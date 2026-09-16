# FireAnchorLevel

The fire grid on a refined level. `erf.fire.anchor_level` names the AMR level
whose grid the fire grid refines; unset (-1) it is the finest level at start-up,
so a single-level deck runs on level 0 as before. On a finer level the fire grid
covers the refined region only, takes one step per step of its level, reads that
level's wind, temperature and humidity, and puts its heat, moisture and smoke into
that level's source, which coarser levels receive by average-down.

## Decks

| deck | levels | what it is |
|---|---|---|
| `inputs_base` | 2 | 16 x 16 x 8 cells of 50 m, a static level-1 box refined 2 x 2 x 1 over x 200-600 m, y 200-500 m (16 x 12 cells of 25 m in boxes of 8 and 4 along y). The fire grid is 64 x 48 cells of 6.25 m over that box. Still air; a prescribed rate of spread per fuel code (1 m/s on code 1, 0.4 m/s on code 4) on a checkerboard map (`fuel_region.asc`, one code per fire cell of the region) grows a disc ignited at (330, 360) m for 40 s, passive coupling |
| `inputs_single` | 1 | the same fire on one level of 25 m cells stepping at 0.5 s, the resolution and step of `inputs_base`'s level 1; its fire grid is the whole domain (128 x 128 cells of 6.25 m) with a map of the whole domain (`fuel_full.asc`) |
| `inputs_heat` | 2 | `inputs_base` made anelastic, nothing burning, a prescribed 1e4 W/m2 disc of 30 m at (350, 350) m on level 1, lagged two-way coupling, no fuel map |
| `inputs_partial_height` | 2 | a level-1 box that stops at 200 m: stops at start-up |
| `inputs_two_patches` | 2 | a second level-1 patch at 650-750 m: stops at start-up |

`make_fuel_maps.py` writes the two maps from one checkerboard by position.

## Running

```
MPIRUN="mpirun -np 2" ./run_anchor_level.sh /path/to/erf_exec
```

The script runs eight variants and `check_anchor_level.py`:

| variant | deck and overrides |
|---|---|
| `single` | `inputs_single` |
| `anchor` | `inputs_base`, checkpoints every 20 steps |
| `restart` | `inputs_base` restarted from `anchor`'s step 20 |
| `heat` | `inputs_heat`, checkpoints every 10 steps |
| `heat_anchor0` | `inputs_heat` with `erf.fire.anchor_level=0` |
| `moved` | `heat`'s step-10 checkpoint restarted with `erf.fire.anchor_level=0`, which must stop |
| `heat_mrf` | `inputs_heat` with `erf.pbl_type=MRF erf.pbl_mrf_fire_thermal_excess=true`, checkpoints every 10 steps |
| `heat_mrf_restart` | `heat_mrf` restarted from its step 10 |

## What is checked, and the measured values

Parity with the single-level run:

| check | measured |
|---|---|
| the fire plotfile of `anchor` covers the refined region | x 200-600 m, y 200-500 m, 64 x 48 cells |
| the start-up line names level 1, no warning | `[FIRE] Fire grid on level 1 of 0-1: x 200 to 600 m, y 200 to 500 m, 64 x 48 fire cells of 6.25 x 6.25 m` |
| arrival time of every fire cell equals `single`'s at the same position | largest difference 0 s over 3072 cells |
| the front is non-trivial | 139 cells burned in both fuel codes, latest arrival 39.5 s |
| `single`'s fire stays inside the region | 0 burned cells outside it |
| statistics CSVs identical | 81 rows each, identical |

Heat on level 1 reaching level 0:

| check | measured |
|---|---|
| the coupling on level 1 takes in the disc's power and places 1 - exp(-400/45) of it | 158 stage lines, energy_in exact, energy_out / energy_in within 1.2e-10 of 0.99986209 |
| Cp times the change of the level-0 integral of rho theta against the placed power over the heated time (tolerance 0.1 %) | 1.17233 GJ against 1.17249 GJ, ratio 0.99986 |
| level 1 averages down onto level 0 | 1.172331 GJ on level 1 and on level 0 over the refined region |
| no heat outside the refined region | 2.7e-11 GJ |
| with `erf.fire.anchor_level=0` the log warns | "level 1 covers 18.75 % of the fire grid" |
| with `erf.fire.anchor_level=0` average-down replaces the heated cells | 0.00000 GJ of 0.97487 GJ kept |

The ratio 0.99986 is the same on one level, compressible or anelastic. The heat
deck is anelastic because the compressible form of it does not close the budget on
two levels: rho theta moves with the pressure waves the heating raises, and the
level-0 integral kept 43 % of the heat after 20 s against 99.99 % on one level. The
difference is the compressible solver's coarse-fine interface, not the fire.

Restart:

| check | measured |
|---|---|
| fire plotfile at step 40 after a restart at step 20 | byte-identical |
| statistics rows after the restart | the last 40 rows of the straight run |
| restart with the fire grid moved to level 0 | stops: "holds the fire state on level 1, but this run puts the fire grid on level 0" |

MRF with its fire thermal excess on level 1. MRF reads the lagged fire flux of the
fire grid's level in a halo of columns around every tile, so the flux carries ghost
columns, filled from the neighbouring boxes after every update and after a restore
with that level's geometry; outside the refined region they hold zero.

| check | measured |
|---|---|
| atmosphere plotfile (both levels) at step 40 after a restart at step 10 | byte-identical, 4 data files |
| fire plotfile at step 40 after that restart | byte-identical |
| MRF is on: the atmosphere differs from `heat`'s | differs |

A build without the flux ghosts stops in the first step of `heat_mrf` with
"(15,7,0,0) is out of bound (16:23,8:15,0:0,0:0)". A build whose restore refills
the ghosts with level 0's geometry clamps level-1 columns to level 0's index range,
which overwrites restored flux beyond column 15; its restarted atmosphere and fire
plotfiles are not byte-identical and the two restart checks fail.

In `ctest -L fire -j 4` on a 2-rank Debug build the eight runs took 140 s.

## CTests

`FireAnchorLevel` (MPI builds, not Windows) runs the script on two ranks. The
start-up checks have their own tests, each passing when the run stops with the
message named:

| test | inputs | message |
|---|---|---|
| `FireAnchorLevel_above_finest_abort` | `inputs_base erf.fire.anchor_level=2` | is above the finest level of this run |
| `FireAnchorLevel_regrid_abort` | `inputs_base erf.regrid_int=10` | regrids it |
| `FireAnchorLevel_partial_height_abort` | `inputs_partial_height` | Cannot decompose in z direction |
| `FireAnchorLevel_two_patches_abort` | `inputs_two_patches` | but the fire grid needs one rectangle |
| `FireAnchorLevel_dust_abort` (dust builds) | `inputs_base erf.dust.enable=true` | The dust layer and the fire-dust coupling run on level 0 |

The unit test `ERF_GTestFireAnchorLevel` checks the fire grid over a refined
region and the maps from fire cells to atmospheric columns (terrain height,
slopes, column grounds, surface fields, flux coarsening, bilinear and nearest wind)
against fields linear or quadratic in position.
