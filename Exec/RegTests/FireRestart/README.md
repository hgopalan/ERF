# FireRestart

Checks that a run restarted from a checkpoint reproduces the uninterrupted
run exactly. For each row three runs are made from one base deck: straight to
200 s, to 100 s with a checkpoint written at step 200, and a restart from that
checkpoint to 200 s. The burned-cell count and the head rate of spread at
200 s of the restarted run must equal those of the straight run to every
printed digit.

```
./run_restart.sh /path/to/erf_exec
```

## The scenario

The FireRosComparison grass fire: 2 x 2 km, 20 x 20 x 30 cells, FM1 at 6%
moisture, a neutral 8 m/s sounding, a MOST surface layer with z0 = 0.1 m,
constant molecular diffusion and no LES, a 100 m ignition disc. Three rows:

- `farsite`: Rothermel on the default Lagrangian path, passive coupling.
  Restores `fire_phi`, `fire_arrival_time`, `fire_ros`, `fire_fuel_load`,
  `fire_fuel_mc` and the displacement accumulator.
- `levelset`: Balbi 2020 on the direction-dependent level-set path, passive
  coupling. Balbi's rate follows the wind, so this row also requires the
  atmosphere itself to resume exactly.
- `exposure`: the `levelset` row with one masked 75 x 100 m box whose upwind
  face the front reaches at about 50 s and the exposure diagnostics on. Its
  accumulators (`FireHeatLoad`, `FirePeakIntensity`, `FireEmberLandings`)
  are checkpointed, and the last line of the exposure CSV written by the
  restarted run must equal the straight run's.
- `spotting`: the `levelset` row with Albini spotting on a fixed seed. The
  seed is the fixed seed plus the fire step, so the row also checks that the
  fire step counter is restored from the checkpoint; before that the
  restarted run drew different brands.
- `crown`: the `farsite` row with crown fire and the canonical canopy;
  `FireCrownActive` and `FireCrownLoad` carry the crown state.
- `coupled`: Rothermel on the level-set path with lagged heat coupling, so
  the fire heats the atmosphere and the flux buffer that the first restarted
  step injects has to come back from the checkpoint.
- `dust`: the `coupled` row with the dust model on top and coupled to the
  fire (crust removal, outflow wind, lofting), with every piece of dust state
  that lives across steps exercised: the deposition accumulator, the 24-hour
  PM averages, the MSHA dose with 60 s shifts, the STEL running average,
  suppression coverage decaying from a raster, a blast before and one after
  the checkpoint, a haul road, the critical-material budget and the PHREEQC
  feedback files. Besides the fire numbers, the last line of every dust CSV
  written by the restarted run must equal the straight run's.

## Reference results

| row | straight cells | straight ROS (m/s) | restarted cells | restarted ROS (m/s) | match |
|---|---|---|---|---|---|
| `farsite` | 70 | 0.250 | 70 | 0.250 | yes |
| `levelset` | 134 | 0.479 | 134 | 0.479 | yes |
| `coupled` | 78 | 0.250 | 78 | 0.250 | yes |
| `dust` | 78 | 0.250 | 78 | 0.250 | yes |

## What this suite found

Its first runs failed on every row, and the failures traced to three bugs
that had been in the code since the corresponding features were added. All
three are fixed in the commits that added this suite.

**Fire restart segfaulted, twice over.** The atmospheric checkpoint reader
copied the fire level set into the `FireLayer` before that object had
allocated any fields (it exists from the constructor, its MultiFabs only after
`initialize()`). And the call that initialises the fire layer sat inside the
"starting from scratch" branch of `InitData_post`, so on a restart the layer
was never set up and the restore call written inside that branch was dead.
The early copy is gone, and initialisation now runs on every start with the
checkpointed fields read over the initialised ones.

**The surface layer restarted with a 0 K surface.** This one is not a fire
bug and affects every ERF run that uses `zlo.type = surface_layer` with an
`input_sounding` initialisation. `InitData_post` sets the surface temperature
and moisture from the sounding's reference values on every start, but on a
restart the sounding was only re-read when nudging from it was enabled, so
those values were zero. With a fixed surface temperature and no heating rate
nothing rewrites the field afterwards, so the MOST iteration ran against a
surface 300 K colder than the air: on the first restarted step the friction
velocity fell from 0.541 to 0.066 m/s and the temperature scale rose from 0
to 3.7 K, and from there the two runs diverged, by about 20% in the
near-surface wind after 200 s here. The state at the checkpoint step itself
was identical to the last digit; a no-slip wall case restarted exactly. The
same drift, at the same magnitude, appears with constant diffusion,
Smagorinsky and the MRF scheme, and on a forced neutral ABL with gravity,
Coriolis and a geostrophic wind, fire off in all cases. The fix reads the
sounding on every restart from an `input_sounding` run and adds `Tsurf` to the
surface-layer checkpoint fields, after which all of those cases restart to
zero difference in every field.

**The lagged fire flux was not checkpointed.** With `coupling_type = lagged`
the flux computed at step n is injected at step n+1 from a buffer, and the
buffer was not written, so the first restarted step injected no heat at all.
That gave a small persistent offset in the `coupled` row (0.01 K in theta,
0.001 m/s in wind at the surface after 200 steps on the ABL case). The
buffers are now written as `FireQAtmPrev` and `FireQLatAtmPrev` and restored.

**Nothing on the dust grid was checkpointed.** The dust layer's deposition
accumulator, 24-hour PM averages, MSHA dose and shift count, suppression
coverage, PHREEQC timing and super-particles all restarted from zero, and its
own step and time counters restarted from zero as well, so shifts and PHREEQC
intervals were measured from the restart. The `dust` row added them: the
persisting fields are written as `Dust*` MultiFabs, the counters as
`DustState`, and the particles as `DustParticles`; with them the row restarts
to the last printed digit of every CSV and bit-for-bit in the dust plotfile,
on one rank and on four. Two details mattered. The dust step runs after the
dycore of the same step, so the first dycore after a restart injects the
emission flux and deposits with the friction velocity of the last step before
the checkpoint; both are checkpointed, and without the friction velocity the
restarted run deposited nothing on its first step and carried a 5% offset in
the accumulator. And the ghost cells come back from the file without a
boundary fill, because the dust kernels read ghost cells at box edges that
the run itself does not refill: filling them on restart changed the edge
values on four ranks.

**The dust layer depended on the domain decomposition.** The straight `dust`
row on four ranks (boxes of 12 and 8 cells) differed from the same row on one
rank, by 30% in parts of the deposition grid, for two reasons, neither of them
a ghost-cell fill. The scratch copies that carry the fire's wind and heat flux
onto the dust grid used the atmosphere geometry's periodicity on MultiFabs
indexed on the dust grid; with a grid ratio of 4 the copy took dust cells
20 cells apart for periodic images of each other and, on more than one rank,
put the fire's heat at three false locations and dropped it from the true one,
so the lofting tripled the emission in the wrong places. And the deposition
kernel read the dust-grid friction velocity with atmosphere indices: on one
box that is the wrong dust cell, on a box whose dust indices start at 48 it is
a read outside the FAB, and one rank's box deposited with garbage. The copies
now use the dust grid's periodicity and the deposition takes the friction
velocity averaged onto the atmosphere grid. With both, the row on four ranks
is bit-for-bit the row on one rank, with periodic and with inflow-outflow
boundaries, and the restart on four ranks is exact. The particle release had
the same kind of indexing slip, placing particles with the atmosphere's
spacing from dust-grid indices, so with a grid ratio above one most were
released outside the domain; it now uses the dust spacing.

**The fire plotfile mislabelled everything after its base block.** Comparing
the fire plotfiles across rank counts on the way showed `fire_fuel_mc_lh` and
`fire_fuel_mc_lw` differing: the catalog names them right after the nineteen
base fields whenever the moisture MultiFab carries the live classes (it
always does), but the writer never copied them, so their two slots held
uninitialized memory and every later field, spotting onwards, sat two slots
away from its name. The writer now copies them.

Restart of the spotting diagnostics and the crown-fire state is not covered
here.
