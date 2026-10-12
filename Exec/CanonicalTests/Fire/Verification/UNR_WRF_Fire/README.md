# UNR_WRF_Fire

## Summary

The WRF-Fire tutorial of the University of Nevada, Reno (UNR,
https://unr-wrf-fire.readthedocs.io, Case Studies 1-3) runs three idealised grass
fires in WRF-Fire and publishes the namelists and figures. These decks set up the
same three cases in ERF, with every setting that acts on the fire or the flow
matched to WRF's one by one (the table below), so that the two fire models can be
compared on the same problems:

- **Case 1**: flat ground, a uniform 5 m/s wind, the fire does not heat the air
  (uncoupled). The head of a long line fire then runs at Rothermel's head rate,
  an exact answer. ERF's head runs at 1.169 m/s against 1.173 m/s from WRF's
  formula (-0.4 %).
- **Case 2**: the same wind over a 200 m high north-south ridge, with the fire
  heating the air (coupled). WRF's default vertical damping, which UNR's namelist
  does not change, damps the vertical wind through the whole 2 km domain and
  blocks the flow at the ridge. With the same damping ERF reproduces UNR's
  pattern; without it the fire crosses the ridge.
- **Case 3**: a turbulent (LES) boundary layer heated from the ground, 2.5 m/s
  wind or none, fire coupled and lit after 30 min of spin-up. Its comparison
  with a new WRF master run, made with the same settings in both codes, is in
  preparation.

Each case has an anelastic deck (the air is treated as incompressible apart from
buoyancy; needs ERF built with FFT, `erf.use_fft = 1`) and a compressible deck at
WRF's time step.

## Running

```
python3 gen_unr_decks.py                       # decks, soundings, ignition lines, ridge (all committed)
MPIRUN="mpirun -np 4" DECKS="case1_compressible" ./run_unr_wrf_fire.sh /path/to/erf_exec
SKIP_RUN=1 DECKS="case1_compressible" ./run_unr_wrf_fire.sh x      # run nothing, check existing output only
```

Without `DECKS` the script runs all eight decks, which takes several hours. A
deck can also be run by hand from this folder, for example
`mpirun -np 4 /path/to/erf_exec inputs_case3_wind2p5_anelastic`; every file a deck
reads (`input_sounding_*`, `line_*.csv`, `terrain_ridge.txt`) is named relative to
this folder, and every deck writes its own plotfiles (`plt_<deck>_*`,
`plt_fire_<deck>_*`) and `fire_stats_<deck>.csv`, so decks can share the folder.

## The cases

The table lists each deck, its setup and its cost. Wall times are on four ranks of
an Apple M2 Pro, read from the runs that produced the results below.

| deck | UNR case | ground, wind | fire heats the air | atmosphere / fire cells | time step, length | wall time |
|---|---|---|---|---|---|---|
| `case1_compressible` | 1 | flat, 5 m/s | no | 52 x 52 x 100 at 100 m / 25 m | 0.25 s, 1 h | about 30 min |
| `case1_anelastic` | 1 | flat, 5 m/s | no | same | 0.25 s, 1 h | about 15-19 min |
| `case2_compressible` | 2 | 200 m ridge, 5 m/s | yes | same | 0.25 s, 1 h | about 45 min |
| `case2_anelastic` | 2 | 200 m ridge, 5 m/s | yes | same | 0.25 s, 1 h | about 40 min (an earlier version of the deck) |
| `case3_wind2p5_anelastic` | 3 | flat, 2.5 m/s, LES | yes | 128 x 128 x 51 at 40 m / 5 m | 0.5 s, 2 h | about 72 min |
| `case3_wind2p5_compressible` | 3 | flat, 2.5 m/s, LES | yes | same | 0.5 s, 2 h | not yet run to the end |
| `case3_nowind_anelastic` | 3 | flat, no wind, LES | yes | same | 0.5 s, 2 h | not yet run to the end |
| `case3_nowind_compressible` | 3 | flat, no wind, LES | yes | same | 0.5 s, 2 h | not yet run to the end |

All cases: a 2 km deep atmosphere, periodic sides, a slip wall on top, the
sounding of UNR's `input_sounding` (300 K to 1 km, then 1 K per 100 m), fuel
category 1 (short grass) at 8 % moisture, and a 1 km ignition line across the
wind (x = 1050 m in cases 1 and 2, lit 10 s after the start, 200 m wide; x = 1020 m
in case 3, lit at 30 min, 80 m wide). The vertical levels are WRF's eta levels
converted to heights: in cases 1 and 2 WRF's default exponential stretch on a
2 km top gives the thickest layer at the ground (49 m) and the thinnest at the top
(5 m), and the fire takes its 6.5 m wind from a log law below the first cell
centre, as WRF does; case 3 uses UNR's hyperbolic stretch (18 m at the ground).
The domain has 52 and 128 cells across instead of WRF's 50 and 125 so that the
boxes divide by the fire refinement.

## How the decks match WRF

Each row gives the WRF value (UNR's namelist where it sets one, otherwise the WRF
default and the code that reads it) and the ERF input that reproduces it. The
comments in the decks name the WRF entry each line reproduces.

| what | WRF | ERF input in these decks |
|---|---|---|
| Wind in the spread rate | `fire_advection = 1` (default): wind factor from the wind speed, times the cosine to the front normal | `directional_wind_coupling = "advective"` (ERF default `"projection"` is WRF's 0) |
| Level-set split | `fire_upwind_split = 0` (default) | `directional_split_hamiltonian = false` (ERF default) |
| Wind reduction factor | `windrf` read but never applied | `use_waf = false` |
| Wind height | `fire_wind_height = 6.5`, log law from z0 = 0.1 m below the first level | `wind_ref_ht = 6.5`, `wind_below_first_cell = "log"`, `wind_sample_z0 = 0.1` |
| Rothermel reaction velocity | exponent A = 1/(4.774 s^0.1 - 7.27) (Rothermel 1972) | `reaction_velocity_formula = "rothermel"` |
| Rothermel fuel load | (1 - M/(1+M)) fgi | `wrf_bmst_compat = true` |
| Heat content | 7496 Btu/lb (`cmbcnst` x 4.30e-4) | custom fuel 1000, `heat_content_J_kg = 17.433e6` |
| Fuel category 1 | fgi 0.166 kg/m2, depth 0.305 m, savr 3500 /ft, Mx 0.12, density 32 lb/ft3 | custom fuel 1000 with the same values (density 512 kg/m3); mineral contents 0.0555 and 0.010 are ERF's fixed values, as WRF's |
| Fuel moisture | `fuelmc_g = 0.08` | `moisture_1hr`, `_10hr`, `_100hr = 0.08`, `moisture_dynamic = false` |
| Sensible heat to the air | burnt mass x (1 - M/(1+M)) x `cmbcnst`; case 1 `fire_atm_feedback = 0` | `fire_atm_feedback = 0.9259` (= 1/1.08) in cases 2 and 3, 0 in case 1 |
| Latent heat to the air | moisture and water of combustion x L_v | `inject_latent = true`: ERF's own form, scaled by the same 0.9259; not compared term by term |
| Burn-out | `fuel_time = weight / 0.85`, weight 7 s | `burnout_model = "sfire"`, `burnout_time_to_efold = 0.85`, `burnout_time_s = 7` |
| Depth of the heat release | `fire_ext_grnd = 50` m (default) | `heat_flux_alfg = 50` |
| Level-set viscosity | `fire_viscosity = 0.4` and `fire_viscosity_bg = 0.4`, in fire cells | `levelset.eps_visc` and `eps_visc_front` = 0.4 fire cells in metres (10 m in cases 1 and 2, 2 m in case 3) |
| Level-set gradient | `fire_upwinding = 7` (cases 1, 2) / 9 (case 3) | `levelset.gradient = "weno5z"` / `"weno5z_front"` |
| Reinitialisation | `fire_lsm_reinit` every step, one pass, the result clamped against the start-of-step field | `levelset.reinit_every = 1`, one pass (default); the clamp against the start-of-step field is ERF's default `levelset.reinit_clamp = "step_start"` |
| Ignition | line, half-width `fire_ignition_radius1`, at `fire_ignition_start_time1` | `ignition.polygon_type = "polyline"`, `polyline_width`, `polygon_time` |
| Vertical damping | cases 1 and 2: `damp_opt = 3`, `zdamp = 5000`, `dampcoef = 0.2` (defaults; UNR does not set them); case 3: `damp_opt = 0` | cases 1 and 2: `rayleigh_damp_W = true`, `rayleigh_zdamp = 5000`, `rayleigh_dampcoef = 0.2`; case 3: none |
| Diffusion | cases 1 and 2: `diff_opt = 0`; case 3: `diff_opt = 2`, `km_opt = 2`, `c_k = 0.10` | cases 1 and 2: none (zero coefficients, which the surface boundary needs to run); case 3: `les_type = "Deardorff"`, `Ck = 0.10` |
| Surface | `isfflx = 0`: drag `tke_drag_coefficient` (0 / 0.005), heat flux `tke_heat_flux` (0 / 0.10 K m/s) | `surface_layer.flux_type = "bulk_coeff"`, `most.Cd`, `most.surf_temp_flux` |
| Start-up perturbation | case 3: a 0.5 K bubble, radii 2500, 2500, 40 m, centred 40 m up on mass point (ide-ids)/2 = 62, x = y = 2460 m | `prob.bubble_dT = 0.5`, `bubble_x/y/z_r`, `bubble_x_c = bubble_y_c = 2460`, `bubble_z_c = 40` |
| Advection | 5th order horizontal, 3rd order vertical | `Upwind_5th`, `Upwind_3rd` |
| Time step | 0.25 s (cases 1, 2), 0.5 s (case 3) | `fixed_dt` the same; ERF picks its own acoustic substeps |
| Top | rigid lid | `zhi.type = "SlipWall"` |
| Coriolis | case 3: `pert_coriolis = .true.` | `use_coriolis = false`: differs (case 3 only; small over 2 h) |

## Results so far

Measured on 2026-10-11 on four ranks with the reinitialisation clamp of the time
(`reinit_clamp = "post_advection"`; the decks now use the default `"step_start"`).
To reproduce the numbers below with the committed decks, add on the command line:

- cases 1 and 2: `erf.fire.levelset.reinit_clamp=post_advection`
- the damped case 2 numbers: also `erf.fire.fire_atm_feedback=0` (they come from
  the case 2 deck with the fire's heat switched off)

A 10 min run of `case1_anelastic` on two ranks gives the same head with either
clamp (4-10 min: 1.1630 and 1.1629 m/s, -0.9 % of WRF's rate).

**Case 1** (`case1_compressible`). The head, where the fire's arrival time crosses
t near the middle of the line, against UNR's figures:

| time after the start | 10 min | 20 min | 40 min |
|---|---|---|---|
| ERF head x | 1.84 km | 2.54 km | 3.94 km |
| UNR figures | 1.65 km | 2.35 km | 3.75 km |

- Head rate 1.169 m/s against 1.173 m/s from WRF's Rothermel formula; between 10
  and 40 min the run's head moves at 1.168 m/s and UNR's at 1.167 m/s.
- The constant +0.19 km offset is set at the start, before 10 min.
- |grad phi| at the front is 1.00: the level set stays a distance function.

**Case 2**. WRF's default `damp_opt = 3` with `zdamp = 5000` m, deeper than the
2 km domain, damps the vertical wind everywhere (by 0.65 of the full rate at the
ground). With that damping ERF reproduces UNR's pattern: the wind near the ground
dies and reverses over the ridge and the fire stalls before it (head at 1.39 km
after 10 min, then 0.02-0.05 m/s). Without damping the flow crosses the ridge
smoothly (7.9 m/s at the crest, 3.9 m/s at its feet) and the coupled fire crosses
it at 2-3 m/s. The damped numbers come from the case 2 deck with the fire's heat
switched off (`erf.fire.fire_atm_feedback=0` on the command line); the coupled
damped deck as committed has not yet been run to the end.

**Case 3** is being compared with a new WRF master run, made with the same
settings in both codes; its results will be added here from that comparison.
Two points for whoever runs it:

- WRF-SFIRE (openwfm) ignores UNR's level-set settings (it turns
  `fire_upwinding = 9` into 3 and does not reinitialise), so the WRF run must use
  WRF master (`ifire = 2`), as UNR did.
- The level set steepens in this case (|grad phi| well above 1 at the front).
  The level-set equation predicts this wherever the spread rate drops behind the
  front, and WRF's reinitialisation strength (one pseudo-time step of 0.01 fire
  cells per step), which the decks keep, cannot hold it at 1;
  `levelset.reinit_iters` and `levelset.reinit_dtau` give a stronger one.

## The check

`check_unr_wrf_fire.py` reads the last fire plotfile of each deck and prints the
head at UNR's figure times next to UNR's numbers. Case 1 has an independent
answer and is checked: the head rate between 10 and 40 min (`--rate-window`)
within 1 % of 1.173 m/s, and |grad phi| at the head within 10 % of 1. Cases 2
and 3 have no independent number; the script prints their heads and rates and
fails only on missing or empty output. On the case 1 run above (the
`"post_advection"` clamp) it prints a head rate of 1.1684 m/s (-0.39 %) and
|grad phi| 1.000. The 1.173 m/s reference is WRF's rate at the 3.79 m/s wind that
the log law gives at 6.5 m below the 24.6 m first cell centre; it holds for these
decks' vertical levels only.

## Files

| file | content |
|---|---|
| `inputs_case*` | the eight decks |
| `input_sounding_case12`, `input_sounding_case3_wind2p5`, `input_sounding_case3_nowind` | UNR's sounding with 5, 2.5 and 0 m/s |
| `line_case12.csv`, `line_case3.csv` | the ignition lines |
| `terrain_ridge.txt` | the case 2 ridge, 105 x 105 points, heights with y varying fastest |
| `gen_unr_decks.py` | writes all of the above |
| `run_unr_wrf_fire.sh`, `check_unr_wrf_fire.py` | run the decks, check the heads |

## Not yet done

- Case 3: results from the comparison with a new WRF master run.
- The coupled, damped case 2 deck, the case 3 compressible decks and the no-wind
  decks run to the end (so far each has passed a zero-step input check and a
  20-step run).
- Cases 1 and 2 repeated with the committed decks (`reinit_clamp = "step_start"`,
  the new default).
