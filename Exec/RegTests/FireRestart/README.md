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
- `coupled`: Rothermel on the level-set path with lagged heat coupling, so
  the fire heats the atmosphere and the flux buffer that the first restarted
  step injects has to come back from the checkpoint.

## Reference results

| row | straight cells | straight ROS (m/s) | restarted cells | restarted ROS (m/s) | match |
|---|---|---|---|---|---|
| `farsite` | 70 | 0.250 | 70 | 0.250 | yes |
| `levelset` | 134 | 0.479 | 134 | 0.479 | yes |
| `coupled` | 78 | 0.250 | 78 | 0.250 | yes |

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

Restart of the spotting diagnostics and the crown-fire state is not covered
here.
