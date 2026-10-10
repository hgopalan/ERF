# FireThresholdIgnition

The temperature-threshold ignition (`erf.fire.ignition.threshold_*`): every
burnable, unburned fire cell whose k = 0 potential temperature exceeds
`threshold_temp` ignites. A grass fire on flat ground is started from a 15 m
disc with the heat coupled back to the atmosphere, so the plume the fire
heats and the 8 m/s wind carries over unburned fuel is what crosses the
threshold. Seven decks share one base:

- `off`: nothing set (the default); `off_key` writes the four default keys
  out and must reproduce it line for line.
- `on`: the threshold at 315 K, one fire cell stamped per hot cell.
- `on_spinup`: the same with `threshold_start_time = 4`, so nothing may
  ignite by threshold before the step whose window contains 4 s (the disc
  fire's plume peaks at 338 K about 4 s after ignition and is back to
  about 305 K by 24 s, once the disc has burned out).
- `on_r5`: the same with a 5 m disc stamped around every hot cell.
- `off_farsite`, `on_farsite`: the pair on the FARSITE path, whose level set
  is the normalised [-1, 1] indicator.

```
MPIRUN="mpirun -np 4" ./run_threshold.sh /path/to/erf_exec
```

The script tabulates, per deck, the steps run, the burning cells at the end,
the steps on which threshold ignition happened, the cells it ignited, the
first such step and the hottest surface temperature seen, and checks that the
default changes nothing, that the threshold ignites cells and the fire ends
larger, that the start time is honoured, that the disc ignites at least as
much as the one-cell stamp, and that the FARSITE path ignites too.

## Results

Measured 2026-10-10 on four ranks, 60 s at a fixed 0.125 s step (480 steps),
from the table `run_threshold.sh` prints: the steps run, the burning fire
cells at the end, the steps with the threshold armed (every step after the
start time, whether or not a cell ignites), the cells ignited by threshold
over the run, the first step with an ignition and the hottest surface
temperature seen while the threshold was armed.

| variant | steps | burning cells at 60 s | threshold armed | cells ignited | first step | T max [K] |
|---|---:|---:|---:|---:|---:|---:|
| off | 480 | 432 | 0 | 0 | - | - |
| off_key | 480 | 432 | 0 | 0 | - | - |
| on | 480 | 3592 | 480 | 1889 | 10 | 376.7 |
| on_spinup | 480 | 3567 | 448 | 1938 | 33 | 367.5 |
| on_r5 | 480 | 16384 | 480 | 13964 | 10 | 395.8 |
| off_farsite | 480 | 895 | 0 | 0 | - | - |
| on_farsite | 480 | 4887 | 480 | 3717 | 9 | 375.2 |

All six checks pass. The 5 m disc deck burns the whole 320 x 80 m fire grid
(16384 cells) by 60 s, so the check that it burns at least as much as the
one-cell stamp compares them a quarter of the way through, at step 120 (5892
against 1343 cells), and requires the disc deck to be still growing there.

Under Rothermel's 0.9 I_R wind limit, the default since 2026-10, the fire
runs at up to 1.1 m/s with heat fluxes up to 1 MW/m2. At the former 0.25 s
step `on_r5` stopped at 20 s with a negative density; with
`erf.fire.wind_limit = fuel_class`, the former cap, it ran, and so did the
ERF-Fire branch before these changes. The base deck's step is therefore
0.125 s, which runs every deck; its margin was not derived (whether the
plume's vertical speed or the heating of the first atmosphere cell, about
20 K per 0.125 s step at 1 MW/m2, set the limit was not isolated). The
start-time check reads the step and the start time from the decks.
