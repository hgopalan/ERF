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
  fire's plume peaks at 322 K about 5 s after ignition and is back below
  305 K by 30 s, once the disc has burned out).
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
