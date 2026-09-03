# FireRestart

Checks that the fire state survives a checkpoint and restart. For each
propagation path three runs are made from one base deck: straight to 200 s, to
100 s with a checkpoint written at step 200, and a restart from that checkpoint
to 200 s. The burned-cell count and the head rate of spread at 200 s of the
restarted run must equal those of the straight run exactly.

```
./run_restart.sh /path/to/erf_exec
```

## The scenario

The FireRosComparison grass fire: 2 x 2 km, 20 x 20 x 30 cells, FM1 at 6%
moisture, a neutral 8 m/s sounding, a 100 m ignition disc, passive coupling,
Rothermel. Two rows:

- `farsite`: the default Lagrangian path, which restores `fire_phi`,
  `fire_arrival_time`, `fire_ros`, `fire_fuel_load`, `fire_fuel_mc` and the
  displacement accumulator.
- `levelset`: the direction-dependent level-set path, which restores the same
  fields minus the accumulator and continues its reinitialisation cadence.

## Reference results

| path | straight cells | straight ROS (m/s) | restarted cells | restarted ROS (m/s) | match |
|---|---|---|---|---|---|
| `farsite` | 70 | 0.250 | 70 | 0.250 | yes |
| `levelset` | 78 | 0.250 | 78 | 0.250 | yes |

## What this suite found

The first run of it segfaulted on both paths, for two reasons that had been
hiding since fire restart was added. The atmospheric checkpoint reader tried
to copy the fire level set into the `FireLayer` before that object had
allocated any fields (the object exists from the constructor, its MultiFabs
only after `initialize()`), and the call that initialises the fire layer sat
inside the "starting from scratch" branch of `InitData_post`, so on a restart
the fire layer was never set up and the first fire step dereferenced empty
fields. Both are fixed in the commit that added this suite: the early copy is
gone, and initialisation runs on every start with the checkpointed fields
read over the initialised ones.

## Why Rothermel, and what is deliberately not tested

Rothermel is used on both rows because its wind cap makes the rate of spread
independent of small changes in the wind. That matters because the
atmosphere itself does not resume identically after an ERF restart when a
surface layer is active: with the same deck and Balbi 2020, whose rate follows
the wind, the straight and restarted runs agree exactly at the first restarted
step and then drift apart, the reference wind at 200 s reaching 4.90 m/s in
the straight run and 6.06 m/s after the restart with constant molecular
diffusion (5.67 against 6.52 m/s with Smagorinsky). The surface drag resumes
weaker after the restart. That is an atmosphere-side property outside the
fire model and is kept out of this suite, which is meant to isolate the fire
state; it is worth its own investigation.

Restart of the spotting diagnostics and the crown-fire state is not covered
here.
