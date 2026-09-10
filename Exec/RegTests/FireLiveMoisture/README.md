# FireLiveMoisture

How the live herbaceous and live woody moisture classes move when
`erf.fire.moisture_dynamic = true`, set by `erf.fire.moisture_live_model`:

- `legacy` (default): the dead-fuel time-lag update with the 100-hour lag,
  bounded to [0.30, 2.50]. That update clamps its input to the dead range
  [0.01, 0.40] first, so a live moisture above 0.40 drops to 0.40 on the
  first step and then decays toward the 0.30 bound.
- `fixed`: the live classes stay at `erf.fire.moisture_live`, as the
  fuel-moisture docs always said. A restart takes them from the inputs, not
  the checkpoint, and the domain-average live moisture that the directional
  level-set path builds its BEHAVE state from is not clamped to [0.30, 2.50].

A grass fire in dry air under BEHAVE, one-way coupled, fuel model 2 (it has a
live herbaceous load, so BEHAVE's live damping and live-to-dead herbaceous
transfer see the live moisture):

- 60 s on the isotropic per-cell BEHAVE path with
  `erf.fire.moisture_live = 0.90` (as `ROS_Models/inputs_fire_phase13_behave_dynamic`):
  the historical deck (`legacy`, no key), the default written out
  (`legacy_key`), and `fixed`.
- Restart: `legacy_chk` writes a checkpoint at step 80 (10 s);
  `legacy_restart` and `fixed_restart` restart from it and run to 20 s.
- Directional path: `fixed_dir020` and `fixed_dir025`, `fixed` with the
  directional level set, `dynamic_transfer_lo = 0.10` and
  `moisture_live` 0.20 and 0.25, 20 s.

```
MPIRUN="mpirun -np 4" ./run_live.sh /path/to/erf_exec
```

`check_live.py` checks that:
- the default written out reproduces the historical deck exactly;
- both 60 s runs start with the live classes at 0.90; `fixed` still holds
  them there at 60 s while `legacy` has dropped them to at most 0.40; the
  dead classes do not depend on the setting; and the rate of spread does;
- `fixed_restart` has its live classes back at 0.90, `legacy_restart` keeps
  the checkpointed values, and the dead classes restart the same;
- the two directional runs hold their live classes and give different fronts.

Measured 2026-09-10 (Release, 2 ranks, macOS):

| variant          | t [s] | live herb / woody | 1-h mean | burned area [m2] | mean ROS, burning cells [m/s] |
|------------------|-------|-------------------|----------|------------------|-------------------------------|
| `legacy`         | 60    | 0.39993           | 0.07918  | 2118.8           | 0.1844                        |
| `legacy_key`     | 60    | 0.39993           | 0.07918  | 2118.8           | 0.1844                        |
| `fixed`          | 60    | 0.90000           | 0.07918  | 1812.5           | 0.1510                        |
| `legacy_restart` | 20    | 0.39998           | 0.07972  | 1106.2           | 0.1839                        |
| `fixed_restart`  | 20    | 0.90000           | 0.07972  | 1062.5           | 0.1506                        |
| `fixed_dir020`   | 20    | 0.20000           | 0.07972  | 853.1            | 0.1861                        |
| `fixed_dir025`   | 20    | 0.25000           | 0.07972  | 853.1            | 0.1865                        |

The legacy live classes fall from 0.90 to the 0.40 clamp on the first step.
BEHAVE then moves 89 % of the live herbaceous load to the dead class instead
of 1/3 (default window 0.30-1.20), so the legacy fire spreads 22 % faster
and burns 17 % more area in 60 s. `legacy_key` matches `legacy` exactly in
the level set, the rate of spread and all five classes; the dead classes are
identical in all runs.

Before the `fixed` restart and averaging fixes, `fixed_restart` came back at
the checkpoint's 0.39999, and the two directional fronts were identical to
the bit (both used 0.30). Now the fronts differ by up to 9e-3 m in the level
set after 20 s; the burned cell count is still the same.

The update itself (the collapse, the bit-for-bit legacy path under both
`erf.fire.emc_model` curves, the held value in any air) is in the unit test
`ERF_GTestFuelMoistureLive`.
