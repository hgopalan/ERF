# FireLiveMoisture

How the live herbaceous and live woody moisture classes move when
`erf.fire.moisture_dynamic = true`, set by `erf.fire.moisture_live_model`:

- `legacy` (default): the dead-fuel time-lag update with the 100-hour lag,
  bounded to [0.30, 2.50]. That update clamps its input to the dead range
  [0.01, 0.40] first, so a live moisture above 0.40 drops to 0.40 on the
  first step and then decays toward the 0.30 bound.
- `fixed`: the live classes stay at `erf.fire.moisture_live` (or at the
  checkpointed value on a restart), as the fuel-moisture docs always said.

Three one-way decks of a grass fire in dry air under BEHAVE, 60 s, with
`erf.fire.moisture_live = 0.90` (as `ROS_Models/inputs_fire_phase13_behave_dynamic`),
fuel model 2 (it has a live herbaceous load, so BEHAVE's live damping and
live-to-dead herbaceous transfer see the live moisture) and the isotropic
per-cell BEHAVE path: the historical deck (no key), the default written out,
and `fixed`.

```
MPIRUN="mpirun -np 4" ./run_live.sh /path/to/erf_exec
```

`check_live.py` compares the fire plotfiles at 0 and 60 s: the default
written out reproduces the historical deck exactly; both runs start with the
live classes at 0.90; `fixed` still holds them there at 60 s while `legacy`
has dropped them to at most 0.40; the dead classes do not depend on the
setting; and the rate of spread does.

Measured 2026-09-10 (Release, 2 ranks, macOS), at 60 s, with the corrected
dead-fuel hysteresis and the single BEHAVE herbaceous transfer of #382:

| variant      | live herb / woody | 1-h mean | burned area [m2] | mean ROS, burning cells [m/s] |
|--------------|-------------------|----------|------------------|-------------------------------|
| `legacy`     | 0.39994           | 0.07963  | 2118.8           | 0.1840                        |
| `legacy_key` | 0.39994           | 0.07963  | 2118.8           | 0.1840                        |
| `fixed`      | 0.90000           | 0.07963  | 1812.5           | 0.1507                        |

The legacy live classes fall from 0.90 to the 0.40 clamp on the first step.
BEHAVE then moves 89 % of the live herbaceous load to the dead class instead
of 1/3 (default window 0.30-1.20), so the legacy fire spreads 22 % faster
and burns 17 % more area in 60 s. `legacy_key` matches `legacy` exactly in
the level set, the rate of spread and all five classes; the dead classes are
identical in all three runs.

The update itself (the collapse, the bit-for-bit legacy path under both
`erf.fire.emc_model` curves, the held value in any air) is in the unit test
`ERF_GTestFuelMoistureLive`.
