# UCMHVACUnit

Phase 5.2 unit test for the SLUCM HVAC waste-heat module.

## Scenarios

Four input files exercise the HVAC block's control-flow branches. Each variant
is a 400 m x 400 m, 4 x 4 x 32 domain running 2 timesteps at CFL=0.5.

| Variant                    | HVAC mode | Setpoint (K) | Occupancy | Canyon T | Expected Q_HVAC |
|----------------------------|-----------|--------------|-----------|----------|-----------------|
| `inputs_off`               | off       | -            | -         | ~293 K   | not printed (block skipped) |
| `inputs_simple_hot`        | simple    | 290          | 1.0       | ~293 K   | > 0 (engaged) |
| `inputs_simple_cold`       | simple    | 310          | 1.0       | ~293 K   | 0 (setpoint gate fires: 293 < 310 - 2) |
| `inputs_simple_unoccupied` | simple    | 290          | 0.0       | ~293 K   | 0 (occupancy gate fires) |

## Running

```bash
for v in off simple_hot simple_cold simple_unoccupied; do
    mpirun -np 2 ../../../build/Exec/erf_exec inputs_${v} > run_${v}.log 2>&1
    echo "=== $v ==="
    grep -E '5\.2\]\[hvac\]|STEP 2 ends|Total Time' run_${v}.log
done
```

Expected output pattern:

```
=== off ===
Coarse STEP 2 ends. TIME = 0.2858234688 ...
Total Time: 0.0...

=== simple_hot ===
[UCM][5.2][hvac] mode=simple hour=12 Q_HVAC=[465.60, 465.60] W/m^2   # step 1
[UCM][5.2][hvac] mode=simple hour=12 Q_HVAC=[473.84, 473.84] W/m^2   # step 2
Coarse STEP 2 ends. TIME = 0.2858234688 ...

=== simple_cold ===
[UCM][5.2][hvac] mode=simple hour=12 Q_HVAC=[0, 0] W/m^2
[UCM][5.2][hvac] mode=simple hour=12 Q_HVAC=[0, 0] W/m^2
Coarse STEP 2 ends. TIME = 0.2858234688 ...

=== simple_unoccupied ===
[UCM][5.2][hvac] mode=simple hour=12 Q_HVAC=[0, 0] W/m^2
[UCM][5.2][hvac] mode=simple hour=12 Q_HVAC=[0, 0] W/m^2
Coarse STEP 2 ends. TIME = 0.2858234688 ...
```

## Files

- `inputs_off`, `inputs_simple_hot`, `inputs_simple_cold`, `inputs_simple_unoccupied` -- ParmParse inputs per variant
- `hvac_hot.csv`, `hvac_cold.csv`, `hvac_unocc.csv` -- HVAC profiles, one per ON variant
- `occupancy_test.csv` -- occupied 24 h (used by hot, cold)
- `occupancy_test_unoccupied.csv` -- unoccupied 24 h (used by unoccupied)
- `hvac_test.csv` -- legacy fixture (unused after this rewire; kept for backward compat)
- `building_layout.csv`, `materials.csv` -- morphology and material library
- `sounding_boston` -- initial atmospheric profile
- `check_hvac.py` -- post-run parser that extracts Q_HVAC per step

## Notes on the debug session (2026-07-28)

During initial bringup, this test hung at step 2 whenever HVAC ran. Root cause
was a Phase 1.4 Bug #9 violation in `ERF_UCMLayer.cpp`: the Q_HVAC_diag
min/max collectives were guarded by `if (IOProcessor())`, so only rank 0 called
them while rank 1 skipped. Subsequent collectives desynchronized and rank 1
entered step 2 while rank 0 was still stuck in step 1.

Fixed in Phase 5.2-hotfix1:
1. Move Q_HVAC_diag min/max OUTSIDE the IOProcessor guard.
2. Add MPI barrier after HVAC/occupancy CSV I/O to sync ranks.
3. Change HVAC-block MFIter source to `*fields.T_canyon_air` (safe binding).
4. Scalar-hoist HVAC profile lookup to satisfy Contract #22 (no std::vector in
   ParallelFor lambda).
