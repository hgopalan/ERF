# UCMBostonMultiLevel — Phase 3.6 Multi-Level One-Way Regression

## Purpose

Verify that the SLUCM stack runs correctly with `anchor_level = 1`.
This is the first canonical test with UCM at a non-zero AMR level.

## Configuration

- Base grid: 20x20x64 (from UCMBoston)
- One refinement level over urban core (5-15 km x 5-15 km, ref_ratio=2)
- UCM anchor_level = 1
- Duration: 600 steps (~14 minutes simulated) — this is a plumbing test
- One-way coupling only (atm_feedback_heat = 0, atm_feedback_momentum = 0)

## Data Files

Required data files are symlinked from ../UCMBoston/:

```bash
for f in materials.csv building_layout.csv inflow_boston.txt sounding_boston; do
    ln -sf ../UCMBoston/$f .
done
```

## Validation Metrics (check_multilevel.py)

1. UCM confirmed running at anchor_level = 1
2. No assertion failures or aborts
3. No NaN or Inf in temperature fields
4. Zero Newton clamps across all steps
5. SEB solver was called at least once

## Running

```bash
../../../Build/erf_slucm inputs_multilevel > run.log 2>&1
python3 check_multilevel.py run.log
```

Exit code 0 = PASS, 1 = FAIL. Should complete in a few minutes.

## Prerequisites

- Phase 3.5c must be merged first (baseline physics working end-to-end)
- Phase 3.1a/b level-awareness cleanup already in place
