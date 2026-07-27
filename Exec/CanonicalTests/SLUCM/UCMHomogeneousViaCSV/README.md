# UCMHomogeneousViaCSV Test

## Purpose
Verify that the Phase 2.1 CSV reader infrastructure produces identical results to the Phase 1.4 homogeneous case when all cells in the CSV are urban with uniform properties.

## Expected Behavior
- Building layout CSV: 8×8 grid, all `is_urban=1`, all cells have height=10m, W_road=W_roof=10m
- Material library CSV: single material (mat_id=1) with properties matching homogeneous Phase 1.4 defaults
- Result: bit-for-bit match with `UCMOneWayInject` output (H_sensible, LE_latent, injection profiles)

## CSV Structure
- **building_layout.csv**: 64 rows (8×8 grid), all uniform morphology
- **materials.csv**: 1 material entry with urban material properties

## Test Workflow
1. Initialize UCM from CSV (not homogeneous fill)
2. Run atmospheric coupling (one-way injection)
3. Compare against `UCMOneWayInject` baseline

## Phase 2.1 Scope
- ✅ CSV readers load per-cell data
- ✅ Per-cell material IDs index the material library
- ✅ is_urban mask is respected in physics (Phase 4.1)
- ❌ Per-cell heterogeneous morphology physics (Phase 2.2)

## Files
- `building_layout.csv` — 8×8 grid, all urban, uniform heights
- `materials.csv` — Single material with typical urban properties
- `inputs` — ERF runtime configuration (CSV paths set)
- `sounding_neutral_abl` — Copy from UCMScaffold (unchanged)
