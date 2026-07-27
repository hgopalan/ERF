# UCMHeterogeneousBlock Test

## Purpose
Verify that the Phase 2.1 CSV reader infrastructure correctly handles heterogeneous inputs:
- Per-cell is_urban mask (with non-urban patch)
- Per-cell material ID indexing into the material library
- Physics respects the is_urban=0 mask (no injection/fluxes in non-urban cells)

## Expected Behavior
- Building layout CSV: 16×16 grid with 4×4 non-urban patch in the middle (6-9, 6-9)
- Urban cells alternate between two materials (mat_id=1 and mat_id=2)
- Non-urban patch has is_urban=0 (LSM/MOST owns these cells; Phase 4.1 bypass not active in Phase 2.1)
- Result: UCM physics active only in urban cells; non-urban patch shows zero injection/fluxes

## CSV Structure
- **building_layout.csv**: 256 rows (16×16 grid)
  - Cells (6-9, 6-9): is_urban=0
  - Remaining cells: is_urban=1, alternating roof_mat_id (1 or 2)
  - All urban cells: height=10m, W_road=W_roof=10m
- **materials.csv**: 2 material entries with distinct properties

## Test Workflow
1. Initialize UCM from CSV
2. Populate per-cell material IDs from CSV
3. Run atmospheric coupling (one-way injection)
4. Verify:
   - H_sensible and LE_latent are zero in non-urban patch
   - H_sensible and LE_latent are nonzero in urban cells
   - Material properties are indexed correctly

## Phase 2.1 Scope
- ✅ CSV readers handle heterogeneous is_urban mask
- ✅ Per-cell material IDs loaded into iMultiFabs
- ✅ Material registry lookup functional
- ❌ is_urban bypass in LSM/MOST (Phase 4.1)
- ❌ Per-cell morphology affecting physics (Phase 2.2)

## Files
- `building_layout.csv` — 16×16 grid, 4×4 non-urban patch in middle
- `materials.csv` — Two materials with distinct properties
- `inputs` — ERF runtime configuration (CSV paths set)
- `sounding_neutral_abl` — Copy from UCMScaffold (unchanged)
