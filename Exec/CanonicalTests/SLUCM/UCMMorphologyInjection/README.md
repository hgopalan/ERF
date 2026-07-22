# Phase 2.6: Morphology-Aware Injection Framework — Canonical Test

## Overview

This test verifies the Phase 2.6 extension to ERF-SLUCM that replaces uniform scalar e-folding depth (`alpha_ucm`) with **per-cell** morphology-driven values based on mean building height (`H_bldg_mean`).

**Physics tested:**
- Road flux injects only at k=klo (surface layer).
- Wall + roof + AH flux decays exponentially with per-cell alpha: `exp(-z/alpha_ij)`.
- Per-cell alpha computed as: `alpha_ij = clamp(alpha_scale * H_bldg_mean, alpha_min, alpha_max)`.
- Tall dense buildings (left half, h≈30m) should inject heat deeper into the atmosphere than short sparse ones (right half, h≈5m).

## Domain & Pattern

- **ATM grid:** 4×4 cells, 64 vertical levels, 1024 m height.
- **UCM grid:** 16×16 cells (grid_ratio=4), aligned with ATM.
- **Urban pattern:** Two vertical stripes.
  - **Left half** (i=0..7): tall dense buildings (h=30m, plan_area_frac=0.6).
  - **Right half** (i=8..15): short sparse buildings (h=5m, plan_area_frac=0.2).
  - All cells are urban (f_urb=1), isolating the alpha-effect from the f_urb effect.

## Files

- `inputs` — ERF parameter file. Key Phase 2.6 settings:
  - `erf.ucm.use_morphology_injection = 1` — enable Phase 2.6 split injection.
  - `erf.ucm.alpha_scale = 1.5`, `alpha_min = 1.0`, `alpha_max = 50.0`.
  - `erf.ucm.ucm_atm_plot_int = 1` — write aggregated ATM plotfile after each step.
  - `erf.ucm.ucm_debug = 1` — enable debug trace.

- `gen_csv.py` — Generate building layout and materials CSVs. Run: `python3 gen_csv.py`.

- `sounding_neutral_abl` — Atmospheric sounding file (neutral, constant θ=300K).

- `check_injection.py` — Verify 8-component ATM plotfile and field values.
  - Assert presence of `H_road_atm` and `H_wallroof_atm` (Phase 2.6).
  - Check left/right stripe heights and flux conservation.

- `check_alpha_effect.py` — Verify morphology-driven injection depth difference.
  - Extract RhoTheta column above tall vs short stripes.
  - Verify tall column heats extend higher.

## Running

```bash
# Generate CSV files
python3 gen_csv.py

# Run simulation (max_step=2, very fast)
mpirun -n 1 ./erf_ucm_morphology_injection inputs 2>&1 | tee run.log

# Verify 8-component ATM plotfile and field values
python3 check_injection.py

# Check alpha effect (optional)
python3 check_alpha_effect.py
```

## Expected Output

- Exit code: 0 at step 2.
- Plotfiles: `plt_ucm_atm_000000`, `plt_ucm_atm_000001` (2D slabs, 8 components).
- Diagnostics: `ucm_diag.dat` (new columns for H_road, H_wallroof, alpha_ij stats).
- Debug trace: `[UCM][2.6]` lines in run.log showing alpha_ij min/max, injection surface vs exponential split.
- check_injection.py: All assertions PASS.
- check_alpha_effect.py: Tall column heat extent > short column heat extent (optional).

## Design Notes

See `Source/UrbanCanopy/UCM_DEVELOPMENT.md` Phase 2.6 section for full design rationale.

Phase 2.6 closes the "uniform alpha_ucm" TODO from Phase 2.5 by making e-folding depth a function of local building morphology, better representing heterogeneous urban canyons (Manhattan vs suburbs).
