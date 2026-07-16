# ERF-LNG Phase 6 Test: LNG_Output

## Purpose

Validates Phase 6 implementation of **output and visualization** for the ERF-LNG module:
1. **2D plotfiles** on the native LNG grid using VisMF with LNGMetadata.json sidecar
2. **Receptor point sampling** at user-defined (x,y) locations, CSV time series
3. **Updated CSV diagnostics** with real `lfl_area_m2` and `ufl_area_m2` from Phase 5

## ATM Configuration

The ATM block is **copied verbatim** from `LNG_GravityCurrent/inputs_lng_gravitycurrent` without any modifications.
This ensures continuity and validates that Phase 6 output does not degrade existing atmospheric behavior.

### Key ATM Settings
- **Domain:** 3000×3000×1024 m, 32×32×64 cells
- **Grid:** Neutral ABL with MRF PBL scheme, SurfaceLayer BC, geostrophic wind = 15 m/s
- **Mandatory constraint:** `amrex.max_grid_size_z = 64` (prevents z-decomposition per Rule B2)

### LNG Phase 6 Settings
- **Grid ratio:** 4 (gives 128×128 LNG grid)
- **Plotfile output:** every 5 steps (`lng_plot_int = 5`)
- **Receptor points:** 2 samples — center (1500, 1500) and downwind (1700, 1500)

## Pass Criteria

1. **Exit code 0**, all 20 steps complete.
2. `[LNG] Writing LNG plotfile plt_lng_XXXXX` printed at steps 5, 10, 15, 20 in stdout.
3. Five plotfile directories exist:
   - `plt_lng_00005/`
   - `plt_lng_00010/`
   - `plt_lng_00015/`
   - `plt_lng_00020/`
4. Each plotfile contains:
   - `Header` (AMReX header file)
   - `Level_0/Cell` (VisMF binary data file)
   - `LNGMetadata.json` (metadata sidecar)
5. `LNGMetadata.json` is valid JSON with `"n_variables": 17`.
6. Two receptor CSV files exist:
   - `lng_receptor_center.csv`
   - `lng_receptor_downwind.csv`
7. Each receptor CSV has exactly **21 lines**: 1 header + 20 data rows (one per step).
8. Columns in receptor CSV: `step,time_s,conc_sfc_kg_m3,vol_fraction,lfl_flag`.
9. `lng_diag.csv` has 21 lines; `lfl_area_m2` and `ufl_area_m2` columns contain non-negative real values (no longer 0.0 after Phase 5 evaporation).
10. `[LNG DEBUG] NaN check PASSED step=20` appears at end (all fields clean).
11. **Regression test:** All 5 prior LNG tests still pass (BuildOnly, PoolEvap, ScalarInjection, WindExtraction, GravityCurrent).
12. **Build test:** Builds with `-DERF_USE_LNG=ON`, no linker errors, no new compile warnings.

## MPI Correctness Notes

Phase 6 implements the MPI-safe patterns from `LNG_MPI_SKILLS.md`:

- **Rule B1 (IOProcessor guard order):**
  - `WriteLNGPlotfile`: VisMF::Write called by ALL ranks (Step 2), IOProcessor writes Header/JSON (Step 3)
  - `append_receptor_sample`: All reductions (`ReduceRealSum`) before IOProcessor guard

- **Rule B4 (FillBoundary periodicity):** Not needed in Phase 6 (no new FillBoundary calls)

- **Rule E1 (write_output duplicate guard):** `m_last_output_step` prevents double CSV rows at final step

- **Rule A2 (build system registration):** `ERF_LNGPlotfile.cpp` in both `Make.package` and `CMake/BuildERFExe.cmake`

## Expected Output Structure

```
plt_lng_00005/
  ├── Header                  (AMReX plotfile header)
  ├── Level_0/
  │   └── Cell               (VisMF binary data, 17 components)
  └── LNGMetadata.json       (metadata: format_version, time, step, grid_ratio, n_variables)

lng_receptor_center.csv       (1500, 1500) time series
lng_receptor_downwind.csv     (1700, 1500) time series
lng_diag.csv                  (updated: real lfl_area_m2, ufl_area_m2)
```

## References

- Pattern source: `Source/Dust/ERF_DustPlotfile.cpp`
- MPI rules: `Source/LNG/LNG_MPI_SKILLS.md` Rules B1, B4, E1, A2
- Phase 6 spec: `Source/LNG/LNG_DEVELOPMENT.md` Phase 6 section
