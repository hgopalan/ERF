# ERF-LNG Phase 7: Regulatory Compliance (NFPA 59A) Test

## Purpose

This canonical test validates Phase 7 NFPA 59A regulatory compliance output for LNG hazard dispersion modeling:

1. **1-hour running exponential moving average** of `lng_conc_sfc` at each LNG grid cell
2. **Exceedance flags** where 1h-average concentration ≥ 1/2 LFL (NFPA 59A exclusion threshold = 2.5 vol%)
3. **Exclusion zone radius** estimation: furthest distance from pool center where exceedance flag is non-zero
4. **Regulatory CSV output** (`lng_regulatory.csv`): per-timestep fence-line concentrations, exclusion zone radius, and exceedance cell count

## Atmospheric Configuration

- **Domain:** 3000 × 3000 × 1024 m
- **Grid:** 32 × 32 × 64 cells (base ATM grid)
- **LNG grid refinement:** grid_ratio = 4 → 128 × 128 LNG cells
- **Time stepping:** fixed dt = 0.5 s, 20 steps total = 10 s simulation
- **Surface:** Monin-Obukhov surface layer, z₀ = 0.1 m
- **PBL:** MRF with Ri_critical = 0.5
- **Wind:** Geostrophic 15 m/s in x-direction
- **Temperature:** Neutral stratification 300 K near surface, 0.003 K/m above

**Critical AMReX Setting:** `amrex.max_grid_size_z = 64` (prevents z-decomposition; required by LNG 2D slab operations)

## LNG Physics

- **Pool:** 500 m² × 0.05 m deep, centered at domain mid-point (1500, 1500) m
- **Spill rate:** 20 kg/s constant LNG injection
- **LNG composition:** CH₄ 90%, C₂H₆ 8%, N₂ 2% → mol. weight 17.4 g/mol
- **Evaporation:** Wind-driven mass transfer using Monin-Obukhov scaling
- **Gravity current:** 2D shallow-water PDEs on LNG grid with Richardson transition
- **Flammability:** Track LFL (5%) and UFL (15%) zones

## Phase 7 Parameters

```
erf.lng.nfpa59a_exclusion_conc = 0.025   # 1/2 LFL = 2.5 vol%
erf.lng.lng_regulatory_file    = "lng_regulatory.csv"
```

## Pass Criteria (12 requirements)

1. **Exit code 0**, simulation completes 20 timesteps without error
2. **`lng_regulatory.csv` file created** with header comment block describing NFPA 59A standard
3. **CSV structure:** `step,time_s,exclusion_zone_radius_m,conc_1h_max_kg_m3,n_cells_exceed` columns
4. **Data rows:** exactly 20 rows (one per timestep)
5. **Early-step behavior:** `exclusion_zone_radius_m = 0.0` for first ~5 steps (insufficient vapor accumulation to exceed threshold)
6. **Concentration growth:** `conc_1h_max_kg_m3` increases monotonically in early steps as vapor accumulates
7. **Exceedance count:** `n_cells_exceed = 0` early, may become non-zero as 1h-average exceeds threshold in later steps
8. **Debug output:** `[LNG DEBUG] Phase 7:` appears exactly 20 times in stdout
9. **Plotfile output:** 2 plotfiles written (steps 0 and 10); each contains `"n_variables": 19` with new `lng_conc_1h_avg` and `lng_exceed_flag` fields
10. **Prior diagnostics:** `lng_diag.csv` has 21 lines (header + 20 data); receptor CSVs written for "center" and "downwind"
11. **NaN check:** "[LNG DEBUG] NaN check PASSED" appears 20 times
12. **Build success:** Compiles without linker errors with `-DERF_USE_LNG=ON`

## MPI Safety Details

### Kernel Patterns

- **`update_lng_1h_average`**: Uses `ParallelFor` with `tilebox()` — GPU-safe, no collective operations. Followed by `FillBoundary` with periodicity.
- **`compute_lng_exceedance`**: Uses `ParallelFor` with `tilebox()` — GPU-safe, no collective operations. Followed by `FillBoundary` with periodicity.
- **`compute_exclusion_zone_radius`**: 
  - **Step 1 (CPU-local):** `LoopOnCpu` on each rank's valid boxes to find local max radius
  - **Step 2 (MPI-collective):** `ReduceRealMax` broadcasts max to all ranks (all ranks participate before any IOProcessor guard)

### MPI Rule B1 Compliance

In `append_lng_regulatory_row()`:
```cpp
// ── ALL MPI-collective reductions first (every rank participates) ───
amrex::Real conc_1h_max = lng_conc_1h_avg.max(0);  // MPI_Allreduce
amrex::Real n_exceed     = lng_exceed_flag.sum(0);  // MPI_Allreduce

// ── File write: IOProcessor only ─────────────────────────────────────
if (!amrex::ParallelDescriptor::IOProcessor()) return;
// ... file write ...
```

All `MultiFab::max/sum` calls complete before `IOProcessor()` guard. If guard preceded collectives, rank 0 would enter `MPI_Allreduce` while rank 1+ have returned → deadlock.

### Grid Construction

- New MultiFabs (`m_lng_conc_1h_avg`, `m_lng_exceed_flag`) allocated on `m_lg.ba` / `m_lg.dm` (LNG grid, not ATM)
- No size-based cell counting (all operations preserve local rank tile sizes)

### Build System

- **Makefile:** `CEXE_sources += LNG/ERF_LNGRegulatory.cpp` + `CEXE_headers += LNG/ERF_LNGRegulatory.H` inside `USE_LNG` guard
- **CMake:** `target_sources()` includes `${SRC_DIR}/LNG/ERF_LNGRegulatory.cpp` inside `if(ERF_ENABLE_LNG)` block

Both registrations are **mandatory** (Rule A2 from `LNG_MPI_SKILLS.md`).

## References

- **NFPA 59A (2023):** *Standard for the Production, Storage, and Handling of Liquefied Natural Gas (LNG)*
- **49 CFR Part 193:** LNG facilities (U.S. federal regulation)
- **Koopman, R.P., 1982:** "Burro LNG spill test series final report" (NTIS historical reference)
- **ERF_DustNAAQSOutput.H:** Pattern source (Dust EPA NAAQS PM₂.₅ averaging and exceedance logic adapted for LNG NFPA 59A)
- **LNG_MPI_SKILLS.md:** Complete MPI rules and bug reference
