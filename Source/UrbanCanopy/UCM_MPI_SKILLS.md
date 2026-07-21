# ERF-SLUCM Module — Development Skills & Bug Fix Reference

Complete record of lessons learned during development of the ERF-SLUCM urban canopy model on the `ERF-SLUCM-phase-1.1` and subsequent branches. Use this as a checklist before merging any new 2D slab module variant.

This file will grow phase by phase as new technical challenges are discovered and documented.

---

## Part A — Architecture & Design Rules

### A1. Follow the Dust and Fire Module Patterns Exactly

**TODO(UCM Phase 1.2):** Document specific Dust/Fire analogs as they are replicated. When in doubt, copy the reference implementation and substitute UCM for module name.

| UCM Component | Reference Analog |
|---------------|------------------|
| `ERF_UCMGrid.H/cpp` | `Source/Dust/ERF_DustGrid.H/cpp` |
| `ERF_UCMPrerequisites.H/cpp` | `Source/Dust/ERF_DustPrerequisites.H/cpp` |
| `ERF_UCMParams.H` | `Source/Dust/ERF_DustParams.H` |
| `(Phase 1.2) ERF_UCMLayer.H/cpp` | `(Phase 2) Source/Dust/ERF_DustLayer.H/cpp` |
| `(Phase 1.3) ERF_UCMAtmCoupling.H/cpp` | `Source/Dust/ERF_DustAtmCoupling.H/cpp` |
| `(Phase 1.3) ERF_UCMWindExtract.H/cpp` | `Source/Dust/ERF_DustWindExtract.H/cpp` |

Diverging silently is the primary source of bugs. Document deviations with `TODO(UCM PhaseX.Y): rationale` comments.

### A2. Build System — Register in Both Make.package and CMake

Every new `.cpp` file must be registered in **both** build systems or one will produce linker errors.

**Make.package pattern:**
```makefile
ifeq ($(USE_UCM), TRUE)
  CEXE_sources += UrbanCanopy/ERF_UCM_New.cpp
endif
```

**CMake pattern (in CMake/BuildERFExe.cmake):**
```cmake
if(ERF_ENABLE_UCM)
  target_sources(${erf_lib_name} PRIVATE ${SRC_DIR}/UrbanCanopy/ERF_UCM_New.cpp)
endif()
```

**Lesson:** Check both files every time a new source is added. Missing from either → linker error.

### A3. Test Case Domain Must Match Baseline

All UCM canonical tests use a consistent minimal domain:

```
geometry.prob_extent = 3000 3000 1024
amr.n_cell           = 8 8 16   (or 32 32 64 for multi-rank tests)
erf.fixed_dt         = 0.5
zlo.type             = "surface_layer"
erf.pbl_type         = "MYNN2.5" (or simplest available)
erf.transport_scalar = true
amrex.max_grid_size_z = 16      # Equal to amr.n_cell(2)
```

Keep domain small for Phase 1 compilation speed; grow to 32×32×64 for multi-rank MPI tests in Phase 3.

### A4. CSV Output Must Not Be Gated on Debug Flag

**TODO(UCM Phase 1.4):** Diagnostic CSV rows must always be written regardless of debug level. Only console print lines gated:

```cpp
// CORRECT:
append_ucm_stats(...);  // always runs

if (m_params.ucm_debug) {
    amrex::Print() << "[UCM DEBUG] ...";  // gated
}
```

**Lesson:** CSV data loss if diagnostics gated on debug flag.

### A5. `amrex::Vector` Cannot Be Implicitly Converted from `std::vector`

**TODO(UCM Phase 1.4):** Any function returning `std::vector<T>` must be explicitly converted to `amrex::Vector<T>`:

```cpp
// WRONG:
amrex::Vector<std::string> var_vec = ucm_plotfile_var_names();

// CORRECT:
auto std_vec = ucm_plotfile_var_names();
amrex::Vector<std::string> var_vec(std_vec.begin(), std_vec.end());
```

---

## Part B — MPI Multi-Rank Rules

### B1. MPI Collectives Must Come BEFORE IOProcessor Guard

**TODO(UCM Phase 1.4):** MultiFab::sum(), max(), min(), contains_nan() are MPI_Allreduce calls. Every rank must execute them before any IOProcessor guard:

```cpp
// WRONG — deadlock with 2+ ranks:
if (!amrex::ParallelDescriptor::IOProcessor()) return;
amrex::Real val = mf.sum(0);  // rank 1 never enters, rank 0 hangs

// CORRECT:
amrex::Real val = mf.sum(0);  // all ranks participate
if (!amrex::ParallelDescriptor::IOProcessor()) return;
amrex::Print() << val;  // only rank 0 writes
```

**Checklist:** Every stats/output function must have MPI collectives first.

### B2. Z-Decomposition Must Be Prevented

**TODO(UCM Phase 1.3+):** AMReX splits domain in z when `amrex.max_grid_size_z` not set. 2D slab operations require full z-column per rank:

```
# inputs file (REQUIRED for ANY run with UCM enabled)
amrex.max_grid_size_z = 16   # equal to amr.n_cell z-dimension
```

Enforce in `check_ucm_prerequisites()`:

```cpp
int domain_nz = atm_domain.length(2);
for (int i = 0; i < ba_atm.size(); ++i) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        ba_atm[i].length(2) == domain_nz,
        "[UCM] Cannot decompose in z. Set: amrex.max_grid_size_z = <nz>");
}
```

**Note:** `amrex.max_grid_size_z` appears in "Unused ParmParse Variables" at end of run (harmless).

### B3. MPI-Safe Field Extraction via ParallelCopy

**TODO(UCM Phase 1.3):** When UCM grid reads ATM fields with `grid_ratio>1` and 2+ ranks, rank-local LNG tiles may access ATM data owned by other ranks. Fix: copy to scratch MultiFabs first:

```cpp
const amrex::Periodicity& per = geom_atm->periodicity();
m_xvel_atm->ParallelCopy(*xvel_mf, 0, 0, 1, nghost, nghost, per);
m_xvel_atm->FillBoundary(per);
// Now safe to read m_xvel_atm on UCM grid
```

Scratch MultiFabs allocated on ATM BoxArray/DistributionMapping in `initialize()`.

### B4. FillBoundary Must Pass Periodicity

**TODO(UCM Phase 1.3):** Every FillBoundary on UCM grid must pass `geom.periodicity()`:

```cpp
// WRONG:
ucm_field.FillBoundary();

// CORRECT:
ucm_field.FillBoundary(geom_ucm.periodicity());
```

**Lesson:** Without periodicity, ghost cells at inter-rank boundaries never exchanged.

### B5. Avoid `average_down` for Non-Coarsenable Fields

**TODO(UCM Phase 2.1):** `average_down` works only when source grid is exact refinement of target. For heterogeneous CSV fields, use manual loop-over-cells coarsening instead.

---

## Part C — Terrain and Atmospheric Coupling

### C1. Always Use `z_phys_cc(i, j, k) - z_phys_cc(i, j, klo)` for Height-Above-Surface

**TODO(UCM Phase 1.3):** All vertical operations must use terrain-aware coordinate. Contract 3 enforces this:

```cpp
// CORRECT:
amrex::Real h_above_surface = z_phys_cc(i, j, k) - z_phys_cc(i, j, klo);
amrex::Real decay = std::exp(-h_above_surface / alpha);

// WRONG (hardcoded z):
amrex::Real decay = std::exp(-z_phys_cc(i, j, k) / alpha);  // ✗
```

**Grep check:** No patterns `z_phys_cc.*k.*0\.0` or `z_phys_cc.*=.*[0-9]` in `Source/UrbanCanopy/`.

### C2. Wind Extraction Reference Height Formula

**TODO(UCM Phase 1.3):** Extract ATM wind at height:

```cpp
z_target = z_phys_cc(i, j, klo) + H_bldg(i, j) + zref
```

where `klo` is k-index of first ATM level, `H_bldg` is local building height, `zref` is `erf.ucm.zref` parameter (default 2 m above roof).

---

## Part D — Data Layout and Ghost Cells

### D1. 2D Slab Ghost Cells Standard

**TODO(UCM Phase 1.2):** All UCM MultiFabs use `amrex::IntVect(1, 1, 0)` ghost cells:

```cpp
amrex::IntVect nghost(1, 1, 0);  // 1 ghost in x, 1 in y, 0 in z
auto mf = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, nghost);
```

---

## Part E — CSV I/O and MPI Broadcasting

### E1. Rank-0-Read + MPI_Bcast Pattern for CSV

**TODO(UCM Phase 2.1):** All CSV input must follow pattern from `LNG_MPI_SKILLS.md`:

1. Rank 0 reads CSV file into POD struct array
2. MPI_Bcast struct array to all ranks (requires fixed-size POD, no std::string)
3. Each rank unpacks into local data structures

**Rationale:** Avoids N-way file system contention and ensures identical parsing.

### E2. No Rank-0-Only File Operations Outside MPI Guard

If a code path writes/reads files, wrap in:

```cpp
if (amrex::ParallelDescriptor::IOProcessor()) {
    // Only rank 0 does file I/O
}
```

---

## Part F — Testing and Validation

### F1. Always Test with `erf.ucm.ucm_debug = true` in Phase Tests

Forces verbose output, making silent data corruption visible.

### F2. Multi-Rank Test Minimum: 2×2 Spatial Domain

Ensures inter-rank ghost cell exchange is exercised.

### F3. Bit-for-Bit Reproducibility Test (Phase 1.4+)

Run same inputs with UCM enabled and disabled, verify plotfiles identical (no physics regression).

---

## Quick Checklist for New SLUCM Phase

**Before opening PR:**

- [ ] File header: `/**`, `@file`, `@brief`, extended description, References
- [ ] Parameter struct (if new): sections with rulers, inline `///< `, defaults  
- [ ] Prerequisites (if new): clear error messages, startup banner
- [ ] Grid function (if new): 3 steps (extract 2D, refine, reuse DM), comment algorithm
- [ ] Make.package: Grid/Prerequisites first, then physics, then headers
- [ ] CMake block: option + compile_definitions + target_sources + include_directories
- [ ] ERF.H: Guarded `#ifdef ERF_USE_UCM`, comment every block `// UCM Phase X.Y`
- [ ] ERF.cpp: Early ParmParse, late prerequisites check
- [ ] All console output prefixed: `[UCM]` or `[UCM DEBUG]`
- [ ] No hardcoded level `0` (use `int lev` parameter instead)
- [ ] Use `amrex::Vector` not `std::vector`
- [ ] MultiFab ghost cells: `IntVect(1,1,0)` for 2D slabs
- [ ] All public functions have `@param[in]`, `@return`, `@throws` Doxygen blocks
- [ ] Test builds and runs with `-DERF_ENABLE_UCM=ON` and `=OFF`
- [ ] No new compiler warnings (`-Wall -Wextra -Wpedantic`)

---

## Known Issues & Workarounds (None Yet — Phase 1.1)

As bugs are discovered and fixed in later phases, document here:

- **Phase X.Y – Issue:** Description. **Workaround/Fix:** Details.

---

## References

- `Source/UrbanCanopy/UCM_DEVELOPMENT.md` — Phase roadmap
- `Source/UrbanCanopy/ERF_UCM.H` — Design contracts
- `Source/Dust/DUST_DEVELOPMENT.md` — Dust module reference
- `Source/LNG/LNG_MPI_SKILLS.md` — MPI lessons for 2D slab modules
