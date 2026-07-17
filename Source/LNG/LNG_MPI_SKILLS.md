# ERF-LNG Module — Development Skills & Bug Fix Reference

Complete record of lessons learned across all PRs (#161–#166 plus Phase 6
post-merge fixes) during the development of the ERF-LNG hazardous gas
dispersion module on the `ERF-HazGas` branch. Use this as a checklist before
merging any new AMReX sub-grid 2D module into ERF.

---

## PR History

| PR | Title | Phase | Merged |
|----|-------|-------|--------|
| [#161](https://github.com/hgopalan/ERF/pull/161) | LNG Phase 1: module stub, parameter framework, debug infrastructure | 1 | 2026-07-15 |
| [#162](https://github.com/hgopalan/ERF/pull/162) | LNG Phase 2: evaporation physics documentation | 2 | 2026-07-15 |
| [#163](https://github.com/hgopalan/ERF/pull/163) | Phase 2 fixes: test reorganisation, domain params, CSV output decoupling | 2 | 2026-07-15 |
| [#164](https://github.com/hgopalan/ERF/pull/164) | LNG Phase 3: one-way 2D→3D vapor injection coupling | 3 | 2026-07-16 |
| [#165](https://github.com/hgopalan/ERF/pull/165) | LNG Phase 4: live wind & surface field extraction | 4 | 2026-07-16 |
| [#166](https://github.com/hgopalan/ERF/pull/166) | LNG Phase 5: gravity current PDEs, Richardson transition, flammability | 5 | 2026-07-16 |
| [#167](https://github.com/hgopalan/ERF/pull/167) | LNG Phase 6: Output & Visualization: plotfile, receptor sampling, CSV | 6 | 2026-07-16 |

Post-merge multi-rank bugs were found during integration testing and fixed
directly on `ERF-HazGas` (not via additional PRs). Phase 7 will be implemented
in a new PR.

---

## Part A — Architecture & Design Rules

### A1. Follow the Dust Module Pattern Exactly

Every LNG sub-system has a proven Dust analog. When in doubt, copy the Dust
implementation and substitute `LNG` for `Dust`. Diverging silently is the
primary source of all bugs found in this project.

| LNG Component | Dust Analog |
|---|---|
| `ERF_LNGGrid.H/cpp` | `ERF_DustGrid.H/cpp` |
| `ERF_LNGPrerequisites.H/cpp` | `ERF_DustPrerequisites.H/cpp` |
| `ERF_LNGLayer.H/cpp` | `ERF_DustLayer.H/cpp` |
| `ERF_LNGAtmCoupling.H/cpp` | `ERF_DustAtmCoupling.H/cpp` |
| `ERF_LNGWindExtract.H/cpp` | `ERF_DustWindExtract.H/cpp` |
| `ERF_LNGStatsOutput.H` | `ERF_DustStatsOutput.H` |
| `ERF_LNGPlotfile.H/cpp` | `ERF_DustPlotfile.H/cpp` |

### A2. Build System — Register in Both Make and CMake

Every new `.cpp` file must be registered in **both** build systems or one will
produce linker errors.

```makefile
# Source/LNG/Make.package
ifeq ($(USE_LNG), TRUE)
  CEXE_sources += ERF_LNGNewFile.cpp
endif
```

```cmake
# CMake/BuildERFExe.cmake
if(ERF_ENABLE_LNG)
  target_sources(... ERF_LNGNewFile.cpp)
endif()
```

**Lesson from PR #165:** `ERF_LNGWindExtract.cpp` was added to `Make.package`
but initially omitted from `CMake/BuildERFExe.cmake` → CMake linker error.

**Lesson from Phase 6:** `ERF_LNGPlotfile.cpp` — same issue. Check both files
every time a new source is added.

### A3. Test Case Domain Must Match the Dust Reference Baseline

All LNG canonical tests use the same ATM domain as `DustCriticalMaterials`:

```
geometry.prob_extent = 3000 3000 1024
amr.n_cell           = 8 8 64   (or 32 32 64 for multi-rank tests)
erf.fixed_dt         = 0.5
zlo.type             = "surface_layer"
erf.pbl_type         = "MRF"
erf.transport_scalar = true
```

The `LNG_Output` Phase 6 test uses `amr.n_cell = 32 32 64` with
`grid_ratio = 4` (128×128 LNG grid) — the most demanding configuration.
All previous single-rank domains work; the multi-rank configuration is the
definitive integration test.

**Lesson from PR #163:** LNG_BuildOnly originally used a toy 16×16×8 domain
that was incompatible with SurfaceLayer → had to be updated to match Dust.

### A4. CSV Output Must Not Be Gated on `lng_debug`

Diagnostic CSV rows must always be written regardless of debug level.
Only the console print lines should be gated:

```cpp
// ✅ CORRECT
append_lng_stats_phase2(...);        // always runs — all ranks reduce first
if (m_params.lng_debug) {
    amrex::Print() << "[LNG DEBUG] ..."; // gated
}
```

**Lesson from PR #163:** CSV rows were originally inside `if (lng_debug)` →
silent data loss in production runs.

### A5. `amrex::Vector` Cannot Be Implicitly Constructed from `std::vector`

`amrex::Vector<T>` is built on `std::vector<T>` but does not define an
implicit conversion constructor. Any function returning `std::vector<std::string>`
(such as `lng_plotfile_var_names()`) must be explicitly converted.

```cpp
// ❌ WRONG — compile error: no viable conversion
amrex::Vector<std::string> var_vec = lng_plotfile_var_names();

// ✅ CORRECT — explicit range construction
auto std_vec = lng_plotfile_var_names();
amrex::Vector<std::string> var_vec(std_vec.begin(), std_vec.end());
```

**File fixed:** `ERF_LNGPlotfile.cpp` line 126

---

## Part B — MPI Multi-Rank Rules

### B1. The IOProcessor Guard Must Come After All MPI Collectives

**This was the final hang bug — found in `ERF_LNGStatsOutput.H`.**

`MultiFab::sum()`, `max()`, `min()`, `contains_nan()` are `MPI_Allreduce`
calls. Every rank must call them. Placing the IO guard before them causes
rank 0 to wait in `MPI_Allreduce` while rank 1 has already returned.

```cpp
// ❌ WRONG — deadlock with 2+ ranks
if (!amrex::ParallelDescriptor::IOProcessor()) return;  // rank 1 exits here
amrex::Real val = mf.sum(0);   // rank 0 hangs — rank 1 never enters

// ✅ CORRECT
amrex::Real val = mf.sum(0);   // all ranks participate in MPI_Allreduce
if (!amrex::ParallelDescriptor::IOProcessor()) return;  // only rank 0 writes
```

**Files affected:** `ERF_LNGStatsOutput.H::append_lng_stats_phase2`

**Checklist:** Search every stats/output header for `IOProcessor()` guards.
If any `MultiFab` operation follows the guard, it is a bug.

**The exception:** `write_lng_stats_header()` is safe because it contains
no MultiFab operations — the guard can come first there.

### B2. Z-Decomposition Must Be Prevented

AMReX splits the domain in z when `amrex.max_grid_size_z` is not set.
The LNG (and Dust) 2D slab operations require each rank to own a full z-column.

```
# inputs file — must be set for ANY run with LNG or Dust enabled
amrex.max_grid_size_z = 64   # equal to amr.n_cell z-dimension
```

Enforce at initialization in `ERF_LNGPrerequisites.cpp` (Check 3):

```cpp
int domain_nz = atm_domain.length(2);
for (int i = 0; i < ba_atm.size(); ++i) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        ba_atm[i].length(2) == domain_nz,
        "[LNG] Cannot decompose in z. Set: amrex.max_grid_size_z = <nz>");
}
```

**Note:** `amrex.max_grid_size_z` is an **unrecognised** ParmParse key for
the ATM solver — it will appear in the "Unused ParmParse Variables" summary
at the end of the run. This is expected and harmless:
```
Unused ParmParse Variables:
  [TOP]::amrex.max_grid_size_z(nvals = 1)  :: [64]
```

### B3. MPI-Safe Wind Extraction via ParallelCopy

`fill_lng_wind_from_interpolation` iterates over the **LNG grid** but accesses
`xvel_mf`/`yvel_mf`/`z_phys_cc_mf` on the **ATM grid**. With `grid_ratio>1`
and 2+ ranks, a LNG tile on rank 0 may try to read ATM data owned by rank 1.

**Fix:** Copy ATM fields into rank-local scratch MultiFabs before reading:

```cpp
// In LNGLayer::advance() — BEFORE fill_lng_wind_from_interpolation
const amrex::Periodicity& per = geom_atm->periodicity();
m_xvel_atm->ParallelCopy(*xvel_mf,       0, 0, 1, 0, 1, per);
m_yvel_atm->ParallelCopy(*yvel_mf,       0, 0, 1, 0, 1, per);
m_zphys_atm->ParallelCopy(*z_phys_cc_mf, 0, 0, 1, 0, 1, per);
m_xvel_atm->FillBoundary(per);
m_yvel_atm->FillBoundary(per);
m_zphys_atm->FillBoundary(per);

fill_lng_wind_from_interpolation(
    *m_lng_wind_ref, *m_xvel_atm, *m_yvel_atm, *m_zphys_atm, ...);
```

Scratch MFs (`m_xvel_atm`, `m_yvel_atm`, `m_zphys_atm`) are allocated on the
**ATM BoxArray/DistributionMapping** in `initialize()` and declared as
`unique_ptr<MultiFab>` members in `ERF_LNGLayer.H`.

### B4. FillBoundary Must Pass Periodicity

All `FillBoundary` calls on the LNG grid must pass `geom_lng.periodicity()`.
Without it, ghost cells at inter-rank boundaries are never exchanged.

```cpp
// ❌ WRONG — ghost cells not exchanged across ranks
lng_pool_mask.FillBoundary();

// ✅ CORRECT
lng_pool_mask.FillBoundary(geom_lng.periodicity());
```

**Files fixed:** `ERF_LNGPool.cpp`, `ERF_LNGGravityCurrent.cpp`

Apply this to every `FillBoundary` call on every LNG MultiFab.

### B5. `average_down` Requires Geometry Arguments

The no-geometry overload of `amrex::average_down` skips the MPI halo exchange
for multi-rank coarsening.

```cpp
// ❌ WRONG — no MPI exchange
amrex::average_down(lng_evap_flux, lng_flux_atm,
                    0, 1, amrex::IntVect(ratio, ratio, 1));

// ✅ CORRECT — mirrors ERF_DustAtmCoupling.H exactly
amrex::average_down(lng_evap_flux, lng_flux_atm,
                    geom_lng, geom_atm,
                    0, 1, amrex::IntVect(ratio, ratio, 1));
```

**File fixed:** `ERF_LNGAtmCoupling.H::coarsen_lng_flux_to_atm`

### B6. `ReduceSum` with `amrex::Loop` Is Not MPI-Collective

`amrex::ReduceSum` with a host `amrex::Loop` lambda performs a CPU-only
thread-local reduction — it does NOT call `MPI_Allreduce`. Results are
per-rank and wrong on multi-rank runs.

```cpp
// ❌ WRONG — not reduced across ranks
amrex::Long cnt = amrex::ReduceSum(mf, 0,
    [=](amrex::Box const& bx, ...) -> amrex::Long {
        amrex::Long n = 0;
        amrex::Loop(bx, [&](int i, int j, int k){ ++n; });
        return n;
    });

// ✅ CORRECT — MPI_Allreduce internally
amrex::Real flag_sum = mf.sum(0);
amrex::Long cnt = static_cast<amrex::Long>(flag_sum);
```

For Ri max/min: write into a scratch `MultiFab` via `ParallelFor`, then call
`scratch.max(0)` / `scratch.min(0)` — both are MPI-collective.

**File fixed:** `ERF_LNGGravityCurrent.cpp` debug block

### B7. Plotfile Write: 4-Step MPI-Safe Pattern

Plotfile writing must follow this strict 4-step pattern to avoid race
conditions with file system operations on multi-rank runs:

```cpp
// Step 1: IOProcessor creates directories; Barrier before collective write
if (amrex::ParallelDescriptor::IOProcessor()) {
    amrex::UtilCreateDirectory(plotfilename, 0755);
    amrex::UtilCreateDirectory(plotfilename + "/Level_0", 0755);
}
amrex::ParallelDescriptor::Barrier();   // ALL ranks wait for dirs to exist

// Step 2: ALL ranks write their owned FABs (MPI-collective)
amrex::VisMF::Write(mf, plotfilename + "/Level_0/Cell");

// Step 3: IOProcessor writes Header and metadata (IOProcessor guard safe here
//         — no MPI collectives inside)
if (amrex::ParallelDescriptor::IOProcessor()) {
    // write Header file
    // write JSON sidecar
}
amrex::ParallelDescriptor::Barrier();   // ALL ranks wait for Header to exist
```

The `Barrier()` after directory creation in Step 1 is **required** — without
it, non-IO ranks may call `VisMF::Write` before the directory exists on the
shared filesystem.

**File:** `ERF_LNGPlotfile.cpp`

---

## Part C — Grid Construction Rules

### C1. LNGGrid Domain Extents Must Come from `atm_geom.Domain()`

`atm_ba[0]` covers only one rank's subdomain. With 2+ ranks, using it gives
wrong LNG grid size.

```cpp
// ❌ WRONG — atm_ba[0] is only rank 0's subdomain
int ihi = atm_ba[0].bigEnd(0);

// ✅ CORRECT — full domain from geometry, rank-independent
const auto& atm_domain = atm_geom.Domain();
int ihi = atm_domain.bigEnd(0);
```

**File fixed:** `ERF_LNGGrid.cpp`

### C2. `MultiFab::size()` Returns Box Count, Not Cell Count

```cpp
// ❌ WRONG — returns number of boxes (e.g. 2 with 2 MPI ranks)
amrex::Long total = mf.size();

// ✅ CORRECT — returns total number of grid points
amrex::Long total = static_cast<amrex::Long>(mf.boxArray().numPts());
```

**File fixed:** `ERF_LNGLayer.cpp` Phase 5 `gc_active_cells` diagnostic
(was printing `-2` = 2 boxes minus 4 mixed cells)

### C3. FillBoundary After Every Physics Update on LNG Grid

After any `ParallelFor` that modifies a MultiFab used in subsequent stencil
operations (e.g. pressure gradients in gravity current), call `FillBoundary`
so ghost cells are valid for the next iteration:

```cpp
// After advance_gravity_current:
lng_gc_h.FillBoundary(geom_lng.periodicity());
lng_gc_u.FillBoundary(geom_lng.periodicity());
lng_gc_v.FillBoundary(geom_lng.periodicity());
lng_gc_ri_flag.FillBoundary(geom_lng.periodicity());

// After pool operations:
lng_pool_depth.FillBoundary(geom_lng.periodicity());
lng_pool_mask.FillBoundary(geom_lng.periodicity());
```

### C4. Multi-Component MultiFabs: Correct Component Indexing

For a 2-component MultiFab (e.g. `m_lng_wind_ref` storing u and v), use
component indices 0 and 1 explicitly when copying to a single-component
output MultiFab:

```cpp
// m_lng_wind_ref has ncomp=2: comp 0 = u, comp 1 = v
// Copy into plotfile MultiFab at components 11 and 12:
amrex::MultiFab::Copy(mf, *lng_layer.get_wind_ref(), 0, 11, 1, 0);  // u
amrex::MultiFab::Copy(mf, *lng_layer.get_wind_ref(), 1, 12, 1, 0);  // v
```

Wrong component indexing produces silent zero values in the plotfile.

**File:** `ERF_LNGPlotfile.cpp`

---

## Part D — Function Signature Rules

### D1. Pass `geom_lng` to Pool Functions That Do FillBoundary

When pool functions need `FillBoundary`, they must receive the geometry
to get the periodicity. Add `const amrex::Geometry& geom_lng` to the
signature rather than deriving it from the MultiFab (MultiFab does not
store its geometry).

```cpp
// ✅ Correct signatures (post-fix)
void update_pool_mask(amrex::MultiFab& lng_pool_mask,
                      const amrex::MultiFab& lng_pool_depth,
                      const amrex::Geometry& geom_lng,          // required
                      amrex::Real depth_threshold = 1.0e-4);

void deplete_pool_from_evaporation(amrex::MultiFab& lng_pool_depth,
                                   const amrex::MultiFab& lng_evap_flux,
                                   const amrex::Geometry& geom_lng,     // required
                                   amrex::Real rho_LNG,
                                   amrex::Real dt,
                                   bool lng_debug);
```

**Files fixed:** `ERF_LNGPool.H`, `ERF_LNGPool.cpp`, call sites in
`ERF_LNGLayer.cpp`

---

## Part E — Output & Diagnostics Rules

### E1. `write_output` Duplicate-Call Guard

`write_output` is called from both `WriteAtIntermediateTime` and
`WriteAtFinalTime`. Without a guard, the last step writes two CSV rows
and potentially two plotfiles.

```cpp
// In ERF_LNGLayer.H private section:
int m_last_output_step = -1;

// In ERF_LNGLayer.cpp::write_output():
if (nstep == m_last_output_step) return;
m_last_output_step = nstep;
```

### E2. Debug Prints Are Not Barriers — Don't Use for Hang Diagnosis

Adding `amrex::Print()` before/after suspected hang points does NOT synchronise
ranks. To diagnose hangs, use explicit `ParallelDescriptor::Barrier()` calls
temporarily, then **remove them before merging** — they impose a global sync
at every call site and will show in benchmarks.

```cpp
// ✅ Temporary hang diagnosis — REMOVE before merging
amrex::Print() << "[LNG DEBUG] PRE-WIND rank=" 
               << amrex::ParallelDescriptor::MyProc() << "\n";
amrex::ParallelDescriptor::Barrier();
// ... suspect call ...
amrex::Print() << "[LNG DEBUG] POST-WIND rank=" 
               << amrex::ParallelDescriptor::MyProc() << "\n";
amrex::ParallelDescriptor::Barrier();
```

The first `POST-` that only appears from one rank identifies the hang location.

### E3. `[LNG DEBUG]` vs `[LNG]` Prefix Convention

| Prefix | When to use |
|---|---|
| `[LNG]` | Always-on info (init summary, phase summaries, plotfile writes) |
| `[LNG DEBUG]` | `if (lng_debug)` per-step diagnostics |
| `[LNG DEBUG3]` | `if (verbose >= 3)` field min/max tables |
| `[LNG COUPLING]` | Phase 3 source term (from `ERF_Advance.cpp`) |
| `[LNG WARNING]` | Non-fatal issues (file open failures, etc.) |

### E4. Receptor Sampling Must Reduce Across Ranks

Point receptor sampling involves locating a cell by physical coordinate
across the LNG grid. The cell may be owned by any rank. The correct pattern
is to check ownership locally and then reduce:

```cpp
// Each rank checks if it owns the receptor cell
amrex::Real local_val = 0.0;
for (amrex::MFIter mfi(conc_sfc); mfi.isValid(); ++mfi) {
    if (mfi.validbox().contains(receptor_iv)) {
        local_val = conc_sfc[mfi](receptor_iv, 0);
    }
}
// MPI reduce — all ranks participate
amrex::ParallelDescriptor::ReduceRealSum(local_val);
// IOProcessor writes
if (amrex::ParallelDescriptor::IOProcessor())
    out << local_val << "\n";
```

**Never use IOProcessor guard before the `ReduceRealSum`** — same rule as B1.

---

## Part F — Inputs File Checklist

Every LNG test inputs file must contain these entries. Missing any one of
them causes silent misconfiguration or a hang.

```
# ── Mandatory for any LNG run with 2+ MPI ranks ──────────────────────────────
amrex.max_grid_size_z = <same as amr.n_cell z>   # prevents z-decomposition
# Note: will appear in "Unused ParmParse Variables" — this is expected.

# ── ERF ATM baseline (must match DustCriticalMaterials) ──────────────────────
erf.prob_name         = "ABL"
geometry.prob_extent  = 3000 3000 1024
geometry.is_periodic  = 1 1 0
zlo.type              = "surface_layer"
erf.most.z0           = 0.1
erf.most.zref         = 24.0
erf.most.surf_temp_flux = 0.0
zhi.type              = "SlipWall"
erf.pbl_type          = "MRF"
erf.transport_scalar  = true
erf.use_gravity       = true
erf.sum_interval      = 1    # required — all-ranks reduction fixed in ERF_LNGStatsOutput.H

# ── LNG mandatory parameters ──────────────────────────────────────────────────
erf.lng.enable        = true
erf.lng.grid_ratio    = <integer>       # must divide amr.n_cell x,y
erf.lng.atm_feedback  = 1.0
erf.lng.lfl_vol_fraction = 0.05
erf.lng.ufl_vol_fraction = 0.15

# ── Phase 6 output (optional but recommended) ─────────────────────────────────
erf.lng.lng_plot_int     = 5            # write plotfile every 5 steps
erf.lng.lng_plot_prefix  = "plt_lng_"
erf.lng.lng_diag_file    = "lng_diag.csv"

# ── Phase 6 receptor sampling (optional) ─────────────────────────────────────
erf.lng.lng_receptor_names = "center" "downwind"
erf.lng.lng_receptor_x     = 1500.0   1700.0
erf.lng.lng_receptor_y     = 1500.0   1500.0
```

---

## Part G — Pre-Merge Checklist for New LNG Phases

Use this checklist before opening a PR for any new LNG phase:

**MPI Safety**
- [ ] All `MultiFab::sum/max/min` calls are BEFORE any `IOProcessor()` guard
- [ ] All `FillBoundary` calls pass `geom_lng.periodicity()`
- [ ] No `amrex::ReduceSum` with `amrex::Loop` for cross-rank quantities
- [ ] No `average_down` calls without geometry arguments
- [ ] No direct `.array(mfi)` access on a MultiFab from a different grid
- [ ] Plotfile write follows the 4-step Barrier pattern (Rule B7)
- [ ] Point receptor sampling uses `ReduceRealSum` (Rule E4)

**Grid Construction**
- [ ] Domain extents use `atm_geom.Domain()`, not `atm_ba[0]`
- [ ] Cell counts use `boxArray().numPts()`, not `size()`
- [ ] New MultiFabs live on the correct BoxArray/DM (LNG grid or ATM grid)
- [ ] ATM-grid scratch MFs allocated for any cross-grid access
- [ ] Multi-component MultiFab copies use explicit component indices

**Build System**
- [ ] New `.cpp` files registered in `Make.package`
- [ ] New `.cpp` files registered in `CMake/BuildERFExe.cmake`
- [ ] No `std::vector` → `amrex::Vector` implicit conversions

**Testing**
- [ ] `amrex.max_grid_size_z = <nz>` in inputs file
- [ ] Test runs with 1 proc (single-rank correctness)
- [ ] Test runs with 2 procs `mpirun -n 2` (multi-rank MPI safety)
- [ ] `erf.sum_interval = 1` set and all steps complete
- [ ] "Unused ParmParse Variables: max_grid_size_z" appears at end (expected)
- [ ] `RHO LNG > 0` in `TIME=` summary by step 2 (confirms scalar injection)

**Output**
- [ ] CSV writes NOT gated on `lng_debug`
- [ ] `write_output` has `m_last_output_step` duplicate guard
- [ ] All console prints use correct `[LNG]` prefix tier
- [ ] Plotfile written at correct intervals (check `[LNG] Writing LNG plotfile`)

**Documentation**
- [ ] `LNG_DEVELOPMENT.md` updated with new phase section
- [ ] New parameters added to `ERF_LNGParams.H` with Doxygen comments
- [ ] Pass criteria listed in PR description
- [ ] `LNG_MPI_SKILLS.md` updated with any new bugs/fixes

---

## Summary Table — All Bugs Found and Fixed

| # | Bug | Phase Found | Symptom | File(s) Fixed | Rule |
|---|---|---|---|---|---|
| 1 | `IOProcessor()` guard before `MultiFab::sum/max` | Post-PR#166 | Hang after step 1 `write_output` | `ERF_LNGStatsOutput.H` | B1 |
| 2 | Z-decomposition not prevented | Post-PR#166 | Wrong `gc_active_cells`, partial slab operations | inputs file + `ERF_LNGPrerequisites.cpp` | B2 |
| 3 | Cross-rank ATM array access in wind extraction | Post-PR#165 | Hang in `fill_lng_wind_from_interpolation` with `grid_ratio>1` | `ERF_LNGLayer.cpp` (ParallelCopy scratch) | B3 |
| 4 | `FillBoundary` without periodicity | Post-PR#164 | Stale ghost cells at rank boundaries → wrong physics | `ERF_LNGPool.cpp`, `ERF_LNGGravityCurrent.cpp` | B4 |
| 5 | `average_down` without geometry arguments | Post-PR#164 | Wrong coarsening on 2+ ranks → wrong scalar injection | `ERF_LNGAtmCoupling.H` | B5 |
| 6 | `atm_ba[0]` for LNG domain extents | Post-PR#163 | Wrong LNG grid size with 2+ ranks | `ERF_LNGGrid.cpp` | C1 |
| 7 | `ReduceSum` with host `Loop` not MPI-collective | Post-PR#166 | Wrong `gc_active_cells=8190`, diagnostic hang | `ERF_LNGGravityCurrent.cpp` | B6 |
| 8 | `MultiFab::size()` used for cell count | Post-PR#166 | `gc_active_cells=-2` displayed | `ERF_LNGLayer.cpp` | C2 |
| 9 | `deplete_pool` / `update_pool_mask` missing `geom` arg | Post-PR#164 | `MultiFab::Geom` compile error | `ERF_LNGPool.H/cpp`, `ERF_LNGLayer.cpp` | D1 |
| 10 | `write_output` called twice at final step | Post-PR#163 | Duplicate CSV rows at last step | `ERF_LNGLayer.cpp` | E1 |
| 11 | CSV output gated on `lng_debug` | PR#163 | No CSV data in production runs | `ERF_LNGLayer.cpp` | A4 |
| 12 | `ERF_LNGWindExtract.cpp` not in CMake | PR#165 | CMake linker error | `CMake/BuildERFExe.cmake` | A2 |
| 13 | LNG_BuildOnly on toy 16×16×8 domain | PR#163 | SurfaceLayer incompatible | inputs_lng_buildonly | A3 |
| 14 | `debug Barrier()` left in production code | Post-PR#166 | Global MPI sync at every step | `ERF_LNGLayer.cpp` | E2 |
| 15 | `std::vector` → `amrex::Vector` implicit conversion | Phase 6 | Compile error in `ERF_LNGPlotfile.cpp` line 126 | `ERF_LNGPlotfile.cpp` | A5 |
| 16 | Plotfile missing `Barrier` after dir creation | Phase 6 | Race condition: non-IO ranks call VisMF before dir exists | `ERF_LNGPlotfile.cpp` | B7 |
| 17 | `ERF_LNGPlotfile.cpp` not in CMake | Phase 6 | CMake linker error | `CMake/BuildERFExe.cmake` | A2 |

---

## Confirmed Working Configurations

The following have been verified to complete all 20 steps without hanging:

| Configuration | n_cell | grid_ratio | LNG cells | Procs | Status |
|---|---|---|---|---|---|
| LNG_GravityCurrent baseline | 8×8×64 | 2 | 16×16 | 1 | ✅ |
| LNG_GravityCurrent multi-rank | 32×32×64 | 4 | 128×128 | 2 | ✅ |
| LNG_Output Phase 6 | 32×32×64 | 4 | 128×128 | 2 | ✅ |
| DustIntegration (reference) | 32×32×64 | 4 | 128×128 | 2 | ✅ |

Key observable indicators of a healthy run:
- `[LNG DEBUG] Prerequisite check 3 passed` at startup
- `RHO LNG > 0` in `TIME=` summary from step 2 onward
- `gc_active_cells` equals `boxArray().numPts()` (16384 for 128×128)
- `[LNG] Writing LNG plotfile plt_lng_XXXXX` at correct intervals
- `[LNG DEBUG] Phase 6: receptor sampling step=N  n_receptors=2`
- `Unused ParmParse Variables: amrex.max_grid_size_z` at end (expected)