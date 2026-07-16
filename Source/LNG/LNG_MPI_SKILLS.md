# ERF-LNG Module — Development Skills & Bug Fix Reference

Complete record of lessons learned across all PRs (#161–#166) during the
development of the ERF-LNG hazardous gas dispersion module on the
`ERF-HazGas` branch. Use this as a checklist before merging any new
AMReX sub-grid 2D module into ERF.

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

Post-merge multi-rank bugs were found during integration testing and fixed
directly on `ERF-HazGas` (not via additional PRs).

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

Scratch MFs are allocated on the **ATM BoxArray/DistributionMapping** in
`initialize()` and declared as `unique_ptr<MultiFab>` in `ERF_LNGLayer.H`.

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

---

## Part D — Function Signature Rules

### D1. Pass `geom_lng` to Pool Functions That Do FillBoundary

When pool functions need `FillBoundary`, they must receive the geometry
to get the periodicity. Add `const amrex::Geometry& geom_lng` to the
signature rather than deriving it from the MultiFab.

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
`WriteAtFinalTime`. Without a guard, the last step writes two CSV rows.

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
at every call site.

### E3. `[LNG DEBUG]` vs `[LNG]` Prefix Convention

| Prefix | When to use |
|---|---|
| `[LNG]` | Always-on info (init summary, phase summaries) |
| `[LNG DEBUG]` | `if (lng_debug)` per-step diagnostics |
| `[LNG DEBUG3]` | `if (verbose >= 3)` field min/max tables |
| `[LNG COUPLING]` | Phase 3 source term application (from ERF_Advance.cpp) |
| `[LNG WARNING]` | Non-fatal issues (file open failures, etc.) |

---

## Part F — Inputs File Checklist

Every LNG test inputs file must contain these entries. Missing any one of
them causes silent misconfiguration or a hang.

```
# ── Mandatory for any LNG run with 2+ MPI ranks ──────────────────────────────
amrex.max_grid_size_z = <same as amr.n_cell z>   # prevents z-decomposition

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
erf.sum_interval      = 1    # required (must call all-ranks reduction, now fixed)

# ── LNG mandatory parameters ──────────────────────────────────────────────────
erf.lng.enable        = true
erf.lng.grid_ratio    = <integer>       # must divide amr.n_cell x,y
erf.lng.atm_feedback  = 1.0
erf.lng.lfl_vol_fraction = 0.05
erf.lng.ufl_vol_fraction = 0.15
```

---

## Part G — Pre-Merge Checklist for New LNG Phases

Use this checklist before opening a PR for any new LNG phase:

**MPI Safety**
- [ ] All `MultiFab::sum/max/min` calls are BEFORE any `IOProcessor()` guard
- [ ] All `FillBoundary` calls pass `geom_lng.periodicity()`
- [ ] No `amrex::ReduceSum` with `amrex::Loop` for cross-rank quantities
- [ ] No `average_down` calls without geometry arguments
- [ ] No direct `.array(mfi)` access on a MultiFab from a different grid when iterating

**Grid Construction**
- [ ] Domain extents use `atm_geom.Domain()`, not `atm_ba[0]`
- [ ] Cell counts use `boxArray().numPts()`, not `size()`
- [ ] New MultiFabs live on the correct BoxArray/DM (LNG grid or ATM grid)
- [ ] ATM-grid scratch MFs allocated for any cross-grid access

**Build System**
- [ ] New `.cpp` files registered in `Make.package`
- [ ] New `.cpp` files registered in `CMake/BuildERFExe.cmake`

**Testing**
- [ ] `amrex.max_grid_size_z = <nz>` in inputs file
- [ ] Test runs with 1 proc (single-rank correctness)
- [ ] Test runs with 2 procs (multi-rank MPI safety)
- [ ] `erf.sum_interval = 1` set and simulation completes all steps

**Output**
- [ ] CSV writes NOT gated on `lng_debug`
- [ ] `write_output` has `m_last_output_step` duplicate guard
- [ ] All console prints use correct `[LNG]` prefix tier

**Documentation**
- [ ] `LNG_DEVELOPMENT.md` updated with new phase section
- [ ] New parameters added to `ERF_LNGParams.H` with Doxygen comments
- [ ] Pass criteria listed in PR description

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