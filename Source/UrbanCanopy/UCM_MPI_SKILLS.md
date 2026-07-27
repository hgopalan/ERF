# ERF-SLUCM Module — Development Skills & Bug Fix Reference

Complete record of lessons learned during development of the ERF-SLUCM urban canopy model on the `ERF-SLUCM-phase-1.1` and subsequent branches. Use this as a checklist before merging any new 2D slab module variant.

This file will grow phase by phase as new technical challenges are discovered and documented.

---

## Part A — Architecture & Design Rules

### A1. Follow the Dust and Fire Module Patterns Exactly

Every UCM sub-system has a proven Dust analog. When in doubt, copy the Dust implementation and substitute UCM for module name.

| UCM Component | Reference Analog |
|---------------|------------------|
| `ERF_UCMParams.H` | `Source/Dust/ERF_DustParams.H` |
| `ERF_UCMGrid.H/cpp` | `Source/Dust/ERF_DustGrid.H/cpp` |
| `ERF_UCMPrerequisites.H/cpp` | `Source/Dust/ERF_DustPrerequisites.H/cpp` |
| `(Phase 1.2) ERF_UCMFields.H` | `(Phase 2) Source/Dust/ERF_DustLayer.H` (field container pattern) |
| `(Phase 1.2) ERF_UCMAllocate.H/cpp` | `Source/LNG/ERF_LNGLayer.cpp` (allocation pattern) |
| `(Phase 1.3) ERF_UCMAtmCoupling.H/cpp` | `Source/Dust/ERF_DustAtmCoupling.H/cpp` |
| `(Phase 1.3) ERF_UCMWindExtract.H/cpp` | `Source/Dust/ERF_DustWindExtract.H/cpp` |

Diverging silently is the primary source of bugs. Document deviations with `TODO(UCM PhaseX.Y): rationale` comments.

### A1.1 Grid Alignment and DistributionMapping Reuse (Phase 1.2 NEW)

The UCM 2D grid is constructed using the exact algorithm from `Source/Dust/ERF_DustGrid.cpp`:

1. **Extract 2D slab:** Take k=0 slice from each box in `ba_atm` via explicit `Box::setSmall(2,0)` and `setBig(2,0)`
2. **Refine BoxArray:** Apply `amrex::refine()` by `IntVect(grid_ratio, grid_ratio, 1)` — this increases box sizes but does NOT change the number of boxes
3. **Reuse DistributionMapping:** Because refinement preserves box count, we reuse `dm_atm` directly — box `i` in refined array is owned by the same rank as box `i` in atmospheric array
4. **Construct 2D Geometry:** Hi-index formula `new_hi = old_hi * grid_ratio + (grid_ratio - 1)`; physical domain x-y unchanged from ATM; z extent set to dummy 1 m

**Critical invariant:** Box `i` in `ba_ucm` is ALWAYS owned by the same rank as box `i` in `ba_atm`. This eliminates inter-rank communication during UCM–ATM coupling, dramatically improving performance.

**Reference:** `Source/LNG/LNG_MPI_SKILLS.md` Section B5 and `Source/Dust/ERF_DustGrid.cpp`

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

### C3. Coarsen + Inject: UCM → ATM Coupling (Phase 1.4 NEW)

**Pattern:** Mirror exactly from `Source/Fire/ERF_FireAtmCoupling.H` and `Source/Dust/ERF_DustAtmCoupling.H`.

```cpp
// Step 1: Coarsen UCM grid flux to ATM grid
coarsen_ucm_flux_to_atm(Q_atm_out, Q_ucm, geom_ucm, geom_atm, grid_ratio, lev);

// Step 2: Apply exponential vertical decay to cc_source
apply_ucm_tendency_to_cc_source(cc_source, Q_atm, z_phys_cc, S_old, geom_atm,
                                is_urban_atm, alpha, feedback, has_moisture, lev);
```

**Coarsening rules:**
- When `grid_ratio == 1`: use `amrex::MultiFab::Copy(dst, src, 0, 0, 1, 0)`
- When `grid_ratio > 1`: use `amrex::average_down(src, dst, 0, 1, IntVect(grid_ratio, grid_ratio, 1))`
- Result is on ATM grid; proceed to injection

**Exponential injection algorithm:**
```cpp
// For each ATM column (i, j):
for (int k = klo; k <= khi; ++k) {
    Real z_sfc = z_phys_cc(i, j, klo);
    Real z_k = z_phys_cc(i, j, k) - z_sfc;
    Real hfx_k = (Q_sfc / Cp) * std::exp(-z_k / alpha);
    
    // Finite difference to get divergence
    Real hfx_kp1 = (Q_sfc / Cp) * std::exp(-(z_k + dz(k)) / alpha);
    Real theta_tend = -rho(k) * (hfx_kp1 - hfx_k) / dz(k);
    
    cc_source(i, j, k, RhoTheta_comp) += feedback * theta_tend;
}
```

**Critical for MPI:** When grid_ratio > 1 and multiple ranks, UCM and ATM grids are **colocated on the same ranks** (Phase 1.2 contract: DistributionMapping reuse). This eliminates inter-rank communication during coarsening.

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

---

## Known Issues & Workarounds

### Phase 1.3 – Bug 7: ParallelFor + Array4 API Misuse (Fixed [`f0b2ef3`](https://github.com/hgopalan/ERF/commit/f0b2ef3))

**Issue:** Phase 1.3 agent incorrectly used AMReX ParallelFor and Array4 APIs:
1. Used `MultiFab[mfi].array()` returning by reference → GPU/memory safety bug
2. Used box-based ParallelFor lambda `[=] AMREX_GPU_DEVICE(const Box& tbx)` → incompatible with AMReX kernel signature
3. Inside ParallelFor, called `LoopConcurrentOnCpu(tbx, ...)` → nested loop incorrect
4. Used `amrex::Copy(...)` function → does not exist in AMReX

**Workaround/Fix:**
- **Always:** Use `mf.array(mfi)` (by value) or `mf.const_array(mfi)` (read-only). Never `mf[mfi].array()` with `auto&`.
- **ParallelFor signature:** `[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { ... }`. Triple int signature, not Box-based.
- **No nested loops:** No `LoopConcurrentOnCpu` inside a `ParallelFor` lambda. Use flat kernel.
- **MultiFab copy:** Use `amrex::MultiFab::Copy(dst, src, srccomp, dstcomp, ncomp, ngrow)` only.

**Prevention:** All Phase 1.4+ code must follow these rules. Grep checklist (acceptance criteria):
```
grep -rn "\[mfi\]\.array()" Source/UrbanCanopy/  → MUST be 0 hits
grep -rn "LoopConcurrentOnCpu" Source/UrbanCanopy/  → MUST be 0 hits (inside ParallelFor)
grep -rn "amrex::Copy(" Source/UrbanCanopy/  → MUST be 0 hits (only MultiFab::Copy)
```

### Phase 1.3 – Bug 8: SurfaceLayer Accessor API Misuse (Fixed [`8c1cddb`](https://github.com/hgopalan/ERF/commit/8c1cddb))

**Issue:** Phase 1.3 agent misunderstood SurfaceLayer's public API:
1. Called `m_SurfaceLayer->get_u_star()[lev]` — missing `lev` argument
2. Called `m_SurfaceLayer->get_theta_star()` — function does not exist
3. Assumed accessors return `Vector<MultiFab*>` → actually return single `MultiFab*`

**Workaround/Fix:**
- **Accessor signature:** All star-value accessors take `int lev` argument: `get_u_star(lev)`, `get_t_star(lev)`, `get_q_star(lev)`.
- **No vector indexing:** Never `get_u_star()[lev]`. Always pass `lev` directly: `get_u_star(lev)`.
- **No theta_star:** Use `get_t_star(lev)` for potential temperature star, not `get_theta_star()`.
- **Dereferencing:** All return `MultiFab*`, so dereference: `*m_SurfaceLayer->get_u_star(lev)` to use by reference.

**Correct usage pattern:**
```cpp
// Phase 1.3 advance call:
auto& u_star = *m_SurfaceLayer->get_u_star(lev);
auto& t_star = *m_SurfaceLayer->get_t_star(lev);
auto& q_star = *m_SurfaceLayer->get_q_star(lev);

// Then use u_star, t_star, q_star as MultiFab references
// (not subscript: u_star[lev] is wrong; u_star is already at lev)
```

**Prevention:** Phase 1.4+ grep checklist:
```
grep -rn "get_theta_star" Source/UrbanCanopy/  → MUST be 0 hits
grep -rn "get_u_star()\[" Source/UrbanCanopy/  → MUST be 0 hits
grep -rn "get_t_star()\[" Source/UrbanCanopy/  → MUST be 0 hits
grep -rn "get_q_star()\[" Source/UrbanCanopy/  → MUST be 0 hits
```

---

## Phase 2.5: Manual weighted aggregation UCM (2D) -> ATM (2D slab)

Unlike Phase 1.4's `amrex::average_down`, Phase 2.5 aggregation needs
per-cell urban-mask weighting and running statistics (`sum` for mean,
`sum_of_squares` for variance). Use a two-nested loop inside a ParallelFor
on the ATM box:

```cpp
ParallelFor(bx_atm, [=] AMREX_GPU_DEVICE (int I, int J, int K) noexcept {
    int n_urb = 0;
    Real Hsum = 0.0, Hsum2 = 0.0;
    for (int dj = 0; dj < gr; ++dj)
    for (int di = 0; di < gr; ++di) {
        const int i_ucm = I*gr + di;
        const int j_ucm = J*gr + dj;
        if (ur_a(i_ucm, j_ucm, 0) == 1) {
            n_urb += 1;
            const Real Hb = Hb_a(i_ucm, j_ucm, 0);
            Hsum  += Hb;
            Hsum2 += Hb * Hb;
        }
    }
    // ... compute mean and std ...
});
```

Conservation convention: total flux is preserved with `(1/N_total)` weighting,
NOT `(1/N_urban)`. Comment above the kernel MUST state which convention is in use.

**Phase 2.5 convention B (conservation-preserving, area-averaged, no injection-side reweight):**
```cpp
Q_atm = (1 / N_total) * sum(is_urban * Q_ucm)
```
This is the reference convention, used on ERF-Hazard branch. The injection kernel
(`apply_ucm_tendency_to_cc_source`) reads Q_atm AS-IS with NO multiplication by `f_urb_atm`.
Total urban heat production is preserved by construction:
  ```
  sum_over_ATM(Q_atm * dA_atm) = sum_over_UCM_urban(Q_ucm * dA_ucm)
  ```
Advantages:
- No division-by-zero risk when f_urb=0.
- No silent over-injection in partial-urban ATM cells.
- Matches ERF-Fire convention (Source/Fire/ERF_FireAtmCoupling.H).

**Lesson (Phase 2.5 conservation fix):** Do NOT post-hoc divide an area-averaged flux
by `f_urb` on the coarsening side unless the injection side multiplies it back (convention A).
Convention B (pure area average, no divide, no reweight) is the reference. Anything else
silently over-injects in partial-urban ATM cells by a factor of `1/f_urb` (e.g., 4× too high
in a 25%-urban cell).

### Phase 2.5-fix2: Three More Lessons

**Lesson 15 (Phase 2.5-fix2):** Do NOT post-hoc divide an area-averaged flux by `f_urb` on the
coarsening side unless the injection side multiplies it back. Convention B (pure area average, no divide)
is the reference. Anything else silently over-injects in partial-urban ATM cells.

**Lesson 16 (Phase 2.5-fix2):** CSV readers must strip UTF-8 BOM and leading/trailing whitespace
before header comparison. Error messages on header mismatch must hex-dump the actual bytes read;
do not use marker characters like `!!!` that can visually corrupt the display.

**Lesson 17 (Phase 2.5-fix2):** Facet-split fluxes (H_road, H_wall, H_roof) must all follow the same
convention: pre-weighted by their area fraction. Never mix per-facet-area and pre-weighted conventions
in the same set. Enforce with a canonical test containing uniform geometry (all three fractions equal),
which forces the three fluxes to be equal.

---

---

## Phase 3.5a-hotfix Cascade — Lessons 18-24

Seven interrelated bugs discovered and fixed during Phase 3.5a → 3.5c integration on `UCMBostonStabilityCorrection`. Full technical narrative in `UCM_DEVELOPMENT.md`; grep patterns and prevention checks below.

### Lesson 18: Numerical Impossibility Is a Coefficient Bug, Not a Physics Bug

If a simulated system exhibits energy loss (or gain) at a rate that exceeds any physical mechanism (e.g., slab cooling at 0.09 K/step ≈ 28 kW/m²), the root cause is almost always a numerical coefficient error, NOT a physics gap.

**Diagnostic pattern:**
```
1. Compute the observed dT/dt from the log
2. Multiply by (rho_cp * dz) to get the implied energy flux [W/m²]
3. Compare to the maximum possible physical flux (SW peak ~1000, LW ~500, sensible ~500)
4. If observed >> physical maximum: it's a coefficient/sign bug in the solver
```

**Anti-pattern:** Speculating about missing physics (radiation trapping, moisture, etc.) when the arithmetic doesn't match any physical mechanism.

### Lesson 19: Compensating Bugs Are the Worst Bugs

When two bugs partially cancel each other's effect, fixing one in isolation makes the simulation WORSE, not better. This misleads developers into reverting the correct fix.

**Case study (Phase 3.5a-hotfix):**
- Bug (old): Wall LW absorbed used full sky LW (~356 W/m²) — over-count
- Bug (still): Wall LW emitted full T_skin⁴ (~410 W/m²) — over-count

Both bugs made wall lose ~50 W/m² net. When we "fixed" absorption to use SVF (correctly reducing to ~103 W/m²), emission was still full — now wall lost ~300 W/m² net. Simulation collapsed harder.

**Prevention:**
- When fixing a physics component, list ALL related terms that could have compensating errors
- Fix them together as a coherent physical unit, not piecemeal
- Test the fix under conditions where each term dominates individually (e.g., no SW, uniform canyon T, high SW noon)

**Grep check for future LW code:**
```
grep -n "LW_down\|LW_absorbed\|LW_emitted\|sigma_sb.*T_skin" Source/UrbanCanopy/
```
Each match must document its SVF/canyon geometry assumption.

### Lesson 20: Sign Convention Documentation Must Live at BOTH Ends of an Interface

Bug 5 (slab-BC sign) occurred because Newton's `ERF_UCMSEBSolver.H` documented "positive H = surface to atm" and Slab's `ERF_UCMSlabConduction.H` documented "positive Q_top = into slab from top." Both docstrings were correct in isolation. The bug lived in the WRAPPER (`advance_slab_conduction_mfi()`), which passed H → Q_top without noting these are OPPOSITE conventions.

**Rule:** Every function boundary where signed physical quantities cross must have an inline comment at the call site documenting BOTH conventions and the reconciliation. Don't rely on docstrings alone — the interface point is where the bug lives.

**Grep check:**
```
grep -B2 -A2 "H_flux\|Q_top\|Q_ucm\|Q_atm" Source/UrbanCanopy/ | grep -v "//"
```
Any signed flux passed across a function boundary without a preceding comment justifying its sign is a latent bug.

### Lesson 21: Tridiagonal Solvers Need a Unit-Test Conservation Check

Bug 6 (TDMA all-plus) would have been caught in seconds by a 3-line unit test:

```cpp
Real T[4] = {293.15, 293.15, 293.15, 293.15};
advance_slab_conduction_column(T, 0.0, 293.15, 1.0, 1.8e6, 0.075, 1.0, 4);
// Assert T[i] == 293.15 for all i to machine precision
```

Zero forcing + uniform IC = zero change. Any drift = coefficient error.

**Rule for future numerical kernels:**
- Every tridiagonal, pentadiagonal, or matrix inversion solver in `Source/UrbanCanopy/` must have a paired unit test verifying its identity behavior with zero forcing.
- Test lives in `Exec/CanonicalTests/SLUCM/UnitTests/` (create if not exists).
- CI runs unit tests on every PR.

### Lesson 22: Astronomical Formulae Must Cite Reference and Include Sentinel Check

Bug 4 (hour angle from midnight vs. noon) fell through the cracks because:
1. The reference (Michalsky 1988) was cited in the docstring but no equation number given
2. No sentinel value ("at noon at Boston in summer, cos_zenith ≈ 0.94, SW ≈ 880 W/m²") was documented
3. The formula was "reviewed" but not tested against a known input/output pair

**Rule for future physics formulae with strong convention dependencies:**
- Cite specific equation number, not just paper (Michalsky 1988, Eq. 3, not just "Michalsky 1988")
- Include a comment with an expected input/output pair for verification
- Add a startup diagnostic printing the sentinel value; a mismatch is immediate cause for investigation

**Example (correct format for the fixed radiation code):**
```cpp
// Michalsky (1988), Eq. 3: solar zenith angle
// Sentinel test: at t_local=noon (12:00 LST), lat=42.36°N, day=172 (June 21):
//   Expected cos_zenith ≈ 0.945, SW_down ≈ 883 W/m² with τ=0.7
// If your startup diagnostic prints values > 5% off these, the formula is wrong.
```

### Lesson 23: Instrumentation-First Debugging Beats Speculative Coding

Every one of the seven bugs was fixed by first adding a diagnostic that produced a specific number violating expectation, and only then changing code. Attempts to jump to speculative fixes (used with a coding agent early in the cycle) produced multiple wrong fixes that had to be reverted, wasting hours per iteration.

**Rule:**
- Before proposing any code change, propose an instrumentation change that would reveal whether the code is producing correct or incorrect intermediate values.
- Run with instrumentation. Get a specific number. Compare to expected. Only THEN change code.
- Log the instrumentation itself as a `[UCM][phase-tag][diagnostic]` line so it can be retained or removed later as a unit.

**Anti-pattern (to avoid):** "Let me fix the LW trapping and the sign convention and the Newton solver all at once." When five things change at once and the sim gets worse, you've lost causality.

### Lesson 24: CSV Material Parameters Are Physics, Not Configuration

Bug 2 (`k_therm = 50 W/m/K` for a wall assembly) illustrates that CSV inputs are not just user preferences — they encode physical assumptions that must be validated.

**Rule:**
- Every `materials.csv` row must have a `description` field with a citation source (WRF UDA, ASHRAE handbook, etc.) — the field was in the reference schema but silently dropped during copy-paste from a docs example.
- The `description` should include the physical interpretation: "brick_concrete" is fine, but "brick_concrete: effective wall assembly, k=1.1 (ASHRAE 90.1 Table 5.5-4, wall type W12)" is auditable.
- Prerequisite check should sanity-bound: `k_therm ∈ [0.05, 5.0] W/m/K` for building assemblies; `k_therm ∈ [0.1, 3.0] W/m/K` for pavements. Values outside these bounds should FAIL prerequisites with a clear error message pointing to the CSV row and the expected range.

**CSV header validation reminder (Lesson 16 from Phase 2.5-fix2):**
- Reader must strip UTF-8 BOM and leading/trailing whitespace
- Error messages on mismatch must hex-dump actual bytes read
- **Do NOT use marker characters like `!!!`** that can visually corrupt display

---

## Consolidated Grep Checklist for Phase 3.5a-hotfix

Add to `run_all_regressions.sh` or CI:

```bash
# Bug 1 & 6 (TDMA coefficients)
grep -E "a_coeff|c_coeff|c_top" Source/UrbanCanopy/ERF_UCMSlabConduction.H | \
  grep -v "^\s*//" | grep -v "\-Fo" && echo "WARNING: positive Fo in TDMA off-diagonal"

# Bug 3 (MOST sign)
grep -E "rho_ref \* Cp \* u_star \* t_star" Source/UrbanCanopy/ERF_UCMLayer.cpp | \
  grep "\-rho" && echo "WARNING: minus sign in H_base; check WRF sign convention"

# Bug 4 (hour angle)
grep -E "hour_angle.*=.*time_s.*/.*3600" Source/UrbanCanopy/ERF_UCMRadiationForcing.H | \
  grep -v "12\.0\|noon" && echo "WARNING: hour_angle not measured from noon"

# Bug 5 (slab BC sign)
grep -A1 "Q_top =" Source/UrbanCanopy/ERF_UCMSlabConduction.H | \
  grep "H_flux_arr" | grep -v "\-H_flux" && echo "WARNING: Q_top not negated from H_flux"

# Bug 7 (canyon LW)
grep -E "LW_wall_eff|LW_road_eff|LW_canyon" Source/UrbanCanopy/ERF_UCMLayer.cpp | \
  head -1 || echo "WARNING: canyon LW trapping not implemented"

# CSV sanity
awk -F, 'NR>1 && $5 > 5.0 {print "WARNING: k_therm=" $5 " exceeds building assembly bound"}' \
  Exec/CanonicalTests/SLUCM/*/materials.csv
```

Any WARNING output from the above is a merge-blocker for Phase 3.5+ code.

---

## Known Issues & Workarounds

As bugs are discovered and fixed in later phases, document here:

- **Phase X.Y – Issue:** Description. **Workaround/Fix:** Details.

---

## References

- `Source/UrbanCanopy/UCM_DEVELOPMENT.md` — Phase roadmap
- `Source/UrbanCanopy/ERF_UCM.H` — Design contracts
- `Source/Dust/DUST_DEVELOPMENT.md` — Dust module reference
- `Source/LNG/LNG_MPI_SKILLS.md` — MPI lessons for 2D slab modules
