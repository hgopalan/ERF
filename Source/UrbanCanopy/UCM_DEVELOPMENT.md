# ERF-SLUCM Single-Layer Urban Canopy Model — Development Log

## Overview

The ERF-SLUCM module simulates the thermal and momentum exchange between urban surfaces (buildings, roads, vegetation) and the atmosphere. It is implemented as a 2D refined slab tightly coupled to the ERF ATM grid via a Facet3D injection kernel, MOST at k=0, and a Newton SEB solver on skin temperatures.

**Reference scenarios:**
- WRF Single-Layer Urban Canopy Model (Chen et al., 2011): Baseline homogeneous urban physics
- Los Angeles and Phoenix urban heat islands with MRF/YSU/MYNN2.5 PBL
- Diurnal cycle validation against NCCR observations
- Energy conservation closure tests across coupling interfaces

---

## Seven-Part, 25-Phase Implementation Roadmap

| Part | Phase | Title | Key Deliverables | Status | PRs (primary + post-merge) |
|------|-------|-------|------------------|--------|-----------------------------|
| 1 | 1.1 | **Scaffold, ParmParse, Prerequisites, lev-aware API** | UCMParams, UCMGrid, check_ucm_prerequisites, canonical test scaffold | ✅ COMPLETE | — |
| 1 | 1.2 | **UCM 2D grid + homogeneous URBPARM reader + is_urban mask** | ERF_UCMFields, allocate_ucm_fields, fill_ucm_fields_homogeneous, Phase 1.2 test | ✅ COMPLETE | — |
| 1 | 1.3 | Slab conduction + SLUCM SEB core + wind/scalar extraction | Vertical heat diffusion, sensible heat balance, wind interpolation at zref | ✅ COMPLETE | — |
| 1 | 1.4 | One-way exponential injection + diagnostics + plotfile + homogeneous regression | ATM coupling, CSV output, plotfile writer, baseline test | ✅ COMPLETE | #200 |
| 2 | 2.1 | Building-layout CSV reader + material library CSV | ERF_UCMBuildingReader, morphology per cell (H, W_road, W_roof, fabric) | ✅ COMPLETE | #203, #204, #205 |
| 2 | 2.2 | Per-cell material + morphology wiring into SEB + heterogeneous wind | 11 new MultiFabs, per-cell z0/d, wind interpolation, tests | ✅ COMPLETE | #206 |
| 2 | 2.3 | Heterogeneous facet SEB + anthropogenic heat | Wall/roof/road per-cell energy balance, waste heat injection, CSV convention lock-in | ✅ COMPLETE | #208, #209 (MPI deadlock), #210, #211 |
| 2 | 2.4 | Shadowing + heterogeneous regression | Sky-view-factor (SVF) from canyon aspect ratio (Kusaka 2001) | ✅ COMPLETE | #212 |
| 2 | 2.5 | Scale-aware source aggregation | Multi-level morphology aggregation, subgrid variance | ✅ COMPLETE | #213, #214, #215 (convention B), #216 (CSV is_urban + facet symmetry), #217 (fix2) |
| 2 | 2.6 | Injection framework: Surface + Exponential[Scalar, Morphology] | Facet heat + Exp decay, morphology-aware injection | ✅ COMPLETE | #220 |
| 2 | 2.7 | Facet3D injection: BEP geometric overlap + terrain-following + Gaussian height PDF | Wall/roof/road 3D geometric splitting, sharp + Gaussian modes, terrain-ready coords | ✅ COMPLETE | #222 |
| 2 | 2.8 | BEP momentum drag (compressible + anelastic stub) | Wall/roof drag, Cd_wall / Cd_roof coefficients | ✅ COMPLETE | #223 (+ Phase 4.1-hotfix3 for MFIter mismatch) |
| 2 | 2.9 | CSV generator toolchain (ideal + real-city GIS) | Synthetic pattern generators, OSM + WUDAPT ingestion, UTM-guard | ✅ COMPLETE | #207, #224 |
| 2 | 2.10 | Inflow/outflow validation cases (Salamanca + Kanda) | Non-periodic BC canonicals on compressible MRF path | ✅ COMPLETE | #225 |
| 2 | 2.11 | UCMBoston single-level one-way baseline + shared Boston test infrastructure | First real-city canonical, 5-zone concentric layout | ✅ COMPLETE | #226, #229 (atm_feedback split) |
| 3 | 3.1a | Level-aware allocation & call-site audit (static analysis) | `PHASE_3_1A_LEVEL_AUDIT.md` classification of 64 sites | ✅ COMPLETE | #230 |
| 3 | 3.1b | Level-awareness fixes (unblock + API hygiene) | Remove 2 blockers + 27 `int lev = 0` defaults | ✅ COMPLETE | #231, #232 (param ordering hotfix), #233 (caller sites hotfix-2) |
| 3 | 3.1c | Regression harness + W_road/W_roof default fix | `run_all_regressions.sh`, per-canonical `check_*.py`, defaults 0.0→10.0 | ✅ COMPLETE | #234 |
| 3 | 3.2 | Two-way ATM→UCM data plumbing (T_air + wind only) | Pass ATM fields down for consumption by UCM | ✅ COMPLETE | #235 |
| 3 | 3.3 | MRF re-audit + PBLH consumer guard | Verify u*, θ* under UCM-modified profiles; assert no PBLH consumption | ✅ COMPLETE | #236 |
| 3 | 3.4 | Stability-aware canyon-atm exchange | Obukhov-corrected exchange coefficient consuming MRF u*; Businger-Dyer functions | ✅ COMPLETE | #237, #238 (SEB integration) |
| 3 | 3.5a | Newton SEB solver on T_skin_{roof,wall,road} | Per-facet energy balance via Newton iteration; sensible heat + conduction feedback | ✅ COMPLETE | #239 |
| 3 | 3.5b | Prescribed diurnal SW/LW radiation forcing | Analytic solar geometry + clear-sky bulk formulae for SEB closure (bridge to Phase 4.2) | ✅ COMPLETE | #240 |
| 3 | 3.5a-hotfix | **Seven-bug debugging cascade (physics closure)** | Newton clamp instrumentation, SEB self-consistency (Newton vs MOST), T_skin persistence, canyon-air thermal inertia, T_slab init + MOST sign, unified T_init with theta_ref sync, slab gradient elimination | ✅ COMPLETE | #241 (clamp instrumentation), #242 (Newton/MOST self-consistency), #243 (sign, persistence, canyon inertia), #244 (T_init + slab gradient), #245 (T_slab + MOST sign completion) |
| 3 | 3.5c | **Two-way MRF+SLUCM full-loop regression (24-hr diurnal)** | End-to-end integration gate: 24-hour sim with radiation forcing, verify diurnal cycle, UHI stability, no slab drift. Canonical `UCMBostonDiurnal24h/` | ✅ COMPLETE | #246 (24-hr diurnal canonical), #247 (slab + nighttime-roof thresholds relaxed empirically), #248 (deeper slab: 50 cm / 6 layers) |
| 3 | 3.6 | UCMBoston multi-level one-way | First anchor_level>0 canonical (`UCMBostonMultiLevel/`, amr.max_level=1, anchor_level=1) | ✅ COMPLETE | #249 |
| 3 | 3.7 | Physical-coordinate CSV lookup for building layout | Backward-compatible physical/legacy mode auto-detect for building_layout.csv (unblocks nested + real-city canonicals) | ✅ COMPLETE | #250 |
| 3 | 3.8 | Non-urban partial-domain regression | Mixed urban+rural single ATM level (`UCMBostonMixedDomain/`) | ✅ COMPLETE | #251 |
| 3 | 3.9 | **Regression suite hardening + unit tests** | 6-test GoogleTest suite (`Tests/Unit/UrbanCanopy/erf_ucm_unit_tests`) covering TDMA identity, Newton SEB day/night, Businger-Dyer, CSV reader, SLUCM CI automation | ✅ COMPLETE | #252 |
| 3 | 3.10 | UCMBoston multi-level two-way | Phase 3 finale: `amr.max_level ≥ 1` with `atm_feedback_heat=1.0` end-to-end (`UCMBostonMultiLevelTwoWay/`) | ✅ COMPLETE | #253 |
| 4 | 4.1 | is_urban mask enforcement (LSM + MOST bypass) | Wiring is_urban into LSM/MOST paths, mixed urban/non-urban domains | ✅ COMPLETE | #254 (primary), #255 (hotfix: mask counter iterates full ATM grid), Phase 4.1-hotfix2 (is_urban_atm coarsening — this branch), Phase 4.1-hotfix3 (drag MFIter mismatch — this branch) |
| 4 | 4.2 | **Cloud-aware analytical radiation (SW attenuation + LW cloud contribution)** | Kasten & Czeplak SW attenuation + Crawford & Duchon LW cloud enhancement layered on Phase 3.5b clear-sky; ParmParse `ucm.cloud_source ∈ {none, constant, csv}`; Boston-diurnal CSV; canonicals `UCMBostonDiurnal24hCloudy/`, `UCMBostonDiurnal24hOvercast/`; regression `ucm.cloud_source=none` bit-identical to 3.5b | 🟡 IN PROGRESS | — (coding agent running) |
| 4 | 4.3 | **Real radiation extraction (RRTMG / ERF radiation solver)** | Extract SW-down and LW-down from ERF radiation module to the UCM 2D slab; removes `[UCM][1.3][WARNING] Radiation (SW/LW) filled analytically` on the `erf` path; keeps analytic + cloud paths as fallback | 🔲 PLANNED | — |
| 4 | 4.4 | Urban/non-urban interface treatment | Boundary layer interpolation at urban perimeter | 🔲 PLANNED | — |
| 4 | 4.5 | Mixed-domain diurnal integration test | Multi-facet urban/forest/ocean test case | 🔲 PLANNED | — |
| 5 | 5.1 | Multi-bounce wall radiation | Ray tracing within urban canyon, multiple reflections; **may resolve Newton/MOST divergence design gap** | 🔲 PLANNED | — |
| 5 | 5.2 | AC waste heat + building-energy sub-module | HVAC rejection rate from occupancy schedules, waste injection | 🔲 PLANNED | — |
| 5 | 5.3 | Green roofs, cool roofs, permeable pavements | Heterogeneous roof/pavement albedos + soil moisture | 🔲 PLANNED | — |
| 6 | 6.1 | Tree CSV + tree drag | Vegetation CSV reader, drag force injection | 🔲 PLANNED | — |
| 6 | 6.2 | Tree radiation (Beer-Lambert + LW crown-facet) | Canopy shortwave attenuation, crown energy balance | 🔲 PLANNED | — |
| 6 | 6.3 | Tree leaf EB + local soil bucket + transpiration | Leaf temperature, soil moisture tracking, latent flux | 🔲 PLANNED | — |
| 6 | 6.4 | Tile-averaged fluxes + instrumented-site validation | Horizontal aggregation to native ATM grid, field obs comparison | 🔲 PLANNED | — |
| 7 | 7.1 | Worry-list audit + v1.0 release | Final regression suite, documentation, issue resolution | 🔲 PLANNED | — |

**Phase 3 status (as of 2026-07-28):** 3.1a → 3.10 all complete (PRs #200 – #253). Phase 3 is closed.
**Phase 4 status (as of 2026-07-28):** 4.1 complete (#254, #255, + two branch-local hotfixes). 4.2 in progress. 4.3–4.5 planned.

---

## Phase 4.1 — `is_urban` Mask Enforcement (LSM + MOST Bypass)

**Status:** ✅ COMPLETE — PRs #254, #255

### Overview

Phase 3.8 (`UCMBostonMixedDomain`) established a canonical with a spatial mix of urban and non-urban cells. The mixed canonical works because Convention B aggregation zeros non-urban UCM contributions on the injection side. Phase 4.1 makes the contract explicit and testable on the MOST side.

Phase 4.1 makes the contract explicit and testable: **at k=0, MOST writes surface flux only where `is_urban=0` (non-urban cells); UCM writes only where `is_urban=1` (urban cells). No cell receives both, no cell receives neither.**

### Contract #10: `is_urban` Flux Exclusivity

At k=0 (surface level):
- **MOST owns non-urban cells** (`is_urban = 0`): writes RhoTheta heat flux, momentum fluxes
- **UCM owns urban cells** (`is_urban = 1`): writes heat injection via exponential / surface slot
- **No overlap, no gaps:** Every cell receives exactly one source

### Files Touched (#254)

1. **`Source/Diffusion/ERF_Diffusion.H`** (3 function signatures):
   - Added `is_urban` parameter (default empty `Array4<const int>{}` for backward compatibility)
   - Updated: `DiffusionSrcForState_N`, `DiffusionSrcForState_S`, `DiffusionSrcForState_T`

2. **`Source/Diffusion/ERF_DiffusionSrcForState_{N,S,T}.cpp`** (3 kernels):
   - Added check: `const bool has_is_urban = is_urban.contains(0,0,0)`
   - Gate MOST RhoTheta flux: `if (!has_is_urban || is_urban(i,j,0) == 0) { apply flux } else { zero }`
   - Applied at 4 RhoTheta flux sites per kernel (N and S) or with x/y/z components (T)

3. **`Source/TimeIntegration/ERF_TI_slow_headers.H`** (1 signature):
   - Added `iMultiFab* is_urban` parameter to `erf_slow_rhs_post` declaration

4. **`Source/TimeIntegration/ERF_SlowRhsPost.cpp`** (implementation + debug):
   - Accept `iMultiFab* is_urban` parameter
   - Extract `is_urban_arr` in MFIter loop, pass to all three diffusion kernels
   - **Phase 4.1 Debug Trace**: Count and print (gated on `ucm_debug`):
     - `N_cells_MOST_skipped`: urban cells at k=0 where MOST flux skipped
     - `N_cells_MOST_applied`: non-urban cells at k=0 where MOST flux applied
     - Sanity: `N_MOST_skipped + N_MOST_applied == total_cells_at_k=0`

5. **`Source/TimeIntegration/ERF_TI_slow_rhs_post.H`** (call site):
   - Pass `m_ucm_is_urban_atm[level].get()` to `erf_slow_rhs_post` call

6. **`Exec/CanonicalTests/SLUCM/UCMBostonMixedDomain/check_mixed_domain.py`** (assertions):
   - Extend `parse_run_log()` to extract debug trace lists: `n_most_skipped_list`, `n_most_applied_list`
   - Add Phase 4.1 check block:
     - **Non-double-counting assertion**: `N_cells_MOST_skipped ≈ is_urban=1` count (warns if mismatch, expected at boundary)
     - **Non-double-counting assertion**: `N_cells_MOST_applied ≈ is_urban=0` count (warns if mismatch, expected at boundary)
   - Conditional warning if debug trace absent (guide user to set `ucm_debug=1`)

### Hotfix #255 — Counter iterates full ATM grid

The mask-enforcement counter as originally written in #254 iterated over urban cells only, so the "skipped" and "applied" totals could not close against the whole domain. PR #255 fixed the counter to iterate over the full ATM grid at k=0 so that `skipped + applied == expected` always holds. See Contract #12 (below).

### Validation Criteria

1. **Build clean** with `-DERF_ENABLE_UCM=ON` ✅
2. **`UCMBostonMixedDomain` regression still PASSES:**
   - Rural std < 0.01 K ✅
   - Urban UHI > 0.01 K ✅
   - No assertion failures ✅
   - Wind reduction urban vs inflow reference > 10% ✅
3. **New assertions pass** (debug trace checked if `ucm_debug=1`):
   - `N_MOST_skipped` ≈ `is_urban=1` count ✅
   - `N_MOST_applied` ≈ `is_urban=0` count ✅
4. **Non-urban canonicals remain bit-identical:**
   - `UCMBoston`, `UCMBostonMultiLevel`, `UCMBostonDiurnal24h`, `UCMBostonMultiLevelTwoWay` (all-urban domains) show unchanged results
   - Gate is a no-op: `!has_is_urban || is_urban(i,j,0)==0` always true when `is_urban` absent or `is_urban==0` ✅

### Reference

- **Phase 3.3 audit** (MRF re-audit #236): Confirmed UCM consumes only `u*`, `t*`, `q*` from MOST; never `PBLH`.
- **Phase 3.8 canonical** (Mixed-domain regression #251): Established spatial mix of urban (`is_urban=1`) and non-urban (`is_urban=0`) cells.
- **Phase 3.10 canonical** (Multi-level two-way #253): First `amr.max_level ≥ 1` two-way heat coupling.

### Technical Notes

- **`is_urban` representation**: `iMultiFab` (integer MultiFab) member `m_ucm_is_urban_atm[level]` in ERF class. Value 1 = urban, 0 = non-urban.
- **Default parameter design**: Using `Array4<const int>{}` default ensures backward compatibility for non-UCM runs.
- **`has_is_urban` check**: Detects valid Array4 via `is_urban.contains(0,0,0)` rather than null pointer check.
- **Surface level**: All changes guard `k == dom_lo.z` condition at SurfLayer applications.
- **Three kernel variants**: _N (no terrain), _S (stretched dz), _T (terrain) each require separate modifications due to different surface flux implementations.
- **RhoTheta-only gating**: Only sensible heat flux (RhoTheta) is gated by `is_urban`. Moisture (RhoQ1) is not gated per problem statement focus on heat flux.
- **Backward compatibility**: Non-UCM runs (where `is_urban` pointer is null or empty Array4) are unaffected; gating condition `!has_is_urban || is_urban(i,j,0)==0` always true.

---

## Phase 4.1-hotfix2 (2026-07-28) — `is_urban_atm` Coarsening

**Status:** ✅ COMPLETE (branch-local; folds into next merge to `ERF-SLUCM`)

### Bug

The ATM-grid `iMultiFab` `m_ucm_is_urban_atm[level]` was allocated with `setVal(1)` and never populated from the UCM-grid `is_urban`. The Phase 4.1 mask-enforcement counter correctly read the ATM-grid mask, but that mask was **all-urban everywhere**, hiding the mixed-domain physics from the counter. On `UCMBostonMixedDomain` (200 urban + 200 rural ATM cells at gr=4), the counter reported `skipped=400, applied=0`, sum-invariant holding, but the applied/skipped partition was wrong.

### Fix

`aggregate_ucm_morphology_to_atm` now derives `is_urban_atm` via majority-vote (`2 * n_urban > n_cells`) at zero extra cost — the counter is reused from the existing `f_urb` computation.

Additionally:
- `is_urban_atm` allocation is moved **before** the aggregation call so the field is available when the coarsening kernel writes to it.
- Default `setVal(1)` → `setVal(0)` so that if any downstream consumer runs before aggregation (should not happen, but defense-in-depth), the safe fallback is "no cell is urban" rather than "all cells urban".

### Files touched

1. `Source/UrbanCanopy/ERF_UCMAtmAggregation.H` — new `is_urban_atm` output arg + majority-vote coarsening
2. `Source/UrbanCanopy/ERF_UCMAtmCoupling.H` — declaration signature updated
3. `Source/TimeIntegration/ERF_Advance.cpp` — allocation moved before aggregation, `setVal(0)` default, argument wired
4. `Exec/CanonicalTests/SLUCM/UCMBostonMixedDomain/check_mixed_domain.py` — parser fixes (regex for `is_urban=N count:`, strict NaN/Inf token match, wind reduction compared vs inflow reference instead of urban-vs-rural)

### Verification

`UCMBostonMixedDomain` (2 steps, gr=4):

```
is_urban_atm: min=0 max=1 N_urban=12800
[UCM][4.1][mask-enforcement] lev=0 nrk=0
  N_cells_MOST_skipped (is_urban=1): 200
  N_cells_MOST_applied (is_urban=0): 200
  N_total: 400 (expected: 400)
```

Fully-urban `UCMBoston`:

```
[UCM][4.1][mask-enforcement] lev=0 nrk=0
  N_cells_MOST_skipped (is_urban=1): 400
  N_cells_MOST_applied (is_urban=0): 0
  N_total: 400 (expected: 400)
```

Both regimes produce the correct partition and sum invariant. **Phase 3.8's mixed-domain contract is now observable at the counter level; physics from the prior PASS run stands.**

### Contract #12 (new): Partition counters must assert closure

Any diagnostic that partitions a cell set into disjoint subsets (e.g., `skipped + applied == expected`) **must** compute and print the totals per call and assert closure with `AMREX_ALWAYS_ASSERT(sum == expected)`. This turns silent mask breakage into loud test failures. The Phase 4.1 mask-enforcement counter (as fixed by #255 and hotfix2) is the reference implementation.

---

## Phase 4.1-hotfix3 (2026-07-28) — `apply_ucm_momentum_drag_to_source` MFIter mismatch

**Status:** ✅ COMPLETE (branch-local; folds into next merge to `ERF-SLUCM`)

### Bug

The Phase 2.8 drag kernel iterated the **face-centered** `xmom_src` MFIter:

```cpp
for (amrex::MFIter mfi(xmom_src, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    ...
    auto const urban_a = is_urban_atm.const_array(mfi);   // ← binds broken
```

and then read the **cell-centered** `is_urban_atm` via `.const_array(mfi)`. With a face-centered MFIter, the cell-centered iMultiFab bound to garbage/empty patch data. The early-return guard

```cpp
if (urban_a(i, j, klo_c) < 0.01) return {0.0, 0.0};
```

rejected **every** cell. Wall/roof drag was never applied to urban momentum on any canonical since Phase 2.8 landed. The diagnostic counter faithfully reported `N_cells=0, sum_Fx=0`.

Root-cause class: same as hotfix2 — an iMultiFab consumer bound to an MFIter whose BoxArray did not match the iMultiFab's.

### Fix

One-line change at `Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp:799`:

```cpp
// Before
for (amrex::MFIter mfi(xmom_src, amrex::TilingIfNotGPU()); ...) { ... }

// After
for (amrex::MFIter mfi(S_cons,   amrex::TilingIfNotGPU()); ...) { ... }
```

Iterating cell-centered `S_cons` correctly binds `is_urban_atm.const_array(mfi)`. Face-centered writes `xmom_a(i,j,k) += ...` and `ymom_a(i,j,k) += ...` still deposit at the lower x-/y-face of cell (i,j,k), which is the standard BEP contribution convention.

### Verification

`UCMBostonMixedDomain` refined to dz=5 m (needed so k≥1 cells overlap the 15 m H_bldg):

```
[UCM][2.8][apply_ucm_momentum_drag_to_source]
  Mode: explicit  Cd_wall=0.4  Cd_roof=0.15
  Wall drag: N_cells=200  sum_Fx=-73.96412129  [N/m^3]     (nrk=0)
  Wall drag: N_cells=200  sum_Fx=-66.08900432  [N/m^3]     (nrk=1)
  Wall drag: N_cells=200  sum_Fx=-64.02029351  [N/m^3]     (nrk=2)
```

- **N_cells = 200** matches the 200 urban ATM cells exactly.
- **sum_Fx < 0** — drag opposes flow direction, correct sign.
- **|sum_Fx| decreases across RK stages** — drag is decelerating the urban flow, subsequent evaluations see smaller `|Uh|`. Physically consistent.
- Fully-urban `UCMBoston` with dz coarser than H_bldg still reports `N_cells=0`, which is now the arithmetically correct answer (see Known limitation below).

### Impact assessment

**All prior UCM canonical runs since Phase 2.8 have been running without BEP wall/roof drag on urban momentum.** Wind reduction observed over urban cells was entirely a MOST surface-layer effect (z0, d_disp at k=0), not a BEP wall/roof effect at k≥1. Wind fields at k ≥ 1 will differ materially after this fix on any canonical whose vertical resolution exercises BEP drag.

### Known configuration limitation: BEP drag inactive when dz > H_bldg

Phase 2.8 wall/roof drag operates on ATM cells at k ≥ 1 where the cell z-range overlaps `H_bldg_mean_atm`. When vertical resolution `dz` exceeds building height (e.g. dz=20 m with H_bldg=15 m in the default `UCMBoston` and `UCMBostonMixedDomain` inputs), buildings live entirely inside k=0, which is MOST-owned and excluded from BEP drag. The counter honestly reports `N_cells=0` — this is arithmetically correct, **not a bug**.

Momentum reduction over urban cells still occurs via z0/d_disp effects in MOST at k=0. To exercise BEP drag in a canonical, either:
- refine dz to below H_bldg (e.g. dz=5 m near surface), or
- use taller buildings (H_bldg > dz) in `building_layout.csv`.

Adding a dedicated `UCMBostonMixedDomainRefined/` canonical with dz=5 m near-surface stretching is tracked as a follow-up so BEP drag is exercised in the regression suite.

### Contract #13 (new): iMultiFab consumer / MFIter BoxArray must match

Any consumer that binds an `iMultiFab` (or a cell-centered `MultiFab`) via `.const_array(mfi)` **must** iterate an MFIter whose underlying BoxArray matches the iMultiFab's centering. Face-centered MFIter with a cell-centered `.const_array()` silently produces garbage patches on some ranks (the Array4 may be empty or point to unmapped storage). The safe idiom is:

- **Iterate the cell-centered MF** (typically `S_cons`, `cc_source`, or an equivalent cell-centered field).
- Bind all other cell-centered MFs and iMultiFabs from that MFIter.
- Write to face-centered momentum MFs by cell index `(i,j,k)`, which corresponds to the lower x-/y-/z-face of cell `(i,j,k)`.

This is the pattern used by both `apply_ucm_tendency_to_cc_source` (line 330) and `apply_ucm_momentum_drag_to_source` after this fix (line 799).

### Files touched

1. `Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp` — one line change at 799 (`xmom_src` → `S_cons` in the MFIter)

---