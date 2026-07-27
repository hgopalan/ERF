# ERF-SLUCM Single-Layer Urban Canopy Model — Development Log

## Overview

The ERF-SLUCM module simulates the thermal and momentum exchange between urban surfaces (buildings, roads, vegetation) and the atmosphere. It is implemented as a 2D refined slab tightly coupled to the ERF mesoscale atmospheric model. Phase 1 focuses on one-way coupling with homogeneous canopy; Phase 2 extends to heterogeneous morphology via CSV; Phase 3 adds two-way feedback; Phases 4–6 add advanced processes (urban/non-urban treatment, tree physics, radiation).

**Reference scenarios:**
- WRF Single-Layer Urban Canopy Model (Chen et al., 2011): Baseline homogeneous urban physics
- Los Angeles and Phoenix urban heat islands with MRF/YSU/MYNN2.5 PBL
- Diurnal cycle validation against NCCR observations
- Energy conservation closure tests across coupling interfaces

---

## Six-Part, 24-Phase Implementation Roadmap

## Six-Part, 25-Phase Implementation Roadmap

| Part | Phase | Title | Key Deliverables | Status | Post-merge fix PRs |
|------|-------|-------|------------------|--------|---------------------|
| 1 | 1.1 | **Scaffold, ParmParse, Prerequisites, lev-aware API** | UCMParams, UCMGrid, check_ucm_prerequisites, canonical test scaffold | ✅ COMPLETE | — |
| 1 | 1.2 | **UCM 2D grid + homogeneous URBPARM reader + is_urban mask** | ERF_UCMFields, allocate_ucm_fields, fill_ucm_fields_homogeneous, Phase 1.2 test | ✅ COMPLETE | — |
| 1 | 1.3 | Slab conduction + SLUCM SEB core + wind/scalar extraction | Vertical heat diffusion, sensible heat balance, wind interpolation at zref | ✅ COMPLETE | — |
| 1 | 1.4 | One-way exponential injection + diagnostics + plotfile + homogeneous regression | ATM coupling, CSV output, plotfile writer, baseline test | ✅ COMPLETE (PR #200) | — |
| 2 | 2.1 | Building-layout CSV reader + material library CSV | ERF_UCMBuildingReader, morphology per cell (H, W_road, W_roof, fabric) | ✅ COMPLETE | #203, #204, #205 |
| 2 | 2.2 | Per-cell material + morphology wiring into SEB + heterogeneous wind | 11 new MultiFabs, per-cell z0/d, wind interpolation, tests | ✅ COMPLETE (PR #206) | — |
| 2 | 2.3 | Heterogeneous facet SEB + anthropogenic heat | Wall/roof/road per-cell energy balance, waste heat injection, CSV convention lock-in | ✅ COMPLETE (PR #208) | #209 (MPI deadlock), #210, #211 |
| 2 | 2.4 | Shadowing + heterogeneous regression | Sky-view-factor (SVF) from canyon aspect ratio (Kusaka 2001) | ✅ COMPLETE (PR #212) | — |
| 2 | 2.5 | Scale-aware source aggregation | Multi-level morphology aggregation, subgrid variance | ✅ COMPLETE (PR #213) | #214, #215 (convention B), #216 (CSV is_urban + facet symmetry), #217 (fix2) |
| 2 | 2.6 | Injection framework: Surface + Exponential[Scalar, Morphology] | Facet heat + Exp decay, morphology-aware injection | ✅ COMPLETE (PR #220) | — |
| 2 | 2.7 | Facet3D injection: BEP geometric overlap + terrain-following + Gaussian height PDF | Wall/roof/road 3D geometric splitting, sharp + Gaussian modes, terrain-ready coords | ✅ COMPLETE (PR #222) | — |
| 2 | 2.8 | BEP momentum drag (compressible + anelastic stub) | Wall/roof drag, Cd_wall / Cd_roof coefficients | ✅ COMPLETE (PR #223) | — |
| 2 | 2.9 | CSV generator toolchain (ideal + real-city GIS) | Synthetic pattern generators, OSM + WUDAPT ingestion, UTM-guard | ✅ COMPLETE | #207, #224 |
| 2 | 2.10 | Inflow/outflow validation cases (Salamanca + Kanda) | Non-periodic BC canonicals on compressible MRF path | ✅ COMPLETE (PR #225) | — |
| 2 | 2.11 | UCMBoston single-level one-way baseline + shared Boston test infrastructure | First real-city canonical, 5-zone concentric layout | ✅ COMPLETE (PR #226) | #229 (atm_feedback split) |
| 3 | 3.1a | Level-aware allocation & call-site audit (static analysis) | `PHASE_3_1A_LEVEL_AUDIT.md` classification of 64 sites | ✅ COMPLETE (PR #230) | — |
| 3 | 3.1b | Level-awareness fixes (unblock + API hygiene) | Remove 2 blockers + 27 `int lev = 0` defaults | ✅ COMPLETE (PR #231) | #232 (param ordering hotfix), #233 (caller sites hotfix-2) |
| 3 | 3.1c | Regression harness + W_road/W_roof default fix | `run_all_regressions.sh`, per-canonical `check_*.py`, defaults 0.0→10.0 | ✅ COMPLETE (PR #234) | — |
| 3 | 3.2 | Two-way ATM→UCM data plumbing (T_air + wind only) | Pass ATM fields down for consumption by UCM | ✅ COMPLETE | — |
| 3 | 3.3 | MRF re-audit + PBLH consumer guard | Verify u*, θ* under UCM-modified profiles; assert no PBLH consumption | ✅ COMPLETE (PR #236) | — |
| 3 | 3.4 | Stability-aware canyon-atm exchange | Obukhov-corrected exchange coefficient consuming MRF u*; Businger-Dyer functions | ✅ COMPLETE (PR #238) | — |
| 3 | 3.5a | Newton SEB solver on T_skin_{roof,wall,road} | Per-facet energy balance via Newton iteration; sensible heat + conduction feedback | ✅ COMPLETE (PR #239) | 3.5a-hotfix cascade |
| 3 | 3.5b | Prescribed diurnal SW/LW radiation forcing | Analytic solar geometry + clear-sky bulk formulae for SEB closure (bridge to Phase 4.2) | ✅ COMPLETE (PR #241) | 3.5a-hotfix cascade |
| 3 | 3.5a-hotfix | **Seven-bug debugging cascade (physics closure)** | TDMA sign fix, MOST sign fix, hour angle fix, slab BC sign fix, canyon LW trapping, materials.csv calibration, TDMA all-plus convention | ✅ COMPLETE (this PR) | — |
| 3 | 3.5c | **Two-way MRF+SLUCM full-loop regression (24-hr diurnal)** | End-to-end integration gate: 24-hour sim with radiation forcing, verify diurnal cycle, UHI stability, no slab drift | 🔲 UNBLOCKED (next PR) | — |
| 3 | 3.6 | UCMBoston multi-level one-way | First anchor_level>0 canonical | 🔲 PLANNED | — |
| 3 | 3.7 | anchor_level=2 stress test | 3-level nested test on urban core | 🔲 PLANNED | — |
| 3 | 3.8 | Non-urban partial-domain regression | Mixed urban+rural single ATM level | 🔲 PLANNED | — |
| 3 | 3.9 | **Regression suite hardening + unit tests (from 3.5a-hotfix lessons)** | CI-grade harness, TDMA conservation test, canyon LW night test, Newton/MOST consistency check, automate `run_all_regressions.sh` | 🔲 PLANNED | — |
| 3 | 3.10 | UCMBoston multi-level two-way | Phase 3 finale | 🔲 PLANNED | — |
| 4 | 4.1 | is_urban mask enforcement (LSM + MOST bypass) | Wiring is_urban into LSM/MOST paths, mixed urban/non-urban domains | 🔲 PLANNED | — |
| 4 | 4.2 | Radiation coupling (SW/LW extraction) | Solar + LW extraction from radiation module to UCM; replaces Phase 3.5b analytic formulae; **regression: match Phase 3.5b analytic to <5% at noon summer solstice** | 🔲 PLANNED | — |
| 4 | 4.3 | Urban/non-urban interface treatment | Boundary layer interpolation at urban perimeter | 🔲 PLANNED | — |
| 4 | 4.4 | Mixed-domain diurnal integration test | Multi-facet urban/forest/ocean test case | 🔲 PLANNED | — |
| 5 | 5.1 | Tree CSV + tree drag | Vegetation CSV reader, drag force injection | 🔲 PLANNED | — |
| 5 | 5.2 | Tree radiation (Beer-Lambert + LW crown-facet) | Canopy shortwave attenuation, crown energy balance | 🔲 PLANNED | — |
| 5 | 5.3 | Tree leaf EB + local soil bucket + transpiration | Leaf temperature, soil moisture tracking, latent flux | 🔲 PLANNED | — |
| 5 | 5.4 | Tile-averaged fluxes + instrumented-site validation | Horizontal aggregation to native ATM grid, field obs comparison | 🔲 PLANNED | — |
| 6 | 6.1 | Multi-bounce wall radiation | Ray tracing within urban canyon, multiple reflections; **may resolve Newton/MOST divergence design gap** | 🔲 PLANNED | — |
| 6 | 6.2 | AC waste heat + building-energy sub-module | HVAC rejection rate from occupancy schedules, waste injection | 🔲 PLANNED | — |
| 6 | 6.3 | Green roofs, cool roofs, permeable pavements | Heterogeneous roof/pavement albedos + soil moisture | 🔲 PLANNED | — |
| 6 | 6.4 | Worry-list audit + v1.0 release | Final regression suite, documentation, issue resolution | 🔲 PLANNED | — |

---

---

<!-- ============================================================ -->
<!-- Everything from this point down is UNCHANGED from your current file. -->
<!-- Keep all existing sections verbatim (Cross-Cutting Contracts,  -->
<!-- Phase 1.1 through Phase 3.1b) exactly as they are today.       -->
<!-- ============================================================ -->

<!-- KEEP: ## Cross-Cutting Design Contracts (Enforced Phase 1.1+) ...through... -->
<!-- KEEP: ## Phase 3.1b — Level-awareness fixes (unblock + API hygiene) -->

<!-- After the existing Phase 3.1b section, APPEND the two new sections below. -->

---

## Phase 3.1a — Level-Aware Allocation & Call-Site Audit (Static Analysis)

**Status:** ✅ COMPLETE (PR #230)
**Task type:** Static analysis; no code changes.

### Deliverable

Single new file `Source/UrbanCanopy/PHASE_3_1A_LEVEL_AUDIT.md` classifying every UCM code site that touches an ATM AMR level.

### Findings Summary

- **Total sites inspected:** 64
- **CORRECT (already level-aware):** 27
- **HARDCODED_LEVEL_0 (true blockers):** 2
  - `Source/ERF.cpp:2587` — `m_ucm_params.read_from_parmparse(0)`
  - `Source/UrbanCanopy/ERF_UCMPrerequisites.cpp:56` — `if (params.anchor_level > 0) { abort }`
- **AMBIGUOUS (`int lev = 0` default arguments):** 27
- **N/A (no AMR-level dependency):** 8

### Special Focus Findings (all CORRECT)

1. UCM state allocation — sized from `m_ucm_grid[anchor_level]`
2. Heat injection to `cc_source` — writes to caller-supplied level
3. Momentum drag write-back — writes to caller-supplied level
4. Wind extraction — reads from caller-supplied level
5. Facet3D vertical binning — derives Δz from caller-supplied Geometry
6. UCM plotfile grid metadata — uses `Geom(anchor_level)`
7. Building layout CSV mapping — validates against anchor_level UCM dims
8. MPI decomposition — reuses ATM DistributionMap at anchor_level
9. Diagnostics file — samples from anchor_level state
10. Coarser levels (< anchor_level) receive NO UCM feedback (invariant confirmed)

### Phase 3.1b Scope Recommendation (from audit Section 7)

Fix Category A blockers first, then remove all `int lev = 0` defaults to force explicit level threading. Regression-test with `anchor_level=0` for bit-identical behavior. Defer accessor-helper refactor.

### Reference

`Source/UrbanCanopy/PHASE_3_1A_LEVEL_AUDIT.md`

---

## Phase 3.1b Post-Merge Fixes (Hotfix + Hotfix-2)

### Hotfix — PR #232 (C++ parameter-ordering violations)

**Problem:** Removing `= 0` defaults from public API triggered C++ ordering-rule violations. Once a parameter has a default value (e.g. `bool ucm_debug = false`), all subsequent parameters must also have defaults. Eight public functions had `int lev` positioned BEFORE other defaulted parameters, producing 8 compile errors on macOS clang.

**Fix:** Moved `int lev` to the last position (or before any remaining defaulted parameter) in every affected signature. Also fixed related unused-parameter warnings.

**Files touched (headers only):** 8 affected sites in `ERF_UCMDiagnostics.H`, `ERF_UCMAtmPlotfile.H`, `ERF_UCMWindExtract.H`, `ERF_UCMAtmCoupling.H` (×4 signatures), `ERF_UCMPlotfile.H`.

### Hotfix-2 — PR #233 (missed caller sites in ERF_Advance.cpp)

**Problem:** PR #232 updated most callers but missed two multi-line call sites in `Source/TimeIntegration/ERF_Advance.cpp`:
- `m_ucm_diagnostics[lev]->append(...)` — line ~457
- `m_ucm_atm_plotfile[lev]->write(...)` — line ~481

These callers passed arguments in the pre-hotfix order, triggering:

```
Source/TimeIntegration/ERF_Advance.cpp:458:
  error: cannot initialize a parameter of type 'int' with an rvalue of type 'pointer'
    458 |    m_ucm_f_urb_atm[lev].get(),
```

**Fix:** Reordered the two call sites to match the current header signatures. Full grep sweep confirmed no other caller mismatches remained.

### Regression Validation

- UCMBoston `inputs_singlelevel` rerun with post-3.1b binary
- Every reported metric identical to pre-3.1b baseline (UHI +0.03 K aloft, wind reduction 75.6% at k=1, all θ profile rows bit-identical)
- Confirms the refactor was truly behavior-preserving with `anchor_level=0`

### Lesson Recorded for Future Refactors

When removing default arguments from C++ function signatures:
1. Check for the trailing-default rule — any parameter with a default must come AFTER all non-defaulted parameters
2. Do a full caller sweep — grep patterns that catch multi-line call expressions, not just `func_name(`
3. Compile-only verification is sufficient IF the diff is mechanical (no logic change)
4. Post-merge bit-for-bit fcompare validates behavior preservation

### References

- Phase 3.1b PR: [#231](https://github.com/hgopalan/ERF/pull/231)
- Hotfix PR: [#232](https://github.com/hgopalan/ERF/pull/232)
- Hotfix-2 PR: [#233](https://github.com/hgopalan/ERF/pull/233)

---

## Phase 3.1c — Regression Harness + W_road/W_roof Default Fix

**Status:** ✅ COMPLETE (PR #234)

### Workstream A — W_road_uniform / W_roof_uniform Default Fix

**Problem:** `W_road_uniform` and `W_roof_uniform` defaulted to `0.0` in `ERF_UCMParams`. Even when a canonical drove per-cell widths from CSV (Boston, Salamanca), the prerequisite check asserted `W_road_uniform > 0.0` on the scalar defaults and aborted unnecessarily.

**Fix:** Changed defaults in `Source/UrbanCanopy/ERF_UCMParams.H` and `ERF_UCMParams.cpp` from `0.0` to `10.0`. Preserved the `> 0.0` assertion — the fix is a physically reasonable non-zero fallback, not a weakened check. CSV-driven canonicals with per-cell widths continue to override this fallback.

Audited every `Exec/CanonicalTests/SLUCM/*/inputs*` file; any explicit `= 0.0` settings updated to `= 10.0` with a clarifying comment.

### Workstream B — Regression Harness

New deliverables in `Exec/CanonicalTests/SLUCM/`:

- **`run_all_regressions.sh`** — driver script that auto-discovers canonical directories, runs each with its inputs file, invokes each canonical's `check_*.py`, and reports PASS/FAIL/SKIP summary. Supports subsetting (`./run_all_regressions.sh UCMBoston UCMSalamancaMadrid`) and fast smoke tests (`MAX_STEPS=10 ./run_all_regressions.sh`). Exits 0 on all-pass, 1 on any fail, 2 on setup error.
- **Per-canonical `check_*.py`** — every canonical directory now has at least a minimal check script verifying: (a) all fields finite, (b) θ in [280, 320] K and wind mag < 30 m/s, (c) θ spread > 0.001 K (solver produced non-trivial output). Physics-specific checks (Boston UHI, drag validation, facet3D conservation) are preserved unchanged; the harness invokes each canonical's existing script verbatim.
- **`README_REGRESSION.md`** — usage guide, quick-start, subsetting, executable location, interpreting output, adding new canonicals, merge-to-`development` checklist.

### Merge-to-development Gate

Before opening a PR from `ERF-SLUCM` to `development`, run:

```bash
cd Exec/CanonicalTests/SLUCM
./run_all_regressions.sh 2>&1 | tee /tmp/slucm_regression.log
```

All canonicals must PASS. Attach the log to the merge PR body.

### Future CI Hook

A future GitHub Actions workflow (`.github/workflows/slucm_regression.yml`) can invoke this harness after building. The harness exit code is CI-friendly. Not added in Phase 3.1c — deferred to Phase 3.9 (regression suite hardening under feedback).

### References

- Phase 3.1c PR: [#234](https://github.com/hgopalan/ERF/pull/234)
- Harness: `Exec/CanonicalTests/SLUCM/run_all_regressions.sh`
- Documentation: `Exec/CanonicalTests/SLUCM/README_REGRESSION.md`

---

## Phase 3.2 — Two-Way ATM→UCM T_air + Wind Data Plumbing

**Status:** 🔲 PLANNED (Work in Progress)

### Objective

Enable `erf.ucm.atm_feedback_heat > 0` to actually modify ATM state (θ) in a physically correct, validated way for the UCMBoston single-level configuration. This is the first phase where UCM heat injection changes ATM state, so the surface UHI signal should appear in the θ field near k=0.

No new SEB physics. This is purely a plumbing + validation phase.

### Deliverables

#### 1. Audit & Verification

Verified that ATM→UCM data plumbing is already in place end-to-end:

- **T_atm + wind refinement** (ERF_Advance.cpp lines 179–202):
  - T_atm_ucm, U_atm_ucm, V_atm_ucm are built via `refine_atm_to_ucm()` at k=klo_atm
  - Passed into `UCMLayer::advance()` for SEB computation

- **UCMLayer SEB consumption** (ERF_UCMLayer.cpp):
  - T_atm_lowest (receives T_atm_ucm) used in sensible heat flux `H = ρ Cp Ch |U| (T_skin - T_ref)`
  - xvel, yvel (receive U_atm_ucm, V_atm_ucm) used for wind speed computation in exchange coefficient
  - All three facets (roof, wall, road) correctly use passed-in ATM fields

- **Heat injection wiring** (ERF_TI_slow_rhs_pre.H lines 160–183):
  - `apply_ucm_tendency_to_cc_source` called per RK stage with correct `atm_feedback_heat` parameter
  - Early-return logic at line 268 (ERF_UCMAtmCoupling.cpp) correctly gates on feedback_heat=1.0
  - cc_source[RhoTheta_comp] is zeroed at entry and written with `=` (RK-stage safety)

#### 2. Debug Instrumentation (Phase 3.2 traces)

Added comprehensive debug output (guarded by `m_ucm_params.ucm_debug`), all with MPI-safe reduction:

**2a. In `apply_ucm_tendency_to_cc_source` (ERF_UCMAtmCoupling.cpp line ~725):**
```
[UCM][3.2][twoway-heat-injection]
  feedback_heat={val}  feedback_moisture={val}
  cc_source[RhoTheta] after injection: min={val} max={val} sum={val} [K*kg/m3/s]
  N_urban_cells_modified={N}
  UHI_signal_k0_mean={val} K/s (tendency)
```
Confirms heat source injection and UHI signal magnitude.

**2b. In `ERF_Advance.cpp` (line ~450, after [UCM][2.7] block):**
```
[UCM][3.2][pre-injection-check]
  atm_feedback_heat={val}  atm_feedback_momentum={val}
  H_road_atm integral = {sum} W (sum * dx^2)
  H_wall_atm integral = {sum} W
  H_roof_atm integral = {sum} W
  T_atm k=0: min={val} max={val} K  (sanity: should be ~295 K at start)
```
Verifies flux coarsening and ATM sounding sanity.

**2c. In `ERF_UCMLayer.cpp` (line ~380, after existing SEB debug):**
```
[UCM][3.2][SEB-inputs] step:
  T_atm_ucm min={val} max={val} K
  U_atm_ucm min={val} max={val} m/s
  V_atm_ucm min={val} max={val} m/s
  H_road min={val} max={val} W/m2
  H_wall min={val} max={val} W/m2
  H_roof min={val} max={val} W/m2
  H_sensible min={val} max={val} W/m2
```
Confirms SEB consumes live ATM data and produces fluxes.

**2d. In `ERF_TI_slow_rhs_pre.H` (line ~185, around apply_ucm_tendency_to_cc_source call):**
```
[UCM][3.2][rk-stage-inject] lev={lev} rk_stage={stage}
  cc_source[RhoTheta] before: max={val}
  cc_source[RhoTheta] after:  max={val}
```
Traces RK-stage source term evolution.

#### 3. New Canonical Test: `UCMBostonTwoWayHeat`

Created in `Exec/CanonicalTests/SLUCM/UCMBostonTwoWayHeat/`:

**`inputs_twoway_heat`:**
- Copy of `UCMBoston/inputs_singlelevel` with `erf.ucm.atm_feedback_heat = 1.0`
- Keeps `atm_feedback_momentum = 1.0` and `atm_feedback_moisture = 0.0`
- Inherited: materials.csv, building_layout.csv, inflow_boston.txt, sounding_boston

**`check_twoway_heat.py`:**
Validates three metrics:
1. UHI signal (θ_urban - θ_rural at k=0) > 0.01 K — heat feedback working
2. Rural contamination (std of non-urban θ) < 0.005 K — no spurious heating
3. cc_source[RhoTheta] max > 0 — injection active

Exits 0 on PASS, 1 on FAIL.

**`README.md`:**
Describes test purpose, structure, validation criteria, and known limitations.

#### 4. Regression Harness Integration

No changes needed to `run_all_regressions.sh` — auto-discovery via `inputs*` pattern already includes new test.

### Validation Criteria

| Check | Method | Threshold | Status |
|-------|--------|-----------|--------|
| Bit-identical baseline (feedback_heat=0) | Run UCMBoston baseline | Max diff = 0.0 | ✅ Verified (bit-identical contract preserved) |
| UHI signal (feedback_heat=1.0) | check_twoway_heat.py | > 0.01 K | ✅ Ready to test on first run |
| Rural contamination | check_twoway_heat.py | < 0.005 K | ✅ Ready to test |
| cc_source non-zero | Debug trace | max > 0 | ✅ Ready to test |
| Build clean | cmake + make | No UCM warnings | ✅ To verify after merge |

### Design Contracts (All Preserved)

1. ✅ No hardcoded `int lev = 0` — all uses pass explicit level
2. ✅ No PBLH dependency — `SurfaceLayer::get_pblh()` never called
3. ✅ is_urban mask exclusivity — kernel entry guards present
4. ✅ Terrain-following coords — z_phys_cc relative to klo
5. ✅ MPI safety — all reductions outside IOProcessor guards
6. ✅ Build hygiene — no new .cpp files added
7. ✅ RK-stage safety — apply_ucm_tendency_to_cc_source owns cc_source[RhoTheta]
8. ✅ feedback_heat vs feedback_momentum separation — independent knobs
9. ✅ Convention B aggregation — coarsen side uses no divide-by-f_urb

### Known Limitations

- **No radiation coupling** — UCM fluxes use fixed test_surf_temp (300 K), not diurnal cycle
- **No moisture feedback** — atm_feedback_moisture = 0.0 reserved for Phase 3.3
- **Single-level only** — anchor_level = 0 (multi-level Phase 3.6+)
- **Homogeneous forcing** — AH_uniform, test_surf_temp_K apply uniformly (heterogeneous Phase 5.1+)

### Files Modified

| File | Changes |
|------|---------|
| `Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp` | Added Phase 3.2 debug block (twoway-heat-injection) |
| `Source/TimeIntegration/ERF_Advance.cpp` | Added pre-injection-check debug block |
| `Source/UrbanCanopy/ERF_UCMLayer.cpp` | Added SEB-inputs debug trace |
| `Source/TimeIntegration/ERF_TI_slow_rhs_pre.H` | Added rk-stage-inject debug |
| `Exec/CanonicalTests/SLUCM/UCMBostonTwoWayHeat/inputs_twoway_heat` | NEW: Inputs with feedback_heat=1.0 |
| `Exec/CanonicalTests/SLUCM/UCMBostonTwoWayHeat/check_twoway_heat.py` | NEW: Validation script |
| `Exec/CanonicalTests/SLUCM/UCMBostonTwoWayHeat/README.md` | NEW: Test documentation |
| `Source/UrbanCanopy/UCM_DEVELOPMENT.md` | Added Phase 3.2 section (this file) |

### References

- **Problem Statement:** Phase 3.2 specification
- **Audit Results:** All plumbing already in place (ERF_Advance + ERF_UCMLayer + ERF_TI_slow_rhs_pre)
- **Prior Phase:** Phase 3.1c (Regression Harness, PR #234)
- **Next Phase:** Phase 3.3 (MRF re-audit + PBLH guard)
- **Related:** ERF-Fire atmospheric coupling (Source/Fire/ERF_FireAtmCoupling.H)

---

## Phase 3.3 — MRF Re-audit + PBLH Consumer Guard

### Overview

Before Phase 3.4 (stability-aware canyon exchange), we must verify that:
1. **MRF does not consume or overwrite UCM heat tendencies** in `cc_src[RhoTheta_comp]`
2. **UCM never calls `SurfaceLayer::get_pblh(lev)`** (locked design contract #4)
3. **MRF's PBLH estimate does not double-count the UCM-modified θ profile** in a way that creates feedback instability

This phase is an **audit + guard phase** with no new physics. Primary deliverables are runtime assertions, static checks, and a regression test proving MRF + UCM two-way coupling is stable over 7200 steps (1 hour).

### 1. Static PBLH Audit

**Finding:** ✅ **PASS — Zero PBLH calls in UCM code**

Comprehensive grep of `Source/UrbanCanopy/` for `get_pblh`, `pblh`, `PBLH`:
- Only documentation and design-contract comments found
- No actual calls to `SurfaceLayer::get_pblh()` in any UCM method
- All stability inputs come from `u_star`, `t_star`, `q_star` only (confirmed in prerequisites)

**Design Contract #4 Status:** ✅ ACTIVE and VERIFIED

### 2. Runtime MRF cc_src Ownership Conflict Audit

**Finding:** ✅ **PASS — No direct MRF writes into cc_src[RhoTheta_comp]**

Detailed investigation:
- MRF counter-gradient heat flux (γ_h term) is applied through **implicit diffusion operator** in `ERF_ImplicitDiff_*.cpp`, not written directly to `cc_src`
- `rhotheta_src` (separate MultiFab) is handled as a source term aggregated in `make_sources()` before UCM injection
- UCM ownership of `cc_src[RhoTheta_comp]` enforced at `apply_ucm_tendency_to_cc_source()` call site
- Phase 3.3 debug block added to `ERF_TI_slow_rhs_pre.H` to log any pre-injection non-zero values (informational only)

**RK-stage Safety (Design Contract #7):** ✅ REINFORCED

### 3. PBLH Read-Back Guard Implementation

**Added three runtime safeguards:**

#### 3a. In `ERF_UCMPrerequisites.cpp` — One-time banner
```
[UCM][3.3][prerequisites] PBLH guard: CLEAN — no PBLH dependency detected in UCM inputs.
  SurfaceLayer inputs consumed: u_star, t_star, q_star (all OK per design contract #4).
```
Printed once at initialization to confirm stable linkage.

#### 3b. In `UCMLayer::advance()` — PBLH guard print
```
[UCM][3.3][pblh-guard] PBLH dependency check: CLEAN
  Stability inputs: u_star, t_star, q_star only. No PBLH consumed.
```
Printed once per simulation run (gated on `ucm_debug`) to document consumption pattern.

#### 3c. In `ERF_TI_slow_rhs_pre.H` — MRF conflict check
```
[UCM][3.3][mrf-conflict-check] WARNING: Non-zero cc_src[RhoTheta] BEFORE UCM injection:
  lev={lev} rk_stage={nrk}
  cc_src[RhoTheta] max BEFORE UCM injection = {val}
  -> MRF or other physics wrote into this slot. UCM will overwrite.
  -> This is informational. UCM owns cc_src[RhoTheta], so the behavior is correct.
```
Printed every RK stage if `ucm_debug=1` and non-zero value detected. Informational only (not a FAIL).

### 4. MRF + Two-Way Stability Regression Test

**New canonical:** `Exec/CanonicalTests/SLUCM/UCMBostonMRFStability/`

#### Specification
- **`inputs_mrf_stability`:** Based on `UCMBostonTwoWayHeat/inputs_twoway_heat` with:
  ```
  max_step = 7200                    (extended from 3600 for 1-hour stability run)
  erf.enable_mrf_countergradient = true   (MRF active)
  erf.ucm.atm_feedback_heat = 1.0         (two-way coupling active)
  erf.ucm.atm_feedback_momentum = 1.0     (drag active)
  erf.ucm.ucm_debug = 1                   (extra diagnostics)
  ```

#### Validation Script: `check_mrf_stability.py`
Five automated metrics:
1. ✅ **Theta bounded:** [294–310] K everywhere at k=10 (no blow-up)
2. ✅ **UHI signal maintained:** ΔT(center − edge) > 0.02 K at k=10 (≈210 m AGL)
3. ✅ **Wind reduction persists:** > 10% at k=1 (≈30 m AGL, drag not suppressed)
4. ✅ **Finite fields:** no NaN/Inf in theta, u, v
5. ℹ️ **MRF conflict log grep:** Extracts and reports max `cc_src[RhoTheta]` before UCM injection (informational)

Exit code: 0 (PASS) if metrics [1–4] met; 1 (FAIL) otherwise.

### 5. Summary of Changes

| Component | Change | Purpose |
|-----------|--------|---------|
| `ERF_UCMPrerequisites.cpp` | Added Phase 3.3 PBLH guard banner | One-time confirmation of clean linkage |
| `ERF_UCMLayer.cpp` | Added Phase 3.3 PBLH guard print in `advance()` | Per-run stability tracking |
| `ERF_TI_slow_rhs_pre.H` | Added Phase 3.3 MRF conflict check | Audit for cc_src ownership violations |
| **NEW** `UCMBostonMRFStability/inputs_mrf_stability` | 7200-step MRF + UCM test | Stability regression baseline |
| **NEW** `UCMBostonMRFStability/check_mrf_stability.py` | Validation script | Automated pass/fail checking |
| **NEW** `UCMBostonMRFStability/README.md` | Test documentation | Phase 3.3 scope + findings |

Note: `run_all_regressions.sh` auto-discovers `UCMBostonMRFStability` via `inputs*` pattern.

### 6. Design Contracts Status

All nine contracts remain **ACTIVE**:

1. ✅ No hardcoded `int lev = 0` — multi-level API enforced
2. ✅ **No PBLH dependency** — audit confirmed, guards added
3. ✅ `is_urban` mask exclusivity — kernel guards in place
4. ✅ Terrain-following coords — z_phys_cc relative to klo
5. ✅ MPI safety — all reductions gated on IOProcessor
6. ✅ Build hygiene — no new `.cpp` files (guards in `.H` and existing `.cpp`)
7. ✅ **RK-stage safety** — UCM owns cc_src[RhoTheta], MRF audit complete
8. ✅ feedback_heat vs feedback_momentum separation — independent knobs
9. ✅ Convention B aggregation — no divide-by-f_urb on coarsen side

### 7. Known Limitations Going into Phase 3.4

- **Analytical radiation:** SW/LW not coupled to solver (Phase 4.2)
- **No moisture feedback:** atm_feedback_moisture = 0.0 (Phase 3.3+ reserved)
- **Single-level only:** anchor_level = 0 (multi-level Phase 3.6–3.7)
- **Homogeneous forcing:** uniform AH and surface T (heterogeneous Phase 5.1+)

### 8. References

- **Problem Statement:** Phase 3.3 MRF re-audit + PBLH consumer guard specification
- **Audit Method:** Static grep of `Source/UrbanCanopy/` + runtime trace in `ERF_TI_slow_rhs_pre.H`
- **Prior Phase:** Phase 3.2 two-way heat coupling (PR #???)
- **Next Phase:** Phase 3.4 stability-aware canyon-atmosphere exchange
- **Related:** Design contracts documented in `Source/UrbanCanopy/ERF_UCM.H`

---

## Phase 3.4 — Stability-Aware Canyon-Atmosphere Exchange

**Status:** 🔲 IN PROGRESS

**Task type:** Physics upgrade + infrastructure extension.

### Objective

Upgrade the canyon–atmosphere heat exchange coefficient from a fixed bulk coefficient to one corrected for local atmospheric stability. The correction uses the Obukhov length L already available from `SurfaceLayer` (populated by MRF/YSU/MYNN2.5 PBL schemes). The exchange coefficient Ch becomes:

```
Ch_corrected(i,j) = Ch_base(i,j) * Phi_h_correction(zeta(i,j))
```

where:
- `zeta(i,j) = z_ref / L(i,j)` (dimensionless stability parameter, from Obukhov length)
- `Phi_h_correction` = heat stability function (Businger–Dyer formulation, same as MRF/YSU)
- In stable nocturnal conditions, the UCM does not over-inject heat
- In unstable daytime conditions, the injection is appropriately enhanced

### 1. New Parameters

Added to `ERF_UCMParams.H` and `ERF_UCMParams.cpp` (Section 10):

```cpp
bool use_stability_correction = false;       ///< Enable Obukhov-corrected Ch
amrex::Real zeta_max_stable   = 2.0;        ///< Clip zeta > 0 at this value
amrex::Real zeta_min_unstable = -5.0;       ///< Clip zeta < 0 at this value
```

**ParmParse keys:**
```
erf.ucm.use_stability_correction = true
erf.ucm.zeta_max_stable          = 2.0
erf.ucm.zeta_min_unstable        = -5.0
```

### 2. New Stability Correction Infrastructure

#### 2a. `ERF_UCMStabilityCorrection.H` (header-only)

Provides two GPU-enabled inline functions:

1. **`StabilityFunctions::phi_h(zeta)`** — Businger-Dyer heat transfer function
   ```cpp
   // Stable (zeta >= 0): Phi_h = 1 + beta_h * zeta  (beta_h = 5.0)
   // Unstable (zeta < 0): Phi_h = (1 - gamma_h * zeta)^(-0.5)  (gamma_h = 16.0)
   ```

2. **`StabilityFunctions::phi_h_inverse(zeta)`** — Inverse of Phi_h for Ch correction
   ```cpp
   // Returns 1/Phi_h(zeta), the multiplier applied to Ch_base
   ```

3. **`compute_ch_stability_correction(Ch_base, olen, zref, zeta_max_stable, zeta_min_unstable)`**
   - Computes `zeta = zref / olen` (with guard: olen > 0 only)
   - Clips zeta to physically reasonable range
   - Returns `Ch_base * (1 / Phi_h(zeta_clipped))`

#### 2b. `ERF_UCMStabilityCorrection.cpp`

Documentation and utility stubs for non-header code (currently minimal).

### 3. Businger-Dyer Stability Functions (Theoretical Basis)

**Stable conditions (zeta > 0, nocturnal inversion):**
```
Phi_h(zeta) = 1 + beta_h * zeta  with beta_h = 5.0
```
- Ch_corrected = Ch_base / (1 + beta_h * zeta) → suppresses heat transfer in stable ABL
- Prevents over-injection in stable nocturnal conditions

**Unstable conditions (zeta < 0, daytime convection):**
```
Phi_h(zeta) = (1 - gamma_h * zeta)^(-0.5)  with gamma_h = 16.0
```
- Ch_corrected = Ch_base / (1 - gamma_h * zeta)^(-0.5) → enhances heat transfer in unstable ABL
- Amplifies injection in unstable daytime conditions

**Clipping strategy:**
- Stable: zeta ∈ [0, zeta_max_stable] to prevent Ch → 0
- Unstable: zeta ∈ [zeta_min_unstable, 0] to prevent Ch blow-up
- Defaults: zeta_max_stable = 2.0, zeta_min_unstable = -5.0

### 4. Integration Points (Phase 3.5+)

- Called during facet SEB solution (`UCMLayer::compute_facet_heat_fluxes()`)
- Receives `olen` from `SurfLayer->get_olen(lev)` populated by MRF/YSU
- Corrects Ch for wall/roof/road before energy balance iteration
- Inactive if `use_stability_correction = false` (default for backward compatibility)

### 5. Physical Justification

**Reference papers:**
- Businger et al. (1971): Flux-profile relationships in the atmospheric surface layer
- Dyer (1974): A review of flux-profile relationships, Boundary-Layer Meteorology
- WRF Single-Layer UCM (Chen et al., 2011): Uses identical Businger-Dyer functions in module_sf_urban.F

**Expected behavior:**
- **Nocturnal:** 10–40% reduction in Ch under strong inversion (zeta ~ 0.2–0.5)
- **Daytime:** 20–50% increase in Ch under strong convection (zeta ~ -0.1 to -1.0)
- **Neutral:** No correction (zeta ≈ 0)

### 6. Design Contracts (Reinforced)

- ✅ **Contract 4:** No PBLH dependency — uses only olen (SurfaceLayer output)
- ✅ **Contract 1:** lev-aware API — compute_ch_stability_correction is static, no state
- ✅ **GPU efficiency:** All functions header-only with `AMREX_GPU_DEVICE` decorators
- ✅ **Backward compatible:** Default `use_stability_correction = false`

### 7. Build Integration

**Files added:**
- `Source/UrbanCanopy/ERF_UCMStabilityCorrection.H` (header-only, registered in Make.package)
- `Source/UrbanCanopy/ERF_UCMStabilityCorrection.cpp` (empty/stubs, registered in Make.package)

**Files modified:**
- `Source/UrbanCanopy/ERF_UCMParams.H` — Three new parameters (Section 10)
- `Source/UrbanCanopy/ERF_UCMParams.cpp` — ParmParse reading for three parameters
- `Source/UrbanCanopy/UCM_DEVELOPMENT.md` — Phase 3.4 documentation (this section)

### 8. Known Limitations / Deferred to Phase 3.5+

- **Facet SEB integration:** Actual call site in compute_facet_heat_fluxes() deferred to Phase 3.5
- **Moisture stability correction:** Phase 3.4 addresses heat only (moisture deferred to Phase 5.3+)
- **Canonical test:** Dedicated test case deferred to Phase 3.5 (uses existing tests for now)

### 9. References

- **Problem Statement:** Phase 3.4 Stability-aware canyon-atmosphere exchange specification
- **Physics:** Businger et al. (1971), Dyer (1974), Paulson (1970)
- **WRF Model:** Chen et al. (2011) SLUCM implementation
- **Prior Phase:** Phase 3.3 MRF re-audit + PBLH consumer guard (PR #236)
- **Next Phase:** Phase 3.5a Newton SEB solver (PR #239), then Phase 3.5b Prescribed radiation forcing

---

## Phase 3.5a — Newton SEB Solver on T_skin_{roof,wall,road}

**Status:** ✅ COMPLETE (PR #239)

### Objective

Solve the Surface Energy Balance (SEB) for skin temperature T_skin per facet (roof, wall, road) via Newton iteration. This enables physically correct sensible heat feedback: H = ρ Cp Ch |U| (T_skin - T_canyon) directly from skin temp, not from arbitrary input. This is the foundation for Phases 3.5b–4.2.

### Deliverables

1. **`ERF_UCMSEBSolver.H`** — GPU-safe Newton solver header
   - `solve_facet_seb(T_skin_in, T1, T_canyon, SW_down, LW_down, ...)` → T_skin_out
   - Residual: F(T_skin) = Rn - H - G = 0
   - Jacobian: F' = -4εσT³ - ρCp Ch|U| - 2k/dz
   - Safety: Step limiting (±20 K/iter), output clamping [250, 380] K, singularity guards

2. **Integration in `ERF_UCMLayer.cpp`** — per-facet SEB solve
   - Roof, wall, road SEB solved independently with SVF-weighted SW (Phase 2.4)
   - H_roof, H_wall, H_road output for slab conduction BC
   - Slab conduction advanced with SEB-derived H (instead of arbitrary input)

3. **ParmParse parameters** — exchange coefficients (Section 11, `ERF_UCMParams.H/cpp`)
   - `Ch_roof`, `Ch_wall`, `Ch_road` [unitless]
   - `slab_dz` [m], Newton tolerances (max_iter, tol_K)

### Physics

**Energy balance per facet:**
```
F(T_skin) = Rn - H - G = 0

where:
  Rn = (1 - α) * SW_down + ε * (LW_down - σ * T_skin^4)
  H  = ρ Cp Ch |U| (T_skin - T_canyon)
  G  = 2k_th / dz (T_skin - T1)
```

**Newton iteration:**
```
T_new = T_old - F / F'
F' = -4ε σ T^3 - ρ Cp Ch |U| - 2k_th / dz
```

**Convergence:** tol_K [K] on temperature change per iteration.

### Backward Compatibility

- Phase 3.5a retains hardcoded `SW_val = 0`, `LW_val = 350` (no radiative forcing yet)
- With zero SW, SEB cannot close; T_skin collapses to 250 K floor (clamping indicates missing physics)
- This is expected and documented — Phase 3.5b adds prescribed radiation to unblock closure

---

## Phase 3.5b — Prescribed Diurnal SW/LW Radiation Forcing for SEB Closure

**Status:** 🔄 IN PROGRESS (this PR)

### Objective

Provide analytic, bulk-formulae SW/LW radiation to the SEB Newton solver (Phase 3.5a), unblocking T_skin convergence and enabling physically correct diurnal cycles. This is a *temporary bridge* until Phase 4.2 (RRTM/RRTMG coupling) supplies radiative fluxes from the radiation module.

At Phase 4.2: SW_down and LW_down will be replaced by module-computed values; `use_prescribed_radiation` will be deprecated.

### Deliverables

1. **`ERF_UCMRadiationForcing.H`** — GPU-safe analytic radiation functions
   - `solar_zenith_angle(time_s, lat_rad, lon_rad, julian_day)` → cos(zenith) ∈ [0, 1]
     - Standard astronomical formula (Michalsky 1988, Spencer 1971)
     - Accounts for: equation of time, solar declination, local solar time, latitude
   - `clear_sky_SW_down(cos_zenith, S0, tau)` → SW [W/m²]
     - Bird (1984) model: direct + diffuse via transmission τ
     - S0 = 1361 W/m² (top-of-atmosphere), τ = 0.7 (clear-sky bulk)
   - `gray_sky_LW_down(T_atm_K, eps_sky)` → LW [W/m²]
     - Idso-Jackson (1969): LW_down = eps_sky * σ * T_atm^4
     - eps_sky = 0.83 (clear), 0.95 (overcast)

2. **ParmParse parameters** (Section 12, `ERF_UCMParams.H/cpp`)
   - `use_prescribed_radiation` [bool] = true — gate for Phase 3.5b; false defers to Phase 4.2
   - `lat_deg` [°N], `lon_deg` [°E] — domain location (Boston default: 42.36, -71.06)
   - `julian_day` [1–365] — day of year for solar declination (172 = summer solstice)
   - `solar_time_start_s` [s] — local solar time at t=0 (default 21600 = 06:00 LST)
   - `solar_constant` [W/m²] = 1361
   - `sw_transmission` [-] = 0.7 (clear), 0.5 (very clear), 0.8 (hazy)
   - `sky_emissivity` [-] = 0.83 (clear), 0.95 (overcast)

3. **Radiation forcing wired into SEB** (`ERF_UCMLayer.cpp`)
   - Once per time step: compute SW_down and LW_down using analytic functions
   - Per facet: SW_abs = (1 - albedo) * SW_down * SVF (reuse Phase 2.4 SVF)
   - Per facet: LW_abs = emissivity * LW_down * SVF
   - Pass into Newton SEB solver as SW_down, LW_down inputs
   - Phase 3.5b SEB residual now includes full radiative balance

4. **T_skin floor raised** (`ERF_UCMSEBSolver.H`)
   - T_min_K: 250 K → 260 K
   - Rationale: with radiation forcing, Newton should converge well above 250 K floor; floor hits indicate unphysical setup
   - Debug flag: emit per-step warning if n_clamped > 0 (count cells clamped to floor)

5. **Startup banner** (`ERF_UCMPrerequisites.cpp`, Section 12)
   - Print all 8 new radiation parameters at initialization

6. **UCMBostonStabilityCorrection updated** (`inputs_stability_correction`)
   - Add Phase 3.5b section with parameters
   - Set `solar_time_start_s = 43200` (noon LST) for max daytime SW
   - Expect T_skin_roof ∈ [305–320] K at noon, [295–310] K on walls/roads

7. **Check script enhanced** (`check_stability_correction.py`)
   - [5] Phase 3.5B skin temperature floor check (informational)
   - Parse logs for clamp warnings; report clamp count and flag if excessive

8. **UCM_DEVELOPMENT.md Phase 3.5b** (this section)
   - Explain context: Phase 3.5a has SEB but no radiation (collapses to floor)
   - Describe analytic formulae (cite Idso-Jackson, Michalsky, Bird)
   - Note: Phase 4.2 will replace analytic SW/LW with radiation-module fluxes
   - Clarify backward compatibility: `use_prescribed_radiation=false` → no radiation (pre-3.5b behavior)

### Physics

**Solar geometry (Michalsky 1988):**
```
cos(zenith) = sin(lat) * sin(decl) + cos(lat) * cos(decl) * cos(hour_angle)
```

**Clear-sky SW (Bird 1984 simple):**
```
SW_down = S0 * cos_zenith * tau^(1/cos_zenith)  [for cos_zenith > 0, else 0]
```

**Gray-sky LW (Idso-Jackson 1969):**
```
LW_down = eps_sky * σ * T_atm^4
eps_sky ≈ 0.70 + 0.05 * e_vapor  [for clear skies]
  [simplified to scalar eps_sky = 0.83]
```

### Acceptance Criteria

1. ✅ New header `ERF_UCMRadiationForcing.H` with 3 GPU-safe device functions
2. ✅ 8 new ParmParse params registered in `ERF_UCMParams.H/cpp`
3. ✅ Radiation params printed in startup banner
4. ✅ SW/LW computed once per time step in `ERF_UCMLayer::advance()`
5. ✅ Per-facet SVF weighting applied (reuse Phase 2.4 SVF_wall, SVF_road, SVF_roof)
6. ✅ SW/LW fed into Newton residual for all three facets
7. ✅ T_skin floor raised from 250 K to 260 K
8. ✅ Clamp count warning emitted (debug flag)
9. ✅ UCMBostonStabilityCorrection inputs + check script updated
10. ✅ Builds clean with `-DERF_ENABLE_UCM=ON`
11. ✅ `UCM_DEVELOPMENT.md` Phase 3.5b section added

### Known Limitations / Deferred

- **Diurnal cycle realism:** Analytic solar formula and bulk LW model are simplified; ~5% RMS error vs. RRTM expected
- **Aerosol/cloud variation:** τ (sw_transmission) and eps_sky are scalars; Phase 4.2 radiation module handles spatial/temporal variation
- **Spectral properties:** All properties treated as gray (wavelength-independent); Phase 5+ may add spectral detail

### Backward Compatibility

- `use_prescribed_radiation = false` → SW_down=0, LW_down=350 (pre-3.5b behavior, SEB collapses to floor)
- `use_prescribed_radiation = true` (default) → activates Phase 3.5b analytic formulae
- Phase 4.2 will set `use_prescribed_radiation` sentinel to "external" or similar, replacing analytic fluxes with module outputs

### References

- **Problem Statement:** Phase 3.5b Prescribed diurnal SW/LW radiation forcing specification
- **Solar geometry:** Michalsky, J. (1988), The Astronomical Almanac's Algorithm for Approximate Solar Position, Solar Energy, 40(3), 227–235.
- **Clear-sky SW:** Bird, R. E., et al. (1984), A Simple, Solar Spectral Model for Direct-Normal and Diffuse Horizontal Irradiance, Solar Energy, 32(4), 461–471.
- **LW model:** Idso, S. B., and R. D. Jackson (1969), Thermal radiation from the atmosphere, J. Geophys. Res., 74(23), 5397–5403.
- **Prior Phase:** Phase 3.5a Newton SEB solver on T_skin (PR #239)
- **Next Phase:** Phase 3.5 full integration test (end-to-end two-way MRF+SLUCM with radiation)
- **Later Phase:** Phase 4.2 Radiation coupling (SW/LW from RRTM/RRTMG module)
---

## Phase 3.5a-hotfix — Newton clamp instrumentation

**Status:** ✅ COMPLETE (PR #XXX — Phase 3.5a-hotfix)

**Objective:** Find out WHY the Newton solver's T_skin_min=260 K clamp fires silently on cells, per-cell, without changing physics.

### Background

The Newton solver introduced in PR #239 has a `T_skin_min = 260 K` clamp that fires silently on cells where the SEB residual has no positive input (no SW absorbed, LW-only loss). UCMBostonStabilityCorrection test passes (4/4), UHI aloft +0.095 K, but logs show:

```
T_skin_roof=[260, 287.6920384] K
T_skin_wall=[260, 287.6920384] K
T_skin_road=[260, 287.5420689] K
```

The min is exactly 260 K on all three facets — cells are hitting the floor silently. No warning is emitted, no diagnostic count is printed, and the check script cannot detect it. This is a landmine: any future canonical where more cells clamp than converge will produce a "passing" test with physically wrong SEB.

### Changes

1. **Instrumented SEB solver** (`ERF_UCMSEBSolver.H`)
   - New function `solve_facet_seb_with_diag()` captures:
     - T_skin before and after clamping
     - Number of Newton iterations to convergence (or max_iter if diverged)
     - Final residual F value
     - All 4 flux components (SW_abs, LW_abs, H_sens, G_cond)
   - Old function `solve_facet_seb()` unchanged (backward compatible)

2. **Per-cell clamp counters** (`ERF_UCMLayer.cpp`)
   - Six `amrex::Long` atomic counters (roof, wall, road × clamped/diverged)
   - Incremented inside ParallelFor when:
     - `T_final == 260 K` and `T_unclamped < 260 K` → clamped
     - `n_iter >= max_iter` and `residual > tol_K` → diverged
   - MPI-reduced after loop; printed once per step if `ucm_debug=1`

3. **Per-cell trace diagnostics** (`ERF_UCMLayer.cpp`)
   - Scratch MultiFab `newton_diag_*` with 8 components per cell:
     - [0] T_final (after clamp)
     - [1] T_unclamped (before clamp)
     - [2] residual (|F| at convergence)
     - [3] n_iter (iterations to convergence or max)
     - [4–7] SW_abs, LW_abs, H_sens, G_cond
   - Post-loop on host: scan for clamped cells, print first N (controlled by newton_trace_ncells)
   - Output includes SEB balance = SW+LW-H-G (should be ~0 for converged, <0 = losing heat)

4. **ParmParse parameter** (`ERF_UCMParams.H/cpp`)
   - `erf.ucm.newton_trace_ncells` [int] = 5
   - Max number of clamped cells to trace per step (0 = disable)
   - Printed in startup banner

5. **Check script enhancement** (`check_stability_correction.py`)
   - New assertion [5] parses `[UCM][3.5A-hotfix][clamp-count]` lines from logs
   - Extracts max clamp counts across all steps (roof, wall, road)
   - Fails if `total_clamped > 10` cell-steps (arbitrary threshold; adjust for domain size)
   - Warns (but doesn't fail) if `total_diverged > 0`
   - Passes if both are zero

6. **Documentation** (`UCM_DEVELOPMENT.md`, this section)
   - Explains context: no radiation forcing in Phase 3.5a → SEB collapses to floor
   - Design decision: clamp value stays at 260 K (removing it allows unphysical negative-K)
   - Trace output shows exactly which cells clamp and why → fix target is unambiguous

### Output Examples

**Clamp-count line (step summary):**
```
[UCM][3.5A-hotfix][clamp-count] step=0 time=0
  Clamped to T_skin_min=260K:  roof=3  wall=5  road=7
  Newton diverged (hit max_iter): roof=0  wall=0  road=0
```

**Clamp-trace line (per-cell detail, first 5 clamped cells):**
```
[UCM][3.5A-hotfix][clamp-trace] ROOF cell (i=10, j=5)
  T_initial=259.8  T_final=260.0  residual=0.025  n_iter=20
  SW_abs=0.0 W/m2  LW_abs=45.0 W/m2  H_sens=60.0 W/m2  G_cond=22.0 W/m2
  Balance: SW+LW-H-G = -37.0 W/m2 (should be ~0 for converged, negative = losing heat)
```

### Physics Notes

**Why the clamp fires:** UCMBostonStabilityCorrection has no radiation forcing (Phase 4.2 pending, or must use Phase 3.5b prescribed radiation). The SEB balance is:

```
F(T_skin) = [SW_absorbed + LW_down - sigma*T_skin^4*emiss] - H_sensible - G_conduction
          = [0 + 350 - 350 - H_sens - G_cond]      (SW=0, LW_down≈LW_emitted)
          = -H_sens - G_cond
```

This has no positive input term, so Newton drives T_skin downward until it hits the floor. The trace output reveals which cells are affected and the imbalance magnitude.

**Resolution:** Phase 3.5b (prescribed diurnal SW/LW forcing) or Phase 4.2 (radiation module) will provide the missing input; then clamp count should drop to zero on daytime canonicals and remain >0 only on nighttime/overcast cells.

### Design Decisions

1. **Clamp value stays at 260 K:** Removing the floor would allow unphysical negative-Kelvin solutions on pathological cells. Instrumenting it reveals exactly which cells hit the floor and why (trace output), making the fix target (radiation input) unambiguous.

2. **Atomic counters only on GPU:** `amrex::Gpu::Atomic::AddNoRet` used for thread-safe increments inside ParallelFor; MPI reduction done outside (collective, safe).

3. **Post-loop trace on host:** MultiFab data is copied to host and scanned sequentially to find clamped cells; one I/O call per cell (gated on IOProcessor). Avoids GPU-hostile Print() inside kernel.

4. **Sentinel for no data:** If `ucm_debug=0`, no diagnostics are printed. Check script handles "not found" case gracefully.

### Backward Compatibility

- Old `solve_facet_seb()` function unchanged; new code path via `solve_facet_seb_with_diag()`
- Parameter `newton_trace_ncells=5` default is non-zero; set to 0 to disable if trace output is too verbose
- Check script tolerance `CLAMP_THRESHOLD=10` is conservative; may need tuning per domain size

### Acceptance Criteria (All Met)

1. ✅ Base branch `ERF-SLUCM`
2. ✅ Three clamp counters + three divergence counters, MPI-reduced correctly
3. ✅ Per-step `[UCM][3.5A-hotfix][clamp-count]` line printed when `ucm_debug=1`
4. ✅ Per-cell `[UCM][3.5A-hotfix][clamp-trace]` lines for first `newton_trace_ncells` clamped cells
5. ✅ `newton_trace_ncells` ParmParse parameter added and printed in startup banner
6. ✅ Check script assertion [5] parses clamp-count lines and asserts total < 10 cell-steps
7. ✅ Builds clean on `-DERF_ENABLE_UCM=ON` and `-DERF_ENABLE_UCM=OFF`
8. ✅ Newton solver algorithm, tolerance, max_iter, and clamp value UNCHANGED
9. ✅ No changes to SVF, aggregation, injection, drag, or radiation code
10. ✅ `UCM_DEVELOPMENT.md` Phase 3.5a-hotfix section added
11. ✅ PR targets `ERF-SLUCM`

### Known Limitations / Deferred

- **Diurnal cycle:** Trace output will show clamping on all cells when SW=0 (nighttime); this is physically expected and will be fixed by Phase 3.5b radiation forcing.
- **Heterogeneous clamping:** If domain has some urban and some non-urban, clamping will occur only on urban cells; check script applies global threshold (may need domain-aware tuning).

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