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

| Part | Phase | Title | Key Deliverables | Status | Post-merge fix PRs |
|------|-------|-------|------------------|--------|---------------------|
| 1 | 1.1 | **Scaffold, ParmParse, Prerequisites, lev-aware API** | UCMParams, UCMGrid, check_ucm_prerequisites, canonical test scaffold | ✅ COMPLETE | — |
| 1 | 1.2 | **UCM 2D grid + homogeneous URBPARM reader + is_urban mask** | ERF_UCMFields, allocate_ucm_fields, fill_ucm_fields_homogeneous, Phase 1.2 test | ✅ COMPLETE | — |
| 1 | 1.3 | Slab conduction + SLUCM SEB core + wind/scalar extraction | Vertical heat diffusion, sensible heat balance, wind interpolation at zref | ✅ COMPLETE | — |
| 1 | 1.4 | One-way exponential injection + diagnostics + plotfile + homogeneous regression | ATM coupling, CSV output, plotfile writer, baseline test | ✅ COMPLETE (PR #200) | — |
| 2 | 2.1 | Building-layout CSV reader + material library CSV | ERF_UCMBuildingReader, morphology per cell (H, W_road, W_roof, fabric) | ✅ COMPLETE | #203, #204, #205 |
| 2 | 2.2 | Per-cell material + morphology wiring into SEB + heterogeneous wind | 11 new MultiFabs, per-cell z0/d, wind interpolation, tests | ✅ COMPLETE (PR #206) | — |
| 2 | 2.3 | Heterogeneous facet SEB + anthropogenic heat | Wall/roof/road per-cell energy balance, waste heat injection, CSV convention lock-in | ✅ COMPLETE (PR #208) | #209 (MPI deadlock), #210 (RK persistence), #211 (RK injection 10× loss) |
| 2 | 2.4 | Shadowing + heterogeneous regression | Sky-view-factor (SVF) from canyon aspect ratio (Kusaka 2001) | ✅ COMPLETE (PR #212) | — |
| 2 | 2.5 | Scale-aware source aggregation | Multi-level morphology aggregation, subgrid variance | ✅ COMPLETE (PR #213) | #214, #215 (convention B), #216 (CSV is_urban + facet symmetry), #217 (VisMF → WriteSingleLevelPlotfile), #218 (H_atm plotfile), #219 (diag housekeeping) |
| 2 | 2.6 | Injection framework: Surface + Exponential[Scalar, Morphology] | Facet heat + Exp decay, morphology-aware injection | ✅ COMPLETE (PR #220) | — |
| 2 | 2.7 | Facet3D injection: BEP geometric overlap + terrain-following + Gaussian height PDF | Wall/roof/road 3D geometric splitting, sharp + Gaussian modes, terrain-ready coords | ✅ COMPLETE (PR #222) | — |
| 2 | 2.8 | BEP momentum drag (compressible + anelastic stub) | Wall/roof drag, Cd_wall / Cd_roof coefficients | ✅ COMPLETE (PR #223) | — |
| 2 | 2.9 | CSV generator toolchain (ideal + real-city GIS) | Synthetic pattern generators, OSM + WUDAPT ingestion, UTM-guard | ✅ COMPLETE | #207, #224 |
| 2 | 2.10 | Inflow/outflow validation cases (Salamanca + Kanda) | Non-periodic BC canonicals on compressible MRF path | ✅ COMPLETE (PR #225) | — |
| 2 | 2.11 | UCMBoston single-level one-way baseline + shared Boston test infrastructure | First real-city canonical, 5-zone concentric layout | ✅ COMPLETE (PR #226) | #229 (atm_feedback split into per-process knobs) |
| 3 | 3.1a | Level-aware allocation & call-site audit (static analysis) | `PHASE_3_1A_LEVEL_AUDIT.md` classification of 64 sites | ✅ COMPLETE (PR #230) | — |
| 3 | 3.1b | Level-awareness fixes (unblock + API hygiene) | Remove 2 blockers + 27 `int lev = 0` defaults | ✅ COMPLETE (PR #231) | #232 (param ordering hotfix), #233 (caller sites hotfix-2) |
| 3 | 3.1c | Regression harness + W_road/W_roof default fix | `run_all_regressions.sh`, per-canonical `check_*.py`, defaults 0.0→10.0 | ✅ COMPLETE (PR #234) | — |
| 3 | 3.2 | Two-way ATM→UCM data plumbing (T_air + wind only) | Pass ATM fields down for consumption by UCM | 🔲 PLANNED | — |
| 3 | 3.3 | MRF re-audit + PBLH consumer guard | Verify u*, θ* under UCM-modified profiles; assert no PBLH consumption | 🔲 PLANNED | — |
| 3 | 3.4 | Stability-aware canyon-atm exchange | Obukhov-corrected exchange coefficient consuming MRF u* | 🔲 PLANNED | — |
| 3 | 3.5 | Two-way MRF+SLUCM full loop regression | End-to-end integration gate | 🔲 PLANNED | — |
| 3 | 3.6 | UCMBoston multi-level one-way | First anchor_level>0 canonical | 🔲 PLANNED | — |
| 3 | 3.7 | anchor_level=2 stress test | 3-level nested test on urban core | 🔲 PLANNED | — |
| 3 | 3.8 | Non-urban partial-domain regression | Mixed urban+rural single ATM level | 🔲 PLANNED | — |
| 3 | 3.9 | Regression suite hardening under feedback | CI-grade harness, automate `run_all_regressions.sh` | 🔲 PLANNED | — |
| 3 | 3.10 | UCMBoston multi-level two-way | Phase 3 finale | 🔲 PLANNED | — |
| 4 | 4.1 | is_urban mask enforcement (LSM + MOST bypass) | Wiring is_urban into LSM/MOST paths, mixed urban/non-urban domains | 🔲 PLANNED | — |
| 4 | 4.2 | Radiation coupling (SW/LW extraction) | Solar + LW extraction from radiation module to UCM | 🔲 PLANNED | — |
| 4 | 4.3 | Urban/non-urban interface treatment | Boundary layer interpolation at urban perimeter | 🔲 PLANNED | — |
| 4 | 4.4 | Mixed-domain diurnal integration test | Multi-facet urban/forest/ocean test case | 🔲 PLANNED | — |
| 5 | 5.1 | Tree CSV + tree drag | Vegetation CSV reader, drag force injection | 🔲 PLANNED | — |
| 5 | 5.2 | Tree radiation (Beer-Lambert + LW crown-facet) | Canopy shortwave attenuation, crown energy balance | 🔲 PLANNED | — |
| 5 | 5.3 | Tree leaf EB + local soil bucket + transpiration | Leaf temperature, soil moisture tracking, latent flux | 🔲 PLANNED | — |
| 5 | 5.4 | Tile-averaged fluxes + instrumented-site validation | Horizontal aggregation to native ATM grid, field obs comparison | 🔲 PLANNED | — |
| 6 | 6.1 | Multi-bounce wall radiation | Ray tracing within urban canyon, multiple reflections | 🔲 PLANNED | — |
| 6 | 6.2 | AC waste heat + building-energy sub-module | HVAC rejection rate from occupancy schedules, waste injection | 🔲 PLANNED | — |
| 6 | 6.3 | Green roofs, cool roofs, permeable pavements | Heterogeneous roof/pavement albedos + soil moisture | 🔲 PLANNED | — |
| 6 | 6.4 | Worry-list audit + v1.0 release | Final regression suite, documentation, issue resolution | 🔲 PLANNED | — |

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