# ERF-SLUCM Single-Layer Urban Canopy Model — Development Log

## Overview

The ERF-SLUCM module simulates the thermal and momentum exchange between urban surfaces (buildings, roads, vegetation) and the atmosphere. It is implemented as a 2D refined slab tightly coupled to the ERF mesoscale atmospheric model. Phase 1 focuses on one-way coupling with homogeneous canopy; Phase 2 extends to heterogeneous morphology via CSV; Phase 3 adds two-way feedback; Phases 4–7 add advanced processes (urban/non-urban treatment, advanced UCM physics, tree physics, v1.0 release).

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
| 2 | 2.8 | BEP momentum drag (compressible + anelastic stub) | Wall/roof drag, Cd_wall / Cd_roof coefficients | ✅ COMPLETE | #223 |
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
| 3 | 3.5a-hotfix | **Seven-bug debugging cascade (physics closure)** | Newton clamp instrumentation, SEB self-consistency, T_skin persistence, canyon-air thermal inertia, T_slab init + MOST sign, TDMA/slab-BC/hour-angle/canyon-LW fixes | ✅ COMPLETE | #241, #242, #243, #244, #245 |
| 3 | 3.5c | **Two-way MRF+SLUCM full-loop regression (24-hr diurnal)** | End-to-end integration gate: 24-hour sim with radiation forcing, verify diurnal cycle, UHI stability, no slab drift. Canonical: `UCMBostonDiurnal24h/` | ✅ COMPLETE | #246, #247 (threshold relax), #248 (50 cm / 6-layer slab) |
| 3 | 3.6 | UCMBoston multi-level one-way | First anchor_level>0 canonical (`UCMBostonMultiLevel/`, amr.max_level=1, anchor_level=1) | ✅ COMPLETE | #249 |
| 3 | 3.7 | Physical-coordinate CSV lookup for building layout | Backward-compatible physical/legacy mode auto-detect for building_layout.csv (unblocks nested + real-city canonicals) | ✅ COMPLETE | #250 |
| 3 | 3.8 | Non-urban partial-domain regression | Mixed urban+rural single ATM level (`UCMBostonMixedDomain/`) | ✅ COMPLETE | #251 |
| 3 | 3.9 | **Regression suite hardening + unit tests** | 6-test GoogleTest suite (`Tests/Unit/UrbanCanopy/erf_ucm_unit_tests`) covering TDMA identity, Newton SEB day/night, Businger-Dyer, CSV reader/consumer; CI workflow `.github/workflows/slucm_regression.yml` (unit + canonical) | ✅ COMPLETE | #252 |
| 3 | 3.10 | UCMBoston multi-level two-way | Phase 3 finale: `amr.max_level ≥ 1` with `atm_feedback_heat=1.0` end-to-end | 🔲 PLANNED | — |
| 4 | 4.1 | is_urban mask enforcement (LSM + MOST bypass) | Wiring is_urban into LSM/MOST paths, mixed urban/non-urban domains | 🔲 PLANNED | — |
| 4 | 4.2 | Radiation coupling (SW/LW extraction) | Solar + LW extraction from radiation module to UCM; replaces Phase 3.5b analytic formulae; **regression: match Phase 3.5b analytic to <5% at noon summer solstice** | 🔲 PLANNED | — |
| 4 | 4.3 | Urban/non-urban interface treatment | Boundary layer interpolation at urban perimeter | 🔲 PLANNED | — |
| 4 | 4.4 | Mixed-domain diurnal integration test | Multi-facet urban/forest/ocean test case | 🔲 PLANNED | — |
| 5 | 5.1 | Multi-bounce wall radiation | Ray tracing within urban canyon, multiple reflections; **may resolve Newton/MOST divergence design gap** | 🔲 PLANNED | — |
| 5 | 5.2 | AC waste heat + building-energy sub-module | HVAC rejection rate from occupancy schedules, waste injection | 🔲 PLANNED | — |
| 5 | 5.3 | Green roofs, cool roofs, permeable pavements | Heterogeneous roof/pavement albedos + soil moisture | 🔲 PLANNED | — |
| 6 | 6.1 | Tree CSV + tree drag | Vegetation CSV reader, drag force injection | 🔲 PLANNED | — |
| 6 | 6.2 | Tree radiation (Beer-Lambert + LW crown-facet) | Canopy shortwave attenuation, crown energy balance | 🔲 PLANNED | — |
| 6 | 6.3 | Tree leaf EB + local soil bucket + transpiration | Leaf temperature, soil moisture tracking, latent flux | 🔲 PLANNED | — |
| 6 | 6.4 | Tile-averaged fluxes + instrumented-site validation | Horizontal aggregation to native ATM grid, field obs comparison | 🔲 PLANNED | — |
| 7 | 7.1 | Worry-list audit + v1.0 release | Final regression suite, documentation, issue resolution | 🔲 PLANNED | — |

**Phase 3 status (as of 2026-07-28):** 3.1a → 3.9 complete (28 PRs, #200–#252). **Only Phase 3.10 (multi-level two-way finale) remains** before Phase 4.

---

