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

| Part | Phase | Title | Key Deliverables | Status |
|------|-------|-------|------------------|--------|
| 1 | 1.1 | **Scaffold, ParmParse, Prerequisites, lev-aware API** | UCMParams, UCMGrid, check_ucm_prerequisites, canonical test scaffold | ✅ COMPLETE |
| 1 | 1.2 | 2D UCM grid + homogeneous URBPARM reader + is_urban stub | ERF_UCMLayer, urban morphology allocation, is_urban iMultiFab | 🔲 PLANNED |
| 1 | 1.3 | Slab conduction + SLUCM SEB core + wind/scalar extraction | Vertical heat diffusion, sensible heat balance, wind interpolation at zref | 🔲 PLANNED |
| 1 | 1.4 | One-way exponential injection + diagnostics + plotfile + homogeneous regression | ATM coupling, CSV output, plotfile writer, baseline test | 🔲 PLANNED |
| 2 | 2.1 | Building-layout CSV reader + material library CSV | ERF_UCMBuildingReader, morphology per cell (H, W_road, W_roof, fabric) | 🔲 PLANNED |
| 2 | 2.2 | Per-cell morphology + heterogeneous canopy wind | Heterogeneous tower morphologies, per-cell wind extraction | 🔲 PLANNED |
| 2 | 2.3 | Heterogeneous facet SEB + anthropogenic heat | Wall/roof/road per-cell energy balance, waste heat injection | 🔲 PLANNED |
| 2 | 2.4 | Shadowing + heterogeneous regression | Sun angle shadow mapping, heterogeneous baseline regression | 🔲 PLANNED |
| 3 | 3.1 | Finest-level anchoring turned on + multi-level regression | anchor_level > 0 enabled, multi-AMR-level UCM slab | 🔲 PLANNED |
| 3 | 3.2 | Two-way feedback + MRF audit + PBLH consumer guard | Inverse coupling (atm_feedback > 0), MRF re-audit for divergence | 🔲 PLANNED |
| 3 | 3.3 | Stability-aware canyon-atm exchange | Obukhov-length dependent exchange, skipping during neutral | 🔲 PLANNED |
| 3 | 3.4 | Two-way MRF regression + energy conservation | MRF+SLUCM validation, energy budget closure test | 🔲 PLANNED |
| 4 | 4.1 | is_urban mask enforcement (LSM + MOST bypass) | Wiring is_urban into LSM/MOST paths, mixed urban/non-urban domains | 🔲 PLANNED |
| 4 | 4.2 | Radiation coupling (SW/LW extraction) | Solar + LW extraction from radiation module to UCM | 🔲 PLANNED |
| 4 | 4.3 | Urban/non-urban interface treatment | Boundary layer interpolation at urban perimeter | 🔲 PLANNED |
| 4 | 4.4 | Mixed-domain diurnal integration test | Multi-facet urban/forest/ocean test case | 🔲 PLANNED |
| 5 | 5.1 | Tree CSV + tree drag | Vegetation CSV reader, drag force injection | 🔲 PLANNED |
| 5 | 5.2 | Tree radiation (Beer-Lambert + LW crown-facet) | Canopy shortwave attenuation, crown energy balance | 🔲 PLANNED |
| 5 | 5.3 | Tree leaf EB + local soil bucket + transpiration | Leaf temperature, soil moisture tracking, latent flux | 🔲 PLANNED |
| 5 | 5.4 | Tile-averaged fluxes + instrumented-site validation | Horizontal aggregation to native ATM grid, field obs comparison | 🔲 PLANNED |
| 6 | 6.1 | Multi-bounce wall radiation | Ray tracing within urban canyon, multiple reflections | 🔲 PLANNED |
| 6 | 6.2 | AC waste heat + building-energy sub-module | HVAC rejection rate from occupancy schedules, waste injection | 🔲 PLANNED |
| 6 | 6.3 | Green roofs, cool roofs, permeable pavements | Heterogeneous roof/pavement albedos + soil moisture | 🔲 PLANNED |
| 6 | 6.4 | Worry-list audit + v1.0 release | Final regression suite, documentation, issue resolution | 🔲 PLANNED |

---

## Cross-Cutting Design Contracts (Enforced Phase 1.1+)

### Contract 1 — `lev`-aware API from day 1

All UCM public functions (constructors, initializers, coupling, diagnostics) take `int lev` as an argument with default value `0`. **No hardcoded `0` may appear inside any file in `Source/UrbanCanopy/` for level indexing.** The only allowed hardcoded `0` is for component indices (e.g., `array[0]`) or spatial indices.

All UCM MultiFab members held by the `ERF` class are declared as `amrex::Vector<std::unique_ptr<amrex::MultiFab>>` sized to `finest_level+1`. In Phase 1.1, only index `anchor_level` is allocated; other indices remain `nullptr`.

### Contract 2 — Anchor level and static refinement

Two runtime parameters lock AMR strategy:

- **`erf.ucm.anchor_level`** (int, default `0`): The single AMR level UCM runs on. Must satisfy `0 ≤ anchor_level ≤ finest_level`. In Phase 1.1, only `anchor_level=0` is exercised; higher values are gated by an assertion in the prerequisites check (Phase 3.1 relaxes this).
- **`erf.ucm.static_refinement`** (bool, default `true`): Required to be `true`. If AMR issues a regrid on `anchor_level` during the run, UCM must error out. In Phase 1.1, install the assertion; regrid-detection hook is a TODO comment (Phase 3.3).

### Contract 3 — Terrain-following coordinates

All vertical operations use `z_phys_cc(i, j, k) - z_phys_cc(i, j, klo)` for height-above-surface. No hardcoded z-coordinate is permitted anywhere in `Source/UrbanCanopy/`. Rules:

1. Wind and scalar extraction (Phase 1.3) use `z_target = z_phys_cc(i, j, klo) + H_bldg(i, j) + zref`.
2. Exponential injection (Phase 1.4) evaluates `exp(-(z_phys_cc(i, j, k) - z_phys_cc(i, j, klo)) / alpha_ucm)` per column, exactly matching the Fire/Dust pattern.
3. Terrain slope > 20° at any urban cell → warning at initialization. > 30° → error unless `erf.ucm.allow_steep_terrain = true`.
4. Prerequisites check (Phase 1.1): if `erf.use_terrain = true`, initialize expecting `z_phys_cc[anchor_level]` populated. If `erf.use_terrain = false`, `z_phys_cc[anchor_level]` is flat by construction; same code path handles both.

### Contract 4 — Zero PBLH dependency in UCM physics

UCM MUST NOT call `SurfaceLayer::get_pblh(lev)` in any code path. All near-surface stability information comes from `u*`, `theta*`, `q*`, and the Obukhov length `L`, all reliably populated by MRF/YSU/MYNN2.5. This is a **hard rule** enforced by code review.

### Contract 5 — `is_urban(i, j)` mask exclusivity

Declare a per-level `ucm_is_urban` `iMultiFab` member on `ERF`, allocated on the UCM 2D slab at `anchor_level` (allocation actually happens Phase 1.2 when UCM grid is created; Phase 1.1 only has declaration). In Phase 1.1, no LSM/MOST bypass hooks are wired; that is Phase 4.1. Only field declaration and TODO comments are added.

### Contract 6 — Build and file conventions (from Fire/Dust/LNG)

1. Every new `.cpp` file MUST be added to BOTH `CMake/BuildERFExe.cmake` (in a new `Source/UrbanCanopy/` block mirroring Dust) AND `Source/UrbanCanopy/Make.package`.
2. Header-only files go into `Source/UrbanCanopy/Make.package` under `CEXE_headers` only.
3. Never wrap `.cpp` file bodies in `#ifdef ERF_USE_UCM`. Only wrap `#include` of UCM headers in consumer files (e.g., `Source/ERF.cpp`) if a build-time toggle is used.
4. All UCM MultiFabs use `amrex::IntVect(1, 1, 0)` ghost cells (1 in x, 1 in y, 0 in z — 2D slab).
5. CSV readers (future phases) MUST follow the rank-0-read + `MPI_Bcast` of POD structs pattern documented in `UCM_MPI_SKILLS.md`. In Phase 1.1, no CSV is read.
6. Never `#include <AMReX_ParallelFor.H>` in stub files. Only include where a `ParallelFor` is actually used (Phase 1.3 and later).

---

## Phase 1.1: Directory Scaffold & ParmParse Integration

Phase 1.1 is the **scaffold phase** that produces a **compilable, no-op module**:

- Registers ParmParse parameters under `erf.ucm.*`
- Validates prerequisites at ERF initialization
- Provides `lev`-aware public API skeletons (all empty stubs)
- Locks in the cross-cutting design contracts
- Ships with one canonical test (UCMScaffold) exercising "module enabled but no physics runs yet"

**Phase 1.1 does NOT compute any physics.** All physics functions are declared and defined as empty stubs returning without touching any MultiFab. Purpose: establish module skeleton, build integration, and parameter/prerequisite infrastructure for subsequent phases.

### All 22 Phase 1.1 Parameters (Defined in ERF_UCMParams.H)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable` | `bool` | `false` | Enable UCM module |
| `ucm_debug` | `bool` | `false` | Per-step debug output |
| `anchor_level` | `int` | `0` | AMR level at which UCM runs |
| `static_refinement` | `bool` | `true` | Refinement must not change during run |
| `grid_ratio` | `int` | `1` | UCM grid refinement vs ATM anchor level |
| `allow_steep_terrain` | `bool` | `false` | Permit slopes > 30° |
| `atm_feedback` | `amrex::Real` | `0.0` | Coupling strength [0,1]; 0=one-way |
| `alpha_ucm` | `amrex::Real` | `10.0` | Exponential injection e-folding depth [m] |
| `zref` | `amrex::Real` | `2.0` | Wind extraction reference height above roof [m] |
| `H_bldg_uniform` | `amrex::Real` | `10.0` | Uniform building height [m] |
| `W_road_uniform` | `amrex::Real` | `10.0` | Uniform road width [m] |
| `W_roof_uniform` | `amrex::Real` | `10.0` | Uniform roof width [m] |
| `albedo_roof` | `amrex::Real` | `0.20` | Homogeneous roof SW albedo |
| `albedo_wall` | `amrex::Real` | `0.20` | Homogeneous wall SW albedo |
| `albedo_road` | `amrex::Real` | `0.15` | Homogeneous road SW albedo |
| `emissivity_roof` | `amrex::Real` | `0.90` | Homogeneous roof LW emissivity |
| `emissivity_wall` | `amrex::Real` | `0.90` | Homogeneous wall LW emissivity |
| `emissivity_road` | `amrex::Real` | `0.94` | Homogeneous road LW emissivity |
| `ucm_plot_int` | `int` | `-1` | Plotfile interval (-1 disabled) |
| `ucm_diag_file` | `std::string` | `"ucm_diag.dat"` | Diagnostics CSV filename |
| `test_ustar` | `amrex::Real` | `0.0` | Placeholder u* before Phase 1.3 [m/s] |
| `test_surf_temp_K` | `amrex::Real` | `293.15` | Placeholder T_sfc before Phase 1.3 [K] |

### Phase 1.1 Deliverables

All files in `Source/UrbanCanopy/`:

1. **ERF_UCMParams.H** – Struct with 22 parameters + `read_from_parmparse(int lev = 0)` method
2. **ERF_UCMParams.cpp** – ParmParse implementation
3. **ERF_UCMPrerequisites.H/.cpp** – 12 checks (range, phase constraints, grid decomposition)
4. **ERF_UCMGrid.H** – `struct UCMGrid` and `create_ucm_grid()` declaration
5. **ERF_UCMGrid.cpp** – Grid creation stub (returns default-constructed `UCMGrid`)
6. **ERF_UCM.H** – Top-level module header with roadmap table and six contracts
7. **ERF_UCM.cpp** – Empty anchor file with TODO comments for all phases
8. **Make.package** – GNUmake registration (Grid, Prerequisites, Params, UCM first; then headers)
9. **CMake/BuildERFExe.cmake block** – Mirroring Dust pattern
10. **Source/ERF.H** – Add `#include` and member `UCMParams m_ucm_params;` (guarded)
11. **Source/ERF.cpp** – Add `m_ucm_params.read_from_parmparse()` and prerequisite check call
12. **Exec/CanonicalTests/SLUCM/UCMScaffold/** – Test directory with inputs, CMakeLists.txt, README

### Build Rules (Phase 1.1+)

- Every `.cpp` file must appear in BOTH `Make.package` (under `CEXE_sources`) AND `CMake/BuildERFExe.cmake` (under `target_sources`)
- Ordering critical: `ERF_UCMGrid.cpp` and `ERF_UCMPrerequisites.cpp` before physics modules
- All `.H` files listed in Make.package `CEXE_headers`
- Header-only files (.H with no corresponding .cpp) only in CEXE_headers

### Test-Input Requirements

Canonical test `UCMScaffold`:
- Domain: 8×8×16 (small, fast)
- PBL: MYNN2.5 (simplest available)
- `amr.max_level = 0` (no AMR)
- `erf.ucm.enable = true`
- `erf.ucm.ucm_debug = true`
- `erf.ucm.anchor_level = 0`
- `erf.ucm.grid_ratio = 1`
- `max_step = 2`, `dt = 1.0`
- `erf.use_terrain = false` (flat terrain)
- No moisture, no radiation (neutral ABL only)
- Sounding: neutral profile or from `Exec/CanonicalTests/Dust/DustIntegration/`

Expected output:
1. Exit code 0 at step 2
2. Startup banner with all UCM parameters (gated by `[UCM]` prefix)
3. Stub debug message: `[UCM DEBUG] create_ucm_grid stub called (Phase 1.1 no-op)`
4. Prerequisite check passes (no aborts)
5. **Phase 1.4 TODO:** Bit-for-bit non-UCM run comparison (proves genuine no-op physics)

---

## References

- **Source/Dust/DUST_DEVELOPMENT.md** (ERF-Hazard branch): Mineral dust module roadmap and design patterns
  https://github.com/hgopalan/ERF/blob/ERF-Hazard/Source/Dust/DUST_DEVELOPMENT.md

- **Source/LNG/LNG_DEVELOPMENT.md** (commit ba55b7307...): LNG hazardous gas module initialization patterns
  https://github.com/hgopalan/ERF/blob/ba55b7307/Source/LNG/LNG_DEVELOPMENT.md

- **Source/LNG/LNG_MPI_SKILLS.md** (commit ba55b7307...): MPI multi-rank lessons for 2D slab modules
  https://github.com/hgopalan/ERF/blob/ba55b7307/Source/LNG/LNG_MPI_SKILLS.md

- **Structural reference files** (patterns only, no modifications):
  - Source/Fire/ERF_FireGrid.H
  - Source/Fire/ERF_FireAtmCoupling.H
  - Source/Dust/ERF_DustGrid.H
  - Source/Dust/ERF_DustAtmCoupling.H
  - Source/Dust/ERF_DustPrerequisites.H
  - Source/LNG/ERF_LNGParams.H
  - Source/LNG/ERF_LNGPrerequisites.H
