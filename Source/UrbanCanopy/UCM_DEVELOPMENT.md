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
| 1 | 1.1 | **Scaffold, ParmParse, Prerequisites, lev-aware API** | UCMParams, UCMGrid, check_ucm_prerequisites, canonical test scaffold | ✅ COMPLETE (with post-merge bug fixes) |
| 1 | 1.2 | **UCM 2D grid + homogeneous URBPARM reader + is_urban mask** | ERF_UCMFields, allocate_ucm_fields, fill_ucm_fields_homogeneous, Phase 1.2 test | 🟢 IN PROGRESS |
| 1 | 1.3 | Slab conduction + SLUCM SEB core + wind/scalar extraction | Vertical heat diffusion, sensible heat balance, wind interpolation at zref | 🔲 PLANNED |
| 1 | 1.4 | One-way exponential injection + diagnostics + plotfile + homogeneous regression | ATM coupling, CSV output, plotfile writer, baseline test | 🔲 PLANNED |
| 2 | 2.1 | Building-layout CSV reader + material library CSV | ERF_UCMBuildingReader, morphology per cell (H, W_road, W_roof, fabric) | ✅ COMPLETE (PRs #203, #204, #205) |
| 2 | 2.2 | Per-cell material + morphology wiring into SEB + heterogeneous wind | 11 new MultiFabs, per-cell z0/d, wind interpolation, tests | 🟢 IN PROGRESS |
| 2 | 2.3 | Heterogeneous facet SEB + anthropogenic heat | Wall/roof/road per-cell energy balance, waste heat injection, CSV convention lock-in | 🟢 IN PROGRESS |
| 2 | 2.4 | Shadowing + heterogeneous regression | Sun angle shadow mapping, heterogeneous baseline regression | 🔲 PLANNED |
| 2 | 2.5 | Scale-aware source aggregation | Multi-level morphology aggregation, subgrid variance | 🔲 PLANNED |
| 2 | 2.6 | Injection framework: Surface + Exponential[Scalar, Morphology] | Facet heat + Exp decay, morphology-aware injection | 🔲 PLANNED |
| 2 | 2.7 | Facet3D injection | True 3D canyon exchange, vertical walls | 🔲 PLANNED |
| 2 | 2.8 | BEP-line injection | Building Energy Performance canyon injection | 🔲 PLANNED |
| 2 | **2.9** | **CSV generator toolchain (ideal + real-city GIS)** | Synthetic pattern generators, OSM + WUDAPT ingestion, UTM-guard | 🟢 IN PROGRESS |
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

---

## Phase 1.1 Post-Merge Bug Fixes

Phase 1.1 was merged to `ERF-SLUCM` branch but produced five bugs that the maintainer fixed by hand:

### Bug 1 — Wrong PBL scheme and broken atmospheric config in the canonical test

**Fix commits:**
- [`7bb380d`](https://github.com/hgopalan/ERF/commit/7bb380d47738c24a47e3af8c7729a23dc815053e) — Update SLUCM UCMScaffold inputs to use neutral ABL atmospheric setup with sounding file
- [`1705e38`](https://github.com/hgopalan/ERF/commit/1705e3844eb232da04151f263cfa38e49b67539a) — Switch SLUCM UCMScaffold PBL to MRF (match neutral_abl)

**Root cause:** Agent synthesized fake sounding instead of using reference file; used wrong sounding column format, wrong `n_cell`, `amr.dt_shrink` (not applicable), wrong boundary types (`slip_wall` vs `SlipWall`), omitted `erf.prob_name`, omitted surface-layer roughness, omitted Coriolis, used `MYNN2.5` instead of `MRF`.

**Phase 1.2 rule:** Use merged `Exec/CanonicalTests/SLUCM/UCMScaffold/inputs` as verbatim baseline. Copy `sounding_neutral_abl` byte-for-byte. Do NOT synthesize soundings.

### Bug 2 — Wrong SolverChoice member for terrain check

**Fix commit:** [`3744d41`](https://github.com/hgopalan/ERF/commit/3744d41f8c42684bc6b8f1ecd356a03886000221) — Fix ERF.cpp

**Root cause:** Called `solverChoice.use_terrain` which is not a member of `SolverChoice`. The correct expression is `(solverChoice.terrain_type != TerrainType::None)`.

**Phase 1.2 rule:** Any terrain-availability check MUST use `(solverChoice.terrain_type != TerrainType::None)`. Do not invent member names.

### Bug 3 — ParmParse read guarded by the very field it was supposed to populate

**Fix commit:** [`927700f`](https://github.com/hgopalan/ERF/commit/927700ff7b4507ff7230498dc817a9bb2ee29306) — Missing Params

**Root cause:** Wrote `if (m_ucm_params.enable) { m_ucm_params.read_from_parmparse(0); }` — but `enable` starts at `false`, so the read never happens and every parameter stays at default. Correct pattern is unconditional read.

**Phase 1.2 rule:** `read_from_parmparse` MUST be called unconditionally at ERF startup. Guards go only on downstream setup (grid creation, allocation), never on the read.

### Bug 4 — Debug messages missing or sparse

Phase 1.1 emitted a startup banner but no per-function debug traces. Users cannot tell whether a UCM code path ran or what intermediate state looks like.

**Phase 1.2 rule (NEW, MANDATORY):** Every non-trivial UCM function MUST emit `[UCM][1.2]` debug message when `params.ucm_debug == true`. See "Debug Message Contract" section below.

### Bug 5 — Hardcoded `0` slipped into level-argument positions

Not yet fixed, but grep the Phase 1.1 code before starting Phase 1.2. If any UCM public function passes hardcoded `0` for a level argument (not a component index), fix as Phase 1.2 drive-by.

**Phase 1.2 rule:** Continue enforcing lev-aware API contract from Phase 1.1.

---

## Phase 1.2: UCM 2D Grid + Homogeneous URBPARM MultiFabs + is_urban Mask

Phase 1.2 builds the **UCM 2D grid infrastructure** and **homogeneous URBPARM MultiFab fields**, plus declares and allocates the `is_urban(i, j)` mask. **No physics computations are performed yet.** After Phase 1.2:

- `create_ucm_grid` is implemented (no longer a stub) and returns a properly-refined 2D BoxArray aligned with ATM DistributionMapping
- All homogeneous URBPARM parameters are broadcast into per-cell MultiFabs on UCM grid
- `ucm_is_urban` iMultiFab is allocated and filled to `1` everywhere (fully urban homogeneous patch)
- A canonical test runs to completion, verifies grid dimensions, spot-checks MultiFab values, bit-for-bit ATM regression
- Every UCM function emits `[UCM][1.2]` debug messages under `ucm_debug = true`

### Phase 1.2 Algorithm — Mirror Dust/Fire Pattern Exactly

#### `create_ucm_grid(ba_atm, dm_atm, geom_atm, grid_ratio, lev)`

1. Extract k=0 slab from each Box in `ba_atm` by explicit Box manipulation (`setSmall(2,0)`, `setBig(2,0)`)
2. Refine 2D BoxArray by `IntVect(grid_ratio, grid_ratio, 1)` using `amrex::refine()`
3. Reuse `dm_atm` directly — refinement preserves box count, so box `i` in refined array is owned by same rank
4. Construct refined 2D Geometry: hi-index = `old_hi * grid_ratio + (grid_ratio - 1)`; physical domain x-y from ATM, z set to dummy 1 m

#### `allocate_ucm_fields(fields, ucm_grid, params, lev)`

Allocates 16 MultiFabs on UCM BoxArray with ghost `IntVect(1,1,0)` and `ncomp=1`:

- **Morphology:** `H_bldg`, `W_road`, `W_roof`
- **Shortwave albedos:** `albedo_roof`, `albedo_wall`, `albedo_road`
- **Longwave emissivities:** `emissivity_roof`, `emissivity_wall`, `emissivity_road`
- **Temperatures (placeholder, Phase 1.3 SEB replaces):** `T_skin_roof`, `T_skin_wall`, `T_skin_road`, `T_canyon_air`
- **Fluxes (placeholder, Phase 1.3 SEB replaces):** `H_sensible`, `LE_latent`
- **Urban mask:** `is_urban` (iMultiFab, 0/1 mask)

#### `fill_ucm_fields_homogeneous(fields, params, lev)`

Sets every field to uniform value from `UCMParams`:

- Morphology ← `params.H_bldg_uniform`, `W_road_uniform`, `W_roof_uniform`
- Albedos ← `params.albedo_roof`, `albedo_wall`, `albedo_road`
- Emissivities ← `params.emissivity_roof`, `emissivity_wall`, `emissivity_road`
- `T_skin_*`, `T_canyon_air` ← `params.test_surf_temp_K`
- `H_sensible`, `LE_latent` ← `0.0`
- `is_urban` ← `1` (everywhere — homogeneous urban patch)

### Debug Message Contract (Phase 1.2)

Every non-trivial UCM function MUST emit debug output when `params.ucm_debug == true`. Format:

```
[UCM][1.2][<function_name>] <description with key values>
```

**Minimum required traces:**

1. **`create_ucm_grid`** — (a) ATM `ba` size and box count, (b) ATM domain extents, (c) `grid_ratio`, (d) UCM `ba` size and box count, (e) UCM domain extents. One message per line.
2. **`allocate_ucm_fields`** — Per-MultiFab: name, box count, ngrow, ncomp. Summary: "allocated N MultiFabs on UCM grid at lev=X"
3. **`fill_ucm_fields_homogeneous`** — Per-field value being set
4. **`all_allocated`** — No output on success; per-field "MISSING: <name>" if any pointer null
5. **`check_ucm_grid_and_fields`** — Phase 1.2 grid-check banner: UCM extents, refinement ratio, ghost cells, allocation status

**Rate-limiting rules:**

- All debug via `amrex::Print()` (IO rank only)
- Never `amrex::AllPrint` (per-rank)
- No debug inside `ParallelFor` kernels
- Emit once per call (not per box/cell)

### Canonical Test (UCMHomogeneousGrid)

New test directory: `Exec/CanonicalTests/SLUCM/UCMHomogeneousGrid/`

**Test configuration:**

- ATM domain: `8 × 8 × 64` cells, MRF PBL, neutral ABL
- UCM grid: `8 × 8 × 1` → refined `16 × 16 × 1` (grid_ratio=2)
- Fields: All homogeneous from ParmParse
- Sounding: Byte-for-byte copy of `UCMScaffold/sounding_neutral_abl`
- Steps: 2 @ 1.0 s `fixed_dt`

**Pass criteria:**

1. Exit code 0 at step 2
2. UCM grid extents `16 × 16 × 1`
3. All Phase 1.2 debug banners printed (see Debug Message Contract above)
4. Spot-checks on MultiFab values match ParmParse (e.g., `H_bldg=10.0`, `albedo_roof=0.20`)
5. Bit-for-bit ATM regression: run with and without UCM enabled; final-step `Rho`, `RhoTheta`, `U`, `V`, `W` identical

### Phase 1.2 Deliverables

All files in `Source/UrbanCanopy/`:

1. **ERF_UCMFields.H** – Struct with 16 MultiFab fields (3 morphology, 6 radiative, 4 temp, 2 flux, 1 mask)
2. **ERF_UCMAllocate.H/.cpp** – `allocate_ucm_fields()`, `fill_ucm_fields_homogeneous()`, `UCMFields::all_allocated()`
3. **ERF_UCMGrid.cpp** – Full implementation with debug messages (replacing Phase 1.1 stub)
4. **ERF_UCMPrerequisites.cpp** – New `check_ucm_grid_and_fields()` function
5. **Modified ERF.H** – Change `ucm_is_urban` standalone iMultiFab to `m_ucm_fields` UCMFields struct Vector
6. **Modified ERF.cpp** – Add grid/field allocation calls in `InitData_post()`
7. **CMake/BuildERFExe.cmake** – Register new files
8. **Make.package** – Register new files
9. **Exec/CanonicalTests/SLUCM/UCMHomogeneousGrid/** – Test directory with inputs, sounding, CMakeLists.txt, README

### Build Integration (Phase 1.2)

- Add `ERF_UCMAllocate.cpp` to both `Make.package` and `CMake/BuildERFExe.cmake`
- Add `ERF_UCMFields.H` and `ERF_UCMAllocate.H` to `CEXE_headers`
- Ensure build succeeds with `-DERF_ENABLE_UCM=ON` and `-DERF_ENABLE_UCM=OFF`

---

## Phase 1.3: Slab Conduction + SLUCM SEB + Terrain-Aware Extraction

Phase 1.3 implements the **Surface Energy Balance (SEB) solver** for the UCM canopy, extracting wind and scalar fields from the atmosphere at the canopy reference height, and computing surface temperatures and sensible/latent heat fluxes.

### Phase 1.3 Status: ✅ COMPLETE

**Post-merge bug fixes** (learned in development):

**Bug 7** – `[f0b2ef3](https://github.com/hgopalan/ERF/commit/f0b2ef3)` "Fix UCMWindExtract: use array() by value, correct ParallelFor lambda signature"
- **Wrong:** Used `MultiFab[mfi].array()` returning by reference (`auto&`)
- **Correct:** Use `mf.array(mfi)` (by value) or `mf.const_array(mfi)` (read-only)
- **Also wrong:** ParallelFor signature `[=] AMREX_GPU_DEVICE(amrex::Box const& tbx) { LoopConcurrentOnCpu(...); }`
- **Correct:** `[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { ... }`
- **Also wrong:** `amrex::Copy(...)` (does not exist)
- **Correct:** `amrex::MultiFab::Copy(dst, src, srccomp, dstcomp, ncomp, ngrow)`

**Bug 8** – `[8c1cddb](https://github.com/hgopalan/ERF/commit/8c1cddb)` "Fix UCM advance call: SurfaceLayer accessors take lev arg and return MultiFab*"
- **Wrong:** Called `m_SurfaceLayer->get_u_star()[lev]` (missing `lev` argument)
- **Correct:** `m_SurfaceLayer->get_u_star(lev)` returns `MultiFab*` (single pointer, not a Vector)
- **Also wrong:** `m_SurfaceLayer->get_theta_star()` (does not exist)
- **Correct:** Use `get_t_star(lev)`, `get_q_star(lev)` for theta and moisture star values
- **Dereferencing:** All SurfaceLayer accessors return `MultiFab*`, so use `*m_SurfaceLayer->get_u_star(lev)` to pass by reference

### Phase 1.3 Deliverables

1. **`ERF_UCMLayer.H/cpp`** — Core SEB solver class with `advance()` method
2. **`ERF_UCMSlabConduction.H`** — Vertical heat conduction kernel (terrain-aware)
3. **`ERF_UCMWindExtract.H/cpp`** — Terrain-following ATM wind/scalar extraction at canopy reference height
4. **Canonical test:** `Exec/CanonicalTests/SLUCM/UCMHomogeneousGrid/` — 1 diurnal cycle with SEB on

### Phase 1.3 Physics

- **Wind extraction:** ATM wind interpolated to height `z_target = z_phys_cc(i,j,klo) + H_bldg(i,j) + zref` using log-law profile
- **Scalar extraction:** Similar interpolation for temperature and moisture
- **SEB:** Canyon air temperature solved via energy balance of sensible/latent fluxes; roof/wall/road temperatures from slab conduction
- **Latent heat:** Computed but set to 0 in Phase 1.3 (placeholder for Phase 2.3 anthropogenic heat)
- **Urban mask:** `is_urban` guarding enforced in every kernel

### Phase 1.3 AMReX API Rules (MANDATORY for Phase 1.4+)

These rules prevent GPU/MPI correctness bugs:

1. **ParallelFor signature:** Always use `[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { ... }`. Never use `auto&` for Array4 or Box-based lambdas.
2. **Array4 access:** Use `mf.array(mfi)` (by value) or `mf.const_array(mfi)` for read-only. Never `mf[mfi].array()` with `auto&`.
3. **MultiFab copy:** Use `amrex::MultiFab::Copy(dst, src, srccomp, dstcomp, ncomp, ngrow)`. Never bare `amrex::Copy(...)`.
4. **SurfaceLayer accessors:** All take `lev` argument and return `MultiFab*` (single pointer). Dereference with `*get_u_star(lev)` etc. Never `get_theta_star()` — use `get_t_star(lev)` instead.

---

## Phase 1.4: One-Way Exponential Injection + Diagnostics + Plotfile

Phase 1.4 turns UCM into a **fully one-way coupled** module. The sensible and latent heat fluxes computed in Phase 1.3 are coarsened to the ATM grid and injected back into `cc_source` using the WRF-SFIRE exponential-decay pattern (Mandel 2011). This is the first phase where UCM affects atmospheric state.

### Phase 1.4 Deliverables

1. **`ERF_UCMAtmCoupling.H/cpp`** — Coarsen + exponential injection pipeline
   - `coarsen_ucm_flux_to_atm()` — Downsample UCM fluxes using `amrex::average_down` when `grid_ratio > 1`
   - `apply_ucm_tendency_to_cc_source()` — Inject vertical exponential tendency into `RhoTheta_comp` and optionally `RhoQ1_comp`

2. **`ERF_UCMPlotfile.H/cpp` + `ERF_UCMPlotfileCatalog.H`** — Output writer
   - `UCMPlotfile` class produces `plt_ucm_NNNNN` files on the native UCM grid
   - 16-component catalog: morphology, albedo/emissivity, temperatures, fluxes, urban mask
   - Called via ERF plotfile hook when `ucm_plot_int > 0`

3. **`ERF_UCMDiagnostics.H/cpp`** — CSV statistics logger
   - Append per-step row to `ucm_diag.dat`: `step, time_s, T_skin_roof_max, T_skin_wall_max, T_skin_road_max, T_canyon_max, H_sensible_max, H_sensible_sum, LE_latent_max`
   - MPI-safe reductions (AllReduce min/max/sum before rank-0 write)
   - Duplicate-write guard; called every step when `ucm_diag_file` specified

4. **Modified `ERF_UCMPrerequisites.cpp`** — Relaxed validation
   - `atm_feedback` now allowed in `[0.0, 1.0]` (was hard-locked at `0.0`)
   - Abort if `atm_feedback < 0.0` or `> 1.0`
   - Startup banner now lists Phase 1.4 parameters: `alpha_ucm`, `atm_feedback`, `ucm_plot_int`, `ucm_diag_file`

5. **Canonical test:** `Exec/CanonicalTests/SLUCM/UCMOneWayInject/`
   - 12-hour diurnal cycle @ hourly steps
   - `atm_feedback = 1.0` (full coupling)
   - `alpha_ucm = 15.0 m` (e-folding depth)
   - Produces `plt_ucm_NNNNN` and `ucm_diag.dat` with injection effects on ATM state

### Phase 1.4 Physics Algorithm

**Exponential vertical injection** (Mandel et al., 2011; mirrors Fire module):

For each ATM column at (i, j):
1. Define surface height: `z_sfc = z_phys_cc(i, j, klo)`
2. For each level k, compute height-above-surface: `z_k = z_phys_cc(i, j, k) - z_sfc`
3. Compute sensible heat tendency:
   ```
   hfx_k = (H_sensible(i, j) / Cp_d) * exp(-z_k / alpha_ucm)
   theta_tend(k) = -rho(k) * (hfx_{k+1} - hfx_k) / dz(k)
   cc_source(i, j, k, RhoTheta_comp) += atm_feedback * theta_tend(k)
   ```
4. If moisture present and `LE_latent` available:
   ```
   le_tendency(k) = -rho(k) * (le_{k+1} - le_k) / dz(k)
   cc_source(i, j, k, RhoQ1_comp) += atm_feedback * le_tendency(k) / L_v
   ```

**Coarsening:** When `grid_ratio > 1`, use `amrex::average_down(src_ucm, dst_atm, 0, 1, IntVect(grid_ratio, grid_ratio, 1))` to downsample fluxes from UCM to ATM grid, preserving area-weighting.

### Phase 1.4 Parameters (ParmParse)

```
erf.ucm.alpha_ucm          [Real] = 15.0 m     # E-folding depth for exponential injection
erf.ucm.atm_feedback       [Real] = 0.0        # Coupling strength: 0=one-way (no feedback), 1=full
erf.ucm.ucm_plot_int       [int]  = 0          # Plotfile interval; 0 = disabled
erf.ucm.ucm_diag_file      [str]  = "ucm_diag.dat"  # CSV diagnostics filename
erf.ucm.sum_interval       [int]  = 1          # Diagnostics write interval (steps)
```

### Phase 1.4 Debug Tracing (MANDATORY)

Format: `[UCM][1.4][<function>] <values>`. Emitted once per call on rank 0:

- **`coarsen_ucm_flux_to_atm`** — min/max flux before/after coarsening; grid_ratio used
- **`apply_ucm_tendency_to_cc_source`** — min/max RhoTheta tendency; k=0 rho and dz; alpha_ucm; atm_feedback; expected surface magnitude
- **`UCMPlotfile::write`** — output filename, step, sim time, component list
- **`UCMDiagnostics::append`** — one-liner CSV row just appended
- **`UCMLayer::advance`** — Phase 1.3 trace + Phase 1.4 "post-injection" line showing whether injection ran

### Phase 1.4 Requirements & Assumptions

- **Still homogeneous:** Phase 1.4 uses homogeneous URBPARM (no CSV). Heterogeneous morphology is Phase 2.1.
- **Radiation coupling deferred:** Analytic diurnal SW/LW is Phase 4.2. Phase 1.4 uses dummy `albedo` and `emissivity` parameters.
- **MRF stability adjustments deferred:** Neutral log-law for exchange coefficient from Phase 1.3. MRF stability-aware tuning is Phase 3.3.
- **One-way only:** Even with `atm_feedback = 1.0`, the ATM state is NOT fed back into UCM (true two-way is Phase 3.2).
- **Latent heat placeholder:** `LE_latent = 0` still in Phase 1.4. Anthropogenic heat injection is Phase 2.3; plant transpiration is Phase 5.3.
- **Backward regression:** With `atm_feedback = 0.0`, must produce identical ATM state as `enable = false` (bit-for-bit).

### Phase 1.4 Build Integration

- Add `ERF_UCMAtmCoupling.cpp`, `ERF_UCMPlotfile.cpp`, `ERF_UCMDiagnostics.cpp` to `Make.package` and `CMake/BuildERFExe.cmake`
- Add `.H` files to `CEXE_headers`
- Modified: `Source/TimeIntegration/ERF_Advance.cpp` — coarsen and inject after SEB, before dycore advance
- Modified: `Source/ERF.H`, `Source/ERF_Constructors.cpp` — Phase 1.4 member vectors

### Acceptance Criteria (Phase 1.4)

See main problem statement for full checklist. Key items:

1. Builds with `-DERF_ENABLE_UCM=ON` and `=OFF`
2. Phase 1.1, 1.2, 1.3 tests still pass (no regression)
3. Phase 1.4 canonical test exits 0, produces plotfiles + CSV
4. Injection-effect check: `atm_feedback=1.0` vs `0.0` ATM state differs >0.5 K at daytime steps
5. Backward regression: `atm_feedback=0.0` matches `enable=false` bit-for-bit
6. All acceptance criteria from problem statement met (20 items total)

---

---

## Phase 2.1: Building-Layout CSV Reader + Material Library CSV

### Phase 2.1 Status

✅ **COMPLETE** — Merged via PRs #203, #204, #205.

**Phase 2.1 Deliverables (from merged PRs):**

1. **`ERF_UCMBuildingReader.H/cpp`** — CSV parsing for building layout
   - Reads `building_layout.csv` with columns: `i, j, height_m, W_road_m, W_roof_m, is_urban, roof_mat_id, wall_mat_id, road_mat_id`
   - Grid-ratio-aware cell mapping (CSV in UCM grid coords)
   - Per-cell `H_bldg`, `W_road`, `W_roof`, `is_urban` mask, material IDs populated

2. **`ERF_UCMMaterialReader.H/cpp`** — CSV parsing for material properties
   - Reads `materials.csv` with columns: `mat_id, albedo, emissivity, k_therm_W_per_mK, rho_cp_J_per_m3K, thickness_m`
   - Populates in-memory `UCMMaterialRegistry` for O(1) lookup by `mat_id`
   - Per-material `UCMMaterial` struct: thermal and optical properties

3. **Modified `fill_ucm_fields_from_csv`** in `ERF_UCMAllocate.cpp`
   - Calls building reader to populate UCM morphology MultiFabs
   - Calls material reader to populate registry
   - Phase 2.1 Bug #12 fixed: `is_urban == 1` guard around all registry lookups

4. **Canonical test:** `Exec/CanonicalTests/SLUCM/UCMHeterogeneousBlock/`
   - 16×16 uniform morphology via CSV (not hardcoded)
   - Verifies CSV parsing + registry lookup work end-to-end

---

## Phase 2.2: Per-Cell Material + Morphology Wiring into SEB

### Phase 2.2 Status

🟢 **IN PROGRESS** — Tasks 1–13 implementation in active development. Goal: wire per-cell material properties (`albedo`, `emissivity`, `k_therm`, `rho_cp`, `slab_L`) and aerodynamic properties (`z0`, `d_disp`) through to SEB and wind extraction kernels.

### Phase 2.2 Deliverables

**Task 1: Add 11 new MultiFabs to UCMFields**
- `k_therm_roof`, `k_therm_wall`, `k_therm_road` — Thermal conductivity [W/m/K] per material
- `rho_cp_roof`, `rho_cp_wall`, `rho_cp_road` — Heat capacity density [J/m³/K]
- `slab_L_roof`, `slab_L_wall`, `slab_L_road` — Material thickness [m]
- `z0_ucm`, `d_disp_ucm` — Aerodynamic roughness and displacement height [m]

**Task 2: Allocate new MultiFabs**
- All 11 fields allocated in `allocate_ucm_fields` with ghost = `IntVect(1,1,0)`

**Task 3: Fill defaults in homogeneous path**
- Homogeneous initialization via `fill_ucm_fields_homogeneous` reads uniform params and broadcasts to all cells

**Task 4: Fill from CSV**
- CSV pathway: `fill_ucm_fields_from_csv` loops per-cell, looks up material via registry, assigns per-cell property
- Phase 2.1 Bug #12 rule enforced: `is_urban == 1` guard around all registry accesses

**Task 5: New function `fill_ucm_z0_and_disp`**
- Computes per-cell `z0 = z0_over_H * H_bldg` and `d_disp = d_over_H * H_bldg`
- One-time initialization (GPU-friendly ParallelFor)
- Safety defaults for non-urban cells

**Task 6: Add ParmParse parameters**
- `z0_over_H` (default 0.1, MacDonald 1998)
- `d_over_H` (default 0.7, WRF convention)
- Both printed in startup banner

**Task 7: Extend wind extraction for per-cell z0/d**
- Modified `fill_ucm_wind_from_interpolation` signature to accept `z0_ucm` and `d_disp_ucm`
- Log-law kernel rewired: `ln((z - d) / z0)` instead of hardcoded `z0=0.1, d=0`

**Task 8: Rewire SEB kernel for per-cell material properties**
- `ERF_UCMLayer.cpp::advance` SEB section reads per-cell `k_therm`, `rho_cp`, `slab_L` arrays
- Replaces scalar `params.k_therm_uniform`, etc. with `k_therm_roof_arr(i,j,0)`, etc.
- Physics unchanged; pure wiring

**Task 9: Call `fill_ucm_z0_and_disp` from `ERF.cpp`**
- Immediately after homogeneous or CSV field fill
- Ensures z0/d computed before any wind extraction

**Task 10: Add Phase 2.2 banner**
- Static one-print verification at top of `UCMLayer::advance`
- Displays min/max of H_bldg, albedo_roof, k_therm_roof, z0, d_disp
- Defense against silent regressions in heterogeneous CSV pipeline

**Task 11: Fix all existing test inputs**
- Removed `erf.fixed_dt` (deprecated timestepping)
- Added `erf.cfl = 0.5` (adaptive dt)
- Adjusted `max_step` for CI compatibility

**Task 12: Create `UCMHeterogeneousMorphology` test**
- 16×16 domain with two-region morphology (5 m vs. 25 m buildings)
- Expected z0 range: 0.5 m – 2.5 m
- Expected d_disp range: 3.5 m – 17.5 m
- Pass: BANNER shows heterogeneous ranges (if collapsed → regression)

**Task 13: Create `UCMHeterogeneousMaterials` test**
- 16×16 domain with uniform 15 m morphology
- Checkerboard material pattern: cool roof (albedo=0.6, k=0.2) vs. dark roof (albedo=0.1, k=2.0)
- Pass: BANNER shows uniform H_bldg but heterogeneous albedo

### Phase 2.2 Parameters (ParmParse)

```
erf.ucm.z0_over_H          [Real] = 0.1     # z0 = z0_over_H * H_bldg
erf.ucm.d_over_H           [Real] = 0.7     # d_disp = d_over_H * H_bldg
```

### Phase 2.2 Key Design Rules

- **R5 (is_urban guard):** Every UCM ParallelFor kernel must start `if (is_urb_a(i,j,0) == 0) return;`
- **R6 (registry safety):** Every `registry.lookup(mat_id)` is inside `if (is_urban == 1) { ... }`
- **R2 (Array4 access):** Use `mf.array(mfi)` (write), never `mf[mfi].array()`
- **R4 (collectives):** `mf.min(0)`, `.max(0)` OUTSIDE `IOProcessor()` blocks

### Phase 2.2 Code Paths

**Homogeneous:**
1. `ERF.cpp` calls `fill_ucm_fields_homogeneous` → broadcasts scalar params to all cells
2. `ERF.cpp` calls `fill_ucm_z0_and_disp` → computes z0, d from H_bldg and params
3. Wind extraction uses per-cell z0, d (even though uniform in this path)

**Heterogeneous (CSV):**
1. `ERF.cpp` calls `fill_ucm_fields_from_csv` → reads CSV, populates per-cell morphology and material registry
2. CSV fill loop guards all registry lookups with `is_urban == 1`
3. `ERF.cpp` calls `fill_ucm_z0_and_disp` → computes per-cell z0, d from heterogeneous H_bldg
4. Wind extraction uses per-cell z0, d (truly heterogeneous)

### Phase 2.2 Test Suite

**Modified existing tests** (Phase 1.1–1.4 regression):
- `UCMScaffold`, `UCMHomogeneousGrid`, `UCMHomogeneousSEB`, `UCMOneWayInject`, `UCMHomogeneousViaCSV`
- All now use `erf.cfl = 0.5` instead of deprecated `erf.fixed_dt`

**New canonical tests** (Phase 2.2 validation):
- `UCMHeterogeneousMorphology` — Two building heights, two material IDs
- `UCMHeterogeneousMaterials` — One building height, checkerboard materials

### Phase 2.2 Acceptance Criteria

1. ✅ Builds with `-DERF_ENABLE_UCM=ON` and `=OFF`
2. ✅ All prior tests (1.1–1.4) still exit 0 after cfl conversion
3. ✅ `UCMHeterogeneousMorphology` exits 0; BANNER shows `H_bldg min=5 max=25`, `z0 min=0.5 max=2.5`, `d_disp min=3.5 max=17.5`
4. ✅ `UCMHeterogeneousMaterials` exits 0; BANNER shows uniform `H_bldg` but `albedo_roof min=0.1 max=0.6`
5. ✅ No `erf.fixed_dt` in any test input
6. ✅ No `amr.max_step` (all converted to `max_step`)
7. ✅ No `get_theta_star` in UCM code
8. ✅ Every UCM ParallelFor guarded by `is_urban` check
9. ✅ Every registry lookup inside `is_urban == 1` block
10. ✅ All collectives (min/max) OUTSIDE `IOProcessor()` guards

### Phase 2.2 Expected Merge Date

End of Phase 2.2 milestone (date TBD). Incremental PR strategy:
- **Phase 2.2a** — Tasks 1–7 (multifabs, allocation, wind extraction)
- **Phase 2.2b** — Tasks 8–10 (SEB rewiring, banner, initialization)
- **Phase 2.2c** — Tasks 11–13 (test suite + validation)

---

## Phase 2.3: Facet-Split Sensible Heat + Anthropogenic Heat + CSV Convention Lock-In

### Phase 2.3 Status

🟢 **IN PROGRESS** — All 14 tasks complete; implementation ready for testing and review.

**Deliverables:**
1. Replace single lumped `H_sensible` with three per-facet MultiFabs: `H_road`, `H_wall`, `H_roof`
2. Add anthropogenic heat MultiFab `AH` with uniform and diurnal profiles
3. Keep `H_sensible = H_road + H_wall + H_roof + AH` for injection backward compatibility
4. Enforce CSV row-count validation and `i,j` index convention lock-in
5. Add two canonical tests: `UCMFacetSplit` (baseline) and `UCMAnthroHeat` (with AH)

### Phase 2.3 Key Implementation Details

**Facet-Split Physics (Task 8):**
- Area fractions: `f_road = 1 - plan_area_frac`, `f_roof = plan_area_frac`, `f_wall = 2*plan_area_frac*H/(W_road+W_roof)`
- Each facet: `H_facet = f_facet * H_base` where `H_base = -ρ*Cp*u*t` from MOST
- Per-cell `plan_area_frac` read from CSV (Phase 2.1 already present)
- Non-urban cells set all fluxes to zero (Rule R5)

**Anthropogenic Heat (Task 7):**
- Two profiles selectable per-cell via `ah_profile_id` iMultiFab (from CSV)
  - `profile_id == 0`: Uniform `AH_uniform_Wm2` [W/m²]
  - `profile_id == 1`: Diurnal cosine: `AH_daytime_peak * max(0, cos(phase))` where phase varies over 86400 s day
- AH added to roof facet (rooftop HVAC convention; future Phase 6.2 will move to BEM)
- Non-urban cells: `AH = 0` (no AH computation)

**CSV Convention Lock-In (Task 6):**
- **Rule R14:** `i,j` in `building_layout.csv` are **UCM cell indices**, not ATM indices
- Valid range: `i ∈ [0, nx_ucm)`, `j ∈ [0, ny_ucm)` where `nx_ucm = n_cell[0] * grid_ratio`
- Total rows must equal `nx_ucm * ny_ucm`
- Reader enforces with `amrex::Abort` if violated; error message includes expected vs. actual count

**Backward Compatibility:**
- Injection still reads `H_sensible` (ERF_UCMAtmCoupling.cpp unchanged)
- Since `H_sensible = H_road + H_wall + H_roof + AH`, ATM receives same total heat
- Phase 1.4 injection kernel untouched → bit-for-bit ATM regression preserved

### Phase 2.3 ParmParse Parameters

```
erf.ucm.plan_area_frac_uniform    [Real] = 0.5   # Plan-area fraction for homogeneous default [-]
erf.ucm.AH_uniform_Wm2            [Real] = 0.0   # Uniform anthropogenic heat [W/m²]
erf.ucm.AH_daytime_peak           [Real] = 20.0  # Peak of diurnal AH [W/m²]
erf.ucm.AH_profile_type_default   [int]  = 0     # 0=uniform, 1=diurnal cosine
```

### Phase 2.3 Test Suite

**Modified existing tests** (Phase 1.1–2.2 regression):
- All 7 existing tests verified to have correct row-count CSVs and UCM index convention
- No breaking changes to existing test structure; only validation adds error guards

**New canonical tests** (Phase 2.3 validation):
- `UCMFacetSplit` — Baseline facet split without AH. Pass: facet fluxes visible in BANNER, `H_road + H_wall + H_roof ≈ H_sensible` in diagnostics
- `UCMAnthroHeat` — Same setup with `AH_uniform_Wm2=30.0`. Pass: `H_roof_max` ~30 W/m² higher than baseline, ATM RhoTheta measurably warmer

### Phase 2.3 Acceptance Criteria

1. ✅ Builds with `-DERF_ENABLE_UCM=ON` and `=OFF`
2. ✅ All prior tests (Phases 1.1–2.2) still exit 0 after Phase 2.3 integration
3. ✅ `UCMFacetSplit` exits 0; BANNER shows `H_road`, `H_wall`, `H_roof` with distinct ranges, `AH_min=0 max=0`
4. ✅ `UCMAnthroHeat` exits 0; BANNER shows `AH_min=30 max=30`, `H_roof_max` ~30 higher than baseline
5. ✅ Facet sum invariant: `H_road_max + H_wall_max + H_roof_max ≈ H_sensible_max` in diagnostics CSV (within round-off)
6. ✅ Reader aborts with clear message if CSV row count ≠ `nx_ucm * ny_ucm`
7. ✅ Every UCM ParallelFor guarded by `is_urban` check (unchanged from Phase 2.2)
8. ✅ `UCM_DEVELOPMENT.md` updated with Phase 2.3 status and CSV convention lock-in note
9. ✅ No regressions in Phase 2.2 BANNER output (extended with Phase 2.3 diagnostics)
10. ✅ ATM regression preserved: final `Rho`, `RhoTheta`, `U`, `V`, `W` match Phase 2.2 (same injection kernel)

---

## Phase 2.3 Bug Fix: RK-Stage Injection Contract

### Problem

In the SLUCM UCMAnthroHeat canonical test, the global `RHO THETA` barely moved
per step (~6.5e4 units vs. the expected ~6e5).  Root cause: the injection call in
`ERF::Advance` was writing into `rhotheta_src[lev]` (a 1-component `MultiFab`),
but `apply_ucm_tendency_to_cc_source` accessed component `RhoTheta_comp = 1` on
that 1-component buffer → out-of-bounds write (silent UB / zeros on GPU).
Additionally the `custom_rhotheta_forcing` path double-counted with the per-stage
injection in `ERF_TI_slow_rhs_pre.H`.

### Fix (this commit)

**`Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp`**
- `apply_ucm_tendency_to_cc_source` now has **overwrite semantics**: it zeroes
  `cc_source[RhoTheta_comp]` (and `cc_source[RhoQ1_comp]` when moisture is on)
  at the top of each call, then writes with `=` (not `+=`).  This is safe because
  `make_sources()` already zeroed `cc_src` before the per-stage injection runs.
- Diagnostic block upgraded: uses `MultiFab::min/max/sum` on the UCM-owned
  component (which equals the UCM tendency only, thanks to overwrite semantics)
  and reports `sum` and cell count, matching the expected debug output.
- Guards indexing `h_a(i, j, klo)` / `urban_a(i, j, klo)` instead of `(i,j,0)`
  (portability fix for non-zero `dom_lo[2]`).

**`Source/TimeIntegration/ERF_Advance.cpp`**
- Removed the redundant `apply_ucm_tendency_to_cc_source(*rhotheta_src[lev], ...)`
  call from the Phase 1.4 block. The only injection path is now per-RK-stage in
  `ERF_TI_slow_rhs_pre.H`.
- Removed `solverChoice.custom_rhotheta_forcing = true` block: UCM does NOT use
  the `rhotheta_src` / `custom_rhotheta_forcing` path.
- `UCMLayer::advance` (SEB) still runs **once per coarse step** here; a new
  `[UCM][step] SEB advanced` print gated on `ucm_debug` confirms this.

**`Source/TimeIntegration/ERF_TI_slow_rhs_pre.H`**
- Per-stage `apply_ucm_tendency_to_cc_source(cc_src, ...)` call now passes
  `m_ucm_params.ucm_debug` (was hardcoded `false`), so the diagnostic banner
  appears once per "Making slow rhs" line — three times per RK3 coarse step —
  with identical `sum` across stages (confirming the lagged-flux contract).

### RK-stage safety contract (recorded here for future maintainers)

`apply_ucm_tendency_to_cc_source` **OWNS** `cc_source[RhoTheta_comp]` (and
optionally `cc_source[RhoQ1_comp]`) for the duration of each RK stage:

1. `UCMLayer::advance` runs **once** per coarse step in `ERF::Advance`.
2. `coarsen_ucm_flux_to_atm` runs **once** per coarse step, caching the lagged
   flux in `m_ucm_H_atm[lev]` / `m_ucm_LE_atm[lev]`.
3. `apply_ucm_tendency_to_cc_source` is called **once per RK stage** from
   `slow_rhs_fun_pre`, after `make_sources()` resets `cc_src` to zero.
4. The function zeros its owned components first and writes with `=`, so the
   stage result is always the UCM-only tendency regardless of prior state.

If any other physics writes into `cc_source[RhoTheta_comp]` per stage on the
same cells, the UCM injection must be moved to a pre-integrator path and
semantics must change back to `+=`.

This mirrors the ERF-Fire explicit-lag convention documented in
`Source/Fire/ERF_FireAtmCoupling.H` on the ERF-Hazard branch.

### Debug comment improvements (same commit)

- `ERF_UCMLayer.cpp`: per-step ATM forcing print now gated on `ucm_debug`
  (was unconditional — printed every timestep).
- `ERF_UCMLayer.cpp`: expensive per-step `min/max` collectives moved inside
  the `ucm_debug` guard (was always computed even with debug off).
- `ERF_UCMLayer.cpp`: per-step debug trace extended with `H_road`, `H_wall`,
  `H_roof`, `AH` min/max.
- `ERF_UCMLayer.cpp`: extraction debug trace added (u*, wind, T_ref stats)
  gated on `ucm_debug`.
- `ERF_UCMDiagnostics.cpp`: verbose CSV append trace now gated on `ucm_debug`
  (was unconditional inside `IOProcessor()` block).
- `ERF_UCMGrid.cpp` / `ERF_UCMGrid.H`: `create_ucm_grid` gains `bool ucm_debug`
  parameter (default `false`); prints gated on it.
- `ERF_UCMPrerequisites.cpp`: Phase 1.2 grid check banner gated on `ucm_debug`.
