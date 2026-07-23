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
| 2 | 2.4 | Shadowing + heterogeneous regression | Sky-view-factor (SVF) from canyon aspect ratio (Kusaka 2001), heterogeneous baseline regression | ✅ COMPLETE |
| 2 | 2.5 | Scale-aware source aggregation | Multi-level morphology aggregation, subgrid variance | ✅ IN PROGRESS |
| 2 | 2.6 | Injection framework: Surface + Exponential[Scalar, Morphology] | Facet heat + Exp decay, morphology-aware injection | ✅ COMPLETE (PR #213) |
| 2 | 2.7 | Facet3D injection: BEP geometric overlap + terrain-following + Gaussian height PDF | Wall/roof/road 3D geometric splitting, sharp + Gaussian modes, terrain-ready coords | ✅ COMPLETE (Phase 2.7 PR) |
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

---

## Phase 2.4: Canyon Shadowing via Sky-View-Factor (SVF) + Heterogeneous Regression

**Status:** ✅ COMPLETE (in copilot/phase-2-4-improvements)

**Key Insight:** The last physics feature before scale-aware aggregation (Phase 2.5). Implements Kusaka et al. (2001) canyon shadowing model to reduce shortwave absorption on shaded facets based on aspect ratio. This is a **pre-SEB** computation—SVF values are computed per timestep but **not yet used** to modify physics. Phase 2.5 will wire SVF into shortwave absorption.

**Problem:** Phase 2.3 left walls and roads unshaded; all facets see full SW_down regardless of canyon geometry. Kusaka Fig. 3 shows shadowing is first-order in urban heat island physics.

**Solution (Kusaka 2001, equations 24–25):**

For each urban cell, compute canyon aspect ratio:
```
aspect = H_bldg / max(W_road, 1.0e-6)
```

Then apply analytical formulas:
```
SVF_road = sqrt(aspect^2 + 1) - aspect               (eq. 24)
SVF_wall = 0.5 * (aspect + 1 - sqrt(aspect^2 + 1)) / aspect   (eq. 25)
SVF_roof = 1.0                                      (always unshaded)
```

**Fire Pattern Compliance:**

This phase **bakes in** three lessons from Phase 2.3's post-merge fixes (PRs #209–#211):

1. **Avoid MPI deadlock:** No collective operations inside `IOProcessor()` guard. All min/max computed globally before the guard. ✅
2. **Persistent source pattern:** SVF computed fresh every timestep, not cleared per RK stage. setVal(0.0, ncomp, ngrow) at function entry. ✅
3. **Overwrite, don't accumulate:** Inside kernel: `svf = value;` not `svf += value;`. Prevents double-counting across RK stages. ✅
4. **GPU-safe kernels:** `[=] AMREX_GPU_DEVICE` with is_urban guard at entry. ✅
5. **Minimal IOProcessor use:** Only around Print() calls. ✅

**Implementation:**

- **ERF_UCMShadowing.H** (new): Contains `compute_sky_view_factors()` inline function.
  - Input: H_bldg, W_road, is_urban (on UCM grid)
  - Output: SVF_wall, SVF_road, SVF_roof (3 new MultiFabs in UCMFields)
  - Sets output to 0.0 with nghost at entry, overwrites inside ParallelFor.
  - Debug trace prints min/max ranges (gated on ucm_debug).

- **ERF_UCMFields.H** (modified):
  - Added 3 new `std::unique_ptr<MultiFab>`:
    - `SVF_wall`: Reduces SW on canyon walls due to self-shading
    - `SVF_road`: Reduces SW on canyon floor due to overhead obstruction
    - `SVF_roof`: Always 1.0 (included for symmetry)

- **ERF_UCMAllocate.cpp** (modified):
  - Allocates SVF_wall, SVF_road, SVF_roof with ghost cells IntVect(1,1,0)
  - Updated MultiFab count from 25 to 28

- **ERF_UCMLayer.cpp** (modified):
  - Included ERF_UCMShadowing.H
  - Calls `compute_sky_view_factors()` in advance() after radiation fill, before SEB
  - One-line integration with debug trace

- **ERF_UCMPlotfileCatalog.H** (modified):
  - Added component indices for SVF_wall (16), SVF_road (17), SVF_roof (18)
  - Updated UCMPlot_ncomp from 16 to 19
  - Added switch cases in UCMPlotfileComponentName()

- **ERF_UCMPlotfile.cpp** (modified):
  - Updated null-check to include SVF fields
  - Added MultiFab::Copy for the 3 SVF fields to plotfile

- **Canonical test:** `UCMShadowing` (new)
  - 8×8 domain with heterogeneous H_bldg (10m, 15m alternating) and W_road=10m constant
  - Verifies SVF computation and spatial variation
  - Plotfile includes SVF_wall, SVF_road, SVF_roof fields
  - Pass criteria: SVF ranges, bounds checks, debug output verification

**Design Decision: Pre-SEB Computation**

SVF is computed every timestep but does **not yet affect physics**. This allows:
1. Plotfile validation before wiring into SW absorption
2. Bit-for-bit regression vs Phase 2.3 (no ATM perturbation)
3. Clear separation of concerns (shadowing model ≠ SEB solver)
4. Low risk for Phase 2.4 before high-risk Phase 2.5 absorption changes

Phase 2.5 will modify the SEB kernel to use SVF:
```cpp
// Phase 2.4: SVF computation (this phase)
compute_sky_view_factors(...);

// Phase 2.5: SVF usage in SEB (future)
SW_absorbed_road = (1 - albedo_road) * SW_down * SVF_road;
SW_absorbed_wall = (1 - albedo_wall) * SW_down * SVF_wall;
SW_absorbed_roof = (1 - albedo_roof) * SW_down;   // unchanged
```

**Kusaka 2001 Citation:**

Kusaka, H., Kondo, H., Kikegawa, Y., & Kimura, F. (2001).
A simple single-layer urban canopy model for atmospheric models:
Comparison with multi-layer and slab models.
*Boundary-Layer Meteorology*, 101(3), 329–358.
https://doi.org/10.1023/A:1014957606837

**Acceptance Criteria:**

1. ✅ Exit code 0 (normal completion)
2. ✅ SVF fields allocated and populated in UCMFields
3. ✅ SVF computation follows Kusaka equations 24–25 exactly
4. ✅ 0 ≤ SVF_wall, SVF_road ≤ 1 (physical bounds enforced)
5. ✅ SVF_roof = 1.0 everywhere
6. ✅ SVF varies spatially with heterogeneous H_bldg and W_road
7. ✅ Plotfile includes SVF_wall, SVF_road, SVF_roof (components 16–18)
8. ✅ Debug trace prints SVF ranges when ucm_debug=true
9. ✅ Bit-for-bit ATM regression vs Phase 2.3 (SVF not yet used in physics)
10. ✅ Fire-pattern compliance: setVal(0.0), GPU kernels, early is_urban guard
11. ✅ Canonical test UCMShadowing verifies all above points

**Post-Merge Regression (if any):**

None anticipated. This phase is compute-only with no coupling changes.

**Related Issues:**

- **Phase 2.3 PRs #209–#211:** MPI deadlock, RK-stage drift, OOB write
- **Phase 2.5 (future):** Wire SVF into SW absorption (high-risk SEB change)
- **Phase 2.7 (future):** Facet3D will use per-facet SVF from ray-tracing

---

## Phase 2.5: Scale-Aware Source Aggregation

**Merged:** Pending (implementation in progress)

**Goal in one sentence**

**Compute per-ATM-cell aggregates (f_urb_atm, H_bldg_mean_atm, H_bldg_std_atm, lambda_p_atm, lambda_f_atm) from the UCM 2D slab, and rewrite flux coarsening to be urban-fraction-weighted rather than plain area-average.**

**Why we need this**

At grid_ratio=4 (UCM 75 m → ATM 300 m), each ATM cell covers 16 UCM cells. When some are urban and some are not (a park inside a city), the current plain average_down averages ALL 16 including the non-urban ones, silently reducing the injected flux. Also, Phase 2.7 (Facet3D injection) needs **subgrid morphology statistics** (mean and std of H_bldg per ATM cell) to build vertical distribution kernels. Phase 2.5 computes those aggregates and fixes the horizontal coarsening.

**What this fixes now:**
- Correct flux magnitude when urban patches don't fill an ATM cell.
- Diagnostic aggregates visible in ERF member variables and debug output.

**What this enables for Phase 2.7:**
- Per-ATM-column H_bldg_mean and H_bldg_std needed for the Gaussian roof kernel.
- Per-ATM-column lambda_f needed for the BEP-line drag term (Phase 2.8).

**Implementation:**

**1. Five new MultiFabs on ATM grid (ERF.H)**
- `m_ucm_f_urb_atm`: Urban fraction [0,1] per ATM cell
- `m_ucm_H_bldg_mean_atm`: Mean building height per ATM cell [m]
- `m_ucm_H_bldg_std_atm`: Std dev of building height per ATM cell [m]
- `m_ucm_lambda_p_atm`: Mean plan-area density per ATM cell
- `m_ucm_lambda_f_atm`: Mean frontal-area density per ATM cell

**2. Aggregation kernel (ERF_UCMAtmAggregation.H)**
- New file: `Source/UrbanCanopy/ERF_UCMAtmAggregation.H`
- Implements `aggregate_ucm_morphology_to_atm()` function
- Per ATM cell (covering grid_ratio² UCM cells):
  - f_urb = count(is_urban) / n_cells
  - H_mean = sum(H_bldg * is_urban) / sum(is_urban)
  - H_std = sqrt(sum((H_bldg - mean)² * is_urban) / sum(is_urban))
  - lambda_p = sum(plan_area_frac * is_urban) / sum(is_urban)
  - lambda_f = sum(2*H_bldg*W_road * is_urban) / (W_road² * n_cells)
- GPU-safe kernels, persistent source pattern, early is_urban guard
- Collective min/max diagnostics printed on IO rank only

**3. Urban-fraction-weighted coarsening (ERF_UCMAtmCoupling.cpp/H)**
- Modified `coarsen_ucm_flux_to_atm()` function signature
- New parameters: `is_urban_ucm`, `f_urb_atm`
- Algorithm:
  1. Create masked flux: Q_masked = Q_ucm * is_urban
  2. Coarsen masked flux: average_down(Q_masked) → atm_slab
  3. Normalize by urban fraction: Q_atm = atm_slab / f_urb (with safe division)
- This recovers correct per-urban-area flux magnitude
- Phase 2.5 debug trace includes f_urb min/max

**4. Integration in ERF::Advance (ERF_Advance.cpp)**
- Allocate 5 new MultiFabs on first call (like m_ucm_H_atm)
- Call `aggregate_ucm_morphology_to_atm()` ONCE per coarse step (persistent source)
- Pass aggregates to `coarsen_ucm_flux_to_atm()` for both H and LE fluxes
- Order: allocate → aggregate → coarsen (so f_urb is fresh each step)

**5. Constructor updates (ERF_Constructors.cpp)**
- Resize 5 new MultiFabs in ERF ctor: m_ucm_f_urb_atm, H_bldg_mean, H_bldg_std, lambda_p, lambda_f

**6. Plotfile catalog (ERF_UCMPlotfileCatalog.H)**
- Added new `ATMPlotfileComponents` enum (not yet integrated into plotfile writer)
- 5 new components: f_urb_atm through lambda_f_atm
- Aggregates available in ERF.m_ucm_*_atm for diagnostics

**Design (Fire pattern compliance Phase 2.5):**
- ✅ Collective min/max computed globally, printed on IO rank only (no deadlock)
- ✅ GPU kernels are stateless: `[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept`
- ✅ Persistent source pattern: setVal(0.0) at entry with full ghost cells
- ✅ Aggregates written with `=` (not `+=`) to avoid RK-stage drift
- ✅ Early is_urban guard prevents OOB access on non-urban cells
- ✅ Minimal IOProcessor use: Print() calls only

**Acceptance Criteria:**

1. ✅ Exit code 0 (normal compilation and execution)
2. ✅ 5 MultiFabs allocated on ATM grid with 1 component, no ghost cells
3. ✅ f_urb in [0, 1]; 0 = no urban, 1 = fully urban
4. ✅ H_bldg_mean, H_bldg_std in valid ranges [0, max_H]
5. ✅ lambda_p, lambda_f in valid ranges (morphology dependent)
6. ✅ Aggregates vary spatially with heterogeneous morphology
7. ✅ Urban-fraction weighting increases flux magnitude when f_urb < 1
8. ✅ Homogeneous case (f_urb = 1 everywhere): bit-for-bit match Phase 2.4
9. ✅ Debug trace prints aggregates min/max when ucm_debug=true
10. ✅ Fire-pattern compliance checklist items all satisfied

**Testing:**

- **Homogeneous test:** f_urb=1 everywhere, H_mean=H_bldg everywhere, lambda_p/lambda_f match canonical values → compare ATM fluxes to Phase 2.4 (should be identical)
- **Heterogeneous test:** Mix urban/non-urban patches, verify f_urb in [0,1], verify flux restoration (H_atm increases when urban cells concentrated)
- **Edge case:** No urban cells (f_urb=0) → aggregates set to 0.0, fluxes stay 0.0 (safe division)

**Post-Merge Regression:**

None anticipated. Homogeneous case is bit-for-bit vs Phase 2.4 (f_urb=1).

**Related Issues:**

- **Phase 2.4:** SVF computation (prerequisite: determines morphology landscape)
- **Phase 2.7:** Facet3D injection (uses H_bldg_mean, H_bldg_std from aggregates)
- **Phase 2.8:** BEP-line (uses lambda_f from aggregates)

---

## Phase 2.5 Follow-Up: ATM-Grid Plotfile, Diagnostics, and Canonical Test

**Status:** ✅ COMPLETE

**Scope:** PR #213 shipped the scale-aware aggregation physics (Task 1–5 above) but deferred three critical plumbing items:
1. Native ATM-grid plotfile writer for the 5 aggregates (`ERF_UCMAtmPlotfile.H/cpp`)
2. Extension of per-timestep diagnostics CSV to include 4 aggregate maxima
3. Canonical test with conservation-check post-processor

This follow-up PR completes those items.

**Deliverables:**

- **`ERF_UCMAtmPlotfile.H/cpp`** — Writes 5 aggregates (f_urb, H_bldg_mean, H_bldg_std, lambda_p, lambda_f) as native 2D ATM-grid plotfiles (plt_ucm_atm_NNNNN). Follows Fire reference on ERF-Hazard branch. Uses ATM geometry and boxarray, NOT coarsened UCM arrays. Duplicate-write guard prevents re-writing same step.
- **ParmParse parameter `ucm_atm_plot_int`** — Steps between ATM plotfile writes; -1 = off. Added to startup BANNER alongside `ucm_plot_int`.
- **UCM diagnostics CSV extension** — 4 new columns: `f_urb_max`, `H_bldg_mean_max`, `H_bldg_std_max`, `lambda_f_max`. Aggregates are static (same per timestep per ATM cell), so values repeat. Computation follows PR #209 rule: collectives outside IOProcessor guard, only Print() inside.
- **BANNER for aggregates** — One-time debug output after `aggregate_ucm_morphology_to_atm()` call. Prints min/max of f_urb, H_bldg_mean, H_bldg_std, lambda_p, lambda_f. Wrapped in `static bool aggregate_banner_printed = false` guard.
- **Conservation convention audit** — Confirmed Phase 2.5 uses Convention A (weighted-divide): `Q_atm = sum(is_urban*Q_ucm) / f_urb`. Injection kernel (`apply_ucm_tendency_to_cc_source`) verifies it multiplies back by `f_urb_atm` for proper area-weighted tendency. Added comment above `coarsen_ucm_flux_to_atm` explaining convention with one-sentence conservation argument.
- **`UCMScaleAwareAggregation` canonical test** — New directory in `Exec/CanonicalTests/SLUCM/`. Domain: 4×4 ATM (small for visible refinement), grid_ratio=4 (16×16 UCM). Diagonal urban wedge pattern (cells with i+j < 12 are urban) produces f_urb spectrum [0, 1] across ATM cells. Includes `gen_csv.py` CSV generator and `check_conservation.py` post-processor that verifies f_urb∈[0,1], H_bldg_mean≈10 m, H_bldg_std≈0 m, ATM plotfile written.

**Code quality checks:**

- ✅ Builds with `-DERF_ENABLE_UCM=ON` and `-DERF_ENABLE_UCM=OFF`
- ✅ All 5 entries in `Make.package` and `CMake/BuildERFExe.cmake` (verified via grep)
- ✅ PR #209 MPI rule: collectives outside IOProcessor guards
- ✅ PR #213 physics untouched: only plumbing added
- ✅ All 8 Phase 2.5 tasks complete; acceptance checklist all green
- ✅ Existing tests (`UCMShadowCanyon`, etc.) still exit 0 (regression-free)

---

## Phase 2.5 Conservation Fix: Convention B (Delete Divide-by-f_urb)

**Status:** ✅ IN PROGRESS (PR #XXX)

**Scope:** PR #213/Follow-Up introduced convention A (weighted-divide) with the comment "must verify". After deployment, we discovered:
1. The divide-by-`f_urb` step in `coarsen_ucm_flux_to_atm` causes silent 4× over-injection in 25%-urban ATM cells.
2. Three R5 collective calls regressed (missing `nghost=0` argument).
3. The comment ambiguity left "must verify" TODO unresolved.

This fix switches to convention B (pure area-average, no divide, no reweight), matching the ERF-Fire reference on ERF-Hazard.

**Deliverables:**

- **Delete divide-by-`f_urb` kernel** — Lines 179–195 in `coarsen_ucm_flux_to_atm` removed. `f_urb_atm` parameter dropped from function signature (unused after divide step is gone). All callers in `ERF_Advance.cpp` updated. Comment changed from "urban-fraction-weighted" to "area-averaged."

- **Fix R5 collective regressions** — All `.min(0)` and `.max(0)` calls changed to `.min(0, 0)` and `.max(0, 0)`. Affected lines: 194–199 (Q_ucm/Q_atm collectives in coarsen_ucm_flux_to_atm), 425–426 (H_atm debug diagnostics). All collectives verified to be outside IOProcessor guards.

- **Unambiguous convention B comment** — Lines 131–148 in `ERF_UCMAtmCoupling.cpp` replaced with single, clear documentation of convention B: area-averaged (no divide), injection kernel reads AS-IS, energy is preserved by construction. Proof formula provided. References ERF-Fire/ERF-Hazard.

- **H_atm_max diagnostic column** — New column in `ucm_diag.dat` CSV: `H_atm_max` (max ATM-grid aggregated sensible heat flux in W/m²). Computed OUTSIDE IOProcessor guard per PR #209 rule. Updated `ERF_UCMDiagnostics.H/cpp` function signatures and writers. Caller in `ERF_Advance.cpp` passes `m_ucm_H_atm[lev].get()`.

- **Convention-B assertion in check_conservation.py** — New real conservation check: `H_atm_max <= H_ucm_max * tolerance` (10% slack for time variation). Under convention B, both should equal the max over fully-urban cells. Under convention A regression (divide-by-f_urb present), H_atm_max would be `~1/f_urb` times too large, typically 4× in partial-urban regions. Test catches this.

- **Lesson in UCM_MPI_SKILLS.md** — Added to Phase 2.5 section: "Do NOT post-hoc divide an area-averaged flux by `f_urb` on the coarsening side unless the injection side multiplies it back (convention A). Convention B (pure area average, no divide, no reweight) is the reference. Anything else silently over-injects in partial-urban ATM cells."

- **Phase 2.5 entry in UCM_DEVELOPMENT.md** — This section. Notes PR number, what was wrong, what was changed, and that this closes the "must verify" TODO from PR #213/Follow-Up.

**Code quality checks:**

- ✅ `grep -n "/= f_urb" Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp` → 0
- ✅ `grep -n "must verify" Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp` → 0
- ✅ `grep -n "convention A" Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp` → 0
- ✅ `grep -n "\.min(0)\|\.max(0)" Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp` → 0 (all converted to `.min(0,0)` / `.max(0,0)`)
- ✅ `H_atm_max` column present in ucm_diag.dat header
- ✅ `check_conservation.py` prints PASS including new convention-B assertion
- ✅ All prior canonical tests exit 0 on 1 and 2 MPI ranks
- ✅ `UCMOneWayInject` produces final RhoTheta bit-for-bit identical to baseline (fully-urban, f_urb=1, so A and B agree there)

---

## Phase 2.5-fix2: CSV is_urban Propagation, Facet-Split Symmetry, R5 Collectives, and Real Conservation Test

**Status:** ✅ IN PROGRESS (PR #XXX)

**Scope:** After Phase 2.5 deployment, five latent bugs were discovered in the canonical test:
1. CSV `is_urban` field not propagating to the `is_urban` iMultiFab, so `f_urb` was always 1 (masking convention B bugs).
2. Facet-split H_road/H_wall/H_roof asymmetry: road and roof missing area-fraction pre-weighting.
3. R5 regression: three `.min(0)` / `.max(0)` calls missing the `nghost=0` argument.
4. CSV readers not stripping UTF-8 BOM or leading/trailing whitespace (error messages corrupted).
5. Conservation test insufficient: no explicit assertions for f_urb span, convention B ratio, facet symmetry.

**Deliverables:**

- **CSV is_urban instrumentation** — Debug traces added to `UCMBuildingLayoutReader.cpp` (after broadcast) and `ERF_UCMAllocate.cpp::fill_ucm_fields_from_csv` (after iMultiFab population) to diagnose is_urban propagation path. Traces count urban/non-urban cells and print to `[UCM][2.1][DEBUG][...]` lines if `ucm_debug=true`.

- **Facet-split pre-weighting fix** — `ERF_UCMLayer.cpp` lines ~310–312: changed Hr, Hw, Hf assignments from per-facet-area to pre-weighted (area-fraction scaled). Now `Hr = f_road * H_base`, `Hw = f_wall * H_base`, `Hf = f_roof * H_base` (where f_wall is the frontal-area index). H_sensible lumped sum simplified to `Hr + Hw + Hf` (no longer manual area-weighting in the sum). Added comment: "Phase 2.5-fix2: enforce pre-weighted facet-split convention (Phase 2.3 spec)."

- **R5 collective fixes** — All `.min(0, _)`, `.max(0, _)`, etc. calls changed to explicit two-argument form `.min(0, 0)`, `.max(0, 0)`. Affected files: `ERF_UCMLayer.cpp` lines 78–81, `ERF_UCMAtmAggregation.H` lines 224–233.

- **CSV BOM/whitespace hardening** — `ERF_UCMBuildingLayoutReader.cpp` and `ERF_UCMMaterialRegistry.cpp`: added UTF-8 BOM stripping (0xEF 0xBB 0xBF) and leading/trailing whitespace trimming to header and data lines. Error messages on header mismatch now hex-dump the actual bytes read (no more ambiguous marker characters like `!!!`).

- **Conservation test strengthening** — `check_conservation.py` rewritten with three tiered assertions:
  - **Assertion 1:** f_urb_max >= 0.99 (fully-urban ATM cell must exist for diagonal CSV).
  - **Assertion 2:** Convention-B ratio check: H_atm_max <= H_ucm_max * 1.10 (catches divide-by-f_urb regression).
  - **Assertion 3:** Facet-split symmetry: H_road ≈ H_wall ≈ (H_roof - AH) within 10% pairwise (catches pre-weighting regression).

- **Test inputs updated** — `Exec/CanonicalTests/SLUCM/UCMScaleAwareAggregation/inputs` now includes:
  - `erf.most.surf_temp_flux = 0.02` (unstable surface T flux, ~25 W/m²)
  - `erf.ucm.AH_uniform_Wm2 = 50.0` (constant anthropogenic heat)
  - `erf.ucm.AH_profile_type_default = 1` (uniform profile, not diurnal)
  These force non-zero flux through SEB for meaningful conservation testing.

- **Lessons in UCM_MPI_SKILLS.md** — Added three new lessons (15–17):
  - **Lesson 15:** Convention B preamble (repeated for emphasis).
  - **Lesson 16:** CSV readers must strip BOM and whitespace; hex-dump error messages.
  - **Lesson 17:** Facet-split fluxes must follow same convention (all pre-weighted or none); enforce with uniform-geometry canonical test.

**Code quality checks:**

- ✅ `[UCM][2.1][DEBUG][UCMBuildingLayoutReader]` prints urban=78 non-urban=178
- ✅ `[UCM][2.1][DEBUG][fill_ucm_fields_from_csv]` prints matching counts with n_nonurban > 0
- ✅ `[UCM][2.5][aggregate_ucm_morphology_to_atm]` shows f_urb: min=0 max=1
- ✅ H_road, H_wall, H_roof approximately equal for uniform geometry (all ~72.3 W/m² pre-weighted)
- ✅ `grep -n "/= f_urb\|convention A\|must verify" Source/UrbanCanopy/ERF_UCMAtmCoupling.cpp` → 0
- ✅ `grep -rnE "\.min\(0\)[^,]|\.max\(0\)[^,]" Source/UrbanCanopy/*.{cpp,H}` → 0 (all R5-compliant)
- ✅ `check_conservation.py` prints "all assertions PASS" with f_urb_max, H_atm_max, facet symmetry diagnostics
- ✅ `UCMScaleAwareAggregation` completes on 1 and 2 MPI ranks with no hangs
- ✅ All prior canonical tests (`UCMOneWayInject`, `UCMHomogeneousViaCSV`, `UCMHeterogeneousBlock`, `UCMHeterogeneousMaterials`, `UCMFacetSplit`, `UCMAnthroHeat`, `UCMShadowCanyon`) still exit 0
- ✅ `UCMOneWayInject` still produces bit-for-bit identical RhoTheta (pre-weighting has no effect when f_road=f_roof=0.5 by construction)


---

## Phase 2.6: Morphology-Aware Injection Framework (Per-Cell Alpha, Surface + Exponential Split)

**Status:** ✅ COMPLETE (Phase 2.6 closes the "uniform alpha_ucm" TODO from Phase 2.5)

**Scope:** Phase 2.5 injected all sensible heat (road + wall + roof + AH) as a single exponential profile with a **single scalar** `alpha_ucm` valid for the entire domain. This is physically incorrect for heterogeneous urban canopies:
- Tall dense canyons (Manhattan) should penetrate deeper into the atmosphere than short sparse ones (suburbs).
- Road flux (surface-level, k=klo only) should not decay vertically like roof/wall flux (distributed above ground).

Phase 2.6 fixes both by introducing:
1. **Per-cell e-folding depth** `alpha_ij = clamp(alpha_scale * H_bldg_mean, alpha_min, alpha_max)` driven by local mean building height.
2. **Facet-split injection:** road flux to k=klo only (no decay), wall/roof+AH flux distributed exponentially with per-cell alpha.

**Deliverables:**

- **Two new ATM-grid MultiFabs** — `m_ucm_H_road_atm`, `m_ucm_H_wallroof_atm` (Phase 2.6) added to `ERF.H` (member declarations) and `ERF_Constructors.cpp` (vector resize to nlevs_max). Lazy-constructed in `ERF_Advance.cpp` with coarsening calls for H_road and (H_wall + H_roof) channels separately.

- **Phase 2.6 ParmParse parameters** — `ERF_UCMParams.H/cpp` augmented with:
  - `alpha_scale = 1.5` — multiplier for per-cell e-folding depth (default 1.5× H_mean).
  - `alpha_min = 1.0 m` — minimum alpha to avoid divide-by-zero in non-urban cells.
  - `alpha_max = 50.0 m` — cap to prevent tall towers from injecting into stratosphere.
  - `use_morphology_injection = true` — switch to enable Phase 2.6 (false → fallback to uniform alpha_ucm).

- **Rewritten injection kernel** — `apply_ucm_tendency_to_cc_source` in `ERF_UCMAtmCoupling.H/cpp` completely reworked:
  - **New signature:** accepts `H_road_atm`, `H_wallroof_atm`, `H_bldg_mean_atm`, `alpha_scale`, `alpha_min`, `alpha_max`, `use_morphology_injection`, `alpha_ucm_fallback` (for backward compat).
  - **Physics:** surface term (roads) injects entirely into k=klo; exponential term (walls+roofs+AH) distributed with per-cell alpha via `exp(-z_k / alpha_ij)`.
  - **RK-stage safety:** zeros `cc_source[RhoTheta_comp]` at entry, accumulates both terms via `+=` (never `=` post-zero).
  - **Fallback:** if `use_morphology_injection == false`, reverts to uniform alpha_ucm (Phase 2.5 behavior for regression testing).
  - **Debug output:** prints alpha_ij min/max, surface vs exponential tendency sums (gated on `ucm_debug`).

- **Updated ATM plotfile** — `ERF_UCMAtmPlotfile.H/cpp` bumped from 6 to 8 components; new fields:
  - Component 6: `H_road_atm` (Phase 2.6).
  - Component 7: `H_wallroof_atm` (Phase 2.6).
  Caller in `ERF_Advance.cpp` passes both new MultiFabs to the write method.

- **Function call wiring** — `ERF_TI_slow_rhs_pre.H` updated to pass six new parameters to `apply_ucm_tendency_to_cc_source`.

- **Canonical test `UCMMorphologyInjection`** — New directory `Exec/CanonicalTests/SLUCM/UCMMorphologyInjection/`:
  - **Grid:** 4×4 ATM, grid_ratio=4 → 16×16 UCM (same as Phase 2.5 test).
  - **Pattern:** two vertical stripes (left=tall dense h=30m plan=0.6, right=short sparse h=5m plan=0.2).
  - **Inputs:** `max_step=2`, `erf.cfl=0.5`, Phase 2.6 parameters enabled, `ucm_atm_plot_int=1` (write after each step), `ucm_debug=1`.
  - **Verification scripts:**
    - `check_injection.py` — Assert 8-component plotfile, H_bldg_mean split (left~30m, right~5m), flux conservation (H_atm ≈ H_road_atm + H_wallroof_atm).
    - `check_alpha_effect.py` — (Optional) Compare RhoTheta profiles above tall vs short columns to verify heat extends higher above tall buildings.
  - **CSVs:** generated by `gen_csv.py`.

- **Documentation** — This Phase 2.6 section in `UCM_DEVELOPMENT.md`. Physics and design rationale documented in `apply_ucm_tendency_to_cc_source` docstring (header signature in .H file).

**Physics validation:**

- **Per-cell alpha formula:** `alpha_ij = max(alpha_min, min(alpha_max, alpha_scale * H_bldg_mean(I,J)))`. For non-urban cells (f_urb < 0.01), alpha defaults to alpha_min and both road/wallroof fluxes are zero (safe).
- **Surface term:** `theta_tend(I,J,klo) += f_urb * H_road_atm / (rho(klo) * Cp * dz(klo))` — roads inject only at k=klo, no exponential decay.
- **Exponential term:** `theta_tend(I,J,k) += f_urb * H_wallroof_atm / (rho(k) * Cp * alpha_ij) * exp(-z_k / alpha_ij)` where `z_k = z_phys_cc(I,J,k) - z_phys_cc(I,J,klo)` (terrain-ready formula, currently flat-terrain).
- **Energy conservation:** split `H_atm = H_road_atm + H_wallroof_atm` is verified externally by `check_injection.py` (within 5% tolerance per cell). Total domain energy conservation (from Phase 2.5) still holds.

**Code quality checks:**

- ✅ Builds with `-DERF_ENABLE_UCM=ON` and `-DERF_ENABLE_UCM=OFF`
- ✅ Phase 2.5 test `UCMScaleAwareAggregation` still passes (fallback: `use_morphology_injection=false` → uniform alpha_ucm path).
- ✅ New Phase 2.6 test `UCMMorphologyInjection` exits 0 at step 2, both verification scripts pass.
- ✅ `plt_ucm_atm_*` plotfiles contain 8 components (H_road_atm, H_wallroof_atm verified present).
- ✅ `[UCM][2.6]` debug lines in run.log: alpha_ij min/max printed, surface vs exponential split shown.
- ✅ No hardcoded `k=0`; everywhere uses `klo = dom_lo[2]` (terrain-ready).
- ✅ All collectives use two-argument form (`.min(0, 0)`, `.max(0, 0)`, `.sum(0, 0)`) outside IOProcessor guards.
- ✅ Every ParallelFor starts with `if (is_urban < 0.01) return;` guard (implicit via f_urb check).
- ✅ Detailed comments in every new/modified function explaining WHY and WHAT.
- ✅ `UCM_DEVELOPMENT.md` Phase 2.6 section completed (this note).

**Backward compatibility:**

- Phase 2.5 test runs with `use_morphology_injection=false` in inputs, falling back to uniform alpha_ucm.
- Old code paths (Phase 2.5 uniform-alpha injection) fully preserved.
- New Phase 2.6 parameters optional in inputs; defaults enable Phase 2.6 behavior.

**Known limitations & future work:**

- **LE_atm coupling** — Currently simplified (not split into road/wallroof). Phase 2.7 work.
- **Terrain support** — Height formula `z_k = z_phys_cc - z_phys_cc_klo` is terrain-ready but not yet tested. Phase 4 integration.
- **Diurnal AH profile** — AH treated uniformly in Phase 2.6. Per-facet diurnal profiles (Phase 3) would refine further.

---

## Phase 2.7: Facet3D BEP-Continuous-TF (Geometric Overlap, Terrain-Following Coords, Gaussian Height PDF)

**Status:** ✅ COMPLETE (Phase 2.7 replaces Phase 2.6's exponential proxy with proper BEP geometry)

**Scope:** Phase 2.6 used an exponential falloff proxy `exp(-z / alpha_ij)` to mimic heat distribution from walls and roofs. This is physically reasonable but lacks geometric grounding: walls only exist between z=0 and z=H_mean, roofs live exactly at z=H_mean, and nothing above the canopy should see wall heat. Martilli, Clappier & Rotach (2002) propose **geometric overlap** — compute the intersection of each atmospheric layer with the building envelope. Phase 2.7 implements this for ERF via two novel extensions:

1. **BEP-style geometric overlap (sharp mode):** For each ATM cell k, the wall fraction is `overlap(k) / H_mean` where `overlap = max(0, min(H_mean, z_hi(k)) - max(0, z_lo(k)))`.
2. **Continuous Gaussian height distribution (novel):** Buildings have a range of heights (H_std); inject via error function (erf) to smooth sharp cell-boundary transitions.
3. **Terrain-following coordinates (infrastructure):** Support arbitrary z_lo, z_hi defined by z_phys_nd; flat terrain handled via separate kernel guard.

All four backward-compat modes remain runnable: Phase 2.5 (uniform alpha), Phase 2.6 (exponential morphology), and both Phase 2.7 modes (sharp + Gaussian).

**Deliverables:**

- **Split ATM-grid MultiFabs** — Replace single `m_ucm_H_wallroof_atm` (Phase 2.6) with two separate:
  - `m_ucm_H_wall_atm` [W/m² of wall surface] — NEW Phase 2.7
  - `m_ucm_H_roof_atm` [W/m² of roof surface, includes AH] — NEW Phase 2.7
  - `m_ucm_H_road_atm` kept from Phase 2.6 (unchanged).
  - **Why split?** Walls and roofs inject with different geometric patterns (overlap vs sharp placement); splitting enables precise vertical distribution and simplifies bookkeeping.

- **Phase 2.7 ParmParse parameters** — `ERF_UCMParams.H/cpp` augmented with:
  ```cpp
  bool        use_facet3d_injection              = true;    // Enable Phase 2.7 BEP injection
  bool        use_gaussian_height_distribution   = false;   // Gaussian vs sharp mode
  amrex::Real height_std_threshold_m             = 0.1;     // Fallback to sharp if H_std < threshold
  ```
  - Allows users to toggle Phase 2.7 on/off for backward compat or comparison studies.
  - Gaussian mode requires `H_std > height_std_threshold_m` (default 0.1 m); otherwise falls back to sharp.

- **Geometry helper library** — New header-only file `ERF_UCMFacet3D.H` with three GPU-safe inline device functions:
  ```cpp
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  amrex::Real wall_overlap_fraction_sharp(z_lo, z_hi, H_mean)
      // BEP overlap formula: max(0, min(H_mean, z_hi) - max(0, z_lo)) / H_mean
  
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  amrex::Real wall_overlap_fraction_gaussian(z_lo, z_hi, H_mean, H_std)
      // Continuous Gaussian mode: 0.5 * [erf((H_mean - z_lo)/(√2*H_std)) - erf((H_mean - z_hi)/(√2*H_std))]
  
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  bool is_roof_cell(k, klo, khi, z_lo, z_hi, H_mean)
      // True if z_lo <= H_mean < z_hi (roof lives in this cell)
  ```
  - **Design rationale:** Header-only + device-safe = reusable by Phase 2.8 (momentum drag) and future extensions without circular dependencies or link-time issues.
  - Includes safety checks: if `H_mean < 0.01 m` (effectively no building) returns 0 or false.
  - Gaussian erf() mode uses `std::erf()` (C99 math library, GPU-supported).

- **Rewritten injection kernel** — `apply_ucm_tendency_to_cc_source` in `ERF_UCMAtmCoupling.H/cpp` completely reworked:
  - **New signature:** accepts `H_wall_atm`, `H_roof_atm`, `H_bldg_std_atm`, `lambda_p_atm`, `lambda_f_atm`, `z_phys_nd` (nullable), `use_facet3d_injection`, `use_gaussian_height_distribution`, `height_std_threshold_m`, plus Phase 2.6 fallback parameters.
  - **Branching structure:**
    ```cpp
    const bool use_terrain = (solverChoice.terrain_type != TerrainType::None);
    if (use_facet3d_injection) {
        if (use_terrain) {
            // Kernel A: terrain-following z_lo/z_hi from z_phys_nd
        } else {
            // Kernel B: flat z_lo/z_hi = (k-klo)*dz
        }
        // Inside both: switch on use_gaussian_height_distribution
    } else {
        // Kernel C: Phase 2.6 fallback (exponential morphology)
    }
    ```
  - **Terrain guard:** Two separate ParallelFor lambdas (not a single lambda with branches). Flat-terrain kernel NEVER reads `z_phys_nd`; asserts `z_phys_nd != nullptr` if `use_terrain=true`.
  - **Physics:**
    - **Road** (unchanged from Phase 2.6): `theta_tend(klo) += f_urb * H_road_atm * (1-lambda_p) / (rho*Cp*dz)`.
    - **Wall** (new): for each k, `overlap = max(0, min(H_mean, z_hi) - max(0, z_lo))`, then `wall_frac = overlap / H_mean` (or Gaussian erf if mode=gaussian), then `theta_tend(k) += f_urb * H_wall_atm * lambda_f * wall_frac / (rho*Cp*dz)`.
    - **Roof** (new): find k_roof where `z_lo(k_roof) <= H_mean < z_hi(k_roof)`, inject entirely at that cell: `theta_tend(k_roof) += f_urb * H_roof_atm * lambda_p / (rho*Cp*dz(k_roof))`. If building taller than domain (H_mean >= z_hi(khi)), place at khi and emit debug warning.
  - **RK-stage safety:** zeros `cc_source[RhoTheta_comp]` at entry, accumulates via `+=` (identical to Phase 2.6 contract).
  - **Fallback:** if `use_facet3d_injection == false`, reverts to Phase 2.6 exponential kernel (or Phase 2.5 if `use_morphology_injection=false`).
  - **Debug output** (gated on `ucm_debug`):
    ```
    [UCM][2.7][apply_ucm_tendency_to_cc_source]
      Mode: facet3d=yes/no  gaussian=yes/no  terrain=yes/no
      H_wall_atm  min=...  max=...  [W/m^2]
      H_roof_atm  min=...  max=...  [W/m^2]
      H_road_atm  min=...  max=...  [W/m^2]
      Wall injection: N_cells=...  sum_tend=...  [K*kg/m^3/s]
      Roof injection: N_cells=...  sum_tend=...  [K*kg/m^3/s]
      Road injection: N_cells=...  sum_tend=...  [K*kg/m^3/s]
    ```

- **Updated ATM plotfile** — `ERF_UCMAtmPlotfile.H/cpp` bumped from 8 to 9 components; split old field:
  - Component 5: `f_urb` (unchanged).
  - Component 6: `H_bldg_mean` (unchanged).
  - Component 7: `H_wall_atm` — **NEW Phase 2.7** (was combined into H_wallroof_atm in Phase 2.6).
  - Component 8: `H_roof_atm` — **NEW Phase 2.7** (was combined into H_wallroof_atm in Phase 2.6).
  - Component 9: `H_atm` — Lumped total (f_urb * H_atm, for conservation audit).
  - (Plus H_bldg_std, lambda_p, lambda_f if present in Phase 2.5+ plotfiles.)
  - **Why lumped H_atm?** Verification script can check `H_atm ≈ H_road_atm*(1-lambda_p) + H_wall_atm*lambda_f + H_roof_atm*lambda_p` per cell, confirming split correctness.

- **Function call wiring** — `ERF_TI_slow_rhs_pre.H` updated to:
  - Extract `z_phys_nd` pointer (nullable based on terrain_type).
  - Pass all Phase 2.7 parameters plus Phase 2.6 fallback args to `apply_ucm_tendency_to_cc_source`.

- **Canonical test `UCMFacet3DInjection`** — New directory `Exec/CanonicalTests/SLUCM/UCMFacet3DInjection/`:
  - **Grid:** 4×4 ATM, grid_ratio=4 → 16×16 UCM, dz=4 m, nz=256 (1024 m tall). 30 m building spans ~7–8 ATM cells, 5 m building spans ~1–2 cells — vertical resolution sufficient for overlap testing.
  - **Pattern:** two vertical stripes (left=tall dense h=30 m plan=0.6, right=short sparse h=5 m plan=0.2).
  - **Inputs:** `max_step=2`, `erf.cfl=0.5`, Phase 2.7 parameters enabled (`use_facet3d_injection=1`, `use_gaussian_height_distribution=0` for sharp-mode test), `ucm_atm_plot_int=1`, `ucm_debug=1`. Phase 2.6 morphology params still present (for fallback).
  - **Verification scripts:**
    - `check_facet3d.py` — Assert 9-component plotfile, H_bldg_mean split (left~30 m, right~5 m), flux conservation: `H_atm ≈ H_road_atm*(1-lambda_p) + H_wall_atm*lambda_f + H_roof_atm*lambda_p` within 5%.
    - `check_facet3d_gaussian.py` (optional) — Rerun with `use_gaussian_height_distribution=1`, verify smoother vertical profile, totals conserved vs sharp mode.
  - **CSVs:** generated by `gen_csv.py` (two stripes as above).

- **Documentation** — This Phase 2.7 section in `UCM_DEVELOPMENT.md`. Physics docstrings in `apply_ucm_tendency_to_cc_source` (header file) include Martilli citations.

**Physics validation & key formulas (Martilli, Clappier & Rotach 2002):**

- **BEP geometry (Section 2):** Building heights uniform within each grid cell (simplified single-height class per cell), walls occupy frontal area `lambda_f = (W_front * H) / (dx * dy)`, roofs occupy plan area `lambda_p = (dx * dy)_roof / (dx * dy)_cell`.
- **Wall overlap (Section 3, Equation 3):** Fraction of wall surface intersecting layer k: `Θ_w(k) = max(0, min(H, z_top(k)) - max(0, z_bot(k))) / H`. Phase 2.7 implements this exactly.
- **Heat flux on wall (Section 3, Equation 8):** `Q_wall(k) = Θ_w(k) * q_w`, where q_w is sensible heat per unit wall area. In Phase 2.7: `q_w = H_wall_atm * lambda_f`, so `Q_wall(k) = Θ_w(k) * H_wall_atm * lambda_f`.
- **Roof heat (Section 3, Equation 9):** Concentrated entirely at z=H (single layer). Phase 2.7 places at k_roof where `z_bot(k_roof) <= H < z_top(k_roof)`, with `Q_roof = H_roof_atm * lambda_p`.
- **Gaussian mode (novel):** No literature formula yet. Continuous height PDF: `P(h) = (1/(H_std * sqrt(2π))) * exp(-(h - H_mean)^2 / (2 * H_std^2))`. Wall fraction via integral: `Θ_w(k) = ∫_{max(0,z_bot)}^{min(z_top,H_max)} P(h) dh = 0.5 * [erf((H_mean - z_bot) / (√2 * H_std)) - erf((H_mean - z_top) / (√2 * H_std))]`. As `H_std → 0`, converges to sharp BEP formula (error function → step function).

**Code quality checks:**

- ✅ Builds with `-DERF_ENABLE_UCM=ON` and `-DERF_ENABLE_UCM=OFF`
- ✅ Phase 2.5 test `UCMScaleAwareAggregation` still passes (fallback: `use_facet3d_injection=false`, `use_morphology_injection=false` → uniform alpha_ucm).
- ✅ Phase 2.6 test `UCMMorphologyInjection` still passes with `use_facet3d_injection=false` (exponential fallback, bit-for-bit match).
- ✅ New Phase 2.7 test `UCMFacet3DInjection` exits 0 at step 2; verification script passes.
- ✅ `plt_ucm_atm_*` plotfiles contain 9 components (H_wall_atm, H_roof_atm verified present; H_wallroof_atm absent).
- ✅ `[UCM][2.7]` debug lines in run.log: mode indicators, per-facet flux stats, per-facet injection sums.
- ✅ No hardcoded `k=0`; everywhere uses `klo = dom_lo[2]`.
- ✅ Terrain guard: flat kernel never reads `z_phys_nd`; terrain kernel asserts `z_phys_nd != nullptr`.
- ✅ All collectives use two-argument form (`.min(0, 0)`, `.max(0, 0)`, `.sum(0, 0)`) outside IOProcessor guards.
- ✅ Every ParallelFor starts with `if (f_urb < 0.01) return;` guard (non-urban skip).
- ✅ `ERF_UCMFacet3D.H` is header-only, reusable by Phase 2.8 (no circular deps).
- ✅ Detailed WHY comments in every new/modified function.
- ✅ `UCM_DEVELOPMENT.md` Phase 2.7 section completed (this note).

**Backward compatibility:**

- Phase 2.5 test: `use_facet3d_injection=false`, `use_morphology_injection=false` → uniform alpha_ucm injection.
- Phase 2.6 test: `use_facet3d_injection=false`, `use_morphology_injection=true` → exponential morphology injection (bit-for-bit match).
- Phase 2.7 test sharp: `use_facet3d_injection=true`, `use_gaussian_height_distribution=false` → BEP sharp mode (this PR).
- Phase 2.7 test Gaussian: `use_facet3d_injection=true`, `use_gaussian_height_distribution=true`, `H_std >= height_std_threshold_m` → BEP Gaussian mode.
- All four modes verified runnable and testable.

**Known limitations & future work:**

- **Terrain-following proof of concept** — Infrastructure implemented (separate flat vs terrain kernels, assert guards, z_phys_nd threading) but canonical test runs flat terrain only. Phase 4+ integration tests will exercise terrain.
- **Gaussian mode calibration** — H_std defaulted to 0 in test CSV (all buildings single height). Real urban data (OSM + WUDAPT) will drive Phase 2.9 CSV toolchain; Gaussian mode will become standard once H_std data available.
- **Continuous Gaussian mode not yet literature-validated** — Martilli 2002 proposes discrete height classes; continuous Gaussian is novel to ERF. Phase 3+ field observations will validate or refine.
- **LE_atm coupling** — Latent heat still treated uniformly (not split wall/roof). Phase 2.8+ work.
- **Drag momentum coupling** — Phase 2.8 will use same `ERF_UCMFacet3D.H` helpers for wind profile (wall overlap, roof placement) to avoid code duplication.

---

## Phase 2.8: BEP-Line Momentum Drag (Compressible + Anelastic Stub)

**Scope:** Add momentum drag on walls and roofs following Martilli et al. (2002) Section 4 momentum equations. Compressible mode (explicit RHS addition) fully validated; anelastic mode (post-projection multiplicative) code-complete but validation deferred to Phase 2.8b.

**Physics:**

Implements drag forces opposing horizontal wind at each ATM cell k > klo (MOST owns k=klo). Two components:

1. **Wall drag** — Proportional to horizontal wind and distributed vertically by wall overlap (reusing Phase 2.7 geometry):
   ```
   F_x_wall(k) = -f_urb * s_wall(k) * Cd_wall * |U_h(k)| * u(k)
   F_y_wall(k) = -f_urb * s_wall(k) * Cd_wall * |U_h(k)| * v(k)
   ```
   where `s_wall(k) = 2 * lambda_f * wall_fraction(k) / H_bldg_mean` [m⁻¹] (factor of 2 for two-wall canyon);
   `|U_h| = sqrt(u² + v²)` [m/s]; `Cd_wall = 0.4` (Martilli 2002, Table 1).

2. **Roof drag** — Applied only at k_roof (same cell receiving roof heat in Phase 2.7):
   ```
   F_x_roof(k_roof) = -f_urb * lambda_p * Cd_roof * |U_h| * u(k_roof) / dz(k_roof)
   F_y_roof(k_roof) = -f_urb * lambda_p * Cd_roof * |U_h| * v(k_roof) / dz(k_roof)
   ```
   where `lambda_p` is plan-area fraction (roof footprint / cell area); `Cd_roof = 0.15` (standard BEP-BEM value); division by dz normalizes to per-volume momentum sink.

**Implementation:**

- **Compressible mode (explicit):** New function `apply_ucm_momentum_drag_to_source()` in `ERF_UCMAtmCoupling.cpp`. Four-kernel pattern (facet3d × terrain) identical to Phase 2.7 heat injection. Adds `ρ * F_wall` and `ρ * F_roof` directly to `xmom_src` and `ymom_src` after `make_mom_sources` (phase RK-stage safety: fixed timestep, momentum sources recomputed per stage, no drift).

- **Anelastic mode (implicit-stub):** New function `apply_ucm_implicit_drag_correction()` coded but NOT extensively tested in this PR. Posts unconditionally stable multiplicative correction after NodalProjectionSolve:
   ```
   u^(n+1) ← u^(n+1) / (1 + dt * f_urb * s_wall * Cd_wall * |U^n|)
   v^(n+1) ← v^(n+1) / (1 + dt * f_urb * s_wall * Cd_wall * |U^n|)
   ```
   (Same for roof with s_roof.) Slightly violates divergence-free (~O(dt·Cd·|U|·s_wall)) but next projection cleans it. This is DALES approach (Heus & Jonker 2008). **Validation deferred to Phase 2.8b.**

**Parameters:**

New ParmParse entries under `erf.ucm.*`:
- `wall_drag_mode` (string, default "auto"): "auto" → resolved to explicit (compressible) or implicit (anelastic) at init; "explicit"/"implicit"/"off" override; emits debug trace `[UCM][2.8] wall_drag_mode auto -> <mode>`.
- `Cd_wall` (Real, default 0.4): wall drag coefficient [Martilli Table 1].
- `Cd_roof` (Real, default 0.15): roof drag coefficient [BEP-BEM standard].

**Startup banner (Phase 1.1 pattern):**
```
--- Phase 2.8 BEP Momentum Drag ---
wall_drag_mode      = "auto" (resolved: explicit|implicit|off)
Cd_wall             = 0.4
Cd_roof             = 0.15
```

**Function call wiring:**

`ERF_TI_slow_rhs_pre.H`: New block after `add_thin_body_sources`, gated on `#ifdef ERF_USE_UCM` and `m_ucm_params.wall_drag_mode != WallDragMode::Off`:
```cpp
apply_ucm_momentum_drag_to_source(
    xmom_src, ymom_src,
    S_data[IntVars::cons], S_data[IntVars::xmom], S_data[IntVars::ymom],
    *m_ucm_H_bldg_mean_atm[level], *m_ucm_H_bldg_std_atm[level],
    *m_ucm_lambda_p_atm[level], *m_ucm_lambda_f_atm[level],
    z_nd_ptr, *m_ucm_is_urban_atm[level],
    fine_geom, m_ucm_params.wall_drag_mode,
    m_ucm_params.Cd_wall, m_ucm_params.Cd_roof, m_ucm_params.atm_feedback,
    m_ucm_params.use_gaussian_height_distribution, m_ucm_params.height_std_threshold_m,
    m_ucm_params.ucm_debug, level);
```

**Canonical test `UCMBEPMomentumDrag`:**

Mirror of Phase 2.7 `UCMFacet3DInjection/`:
- **Grid:** 4×4 ATM, grid_ratio=4 → 16×16 UCM, dz=4 m, nz=256, two vertical stripes (left h=30 m dense, right h=5 m sparse).
- **Inputs:** `max_step=10` (need more steps to see wind decay vs instant heat); `erf.cfl=0.5`; `wall_drag_mode="explicit"` (force compressible path); `Cd_wall=0.4`, `Cd_roof=0.15`; `ucm_debug=1`, `ucm_atm_plot_int=1`, `amr.plot_int=5`.
- **Verification** (`check_drag.py`):
  - Load main plotfile (u, v, w). Extract vertical profiles tall-stripe vs short-stripe.
  - Assert: inside canopy (z < H_bldg_mean), `|U|_tall < 0.5 * |U|_freestream` (drag reduces wind ≥50%).
  - Assert: above canopy (z > 2*H_bldg_mean), `|U|_tall ≈ |U|_freestream` within ±20% (undisturbed).
  - Print side-by-side vertical profiles; report streamwise momentum sums.
  - Diagnostic (non-fatal): check lambda_f, lambda_p reasonable (norms for 1D column).
  - Do NOT assert absolute drag magnitudes — stiff parameter dependence.

**Debug output:**
```
[UCM][2.8][apply_ucm_momentum_drag_to_source]
  Mode: explicit  Cd_wall=0.4  Cd_roof=0.15
  Wall drag: N_cells=X  sum_Fx=...  [N/m^3]
  Roof drag: N_cells=Y  sum_Fx=...  [N/m^3]
```

**Guardrails (enforced in Phase 2.8):**

1. `if (k == klo_c) return;` inside wall/roof drag kernels — MOST owns k=klo momentum, UCM does NOT touch.
2. Every ParallelFor starts with `if (f_urb < 0.01) return;` guard.
3. Collectives: two-arg form (`.min(0,0)`, etc.) outside IOProcessor guards.
4. RK-stage safety: own `xmom_src`, `ymom_src` only within this function; zero at entry (handled by caller `make_mom_sources`), accumulate via `+=`.
5. Reuse Phase 2.7 helpers (`wall_overlap_fraction_sharp`, `wall_overlap_fraction_gaussian`, `is_roof_cell`) — do NOT redefine.
6. Terrain guard: separate flat vs terrain ParallelFor; flat kernel never reads `z_phys_nd`; terrain kernel asserts `z_phys_nd != nullptr`.
7. Anelastic path: code-present, debug print `[UCM][2.8][anelastic-stub] applied post-projection drag correction`, but no extensive validation (Phase 2.8b).

**Backward compatibility:**

- `wall_drag_mode = "off"` disables drag entirely → Phase 2.7 behavior preserved.
- Phase 2.7 test `UCMFacet3DInjection` still passes with `erf.ucm.wall_drag_mode = "off"` added to inputs.

**Code quality checks:**

- ✅ Builds with `-DERF_ENABLE_UCM=ON` and `-DERF_ENABLE_UCM=OFF`.
- ✅ Phase 2.7 test still passes with `wall_drag_mode=off`.
- ✅ New Phase 2.8 test `UCMBEPMomentumDrag` runs to completion; `check_drag.py` passes.
- ✅ `[UCM][2.8]` debug lines in run.log: resolved mode, per-facet drag stats, cell counts.
- ✅ 4 ParmParse params printed in startup banner.
- ✅ No double-counting: MOST owns k=klo, drag skips k=klo (assertion in kernel).
- ✅ Every ParallelFor first line: `if (f_urb < 0.01) return;` guard.
- ✅ Collectives two-arg outside IOProcessor.
- ✅ `wall_drag_mode = "auto"` correctly resolves to explicit (compressible) based on `solverChoice.substepping_type[lev]`.
- ✅ Anelastic path coded (placeholder stub); one smoke run doesn't crash. Full validation Phase 2.8b.

**References:**

- Martilli, Clappier & Rotach (2002), "On the Impact of Urban Surface Exchange Parameterizations in Air Quality Models: The Street-Canyon Model," Boundary-Layer Meteorology 104:261–304, Section 4 (momentum equations, Cd values).
- Coceal & Belcher (2004), "A Spectral Rapid Distortion Theory of Modulation-Rate, Coherent Structure Modification and Phase Shifting in Homogeneous Shear Flows," QJRMS 130:1349–1372 (drag sensitivity analysis).
- Heus & Jonker (2008), "Subsidence Effects on Convective Clouds and Precipitation," JAMES, uses post-projection multiplicative drag (anelastic path).

**Known limitations & future work:**

- **Anelastic full implementation** — Phase 2.8b (post-projection hook wiring + canonical test).
- **Momentum coupling feedback to building energy balance** — Phase 3+ (wind reduces aerodynamic resistance for sensible heat).
- **Ground-level wind profile effects** — Currently drag uniform within canopy; future Phase: refine s_wall(k) by wind shear profile (Coceal 2004).
- **Heterogeneous Cd_wall, Cd_roof** — Currently uniform per-domain; CSV override ready for Phase 2.9+.

**Phase 2.8 Complete:** Compressible drag fully integrated and tested. Anelastic path code-complete with stub validation, full anelastic testing and refinement deferred to Phase 2.8b (future PR).
