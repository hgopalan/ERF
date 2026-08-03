/**
 * @file ERF_UCM.H
 * @brief Top-level module header for the ERF-SLUCM urban canopy model
 *
 * ## Module Purpose
 *
 * The Single-Layer Urban Canopy Model (SLUCM) simulates the thermal and momentum
 * exchange between urban surfaces (buildings, roads, vegetation) and the atmosphere.
 * It is implemented as a 2D refined slab coupled to the ERF mesoscale atmospheric model.
 * Phase 1 focuses on one-way coupling; Phase 3 extends to two-way feedback.
 *
 * ## Six-Part Development Roadmap
 *
 * | Part | Phase | Title | Status |
 * |------|-------|-------|--------|
 * | 1 | 1.1 | Scaffold, ParmParse, Prerequisites, lev-aware API | ✅ IN PROGRESS |
 * | 1 | 1.2 | UCM 2D grid + homogeneous URBPARM reader + is_urban stub | 🔲 PLANNED |
 * | 1 | 1.3 | Slab conduction + SLUCM SEB core + wind/scalar extraction | 🔲 PLANNED |
 * | 1 | 1.4 | One-way injection + diagnostics + plotfile + homogeneous regression | 🔲 PLANNED |
 * | 2 | 2.1 | Building-layout CSV reader + material library CSV | 🔲 PLANNED |
 * | 2 | 2.2 | Per-cell morphology + heterogeneous canopy wind | 🔲 PLANNED |
 * | 2 | 2.3 | Heterogeneous facet SEB + anthropogenic heat | 🔲 PLANNED |
 * | 2 | 2.4 | Shadowing + heterogeneous regression | 🔲 PLANNED |
 * | 3 | 3.1 | Finest-level anchoring + multi-level regression | 🔲 PLANNED |
 * | 3 | 3.2 | Two-way feedback + MRF audit + PBLH consumer guard | 🔲 PLANNED |
 * | 3 | 3.3 | Stability-aware canyon-atm exchange | 🔲 PLANNED |
 * | 3 | 3.4 | Two-way MRF regression + energy conservation | 🔲 PLANNED |
 * | 4 | 4.1 | is_urban mask enforcement (LSM + MOST bypass) | 🔲 PLANNED |
 * | 4 | 4.2 | Radiation coupling (SW/LW extraction) | 🔲 PLANNED |
 * | 4 | 4.3 | Urban/non-urban interface treatment | 🔲 PLANNED |
 * | 4 | 4.4 | Mixed-domain diurnal integration test | 🔲 PLANNED |
 * | 5 | 5.1 | Tree CSV + tree drag | 🔲 PLANNED |
 * | 5 | 5.2 | Tree radiation (Beer-Lambert + LW crown-facet) | 🔲 PLANNED |
 * | 5 | 5.3 | Tree leaf EB + local soil bucket + transpiration | 🔲 PLANNED |
 * | 5 | 5.4 | Tile-averaged fluxes + instrumented-site validation | 🔲 PLANNED |
 * | 6 | 6.1 | Multi-bounce wall radiation | 🔲 PLANNED |
 * | 6 | 6.2 | AC waste heat + building-energy sub-module | 🔲 PLANNED |
 * | 6 | 6.3 | Green roofs, cool roofs, permeable pavements | 🔲 PLANNED |
 * | 6 | 6.4 | Worry-list audit + v1.0 release | 🔲 PLANNED |
 *
 * ## Six Cross-Cutting Design Contracts (Enforced Phase 1.1+)
 *
 * **Contract 1 — lev-aware API from day 1:**
 * All UCM functions take `int lev` argument (default 0). No hardcoded `0` for
 * level indexing. UCM MultiFabs are Vector<unique_ptr> sized to finest_level+1;
 * only anchor_level allocated in Phase 1.1.
 *
 * **Contract 2 — Anchor level and static refinement:**
 * `anchor_level` (int, default 0): Single AMR level UCM runs on.
 * `static_refinement` (bool, default true): Regrid errors out if true.
 * Phase 1.1: anchor_level=0 only; Phase 3.1 enables higher levels.
 *
 * **Contract 3 — Terrain-following coordinates:**
 * All vertical ops use `z_phys_cc(i,j,k) - z_phys_cc(i,j,klo)` for height-above-surface.
 * Wind extraction: `z_target = z_phys_cc(i,j,klo) + H_bldg(i,j) + zref`.
 * Exponential injection: `exp(-(z_phys_cc(i,j,k) - z_phys_cc(i,j,klo)) / alpha_ucm)`.
 * Slope > 20° → warning; > 30° → error (unless allow_steep_terrain=true).
 * No hardcoded z-coordinates permitted anywhere.
 *
 * **Contract 4 — Zero PBLH dependency:**
 * UCM MUST NOT call `SurfaceLayer::get_pblh(lev)`. All stability from u*, theta*, q*, L.
 *
 * **Contract 5 — is_urban mask exclusivity:**
 * `ucm_is_urban` iMultiFab declared on ERF (allocated Phase 1.2).
 * LSM/MOST bypass hooks: Phase 4.1.
 *
 * **Contract 6 — Build and file conventions:**
 * Every .cpp in Source/UrbanCanopy/ in BOTH Make.package AND CMake/BuildERFExe.cmake.
 * All MultiFabs use IntVect(1,1,0) ghost cells (2D slab). Header-only in CEXE_headers only.
 * ParallelFor headers only where used.
 *
 * ## References (MUST be read before coding)
 *
 *   1. Source/Dust/DUST_DEVELOPMENT.md (ERF-Hazard branch)
 *      https://github.com/hgopalan/ERF/blob/ERF-Hazard/Source/Dust/DUST_DEVELOPMENT.md
 *
 *   2. Source/LNG/LNG_DEVELOPMENT.md (commit ba55b7307...)
 *      https://github.com/hgopalan/ERF/blob/ba55b7307/Source/LNG/LNG_DEVELOPMENT.md
 *
 *   3. Source/LNG/LNG_MPI_SKILLS.md (commit ba55b7307...)
 *      https://github.com/hgopalan/ERF/blob/ba55b7307/Source/LNG/LNG_MPI_SKILLS.md
 *
 *   Structural references (no modifications, patterns only):
 *   - Source/Fire/ERF_FireGrid.H
 *   - Source/Fire/ERF_FireAtmCoupling.H
 *   - Source/Dust/ERF_DustGrid.H
 *   - Source/Dust/ERF_DustAtmCoupling.H
 *   - Source/Dust/ERF_DustPrerequisites.H/.cpp
 *   - Source/Dust/ERF_DustParams.H
 *   - Source/LNG/ERF_LNGParams.H
 *   - Source/LNG/ERF_LNGPrerequisites.H/.cpp
 */

#ifndef ERF_UCM_H
#define ERF_UCM_H

#include <UrbanCanopy/ERF_UCMParams.H>
#include <UrbanCanopy/ERF_UCMGrid.H>
#include <UrbanCanopy/ERF_UCMFields.H>
#include <UrbanCanopy/ERF_UCMAllocate.H>
#include <UrbanCanopy/ERF_UCMPrerequisites.H>
#include <UrbanCanopy/ERF_UCMWindExtract.H>
#include <UrbanCanopy/ERF_UCMSlabConduction.H>
#include <UrbanCanopy/ERF_UCMLayer.H>
// Phase 1.4: atmospheric coupling, plotfile, diagnostics
#include <UrbanCanopy/ERF_UCMAtmCoupling.H>
#include <UrbanCanopy/ERF_UCMAtmAggregation.H>
#include <UrbanCanopy/ERF_UCMTreeDrag.H>
#include <UrbanCanopy/ERF_UCMPlotfile.H>
#include <UrbanCanopy/ERF_UCMDiagnostics.H>

// Phase 2.1 / 6.1: CSV readers for heterogeneous inputs
#include <UrbanCanopy/ERF_UCMBuildingLayoutReader.H>
#include <UrbanCanopy/ERF_UCMTreeLayoutReader.H>
#include <UrbanCanopy/ERF_UCMMaterialRegistry.H>

// Forward declarations for future phases
// class UCMAtmCoupling;
// class UCMPlotfile;

#endif // ERF_UCM_H
