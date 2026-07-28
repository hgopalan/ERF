/**
 * @file ERF_UCMViewFactors.H
 * @brief Phase 5.1a: Multi-facet view-factor computation for urban canyons
 *
 * Computes per-cell view factors between (wall, road, sky) facets using
 * standard 2D infinite-canyon geometry (Hottel's crossed-string method).
 *
 * This is Phase 5.1a — GEOMETRY ONLY. The view factors are computed and
 * stored but NOT consumed by any SEB or radiation path in this phase.
 * Phase 5.1b will introduce a radiosity solver that consumes these fields
 * for SW multi-bounce. Phase 5.1c will extend to LW multi-bounce.
 *
 * View-factor conventions (per-cell scalars, all in [0, 1]):
 *   F_wall_sky   = wall to sky hemisphere
 *   F_wall_wall  = wall to opposite wall  (one wall to the other)
 *   F_wall_road  = wall to road
 *   F_road_sky   = road to sky hemisphere
 *   F_road_wall  = road to ONE wall  (multiply by 2 for total)
 *   F_roof_sky   = 1.0 always (unshaded from above)
 *
 * Closure identities (verified in unit tests):
 *   F_wall_sky + F_wall_wall + F_wall_road == 1
 *   F_road_sky + 2 * F_road_wall == 1
 *
 * References:
 *   - Hottel, H. C. (1954), Radiant heat transmission, in *Heat Transmission*,
 *     3rd ed., McGraw-Hill (crossed-string method for 2D geometry).
 *   - Kusaka et al. (2001), A simple single-layer urban canopy model for
 *     atmospheric models, Bound.-Layer Meteorol., 101, 329–358.
 *   - Sparrow, E. M., and R. D. Cess (1978), *Radiation Heat Transfer*,
 *     Hemisphere Publishing (comprehensive analytic view-factor formulae).
 */

#ifndef ERF_UCM_VIEW_FACTORS_H
#define ERF_UCM_VIEW_FACTORS_H

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

/**
 * @brief Compute per-cell multi-facet view factors from morphology
 *
 * For each urban cell, computes the six view factors listed in the file header
 * from the local canyon aspect ratio r = H_bldg / W_road.
 *
 * Non-urban cells (is_urban == 0) get zero view factors except F_roof_sky = 1
 * (roof always sees full sky).
 *
 * Degenerate canyon (W_road ≤ ε or H_bldg ≤ ε) → treat as isolated building:
 *   F_wall_sky = 1, F_wall_wall = 0, F_wall_road = 0
 *   F_road_sky = 1, F_road_wall = 0
 *
 * @param[out] F_wall_sky   Wall→sky view factor MultiFab
 * @param[out] F_wall_wall  Wall→opposite-wall view factor MultiFab
 * @param[out] F_wall_road  Wall→road view factor MultiFab
 * @param[out] F_road_sky   Road→sky view factor MultiFab
 * @param[out] F_road_wall  Road→one-wall view factor MultiFab
 * @param[out] F_roof_sky   Roof→sky view factor MultiFab (uniformly 1.0)
 * @param[in]  H_bldg       Building height MultiFab
 * @param[in]  W_road       Road width MultiFab
 * @param[in]  is_urban     Urban mask iMultiFab
 * @param[in]  lev          AMR level (must be anchor_level per Contract #1)
 * @param[in]  ucm_debug    If true, print field-range banner on IO rank
 */
void compute_view_factors(amrex::MultiFab& F_wall_sky,
                          amrex::MultiFab& F_wall_wall,
                          amrex::MultiFab& F_wall_road,
                          amrex::MultiFab& F_road_sky,
                          amrex::MultiFab& F_road_wall,
                          amrex::MultiFab& F_roof_sky,
                          const amrex::MultiFab& H_bldg,
                          const amrex::MultiFab& W_road,
                          const amrex::iMultiFab& is_urban,
                          int lev,
                          bool ucm_debug);

#endif  // ERF_UCM_VIEW_FACTORS_H
