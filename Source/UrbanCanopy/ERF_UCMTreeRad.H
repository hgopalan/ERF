/**
 * @file ERF_UCMTreeRad.H
 * @brief Phase 6.2a Beer-Lambert SW attenuation through tree crown.
 *
 * Header-only after Phase 6.2a-hotfix1 (avoids external linkage vs.
 * AMREX_FORCE_INLINE mismatch — see UCM_MPI_SKILLS.md Lesson 28).
 */

#ifndef ERF_UCM_TREE_RAD_H
#define ERF_UCM_TREE_RAD_H

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <UrbanCanopy/ERF_UCMParams.H>

/**
 * @brief Apply Beer-Lambert SW attenuation to SEB facet inputs and update absorbed SW diagnostic.
 *
 * Called per UCM cell within the existing MFIter loop in ERF_UCMLayer::advance_ucm_layer.
 * Modifies SW input scalars in-place and accumulates absorbed SW into Q_tree_SW_abs.
 *
 * Contract #26: Attenuation is applied to SEB SW input path (roof, wall, road facets),
 * NOT to the upstream radiation MultiFab. The attenuation factor tau_tree is computed
 * per facet based on the path length through the crown using Beer-Lambert law:
 *   tau_tree = exp(-k_ext * LAD_bulk * L_path)
 * where L_path is the sunlit path length inside the crown for each facet.
 *
 * Contract #27: The Newton SEB stays 3-variable. No crown facet is added in Phase 6.2a.
 *
 * Contract #28: Phase 6.2a has double-gate safety: if tree_drag_mode=off, is_tree is
 * zero everywhere and this is a no-op. If tree_rad_mode=off, this function is not called.
 *
 * Per-facet path-length rules (Phase 6.2a-hotfix1):
 *   Roof (top of building at z = H_roof == H_bldg):
 *     If roof is above crown top (H_roof >= H_tree): no attenuation.
 *     Else roof lies inside or below crown: path spans from roof up to crown top,
 *     clipped to crown envelope: L_path_roof = H_tree - max(H_roof, H_crown_base).
 *   Wall (canyon face, 0 <= z <= H_wall == H_bldg):
 *     Wall sees crown along the portion of its height inside [H_crown_base, min(H_tree, H_wall)].
 *     Use a linear blend of sunlit (above/below crown) and shaded (inside crown) fractions:
 *       f_wall_in_crown = max(0, min(H_wall, H_tree) - H_crown_base) / max(H_wall, eps)
 *       tau_wall = (1 - f_wall_in_crown) + f_wall_in_crown * exp(-k_ext * LAD_bulk * crown_depth)
 *   Road (canyon floor at z = 0):
 *     Sky view down to z = 0 always spans the full crown depth.
 *     L_path_road = crown_depth (= H_tree - H_crown_base).
 *
 * The H_road argument is retained for ABI stability (Phase 6.2b will re-purpose it
 * for the multi-layer road envelope). It is unused in Phase 6.2a-hotfix1.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void apply_ucm_tree_rad_beer_lambert(
    amrex::Real&           SW_roof_input,
    amrex::Real&           SW_wall_input,
    amrex::Real&           SW_road_input,
    amrex::Real&           Q_tree_SW_abs_cell,
    int                    is_tree,
    amrex::Real            H_tree,
    amrex::Real            H_crown_base,
    amrex::Real            LAD_bulk,
    amrex::Real            k_ext,
    amrex::Real            H_roof,
    amrex::Real            H_wall,
    amrex::Real            /*H_road*/,          // unused in 6.2a-hotfix1; kept for ABI
    amrex::Real            area_frac_roof,
    amrex::Real            area_frac_wall,
    amrex::Real            area_frac_road,
    TreeRadMode            rad_mode)
{
    // Phase 6.2a: Beer-Lambert SW attenuation
    // No-op if tree radiation mode is off or is_tree is zero (double-gate safety)
    if (rad_mode != TreeRadMode::BeerLambert || is_tree == 0) {
        return;
    }

    // Ensure physical bounds
    if (k_ext < 0.0 || LAD_bulk < 0.0) {
        return;
    }
    if (H_crown_base < 0.0 || H_tree <= H_crown_base) {
        return;
    }

    // Compute the crown depth
    const amrex::Real crown_depth = H_tree - H_crown_base;

    // ---- Per-facet path length (Phase 6.2a-hotfix1) ----

    // Roof: from roof surface up to crown top, clipped to crown envelope.
    // If roof sits above the crown, no attenuation.
    const amrex::Real L_path_roof = amrex::max(
        amrex::Real(0.0),
        H_tree - amrex::max(H_roof, H_crown_base));

    // Wall: fraction of wall height that lies inside the crown envelope.
    // Wall segment above the crown (or below crown_base) sees unattenuated SW;
    // segment inside the crown sees full-crown-depth attenuation.
    const amrex::Real wall_top     = amrex::min(H_wall, H_tree);
    const amrex::Real wall_in      = amrex::max(amrex::Real(0.0), wall_top - H_crown_base);
    const amrex::Real f_wall_in    = wall_in / amrex::max(H_wall, amrex::Real(1.0e-6));

    // Road at z = 0 always sees SW passing through the FULL crown depth.
    const amrex::Real L_path_road  = crown_depth;

    // ---- Transmission factors ----

    const amrex::Real tau_crown_full = std::exp(-k_ext * LAD_bulk * crown_depth);
    const amrex::Real tau_roof = std::exp(-k_ext * LAD_bulk * L_path_roof);
    // Linear blend: unshaded fraction stays at 1, shaded fraction gets tau_crown_full.
    const amrex::Real tau_wall = (amrex::Real(1.0) - f_wall_in)
                               + f_wall_in * tau_crown_full;
    const amrex::Real tau_road = std::exp(-k_ext * LAD_bulk * L_path_road);

    // Compute absorbed SW for diagnostic: Q_abs = (1 - tau) * SW_incident
    const amrex::Real Q_abs_roof = (amrex::Real(1.0) - tau_roof)
                                 * amrex::max(amrex::Real(0.0), SW_roof_input) * area_frac_roof;
    const amrex::Real Q_abs_wall = (amrex::Real(1.0) - tau_wall)
                                 * amrex::max(amrex::Real(0.0), SW_wall_input) * area_frac_wall;
    const amrex::Real Q_abs_road = (amrex::Real(1.0) - tau_road)
                                 * amrex::max(amrex::Real(0.0), SW_road_input) * area_frac_road;

    // Accumulate total absorbed SW
    Q_tree_SW_abs_cell += Q_abs_roof + Q_abs_wall + Q_abs_road;

    // Apply attenuation to SW inputs (multiply by tau)
    SW_roof_input *= tau_roof;
    SW_wall_input *= tau_wall;
    SW_road_input *= tau_road;
}

#endif