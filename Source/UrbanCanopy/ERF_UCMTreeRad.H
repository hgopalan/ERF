/**
 * @file ERF_UCMTreeRad.H
 * @brief Phase 6.2a Beer-Lambert SW attenuation through tree crown.
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
 */
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
    amrex::Real            H_road,
    amrex::Real            area_frac_roof,
    amrex::Real            area_frac_wall,
    amrex::Real            area_frac_road,
    TreeRadMode            rad_mode);

#endif
