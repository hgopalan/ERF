#include <UrbanCanopy/ERF_UCMTreeRad.H>

#include <cmath>

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
    amrex::Real            H_road,
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

    // Compute path length for each facet
    // Roof: full crown depth
    const amrex::Real L_path_roof = amrex::max(0.0, amrex::min(H_roof, H_tree) - H_crown_base);
    // Wall: path length if H_wall is within crown
    const amrex::Real L_path_wall = (H_wall >= H_crown_base && H_wall <= H_tree)
                                    ? amrex::min(crown_depth, amrex::max(H_roof, H_wall) - H_crown_base)
                                    : 0.0;
    // Road: path length if H_road is within crown
    const amrex::Real L_path_road = (H_road >= H_crown_base && H_road <= H_tree)
                                    ? amrex::min(crown_depth, amrex::max(H_roof, H_road) - H_crown_base)
                                    : 0.0;

    // Compute transmission factor tau = exp(-k_ext * LAD_bulk * L_path)
    const amrex::Real tau_roof = std::exp(-k_ext * LAD_bulk * L_path_roof);
    const amrex::Real tau_wall = std::exp(-k_ext * LAD_bulk * L_path_wall);
    const amrex::Real tau_road = std::exp(-k_ext * LAD_bulk * L_path_road);

    // Compute absorbed SW for diagnostic: Q_abs = (1 - tau) * SW_incident
    const amrex::Real Q_abs_roof = (1.0 - tau_roof) * amrex::max(0.0, SW_roof_input) * area_frac_roof;
    const amrex::Real Q_abs_wall = (1.0 - tau_wall) * amrex::max(0.0, SW_wall_input) * area_frac_wall;
    const amrex::Real Q_abs_road = (1.0 - tau_road) * amrex::max(0.0, SW_road_input) * area_frac_road;

    // Accumulate total absorbed SW
    Q_tree_SW_abs_cell += Q_abs_roof + Q_abs_wall + Q_abs_road;

    // Apply attenuation to SW inputs (multiply by tau)
    SW_roof_input *= tau_roof;
    SW_wall_input *= tau_wall;
    SW_road_input *= tau_road;
}
