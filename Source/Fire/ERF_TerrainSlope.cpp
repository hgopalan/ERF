/**
 * \file ERF_TerrainSlope.cpp
 *
 * \brief Implementation of terrain slope computation.
 */

#include "ERF_TerrainSlope.H"
#include <AMReX_ParallelFor.H>
#include <cmath>

void compute_terrain_slopes(
    amrex::MultiFab& fire_slopes,
    const amrex::MultiFab& z_phys_nd,
    const amrex::Geometry& geom_atm,
    const FireGrid& fg)
{
    using amrex::ParallelFor;

    amrex::Real dx_atm = geom_atm.CellSize(0);
    amrex::Real dy_atm = geom_atm.CellSize(1);
    amrex::Real dx_fire = fg.geom.CellSize(0);
    amrex::Real dy_fire = fg.geom.CellSize(1);
    int C = fg.C;

    // Iterate over fire slopes boxes
    for (amrex::MFIter mfi(fire_slopes); mfi.isValid(); ++mfi) {
        const amrex::Box& fire_box = mfi.tilebox();
        auto slope_arr = fire_slopes.array(mfi);

        // Atmospheric z_phys_nd array (on atmospheric grid)
        // Note: This requires that z_phys_nd is extended to cover the fire grid domain
        // with appropriate interpolation/extension
        auto z_arr = z_phys_nd.const_array(mfi);

        // Compute slopes at fire grid cell centers
        ParallelFor(fire_box, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            int i_f = iv[0];
            int j_f = iv[1];
            int k_f = iv[2];

            // Fire cell indices map to atmospheric nodal/cell indices
            // For centered differences on fire grid
            int i_f_l = i_f - 1;
            int i_f_r = i_f + 1;
            int j_f_b = j_f - 1;
            int j_f_t = j_f + 1;

            // Get heights at left/right and bottom/top on fire grid
            // (This is a simplified computation; in practice, you'd interpolate from coarser grid)
            amrex::Real z_c = z_arr(i_f, j_f, 0, 0);  // Center
            amrex::Real z_l = (i_f_l >= 0) ? z_arr(i_f_l, j_f, 0, 0) : z_c;
            amrex::Real z_r = z_arr(i_f_r, j_f, 0, 0);
            amrex::Real z_b = (j_f_b >= 0) ? z_arr(i_f, j_f_b, 0, 0) : z_c;
            amrex::Real z_t = z_arr(i_f, j_f_t, 0, 0);

            // Centered differences
            amrex::Real dz_dx = (z_r - z_l) / (2.0 * dx_fire);
            amrex::Real dz_dy = (z_t - z_b) / (2.0 * dy_fire);

            slope_arr(i_f, j_f, k_f, 0) = dz_dx;
            slope_arr(i_f, j_f, k_f, 1) = dz_dy;
        });
    }
}
