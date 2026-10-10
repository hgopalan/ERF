#ifndef ERF_FUEL_BLENDING_H
#define ERF_FUEL_BLENDING_H

#include <AMReX_MultiFab.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <ERF_FireGrid.H>

/**
 * @file ERF_FuelBlending.H
 * @brief Fuel boundary blending for multi-fuel simulations
 *
 * Ported from wildfire_levelset/src/two_fuel_blending.H
 *
 * Blend rate-of-spread (ROS) at spatial boundaries between different fuel model zones.
 * For cells at fuel boundaries, ROS is smoothly transitioned to reduce discontinuities
 * in fire spread behavior.
 *
 * Reference:
 *   Finney, M.A. (1998). FARSITE: Fire Area Simulator. RMRS-RP-4, Appendix B.
 */

/**
 * @brief Blend ROS at spatial boundaries between different fuel model zones.
 *
 * For each cell that has a different fuel code from at least one of its
 * 4 neighbours (i±1, j) and (i, j±1):
 *   ros(i,j,0) = (1 - f) * ros(i,j,0) + f * mean(ros of differently-coded neighbours)
 *
 * where f = blending_fraction (typically 0.1–0.5).
 *
 * When blending_fraction = 0 this function returns immediately (no-op).
 *
 * The neighbours are read from one-ghost copies of the rate and the fuel
 * code, exchanged across boxes and periodic faces and extrapolated outside a
 * non-periodic face (fire_fill_boundary), with the stencil bounded by the
 * domain. Until 2026-10 the stencil read and wrote fire_ros in place and was
 * cut at the box: thread (i, j) read ros(i +- 1, j) while those threads
 * overwrote it (loop-order dependent on a CPU, a race on a GPU), and an edge
 * on a box face was never blended, so the answer changed with the rank count.
 *
 * Reference:
 *   Finney, M.A. (1998). FARSITE: Fire Area Simulator. RMRS-RP-4, Appendix B.
 *
 * @param[inout] fire_ros          Rate-of-spread field to blend
 * @param[in]    fire_fuel_model   Per-cell fuel model codes (integer stored as Real)
 * @param[in]    blending_fraction Blending weight [0-1]
 * @param[in]    geom              Fire-grid geometry (periodicity, domain)
 */
inline void apply_fuel_boundary_blending(amrex::MultiFab& fire_ros,
                                          const amrex::MultiFab& fire_fuel_model,
                                          amrex::Real blending_fraction,
                                          const amrex::Geometry& geom)
{
    // No-op if blending is disabled
    if (blending_fraction <= 0.0) { return; }

    // Clamp blending fraction to [0, 1]
    blending_fraction = amrex::min(static_cast<amrex::Real>(blending_fraction), static_cast<amrex::Real>(1.0));

    const amrex::IntVect ng(1, 1, 0);
    amrex::MultiFab ros_in(fire_ros.boxArray(), fire_ros.DistributionMap(), 1, ng);
    amrex::MultiFab fuel_in(fire_fuel_model.boxArray(), fire_fuel_model.DistributionMap(), 1, ng);
    amrex::MultiFab::Copy(ros_in, fire_ros, 0, 0, 1, 0);
    amrex::MultiFab::Copy(fuel_in, fire_fuel_model, 0, 0, 1, 0);
    fire_fill_boundary(ros_in, geom);
    fire_fill_boundary(fuel_in, geom);
    // a periodic direction reaches its neighbour across the face; a
    // non-periodic one stops at the domain (the extrapolated ghost holds the
    // cell's own code, so it never counts as a boundary either way)
    amrex::Box dom = geom.Domain();
    for (int d = 0; d < 2; ++d) { if (geom.isPeriodic(d)) { dom.grow(d, 1); } }
    const amrex::Box valid_box = dom;

    // ParallelFor over ROS field
    for (amrex::MFIter mfi(fire_ros); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();

        auto ros_out        = fire_ros.array(mfi);          // Array4 by value — required for GPU lambda capture
        auto ros_arr        = ros_in.const_array(mfi);
        auto fuel_arr       = fuel_in.const_array(mfi);     // Array4 by value

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // Get fuel code at this cell
            int fuel_c = static_cast<int>(fuel_arr(i, j, k) + 0.5);

            // Check 4 neighbours: (i-1,j), (i+1,j), (i,j-1), (i,j+1)
            bool is_boundary = false;
            int fuel_i_minus = fuel_c;
            int fuel_i_plus  = fuel_c;
            int fuel_j_minus = fuel_c;
            int fuel_j_plus  = fuel_c;

            if (valid_box.contains(amrex::IntVect(i - 1, j, k))) {
                fuel_i_minus = static_cast<int>(fuel_arr(i - 1, j, k) + 0.5);
                if (fuel_i_minus != fuel_c) { is_boundary = true; }
            }
            if (valid_box.contains(amrex::IntVect(i + 1, j, k))) {
                fuel_i_plus = static_cast<int>(fuel_arr(i + 1, j, k) + 0.5);
                if (fuel_i_plus != fuel_c) { is_boundary = true; }
            }
            if (valid_box.contains(amrex::IntVect(i, j - 1, k))) {
                fuel_j_minus = static_cast<int>(fuel_arr(i, j - 1, k) + 0.5);
                if (fuel_j_minus != fuel_c) { is_boundary = true; }
            }
            if (valid_box.contains(amrex::IntVect(i, j + 1, k))) {
                fuel_j_plus = static_cast<int>(fuel_arr(i, j + 1, k) + 0.5);
                if (fuel_j_plus != fuel_c) { is_boundary = true; }
            }

            // If at a boundary, blend ROS with differently-coded neighbours
            if (is_boundary) {
                amrex::Real ros_blend = 0.0;
                int n_diff = 0;

                if (valid_box.contains(amrex::IntVect(i - 1, j, k)) && fuel_i_minus != fuel_c) {
                    ros_blend += ros_arr(i - 1, j, k);
                    ++n_diff;
                }
                if (valid_box.contains(amrex::IntVect(i + 1, j, k)) && fuel_i_plus != fuel_c) {
                    ros_blend += ros_arr(i + 1, j, k);
                    ++n_diff;
                }
                if (valid_box.contains(amrex::IntVect(i, j - 1, k)) && fuel_j_minus != fuel_c) {
                    ros_blend += ros_arr(i, j - 1, k);
                    ++n_diff;
                }
                if (valid_box.contains(amrex::IntVect(i, j + 1, k)) && fuel_j_plus != fuel_c) {
                    ros_blend += ros_arr(i, j + 1, k);
                    ++n_diff;
                }

                if (n_diff > 0) {
                    ros_blend /= static_cast<amrex::Real>(n_diff);
                    amrex::Real ros_orig = ros_arr(i, j, k);
                    ros_out(i, j, k) = (1.0 - blending_fraction) * ros_orig +
                                       blending_fraction * ros_blend;
                }
            }
        });
    }
    // The kernels read ros_in and fuel_in, which are freed on return.
    amrex::Gpu::streamSynchronize();
}

#endif  // ERF_FUEL_BLENDING_H