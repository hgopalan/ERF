#ifndef ERF_FIRE_IGNITION_H
#define ERF_FIRE_IGNITION_H

/**
 * @file ERF_FireIgnition.H
 * @brief Fire ignition initialization for level-set field
 *
 * Initializes the level-set field phi for a circular ignition point.
 */

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Reduce.H>
#include <AMReX_ParallelDescriptor.H>
#include <ERF_FireGrid.H>
#include <cmath>

/**
 * @brief Initialize the level-set field phi for a circular ignition point.
 *
 * Sets phi = -(r - dist(x,y, cx, cy)) / r for cells within r metres of (cx, cy),
 * and phi = +1 elsewhere. This gives a smooth signed-distance initialisation
 * within the ignition circle and +1 (unburned) outside.
 *
 * Parameters read from ParmParse("erf.fire"):
 *   erf.fire.ignition_x   [m]
 *   erf.fire.ignition_y   [m]
 *   erf.fire.ignition_r   [m]  (default 20.0)
 *
 * @param[in,out] phi     Level-set field to initialize
 * @param[in]     geom_fire Fire grid geometry
 * @param[in]     cx      Ignition center x [m]
 * @param[in]     cy      Ignition center y [m]
 * @param[in]     radius  Ignition radius [m]
 */
inline
void initialize_ignition(
    amrex::MultiFab& phi,
    const amrex::Geometry& geom_fire,
    amrex::Real cx, amrex::Real cy, amrex::Real radius,
    bool normalized = true)
{
    using namespace amrex;

    const auto prob_lo = geom_fire.ProbLoArray();
    const auto dx      = geom_fire.CellSizeArray();

    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto p = phi.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real x = prob_lo[0] + (i + 0.5_rt) * dx[0];
            Real y = prob_lo[1] + (j + 0.5_rt) * dx[1];
            Real dist = std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));

            // Signed distance: negative inside ignition circle, positive outside.
            //
            // normalized = true  : divided by the radius and clamped to [-1, 1].
            //   This is the convention the FARSITE path uses, where phi is
            //   effectively an indicator and the magnitude only has to clear
            //   farsite.phi_threshold.
            //
            // normalized = false : the true signed distance in metres, unclamped.
            //   The level-set solver needs this. Its advection, its Godunov
            //   Hamiltonian and the Russo-Smereka subcell term are all derived
            //   for |grad phi| = 1 with phi in metres, and clamping flattens the
            //   field outside the band so the front eventually advances into
            //   ground carrying no gradient information.
            const Real sdf = dist - radius;
            p(i,j,k) = normalized
                     ? amrex::max(amrex::min(sdf / radius, 1.0_rt), -1.0_rt)
                     : sdf;
        });
    }
}

/**
 * @brief Temperature-threshold ignition.
 *
 * Every burnable, unburned fire cell (phi >= 0, not in the non-burnable mask)
 * whose near-surface air temperature exceeds T_thr is ignited as a disc of
 * radius `radius` around its centre, stamped with the min() semantics of the
 * scheduled ignitions so existing fire is never overwritten. The stamp is the
 * signed distance to the disc, in metres on the level-set path and divided
 * by the radius and clamped to [-1, 1] on the FARSITE path, matching
 * initialize_ignition(). The hot cell itself gets -radius (or -1).
 *
 * The temperature field is whatever the caller maps onto the fire grid; the
 * fire layer passes the k = 0 potential temperature it already holds for the
 * fuel-moisture update, so in a coupled run the fire's own heating can start
 * new fire and in a one-way run only the initial atmosphere can.
 *
 * @param[in,out] phi         Level-set field on the fire grid
 * @param[in]     temp        Near-surface temperature on the fire grid [K]
 * @param[in]     nonburnable Non-burnable mask (> 0.5 blocks ignition), or nullptr
 * @param[in]     geom_fire   Fire grid geometry
 * @param[in]     T_thr       Ignition temperature [K]
 * @param[in]     radius      Disc radius [m] stamped around each hot cell (> 0)
 * @param[in]     normalized  FARSITE [-1, 1] convention when true, metres when false
 * @return Number of cells that went from unburned to burning, summed over ranks
 */
inline
amrex::Long apply_threshold_ignition(
    amrex::MultiFab& phi,
    const amrex::MultiFab& temp,
    const amrex::MultiFab* nonburnable,
    const amrex::Geometry& geom_fire,
    amrex::Real T_thr, amrex::Real radius,
    bool normalized)
{
    using namespace amrex;

    const auto dx = geom_fire.CellSizeArray();
    const Real dmin_cell = amrex::min(dx[0], dx[1]);
    const int ns = amrex::max(1, static_cast<int>(std::ceil(radius / dmin_cell)));

    // Pass 1: flag the hot, burnable, unburned cells, with enough ghost cells
    // for the stencil of pass 2 (filled across ranks and periodic faces, and
    // copied from the nearest interior cell outside a non-periodic face).
    MultiFab flag(phi.boxArray(), phi.DistributionMap(), 1, ns);
    flag.setVal(0.0_rt);
    const bool has_nb = (nonburnable != nullptr);
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const& f = flag.array(mfi);
        auto const& p = phi.const_array(mfi);
        auto const& T = temp.const_array(mfi);
        Array4<const Real> nb;
        if (has_nb) { nb = nonburnable->const_array(mfi); }
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const bool burnable = !has_nb || nb(i, j, k) < 0.5_rt;
            f(i, j, k) = (burnable && p(i, j, k) >= 0.0_rt && T(i, j, k) > T_thr) ? 1.0_rt : 0.0_rt;
        });
    }
    fire_fill_boundary(flag, geom_fire);

    // Pass 2: every cell within `radius` of a flagged cell takes the signed
    // distance to the nearest flagged centre, min-merged with the fire it
    // already carries. Cells that turn from unburned to burning are counted.
    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<long> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const& f = flag.const_array(mfi);
        auto const& p = phi.array(mfi);
        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                Real dmin = static_cast<Real>(1.0e30);
                for (int dj = -ns; dj <= ns; ++dj) {
                    for (int di = -ns; di <= ns; ++di) {
                        if (f(i + di, j + dj, k) > 0.5_rt) {
                            const Real ddx = static_cast<Real>(di) * dx[0];
                            const Real ddy = static_cast<Real>(dj) * dx[1];
                            dmin = amrex::min(dmin, std::sqrt(ddx * ddx + ddy * ddy));
                        }
                    }
                }
                if (dmin >= radius) { return {0L}; }
                const Real old   = p(i, j, k);
                const Real sdf   = dmin - radius;
                const Real stamp = normalized
                                 ? amrex::max(amrex::min(sdf / radius, 1.0_rt), -1.0_rt)
                                 : sdf;
                p(i, j, k) = amrex::min(old, stamp);
                return {(old >= 0.0_rt && p(i, j, k) < 0.0_rt) ? 1L : 0L};
            });
    }
    amrex::Long n_ignited = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceLongSum(n_ignited);
    return n_ignited;
}

#endif // ERF_FIRE_IGNITION_H
