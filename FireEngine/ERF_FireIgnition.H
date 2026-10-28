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
 * Stamps the signed distance to the disc edge, sdf = dist(x,y; cx,cy) - r:
 * with normalized = true (FARSITE) sdf / r clamped to [-1, 1], with
 * normalized = false (level set) sdf in metres, unclamped (see the comment in
 * the kernel). The caller passes the centre and radius it read from
 * erf.fire.ignition_x/y/r; nothing is read from ParmParse here.
 *
 * @param[in,out] phi     Level-set field to initialize
 * @param[in]     geom_fire Fire grid geometry
 * @param[in]     cx      Ignition center x [m]
 * @param[in]     cy      Ignition center y [m]
 * @param[in]     radius  Ignition radius [m]; <= 0 means no disc
 * @param[in]     normalized  true: sdf / r clamped to [-1, 1]; false: sdf in metres
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
    // "No disc" on the level-set path: every cell the domain diagonal from a
    // front, a finite distance the reinitialisation and the advection keep
    // where it is (the advection moves nothing on a uniform field, and the
    // reinitialisation ends with min(new, old), so it cannot raise it). The
    // distance to the ignition point (the form until 2026-10) was a disc of
    // zero radius that the advection grew from the first step. A scheduled
    // or polygon ignition min-merges a whole-grid distance field. A
    // threshold ignition stamps its disc plus a distance band
    // (apply_threshold_ignition), so the field steps from that band to this
    // value; the upwind scheme carries the step outward as a smeared step
    // that the front catches up with late, by a lag that grows slowly with
    // the distance covered and with ln(far / dx) (
    // about 20 to 60 steps over 1 to 40 cells in a sweep of the scheme). The
    // distance to a fire far away gives the same step, so spot landings and
    // threshold ignitions lagged this way before 2026-10 too; a whole-grid
    // distance merge for them would remove it.
    // The full diagonal, never capped: an ignition's distance field
    // min-merged over it then keeps |grad phi| = 1 everywhere in the domain
    // (a cap left a flat plateau beyond it on a wider grid). The firebreak
    // mask marks its cells from the barrier shapes (mark_firebreak_cells in
    // ERF_FireBreak.H), not from large values of phi.
    const Real far = std::sqrt(geom_fire.ProbLength(0) * geom_fire.ProbLength(0)
                             + geom_fire.ProbLength(1) * geom_fire.ProbLength(1));

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
            //   Hamiltonian and the reinitialisation are all derived
            //   for |grad phi| = 1 with phi in metres, and clamping flattens the
            //   field outside the band so the front eventually advances into
            //   ground carrying no gradient information.
            const Real sdf = dist - radius;
            // radius <= 0 is "no disc" (the schedule, polygon, line or threshold ignites
            // instead): unburned everywhere, and no division by the radius.
            // (floored radius inside the division: the unselected sdf/0 of
            // the "no disc" case would be speculated under the fpe traps)
            const Real sdf_n = amrex::max(amrex::min(sdf / amrex::max(radius, 1.0e-30_rt), 1.0_rt), -1.0_rt);
            p(i,j,k) = normalized
                     ? ((radius > 0.0_rt) ? sdf_n : 1.0_rt)
                     : ((radius > 0.0_rt) ? sdf : far);
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
 * fire layer passes the k = 0 air temperature (theta times the Exner function,
 * compute_t_from_conservative) it already holds for the fuel-moisture update,
 * so in a coupled run the fire's own heating can start new fire and in a
 * one-way run only the initial atmosphere can.
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
    // On the level-set path the stamp also writes the (positive) distance to
    // the hot cell over a band beyond the disc. A stamp of the disc alone
    // left its cells a deep local minimum in whatever phi the neighbours held
    // (the no-disc start, the domain diagonal, or the distance to a fire far
    // away): the artificial viscosity reads that jump as a Laplacian, a lift
    // of 1.6 eps phi_bg / dx in one substep at CFL 0.4, and a one-cell
    // ignition went back above zero once phi_bg exceeded about
    // dx^2 / (1.6 eps) (5.6 km on 30 m cells, 160 m on 5 m cells), sooner over
    // a few substeps, so the fire went out. The band is three cells (the
    // reach of the WENO5 stencil and of the Laplacian) and at least 8 m: on
    // fine cells the viscosity carries the lift in over several cells before
    // the jump collapses (a three-cell band let a 1 m-cell ignition under a
    // 14 km background go out within 20 substeps; 8 m holds it, and in a
    // sweep of the scheme held to 1000 km backgrounds on 1 to 1.5 m cells).
    // The band cells stay unburned (only a crossing below zero counts as an
    // ignition); the FARSITE indicator has no gradient to keep and takes the
    // disc alone.
    const Real band = normalized ? 0.0_rt : amrex::max(3.0_rt * dmin_cell, Real(8.0));
    const int ns = amrex::max(1, static_cast<int>(std::ceil((radius + band) / dmin_cell)));

    // Pass 1: flag the hot, burnable, unburned cells, with enough ghost cells
    // in x and y for the stencil of pass 2 (filled across ranks and periodic
    // faces, and copied from the nearest interior cell outside a
    // non-periodic face); none in z, where the fire grid is one cell.
    MultiFab flag(phi.boxArray(), phi.DistributionMap(), 1, IntVect(AMREX_D_DECL(ns, ns, 0)));
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
    // Nothing hot anywhere (the usual step): no halo exchange and no stencil
    // pass. The sum is global, so every rank takes the same branch.
    if (flag.sum(0) < 0.5_rt) { return amrex::Long(0); }
    fire_fill_boundary(flag, geom_fire);

    // Pass 2: every cell within `radius` (plus the band on the level-set
    // path) of a flagged cell takes the signed distance to the nearest
    // flagged centre, min-merged with the fire it already carries. Cells that
    // turn from unburned to burning are counted.
    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<amrex::Long> reduce_data(reduce_op);   // amrex::Long: `long` is 32-bit on MSVC
    using ReduceTuple = typename decltype(reduce_data)::Type;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const& f = flag.const_array(mfi);
        auto const& p = phi.array(mfi);
        Array4<const Real> nb2;
        if (has_nb) { nb2 = nonburnable->const_array(mfi); }
        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                // cells in the non-burnable mask take no stamp
                if (has_nb && nb2(i, j, k) >= 0.5_rt) { return {amrex::Long(0)}; }
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
                if (dmin >= radius + band) { return {amrex::Long(0)}; }
                const Real old   = p(i, j, k);
                const Real sdf   = dmin - radius;
                // floored radius (threshold_radius = 0 means one cell and the
                // unselected sdf/0 would be speculated under the fpe traps)
                const Real stamp = normalized
                                 ? amrex::max(amrex::min(sdf / amrex::max(radius, 1.0e-30_rt), 1.0_rt), -1.0_rt)
                                 : sdf;
                p(i, j, k) = amrex::min(old, stamp);
                return {(old >= 0.0_rt && p(i, j, k) < 0.0_rt) ? amrex::Long(1) : amrex::Long(0)};
            });
    }
    amrex::Long n_ignited = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceLongSum(n_ignited);
    return n_ignited;
}

#endif // ERF_FIRE_IGNITION_H
