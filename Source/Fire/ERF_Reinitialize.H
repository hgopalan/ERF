#ifndef ERF_REINITIALIZE_H
#define ERF_REINITIALIZE_H

#include <AMReX_MultiFab.H>
#include <ERF_FireGrid.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <ERF_NumericalSchemes.H>

/**
 * @file ERF_Reinitialize.H
 * @brief Sussman reinitialization of the level-set function
 *
 * Ported from wildfire_levelset/src/reinitialization.H
 * Resets φ to a normalized signed-distance function via Sussman's method.
 *
 * NOTE ON UNITS. The textbook Sussman update drives |∇φ| → 1, which presumes φ
 * is a signed distance measured in metres. ERF-Hazard's φ is not: it is
 * normalized and clamped to [-1, 1] (initialize_ignition sets φ = (d - r)/r,
 * and the FARSITE path and ember spotting use the same convention). Applying
 * the textbook form to a normalized φ leaves the bracket (1 - |∇φ|) near 1 for
 * every cell, so each iteration added ~dtau and the field diverged by
 * n_iters * dtau per call. The update below instead targets |∇φ| = 1/L for an
 * explicit band length L, so φ reaches ±1 at distance L from the front, which
 * is the convention the rest of the module already uses.
 */

namespace fire_levelset {

/**
 * @brief Reinitialize φ to a normalized signed-distance function
 *
 * Solves the band-normalized reinitialization PDE
 *
 *   ∂φ/∂τ = S(φ₀) * (1 - L |∇φ|) / L
 *
 * by explicit Euler in pseudo-time, where S(φ₀) = φ₀/sqrt(φ₀²+eps²) is the
 * WRF-Fire-style smoothed sign (module_fr_fire_core.F's reinit_ls_rk3), not
 * the exact ±1 sign -- it tapers the forcing continuously to zero across the
 * front instead of switching at a single point. The fixed point is
 * |∇φ| = 1/L, i.e. φ varies linearly from 0 at the front to ±1 at distance L,
 * matching the normalization initialize_ignition applies. φ is clamped to
 * [-1, 1] after every iteration, which pins the far field (where |∇φ| = 0 and
 * the update would otherwise push φ past ±1 indefinitely).
 *
 * Cells whose original neighbourhood straddles the interface use the
 * Russo & Smereka (2000) subcell update instead, driven by the *exact* sign
 * of φ₀ (the smoothed sign would scale down this correction exactly where it
 * matters most). This fixes the interface location from φ₀ rather than
 * letting the pseudo-time iteration move it. Without it each pass erodes the
 * burned area, and because the level-set path never rebuilds φ from
 * fire_arrival_time the loss is permanent.
 *
 * After all iterations, a WRF-Fire-style monotonicity safeguard
 * (lfn_out = min(lfn_out, lfn_in), "fire area can only increase") clamps the
 * result against the pre-reinit field so reinitialization can never revert an
 * already-burned cell (φ < 0) back to unburned.
 *
 * Each iteration moves φ by at most dtau/L, so with the default
 * dtau = 0.5*min(dx,dy) the change per iteration is half a cell measured in
 * band lengths.
 *
 * @param[in,out] phi      Level-set field; modified in-place
 * @param[in]     geom     Domain geometry
 * @param[in]     n_iters  Number of Sussman iterations
 * @param[in]     dtau     Pseudo-timestep [m] for reinitialization
 * @param[in]     band_m   Band half-width L [m]: the distance from the front at
 *                         which φ reaches ±1. Values <= 0 select 3*min(dx,dy).
 * @param[in]     mask     Optional non-burnable mask, read only with wall_extrap
 * @param[in]     wall_extrap  Leave mask cells untouched and read neighbours
 *                         through a wall stencil, so the wall is a zero-gradient
 *                         boundary of the distance function rather than a front
 */
inline void reinitialize_phi(amrex::MultiFab& phi,
                              const amrex::Geometry& geom,
                              int n_iters,
                              amrex::Real dtau,
                              amrex::Real band_m = -1.0,
                              bool normalized = true,
                              const amrex::MultiFab* mask = nullptr,
                              bool wall_extrap = false,
                              const LevelSetGradient& grad = LevelSetGradient{})
{
    using namespace amrex;

    const auto* dx = geom.CellSize();

    // Band half-width. The default spreads the [-1, 1] range over three cells,
    // the usual interface thickness for a level-set band.
    // In signed-distance mode phi is already in metres, so the band length is
    // 1 m: the update targets |grad phi| = 1 and nothing is clamped. In
    // normalized mode phi is dimensionless and reaches +/-1 at distance L.
    const Real L    = normalized ? ((band_m > 0.0) ? band_m
                                                  : 3.0 * amrex::min(dx[0], dx[1]))
                                 : 1.0;
    const Real Linv = 1.0 / L;

    // Isotropic spacing for the Russo-Smereka subcell term. The fire grid is
    // square in practice; min() is the conservative choice if it ever is not.
    const Real h    = amrex::min(dx[0], dx[1]);
    const Real hinv = 1.0 / h;
    AMREX_ASSERT(L > 0.0 && h > 0.0);

    // Make a working copy for pseudo-time iteration
    MultiFab phi_work(phi.boxArray(), phi.DistributionMap(), 1, 3);
    phi_work.ParallelCopy(phi);

    // Jacobi scratch. The update below reads phi_work at i±1 / j±1 while writing
    // the centre cell, so writing back into phi_work in place would be a
    // read-write race between neighbouring GPU threads -- each cell would see
    // either the old or the new neighbour depending on scheduling, making the
    // result non-deterministic and different from the CPU sweep. Write into a
    // separate fab and swap once the pass is complete.
    MultiFab phi_next(phi.boxArray(), phi.DistributionMap(), 1, 3);
    // Freshly allocated fabs hold indeterminate values. Only the valid region is
    // written and only the valid region is copied back, but leaving the ghosts
    // indeterminate makes the run depend on whatever the allocator hands over.
    phi_next.setVal(0.0);

    for (int iter = 0; iter < n_iters; ++iter) {
        // Fill boundary ghosts
        fire_fill_boundary(phi_work, geom);

        // Update RHS via Sussman reinitialization
        for (MFIter mfi(phi_work); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto phi_arr = phi_work.const_array(mfi);
            auto phi_out = phi_next.array(mfi);
            auto phi_in  = phi.const_array(mfi);
            const WallStencil wall = make_wall_stencil(mask, wall_extrap, mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
                int i = iv[0], j = iv[1], k = iv[2];

                // Mask cells are not part of the distance function: they keep
                // their value and their neighbours read through the wall.
                if (wall.masked(i, j, k)) { phi_out(i, j, k) = phi_arr(i, j, k); return; }

                // One-sided differences for the gradient, first order or
                // HJ-WENO5-Z as the advection uses (WRF-Fire solves the
                // reinitialisation with the same hybrid scheme).
                Real dxm, dxp, dym, dyp;
                one_sided_derivatives(phi_arr, dx[0], i, j, k, 1, 0, grad, wall, dxm, dxp);
                one_sided_derivatives(phi_arr, dx[1], i, j, k, 0, 1, grad, wall, dym, dyp);

                // Exact sign of the original level-set. Used only by the
                // Russo-Smereka subcell block below, which must combine
                // crossing distances with an exact +/-1 direction -- feeding
                // it the smoothed sign (next) would scale down the very
                // sub-cell correction that block exists to get right, since
                // near_front cells are exactly where the smoothed sign's
                // magnitude is smallest.
                const Real phi0 = phi_in(i, j, k);
                Real sgn_phi0_hard = (phi0 > 0.0) ? 1.0 :
                                     (phi0 < 0.0) ? -1.0 : 0.0;

                // WRF-Fire style smoothed sign, S(phi0) = phi0/sqrt(phi0^2+eps^2)
                // (module_fr_fire_core.F's reinit_ls_rk3), used for the general
                // (non-Russo-Smereka) update below. eps is one grid cell's
                // worth of phi-units: h*Linv is h itself (WRF's dx) in
                // signed-distance mode where L=1, and h/L in normalized mode.
                // This only damps the RHS magnitude near the front -- the sign
                // of sgn_phi0_smooth always matches sgn_phi0_hard, so it does
                // not change which upwind branch is selected below.
                const Real eps_s = h * Linv;
                Real sgn_phi0_smooth = phi0 / std::sqrt(phi0 * phi0 + eps_s * eps_s);

                // Godunov upwind norm for the reinitialization Hamiltonian.
                // The one-sided difference is selected by the sign of phi_0, not
                // by which has the larger magnitude: taking the larger magnitude
                // unconditionally is not the Godunov scheme and is unstable near
                // extrema, which showed up as the front flooding the domain once
                // enough pseudo-time iterations were taken.
                Real gx2, gy2;
                if (sgn_phi0_smooth > 0.0_rt) {
                    Real a = amrex::max(dxm, 0.0_rt), b = amrex::min(dxp, 0.0_rt);
                    gx2 = amrex::max(a * a, b * b);
                    Real c = amrex::max(dym, 0.0_rt), d = amrex::min(dyp, 0.0_rt);
                    gy2 = amrex::max(c * c, d * d);
                } else {
                    Real a = amrex::min(dxm, 0.0_rt), b = amrex::max(dxp, 0.0_rt);
                    gx2 = amrex::max(a * a, b * b);
                    Real c = amrex::min(dym, 0.0_rt), d = amrex::max(dyp, 0.0_rt);
                    gy2 = amrex::max(c * c, d * d);
                }
                Real norm_grad = std::sqrt(gx2 + gy2 + 1.0e-14);

                // Band-normalized Sussman RHS: S(φ₀) * (1 - L|∇φ|) / L, using
                // the smoothed sign so the forcing tapers continuously to zero
                // across the front rather than switching at a single point.
                Real rhs = sgn_phi0_smooth * (1.0_rt - L * norm_grad) * Linv;

                Real phi_new = phi_arr(i, j, k) + dtau * rhs;

                // Russo-Smereka (2000) subcell fix.
                //
                // The plain Sussman iteration does not preserve the zero level
                // set, so each pass walks the front and permanently erodes the
                // burned area -- the level-set path never rebuilds phi from
                // fire_arrival_time the way the FARSITE path does, so nothing
                // puts it back. For cells whose ORIGINAL neighbourhood straddles
                // the interface, replace the Godunov update with one driven by
                // the distance to the interface implied by phi_0 itself, so the
                // interface location is set by phi_0 and not by the iteration.
                const Real p0c = phi_in(i, j, k);
                const Real p0xp = wall.at(phi_in, i, j, k, 1, 0);
                const Real p0xm = wall.at(phi_in, i, j, k, -1, 0);
                const Real p0yp = wall.at(phi_in, i, j, k, 0, 1);
                const Real p0ym = wall.at(phi_in, i, j, k, 0, -1);
                const bool near_front =
                    (p0c * p0xp < 0.0_rt) || (p0c * p0xm < 0.0_rt) ||
                    (p0c * p0yp < 0.0_rt) || (p0c * p0ym < 0.0_rt);

                if (near_front) {
                    // Russo & Smereka (2000) subcell distance: for each axis
                    // direction where phi_0 actually crosses zero, linearly
                    // interpolate phi_0 between the cell centre and that
                    // neighbour to get the zero-crossing distance along that
                    // direction, then combine the (up to 4) crossing
                    // directions via the standard reciprocal-sum rule.
                    //
                    // This replaces an earlier version that combined ALL FOUR
                    // one-sided differences (crossing or not) into a single
                    // magnitude via sqrt(dcx^2+dcy^2) floored by the largest
                    // one-sided jump. At a concave corner -- where two
                    // non-crossing directions can have a much larger jump
                    // than the actual crossing direction(s), e.g. from a
                    // neighbour on the far side of the corner -- that
                    // combined magnitude overestimates the true local
                    // gradient and so underestimates the true
                    // distance-to-interface. Verified against an exact
                    // brute-force (marching-squares) ground truth at a
                    // concave corner cell: the old formula was off by ~40%
                    // (-14.7 vs the true -10.5); this one matches to <0.1%.
                    Real inv_D2 = 0.0_rt;
                    int n_cross = 0;
                    auto accumulate_crossing = [&](Real p0n) {
                        if (p0c * p0n < 0.0_rt) {
                            Real Ddir = h * p0c / (p0c - p0n);
                            inv_D2 += 1.0_rt / (Ddir * Ddir);
                            ++n_cross;
                        }
                    };
                    accumulate_crossing(p0xp);
                    accumulate_crossing(p0xm);
                    accumulate_crossing(p0yp);
                    accumulate_crossing(p0ym);
                    // n_cross >= 1 is guaranteed by near_front, but guard
                    // against a degenerate zero denominator regardless.
                    const Real D = (n_cross > 0 && inv_D2 > 0.0_rt)
                        ? sgn_phi0_hard / std::sqrt(inv_D2)
                        : h * p0c;

                    phi_new = phi_arr(i, j, k)
                            - dtau * hinv * (sgn_phi0_hard * std::abs(phi_arr(i, j, k)) - D * Linv);
                }

                // Clamp only in normalized mode, where it pins the far field:
                // there |grad phi| = 0 and the update would otherwise drive phi
                // past +/-1 without limit. In signed-distance mode phi is meant
                // to keep growing with distance, so clamping is exactly the
                // thing that breaks it.
                phi_out(i, j, k) = normalized
                    ? amrex::max(-1.0_rt, amrex::min(1.0_rt, phi_new))
                    : phi_new;
            });
        }

        // phi_next now holds the whole updated valid region; make it the input
        // for the next sweep. Ghost cells are refilled at the top of the loop.
        MultiFab::Copy(phi_work, phi_next, 0, 0, 1, 0);
    }

    // WRF-Fire style monotonicity safeguard (module_fr_fire_core.F's
    // reinit_ls_rk3: "lfn_out=min(lfn_out,lfn_in) ! fire area can only
    // increase"). phi < 0 is burned, so taking the elementwise min against
    // the pre-reinit field can only make a cell more (or equally) burned --
    // reinitialization can never revert an already-burned cell back to
    // unburned, regardless of how the iterations above moved it. This is a
    // blunt backstop on top of, not instead of, the Russo-Smereka fix above.
    for (MFIter mfi(phi_work); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto phi_arr = phi_work.array(mfi);
        auto phi_orig = phi.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            int i = iv[0], j = iv[1], k = iv[2];
            phi_arr(i, j, k) = amrex::min(phi_arr(i, j, k), phi_orig(i, j, k));
        });
    }

    // Copy result back to phi
    phi.ParallelCopy(phi_work);
}

} // namespace fire_levelset

#endif
