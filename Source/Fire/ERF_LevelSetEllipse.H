#ifndef ERF_LEVELSET_ELLIPSE_H
#define ERF_LEVELSET_ELLIPSE_H

#include <AMReX_MultiFab.H>
#include <AMReX_Gpu.H>
#include <cmath>
#include <ERF_LevelSetAdvection.H>
#include <ERF_FarsiteEllipse.H>

/**
 * @file ERF_LevelSetEllipse.H
 * @brief Wind-driven spread ellipse on the level-set path (optional)
 *
 * The FARSITE family (Finney 1998; also ELMFIRE, Prometheus, Cell2Fire)
 * shapes a fire with Huygens' principle: every point of the front spreads as
 * a small ellipse whose length-to-width ratio LB is an empirical function of
 * the midflame wind (Anderson 1983), whose head-to-back ratio follows from it
 * (Alexander 1985),
 *
 *   HB = (LB + sqrt(LB^2 - 1)) / (LB - sqrt(LB^2 - 1)),
 *
 * and whose axes are set by the head rate R:
 *
 *   b = (R + R/HB) / 2   (semi-major rate, along the wind)
 *   a = b / LB           (semi-minor rate, across the wind)
 *   c = b - R/HB         (offset of the ellipse centre from the ignition point)
 *
 * so that the head runs at b + c = R, the back at b - c = R/HB and the flanks
 * at a. The level set needs the normal speed of that envelope, which is the
 * support function of the ellipse: for an outward normal at angle theta to the
 * wind,
 *
 *   R_n(theta) = c cos(theta) + sqrt(b^2 cos^2(theta) + a^2 sin^2(theta)).
 *
 * This is the Richards (1990) ellipse written as a normal speed, the form the
 * level-set implementations of FARSITE-like models use. With no wind the
 * ellipse is a circle of radius R.
 *
 * The projection-based directional spread in ERF_DirectionalRos.H is the
 * default way to get head, flank and backing rates on this path, and its
 * header explains why an imposed ellipse double counts a resolved wind. The
 * ellipse is offered as an option all the same: it reproduces the saturation
 * of the observed length-to-width ratio and a backing rate below the no-wind
 * rate, both of which the projection lacks, and it is what the other fire
 * models compute, so it is the right thing to run when comparing with them.
 * It is selected with erf.fire.levelset.ellipse and is off by default. The
 * same ellipse is the default spread shape of the FARSITE path
 * (erf.fire.farsite.shape, ERF_FarsiteEllipse.H).
 */

// SpreadEllipse, spread_ellipse() and ellipse_normal_speed() live in
// ERF_FarsiteEllipse.H, which the FARSITE path shares with this one.

/// Below this wind speed [m/s] the ellipse is a circle of radius R.
inline constexpr amrex::Real ELLIPSE_WIND_MIN = 1.0e-3;
/// |grad(phi)| below this leaves the cell with the head rate.
inline constexpr amrex::Real ELLIPSE_GRAD_MIN = 1.0e-12;

/**
 * @brief Fill the ellipse normal speed from the front normal
 *
 * @param[out] R_mf     Normal speed [m/s]
 * @param[in]  phi      Level-set field, needs one ghost cell
 * @param[in]  wind     Effective (midflame) wind [m/s], two components
 * @param[in]  R_head   Head rate of spread [m/s] from the rate-of-spread model
 * @param[in]  geom     Fire-grid geometry
 * @param[in]  lw_fixed Length-to-width ratio when > 0, else Anderson from the wind
 * @param[in]  lw_max   Cap on the length-to-width ratio
 * @param[in]  mask     Optional non-burnable mask, read only with wall_extrap
 * @param[in]  wall_extrap  Take the front normal through a wall stencil
 */
inline void fill_ellipse_ros(amrex::MultiFab& R_mf,
                             const amrex::MultiFab& phi,
                             const amrex::MultiFab& wind,
                             const amrex::MultiFab& R_head,
                             const amrex::Geometry& geom,
                             amrex::Real lw_fixed,
                             amrex::Real lw_max,
                             const amrex::MultiFab* mask = nullptr,
                             bool wall_extrap = false)
{
    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dy = geom.CellSize(1);

    for (amrex::MFIter mfi(R_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto R_arr    = R_mf.array(mfi);
        auto phi_arr  = phi.const_array(mfi);
        auto wind_arr = wind.const_array(mfi);
        auto Rh_arr   = R_head.const_array(mfi);
        const fire_levelset::WallStencil wall =
            fire_levelset::make_wall_stencil(mask, wall_extrap, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            const int i = iv[0];
            const int j = iv[1];
            const int k = iv[2];

            const amrex::Real R = Rh_arr(i, j, k);
            const amrex::Real u = wind_arr(i, j, k, 0);
            const amrex::Real v = wind_arr(i, j, k, 1);
            const amrex::Real U = std::sqrt(u * u + v * v);
            if (R <= 0.0 || U < ELLIPSE_WIND_MIN) { R_arr(i, j, k) = amrex::max(R, amrex::Real(0.0)); return; }

            // Front normal from central differences, pointing into the unburned fuel.
            const amrex::Real dpdx = (wall.at(phi_arr, i, j, k, 1, 0)
                                      - wall.at(phi_arr, i, j, k, -1, 0)) / (2.0 * dx);
            const amrex::Real dpdy = (wall.at(phi_arr, i, j, k, 0, 1)
                                      - wall.at(phi_arr, i, j, k, 0, -1)) / (2.0 * dy);
            const amrex::Real grad = std::sqrt(dpdx * dpdx + dpdy * dpdy);
            // unit vectors from floored magnitudes, before the early return
            // (the unselected 0/0 away from the front is speculated)
            const amrex::Real inv_grad = amrex::Real(1.0) / amrex::max(grad, amrex::Real(ELLIPSE_GRAD_MIN));
            const amrex::Real inv_U    = amrex::Real(1.0) / amrex::max(U,    amrex::Real(ELLIPSE_WIND_MIN));
            if (grad <= ELLIPSE_GRAD_MIN) { R_arr(i, j, k) = R; return; }

            const amrex::Real nx = dpdx * inv_grad, ny = dpdy * inv_grad;
            const amrex::Real wx = u * inv_U,       wy = v * inv_U;
            const amrex::Real cos_th = nx * wx + ny * wy;
            const amrex::Real sin_th = nx * wy - ny * wx;
            const SpreadEllipse e = spread_ellipse(R, U, lw_fixed, lw_max);
            R_arr(i, j, k) = ellipse_normal_speed(e, cos_th, sin_th);
        });
    }
}

/**
 * @brief Advect the level set at the ellipse normal speed
 *
 * Same Godunov / SSP-RK3 scheme as the isotropic and directional paths; the
 * speed is rebuilt from the front normal at every RK stage.
 */
inline void advect_levelset_ellipse_rk3(amrex::MultiFab& phi,
                                        const amrex::MultiFab& wind,
                                        const amrex::MultiFab& R_head,
                                        const amrex::MultiFab& slopes,
                                        const amrex::Geometry& geom,
                                        amrex::Real dt,
                                        amrex::Real eps_visc,
                                        amrex::Real lw_fixed,
                                        amrex::Real lw_max,
                                        const amrex::MultiFab* nonburnable = nullptr,
                                        bool wall_extrap = false,
                                        const fire_levelset::LevelSetGradient& grad_scheme = fire_levelset::LevelSetGradient{},
                                        fire_levelset::Scratch* scratch = nullptr)
{
    fire_levelset::advect_levelset_rk3_with_fill(phi, slopes, geom, dt, eps_visc,
        [&](amrex::MultiFab& R, const amrex::MultiFab& phi_stage) {
            fill_ellipse_ros(R, phi_stage, wind, R_head, geom, lw_fixed, lw_max, nonburnable, wall_extrap);
        }, nonburnable, wall_extrap, grad_scheme, nullptr, scratch);
}

#endif
