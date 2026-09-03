#ifndef ERF_DIRECTIONAL_ROS_H
#define ERF_DIRECTIONAL_ROS_H

#include <AMReX_MultiFab.H>
#include <AMReX_Gpu.H>
#include <cmath>
#include <ERF_Rothermel.H>
#include <ERF_BehaveModel.H>
#include <ERF_MacArthurModel.H>
#include <ERF_CheneyGouldModel.H>
#include <ERF_LevelSetAdvection.H>

/**
 * @file ERF_DirectionalRos.H
 * @brief Direction-dependent rate of spread on the level-set path
 *
 * The level set propagates a front at the normal speed it is given, so the
 * shape of the fire follows from the rate of spread rather than from an imposed
 * template. Handed a single scalar rate, it grows a circle at the head-fire
 * rate: flanks and backing fire advance as fast as the head, which is the
 * behaviour of every rate-of-spread model on this path except Balbi with
 * erf.fire.balbi.directional.
 *
 * The fix is not to bolt on the FARSITE ellipse. That ellipse exists because a
 * Lagrangian marker method has to be told how fast each marker moves relative to
 * the wind, and its length-to-width ratio is an empirical function of midflame
 * wind speed — a stand-in for a flow field that ERF resolves. Imposing it on top
 * of a resolved wind is the same double counting as applying the FARSITE terrain
 * wind factors over resolved terrain flow.
 *
 * Instead the driving vectors are projected onto the front normal
 * n = grad(phi)/|grad(phi)| and the selected model is evaluated with the
 * projected scalars:
 *
 *   R(n) = model( max(U . n, 0), max(grad(z) . n, 0) )
 *
 * At the head n is aligned with the wind and the model sees the full wind and
 * upslope, so R is the head rate. On the flanks the projections vanish and R
 * falls back to the model's no-wind, no-slope rate. Backing is the same, since
 * a wind blowing out of the unburned fuel does not drive the front into it, and
 * Rothermel's slope factor is quadratic and so cannot represent downslope
 * retardation; both are clamped at zero rather than reversed.
 *
 * Two limits of this are worth knowing. The projection reproduces neither the
 * saturation of the observed length-to-width ratio, which the Anderson fit caps
 * at 8, nor a backing rate below the no-wind rate, both of which the empirical
 * ellipse carries as calibration. And making R depend on the normal turns the
 * level-set equation into a general Hamilton-Jacobi problem, while the Godunov
 * upwinding used for |grad(phi)| is derived for a direction-independent speed;
 * freezing R(n) from central differences at each Runge-Kutta stage, as here, is
 * the usual practical approximation rather than the correct anisotropic flux.
 *
 * Balbi has its own directional path in ERF_BalbiModel.H, which carries the
 * per-cell moisture, surface temperature and heat-flux couplings that model
 * supports; this covers the remaining models.
 */

/**
 * @struct DirectionalRosState
 * @brief Precomputed coefficients for whichever ROS model is selected
 *
 * Trivially copyable so it can be captured by a device lambda. Only the member
 * matching `model` is read.
 */
struct DirectionalRosState
{
    /// 0 = Rothermel, 1 = BEHAVE, 2 = MacArthur, 3 = Cheney-Gould
    int model = 0;

    RothermelComputed   rc{};
    BehaveState         bs{};
    CheneyGouldComputed cgc{};

    /// Cheney-Gould inputs, which its per-cell function takes directly
    amrex::Real cg_moisture = 10.0;   ///< Dead fine fuel moisture [%]
    amrex::Real cg_curing   = 1.0;    ///< Degree of curing [0-1]
};

/// Model selectors, matching DirectionalRosState::model
inline constexpr int DIRECTIONAL_ROS_ROTHERMEL    = 0;
inline constexpr int DIRECTIONAL_ROS_BEHAVE       = 1;
inline constexpr int DIRECTIONAL_ROS_MACARTHUR    = 2;
inline constexpr int DIRECTIONAL_ROS_CHENEY_GOULD = 3;

/// |grad(phi)| below which the front normal is taken as undefined and the
/// isotropic wind and slope magnitudes are used instead.
inline constexpr amrex::Real DIRECTIONAL_ROS_GRAD_MIN = 1.0e-12;

/**
 * @brief Rate of spread along one direction
 *
 * @param[in] st   Precomputed coefficients for the selected model
 * @param[in] U_n  Wind component along the spread direction [m/s]
 * @param[in] s_n  Terrain slope along the spread direction [-]
 * @return ROS [m/s]
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real directional_ros_cell(const DirectionalRosState& st,
                                 amrex::Real U_n,
                                 amrex::Real s_n) noexcept
{
    // Only a wind pushing into the unburned fuel drives the front, and only an
    // upslope component enhances it; the models take magnitudes, so the
    // projections are clamped rather than passed signed.
    const amrex::Real U = amrex::max(U_n, amrex::Real(0.0));
    const amrex::Real s = amrex::max(s_n, amrex::Real(0.0));

    switch (st.model) {
        case DIRECTIONAL_ROS_BEHAVE:
            return behave_ros_cell(U, amrex::Real(0.0), s, amrex::Real(0.0), st.bs);
        case DIRECTIONAL_ROS_MACARTHUR:
            return macarthur_ros(U);
        case DIRECTIONAL_ROS_CHENEY_GOULD:
            return cheney_gould_ros(U, st.cgc.ros_backing, st.cg_moisture, st.cg_curing);
        default:
            return rothermel_ros_cell(U, amrex::Real(0.0), s, amrex::Real(0.0), st.rc);
    }
}

/**
 * @brief Fill a direction-dependent ROS field from the front normal
 *
 * Cells away from the front, where |grad(phi)| underflows, fall back to the
 * isotropic magnitudes so the field stays defined everywhere.
 *
 * @param[out] R_mf    Directional ROS [m/s]
 * @param[in]  phi     Level-set field, needs one ghost cell
 * @param[in]  wind    Effective wind [m/s]
 * @param[in]  slopes  Terrain slopes [dz/dx, dz/dy]
 * @param[in]  geom    Fire-grid geometry
 * @param[in]  st      Precomputed coefficients for the selected model
 */
inline void fill_directional_ros(amrex::MultiFab& R_mf,
                                 const amrex::MultiFab& phi,
                                 const amrex::MultiFab& wind,
                                 const amrex::MultiFab& slopes,
                                 const amrex::Geometry& geom,
                                 const DirectionalRosState& st)
{
    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dy = geom.CellSize(1);

    for (amrex::MFIter mfi(R_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto R_arr      = R_mf.array(mfi);
        auto phi_arr    = phi.const_array(mfi);
        auto wind_arr   = wind.const_array(mfi);
        auto slopes_arr = slopes.const_array(mfi);

        constexpr amrex::Real grad_min = DIRECTIONAL_ROS_GRAD_MIN;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            const int i = iv[0];
            const int j = iv[1];
            const int k = iv[2];

            const amrex::Real u = wind_arr(i, j, k, 0);
            const amrex::Real v = wind_arr(i, j, k, 1);
            const amrex::Real dzdx = slopes_arr(i, j, k, 0);
            const amrex::Real dzdy = slopes_arr(i, j, k, 1);

            // Front normal from central differences; it points toward increasing
            // phi, that is into the unburned fuel.
            const amrex::Real dpdx = (phi_arr(i+1, j, k) - phi_arr(i-1, j, k)) / (2.0 * dx);
            const amrex::Real dpdy = (phi_arr(i, j+1, k) - phi_arr(i, j-1, k)) / (2.0 * dy);
            const amrex::Real grad = std::sqrt(dpdx*dpdx + dpdy*dpdy);

            amrex::Real U_n, s_n;
            if (grad > grad_min) {
                const amrex::Real nx = dpdx / grad;
                const amrex::Real ny = dpdy / grad;
                U_n = u*nx + v*ny;
                s_n = dzdx*nx + dzdy*ny;
            } else {
                U_n = std::sqrt(u*u + v*v);
                s_n = std::sqrt(dzdx*dzdx + dzdy*dzdy);
            }

            R_arr(i, j, k) = directional_ros_cell(st, U_n, s_n);
        });
    }
}

/**
 * @brief Advect the level set with a direction-dependent ROS
 *
 * Same WENO5-Z / SSP-RK3 scheme as advect_levelset_weno5z_rk3(), except that
 * the ROS field is rebuilt from the front normal at the start of every RK stage
 * so R follows the front as it turns.
 *
 * @param[in,out] phi       Level-set field
 * @param[in]     wind      Effective wind [m/s]
 * @param[in]     slopes    Terrain slopes [dz/dx, dz/dy]
 * @param[in]     geom      Fire-grid geometry
 * @param[in]     dt        Timestep [s]
 * @param[in]     eps_visc  Artificial viscosity coefficient
 * @param[in]     st        Precomputed coefficients for the selected model
 */
inline void advect_levelset_directional_rk3(amrex::MultiFab& phi,
                                            const amrex::MultiFab& wind,
                                            const amrex::MultiFab& slopes,
                                            const amrex::Geometry& geom,
                                            amrex::Real dt,
                                            amrex::Real eps_visc,
                                            const DirectionalRosState& st)
{
    using namespace amrex;

    const auto* dx = geom.CellSize();

    const int nghost = 3;
    MultiFab phi_n(phi.boxArray(), phi.DistributionMap(), 1, nghost);
    MultiFab phi_star(phi.boxArray(), phi.DistributionMap(), 1, nghost);
    MultiFab phi_hat(phi.boxArray(), phi.DistributionMap(), 1, nghost);
    MultiFab rhs(phi.boxArray(), phi.DistributionMap(), 1, 0);
    MultiFab R_dir(phi.boxArray(), phi.DistributionMap(), 1, 0);

    phi_n.ParallelCopy(phi);
    phi_n.FillBoundary(geom.periodicity());

    // Stage 1: phi* = phi^n + dt L(phi^n)
    phi_star.ParallelCopy(phi_n);
    phi_star.FillBoundary(geom.periodicity());
    fill_directional_ros(R_dir, phi_star, wind, slopes, geom, st);
    fire_levelset::compute_levelset_rhs(rhs, phi_star, R_dir, dx[0], dx[1], eps_visc, &slopes);
    for (MFIter mfi(phi_star); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto ps = phi_star.array(mfi);
        auto r  = rhs.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            ps(iv) = ps(iv) + dt * r(iv);
        });
    }
    phi_star.FillBoundary(geom.periodicity());

    // Stage 2: phi** = (3/4)phi^n + (1/4)(phi* + dt L(phi*))
    fill_directional_ros(R_dir, phi_star, wind, slopes, geom, st);
    fire_levelset::compute_levelset_rhs(rhs, phi_star, R_dir, dx[0], dx[1], eps_visc, &slopes);
    phi_hat.ParallelCopy(phi_n);
    for (MFIter mfi(phi_hat); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto ph = phi_hat.array(mfi);
        auto ps = phi_star.const_array(mfi);
        auto r  = rhs.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            ph(iv) = 0.75_rt * ph(iv) + 0.25_rt * (ps(iv) + dt * r(iv));
        });
    }
    phi_hat.FillBoundary(geom.periodicity());

    // Stage 3: phi^{n+1} = (1/3)phi^n + (2/3)(phi** + dt L(phi**))
    fill_directional_ros(R_dir, phi_hat, wind, slopes, geom, st);
    fire_levelset::compute_levelset_rhs(rhs, phi_hat, R_dir, dx[0], dx[1], eps_visc, &slopes);
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto pnew = phi.array(mfi);
        auto ph   = phi_hat.const_array(mfi);
        auto r    = rhs.const_array(mfi);
        auto pn   = phi_n.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            pnew(iv) = (1.0_rt/3.0_rt) * pn(iv) + (2.0_rt/3.0_rt) * (ph(iv) + dt * r(iv));
        });
    }
}

#endif
