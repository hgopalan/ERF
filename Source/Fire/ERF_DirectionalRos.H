#ifndef ERF_DIRECTIONAL_ROS_H
#define ERF_DIRECTIONAL_ROS_H

#include <AMReX_MultiFab.H>
#include <AMReX_Gpu.H>
#include <cmath>
#include <ERF_Rothermel.H>
#include <ERF_BehaveModel.H>
#include <ERF_MacArthurModel.H>
#include <ERF_CheneyGouldModel.H>
#include <ERF_FbpModel.H>
#include <ERF_LevelSetAdvection.H>
#include <ERF_LevelSetEllipse.H>

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
 * Three limits of this are worth knowing. The projection reproduces neither the
 * saturation of the observed length-to-width ratio, which the Anderson fit caps
 * at 8, nor a backing rate below the no-wind rate, both of which the empirical
 * ellipse carries as calibration. Making R depend on the normal turns the
 * level-set equation into a general Hamilton-Jacobi problem, while the Godunov
 * upwinding used for |grad(phi)| is derived for a direction-independent speed;
 * freezing R(n) from central differences at each Runge-Kutta stage, as here, is
 * the usual practical approximation rather than the correct anisotropic flux.
 *
 * And a normal speed is not a rate of spread in a direction. From a point, the
 * exact (viscosity) solution of phi_t + R(n) |grad(phi)| = 0 is the Wulff shape
 * {x : x . n <= t R(n) for every n}, whose extent along a direction d is the
 * minimum over n . d > 0 of R(n) / (n . d), not R(d); the two agree only when R
 * is the support function of a convex set. Rothermel's R0 (1 + phi_w (U n_x)^B)
 * is not once phi_w (B - 1) > 1, nor R0 (1 + phi_s (s n_x)^2) once phi_s > 1
 * (the terrain projection of |grad(phi)| adds a factor 1/sqrt(1 + (s . n)^2)
 * that changes little): the head of a point fire becomes a wedge of oblique
 * facets whose tip runs at R0 B/(B - 1) (phi_w (B - 1))^(1/B) in wind, about
 * 2 sqrt(phi_s) R0 on a slope, well below the head rate. A straight line fire
 * keeps n along the wind and is not affected. A better scheme would not help,
 * since it is the exact solution that falls short; this one lands between the
 * tip speed and the head rate (Exec/RegTests/FireDirectionalShape).
 *
 * erf.fire.directional_shape = "ellipse" (DIRECTIONAL_SHAPE_ELLIPSE, opt-in)
 * takes the normal speed instead from the support function of an ellipse, which
 * is convex and so its own Wulff shape, built from the same model:
 *
 *   head   R_h = R0 + g |dR_w w + dR_s s|, along that vector
 *   back   R0
 *   flank  R0
 *
 * with R0 = model(0, 0), dR_w = model(|U|, 0) - R0 and dR_s = model(0, |grad z|)
 * - R0 the wind and slope increments along the unit wind and upslope vectors w
 * and s, added as vectors the way BEHAVE adds Rothermel's wind and slope
 * factors, and g = (model(|U|, |grad z|) - R0) / (dR_w + dR_s), which is 1 for
 * the additive Rothermel form and makes aligned wind and slope give the model's
 * combined rate for the others. With aligned drivers the head, back and flanks
 * are the projection's own rates at those three normals, and the ellipse fills
 * in the directions between them convexly. Its semi-axes are b = (R_h + R0)/2
 * along the head and a = R0 across it, with the centre offset c = (R_h - R0)/2
 * (ERF_LevelSetEllipse.H has the support function), so the length-to-width ratio
 * is (R_h/R0 + 1)/2, uncapped like the projection's.
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
    /// 0 = Rothermel, 1 = BEHAVE, 2 = MacArthur, 3 = Cheney-Gould, 4 = FBP
    int model = 0;

    RothermelComputed   rc{};
    BehaveState         bs{};
    CheneyGouldComputed cgc{};
    FbpComputed         fbp{};

    /// Cheney-Gould inputs, which its per-cell function takes directly
    amrex::Real cg_moisture = 10.0;   ///< Dead fine fuel moisture [%]
    amrex::Real cg_curing   = 1.0;    ///< Degree of curing [0-1]
};

/// Model selectors, matching DirectionalRosState::model
inline constexpr int DIRECTIONAL_ROS_ROTHERMEL    = 0;
inline constexpr int DIRECTIONAL_ROS_BEHAVE       = 1;
inline constexpr int DIRECTIONAL_ROS_MACARTHUR    = 2;
inline constexpr int DIRECTIONAL_ROS_CHENEY_GOULD = 3;
inline constexpr int DIRECTIONAL_ROS_FBP          = 4;

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
        case DIRECTIONAL_ROS_FBP:
            return fbp_ros(st.fbp, U, s);
        default:
            return rothermel_ros_cell(U, amrex::Real(0.0), s, amrex::Real(0.0), st.rc);
    }
}

/// Shape of the direction-dependent rate, erf.fire.directional_shape
inline constexpr int DIRECTIONAL_SHAPE_PROJECTION = 0;  ///< the model evaluated along the front normal
inline constexpr int DIRECTIONAL_SHAPE_ELLIPSE    = 1;  ///< support function of the model's spread ellipse

/**
 * @struct DirectionalEllipse
 * @brief Spread ellipse built from the selected model, and the direction of its head
 */
struct DirectionalEllipse
{
    SpreadEllipse e{};       ///< head rate b + c, back b - c, flank a [m/s]
    amrex::Real hx = 1.0;    ///< unit vector along the head
    amrex::Real hy = 0.0;
};

/**
 * @brief Spread ellipse whose head, back and flank rates come from the model
 *
 * Head R0 + g |dR_w w + dR_s s| along that vector, back and flanks R0; see the
 * file comment.
 *
 * @param[in] st          Precomputed coefficients for the selected model
 * @param[in] u, v        Wind [m/s]
 * @param[in] dzdx, dzdy  Terrain slopes [-]
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
DirectionalEllipse directional_ellipse_cell(const DirectionalRosState& st,
                                            amrex::Real u, amrex::Real v,
                                            amrex::Real dzdx, amrex::Real dzdy) noexcept
{
    constexpr amrex::Real tiny = 1.0e-12;
    const amrex::Real zero = 0.0;
    const amrex::Real U = std::sqrt(u*u + v*v);
    const amrex::Real S = std::sqrt(dzdx*dzdx + dzdy*dzdy);

    const amrex::Real R0   = directional_ros_cell(st, zero, zero);
    const amrex::Real dR_w = amrex::max(directional_ros_cell(st, U, zero) - R0, zero);
    const amrex::Real dR_s = amrex::max(directional_ros_cell(st, zero, S) - R0, zero);
    const amrex::Real dR   = amrex::max(directional_ros_cell(st, U, S) - R0, zero);

    // The wind and slope increments as vectors along the wind and upslope
    amrex::Real px = 0.0, py = 0.0;
    if (U > tiny) { px += dR_w * u / U;    py += dR_w * v / U; }
    if (S > tiny) { px += dR_s * dzdx / S; py += dR_s * dzdy / S; }
    const amrex::Real P = std::sqrt(px*px + py*py);
    const amrex::Real g = (dR_w + dR_s > tiny) ? dR / (dR_w + dR_s) : amrex::Real(1.0);

    DirectionalEllipse d;
    if (P > tiny) { d.hx = px / P; d.hy = py / P; }
    const amrex::Real R_h = R0 + g * P;
    d.e.b  = amrex::Real(0.5) * (R_h + R0);
    d.e.c  = amrex::Real(0.5) * (R_h - R0);
    d.e.a  = R0;
    d.e.LB = (R0 > tiny) ? d.e.b / R0 : amrex::Real(1.0e12);
    d.e.HB = (R0 > tiny) ? R_h / R0   : amrex::Real(1.0e12);
    return d;
}

/// Normal speed of a DirectionalEllipse for the unit front normal (nx, ny).
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real directional_ellipse_speed(const DirectionalEllipse& d,
                                      amrex::Real nx, amrex::Real ny) noexcept
{
    return ellipse_normal_speed(d.e, nx * d.hx + ny * d.hy, nx * d.hy - ny * d.hx);
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
 * @param[in]  mask    Optional non-burnable mask, read only with wall_extrap
 * @param[in]  wall_extrap  Take the front normal through a wall stencil, so a
 *                     cell next to a masked wall sees a normal along the wall
 *                     rather than into it
 * @param[in]  shape   DIRECTIONAL_SHAPE_PROJECTION (the model along the normal)
 *                     or DIRECTIONAL_SHAPE_ELLIPSE (the model's spread ellipse)
 */
inline void fill_directional_ros(amrex::MultiFab& R_mf,
                                 const amrex::MultiFab& phi,
                                 const amrex::MultiFab& wind,
                                 const amrex::MultiFab& slopes,
                                 const amrex::Geometry& geom,
                                 const DirectionalRosState& st,
                                 const amrex::MultiFab* mask = nullptr,
                                 bool wall_extrap = false,
                                 int shape = DIRECTIONAL_SHAPE_PROJECTION)
{
    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dy = geom.CellSize(1);

    for (amrex::MFIter mfi(R_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto R_arr      = R_mf.array(mfi);
        auto phi_arr    = phi.const_array(mfi);
        auto wind_arr   = wind.const_array(mfi);
        auto slopes_arr = slopes.const_array(mfi);
        const fire_levelset::WallStencil wall =
            fire_levelset::make_wall_stencil(mask, wall_extrap, mfi);

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
            const amrex::Real dpdx = (wall.at(phi_arr, i, j, k, 1, 0)
                                      - wall.at(phi_arr, i, j, k, -1, 0)) / (2.0 * dx);
            const amrex::Real dpdy = (wall.at(phi_arr, i, j, k, 0, 1)
                                      - wall.at(phi_arr, i, j, k, 0, -1)) / (2.0 * dy);
            const amrex::Real grad = std::sqrt(dpdx*dpdx + dpdy*dpdy);

            if (shape == DIRECTIONAL_SHAPE_ELLIPSE) {
                // Support function of the model's spread ellipse, and its head
                // rate where the normal is undefined
                const DirectionalEllipse de = directional_ellipse_cell(st, u, v, dzdx, dzdy);
                R_arr(i, j, k) = (grad > grad_min)
                    ? directional_ellipse_speed(de, dpdx / grad, dpdy / grad)
                    : de.e.b + de.e.c;
                return;
            }

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
 * Same Godunov / SSP-RK3 scheme as advect_levelset_weno5z_rk3(), except that
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
 * @param[in]     shape     DIRECTIONAL_SHAPE_PROJECTION or DIRECTIONAL_SHAPE_ELLIPSE
 */
inline void advect_levelset_directional_rk3(amrex::MultiFab& phi,
                                            const amrex::MultiFab& wind,
                                            const amrex::MultiFab& slopes,
                                            const amrex::Geometry& geom,
                                            amrex::Real dt,
                                            amrex::Real eps_visc,
                                            const DirectionalRosState& st,
                                            const amrex::MultiFab* nonburnable = nullptr,
                                            bool wall_extrap = false,
                                            const fire_levelset::LevelSetGradient& grad_scheme = fire_levelset::LevelSetGradient{},
                                            int shape = DIRECTIONAL_SHAPE_PROJECTION)
{
    fire_levelset::advect_levelset_rk3_with_fill(phi, slopes, geom, dt, eps_visc,
        [&](amrex::MultiFab& R, const amrex::MultiFab& phi_stage) {
            fill_directional_ros(R, phi_stage, wind, slopes, geom, st, nonburnable, wall_extrap, shape);
        }, nonburnable, wall_extrap, grad_scheme);
}

#endif
