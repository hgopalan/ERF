/**
 * @file ERF_DustThreshold.H
 * @brief Threshold friction velocity computation for the ERF-Dust emission layer.
 *
 * Provides inline GPU-device functions and a MultiFab kernel for computing
 * the threshold friction velocity u*_t at each dust grid cell.
 *
 * Emission begins when u* (from the surface layer) exceeds u*_t.
 *
 * Base threshold, evaluated for the SALTATING grains (erf.dust.saltation_diameter,
 * default 75 um), not for the emitted dust bins: Marticorena & Bergametti (1995)
 * drive the sandblasting flux with the threshold of the soil's coarse mode, and
 * the threshold-versus-size relation has its minimum there. Two forms:
 *
 *   shao_lu (default, Shao & Lu 2000):
 *     u*_t = sqrt( A_N * ( rho_p g d / rho_a  +  gamma / (rho_a d) ) )
 *     with A_N = 0.0123 and gamma = 1.65e-4 kg/s^2 (the cohesion term, which
 *     dominates below ~50 um and makes the curve non-monotone in d).
 *   bagnold (Bagnold 1941, inertial branch only, valid above ~100 um):
 *     u*_t = A * sqrt( rho_p g d / rho_a )
 *
 * Until October 2026 the base was Bagnold's formula at the 7 um bin-0 diameter,
 * 0.0385 m/s: 13x below Shao-Lu at that size and 5x below the physical minimum,
 * so every case ran saturated (u*_t / u* ~ 0.08) and the surface-state modifiers
 * acted at a few percent with the wrong sign (the Owen factor (1-R)(1+R)^2 rises
 * with R below 1/3). erf.dust.ustar_t_base >= 0 still overrides the formula.
 *
 * Modifier chain (every factor >= 1 raises u*_t, inhibiting emission):
 *   f_chem  = (1 + alpha_c * crust_index) * (1 + alpha_e * efflorescence)
 *   f_moist = 1 + beta_m  * surface_moisture
 *   f_supp  = 1 + gamma_s * suppression
 *
 *   u*_t_final = u*_t_base * f_chem * f_moist * f_supp * f_slope
 *   clamped to [USTAR_T_MIN, USTAR_T_MAX].
 *
 * (The code divided by f_moist * f_supp until October 2026, so a wet or treated
 * surface emitted MORE; the DustThreshold gtests pin the direction.)
 *
 * Slope factor, Iversen & Rasmussen (1994), signed along the wind:
 *   f_slope = sqrt( cos(theta) + sin(theta) / tan(35 deg) ),  theta the slope angle
 *   along the wind at zref (positive upslope, negative downslope); isotropic in
 *   |grad z| where the wind is zero. Floored at sqrt(0.1).
 *
 * References:
 *   Bagnold, R.A. (1941). The Physics of Blown Sand and Desert Dunes.
 *     Methuen, London.
 *   Marticorena, B., & Bergametti, G. (1995). J. Geophys. Res., 100, 16415.
 *     https://doi.org/10.1029/95JD00690
 *   Shao, Y., & Lu, H. (2000). J. Geophys. Res., 105, 22437.
 *     https://doi.org/10.1029/2000JD900304
 *   Iversen, J.D., & White, B.R. (1982). Sedimentology, 29, 111.
 *     https://doi.org/10.1111/j.1365-3091.1982.tb01713.x
 *   Iversen, J.D., & Rasmussen, K.R. (1994). Sedimentology, 41, 721.
 *     https://doi.org/10.1111/j.1365-3091.1994.tb01419.x
 *   Fecan, F., Marticorena, B., & Bergametti, G. (1999). Ann. Geophys., 17, 149.
 *     https://doi.org/10.1007/s00585-999-0149-7
 */

#ifndef ERF_DUST_THRESHOLD_H
#define ERF_DUST_THRESHOLD_H

#ifdef ERF_USE_DUST

#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>
#include <cmath>

namespace DustThresholdConst {
    constexpr amrex::Real USTAR_T_MIN  = 0.001;   ///< Floor for u*_t [m/s].
    constexpr amrex::Real USTAR_T_MAX  = 5.0;    ///< Ceiling for u*_t [m/s].
    constexpr amrex::Real RHO_AIR_STD  = 1.225;  ///< Standard air density [kg/m^3].
    constexpr amrex::Real G_ACCEL      = 9.81;   ///< Gravitational acceleration [m/s^2].
    constexpr amrex::Real BETA_MOIST   = 4.0;    ///< Moisture inhibition coefficient [-] on the
                                                  ///<   [0,1] moisture flag (an ad hoc linear form,
                                                  ///<   not Fecan et al. 1999, which needs a
                                                  ///<   gravimetric soil moisture).
    constexpr amrex::Real GAMMA_SUPP   = 6.0;    ///< Suppression elevation coefficient [-].
    constexpr amrex::Real TAN_REPOSE   = 0.7002075382097097; ///< tan(35 deg), the angle of repose
                                                  ///<   in Iversen & Rasmussen (1994).
    constexpr amrex::Real SLOPE_FLOOR  = 0.1;    ///< Floor of the squared slope factor [-].
    constexpr amrex::Real SHAO_LU_A_N  = 0.0123; ///< Shao & Lu (2000) A_N [-].
    constexpr amrex::Real SHAO_LU_GAMMA = 1.65e-4; ///< Shao & Lu (2000) cohesion gamma [kg/s^2].
}

/**
 * @brief Bagnold (1941) fluid threshold, inertial branch: u*_t = A sqrt(rho_p g d / rho_a).
 *
 * Valid above ~100 um; it has no cohesion term and keeps falling as d shrinks,
 * which is why it must not be evaluated at a dust-bin diameter.
 *
 * @param[in] A        Bagnold coefficient (0.1) [-].
 * @param[in] rho_p    Particle density [kg/m^3].
 * @param[in] d_m      Saltating grain diameter [m].
 * @param[in] rho_a    Air density [kg/m^3]. If <= 0, RHO_AIR_STD is used.
 *
 * @return u*_t [m/s].
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_ustar_t_bagnold(amrex::Real A, amrex::Real rho_p,
                                     amrex::Real d_m, amrex::Real rho_a) noexcept
{
    amrex::Real rho = (rho_a > 0.0) ? rho_a : DustThresholdConst::RHO_AIR_STD;
    return A * std::sqrt(rho_p * DustThresholdConst::G_ACCEL * d_m / rho);
}

/**
 * @brief Shao & Lu (2000) threshold with the cohesion term:
 *        u*_t = sqrt( A_N ( rho_p g d / rho_a + gamma / (rho_a d) ) ).
 *
 * Non-monotone in d with a minimum near 75 um (0.204 m/s for quartz in
 * standard air): the inertial term grows with d, the cohesion term with 1/d.
 *
 * @param[in] A_N      Shao-Lu coefficient (0.0123) [-].
 * @param[in] gamma    Cohesion parameter (1.65e-4) [kg/s^2].
 * @param[in] rho_p    Particle density [kg/m^3].
 * @param[in] d_m      Saltating grain diameter [m].
 * @param[in] rho_a    Air density [kg/m^3]. If <= 0, RHO_AIR_STD is used.
 *
 * @return u*_t [m/s].
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_ustar_t_shao_lu(amrex::Real A_N, amrex::Real gamma, amrex::Real rho_p,
                                     amrex::Real d_m, amrex::Real rho_a) noexcept
{
    amrex::Real rho = (rho_a > 0.0) ? rho_a : DustThresholdConst::RHO_AIR_STD;
    amrex::Real d   = amrex::max(d_m, amrex::Real(1.0e-9));
    return std::sqrt(A_N * (rho_p * DustThresholdConst::G_ACCEL * d / rho + gamma / (rho * d)));
}

/**
 * @brief Compute moisture inhibition factor f_moist.
 *
 * f_moist = 1 + beta_m * w; w clamped to [0,1].
 * An ad-hoc linear inhibition on the [0, 1] moisture flag of the surface map,
 * not Fecan et al. (1999)'s factor sqrt(1 + 1.21 (w - w')^0.68), which needs
 * a gravimetric soil moisture the dust module does not carry.
 *
 * @param[in] w Surface moisture [0,1]. Values outside this range are clamped.
 *
 * @return f_moist, always >= 1.0 (it multiplies the threshold).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_moisture_inhibition(amrex::Real w) noexcept
{
    w = amrex::min(amrex::max(w, amrex::Real(0.0)), amrex::Real(1.0));
    return amrex::Real(1.0) + DustThresholdConst::BETA_MOIST * w;
}

/**
 * @brief Compute suppression elevation factor f_supp.
 *
 * f_supp = 1 + gamma_s * s; s clamped to [0,1].
 *
 * @param[in] s Suppression agent coverage [0,1]. Values outside this range are clamped.
 *
 * @return f_supp, always >= 1.0 (it multiplies the threshold).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_suppression_factor(amrex::Real s) noexcept
{
    s = amrex::min(amrex::max(s, amrex::Real(0.0)), amrex::Real(1.0));
    return amrex::Real(1.0) + DustThresholdConst::GAMMA_SUPP * s;
}

/**
 * @brief Slope factor on the threshold, Iversen & Rasmussen (1994).
 *
 * f_slope = sqrt( max(SLOPE_FLOOR, cos(theta) + sin(theta) / tan(35 deg)) ) with
 * theta the terrain slope angle along the wind: positive where the wind blows
 * uphill (the threshold rises), negative on the lee face (it falls). With no wind
 * (|U| = 0) the isotropic form with |grad z| is used, as before October 2026
 * when every face paid the upslope penalty.
 *
 * @param[in] slope_x  dz/dx [m/m].
 * @param[in] slope_y  dz/dy [m/m].
 * @param[in] wind_u   Wind at zref, x component [m/s].
 * @param[in] wind_v   Wind at zref, y component [m/s].
 *
 * @return f_slope [-], 1 on flat ground.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_slope_factor(amrex::Real slope_x, amrex::Real slope_y,
                                 amrex::Real wind_u, amrex::Real wind_v) noexcept
{
    const amrex::Real slope_mag = std::sqrt(slope_x * slope_x + slope_y * slope_y);
    if (!(slope_mag > 0.0)) return amrex::Real(1.0);
    const amrex::Real spd = std::sqrt(wind_u * wind_u + wind_v * wind_v);
    // slope along the wind (signed), or the magnitude with no wind
    const amrex::Real s_along = (spd > 0.0)
        ? (slope_x * wind_u + slope_y * wind_v) / spd
        : slope_mag;
    const amrex::Real hyp       = std::sqrt(amrex::Real(1.0) + s_along * s_along);
    const amrex::Real cos_theta = amrex::Real(1.0) / hyp;
    const amrex::Real sin_theta = s_along / hyp;
    return std::sqrt(amrex::max(amrex::Real(DustThresholdConst::SLOPE_FLOOR),
                                cos_theta + sin_theta / amrex::Real(DustThresholdConst::TAN_REPOSE)));
}

/**
 * @brief Compute final threshold friction velocity with all modifiers.
 *
 *   f_chem  = (1 + alpha_crust * crust_index) * (1 + alpha_efflor * efflorescence)
 *   f_moist = 1 + beta_m * surface_moisture
 *   f_supp  = 1 + gamma_s * suppression
 *
 *   u*_t = u*_t_base * f_chem * f_moist * f_supp * f_slope
 *   clamped to [USTAR_T_MIN, USTAR_T_MAX].
 *
 * Every modifier is >= 1 on flat ground and raises u*_t (harder to emit). A
 * fully crusted soil (crust_index 1) is protected; a burned cell whose crust
 * was removed emits more; a wet or suppressant-treated cell emits less.
 *
 * @param[in] ustar_t_base       Base threshold for the saltating grains [m/s].
 * @param[in] crust_index        Crust strength [0,1]. 0=uncrusted (bare), 1=fully crusted (protected).
 * @param[in] efflorescence      Efflorescence coverage [0,1].
 * @param[in] alpha_crust        Crust enhancement coefficient [-].
 * @param[in] alpha_efflor       Efflorescence enhancement coefficient [-].
 * @param[in] surface_moisture   Surface moisture flag [0,1].
 * @param[in] suppression        Suppression coverage [0,1].
 * @param[in] slope_x            Terrain slope dz/dx [m/m].
 * @param[in] slope_y            Terrain slope dz/dy [m/m].
 * @param[in] wind_u             Wind at zref, x [m/s] (0 = isotropic slope factor).
 * @param[in] wind_v             Wind at zref, y [m/s].
 *
 * @return Final u*_t clamped to [USTAR_T_MIN, USTAR_T_MAX] [m/s].
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_ustar_t_full(amrex::Real ustar_t_base,
                                  amrex::Real crust_index,
                                  amrex::Real efflorescence,
                                  amrex::Real alpha_crust,
                                  amrex::Real alpha_efflor,
                                  amrex::Real surface_moisture,
                                  amrex::Real suppression,
                                  amrex::Real slope_x = 0.0,
                                  amrex::Real slope_y = 0.0,
                                  amrex::Real wind_u  = 0.0,
                                  amrex::Real wind_v  = 0.0) noexcept
{
    // Chemistry enhancement from crust and efflorescence (PHREEQC).
    // Reference: Marticorena & Bergametti (1995), https://doi.org/10.1029/95JD00690
    const amrex::Real crust  = amrex::min(amrex::max(crust_index,   amrex::Real(0.0)), amrex::Real(1.0));
    const amrex::Real efflor = amrex::min(amrex::max(efflorescence, amrex::Real(0.0)), amrex::Real(1.0));
    amrex::Real f_chem = (amrex::Real(1.0) + alpha_crust * crust) * (amrex::Real(1.0) + alpha_efflor * efflor);
    f_chem = amrex::max(f_chem, amrex::Real(0.01));

    const amrex::Real f_moist = compute_moisture_inhibition(surface_moisture);
    const amrex::Real f_supp  = compute_suppression_factor(suppression);
    const amrex::Real f_slope = compute_slope_factor(slope_x, slope_y, wind_u, wind_v);

    const amrex::Real ut = ustar_t_base * f_chem * f_moist * f_supp * f_slope;

    return amrex::min(amrex::max(ut, amrex::Real(DustThresholdConst::USTAR_T_MIN)),
                      amrex::Real(DustThresholdConst::USTAR_T_MAX));
}

/**
 * @brief Recompute dust_ustar_t at all dust grid cells.
 *
 * GPU ParallelFor over dust grid tiles. Applies the full modifier chain:
 * crust/efflorescence (PHREEQC), moisture inhibition, suppression, slope.
 * Called from DustLayer::initialize and DustLayer::advance.
 *
 * @param[in,out] dust_ustar_t      Threshold friction velocity [m/s].
 * @param[in]     dust_ustar_base   Base threshold (read-only) [m/s].
 * @param[in]     dust_crust_index  Crust strength index [0,1].
 * @param[in]     dust_efflor       Efflorescence fraction [0,1].
 * @param[in]     dust_surf_moist   Surface moisture flag [0,1].
 * @param[in]     dust_suppression  Suppression coverage [0,1].
 * @param[in]     alpha_crust       Crust coefficient [-].
 * @param[in]     alpha_efflor      Efflorescence coefficient [-].
 * @param[in]     dust_slopes       Terrain slopes (2 comp) or nullptr for flat ground.
 * @param[in]     dust_wind_ref     Wind at zref (2 comp) or nullptr: the slope factor
 *                                  is then isotropic in |grad z|.
 */
inline void recompute_dust_ustar_t(
    amrex::MultiFab&       dust_ustar_t,
    const amrex::MultiFab& dust_ustar_base,
    const amrex::MultiFab& dust_crust_index,
    const amrex::MultiFab& dust_efflor,
    const amrex::MultiFab& dust_surf_moist,
    const amrex::MultiFab& dust_suppression,
    amrex::Real            alpha_crust,
    amrex::Real            alpha_efflor,
    const amrex::MultiFab* dust_slopes = nullptr,
    const amrex::MultiFab* dust_wind_ref = nullptr)
{
    for (amrex::MFIter mfi(dust_ustar_t, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto ut_arr     = dust_ustar_t.array(mfi);
        auto base_arr   = dust_ustar_base.const_array(mfi);
        auto crust_arr  = dust_crust_index.const_array(mfi);
        auto efflor_arr = dust_efflor.const_array(mfi);
        auto moist_arr  = dust_surf_moist.const_array(mfi);
        auto supp_arr   = dust_suppression.const_array(mfi);
        amrex::Array4<const amrex::Real> slope_arr;
        amrex::Array4<const amrex::Real> wind_arr;
        const bool have_slopes = (dust_slopes != nullptr);
        const bool have_wind   = (dust_wind_ref != nullptr);
        if (have_slopes) { slope_arr = dust_slopes->const_array(mfi); }
        if (have_wind)   { wind_arr  = dust_wind_ref->const_array(mfi); }
        const amrex::Real ac = alpha_crust;
        const amrex::Real ae = alpha_efflor;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real slope_x = have_slopes ? slope_arr(i,j,k,0) : amrex::Real(0.0);
            const amrex::Real slope_y = have_slopes ? slope_arr(i,j,k,1) : amrex::Real(0.0);
            const amrex::Real wind_u  = have_wind ? wind_arr(i,j,k,0) : amrex::Real(0.0);
            const amrex::Real wind_v  = have_wind ? wind_arr(i,j,k,1) : amrex::Real(0.0);
            ut_arr(i,j,k) = compute_ustar_t_full(
                base_arr(i,j,k), crust_arr(i,j,k), efflor_arr(i,j,k),
                ac, ae, moist_arr(i,j,k), supp_arr(i,j,k), slope_x, slope_y, wind_u, wind_v);
        });
    }
}

#endif // ERF_USE_DUST

#endif // ERF_DUST_THRESHOLD_H
