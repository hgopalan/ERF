/**
 * @file ERF_DustSuppression.H
 * @brief Dust suppression agent degradation model for the ERF-Dust 2D layer.
 *
 * Tracks the decay of dust suppression agent (water, MgCl2, lignin
 * sulfonate) applied to haul roads and tailings surfaces. The coverage fraction
 * in dust_suppression [0,1] decays exponentially with an e-folding time that
 * depends on surface temperature and wind speed (half-life = 0.693 tau). When
 * coverage falls below DustSuppressionConst::COVERAGE_FLOOR, it is set to zero.
 * T_surf and u_10m are domain-wide scalars: the maximum surface temperature
 * and the maximum wind SPEED at zref (DustLayer::advance).
 *
 * Decay model:
 *   coverage_new = coverage_old * exp(-dt / tau_eff)
 *
 * where the effective e-folding time tau_eff [s] is:
 *   tau_eff = tau_base / (f_T * f_wind)
 *
 *   f_T    = exp(k_T * (T_surf - T_ref))      temperature acceleration
 *   f_wind = 1 + k_wind * u_10m               wind enhancement
 *
 * k_T and k_wind are tunable coefficients. Warmer temperatures and higher
 * winds reduce the effective half-life (faster agent degradation).
 *
 * T_surf is the domain maximum of the surface temperature (the surface
 * layer's potential temperature converted with the surface Exner function;
 * until October 2026 the potential temperature itself was used, a factor 2 on
 * the decay rate at 850 hPa) and u_10m the domain maximum wind speed at zref;
 * without a coupled atmosphere the test_* placeholders of DustParams.
 *
 * Structural reference: ERF_FireBreak.H in Source/Fire/ (local 2D property
 * modification on the fire grid — same pattern adapted for suppression).
 *
 * References:
 *   Gillies, J.A., et al. (2005). Effectiveness of water and chemical
 *     dust suppressants on unpaved roads. J. Air Waste Manage. Assoc.,
 *     55, 1168-1176. https://doi.org/10.1080/10473289.2005.10464701
 *   U.S. EPA AP-42, Chapter 13.2.2 (Unpaved Roads).
 *     https://www.epa.gov/air-emissions-factors-and-quantification/
 *     ap-42-compilation-air-emission-factors
 *   MSHA 30 CFR Part 70 (Respirable Dust Exposure Limits).
 *     https://www.ecfr.gov/current/title-30/chapter-I/subchapter-O/part-70
 */

#ifndef ERF_DUST_SUPPRESSION_H
#define ERF_DUST_SUPPRESSION_H

#ifdef ERF_USE_DUST

#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>
#include <cmath>

namespace DustSuppressionConst {
    // Default base half-life [s] of suppression agent under reference conditions.
    // Water: ~3600 s (1 hr). Chemical agents (MgCl2): ~43200 s (12 hr).
    // Reference: Gillies et al. (2005).
    constexpr amrex::Real TAU_BASE_S       = 3600.0;

    // Reference surface temperature [K] for half-life calculation.
    constexpr amrex::Real T_REF_K          = 293.15;  // 20 C

    // Temperature acceleration coefficient [1/K].
    // f_T = exp(k_T * (T - T_ref)): higher T -> faster degradation.
    constexpr amrex::Real K_TEMP           = 0.05;

    // Wind enhancement coefficient [s/m].
    // f_wind = 1 + k_wind * u_10m.
    constexpr amrex::Real K_WIND           = 0.1;

    // Coverage floor: values below this are set to zero.
    constexpr amrex::Real COVERAGE_FLOOR   = 0.01;

    // Re-treatment flag threshold: when coverage drops below this value,
    // a per-cell re-treatment flag is set to 1.0 in dust_retreat_flag.
    constexpr amrex::Real RETREAT_THRESHOLD = 0.2;
}

/**
 * @brief Compute temperature acceleration factor for suppression degradation.
 *
 * f_T = exp(k_T * (T_surf - T_ref))
 * T_surf and T_ref in Kelvin. f_T >= 1 when T > T_ref.
 * Clamped to [0.1, 10.0] to prevent extreme values.
 *
 * Reference: Gillies et al. (2005), https://doi.org/10.1080/10473289.2005.10464701
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_temp_factor_supp(amrex::Real T_surf_K) noexcept
{
    amrex::Real dT = T_surf_K - DustSuppressionConst::T_REF_K;
    amrex::Real fT = std::exp(DustSuppressionConst::K_TEMP * dT);
    return amrex::min(static_cast<amrex::Real>(amrex::max(static_cast<amrex::Real>(fT), static_cast<amrex::Real>(0.1))), static_cast<amrex::Real>(10.0));
}

/**
 * @brief Compute wind enhancement factor for suppression degradation.
 *
 * f_wind = 1 + k_wind * u_10m
 * u_10m in [m/s]. f_wind >= 1.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_wind_factor_supp(amrex::Real u_10m) noexcept
{
    amrex::Real u = amrex::max(static_cast<amrex::Real>(u_10m), static_cast<amrex::Real>(0.0));
    return 1.0 + DustSuppressionConst::K_WIND * u;
}

/**
 * @brief Advance suppression coverage for one timestep at a single cell.
 *
 * coverage_new = coverage_old * exp(-dt / tau_eff)
 * tau_eff = tau_base / (f_T * f_wind)
 * Values below COVERAGE_FLOOR set to zero.
 *
 * @param coverage_old  Current coverage fraction [0,1].
 * @param T_surf_K      Surface temperature [K].
 * @param u_10m         10 m wind speed [m/s].
 * @param dt_s          Timestep [s].
 * @param tau_base_s    Base e-folding time [s] (erf.dust.supp_tau_base_s).
 * @return Updated coverage fraction [0,1].
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real advance_suppression_coverage(amrex::Real coverage_old,
                                          amrex::Real T_surf_K,
                                          amrex::Real u_10m,
                                          amrex::Real dt_s,
                                          amrex::Real tau_base_s) noexcept
{
    if (coverage_old <= DustSuppressionConst::COVERAGE_FLOOR) return 0.0;
    amrex::Real fT    = compute_temp_factor_supp(T_surf_K);
    amrex::Real fwind = compute_wind_factor_supp(u_10m);
    amrex::Real tau_eff = tau_base_s / (fT * fwind);
    tau_eff = amrex::max(static_cast<amrex::Real>(tau_eff), static_cast<amrex::Real>(1.0));  // floor at 1 s to prevent divide-by-zero
    amrex::Real c_new = coverage_old * std::exp(-dt_s / tau_eff);
    if (c_new < DustSuppressionConst::COVERAGE_FLOOR) c_new = 0.0;
    return c_new;
}

/**
 * @brief Advance dust_suppression coverage and set re-treatment flag MultiFab.
 *
 * GPU ParallelFor over dust grid tiles. Decays suppression coverage each
 * timestep using advance_suppression_coverage. When coverage drops below
 * RETREAT_THRESHOLD, sets dust_retreat_flag = 1.0 at that cell.
 *
 * Called from DustLayer::advance before recompute_dust_ustar_t so that
 * updated suppression is used in the threshold computation.
 *
 * @param[in,out] dust_suppression    Suppression coverage [0,1].
 * @param[out]    dust_retreat_flag   Re-treatment flag [0=ok, 1=retreat needed].
 * @param[in]     dust_treated_mask   1 where the cell was ever treated.
 * @param[in]     T_surf_K            Surface temperature [K] (uniform placeholder).
 * @param[in]     u_10m               10 m wind speed [m/s] (uniform placeholder).
 * @param[in]     dt_s                Timestep [s].
 * @param[in]     tau_base_s          Base half-life [s].
 */
inline void advance_dust_suppression(
    amrex::MultiFab& dust_suppression,
    amrex::MultiFab& dust_retreat_flag,
    const amrex::MultiFab& dust_treated_mask,
    amrex::Real      T_surf_K,
    amrex::Real      u_10m,
    amrex::Real      dt_s,
    amrex::Real      tau_base_s)
{
    const amrex::Real retreat_thresh = DustSuppressionConst::RETREAT_THRESHOLD;

    for (amrex::MFIter mfi(dust_suppression, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto supp_arr   = dust_suppression.array(mfi);
        auto retreat_arr= dust_retreat_flag.array(mfi);
        auto treated    = dust_treated_mask.const_array(mfi);
        const amrex::Real T   = T_surf_K;
        const amrex::Real u   = u_10m;
        const amrex::Real dt  = dt_s;
        const amrex::Real tau = tau_base_s;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real c_new = advance_suppression_coverage(
                supp_arr(i,j,k), T, u, dt, tau);
            supp_arr(i,j,k) = c_new;
            // Re-treatment flag: a treated cell whose coverage has worn below
            // the threshold, including one worn to zero (a cell at zero used to
            // lose the flag and could not be told from one never treated).
            retreat_arr(i,j,k) = (treated(i,j,k) > amrex::Real(0.5) &&
                                  c_new < retreat_thresh) ? amrex::Real(1.0) : amrex::Real(0.0);
        });
    }
}

#endif // ERF_USE_DUST

#endif // ERF_DUST_SUPPRESSION_H
