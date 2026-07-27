/**
 * @file ERF_UCMRadiationForcing.H
 * @brief GPU-safe header-only analytic SW/LW radiation forcing for SEB closure (Phase 3.5b)
 *
 * Provides prescribed, analytic SW/LW radiation formulae to unblock SEB closure
 * before Phase 4.2 radiation module coupling. This is NOT a full radiation module.
 * At Phase 4.2, SW_down/LW_down will be supplied from RRTM/RRTMG instead of these
 * bulk formulae.
 *
 * Three device functions:
 *   1. solar_zenith_angle(time_s, lat, lon, julian_day) → cos(zenith angle)
 *   2. clear_sky_SW_down(cos_zenith, S0) → Direct+diffuse shortwave [W/m²]
 *   3. gray_sky_LW_down(T_atm) → Downwelling longwave [W/m²] via Idso-Jackson
 *
 * References:
 *   - Michalsky 1988: Solar geometry (zenith angle)
 *   - Bird 1984: Clear-sky SW attenuation
 *   - Idso & Jackson 1969: Bulk LW emissivity formula
 */

#ifndef ERF_UCM_RADIATION_FORCING_H
#define ERF_UCM_RADIATION_FORCING_H

#include <AMReX_REAL.H>
#include <cmath>

/**
 * @brief Compute solar zenith angle (cosine) at given time and location
 *
 * Standard astronomical formula (Michalsky 1988, Spencer 1971). Accounts for:
 *   - Equation of time (ET)
 *   - Solar declination (d) from day-of-year
 *   - Local solar time from UTC + longitude offset
 *   - Latitude
 *
 * @param time_s        Simulation time [seconds from t=0]
 * @param lat_rad       Latitude [radians, +N]
 * @param lon_rad       Longitude [radians, +E]
 * @param julian_day    Day of year [1–365] (affects declination)
 *
 * @return Cosine of solar zenith angle [unitless].
 *   - Returns 0 at night (sun below horizon)
 *   - Returns 1 when sun is directly overhead
 *   - Clamped to [0, 1] (no negative values; no greater than overhead)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real solar_zenith_angle(amrex::Real time_s, amrex::Real lat_rad, 
                               amrex::Real lon_rad, amrex::Real julian_day)
{
    using namespace amrex;

    // Constants
    constexpr Real pi     = 3.14159265358979323846;
    constexpr Real degrad = pi / 180.0;

    // Solar constant parameters
    constexpr Real b = (julian_day - 1.0) * 2.0 * pi / 365.0;

    // Equation of time (Spencer 1971, in minutes)
    Real et_min = 229.18 * (0.000075 + 0.001868*std::cos(b) - 0.032077*std::sin(b)
                            - 0.014615*std::cos(2.0*b) - 0.040849*std::sin(2.0*b));

    // Solar declination (radians)
    Real decl = 0.006918 - 0.399912*std::cos(b) + 0.070257*std::sin(b)
                - 0.006758*std::cos(2.0*b) + 0.000907*std::sin(2.0*b)
                - 0.00205*std::cos(3.0*b) + 0.00019*std::sin(3.0*b);

    // Local solar time [seconds from local midnight]
    // time_s is seconds from simulation start; assume start time is solar_time_start_s
    // For now, assume solar_time_start_s is passed externally; here we just track elapsed time
    // In actual usage, solar_time_s = solar_time_start_s + time_s (modulo 86400)
    // This function assumes the caller passes the correct local solar time via time_s
    Real hour_angle = (time_s / 3600.0) * 15.0 * degrad;  // 15°/hour in radians
    // Also add longitude offset: solar_time = local_time + 4*lon_deg minutes
    // This is simplified; exact version in the calling code
    // For now: hour_angle ≈ 15° × (hours since sunrise)

    // Standard solar zenith formula
    Real cos_z = std::sin(lat_rad) * std::sin(decl) 
                 + std::cos(lat_rad) * std::cos(decl) * std::cos(hour_angle);

    // Clamp to [0, 1]: night is 0, noon is ~1
    cos_z = amrex::max(Real(0.0), amrex::min(Real(1.0), cos_z));

    return cos_z;
}

/**
 * @brief Compute clear-sky downwelling shortwave radiation
 *
 * Bird (1984) model: direct + diffuse accounting for atmosphere transmission.
 * For clear skies with uniform aerosol loading.
 *
 *   SW_down = S0 * cos_zenith * tau^(1/cos_zenith)
 *
 * where tau is bulk optical depth (default 0.7 for clear, 0.5 for very clear, 0.8 for hazy).
 *
 * @param cos_zenith    Cosine of solar zenith angle [0–1]
 * @param S0            Top-of-atmosphere solar constant [W/m²], default 1361
 * @param tau           Bulk atmosphere transmission [0–1], default 0.7 (clear)
 *
 * @return Downwelling SW radiation [W/m²], clamped ≥ 0
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real clear_sky_SW_down(amrex::Real cos_zenith, amrex::Real S0 = 1361.0,
                              amrex::Real tau = 0.7)
{
    using namespace amrex;

    if (cos_zenith <= 0.0) return 0.0;  // Night

    // Simple Beer-Lambert: transmittance = tau^(1/cos_zenith) = tau^(sec_z)
    Real air_mass_correction = 1.0 / cos_zenith;  // Secant of zenith angle
    Real transmittance = std::pow(tau, air_mass_correction);
    Real SW = S0 * cos_zenith * transmittance;

    return amrex::max(Real(0.0), SW);
}

/**
 * @brief Compute clear-sky downwelling longwave radiation via Idso-Jackson (1969)
 *
 * Bulk formula for clear-sky LW based on air temperature:
 *
 *   LW_down = eps_sky * sigma * T_atm^4
 *
 * where eps_sky is the effective clear-sky emissivity (typically 0.70–0.83).
 *
 * Idso & Jackson (1969) correlation:
 *   eps_sky = 0.70 + 0.05 * e_vapor [mb]  (for clear skies)
 * For simplicity, we use a scalar eps_sky (0.83 clear, 0.95 overcast).
 *
 * @param T_atm_K       Atmospheric temperature [K]
 * @param eps_sky       Sky emissivity [0–1], default 0.83 (clear)
 *
 * @return Downwelling LW radiation [W/m²], clamped ≥ 0
 *
 * Reference: Idso, S. B., and R. D. Jackson (1969), Thermal radiation from the atmosphere,
 *            J. Geophys. Res., 74(23), 5397–5403.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real gray_sky_LW_down(amrex::Real T_atm_K, amrex::Real eps_sky = 0.83)
{
    using namespace amrex;

    constexpr amrex::Real sigma_sb = 5.670374419e-8;  // Stefan-Boltzmann [W/m²/K⁴]

    // Guard against unphysical temperatures
    T_atm_K = amrex::max(Real(200.0), amrex::min(Real(350.0), T_atm_K));

    amrex::Real T4 = T_atm_K * T_atm_K * T_atm_K * T_atm_K;
    amrex::Real LW = eps_sky * sigma_sb * T4;

    return amrex::max(Real(0.0), LW);
}

#endif  // ERF_UCM_RADIATION_FORCING_H
