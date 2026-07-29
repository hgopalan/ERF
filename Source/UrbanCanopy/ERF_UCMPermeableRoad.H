/**
 * @file ERF_UCMPermeableRoad.H
 * @brief GPU-safe header-only permeable pavement evaporation solver (Phase 5.3)
 *
 * Phase 5.3: Simple permeable road model with bulk-resistance latent heat flux.
 *
 * Physics:
 *  - Similar to ERF_UCMGreenRoof.H but applied to permeable road surface
 *  - Latent heat flux: LE = (rho * Lv) * (q_sat(T_skin) - q_canyon) / (r_s + r_a)
 *    where:
 *      r_s    = surface resistance for bare soil [s/m] (typically larger than green roof)
 *      r_a    = aerodynamic resistance [s/m]
 *  - Water bucket depletion: dW/dt = -LE / (rho_water * L_v) [m/s]
 *
 * Differences from green roof:
 *  - No stomatal resistance (bare soil surface)
 *  - Typically deeper water storage (0.2 m vs 0.1 m)
 *  - Higher thermal properties (asphalt + soil vs vegetation soil)
 */

#ifndef ERF_UCM_PERMEABLE_ROAD_H
#define ERF_UCM_PERMEABLE_ROAD_H

#include <AMReX_REAL.H>
#include <cmath>

/**
 * @brief Compute saturation specific humidity [kg/kg] via Magnus formula
 *
 * (Identical to green roof implementation for consistency)
 *
 * @param T_K  Temperature [K]
 * @return q_sat [kg/kg]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real qsat_magnus_road(amrex::Real T_K)
{
    constexpr amrex::Real a_magnus  = 17.27;
    constexpr amrex::Real b_magnus  = 237.7;
    constexpr amrex::Real eps       = 0.622;

    amrex::Real T_C = T_K - 273.15;
    amrex::Real es = 6.112 * std::exp((a_magnus * T_C) / (b_magnus + T_C));
    amrex::Real q_sat = eps * es / (1013.25 - es);

    return amrex::max(q_sat, 0.0);
}

/**
 * @brief Compute permeable road latent heat flux (evaporation) via bulk resistance
 *
 * Bare-soil evaporation model: LE = (ρ * Lv) * Δq / (r_s + r_a)
 *
 * For permeable road, r_s is soil surface resistance (no stomata), typically
 * ~200–400 s/m for moist bare soil to ~1000+ s/m for dry soil.
 * Phase 5.3 uses fixed r_s; Phase 5.3-followup can add soil-moisture-dependent
 * surface resistance.
 *
 * @param[in] soil_moisture             Current soil moisture [m³/m³]
 * @param[in] T_skin_road               Road surface skin temperature [K]
 * @param[in] T_canyon_air              Canyon air temperature [K]
 * @param[in] q_canyon_air              Canyon air specific humidity [kg/kg]
 * @param[in] wind_speed_ref            Reference height wind speed [m/s]
 * @param[in] Ch                        Bulk heat transfer coefficient [-]
 * @param[in] r_soil_s_per_m            Surface (soil) resistance [s/m] (~200 for bare soil)
 *
 * @return LE [W/m²]  Positive = surface loses latent heat (cooling)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_permeable_road_LE(
    amrex::Real soil_moisture,           // [m³/m³]
    amrex::Real T_skin_road,             // [K]
    amrex::Real T_canyon_air,            // [K]
    amrex::Real q_canyon_air,            // [kg/kg]
    amrex::Real wind_speed_ref,          // [m/s]
    amrex::Real Ch,                      // [-]
    amrex::Real r_soil_s_per_m)          // [s/m]
{
    constexpr amrex::Real rho_air    = 1.225;           // Air density [kg/m³]
    constexpr amrex::Real Lv         = 2.5e6;           // Latent heat of vaporization [J/kg]
    constexpr amrex::Real U_min      = 0.01;            // Minimum wind speed [m/s]
    constexpr amrex::Real theta_min  = 1.e-6;           // Minimum soil moisture [m³/m³]

    amrex::Real W = amrex::max(soil_moisture, 0.0);
    amrex::Real U = amrex::max(std::abs(wind_speed_ref), U_min);

    // Aerodynamic resistance: r_a ≈ 1 / (Ch * |U|) [s/m]
    amrex::Real r_aero = 1.0 / (Ch * U);

    // Bulk total resistance [s/m]
    amrex::Real r_total = r_soil_s_per_m + r_aero;
    if (r_total < 1.e-6) r_total = 1.e-6;

    // Saturation specific humidity at road surface temperature
    amrex::Real q_sat_skin = qsat_magnus_road(T_skin_road);

    // Specific humidity deficit [kg/kg]
    amrex::Real dq = amrex::max(q_sat_skin - q_canyon_air, 0.0);

    // Latent heat flux [W/m²] (positive = cooling)
    // Modulate by soil moisture availability
    amrex::Real moisture_factor = (W > theta_min) ? 1.0 : 0.0;

    amrex::Real LE = rho_air * Lv * dq / r_total * moisture_factor;

    return LE;
}

/**
 * @brief Update permeable road soil moisture bucket via water balance
 *
 * Same bucket model as green roof: dW/dt = -LE / (ρ_water * Lv) [m/s]
 *
 * @param[in,out] soil_moisture  Current soil moisture [m³/m³]
 * @param[in] LE                 Latent heat flux [W/m²]
 * @param[in] dt                 Timestep [s]
 * @param[in] W_max              Water bucket capacity [m H2O]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void update_permeable_road_moisture(
    amrex::Real& soil_moisture,
    amrex::Real LE,
    amrex::Real dt,
    amrex::Real W_max)
{
    constexpr amrex::Real rho_water = 1000.0;  // Water density [kg/m³]
    constexpr amrex::Real Lv        = 2.5e6;   // Latent heat of vaporization [J/kg]

    amrex::Real dW = -LE / (rho_water * Lv) * dt;  // [m H2O]

    soil_moisture += dW / W_max;
    soil_moisture = amrex::max(amrex::min(soil_moisture, 1.0), 0.0);
}

#endif // ERF_UCM_PERMEABLE_ROAD_H
