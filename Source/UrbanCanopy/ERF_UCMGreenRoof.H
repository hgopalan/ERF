/**
 * @file ERF_UCMGreenRoof.H
 * @brief GPU-safe header-only green roof latent heat (evapotranspiration) solver (Phase 5.3)
 *
 * Phase 5.3: Simple green roof model with bulk-resistance latent heat flux.
 *
 * Physics:
 *  - Latent heat flux: LE = (rho * Lv) * (q_sat(T_skin) - q_canyon) / (r_s + r_a)
 *    where:
 *      q_sat(T)    = saturation specific humidity [kg/kg]
 *      q_canyon    = canyon-air specific humidity [kg/kg]
 *      r_s         = stomatal resistance [s/m]
 *      r_a         = aerodynamic resistance [s/m] ≈ 1 / (Ch * |U|)
 *  - Water bucket depletion: dW/dt = -LE / (rho_water * L_v) [m/s]
 *    where rho_water ≈ 1000 kg/m³, L_v ≈ 2.5e6 J/kg
 *
 * Limitations (Phase 5.3):
 *  - No stomatal closure on drought (TODO Phase 5.3-followup)
 *  - Simplified aerodynamic resistance (MOST-derived Ch * |U|)
 */

#ifndef ERF_UCM_GREEN_ROOF_H
#define ERF_UCM_GREEN_ROOF_H

#include <AMReX_REAL.H>
#include <cmath>

/**
 * @brief Compute saturation specific humidity [kg/kg] via Magnus formula
 *
 * Simple approximation valid for T ∈ [250, 320] K.
 *
 * @param T_K  Temperature [K]
 * @return q_sat [kg/kg]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real qsat_magnus(amrex::Real T_K)
{
    constexpr amrex::Real a_magnus  = 17.27;  // Magnus coefficient
    constexpr amrex::Real b_magnus  = 237.7;  // Magnus coefficient [°C]
    constexpr amrex::Real eps       = 0.622;  // R_d / R_v (molecular weight ratio)

    // Convert to Celsius
    amrex::Real T_C = T_K - 273.15;

    // Saturation vapor pressure [hPa]
    amrex::Real es = 6.112 * std::exp((a_magnus * T_C) / (b_magnus + T_C));

    // Saturation specific humidity [kg/kg]
    // q_sat = eps * e_s / (P - e_s), assuming P ≈ 1013.25 hPa ≈ 1 atm
    amrex::Real q_sat = eps * es / (1013.25 - es);

    return amrex::max(q_sat, 0.0);
}

/**
 * @brief Compute green roof latent heat flux (evapotranspiration) via bulk resistance
 *
 * Simple implementation: LE = (ρ * Lv) * Δq / (r_s + r_a)
 *
 * @param[in] soil_moisture       Current soil moisture [m³/m³]; clamped to [0, W_max]
 * @param[in] T_skin_roof         Green roof skin temperature [K]
 * @param[in] T_canyon_air        Canyon air temperature [K]
 * @param[in] q_canyon_air        Canyon air specific humidity [kg/kg]
 * @param[in] wind_speed_ref      Reference height wind speed [m/s]
 * @param[in] Ch                  Bulk heat transfer coefficient [-]
 * @param[in] r_stomatal_s_per_m  Stomatal resistance [s/m]
 *
 * @return LE [W/m²]  Positive = surface loses latent heat (cooling)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_green_roof_LE(
    amrex::Real soil_moisture,           // [m³/m³]
    amrex::Real T_skin_roof,             // [K]
    amrex::Real T_canyon_air,            // [K]
    amrex::Real q_canyon_air,            // [kg/kg]
    amrex::Real wind_speed_ref,          // [m/s]
    amrex::Real Ch,                      // [-]
    amrex::Real r_stomatal_s_per_m)      // [s/m]
{
    constexpr amrex::Real rho_air    = 1.225;           // Air density [kg/m³]
    constexpr amrex::Real Lv         = 2.5e6;           // Latent heat of vaporization [J/kg]
    constexpr amrex::Real U_min      = 0.01;            // Minimum wind speed [m/s]
    constexpr amrex::Real theta_min  = 1.e-6;           // Minimum soil moisture [m³/m³]

    // Guard against unphysical values
    amrex::Real W = amrex::max(soil_moisture, 0.0);
    amrex::Real U = amrex::max(std::abs(wind_speed_ref), U_min);

    // Aerodynamic resistance: r_a ≈ 1 / (Ch * |U|) [s/m]
    amrex::Real r_aero = 1.0 / (Ch * U);

    // Bulk total resistance [s/m]
    amrex::Real r_total = r_stomatal_s_per_m + r_aero;
    if (r_total < 1.e-6) r_total = 1.e-6;  // Guard division

    // Saturation specific humidity at skin temperature
    amrex::Real q_sat_skin = qsat_magnus(T_skin_roof);

    // Specific humidity deficit [kg/kg]
    amrex::Real dq = amrex::max(q_sat_skin - q_canyon_air, 0.0);

    // Latent heat flux [W/m²] (positive = cooling)
    // Modulate by soil moisture availability (simple linear reduction if dry)
    // TODO(Phase 5.3-followup): Add stomatal closure model based on soil moisture stress
    amrex::Real moisture_factor = (W > theta_min) ? 1.0 : 0.0;

    amrex::Real LE = rho_air * Lv * dq / r_total * moisture_factor;

    return LE;
}

/**
 * @brief Update green roof soil moisture bucket via water balance
 *
 * Simple bucket model: dW/dt = -LE / (ρ_water * Lv) [m/s or m³/m³/s]
 *
 * Clamped to [0, W_max].
 *
 * @param[in,out] soil_moisture  Current soil moisture [m³/m³]
 * @param[in] LE                 Latent heat flux [W/m²]
 * @param[in] dt                 Timestep [s]
 * @param[in] W_max              Water bucket capacity [m H2O → m³/m³ via depth/thickness]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void update_green_roof_moisture(
    amrex::Real& soil_moisture,
    amrex::Real LE,
    amrex::Real dt,
    amrex::Real W_max)
{
    constexpr amrex::Real rho_water = 1000.0;  // Water density [kg/m³]
    constexpr amrex::Real Lv        = 2.5e6;   // Latent heat of vaporization [J/kg]

    // Negative LE means condensation (recharge); positive means evapotranspiration (depletion)
    amrex::Real dW = -LE / (rho_water * Lv) * dt;  // [m H2O]

    soil_moisture += dW / W_max;  // Convert to [m³/m³]
    soil_moisture = amrex::max(amrex::min(soil_moisture, 1.0), 0.0);
}

#endif // ERF_UCM_GREEN_ROOF_H
