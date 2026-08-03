/**
 * @file ERF_UCMCrownTranspiration.H
 * @brief GPU-safe header-only crown latent heat (transpiration) solver (Phase 6.3)
 *
 * Phase 6.3: Crown transpiration model with bulk stomatal resistance.
 *
 * Physics:
 *  - Latent heat flux: LE_crown = LAI * ρ * Lv * (q_sat(T) - q_canyon) 
 *                                   / (r_boundary + r_stomatal)
 *    where:
 *      LAI             = Leaf Area Index [-]
 *      ρ               = air density [kg/m³]
 *      Lv              = latent heat of vaporization [J/kg]
 *      q_sat(T)        = saturation specific humidity [kg/kg]
 *      q_canyon        = canyon specific humidity [kg/kg]
 *      r_boundary      = aerodynamic leaf boundary layer resistance [s/m]
 *                        = 1 / (Ch_leaf * |U|); guarded against div-by-zero
 *      r_stomatal      = bulk stomatal resistance [s/m]
 *  - Sign convention: LE_crown > 0 means cooling (latent heat loss)
 *  - Clamped to [0, 500] W/m² to prevent unphysical dew and edge-case blowup
 *  - Returns 0.0 if q_sat <= q_canyon (no transpiration when saturated)
 *
 * Limitations (Phase 6.3):
 *  - Bulk resistance model (no Ball-Berry stomatal closure yet; deferred to Phase 6.4)
 *  - Uses constant LAI_default; per-cell LAI deferred to Phase 6.4
 *  - No soil-water coupling (deferred to Phase 6.4)
 */

#ifndef ERF_UCM_CROWN_TRANSPIRATION_H
#define ERF_UCM_CROWN_TRANSPIRATION_H

#include <AMReX_REAL.H>
#include <cmath>

/**
 * @brief Compute saturation specific humidity [kg/kg] via Magnus formula
 *
 * Magnus formula: es = 611.0 * exp(a * T_C / (T_C + b)) [Pa]
 * With a=17.27, b=237.3, valid for T ∈ [250, 320] K.
 * Follows the Phase 5.3 green-roof pattern with Magnus coefficients.
 *
 * @param T_K  Temperature [K]
 * @return q_sat [kg/kg]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real qsat_magnus_crown(amrex::Real T_K)
{
    constexpr amrex::Real a_magnus  = 17.27;   // Magnus coefficient
    constexpr amrex::Real b_magnus  = 237.3;   // Magnus coefficient [°C]
    constexpr amrex::Real eps       = 0.622;   // R_d / R_v (molecular weight ratio)
    constexpr amrex::Real P_atm     = 101325.0; // Atmospheric pressure [Pa]

    // Convert to Celsius
    amrex::Real T_C = T_K - 273.15;

    // Saturation vapor pressure [Pa] (Magnus formula)
    amrex::Real es = 611.0 * std::exp((a_magnus * T_C) / (T_C + b_magnus));

    // Saturation specific humidity [kg/kg]
    // q_sat = 0.622 * es / P_atm
    amrex::Real q_sat = eps * es / P_atm;

    return amrex::max(q_sat, 0.0);
}

/**
 * @brief Compute crown latent heat flux (transpiration) via bulk stomatal resistance
 *
 * Implementation: LE_crown = LAI * (ρ * Lv) * Δq / (r_boundary + r_stomatal)
 *
 * Per the problem statement (Contract #36), this is a semi-implicit pre-Newton
 * computation that is wired into the 4-var SEB solver as an extra cooling term
 * in the crown row. The computation uses T_crown_lagged (from previous timestep),
 * matching how green-roof uses T_skin_roof.
 *
 * @param[in] LAI                    Leaf Area Index [-]
 * @param[in] T_crown_lagged         Crown temperature from previous timestep [K]
 * @param[in] q_canyon               Canyon specific humidity [kg/kg]
 * @param[in] wind_speed_ref         Reference height wind speed [m/s]
 * @param[in] Ch_leaf                Leaf drag coefficient (for boundary layer resistance) [-]
 * @param[in] r_stomatal_leaf_s_per_m Bulk stomatal resistance [s/m]
 *
 * @return LE_crown [W/m²]  Positive = crown loses latent heat (cooling); clamped to [0, 500]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_crown_transpiration_LE(
    amrex::Real LAI,                     // [-]
    amrex::Real T_crown_lagged,          // [K]
    amrex::Real q_canyon,                // [kg/kg]
    amrex::Real wind_speed_ref,          // [m/s]
    amrex::Real Ch_leaf,                 // [-]
    amrex::Real r_stomatal_leaf_s_per_m) // [s/m]
{
    constexpr amrex::Real rho_air    = 1.2;           // Air density [kg/m³]
    constexpr amrex::Real Lv         = 2.5e6;         // Latent heat of vaporization [J/kg]
    constexpr amrex::Real U_min      = 1.e-6;         // Minimum wind speed guard [m/s]
    constexpr amrex::Real LE_max     = 500.0;         // Maximum LE to prevent blowup [W/m²]

    // Guard against unphysical LAI
    amrex::Real LAI_clamped = amrex::max(LAI, 0.0);

    // Guard against zero/negative wind (avoid division by zero)
    amrex::Real U = amrex::max(std::abs(wind_speed_ref), U_min);

    // Aerodynamic leaf boundary layer resistance: r_boundary = 1 / (Ch_leaf * U) [s/m]
    // Guard against very large resistance due to small Ch_leaf
    amrex::Real r_boundary = 1.0 / (Ch_leaf * U);
    if (r_boundary > 1.e6) r_boundary = 1.e6;

    // Bulk total resistance [s/m]
    amrex::Real r_total = r_stomatal_leaf_s_per_m + r_boundary;
    if (r_total < 1.e-6) r_total = 1.e-6;  // Guard division

    // Saturation specific humidity at crown temperature (lagged from previous timestep)
    amrex::Real q_sat_crown = qsat_magnus_crown(T_crown_lagged);

    // Clamp q_canyon to physically reasonable range [0, 0.02] kg/kg (matches green-roof line 364)
    amrex::Real q_canyon_clamped = amrex::max(amrex::min(q_canyon, 0.02), 0.0);

    // Specific humidity deficit [kg/kg]
    // Only transpire if q_sat > q_canyon; otherwise no driving gradient
    amrex::Real dq = amrex::max(q_sat_crown - q_canyon_clamped, 0.0);

    // Latent heat flux [W/m²] (positive = cooling)
    amrex::Real LE_crown = LAI_clamped * rho_air * Lv * dq / r_total;

    // Clamp to [0, 500] W/m² to prevent unphysical dew and edge-case blowup
    LE_crown = amrex::min(amrex::max(LE_crown, 0.0), LE_max);

    return LE_crown;
}

#endif // ERF_UCM_CROWN_TRANSPIRATION_H
