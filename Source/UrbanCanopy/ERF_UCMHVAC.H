/**
 * @file ERF_UCMHVAC.H
 * @brief HVAC waste heat compute kernel (Phase 5.2, header-only)
 *
 * Provides the core compute_hvac_waste_heat() kernel for calculating HVAC
 * waste heat from slab-conduction residual cooling load, gated by occupancy
 * and setpoint.
 *
 * References:
 *   - Phase 5.2 problem statement (HVAC waste heat, Contract #21)
 *   - ERF_UCMRadiosity.H (similar header-only pattern)
 */

#ifndef ERF_UCM_HVAC_H
#define ERF_UCM_HVAC_H

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

/**
 * @brief Compute HVAC waste heat from slab-conduction residual cooling load
 *
 * Physical model:
 * 1. Compute cooling load from residual slab conduction heat flux (walls + roof)
 * 2. Gate by occupancy: if unoccupied (f_occ ≤ 0), HVAC off
 * 3. Gate by setpoint: if T_canyon >= T_setpoint - hysteresis, HVAC off (no cooling needed)
 * 4. Convert load to waste heat via COP: Q_HVAC = Q_load × (1 + 1/COP)
 *
 * Sanity checks (guaranteed by this function):
 * - Returns 0 if mode gates (occupancy, temperature) are off
 * - Q_HVAC ≥ 0 always (HVAC only injects heat, never removes it)
 * - COP clamped to ≥ 1.0 to avoid division by zero or negative waste heat
 *
 * @param[in]  H_wall_slab   Slab conduction flux through walls [W/m²]
 * @param[in]  H_roof_slab   Slab conduction flux through roof [W/m²]
 * @param[in]  T_can         Canyon air temperature [K]
 * @param[in]  T_setpoint    Indoor setpoint temperature [K]
 * @param[in]  hysteresis_K  Setpoint hysteresis (typ. 2 K) [K]
 * @param[in]  cop           HVAC coefficient of performance [-]
 * @param[in]  f_occ         Occupancy fraction (0–1) [-]
 * @return                   HVAC waste heat rate injected as AH [W/m²]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_hvac_waste_heat(amrex::Real H_wall_slab, amrex::Real H_roof_slab,
                                     amrex::Real T_can, amrex::Real T_setpoint,
                                     amrex::Real hysteresis_K, amrex::Real cop,
                                     amrex::Real f_occ)
{
    // Gate 1: Setpoint-based gate (HVAC off if canyon is cold enough)
    if (T_can < T_setpoint - hysteresis_K) {
        return 0.0;
    }

    // Gate 2: Occupancy-based gate (HVAC off if unoccupied)
    if (f_occ <= 0.0) {
        return 0.0;
    }

    // Compute cooling load: occupancy-weighted sum of wall and roof slab conduction fluxes
    // Only positive (downward/inward) heat fluxes contribute to cooling load
    const amrex::Real Q_load = f_occ * amrex::max(0.0, H_wall_slab + H_roof_slab);

    // If cooling load is zero or negative, HVAC is off
    if (Q_load <= 0.0) {
        return 0.0;
    }

    // Clamp COP to physical range (>= 1.0) to avoid division by zero
    const amrex::Real cop_safe = amrex::max(cop, amrex::Real(1.0));

    // HVAC waste heat: Q_HVAC = Q_load * (1 + 1/COP)
    // This represents the heat rejected to outdoor air (load + compressor work)
    const amrex::Real Q_HVAC = Q_load * (1.0 + 1.0 / cop_safe);

    // Ensure non-negative output (should always be true given above checks)
    return amrex::max(0.0, Q_HVAC);
}

#endif  // ERF_UCM_HVAC_H
