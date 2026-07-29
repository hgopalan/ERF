/**
 * @file ERF_UCMHVAC.H
 * @brief HVAC waste heat compute kernel (Phase 5.2, extended Phase 5.5, header-only)
 *
 * Provides the core compute_hvac_waste_heat() kernel for calculating HVAC
 * waste heat from slab-conduction residual cooling load, gated by occupancy
 * and setpoint. Phase 5.5 adds COP degradation with outdoor temperature and
 * sensible/latent split.
 *
 * References:
 *   - Phase 5.2 problem statement (HVAC waste heat, Contract #21)
 *   - Phase 5.5 problem statement (COP degradation, sensible/latent split)
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
 * 4. Degrade COP with outdoor temperature: COP_eff = COP * max(0.3, 1 - alpha * max(0, T_out - T_rated))
 * 5. Convert load to waste heat via effective COP: Q_HVAC = Q_load × (1 + 1/COP_eff)
 * 6. Split into sensible/latent: H = sensible_frac * Q_HVAC, LE = (1 - sensible_frac) * Q_HVAC
 *
 * Sanity checks (guaranteed by this function):
 * - Returns 0 if mode gates (occupancy, temperature) are off
 * - Q_HVAC_sensible, Q_HVAC_latent ≥ 0 always
 * - COP_eff clamped to ≥ 0.3 × rated to avoid extreme rejections at high T_outdoor
 *
 * @param[in]  H_wall_slab         Slab conduction flux through walls [W/m²]
 * @param[in]  H_roof_slab         Slab conduction flux through roof [W/m²]
 * @param[in]  T_can               Canyon air temperature [K]
 * @param[in]  T_setpoint          Indoor setpoint temperature [K]
 * @param[in]  hysteresis_K        Setpoint hysteresis (typ. 2 K) [K]
 * @param[in]  cop_rated           HVAC coefficient of performance at rating [-]
 * @param[in]  f_occ               Occupancy fraction (0–1) [-]
 * @param[in]  T_outdoor           Outdoor (canyon) air temperature for degradation [K] (Phase 5.5)
 * @param[in]  alpha_degradation   COP degradation slope [-/K] (Phase 5.5)
 * @param[in]  sensible_fraction   Fraction of Q_HVAC that is sensible (0-1) (Phase 5.5)
 * @param[out] Q_HVAC_sensible     Sensible waste heat rate [W/m²] (Phase 5.5)
 * @param[out] Q_HVAC_latent       Latent waste heat rate [W/m²] (Phase 5.5)
 * @return                         Total HVAC waste heat rate (sensible + latent) [W/m²]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_hvac_waste_heat(amrex::Real H_wall_slab, amrex::Real H_roof_slab,
                                     amrex::Real T_can, amrex::Real T_setpoint,
                                     amrex::Real hysteresis_K, amrex::Real cop_rated,
                                     amrex::Real f_occ,
                                     amrex::Real T_outdoor, amrex::Real alpha_degradation,
                                     amrex::Real sensible_fraction,
                                     amrex::Real& Q_HVAC_sensible, amrex::Real& Q_HVAC_latent)
{
    // Initialize outputs
    Q_HVAC_sensible = 0.0;
    Q_HVAC_latent = 0.0;

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

    // Phase 5.5: COP degradation with outdoor temperature
    // COP_effective = COP_rated * max(0.3, 1.0 - alpha * max(0, T_outdoor - T_rated))
    // T_rated = 308.15 K (ISO 5151 rating point, ~35°C)
    const amrex::Real T_rated = 308.15;
    const amrex::Real T_excess = amrex::max(0.0, T_outdoor - T_rated);
    const amrex::Real degradation_factor = amrex::max(0.3, 1.0 - alpha_degradation * T_excess);
    const amrex::Real cop_effective = cop_rated * degradation_factor;

    // Clamp effective COP to physical range (>= 1.0) to avoid division by zero
    const amrex::Real cop_safe = amrex::max(cop_effective, amrex::Real(1.0));

    // HVAC waste heat: Q_HVAC = Q_load * (1 + 1/COP)
    // This represents the heat rejected to outdoor air (load + compressor work)
    const amrex::Real Q_HVAC_total = Q_load * (1.0 + 1.0 / cop_safe);

    // Phase 5.5: Split into sensible and latent components
    Q_HVAC_sensible = sensible_fraction * Q_HVAC_total;
    Q_HVAC_latent = (1.0 - sensible_fraction) * Q_HVAC_total;

    // Ensure non-negative output (should always be true given above checks)
    return amrex::max(0.0, Q_HVAC_total);
}

#endif  // ERF_UCM_HVAC_H
