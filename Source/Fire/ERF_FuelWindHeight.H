#ifndef ERF_FUEL_WIND_HEIGHT_H
#define ERF_FUEL_WIND_HEIGHT_H

#include <vector>
#include <AMReX_REAL.H>
#include <ERF_FuelModels.H>

/**
 * @file ERF_FuelWindHeight.H
 * @brief Per-fuel wind height (fcwh) table
 *
 * fcwh[k]: height [m] above terrain at which wind is extracted for fuel slot k
 *          (equivalent to WRF-SFIRE fcwh per fuel category). Slot 0 is the
 *          non-burnable slot and is unused; valid range 1..FUEL_SLOT_COUNT-1.
 *
 * Default values follow WRF-SFIRE namelist.fire defaults: fcwh is 6.096 m
 * (20 ft) for every fuel model, consistent with the BEHAVE midflame height
 * default.
 *
 * Surface roughness is not carried per fuel model. The roughness the fire model
 * uses comes from the surface layer (erf.most.z0) and, for the two-height log
 * interpolation of the sampled wind, from erf.fire.wind_sample_z0.
 *
 * References:
 *   Andrews, P.L. (1986). BEHAVE: Fire Behavior Prediction and Fuel Modeling
 *     System. USDA Forest Service GTR INT-194.
 *   WRF-SFIRE module_fr_sfire_phys.F, fcwh data statement.
 */

/**
 * @brief Build the per-fuel wind height (fcwh) table, one entry per fuel slot
 *
 * @param[in] global_z_ref    Global fallback wind reference height [m]
 * @param[in] use_per_fuel    When true, use WRF-SFIRE fcwh defaults per fuel;
 *                            when false, all entries equal global_z_ref
 * @return host-side vector of size FUEL_SLOT_COUNT; slot 0 unused,
 *         slots 1..FUEL_SLOT_COUNT-1 valid
 *
 * Slot 0 is the non-burnable slot and is left at zero.
 * When use_per_fuel=false, every other entry is set to global_z_ref
 * (backward-compatible, zero cost).
 * When use_per_fuel=true, every other entry is set to the WRF-SFIRE default
 * (6.096 m).
 */
inline std::vector<amrex::Real> build_fcwh_table(
    amrex::Real global_z_ref,
    bool use_per_fuel = false)
{
    // Sized for every fuel slot (1-13 Anderson, 14-53 Scott-Burgan,
    // 54-69 deck-defined); all of them take the Anderson heights.
    std::vector<amrex::Real> fcwh(FUEL_SLOT_COUNT, 0.0);

    if (use_per_fuel) {
        // WRF-SFIRE defaults: all fuel models use 6.096 m (20 ft)
        for (int i = 1; i < FUEL_SLOT_COUNT; ++i) {
            fcwh[i] = 6.096;
        }
    } else {
        // Uniform: all entries use global_z_ref
        for (int i = 1; i < FUEL_SLOT_COUNT; ++i) {
            fcwh[i] = global_z_ref;
        }
    }

    return fcwh;
}

#endif
