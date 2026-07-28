/**
 * @file ERF_UCMRadiationExtraction.H
 * @brief Placeholder stub for Phase 4.3 real radiation extraction
 *
 * When `erf.ucm.radiation_source = erf`, the SEB radiation path calls
 * `extract_radiation_from_erf` to obtain SW_down and LW_down at UCM cell centers
 * from ERF's radiation module.
 *
 * **THIS FILE IS A PLACEHOLDER.** The extraction is not yet implemented because
 * ERF's radiation interface is undergoing upstream changes. The stub:
 *   - Returns sentinel values (-999.0) for both SW and LW
 *   - Prints a one-time warning on the IO rank
 *   - Does NOT touch any ERF core radiation state
 *
 * When RRTMG is stable, replace the stub body with actual extraction:
 *   1. Access ERF's radiation solver output arrays (SW_down, LW_down at klo)
 *   2. Interpolate/copy to UCM cell centers if grid ratio > 1
 *   3. Convert units if needed (ERF radiation may use different conventions)
 *   4. Delete the sentinel + warning
 *
 * References:
 *   - Phase 4.3 problem statement (this task)
 *   - Phase 3.5b: analytic path (backward-compatible default)
 *   - Phase 4.2: cloud attenuation layer (still applies to analytic path)
 */

#ifndef ERF_UCM_RADIATION_EXTRACTION_H
#define ERF_UCM_RADIATION_EXTRACTION_H

#include <AMReX_REAL.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

/**
 * @brief Extract SW_down and LW_down from ERF radiation module (PLACEHOLDER)
 *
 * @param[out] SW_down    Downwelling shortwave [W/m²]
 * @param[out] LW_down    Downwelling longwave  [W/m²]
 * @param[in]  time_s     Simulation time [s] (unused in stub)
 *
 * @note Placeholder returns sentinel values (-999.0) and prints one-time warning.
 *       Real implementation deferred until RRTMG interface stabilizes.
 */
inline void extract_radiation_from_erf(amrex::Real& SW_down,
                                        amrex::Real& LW_down,
                                        amrex::Real time_s)
{
    amrex::ignore_unused(time_s);

    // Sentinel values — clearly identifiable as "not real"
    SW_down = -999.0;
    LW_down = -999.0;

    // One-time warning
    static bool warned = false;
    if (!warned && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print()
            << "[UCM][4.3][PLACEHOLDER] Real radiation extraction is not yet "
            << "implemented. Returning sentinel SW=-999.0, LW=-999.0.\n"
            << "[UCM][4.3][PLACEHOLDER] For working radiation, set "
            << "erf.ucm.radiation_source=analytic (default).\n"
            << "[UCM][4.3][PLACEHOLDER] Full RRTMG extraction deferred until "
            << "ERF core radiation interface stabilizes.\n";
        warned = true;
    }
}

#endif  // ERF_UCM_RADIATION_EXTRACTION_H
