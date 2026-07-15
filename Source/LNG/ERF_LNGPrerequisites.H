/**
 * @file ERF_LNGPrerequisites.H
 * @brief Pre-initialization validation for LNG module
 * @details
 * Declares the prerequisite checking function and initialization summary printer.
 * 
 * @note Phase 1 checks:
 *   1. grid_ratio >= 1
 *   2. ATM boxes divisible by grid_ratio in x,y
 *   3. LNG composition mole fractions sum to [0.99, 1.01]
 *   4. lfl_vol_fraction < ufl_vol_fraction
 *   5. spill_rate_kg_s >= 0
 *   6. All print initialization summary if verbose >= 1
 * 
 * @see Source/Dust/ERF_DustPrerequisites.H — Dust module prerequisite pattern (reference)
 */

#ifndef ERF_LNG_PREREQUISITES_H
#define ERF_LNG_PREREQUISITES_H

#include "ERF_LNGParams.H"
#include <AMReX_Geometry.H>

/**
 * @brief Validate LNG configuration and print initialization summary
 * @param[in] params       LNG parameters
 * @param[in] geom_atm     Atmospheric level-0 geometry
 * @details
 * Performs 5 prerequisite checks. If any fatal check fails, calls amrex::Abort()
 * with descriptive error message. Warnings are printed but do not halt execution.
 * 
 * If verbose >= 1, prints an initialization summary block:
 * ```
 * [LNG] ============================================================
 * [LNG] ERF-LNG Phase 1 initialized
 * [LNG]   Pool area       : 100.00 m^2
 * [LNG]   Spill rate      : 10.00 kg/s
 * [LNG]   LNG composition : CH4=0.90  C2H6=0.08  N2=0.02
 * [LNG]   Mol. weight     : 17.40 g/mol
 * [LNG]   Boiling point   : 111.70 K
 * [LNG]   LFL/UFL         : 0.050 / 0.150 (vol/vol)
 * [LNG]   Grid ratio      : 1
 * [LNG]   ATM feedback    : 1.00
 * [LNG]   Debug mode      : on
 * [LNG]   Verbose level   : 2
 * [LNG] ============================================================
 * ```
 * 
 * @throws amrex::Abort if fatal prerequisites violated
 */
void check_lng_prerequisites(const LNGParams& params,
                              const amrex::Geometry& geom_atm);

#endif /* ERF_LNG_PREREQUISITES_H */
