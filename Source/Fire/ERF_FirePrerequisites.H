#ifndef ERF_FIRE_PREREQUISITES_H
#define ERF_FIRE_PREREQUISITES_H

#include <ERF_FireParams.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <ERF_SurfacePrecipitation.H>

// Forward declaration
class SurfaceLayer;
class ERF;

/**
 * @file ERF_FirePrerequisites.H
 * @brief Fire module prerequisite checks at startup
 *
 * Verifies that ERF is configured correctly for fire simulation:
 * - MOST surface layer is active with proper boundary conditions
 * - MOST diagnostics (u*, z0, L, wind) are computed
 * - The level the fire grid refines exists, keeps its boxes and has no dust layer above level 0
 * - No z-direction MPI decomposition (all z-levels on same rank)
 * - The level's boxes cover one rectangle
 * - Grid divisibility by refinement factor
 * - The domain is deeper than the wind's sampling height (necessary, not
 *   sufficient: terrain is not included)
 *
 * All checks use AMREX_ALWAYS_ASSERT_WITH_MESSAGE for actionable diagnostics.
 */

/**
 * @brief The AMR level the fire grid refines: erf.fire.anchor_level, or the
 *        finest level when it is -1 (the default).
 *
 * Aborts, naming the input, when the level is above the finest level.
 *
 * @param[in] fire_params   Fire configuration parameters
 * @param[in] finest_level  Finest level of the run at start-up
 */
int fire_anchor_level(const FireParams& fire_params, int finest_level);

/**
 * @brief Start-up check of erf.fire.precip_source = atmosphere: the microphysics
 *        scheme must provide surface precipitation accumulators.
 *
 * Aborts, naming erf.fire.precip_source and erf.moisture_model, when the
 * sources are empty (no moisture model, Kessler_NoRain, saturation adjustment
 * only, ...). A no-op for the uniform source. Called after the microphysics of
 * the fire grid's level is initialised.
 *
 * @param[in] fire_params  Fire configuration parameters
 * @param[in] sources      Microphysics::Get_Surface_Precip_Accumulation_Ptrs(lev)
 */
void verify_fire_precip_source(const FireParams& fire_params,
                               const SurfacePrecipAccumulationSources& sources);

/**
 * @brief Start-up check of the level-set artificial viscosity against the
 *        linear stability limit of the substep (levelset_viscous_stability).
 *
 * Each viscosity is checked with the one-sided derivatives used where it
 * acts: erf.fire.levelset.eps_visc away from the front (first order with
 * gradient "upwind" and "weno5z_front", WENO5 with "weno5z"), and
 * erf.fire.levelset.eps_visc_front, when it is >= 0, near the front (WENO5
 * inside the band of "weno5z_front"). With "weno5z_front" the coefficient
 * just beyond the WENO band, eps_visc_front itself or a value of the blend
 * toward eps_visc, is also checked with first-order differences when it is
 * larger than eps_visc, and, with erf.fire.levelset.wall_extrapolate, every
 * WENO5 coefficient again with first-order differences (the stencil falls back
 * to them next to a non-burnable cell). Aborts, naming the inputs and the largest stable
 * viscosity and cfl, when a coefficient is unstable for any front direction
 * (levelset_substep_gain > 1): every curved front has a part facing the
 * least favourable direction. A no-op on the FARSITE path.
 *
 * The check takes R_max as the factor of the viscous term. With
 * erf.fire.directional_split_hamiltonian the term is R0 eps lap(phi), R0
 * the no-wind, no-slope rate, while the substep is still set by R_max, so
 * the check is conservative there by about R_max / R0.
 *
 * @param[in] fire_params  Fire configuration parameters (cfl > 0)
 * @param[in] dx, dy       Fire cell size [m]
 */
void verify_levelset_viscous_stability(const FireParams& fire_params,
                                       amrex::Real dx, amrex::Real dy);

/**
 * @brief Verify fire module prerequisites in ERF configuration
 *
 * Called once at startup from FireLayer::initialize() after SurfaceLayer is ready.
 * Aborts with clear diagnostic messages if any requirement is not met.
 *
 * @param[in] erf              ERF instance (for grids, dmap, Geom access)
 * @param[in] lev              AMR level the fire grid refines (fire_anchor_level)
 * @param[in] surface_layer    Pointer to initialized SurfaceLayer
 * @param[in] fire_params      Fire configuration parameters
 *
 * Checks performed (in order):
 * 1. SurfaceLayer pointer is not null (zlo.type = surface_layer)
 * 2. 0 <= lev <= finest level
 * 3. On a level above 0: static refinement (erf.regrid_int <= 0)
 * 4. On a level above 0: no dust layer (erf.dust.enable)
 * 7. Every box of the level spans its full height (no z-decomposition)
 * 7b. The level's boxes cover one rectangle in x and y
 * 8. grid_ratio >= 1
 * 9. All boxes divisible by grid_ratio in x and y
 * 10. Distribution map size matches grid size
 * 11. Domain z, x and y indices start at 0
 * 12. Domain depth (prob_hi - prob_lo in z) > the wind's sampling height (FireParams::sampled_wind_ht,
 *     or erf.fire.wind_sample_ht when set); terrain is not included
 * 13-14. Level-set and FARSITE divisors and step sizes, and the level-set
 *        artificial viscosity against the substep's stability limit
 *        (verify_levelset_viscous_stability)
 */
void verify_fire_prerequisites(const ERF& erf,
                               int lev,
                               const SurfaceLayer* surface_layer,
                               const FireParams& fire_params);

#endif
