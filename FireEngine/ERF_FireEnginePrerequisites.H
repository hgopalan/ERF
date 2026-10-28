#ifndef ERF_FIRE_ENGINE_PREREQUISITES_H
#define ERF_FIRE_ENGINE_PREREQUISITES_H

#include <ERF_FireParams.H>
#include <AMReX_REAL.H>

class ERF;

/**
 * @file ERF_FireEnginePrerequisites.H
 * @brief Start-up checks of the native fire engine's own inputs
 *
 * Run by FireLayer::verify_prerequisites(), which FireCoupler::initialize()
 * calls after the shared checks of the coupling (ERF_FirePrerequisites.H): the domain against the engine's wind sampling
 * height, the level-set and FARSITE divisors and step sizes, and the level-set
 * artificial viscosity against the substep's stability limit.
 */

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
 * @brief Verify the native fire engine's inputs against the grid it runs on
 *
 * Called once at startup from FireLayer::verify_prerequisites(), after the
 * shared checks of verify_fire_prerequisites() and before the fire grid is built. Aborts with clear diagnostic messages
 * if any requirement is not met.
 *
 * @param[in] erf          ERF instance (for Geom access)
 * @param[in] lev          AMR level the fire grid refines
 * @param[in] fire_params  Fire configuration parameters
 *
 * Checks performed (in order, numbered on from the shared checks):
 * 12. Domain depth (prob_hi - prob_lo in z) > the wind's sampling height (FireParams::sampled_wind_ht,
 *     or erf.fire.wind_sample_ht when set); terrain is not included
 * 13-14. Level-set and FARSITE divisors and step sizes, and the level-set
 *        artificial viscosity against the substep's stability limit
 *        (verify_levelset_viscous_stability)
 */
void verify_native_fire_prerequisites(const ERF& erf,
                                      int lev,
                                      const FireParams& fire_params);

#endif
