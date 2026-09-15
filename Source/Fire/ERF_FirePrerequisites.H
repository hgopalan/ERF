#ifndef ERF_FIRE_PREREQUISITES_H
#define ERF_FIRE_PREREQUISITES_H

#include <ERF_FireParams.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>

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
 * - Domain height is sufficient for wind extraction
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
 * 12. Domain height > wind_ref_ht
 * 13-14. Level-set and FARSITE divisors and step sizes
 */
void verify_fire_prerequisites(const ERF& erf,
                               int lev,
                               const SurfaceLayer* surface_layer,
                               const FireParams& fire_params);

#endif
