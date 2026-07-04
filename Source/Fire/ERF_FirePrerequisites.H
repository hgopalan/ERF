/**
 * \file ERF_FirePrerequisites.H
 *
 * \brief Validation checks for fire module prerequisites.
 *
 * Verifies that the ERF configuration and SurfaceLayer setup meet requirements
 * for the fire module to function correctly. All checks must pass before
 * fire simulation can proceed.
 *
 * Key assumptions:
 * - Single AMR level (level 0 only)
 * - No z-direction MPI decomposition (each rank holds all k-levels)
 * - SurfaceLayer with MOST active
 * - Surface roughness set
 * - Wind and flux data available
 *
 * Ported concepts from: hgopalan/wildfire_levelset validation patterns
 */

#ifndef ERF_FIRE_PREREQUISITES_H
#define ERF_FIRE_PREREQUISITES_H

#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Print.h>

#include "ERF_SurfaceLayer.H"
#include "ERF_FireParams.H"

/**
 * \brief Verify that fire module prerequisites are satisfied.
 *
 * Performs validation in the following order. Each failure calls
 * AMREX_ALWAYS_ASSERT_WITH_MESSAGE with an actionable error message.
 *
 * \param phys_bc_type   Boundary condition types (from ERF)
 * \param surface_layer  Pointer to SurfaceLayer (may be nullptr)
 * \param grids          Vector of BoxArrays (level 0 used)
 * \param dmap           Vector of DistributionMappings (level 0 used)
 * \param geom           Vector of Geometries (level 0 used)
 * \param fire_params    Fire parameters struct
 * \param lev            AMR level to check (default 0)
 *
 * Checks (in order):
 * 1. Surface layer BC type is set to "surface_layer" (not others like "wall")
 * 2. SurfaceLayer pointer is not null
 * 3. u_star field has been allocated and computed
 * 4. z0 (roughness) field has been set
 * 5. olen (Obukhov length) field has been computed
 * 6. MAC-averaged velocity components are available
 * 7. No z-direction domain decomposition (MPI z-split check):
 *    Each box must have z-extent equal to full domain z-extent
 * 8. grid_ratio parameter is valid (>= 1)
 * 9. All box x,y lengths are divisible by grid_ratio
 * 10. DistributionMapping consistency check
 * 11. Domain z-index starts at 0 (AMR level 0 at ground)
 * 12. Domain height >= wind reference height (6.1 m default)
 */
void verify_fire_prerequisites(
    const amrex::Vector<amrex::BCRec>& phys_bc_type,
    const SurfaceLayer* surface_layer,
    const amrex::Vector<amrex::BoxArray>& grids,
    const amrex::Vector<amrex::DistributionMapping>& dmap,
    const amrex::Vector<amrex::Geometry>& geom,
    const FireParams& fire_params,
    int lev = 0);

#endif // ERF_FIRE_PREREQUISITES_H
