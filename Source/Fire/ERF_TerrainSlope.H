/**
 * \file ERF_TerrainSlope.H
 *
 * \brief Computation of terrain slopes for fire module.
 *
 * Computes terrain slopes (dz/dx, dz/dy) at fire grid cell centers via
 * bilinear interpolation from coarser nodal terrain heights.
 *
 * References:
 * - Slope computation via centered finite differences on staggered grids
 *
 * Ported from: hgopalan/wildfire_levelset/src/terrain_slope.H
 */

#ifndef ERF_FIRE_TERRAIN_SLOPE_H
#define ERF_FIRE_TERRAIN_SLOPE_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>

#include "ERF_FireGrid.H"

/**
 * \brief Compute terrain slopes on the fire grid.
 *
 * Reads nodal terrain heights from the coarser atmospheric grid and computes
 * slope components (dz/dx, dz/dy) at fire grid cell centers via bilinear
 * interpolation followed by centered differences.
 *
 * This function is called once at initialization and the result is stored
 * in a static fire_slopes MultiFab for reuse across fire timesteps.
 *
 * \param fire_slopes   Output: 2 comp (dz/dx, dz/dy), fire grid resolution
 * \param z_phys_nd     Input: ERF nodal terrain height [m], atmospheric grid
 * \param geom_atm      Atmospheric Geometry
 * \param fg            FireGrid structure (for refinement factor and geometry)
 *
 * Algorithm:
 * 1. Interpolate z_phys_nd from nodal (atmospheric) to cell-centered (fire)
 * 2. Compute centered differences of interpolated heights for slopes
 * 3. Store slopes in fire_slopes MultiFab (1-2 layers of ghost cells may be needed)
 *
 * Notes:
 * - z_phys_nd is on the nodal grid of the atmospheric domain
 * - Fire grid is C× finer in x and y
 * - Slopes computed using 2-point centered differences
 */
void compute_terrain_slopes(
    amrex::MultiFab& fire_slopes,
    const amrex::MultiFab& z_phys_nd,
    const amrex::Geometry& geom_atm,
    const FireGrid& fg);

#endif // ERF_FIRE_TERRAIN_SLOPE_H
