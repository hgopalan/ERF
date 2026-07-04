/**
 * \file ERF_FireGrid.H
 *
 * \brief Fire simulation grid refinement and decomposition.
 *
 * Defines the FireGrid data structure that holds the refined fire simulation
 * grid (C× finer in x and y than atmospheric). Provides utilities to create
 * the fire grid from the atmospheric grid while preserving MPI decomposition.
 *
 * References:
 * - AMReX BoxArray and DistributionMapping documentation
 *
 * Ported from concepts in: hgopalan/wildfire_levelset/src/terrain_slope.H
 */

#ifndef ERF_FIRE_GRID_H
#define ERF_FIRE_GRID_H

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>

/**
 * \struct FireGrid
 *
 * Holds grid structures for the fire simulation domain (C× refinement in x,y).
 * The fire grid is a 2D slab (z-direction length = 1) on the atmospheric level 0.
 * Each MPI rank owns the fire cells corresponding to its atmospheric columns.
 */
struct FireGrid
{
    amrex::BoxArray              ba;        ///< Boxes: C*nx × C*ny × 1 (2D slab)
    amrex::DistributionMapping   dm;        ///< Same (i,j) ownership as atmospheric level-0
    amrex::Geometry              geom;      ///< 2D physical domain [m]
    int                          C = 1;     ///< Refinement factor (x,y only)
};

/**
 * \brief Create a refined fire grid from atmospheric grid.
 *
 * Creates a 2D fire grid (C× finer in x and y) that preserves the MPI
 * decomposition of the atmospheric grid. Each rank owns the fire cells
 * for its atmospheric columns.
 *
 * \param ba_atm           Atmospheric level-0 BoxArray
 * \param dm_atm           Atmospheric level-0 DistributionMapping
 * \param geom_atm         Atmospheric Geometry
 * \param C                Refinement factor (fire grid is C× finer in x,y)
 * \return                 FireGrid with refined boxes, mapping, and 2D geometry
 *
 * Algorithm:
 * 1. Extract k=0 (bottom) slice from ba_atm to get 2D boxes
 * 2. Refine by C in x,y (z stays 1)
 * 3. Create DistributionMapping by refining dm_atm by {C, C, 1}
 * 4. Build 2D Geometry from geom_atm
 */
FireGrid create_fire_grid(const amrex::BoxArray& ba_atm,
                          const amrex::DistributionMapping& dm_atm,
                          const amrex::Geometry& geom_atm,
                          int C);

#endif // ERF_FIRE_GRID_H
