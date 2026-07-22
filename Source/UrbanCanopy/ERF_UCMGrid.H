/**
 * @file ERF_UCMGrid.H
 * @brief Definition of the SLUCM computational grid
 *
 * The SLUCM grid is a 2D horizontal slab covering the same physical domain
 * as the ERF level-0 atmospheric grid, optionally refined by grid_ratio in
 * x and y directions. The vertical extent is a dummy 1 m slab at k=0.
 *
 * MPI decomposition follows the atmospheric layout to minimize inter-rank
 * communication during coupling operations. Each rank owns identically-
 * positioned horizontal tiles in both ATM and UCM grids.
 *
 * References:
 *  - Source/Fire/ERF_FireGrid.H
 *  - Source/Dust/ERF_DustGrid.H
 *  - AMReX BoxArray and Geometry documentation
 */

#ifndef ERF_UCM_GRID_H
#define ERF_UCM_GRID_H

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>

/**
 * @struct UCMGrid
 * @brief Container for the SLUCM computational grid infrastructure
 *
 * Holds all necessary AMR/geometric objects describing the 2D SLUCM grid.
 * The grid is a horizontal slab (k-extent = 1 cell) with x-y extent
 * optionally refined by grid_ratio relative to the ATM level 0 grid.
 */
struct UCMGrid
{
    amrex::BoxArray            ba;          ///< 2D BoxArray (refined nx × ny × 1)
    amrex::DistributionMapping dm;          ///< MPI distribution (aligned with ATM layout)
    amrex::Geometry            geom;        ///< 2D Geometry in physical coordinates [m]
    int                        grid_ratio;  ///< Horizontal refinement factor (unitless)
    int                        lev;         ///< ATM level this grid was built from
};

/**
 * @brief Construct a 2D SLUCM grid from the atmospheric level-0 grid
 *
 * Builds a refined 2D horizontal slab covering the same physical domain as
 * the atmospheric level-0 grid, with optional refinement in x-y by grid_ratio.
 * The MPI decomposition mirrors the atmospheric layout to ensure each rank owns
 * identically-positioned horizontal tiles, avoiding inter-rank communication.
 *
 * Algorithm:
 *  1. Extract k=0 slab from each box in ba_atm by explicit Box manipulation
 *     (setSmall(2, 0); setBig(2, 0)) to create a proper 2D BoxArray.
 *  2. Refine the 2D BoxArray by IntVect{grid_ratio, grid_ratio, 1} using
 *     amrex::refine() global function.
 *  3. Reuse dm_atm directly: amrex::refine() preserves the number of boxes
 *     (only changes box sizes), so box i in refined array is owned by the same
 *     rank as box i in atmospheric array.
 *  4. Construct refined 2D Geometry:
 *     - Index-space domain: hi = old_hi * grid_ratio + (grid_ratio - 1)
 *     - Keep x-y physical extent from geom_atm
 *     - Set z physical extent to 1 m as dummy value
 *     - Use cartesian coordinates with {false, false, false} periodic flags
 *
 * Domain decomposition:
 *  Reusing dm_atm after refinement ensures each rank owns identically-
 *  positioned horizontal tiles in both grids, eliminating inter-rank
 *  communication overhead during coupling operations.
 *
 * References:
 *  - AMReX BoxArray and refinement:
 *    https://amrex-codes.github.io/amrex/docs_html/BoxArray.html
 *  - AMReX makeSlab utility:
 *    https://amrex-codes.github.io/amrex/docs_html/utilities_html
 *
 * @param[in] ba_atm        Atmospheric level-0 BoxArray
 * @param[in] dm_atm        Atmospheric level-0 DistributionMapping
 * @param[in] geom_atm      Atmospheric level-0 Geometry
 * @param[in] grid_ratio    Horizontal refinement factor (>= 1, default 1)
 * @param[in] lev           ATM level (default 0; Phase 1.1 only uses lev=0)
 * @return                  UCMGrid with refined 2D domain
 */
UCMGrid create_ucm_grid(const amrex::BoxArray&          ba_atm,
                        const amrex::DistributionMapping& dm_atm,
                        const amrex::Geometry&          geom_atm,
                        int                             grid_ratio,
                        int                             lev = 0,
                        bool                            ucm_debug = false);

#endif // ERF_UCM_GRID_H
