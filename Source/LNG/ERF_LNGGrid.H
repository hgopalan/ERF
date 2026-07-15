/**
 * @file ERF_LNGGrid.H
 * @brief LNG grid management (2D or 3D slab)
 * @details
 * The LNGGrid struct manages the computational grid for LNG modeling.
 * In Phase 1, it is structured as a 2D slab (k=1 in z) on a refined horizontal grid.
 * Future phases may extend to full 3D if volumetric source terms are needed.
 * 
 * The grid is constructed from the atmospheric level-0 grid by:
 * 1. Extracting the k=0 slice (lowest atmosphere level)
 * 2. Refining horizontally by grid_ratio in x and y
 * 3. Creating a 2D geometry with appropriate index-space bounds
 * 
 * @note Phase 1 allocates only the geometry; MultiFabs are allocated in LNGLayer::initialize().
 * 
 * @see Source/Dust/ERF_DustGrid.H — Dust module grid pattern (reference)
 */

#ifndef ERF_LNG_GRID_H
#define ERF_LNG_GRID_H

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

/**
 * @struct LNGGrid
 * @brief Container for LNG computational grid (2D slab)
 * @details
 * Holds the distributed grid arrays and geometry for LNG fields.
 * All LNG MultiFabs are allocated using this BoxArray and DistributionMapping.
 */
struct LNGGrid {
    amrex::BoxArray           ba;           ///< 2D box array (nx_refined, ny_refined, nz=1)
    amrex::DistributionMapping dm;          ///< Distribution mapping (load balancing)
    amrex::Geometry           geom;         ///< 2D geometry with scaled physical domain
    int                       grid_ratio;   ///< Horizontal refinement factor from ATM grid
    
    /**
     * @brief Construct the LNG 2D grid from atmospheric level-0 grid
     * @param[in] atm_ba    Atmospheric BoxArray at level 0
     * @param[in] atm_dm    Atmospheric DistributionMapping at level 0
     * @param[in] atm_geom  Atmospheric Geometry at level 0
     * @param[in] ratio     Horizontal refinement factor (grid_ratio parameter)
     * @details
     * Algorithm:
     * 1. Extract k=0 domain bounds: ihi = atm_ba[0].bigEnd(0), etc.
     * 2. Create 2D boxes by:
     *    - Refining each ATM box: new_box = atm_box * ratio (with k=0 mapped to k=0 in 2D)
     *    - Setting nz=1 (single LNG slab layer)
     * 3. Create LNG 2D Geometry with:
     *    - Index-space: [0, ihi*ratio+1] x [0, jhi*ratio+1] x [0, 0] (k=1 only)
     *    - Physical-space: same x,y as ATM domain, z in [0, dz_LNG] where dz_LNG is grid cell height
     * 4. Reuse ATM DistributionMapping (refined boxes maintain load balance)
     */
    void build(const amrex::BoxArray& atm_ba,
               const amrex::DistributionMapping& atm_dm,
               const amrex::Geometry& atm_geom,
               int ratio);
};

#endif /* ERF_LNG_GRID_H */
