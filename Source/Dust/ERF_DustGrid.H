/**
 * @file ERF_DustGrid.H
 * @brief Definition of the 2D dust emission grid and factory function.
 *
 * The dust grid is a horizontal slab covering the same physical domain as the
 * ERF level-0 atmospheric grid, optionally refined by grid_ratio in x and y.
 * The MPI decomposition follows the atmospheric DistributionMapping to avoid
 * inter-rank communication during coupling phases.
 *
 * References:
 *   AMReX BoxArray refinement:
 *     https://amrex-codes.github.io/amrex/docs_html/BoxArray.html
 *   AMReX makeSlab utility:
 *     https://amrex-codes.github.io/amrex/docs_html/utilities_html
 */

#ifndef ERF_DUST_GRID_H
#define ERF_DUST_GRID_H

#include <AMReX_BoxArray.H>
#include <AMReX_MultiFab.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>

/**
 * @struct DustGrid
 * @brief Container for the 2D dust emission grid infrastructure.
 *
 * Holds the BoxArray, DistributionMapping, Geometry, and refinement factor
 * of the 2D dust grid. The 2D slab has k-extent = 1 (k ∈ [0,0]) and x-y
 * extent refined by grid_ratio relative to the atmospheric grid.
 */
struct DustGrid
{
    amrex::BoxArray           ba;          ///< 2D BoxArray (grid_ratio·nx × grid_ratio·ny × 1)
    amrex::DistributionMapping dm;         ///< Distribution map aligned with atmospheric layout
    amrex::Geometry           geom;        ///< 2D geometry in physical coordinates [m]
    int                       grid_ratio;  ///< Refinement factor
};

/**
 * @brief Construct a 2D dust emission grid from the atmospheric level-0 grid.
 *
 * The dust grid is a horizontal slab (k-extent = 1) covering the same
 * physical domain as the atmospheric level-0 grid, optionally refined by
 * grid_ratio in x and y. The MPI decomposition follows the atmospheric
 * DistributionMapping so that each rank owns the same horizontal tiles in
 * both grids.
 *
 * Algorithm:
 *   1. Extract k=0 slab from each box in ba_atm using makeSlab and explicit
 *      box iteration to construct a proper 2D BoxArray.
 *   2. Refine the 2D BoxArray by IntVect{grid_ratio, grid_ratio, 1} using
 *      the global amrex::refine() function.
 *   3. Reuse dm_atm directly (refinement does not change the number of boxes,
 *      only their size, so box i in the refined array is owned by the same
 *      rank as box i in the atmospheric array).
 *   4. Construct a refined 2D Geometry whose index-space domain spans
 *      [0, grid_ratio*nx - 1] x [0, grid_ratio*ny - 1] x [0, 0], computed as
 *      hi = old_hi * grid_ratio + (grid_ratio - 1) for proper zero-based
 *      indexing. Keep the same physical (RealBox) extent as geom_atm in x-y;
 *      the z physical extent is set to 1 m as a dummy value (the dust grid
 *      is 2D). Use explicit Geometry constructor with cartesian coordinates
 *      and periodic flags.
 *
 * Domain decomposition:
 *   The distribution mapping dm_atm is reused after refinement because
 *   amrex::refine() on a BoxArray preserves the number of boxes. This ensures
 *   each rank owns identically-positioned horizontal tiles in both grids,
 *   avoiding inter-rank communication during coupling.
 *
 * References:
 *   AMReX BoxArray refinement:
 *     https://amrex-codes.github.io/amrex/docs_html/BoxArray.html
 *   AMReX makeSlab utility:
 *     https://amrex-codes.github.io/amrex/docs_html/utilities_html
 *
 * @param[in] ba_atm       Atmospheric level-0 BoxArray.
 * @param[in] dm_atm       Atmospheric level-0 DistributionMapping.
 * @param[in] geom_atm     Atmospheric level-0 Geometry.
 * @param[in] grid_ratio   Horizontal refinement factor (>= 1).
 * @return DustGrid with refined 2D domain.
 */
DustGrid create_dust_grid(const amrex::BoxArray&          ba_atm,
                          const amrex::DistributionMapping& dm_atm,
                          const amrex::Geometry&          geom_atm,
                          int                             grid_ratio);

/**
 * @brief FillBoundary plus a nearest-interior fill of the ghost cells outside a
 *        non-periodic face, the dust-grid twin of the fire grid's helper.
 *
 * FillBoundary leaves the cells outside a non-periodic domain face untouched, so
 * a stencil read there sees whatever the field was allocated or last set with
 * (the slopes were setVal(0), which made the curvature of the outermost row and
 * column at an outflow face -s/(2 dx) instead of 0). Every dust-grid exchange of
 * a field read with a stencil goes through here.
 */
inline void dust_fill_boundary (amrex::MultiFab& mf, const amrex::Geometry& geom)
{
    mf.FillBoundary(geom.periodicity());
    const auto ng = mf.nGrowVect();
    const bool px = geom.isPeriodic(0) || ng[0] == 0;
    const bool py = geom.isPeriodic(1) || ng[1] == 0;
    const bool pz = geom.isPeriodic(2) || ng[2] == 0;
    if (px && py && pz) { return; }
    const amrex::Box& dom = geom.Domain();
    const auto dlo = dom.smallEnd();
    const auto dhi = dom.bigEnd();
    const int ncomp = mf.nComp();
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box& gbx = mfi.fabbox();
        if (dom.contains(gbx)) { continue; }
        auto a = mf.array(mfi);
        amrex::ParallelFor(gbx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            int ic = i, jc = j, kc = k;
            if (!px) { ic = amrex::min(amrex::max(i, dlo[0]), dhi[0]); }
            if (!py) { jc = amrex::min(amrex::max(j, dlo[1]), dhi[1]); }
            if (!pz) { kc = amrex::min(amrex::max(k, dlo[2]), dhi[2]); }
            if (ic != i || jc != j || kc != k) { a(i, j, k, n) = a(ic, jc, kc, n); }
        });
    }
}

#endif // ERF_DUST_GRID_H
