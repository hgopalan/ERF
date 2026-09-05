#ifndef ERF_FIRE_GRID_H
#define ERF_FIRE_GRID_H

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>

/**
 * @file ERF_FireGrid.H
 * @brief Fire grid data structure and construction
 *
 * Defines a 2D grid for fire simulation on the horizontal plane, refined by
 * factor C relative to the atmospheric level-0 grid. The fire grid maintains
 * x-y domain decomposition aligned with the atmospheric MPI layout.
 */

/**
 * @struct FireGrid
 * @brief 2D horizontal fire computational grid
 *
 * Represents a 2D slab (kmax=1) with dimensions C*nx × C*ny × 1,
 * where C is the refinement factor and (nx, ny) is the atmospheric grid size.
 */
struct FireGrid
{
    amrex::BoxArray              ba;   ///< 2D BoxArray (C*nx × C*ny × 1)
    amrex::DistributionMapping   dm;   ///< Distribution map (x-y decomposition aligned with atm)
    amrex::Geometry              geom; ///< 2D Geometry in physical coordinates [m]
    int                          C;    ///< Refinement factor
};

/**
 * @brief Construct a refined 2D fire grid from atmospheric level-0
 *
 * @param[in] ba_atm       Atmospheric level-0 BoxArray
 * @param[in] dm_atm       Atmospheric level-0 DistributionMapping
 * @param[in] geom_atm     Atmospheric level-0 Geometry
 * @param[in] C            Refinement factor (grid_ratio from FireParams)
 *
 * @return FireGrid with refined 2D domain
 *
 * Algorithm:
 * 1. Extract k=0 2D slice from each box in ba_atm using makeSlab()
 * 2. Refine the 2D BoxArray by {C, C, 1} using amrex::refine()
 * 3. Refine DistributionMapping by {C, C, 1} to align ownership
 * 4. Create 2D Geometry from geom_atm (drop z extent, keep x-y physical domain)
 */
FireGrid create_fire_grid(const amrex::BoxArray& ba_atm,
                          const amrex::DistributionMapping& dm_atm,
                          const amrex::Geometry& geom_atm,
                          int C);

#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>

/**
 * Exchange the ghost cells of a fire-grid MultiFab and fill the ones outside a
 * non-periodic face of the domain by copying the nearest interior value.
 *
 * FillBoundary alone leaves those cells untouched, so with an inflow-outflow
 * atmosphere (the fire grid inherits its periodicity) the WENO stencil of the
 * level set read whatever the allocator had left there. Whether that was
 * benign depended on the box layout: one box per rank happened to run, two
 * boxes on a rank blew the level set up at the inflow face within three steps.
 * Every fire-grid exchange goes through here.
 */
inline void fire_fill_boundary (amrex::MultiFab& mf, const amrex::Geometry& geom)
{
    mf.FillBoundary(geom.periodicity());
    const auto ng = mf.nGrowVect();
    const bool px = geom.isPeriodic(0) || ng[0] == 0;
    const bool py = geom.isPeriodic(1) || ng[1] == 0;
    const bool pz = geom.isPeriodic(2) || ng[2] == 0;
    if (px && py && pz) { return; }   // nothing outside a non-periodic face to fill
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

#endif
