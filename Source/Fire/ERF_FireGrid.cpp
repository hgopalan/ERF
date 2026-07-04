/**
 * \file ERF_FireGrid.cpp
 *
 * \brief Implementation of fire grid creation and refinement.
 */

#include "ERF_FireGrid.H"
#include <AMReX_Orientation.H>
#include <AMReX_Print.H>

FireGrid create_fire_grid(const amrex::BoxArray& ba_atm,
                          const amrex::DistributionMapping& dm_atm,
                          const amrex::Geometry& geom_atm,
                          int C)
{
    FireGrid fg;
    fg.C = C;

    // 1. Extract k=0 slice from atmospheric BoxArray
    //    We want to get a 2D BoxArray by taking just the k=0 plane
    int kmax = geom_atm.Domain().length(2);
    int kmin = geom_atm.Domain().smallEnd(2);
    
    amrex::BoxArray ba_2d(ba_atm);
    
    // Convert each 3D box to a 2D box by setting z-range to [kmin, kmin+1)
    for (int i = 0; i < ba_2d.size(); ++i) {
        amrex::Box b = ba_2d[i];
        b.setSmall(2, kmin);
        b.setBig(2, kmin);
        ba_2d.set(i, b);
    }

    // 2. Refine the 2D BoxArray by C in x and y (z stays as single-cell slab)
    amrex::IntVect ref_ratio(C, C, 1);
    fg.ba = amrex::refine(ba_2d, ref_ratio);

    // 3. Create DistributionMapping by refining the atmospheric DM by {C, C, 1}
    //    This ensures each rank owns the refined cells corresponding to its columns
    fg.dm = amrex::refine(dm_atm, ref_ratio);

    // 4. Build 2D Geometry from atmospheric geometry
    //    Physical domain: [prob_lo, prob_hi] in x,y only
    //    Computational domain: [0, C*nx) × [0, C*ny) × [0, 1)
    amrex::RealBox real_box(geom_atm.ProbLo(), geom_atm.ProbHi());
    
    // Set z extent to just the z=0 cell height (or minimal)
    // Domain box: refined version of k=0 box
    amrex::Box domain_2d = amrex::refine(geom_atm.Domain(), ref_ratio);
    domain_2d.setBig(2, domain_2d.smallEnd(2));  // Make it a single slab
    
    fg.geom = amrex::Geometry(domain_2d, real_box,
                              geom_atm.Coord(), geom_atm.isPeriodic());

    return fg;
}
