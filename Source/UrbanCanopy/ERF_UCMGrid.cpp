/**
 * @file ERF_UCMGrid.cpp
 * @brief Implementation of the SLUCM grid factory function
 *
 * Constructs a 2D refined grid from the atmospheric level-0 domain.
 * Phase 1.1: Stub implementation with debug message.
 * Phase 1.2: Real refinement logic will be added.
 *
 * References:
 *  - Source/Fire/ERF_FireGrid.cpp
 *  - Source/Dust/ERF_DustGrid.cpp
 */

#include <ERF_UCMGrid.H>
#include <AMReX_Box.H>
#include <AMReX_BoxList.H>
#include <AMReX_IntVect.H>
#include <AMReX_Print.H>

using namespace amrex;

UCMGrid
create_ucm_grid(const BoxArray& ba_atm,
                const DistributionMapping& dm_atm,
                const Geometry& geom_atm,
                int grid_ratio,
                int lev)
{
    UCMGrid ug;
    ug.grid_ratio = grid_ratio;
    ug.lev = lev;

    // TODO(UCM Phase 1.2): Implement real grid refinement
    // For now, return stub grid structure to enable compilation
    
    // Step 1: Extract k=0 2D slice from each atmospheric box
    Box domain_2d = geom_atm.Domain();
    domain_2d.setSmall(2, 0);  // Set k minimum to 0
    domain_2d.setBig(2, 0);     // Set k maximum to 0 (1 cell in z)

    Vector<Box> box_list_2d;
    for (int i = 0; i < ba_atm.size(); ++i) {
        Box b = ba_atm[i];
        b.setSmall(2, 0);
        b.setBig(2, 0);
        box_list_2d.push_back(b);
    }
    BoxList bl_2d(std::move(box_list_2d));
    BoxArray ba_2d(bl_2d);

    // Step 2: Refine the 2D BoxArray (Phase 1.2 TODO: actual refinement)
    IntVect ref_ratio(grid_ratio, grid_ratio, 1);
    BoxArray ba_ucm = amrex::refine(ba_2d, ref_ratio);

    // Step 3: DistributionMapping unchanged
    // amrex::refine() preserves number of boxes, so rank ownership unchanged
    DistributionMapping dm_ucm = dm_atm;

    // Step 4: Create refined 2D Geometry
    Box atm_domain = geom_atm.Domain();
    Box atm_2d_full = makeSlab(atm_domain, 2, 0);

    // Index-space domain: scale hi-end for refined grid
    // Formula: new_hi = old_hi * grid_ratio + (grid_ratio - 1)
    Box ucm_domain(atm_2d_full.smallEnd(),
                   IntVect(atm_2d_full.bigEnd(0) * grid_ratio + (grid_ratio - 1),
                           atm_2d_full.bigEnd(1) * grid_ratio + (grid_ratio - 1),
                           0));

    // Physical domain: keep x-y from ATM, set z to dummy 1 m
    RealBox prob_domain_2d = geom_atm.ProbDomain();
    prob_domain_2d.setHi(2, prob_domain_2d.lo(2) + 1.0);

    // Create 2D Geometry with cartesian coordinates, no periodicity
    Geometry geom_ucm_2d(ucm_domain, prob_domain_2d,
                         CoordSys::cartesian,
                         {false, false, false});

    ug.ba = ba_ucm;
    ug.dm = dm_ucm;
    ug.geom = geom_ucm_2d;

    // Phase 1.1 debug message
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[UCM DEBUG] create_ucm_grid stub called (Phase 1.1 no-op)\n";
    }

    return ug;
}
