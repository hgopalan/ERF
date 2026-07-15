/**
 * @file ERF_LNGGrid.cpp
 * @brief LNG grid construction implementation
 * @details
 * Implements LNGGrid::build() which creates a 2D computational grid
 * refined from the atmospheric level-0 grid by grid_ratio.
 */

#include "ERF_LNGGrid.H"

void LNGGrid::build(const amrex::BoxArray& atm_ba,
                    const amrex::DistributionMapping& atm_dm,
                    const amrex::Geometry& atm_geom,
                    int ratio)
{
    grid_ratio = ratio;
    
    // Extract k=0 ATM domain bounds
    const auto& atm_box0 = atm_ba[0];
    int ihi = atm_box0.bigEnd(0);
    int jhi = atm_box0.bigEnd(1);
    // klo = 0, khi = 0 for k=0 slab in ATM
    
    // Refine ATM boxes horizontally by ratio; keep k=0 mapping
    amrex::BoxArray refined_ba(atm_ba);
    refined_ba.refine(amrex::IntVect(ratio, ratio, 1));
    
    // Convert to 2D: set nz=1 (k from 0 to 0, which is 1 cell)
    amrex::BoxList lng_bl;
    for (int i = 0; i < refined_ba.size(); ++i) {
        amrex::Box b = refined_ba[i];
        b.setSmall(2, 0);      // ksmall = 0
        b.setBig(2, 0);        // kbig = 0 (1 cell in z)
        lng_bl.push_back(b);
    }
    ba = amrex::BoxArray(lng_bl);
    
    // Create 2D Geometry
    // Physical domain: same x,y as ATM; z spans single LNG cell
    amrex::RealBox lng_prob_domain;
    const auto& atm_prob_domain = atm_geom.ProbDomain();
    lng_prob_domain.setLo(0, atm_prob_domain.lo(0));
    lng_prob_domain.setLo(1, atm_prob_domain.lo(1));
    lng_prob_domain.setLo(2, atm_prob_domain.lo(2));  // Use ATM z_lo as LNG z_lo
    
    lng_prob_domain.setHi(0, atm_prob_domain.hi(0));
    lng_prob_domain.setHi(1, atm_prob_domain.hi(1));
    lng_prob_domain.setHi(2, atm_prob_domain.lo(2) + (atm_geom.CellSize(2)));  // One ATM cell height
    
    // Index-space domain: [0, ihi*ratio] x [0, jhi*ratio] x [0, 0]
    amrex::Box lng_domain;
    lng_domain.setSmall(amrex::IntVect(0, 0, 0));
    lng_domain.setBig(amrex::IntVect(ihi*ratio+1, jhi*ratio+1, 0));
    
    // Create geometry (all periodic in x,y; non-periodic in z)
    amrex::Array<int, 3> is_periodic = {{1, 1, 0}};  // x,y periodic; z not
    geom.define(lng_domain, &lng_prob_domain, amrex::CoordSys::cartesian, is_periodic.data());
    
    // Reuse atmospheric distribution mapping (refined boxes preserve load balance)
    dm = atm_dm;
}