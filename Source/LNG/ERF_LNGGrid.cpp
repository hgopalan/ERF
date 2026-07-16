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

    // Extract full ATM domain bounds from geometry — NOT from atm_ba[0].
    // With MPI decomposition, atm_ba[0] covers only one rank's subdomain.
    // Using atm_ba[0].bigEnd() gives wrong ihi/jhi when n_boxes > 1.
    const auto& atm_domain = atm_geom.Domain();
    int ihi = atm_domain.bigEnd(0);
    int jhi = atm_domain.bigEnd(1);

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
    lng_prob_domain.setLo(2, atm_prob_domain.lo(2));

    lng_prob_domain.setHi(0, atm_prob_domain.hi(0));
    lng_prob_domain.setHi(1, atm_prob_domain.hi(1));
    lng_prob_domain.setHi(2, atm_prob_domain.lo(2) + atm_geom.CellSize(2));

    // Index-space domain: [0, (ihi+1)*ratio - 1] x [0, (jhi+1)*ratio - 1] x [0, 0]
    // Derived from full ATM domain, so correct for any number of MPI ranks.
    amrex::Box lng_domain;
    lng_domain.setSmall(amrex::IntVect(0, 0, 0));
    lng_domain.setBig(amrex::IntVect((ihi+1)*ratio - 1, (jhi+1)*ratio - 1, 0));

    // Create geometry (x,y periodic; z not)
    amrex::Array<int, 3> is_periodic = {{1, 1, 0}};
    geom.define(lng_domain, &lng_prob_domain, amrex::CoordSys::cartesian, is_periodic.data());

    // Reuse atmospheric distribution mapping (refined boxes preserve load balance)
    dm = atm_dm;
}