/**
 * @file ERF_DustAtmCoupling.cpp
 * @brief Implementation of dust-to-atmosphere coupling functions.
 */

#ifdef ERF_USE_DUST

#include <ERF_DustAtmCoupling.H>
#include <AMReX_MFIter.H>

using namespace amrex;

void apply_dust_tendency_to_cc_source(
    MultiFab&       cc_source,
    const MultiFab& Q_dust_atm,
    const MultiFab& detJ,
    const Geometry& geom_atm,
    int             dust_scalar_comp,
    Real            feedback,
    bool            dust_debug)
{
    if (feedback <= 0.0) return;

    const Box& domain = geom_atm.Domain();
    const int klo = domain.smallEnd(2);
    const Real dz_avg = geom_atm.CellSize(2);

    for (MFIter mfi(cc_source, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        // one plane: the injection is at klo only (the kernel used to run over
        // the whole tile and return in every other cell)
        if (bx.smallEnd(2) > klo) continue;
        Box bx_sfc = bx; bx_sfc.setSmall(2, klo); bx_sfc.setBig(2, klo);
        auto src_arr  = cc_source.array(mfi);
        auto q_arr    = Q_dust_atm.const_array(mfi);
        auto dj_arr   = detJ.const_array(mfi);
        const int comp = dust_scalar_comp;
        const Real fb  = feedback;

        ParallelFor(bx_sfc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // the lowest cell's thickness
            Real dz = dj_arr(i,j,k) * dz_avg;
            if (dz <= 1.0e-10) dz = dz_avg;
            // d(RhoDust)/dt = F_dust * feedback / h_0
            src_arr(i, j, k, comp) += fb * q_arr(i, j, 0) / dz;
        });
    }

    if (dust_debug) {
        Real F_max   = Q_dust_atm.max(0);
        Real tend_max= cc_source.max(dust_scalar_comp);
        Real tend_sum= cc_source.sum(dust_scalar_comp);
        amrex::Print() << "[DUST COUPLING] F_dust_max=" << F_max
                       << " kg/m^2/s  RhoDust_tend max=" << tend_max
                       << " sum=" << tend_sum << " [kg/m^3/s]\n";
    }
}

#endif // ERF_USE_DUST
