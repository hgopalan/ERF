/**
 * @file ERF_LNGAtmCoupling.cpp
 * @brief LNG → atmosphere injection coupling (Phase 3)
 * @details
 * Implements one-way coupling: 2D LNG evaporation flux → 3D ATM passive scalar at k=0.
 * Uses one-step explicit lag (flux from step n injected at step n+1).
 * lng_flux_atm lives on the ATM-grid k=0 slab (same BoxArray as cc_source at k=0),
 * so const_array(mfi) is valid when iterating over cc_source.
 */

#include <ERF_LNGAtmCoupling.H>
#include <AMReX_MFIter.H>
#include <AMReX_Print.H>

void apply_lng_tendency_to_cc_source(
    amrex::MultiFab&       cc_source,
    const amrex::MultiFab& lng_flux_atm,
    const amrex::MultiFab& z_phys_cc,
    const amrex::Geometry& geom_atm,
    int                    lng_scalar_comp,
    amrex::Real            feedback,
    bool                   lng_debug)
{
    if (feedback <= 0.0) return;
    if (lng_scalar_comp < 0) return;

    const amrex::Box& domain = geom_atm.Domain();
    int klo = domain.smallEnd(2);
    int khi = domain.bigEnd(2);
    const auto& dx = geom_atm.CellSize();
    amrex::Real dz_avg = dx[2];

    // lng_flux_atm is on the ATM k=0 slab BoxArray — same as cc_source at k=0.
    // Iterate over cc_source and read flux from the matching k=0 tile.
    for (amrex::MFIter mfi(cc_source, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto src_arr = cc_source.array(mfi);
        // Use ParallelCopy-safe access: lng_flux_atm has same xy BoxArray as cc_source
        auto q_arr   = lng_flux_atm.const_array(mfi);
        auto z_arr   = z_phys_cc.const_array(mfi);
        const int comp = lng_scalar_comp;
        const amrex::Real fb = feedback;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (k != klo) return;
            amrex::Real dz = (k < khi) ? (z_arr(i,j,k+1) - z_arr(i,j,k)) : dz_avg;
            if (dz <= 1.0e-10) dz = dz_avg;
            // d(RhoLNG)/dt = F_evap * feedback / dz
            src_arr(i, j, k, comp) += fb * q_arr(i, j, 0) / dz;
        });
    }

    if (lng_debug) {
        amrex::Real F_max    = lng_flux_atm.max(0);
        amrex::Real tend_max = cc_source.max(lng_scalar_comp);
        amrex::Real tend_sum = cc_source.sum(lng_scalar_comp);
        amrex::Print() << "[LNG COUPLING] Phase 3: F_evap_max=" << F_max
                       << " kg/m^2/s  RhoLNG_tend_max=" << tend_max
                       << " kg/m^3/s  sum=" << tend_sum << " kg/m^3/s\n";
    }
}