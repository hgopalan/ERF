/**
 * @file ERF_LNGEvaporation.cpp
 * @brief Implementation of LNG evaporation kernel
 * @note Phase 2: evaporation model with Chilton-Colburn mass transfer
 * @ref ERF_LNGEvaporation.H — kernel declarations and theory
 */

#include "ERF_LNGEvaporation.H"
#include <AMReX_MFIter.H>
#include <AMReX_Geometry.H>

void compute_lng_evap_flux(amrex::MultiFab& lng_evap_flux,
                            amrex::MultiFab& lng_latent_flux,
                            const amrex::MultiFab& lng_pool_mask,
                            const amrex::MultiFab& lng_ustar,
                            amrex::Real z_ref,
                            amrex::Real z0,
                            amrex::Real rho_vapor,
                            amrex::Real Hv,
                            bool lng_debug)
{
    using namespace LNGEvapConst;
    
    // GPU kernel over 2D LNG grid
    for (amrex::MFIter mfi(lng_evap_flux, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& bx = mfi.tilebox();
        const auto& mask_fab = lng_pool_mask[mfi];
        const auto& ustar_fab = lng_ustar[mfi];
        auto& evap_fab = lng_evap_flux[mfi];
        auto& latent_fab = lng_latent_flux[mfi];
        
        // Parallel for loop over 2D box
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            if (mask_fab(i, j, k) > 0.5) {
                // Pool cell: compute evaporation
                amrex::Real u_star = ustar_fab(i, j, k);
                amrex::Real F_evap = compute_evap_flux_boiling(u_star, z_ref, z0, rho_vapor);
                amrex::Real Q_latent = compute_latent_heat_flux(F_evap, Hv);
                
                evap_fab(i, j, k) = F_evap;
                latent_fab(i, j, k) = Q_latent;
            } else {
                // No pool: zero flux
                evap_fab(i, j, k) = 0.0;
                latent_fab(i, j, k) = 0.0;
            }
        });
    }
    
    // Fill ghost cells
    lng_evap_flux.FillBoundary();
    lng_latent_flux.FillBoundary();
    
    // Debug summary
    if (lng_debug) {
        amrex::Real evap_max = lng_evap_flux.max(0);
        amrex::Real evap_sum = lng_evap_flux.sum(0);
        amrex::Real latent_max = lng_latent_flux.max(0);
        amrex::Real active_cells = lng_pool_mask.sum(0);
        
        amrex::Print() << "[LNG DEBUG] Phase 2: evap step"
                       << "  evap_flux_max=" << evap_max << " kg/m^2/s"
                       << "  evap_flux_sum=" << evap_sum << " kg/m^2/s"
                       << "  latent_flux_max=" << latent_max << " W/m^2"
                       << "  active_cells=" << (long)active_cells << "\n";
    }
}
