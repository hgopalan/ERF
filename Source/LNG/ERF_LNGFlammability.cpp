/**
 * @file ERF_LNGFlammability.cpp
 * @brief Implementation of flammability zone computation for LNG Phase 5
 * 
 * Converts mass concentration to volume fraction and computes LFL/UFL exceedance zones.
 */

#include "ERF_LNGFlammability.H"

#ifdef ERF_USE_LNG

#include <AMReX_MFIter.H>
#include <AMReX_Reduce.H>
#include <AMReX_Print.H>

/**
 * @brief Compute LFL and UFL exceedance masks from concentration field
 * 
 * Algorithm:
 * 1. For each cell in a ParallelFor:
 *    a) Read concentration value from lng_conc_sfc
 *    b) Convert to volume fraction using rho_to_vol_fraction()
 *    c) Compare against LFL and UFL thresholds
 *    d) Set mask = 1 if threshold exceeded, else 0
 * 2. Optional debug output with zone statistics
 */
void compute_flammability_masks(amrex::MultiFab&       lng_lfl_mask,
                                 amrex::MultiFab&       lng_ufl_mask,
                                 const amrex::MultiFab& lng_conc_sfc,
                                 amrex::Real            rho_vapor,
                                 amrex::Real            mol_weight_LNG,
                                 amrex::Real            lfl_threshold,
                                 amrex::Real            ufl_threshold)
{
    for (amrex::MFIter mfi(lng_lfl_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        
        auto lfl_mask = lng_lfl_mask.array(mfi);
        auto ufl_mask = lng_ufl_mask.array(mfi);
        auto conc = lng_conc_sfc.const_array(mfi);
        
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real conc_val = conc(i, j, k);
            
            // Convert mass concentration to volume fraction
            amrex::Real vol_frac = rho_to_vol_fraction(conc_val, rho_vapor, mol_weight_LNG);
            
            // Check LFL threshold
            lfl_mask(i, j, k) = (vol_frac >= lfl_threshold) ? 1.0 : 0.0;
            
            // Check UFL threshold
            ufl_mask(i, j, k) = (vol_frac >= ufl_threshold) ? 1.0 : 0.0;
        });
    }
}

/**
 * @brief Compute LFL zone area [m^2]
 * 
 * Sums all cells where lfl_mask = 1 and multiplies by cell area.
 */
amrex::Real compute_lfl_area(const amrex::MultiFab& lng_lfl_mask,
                               const amrex::Geometry& geom_lng)
{
    amrex::Real cell_area = 1.0;
    const auto& dx = geom_lng.CellSize();
    if (dx[0] > 0.0) {
        cell_area = dx[0] * dx[1];  // Assuming square grid in 2D
    }
    
    // Sum all mask values
    amrex::Real mask_sum = lng_lfl_mask.sum(0);
    
    return mask_sum * cell_area;
}

/**
 * @brief Compute UFL zone area [m^2]
 * 
 * Sums all cells where ufl_mask = 1 and multiplies by cell area.
 */
amrex::Real compute_ufl_area(const amrex::MultiFab& lng_ufl_mask,
                               const amrex::Geometry& geom_lng)
{
    amrex::Real cell_area = 1.0;
    const auto& dx = geom_lng.CellSize();
    if (dx[0] > 0.0) {
        cell_area = dx[0] * dx[1];  // Assuming square grid in 2D
    }
    
    // Sum all mask values
    amrex::Real mask_sum = lng_ufl_mask.sum(0);
    
    return mask_sum * cell_area;
}

#endif /* ERF_USE_LNG */
