/**
 * @file ERF_LNGFlammability.H
 * @brief Flammability zone computation from concentration fields (Phase 5)
 * 
 * Converts mass concentration (RhoLNG) to volume fraction and identifies
 * LFL (Lower Flammability Limit) and UFL (Upper Flammability Limit) zones.
 * 
 * References:
 *   NFPA 59A — Recommended Practice for the Siting of Liquefied Natural Gas Facilities
 *   Koopman, R.P., 1982: "Burro LNG spill test series final report" NTIS
 */

#ifndef ERF_LNG_FLAMMABILITY_H
#define ERF_LNG_FLAMMABILITY_H

#ifdef ERF_USE_LNG

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>

namespace LNGFlammabilityConst {
    /// Molecular weight of air [g/mol]
    constexpr amrex::Real MOL_WEIGHT_AIR = 28.97;
    
    /// Standard air density [kg/m^3] at 15°C, 1 atm
    constexpr amrex::Real RHO_AIR_STD    = 1.225;
}

/**
 * @brief Convert mass concentration to volume fraction
 * 
 * vol_frac = (RhoLNG / rho_vapor) * (M_air / M_LNG)
 * 
 * Using ideal gas law, the volume fraction is:
 *   vol_frac = (ρ_LNG / ρ_vapor) * (M_air / M_LNG)
 * 
 * where ρ_LNG is mass concentration and ρ_vapor is the reference vapor density.
 * 
 * @param[in] rho_lng          Mass concentration [kg/m^3]
 * @param[in] rho_vapor        Reference vapor density [kg/m^3]
 * @param[in] mol_weight_LNG   Molecular weight of LNG [g/mol]
 * @return Volume fraction [-]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rho_to_vol_fraction(amrex::Real rho_lng, amrex::Real rho_vapor,
                                 amrex::Real mol_weight_LNG) noexcept
{
    if (rho_vapor < 1.0e-10 || mol_weight_LNG < 1.0e-10) return 0.0;
    return (rho_lng / rho_vapor) * (LNGFlammabilityConst::MOL_WEIGHT_AIR / mol_weight_LNG);
}

/**
 * @brief Compute LFL and UFL exceedance masks from concentration field
 * 
 * For each cell, converts mass concentration to volume fraction and compares
 * against LFL and UFL thresholds. Outputs binary masks (1 = zone exceeded, 0 = not).
 * 
 * @param[out] lng_lfl_mask        LFL mask [0/1]: 1 where vol_frac >= LFL threshold
 * @param[out] lng_ufl_mask        UFL mask [0/1]: 1 where vol_frac >= UFL threshold
 * @param[in]  lng_conc_sfc        Concentration field [kg/m^3]
 * @param[in]  rho_vapor           Reference vapor density [kg/m^3]
 * @param[in]  mol_weight_LNG      Molecular weight of LNG [g/mol]
 * @param[in]  lfl_threshold       LFL volume fraction threshold [-]
 * @param[in]  ufl_threshold       UFL volume fraction threshold [-]
 */
void compute_flammability_masks(amrex::MultiFab&       lng_lfl_mask,
                                 amrex::MultiFab&       lng_ufl_mask,
                                 const amrex::MultiFab& lng_conc_sfc,
                                 amrex::Real            rho_vapor,
                                 amrex::Real            mol_weight_LNG,
                                 amrex::Real            lfl_threshold,
                                 amrex::Real            ufl_threshold);

/**
 * @brief Compute LFL zone area [m^2]
 * 
 * Returns the total area of cells where LFL mask = 1.
 * 
 * area = sum(lfl_mask) * dx * dy
 * 
 * @param[in] lng_lfl_mask  LFL mask [0/1]
 * @param[in] geom_lng      LNG geometry
 * @return LFL area [m^2]
 */
amrex::Real compute_lfl_area(const amrex::MultiFab& lng_lfl_mask,
                               const amrex::Geometry& geom_lng);

/**
 * @brief Compute UFL zone area [m^2]
 * 
 * Returns the total area of cells where UFL mask = 1.
 * 
 * area = sum(ufl_mask) * dx * dy
 * 
 * @param[in] lng_ufl_mask  UFL mask [0/1]
 * @param[in] geom_lng      LNG geometry
 * @return UFL area [m^2]
 */
amrex::Real compute_ufl_area(const amrex::MultiFab& lng_ufl_mask,
                               const amrex::Geometry& geom_lng);

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_FLAMMABILITY_H */
