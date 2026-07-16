/**
 * @file ERF_LNGAtmCoupling.H
 * @brief LNG ↔ atmosphere coupling routines (Phase 3+)
 * @details
 * Phase 3: implements 2D→3D injection of LNG vapor flux into atmospheric scalar.
 * 
 * @see Source/Dust/ERF_DustAtmCoupling.H — Dust module coupling pattern (reference)
 */

#ifndef ERF_LNG_ATM_COUPLING_H
#define ERF_LNG_ATM_COUPLING_H

#ifdef ERF_USE_LNG

#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Geometry.H>

/**
 * @brief Coarsen LNG flux from LNG grid to atmospheric grid
 * @param[in,out] lng_flux_atm   Output flux on ATM grid [kg/m^2/s]
 * @param[in] lng_evap_flux      Input flux on LNG grid [kg/m^2/s]
 * @param[in] geom_lng           LNG geometry
 * @param[in] geom_atm           ATM geometry
 * @param[in] grid_ratio         LNG refinement ratio vs ATM
 * @details
 * Uses amrex::average_down() to coarsen evaporation flux from LNG grid to ATM level-0 2D slab.
 * grid_ratio=1: direct copy. grid_ratio>1: area-weighted average.
 */
inline void coarsen_lng_flux_to_atm(
    amrex::MultiFab&       lng_flux_atm,
    const amrex::MultiFab& lng_evap_flux,
    const amrex::Geometry& geom_lng,
    const amrex::Geometry& geom_atm,
    int                    grid_ratio)
{
    if (grid_ratio == 1) {
        amrex::MultiFab::Copy(lng_flux_atm, lng_evap_flux, 0, 0, 1, 0);
    } else {
        amrex::average_down(lng_evap_flux, lng_flux_atm,
                            0, 1, amrex::IntVect(grid_ratio, grid_ratio, 1));
    }
}

/**
 * @brief Apply LNG tendency to atmosphere mass source
 * @param[in,out] cc_source      Atmosphere source term (RhoDT, RhoYv, etc.)
 * @param[in] lng_flux_atm       LNG flux on ATM grid [kg/m^2/s]
 * @param[in] z_phys_cc          Atmosphere physical height at cell centers [m]
 * @param[in] geom_atm           Atmosphere geometry
 * @param[in] lng_scalar_comp    Component index in cc_source for LNG scalar
 * @param[in] feedback           Coupling coefficient [0, 1]
 * @param[in] lng_debug          Debug output flag
 * @details
 * Phase 3: Distributes evaporation flux over lowest k=0 layer:
 *   dRhoLNG/dt = F_evap * feedback / dz_k0  [kg/m^3/s]
 * This is a declaration only; implementation in ERF_LNGAtmCoupling.cpp
 */
void apply_lng_tendency_to_cc_source(
    amrex::MultiFab&       cc_source,
    const amrex::MultiFab& lng_flux_atm,
    const amrex::MultiFab& z_phys_cc,
    const amrex::Geometry& geom_atm,
    int                    lng_scalar_comp,
    amrex::Real            feedback,
    bool                   lng_debug);

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_ATM_COUPLING_H */