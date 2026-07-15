/**
 * @file ERF_LNGAtmCoupling.H
 * @brief LNG ↔ atmosphere coupling routines (Phase 3+)
 * @details
 * Header-only stubs for atmosphere coupling.
 * These will be implemented in Phase 3 when two-way feedback is added.
 * 
 * @see Source/Dust/ERF_DustAtmCoupling.H — Dust module coupling pattern (reference)
 */

#ifndef ERF_LNG_ATM_COUPLING_H
#define ERF_LNG_ATM_COUPLING_H

#ifdef ERF_USE_LNG

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>

/**
 * @brief Coarsen LNG flux from LNG grid to atmospheric grid
 * @param[in,out] lng_flux_atm   Output flux on ATM grid [kg/m^2/s]
 * @param[in] lng_flux           Input flux on LNG grid [kg/m^2/s]
 * @param[in] geom_lng           LNG geometry
 * @param[in] geom_atm           ATM geometry
 * @param[in] grid_ratio         LNG refinement ratio vs ATM
 * @details
 * Phase 1: stub (does nothing).
 * Phase 3: will use amrex::average_down() with refinement ratio to coarsen.
 */
inline void coarsen_lng_flux_to_atm(
    amrex::MultiFab&       lng_flux_atm,
    const amrex::MultiFab& lng_flux,
    const amrex::Geometry& geom_lng,
    const amrex::Geometry& geom_atm,
    int                    grid_ratio)
{
    // Phase 1: stub
}

/**
 * @brief Apply LNG tendency to atmosphere energy/mass source
 * @param[in,out] cc_source      Atmosphere source term (RhoDT, RhoYv, etc.)
 * @param[in] lng_flux_atm       LNG flux on ATM grid [kg/m^2/s] or [W/m^2]
 * @param[in] z_phys_cc          Atmosphere physical height at cell centers [m]
 * @param[in] geom_atm           Atmosphere geometry
 * @param[in] feedback_strength  Coupling coefficient [0, 1]
 * @details
 * Phase 1: stub (does nothing).
 * Phase 3: will distribute flux over lowest k=0 layer:
 *   dE/dt = -flux / dz_k0  [W/m^3] (negative = cooling from evaporation)
 */
inline void apply_lng_tendency_to_cc_source(
    amrex::MultiFab&       cc_source,
    const amrex::MultiFab& lng_flux_atm,
    const amrex::MultiFab& z_phys_cc,
    const amrex::Geometry& geom_atm,
    amrex::Real            feedback_strength)
{
    // Phase 1: stub
}

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_ATM_COUPLING_H */
