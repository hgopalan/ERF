/**
 * @file ERF_LNGWindExtract.H
 * @brief Wind extraction from atmosphere (Phase 4+)
 * @details
 * Header-only stubs for extracting wind and boundary layer data from atmosphere.
 * These will be implemented in Phase 4 when ATM coupling is added.
 * 
 * @see Source/Dust/ERF_DustWindExtract.H — Dust module wind extraction pattern (reference)
 */

#ifndef ERF_LNG_WIND_EXTRACT_H
#define ERF_LNG_WIND_EXTRACT_H

#ifdef ERF_USE_LNG

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>

/**
 * @brief Fill LNG wind field by interpolating from atmosphere at reference height
 * @param[in,out] lng_wind_ref   Output wind on LNG grid [m/s] (2 components: u, v)
 * @param[in] xvel_mf            ATM u-velocity
 * @param[in] yvel_mf            ATM v-velocity
 * @param[in] z_phys_cc_mf       ATM physical height [m]
 * @param[in] geom_atm           ATM geometry
 * @param[in] zref               Reference height [m]
 * @param[in] grid_ratio         LNG refinement ratio
 * @details
 * Phase 1: stub (does nothing).
 * Phase 4: will interpolate wind to reference height zref (default 10 m),
 *          map from ATM columns to LNG grid, and fill output MultiFab.
 */
inline void fill_lng_wind_from_interpolation(
    amrex::MultiFab&       lng_wind_ref,
    const amrex::MultiFab& xvel_mf,
    const amrex::MultiFab& yvel_mf,
    const amrex::MultiFab& z_phys_cc_mf,
    const amrex::Geometry& geom_atm,
    amrex::Real            zref,
    int                    grid_ratio)
{
    // Phase 1: stub
}

/**
 * @brief Fill LNG friction velocity from surface layer
 * @param[in,out] lng_ustar      Output u* on LNG grid [m/s]
 * @param[in] ustar_atm          ATM friction velocity [m/s]
 * @param[in] geom_atm           ATM geometry
 * @param[in] grid_ratio         LNG refinement ratio
 * @details
 * Phase 1: stub (does nothing).
 * Phase 4: will map ATM u* to LNG grid cells (replicating ATM value within each LNG block).
 */
inline void fill_lng_ustar_from_surface_layer(
    amrex::MultiFab&       lng_ustar,
    const amrex::MultiFab& ustar_atm,
    const amrex::Geometry& geom_atm,
    int                    grid_ratio)
{
    // Phase 1: stub
}

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_WIND_EXTRACT_H */
