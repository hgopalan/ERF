/**
 * @file ERF_LNGWindExtract.H
 * @brief Wind and surface field extraction from ERF 3D solver to the 2D LNG grid.
 *
 * The wind interpolation algorithm copies fill_fire_wind_from_interpolation
 * from Source/Fire/ERF_FireWindExtract.cpp, substituting LNGGrid for FireGrid
 * and grid_ratio for C. 
 *
 * SurfaceLayer accessors used:
 *   get_u_star(0)  — friction velocity [m/s]
 *   get_t_surf(0)  — surface skin temperature [K]
 *   get_pblh(0)    — PBL height [m]
 *
 * References:
 *   Source/Dust/ERF_DustWindExtract.H (Phase 9 analog):
 *     Dust module wind extraction pattern
 *   Source/Fire/ERF_FireWindExtract.H (algorithm source)
 *   Hong & Pan (1996). Mon. Wea. Rev., 124, 2322.
 *     https://doi.org/10.1175/1520-0493(1996)124<2322:NBLVDI>2.0.CO;2
 */

#ifndef ERF_LNG_WIND_EXTRACT_H
#define ERF_LNG_WIND_EXTRACT_H

#ifdef ERF_USE_LNG

#include <AMReX_MultiFab.H>
#include <ERF_LNGGrid.H>

// Vertical wind interpolation to zref. Direct analog of fill_dust_wind_from_interpolation
// in Source/Dust/ERF_DustWindExtract.cpp. Steps per LNG cell:
//   - map to ATM column: i_a = i_l / grid_ratio, j_a = j_l / grid_ratio
//   - find k bracket at z_surf + zref
//   - average face-staggered u/v to cell centres
//   - linearly interpolate to z_target
void fill_lng_wind_from_interpolation(
    amrex::MultiFab&       lng_wind_ref,   // ncomp=2, u and v [m/s]
    const amrex::MultiFab& xvel_mf,
    const amrex::MultiFab& yvel_mf,
    const amrex::MultiFab& z_phys_cc_mf,
    const LNGGrid&         lg,
    amrex::Real            zref,
    int                    nz,
    bool                   lng_debug = false);

// Map u* from SurfaceLayer::get_u_star(0) to LNG grid (i_a=i/C, j_a=j/C, k=0).
void fill_lng_ustar_from_surface_layer(
    amrex::MultiFab&       lng_ustar,
    const amrex::MultiFab& ustar_atm,
    const LNGGrid&         lg,
    bool                   lng_debug = false);

// Generic: map any scalar ATM k=0 field to LNG grid. Used for T_sfc and PBLH.
void fill_lng_scalar_from_atm(
    amrex::MultiFab&       lng_field,
    const amrex::MultiFab& atm_field,
    const LNGGrid&         lg,
    bool                   lng_debug = false,
    const std::string&     field_name = "");

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_WIND_EXTRACT_H */
