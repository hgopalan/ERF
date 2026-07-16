#ifndef ERF_LNG_ATM_COUPLING_H
#define ERF_LNG_ATM_COUPLING_H

#ifdef ERF_USE_LNG

#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Geometry.H>

/**
 * @brief Coarsen LNG flux from LNG grid to atmospheric k=0 slab.
 * Mirrors coarsen_dust_flux_to_atm in ERF_DustAtmCoupling.H exactly.
 * Uses the geom overload of average_down which is MPI-safe.
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
        // Use the geometry overload — required for correct MPI behaviour
        // with multiple ranks. Mirrors Dust:
        //   average_down(Q_dust, Q_dust_atm, geom_dust, geom_atm, 0, 1, ratio)
        amrex::average_down(lng_evap_flux, lng_flux_atm,
                            geom_lng, geom_atm,
                            0, 1, amrex::IntVect(grid_ratio, grid_ratio, 1));
    }
}

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