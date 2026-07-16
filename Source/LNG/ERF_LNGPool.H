#ifndef ERF_LNG_POOL_H
#define ERF_LNG_POOL_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Print.H>

void update_pool_mask(amrex::MultiFab& lng_pool_mask,
                      const amrex::MultiFab& lng_pool_depth,
                      const amrex::Geometry& geom_lng,
                      amrex::Real depth_threshold = 1.0e-4);

void apply_spill_source(amrex::MultiFab& lng_pool_depth,
                        const amrex::Geometry& geom_lng,
                        amrex::Real spill_rate_kg_s,
                        amrex::Real rho_LNG,
                        amrex::Real pool_area_m2,
                        amrex::Real cx, amrex::Real cy,
                        amrex::Real dt);

void deplete_pool_from_evaporation(amrex::MultiFab& lng_pool_depth,
                                   const amrex::MultiFab& lng_evap_flux,
                                   const amrex::Geometry& geom_lng,
                                   amrex::Real rho_LNG,
                                   amrex::Real dt,
                                   bool lng_debug);

amrex::Real compute_pool_mass(const amrex::MultiFab& lng_pool_depth,
                               const amrex::Geometry& geom_lng,
                               amrex::Real rho_LNG);

amrex::Real compute_pool_area(const amrex::MultiFab& lng_pool_mask,
                               const amrex::Geometry& geom_lng);

#endif /* ERF_LNG_POOL_H */