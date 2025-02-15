#ifndef ERF_INTEGRATION_H_
#define ERF_INTEGRATION_H_

#include <AMReX_MultiFab.H>
#include "ERF_DataStruct.H"
#include "ERF_InputSoundingData.H"
#include "ERF_TurbPertStruct.H"

#ifdef ERF_USE_EB
#include <AMReX_EBMultiFabUtil.H>
#endif

/**
 * Function for computing the buoyancy term to be used in the evolution
 * equation for the z-component of momentum in the slow integrator.  There
 * are three options for how buoyancy is computed (two are the same in the absence of moisture).
 */
void make_buoyancy (amrex::Vector<  amrex::MultiFab>& S_data,
                    const           amrex::MultiFab & S_prim,
                    amrex::MultiFab& buoyancy,
                    const amrex::Geometry geom,
                    const SolverChoice& solverChoice,
                    const amrex::MultiFab& base_state,
                    const int n_qstate,
                    const int anelastic);

void make_sources (int level, int nrk,
                   amrex::Real dt, amrex::Real time,
                   amrex::Vector<amrex::MultiFab>& S_data,
                   const amrex::MultiFab& S_prim,
                         amrex::MultiFab& cc_source,
                   std::unique_ptr<amrex::MultiFab>& z_phys_cc,
#if defined(ERF_USE_RRTMGP)
                   const amrex::MultiFab* qheating_rates,
#endif
                   const amrex::Geometry geom,
                   const SolverChoice& solverChoice,
                   std::unique_ptr<amrex::MultiFab>& mapfac_u,
                   std::unique_ptr<amrex::MultiFab>& mapfac_v,
                   std::unique_ptr<amrex::MultiFab>& mapfac_m,
                   const amrex::Real* dptr_rhotheta_src,
                   const amrex::Real* dptr_rhoqt_src,
                   const amrex::Real* dptr_wbar_sub,
                   const amrex::Vector<amrex::Real*> d_rayleigh_ptrs_at_lev,
                   InputSoundingData& input_sounding_data,
                   TurbulentPerturbation& turbPert);

void make_mom_sources (int level, int nrk,
                       amrex::Real dt, amrex::Real time,
                       amrex::Vector<amrex::MultiFab>& S_data,
                       const amrex::MultiFab& S_prim,
                       std::unique_ptr<amrex::MultiFab>& z_phys_nd,
                       std::unique_ptr<amrex::MultiFab>& z_phys_cc,
                       const amrex::MultiFab& xvel,
                       const amrex::MultiFab& yvel,
                       const amrex::MultiFab& wvel,
                             amrex::MultiFab& xmom_source,
                             amrex::MultiFab& ymom_source,
                             amrex::MultiFab& zmom_source,
                       const amrex::MultiFab& base_state,
                             amrex::MultiFab* forest_drag,
                       const amrex::Geometry geom,
                       const SolverChoice& solverChoice,
                       std::unique_ptr<amrex::MultiFab>& mapfac_m,
                       std::unique_ptr<amrex::MultiFab>& mapfac_u,
                       std::unique_ptr<amrex::MultiFab>& mapfac_v,
                       const amrex::Real* dptr_rhotheta_src,
                       const amrex::Real* dptr_rhoqt_src,
                       const amrex::Real* dptr_wbar_sub,
                       const amrex::Vector<amrex::Real*> d_rayleigh_ptrs_at_lev,
                       const amrex::Vector<amrex::Real*> d_sponge_ptrs_at_lev,
                       InputSoundingData& input_sounding_data,
                       const int n_qstate);

void add_thin_body_sources (amrex::MultiFab& xmom_source,
                            amrex::MultiFab& ymom_source,
                            amrex::MultiFab& zmom_source,
                            std::unique_ptr<amrex::iMultiFab>& xflux_imask_lev,
                            std::unique_ptr<amrex::iMultiFab>& yflux_imask_lev,
                            std::unique_ptr<amrex::iMultiFab>& zflux_imask_lev,
                            std::unique_ptr<amrex::MultiFab>& thin_xforce_lev,
                            std::unique_ptr<amrex::MultiFab>& thin_yforce_lev,
                            std::unique_ptr<amrex::MultiFab>& thin_zforce_lev);

#if defined(ERF_USE_NETCDF)
void
moist_set_rhs (const amrex::Box& tbx,
               const amrex::Box& gtbx,
               const amrex::Array4<amrex::Real const>& old_cons,
               const amrex::Array4<amrex::Real const>& new_cons,
               const amrex::Array4<amrex::Real      >& cell_rhs,
               const amrex::Real& bdy_time_interval,
               const amrex::Real& start_bdy_time,
               const amrex::Real& new_stage_time,
               const amrex::Real& dt,
               int width, int set_width,
               const amrex::Box& domain,
               amrex::Vector<amrex::Vector<amrex::FArrayBox>>& bdy_data_xlo,
               amrex::Vector<amrex::Vector<amrex::FArrayBox>>& bdy_data_xhi,
               amrex::Vector<amrex::Vector<amrex::FArrayBox>>& bdy_data_ylo,
               amrex::Vector<amrex::Vector<amrex::FArrayBox>>& bdy_data_yhi);
#endif

void ApplySpongeZoneBCsForCC (const SpongeChoice& spongeChoice,
                              const amrex::Geometry geom,
                              const amrex::Box& bx,
                              const amrex::Array4<amrex::Real>& cell_rhs,
                              const amrex::Array4<const amrex::Real>& cell_data);

void ApplySpongeZoneBCsForMom (const SpongeChoice& spongeChoice,
                               const amrex::Geometry geom,
                               const amrex::Box& tbx,
                               const amrex::Box& tby,
                               const amrex::Box& tbz,
                               const amrex::Array4<amrex::Real>& rho_u_rhs,
                               const amrex::Array4<amrex::Real>& rho_v_rhs,
                               const amrex::Array4<amrex::Real>& rho_w_rhs,
                               const amrex::Array4<const amrex::Real>& rho_u,
                               const amrex::Array4<const amrex::Real>& rho_v,
                               const amrex::Array4<const amrex::Real>& rho_w);

void ApplySpongeZoneBCsForMom_ReadFromFile (const SpongeChoice& spongeChoice,
                                              const amrex::Geometry geom,
                                              const amrex::Box& tbx,
                                              const amrex::Box& tby,
                                              const amrex::Array4<const amrex::Real>& cell_data,
                                              const amrex::Array4<amrex::Real>& rho_u_rhs,
                                              const amrex::Array4<amrex::Real>& rho_v_rhs,
                                              const amrex::Array4<const amrex::Real>& rho_u,
                                              const amrex::Array4<const amrex::Real>& rho_v,
                                              const amrex::Vector<amrex::Real*> d_sponge_ptrs_at_lev);
#endif
