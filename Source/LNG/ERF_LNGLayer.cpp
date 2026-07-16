/**
 * @file ERF_LNGLayer.cpp
 * @brief LNG container class implementation
 * @details
 * Implements LNGLayer methods for grid construction, MultiFab allocation,
 * and timestep cycling. Phase 1 is zero-physics stub; Phase 2 adds evaporation.
 * @note Phase 2: pool evaporation model, pool depletion, mass tracking
 */

#include "ERF_LNGLayer.H"
#include "ERF_LNGPrerequisites.H"
#include "ERF_LNGStatsOutput.H"
#include "ERF_LNGEvaporation.H"
#include "ERF_LNGPool.H"
#include "ERF_LNGAtmCoupling.H"
#include "ERF.H"
#include "ERF_IndexDefines.H"
#include <AMReX_MultiFab.H>
#include <AMReX_Print.h>
#include <cmath>
#include <iomanip>

void LNGLayer::initialize(const ERF& erf, const LNGParams& params)
{
    m_params = params;
    
    // Validate prerequisites
    check_lng_prerequisites(params, erf.Geom(0));
    
    // Build 2D LNG grid from ATM level-0 grid
    m_lg.build(erf.boxArray(0), erf.DistributionMap(0), erf.Geom(0), params.grid_ratio);
    
    // Allocate all MultiFabs: 1 component, 1 ghost cell
    const int ncomp = 1;
    const int nghost = 1;
    m_lng_pool_depth    = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_pool_mask     = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_evap_flux     = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_latent_flux   = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_vapor_conc    = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    
    // ATM-grid flux (to be coarsened)
    const int natm_ratio = params.grid_ratio;
    amrex::BoxArray atm_ba_coarse = erf.boxArray(0);
    if (natm_ratio > 1) atm_ba_coarse.coarsen(natm_ratio);
    m_lng_flux_atm      = std::make_unique<amrex::MultiFab>(atm_ba_coarse, erf.DistributionMap(0), ncomp, nghost);
    
    // Wind field (2 components: u, v)
    m_lng_wind_ref      = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, 2, nghost);
    
    // Scalars
    m_lng_ustar         = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_tsfc          = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_pblh          = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_conc_sfc      = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_lfl_mask      = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_ufl_mask      = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    
    // Initialize all to 0.0
    m_lng_pool_depth->setVal(0.0);
    m_lng_pool_mask->setVal(0.0);
    m_lng_evap_flux->setVal(0.0);
    m_lng_latent_flux->setVal(0.0);
    m_lng_vapor_conc->setVal(0.0);
    m_lng_flux_atm->setVal(0.0);
    m_lng_wind_ref->setVal(0.0);
    m_lng_ustar->setVal(0.0);
    m_lng_tsfc->setVal(params.test_surf_temp_K);  // Set to test temperature
    m_lng_pblh->setVal(1000.0);  // Placeholder 1 km PBL height
    m_lng_conc_sfc->setVal(0.0);
    m_lng_lfl_mask->setVal(0.0);
    m_lng_ufl_mask->setVal(0.0);
    
    // Set initial pool region: circle of radius sqrt(area/π) at domain center
    amrex::Real pool_radius = std::sqrt(params.pool_area_m2 / M_PI);
    const auto& geom_lng = m_lg.geom;
    const auto& prob_domain = geom_lng.ProbDomain();
    amrex::Real pool_center_x = 0.5 * (prob_domain.lo(0) + prob_domain.hi(0));
    amrex::Real pool_center_y = 0.5 * (prob_domain.lo(1) + prob_domain.hi(1));
    
    // Fill pool_mask and pool_depth
    for (amrex::MFIter mfi(*m_lng_pool_mask); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.validbox();
        auto pool_mask_arr = (*m_lng_pool_mask)[mfi].array();
        auto pool_depth_arr = (*m_lng_pool_depth)[mfi].array();
        
        amrex::ParallelFor(bx, [=] (amrex::IntVect const& iv) noexcept {
            amrex::Real x = geom_lng.CellCenter(iv[0], 0);
            amrex::Real y = geom_lng.CellCenter(iv[1], 1);
            amrex::Real r = std::sqrt((x - pool_center_x)*(x - pool_center_x) + 
                                      (y - pool_center_y)*(y - pool_center_y));
            if (r <= pool_radius) {
                pool_mask_arr(iv) = 1.0;
                pool_depth_arr(iv) = params.pool_depth_init_m;
            } else {
                pool_mask_arr(iv) = 0.0;
                pool_depth_arr(iv) = 0.0;
            }
        });
    }
    
    // Phase 2: Set atmospheric placeholders and initialize pool tracking
    m_lg_z0 = params.z0_lng;
    
    // Set pool center (or use domain center if -1)
    if (m_pool_cx < 0.0 || m_pool_cy < 0.0) {
       m_pool_cx = 0.5 * (prob_domain.lo(0) + prob_domain.hi(0));
       m_pool_cy = 0.5 * (prob_domain.lo(1) + prob_domain.hi(1));
    }
    
    // Set atmospheric placeholders (Phase 4 will replace with live extraction)
    m_lng_ustar->setVal(params.test_ustar);
    m_lng_tsfc->setVal(params.test_surf_temp_K);
     
    // Phase 3: Set scalar component index for atmosphere coupling
    // Use RhoScalar_comp + 1 (same pattern as Dust: m_dust_scalar_comp = RhoScalar_comp + 1)
    m_lng_scalar_comp = RhoScalar_comp + 1;
     
    // Compute initial pool diagnostics for debug output
    amrex::Real pool_mass_init = compute_pool_mass(*m_lng_pool_depth, geom_lng, params.rho_LNG);
    amrex::Real pool_area_init = compute_pool_area(*m_lng_pool_mask, geom_lng);
     
    // Phase 1 initialization debug output
    if (params.lng_debug) {
        amrex::Print() << "[LNG] ===== ERF-LNG Phase 1 initialized =====\n"
                       << "[LNG DEBUG] Phase 1: pool_centre=(" << m_pool_cx << ", " << m_pool_cy << ") m  "
                       << "area=" << pool_area_init << " m^2  "
                       << "depth=" << params.pool_depth_init_m << " m\n"
                       << "[LNG DEBUG] Phase 1: mol_weight_LNG=" << params.mol_weight_LNG << " g/mol  "
                       << "LFL=" << params.LFL_percent << "%  "
                       << "UFL=" << params.UFL_percent << "%\n"
                       << "[LNG DEBUG] Phase 1: grid_ratio=" << params.grid_ratio << "  "
                       << "feedback=" << params.atm_feedback << "  "
                       << "verbose=" << params.verbose << "  "
                       << "debug=" << (params.lng_debug ? "ON" : "OFF") << "\n"
                       << "[LNG DEBUG] Phase 1: LNGGrid created " << m_lg.ba.size() 
                       << " boxes, grid_ratio=" << params.grid_ratio << "\n"
                       << "[LNG DEBUG] Phase 1: MultiFabs allocated (pool_depth, pool_mask, evap_flux, "
                       << "latent_flux, vapor_conc, flux_atm, wind_ref, ustar, tsfc, pblh, conc_sfc, "
                       << "lfl_mask, ufl_mask) ncomp=1\n"
                       << "[LNG DEBUG] Phase 3: lng_scalar_comp=" << m_lng_scalar_comp
                       << " (RhoScalar_comp+1)\n";
    }
     
    // Write CSV diagnostic header
    write_lng_stats_header(params.lng_diag_file);
     
    if (params.lng_debug) {
        amrex::Print() << "[LNG DEBUG] Phase 1: lng_diag.csv header written\n";
    }
     
    // Phase 2 initialization debug output
    if (params.lng_debug || params.verbose >= 1) {
       amrex::Print() << "[LNG DEBUG] Phase 2: pool evaporation model initialized\n"
                      << "[LNG DEBUG] Phase 2:   pool_centre=(" << m_pool_cx << ", " << m_pool_cy << ") m\n"
                      << "[LNG DEBUG] Phase 2:   pool_area_init=" << pool_area_init << " m^2  "
                      << "pool_depth_init=" << params.pool_depth_init_m << " m\n"
                      << "[LNG DEBUG] Phase 2:   pool_mass_init=" << pool_mass_init << " kg\n"
                      << "[LNG DEBUG] Phase 2:   rho_LNG=" << params.rho_LNG << " kg/m^3  "
                      << "Hv=" << params.Hv_LNG << " J/kg  "
                      << "rho_vapor_ref=" << params.rho_vapor_ref << " kg/m^3\n"
                      << "[LNG DEBUG] Phase 2:   test_ustar=" << params.test_ustar << " m/s  "
                      << "test_surf_temp=" << params.test_surf_temp_K << " K\n"
                      << "[LNG DEBUG] Phase 2:   z0_lng=" << m_lg_z0 << " m  zref=" << params.zref << " m\n"
                      << "[LNG DEBUG] Phase 2:   evap model: k_mass = u* * kappa / (Sc^(2/3) * ln(zref/z0))\n";
    }
     
    // Print per-step debug header if enabled
    if (params.lng_debug) {
      int pool_cells = amrex::ReduceSum(*m_lng_pool_mask, 0,
            [=] (amrex::Box const& bx, amrex::Array4<amrex::Real const> const& arr) -> int
            {
                int count = 0;
                amrex::Loop(bx, [&] (amrex::IntVect const& iv) {
                    if (arr(iv) > 0.5) ++count;
                });
                return count;
            });
          
        amrex::Print() << "[LNG DEBUG] Initial Step  " << std::setw(4) << m_step 
                       << "  time=" << std::scientific << std::setprecision(3) << m_time 
                       << "  pool_cells=" << pool_cells
                       << "  pool_mass=" << pool_mass_init << " kg\n";
    }
}

void LNGLayer::advance(amrex::Real dt, const LNGParams& params,
                       const amrex::MultiFab* xvel_mf,
                       const amrex::MultiFab* yvel_mf,
                       const amrex::MultiFab* zvel_mf,
                       const amrex::MultiFab* z_phys_cc_mf,
                       const amrex::Geometry* geom_atm,
                       int nz)
{
    // Phase 2: Full evaporation physics sequence
    
    // Step A: Update m_step and m_time
    ++m_step;
    m_time += dt;
    
    // Step B: Per-step entry debug print
    if (params.lng_debug) {
        amrex::Real pool_mass = compute_pool_mass(*m_lng_pool_depth, m_lg.geom, params.rho_LNG);
        amrex::Real ef_max = m_lng_evap_flux->max(0);
        amrex::Real vc_max = m_lng_vapor_conc->max(0);
        
        amrex::Print() << "[LNG DEBUG] advance: step=" << m_step 
                       << "  time=" << std::scientific << std::setprecision(3) << m_time << " s"
                       << "  dt=" << dt << " s"
                       << "  pool_mass=" << pool_mass << " kg"
                       << "  evap_flux_max=" << ef_max << " kg/m^2/s"
                       << "  vapor_conc_max=" << vc_max << " kg/m^3\n";
    }
    
    // Step C: Set atmospheric state (placeholder path for Phase 2)
    bool have_atm = (xvel_mf && yvel_mf && z_phys_cc_mf && nz > 0);
    
    if (have_atm) {
        if (params.lng_debug) {
            amrex::Print() << "[LNG DEBUG] Phase 2: have_atm=true but ATM extraction not yet active (Phase 4)\n";
        }
    }
    
    // Always use placeholders in Phase 2
    m_lng_ustar->setVal(params.test_ustar);
    m_lng_tsfc->setVal(params.test_surf_temp_K);
    
    if (params.lng_debug) {
        amrex::Print() << "[LNG DEBUG] Phase 2: using placeholder u*=" << params.test_ustar
                       << " m/s  T_sfc=" << params.test_surf_temp_K << " K\n";
    }
    
    // Step D: Apply spill source
    if (params.spill_rate_kg_s > 0.0 && dt > 0.0) {
        apply_spill_source(*m_lng_pool_depth, m_lg.geom,
                           params.spill_rate_kg_s, params.rho_LNG,
                           params.pool_area_m2, m_pool_cx, m_pool_cy, dt);
        if (params.lng_debug) {
            amrex::Print() << "[LNG DEBUG] Phase 2: spill source applied  rate="
                           << params.spill_rate_kg_s << " kg/s  dt=" << dt << " s\n";
        }
    }
    
    // Step E: Compute evaporation flux
    compute_lng_evap_flux(*m_lng_evap_flux, *m_lng_latent_flux,
                          *m_lng_pool_mask, *m_lng_ustar,
                          params.zref, m_lg_z0,
                          params.rho_vapor_ref, params.Hv_LNG,
                          params.lng_debug);
    
    // Step F: Deplete pool from evaporation
    if (dt > 0.0) {
        deplete_pool_from_evaporation(*m_lng_pool_depth, *m_lng_evap_flux,
                                      params.rho_LNG, dt, params.lng_debug);
    }
    
    // Step G: Update pool mask
    update_pool_mask(*m_lng_pool_mask, *m_lng_pool_depth);
    
    // Step H: Compute and print mass budget
    amrex::Real pool_mass  = compute_pool_mass(*m_lng_pool_depth, m_lg.geom, params.rho_LNG);
    amrex::Real pool_area  = compute_pool_area(*m_lng_pool_mask,  m_lg.geom);
    amrex::Real ef_max     = m_lng_evap_flux->max(0);
    amrex::Real ef_sum     = m_lng_evap_flux->sum(0);
    amrex::Real lf_max     = m_lng_latent_flux->max(0);
    amrex::Real mask_cells = m_lng_pool_mask->sum(0);
    
    if (params.lng_debug) {
        amrex::Print() << "[LNG DEBUG] Phase 2: step=" << m_step
                       << "  pool_mass=" << pool_mass << " kg"
                       << "  pool_area=" << pool_area << " m^2"
                       << "  active_cells=" << (long)mask_cells << "\n"
                       << "[LNG DEBUG] Phase 2:   evap_flux_max=" << ef_max << " kg/m^2/s"
                       << "  evap_flux_sum=" << ef_sum << " kg/m^2/s"
                       << "  latent_flux_max=" << lf_max << " W/m^2\n";
    }
    
    // Step I: NaN check
    if (params.lng_debug) {
        bool nan_found = false;
        if (m_lng_pool_depth->contains_nan(0))   nan_found = true;
        if (m_lng_pool_mask->contains_nan(0))    nan_found = true;
        if (m_lng_evap_flux->contains_nan(0))    nan_found = true;
        if (m_lng_latent_flux->contains_nan(0))  nan_found = true;
        if (m_lng_vapor_conc->contains_nan(0))   nan_found = true;
        if (m_lng_ustar->contains_nan(0))        nan_found = true;
        if (m_lng_tsfc->contains_nan(0))         nan_found = true;
        
        if (nan_found) {
            amrex::Abort("[LNG] NaN detected in LNG MultiFab at step " + std::to_string(m_step));
        } else {
            amrex::Print() << "[LNG DEBUG] NaN check PASSED step=" << m_step << "\n";
        }
    }
    
    // Verbose=3: print min/max of all fields
    if (params.verbose >= 3) {
        amrex::Print() << "[LNG DEBUG3] step=" << m_step << "\n"
                       << "[LNG DEBUG3]   lng_pool_depth   min=" << std::scientific 
                       << std::setprecision(3) << m_lng_pool_depth->min(0) << "  max=" 
                       << m_lng_pool_depth->max(0) << "  m\n"
                       << "[LNG DEBUG3]   lng_pool_mask    min=" << m_lng_pool_mask->min(0) << "  max=" 
                       << m_lng_pool_mask->max(0) << "\n"
                       << "[LNG DEBUG3]   lng_evap_flux    min=" << m_lng_evap_flux->min(0) << "  max=" 
                       << m_lng_evap_flux->max(0) << "  kg/m^2/s\n"
                       << "[LNG DEBUG3]   lng_latent_flux  min=" << m_lng_latent_flux->min(0) << "  max=" 
                       << m_lng_latent_flux->max(0) << "  W/m^2\n"
                       << "[LNG DEBUG3]   lng_ustar        min=" << m_lng_ustar->min(0) << "  max=" 
                       << m_lng_ustar->max(0) << "  m/s\n"
                       << "[LNG DEBUG3]   lng_tsfc         min=" << m_lng_tsfc->min(0) << "  max=" 
                       << m_lng_tsfc->max(0) << "  K\n";
    }
}

void LNGLayer::apply_to_cc_source(amrex::MultiFab& cc_source,
                                  const amrex::MultiFab& z_phys_cc,
                                  const amrex::Geometry& geom_atm)
{
    if (!m_lng_flux_atm) return;
    if (m_params.atm_feedback <= 0.0) return;

    // Step 1: zero the ATM-grid flux buffer
    m_lng_flux_atm->setVal(0.0);

    // Step 2: sum all evap flux into single-component flux buffer
    amrex::MultiFab::Copy(*m_lng_flux_atm, *m_lng_evap_flux, 0, 0, 1,
                          amrex::IntVect(0));

    // Step 3: coarsen from LNG grid to ATM level-0 2D slab
    coarsen_lng_flux_to_atm(*m_lng_flux_atm, *m_lng_evap_flux,
                             m_lg.geom, geom_atm, m_lg.grid_ratio);

    // Step 4: inject into cc_source at k=0
    apply_lng_tendency_to_cc_source(cc_source, *m_lng_flux_atm, z_phys_cc,
                                    geom_atm, m_lng_scalar_comp,
                                    m_params.atm_feedback,
                                    m_params.lng_debug);

    if (m_params.lng_debug) {
        amrex::Real F_max = m_lng_flux_atm->max(0);
        amrex::Print() << "[LNG DEBUG] Phase 3: apply_to_cc_source step=" << m_step
                       << "  F_evap_atm_max=" << F_max << " kg/m^2/s"
                       << "  scalar_comp=" << m_lng_scalar_comp
                       << "  feedback=" << m_params.atm_feedback << "\n";
    }
}

void LNGLayer::extract_atm_return_fields(const amrex::MultiFab& S_new_cons,
                                         const amrex::Geometry& geom_atm)
{
    // Phase 1 stub: no atmosphere extraction yet
}

void LNGLayer::write_output(int nstep, double cur_time, bool is_final)
{
    if (m_params.lng_plot_int > 0 && nstep % m_params.lng_plot_int == 0) {
        amrex::Print() << "[LNG] Plotfile write stub at step " << nstep << "\n";
    }
    
    // Always write CSV row (not gated on lng_debug)
    append_lng_stats_phase2(nstep, cur_time, m_params.lng_diag_file,
                            m_lng_pool_depth.get(), m_lng_pool_mask.get(),
                            m_lng_evap_flux.get(), m_lng_vapor_conc.get(),
                            m_lg.geom, m_params.rho_LNG);
    
    if (m_params.lng_debug) {
        amrex::Real pool_mass = compute_pool_mass(*m_lng_pool_depth, m_lg.geom, m_params.rho_LNG);
        
        amrex::Print() << "[LNG DEBUG] write_output step=" << nstep 
                       << "  time=" << std::scientific << std::setprecision(3) << cur_time 
                       << "  pool_mass=" << pool_mass << " kg\n";
    }
}