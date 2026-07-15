/**
 * @file ERF_LNGLayer.cpp
 * @brief LNG container class implementation
 * @details
 * Implements LNGLayer methods for grid construction, MultiFab allocation,
 * and timestep cycling (all stubs in Phase 1).
 */

#include "ERF_LNGLayer.H"
#include "ERF_LNGPrerequisites.H"
#include "ERF_LNGStatsOutput.H"
#include "ERF.H"
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <cmath>

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
        auto& pool_mask_arr = (*m_lng_pool_mask)[mfi];
        auto& pool_depth_arr = (*m_lng_pool_depth)[mfi];
        
        amrex::ParallelFor(bx, [=] (amrex::IntVect const& iv) noexcept {
            amrex::Real x = geom_lng.CellCenter(iv, 0);
            amrex::Real y = geom_lng.CellCenter(iv, 1);
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
    
    // Write CSV diagnostic header
    write_lng_stats_header(params.lng_diag_file);
    
    // Print per-step debug header if enabled
    if (params.lng_debug) {
        int pool_cells = 0;
        for (amrex::MFIter mfi(*m_lng_pool_mask); mfi.isValid(); ++mfi) {
            const auto& pool_mask_arr = (*m_lng_pool_mask)[mfi];
            pool_cells += amrex::ReduceOps::Sum<int>::value(
                amrex::ReduceData<amrex::ReduceOps::Sum<int>>,
                mfi.validbox(),
                pool_mask_arr,
                [=] (amrex::IntVect const& iv) { return (pool_mask_arr(iv) > 0.5) ? 1 : 0; });
        }
        amrex::Print() << "[LNG DEBUG] Step  " << std::setw(4) << m_step 
                       << "  time=" << std::scientific << std::setprecision(3) << m_time 
                       << "  pool_cells=" << pool_cells
                       << "  evap_flux_max=0.000e+00 kg/m^2/s  vapor_conc_max=0.000e+00 kg/m^3\n";
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
    // Phase 1 stub: no physics, just housekeeping
    ++m_step;
    m_time += dt;
    
    if (params.lng_debug) {
        amrex::Print() << "[LNG DEBUG] Step " << std::setw(4) << m_step 
                       << "  time=" << std::scientific << std::setprecision(3) << m_time 
                       << "  advance() stub — no physics in Phase 1\n";
        
        // Check for NaNs in all MultiFabs
        bool has_nan = false;
        if (m_lng_pool_depth->contains_nan(0))   has_nan = true;
        if (m_lng_pool_mask->contains_nan(0))    has_nan = true;
        if (m_lng_evap_flux->contains_nan(0))    has_nan = true;
        if (m_lng_latent_flux->contains_nan(0))  has_nan = true;
        if (m_lng_vapor_conc->contains_nan(0))   has_nan = true;
        if (m_lng_wind_ref->contains_nan(0))     has_nan = true;
        if (m_lng_wind_ref->contains_nan(1))     has_nan = true;
        
        if (has_nan) {
            amrex::Abort("[LNG] NaN detected in MultiFab at step " + std::to_string(m_step));
        }
        amrex::Print() << "[LNG DEBUG] NaN check PASSED\n";
    }
    
    // Verbose=3: print min/max of all fields
    if (params.verbose >= 3) {
        amrex::Print() << "[LNG DEBUG3]   lng_pool_depth   min=" << std::scientific 
                       << std::setprecision(3) << m_lng_pool_depth->min(0) << "  max=" 
                       << m_lng_pool_depth->max(0) << "  m\n";
        amrex::Print() << "[LNG DEBUG3]   lng_evap_flux    min=" << std::scientific 
                       << std::setprecision(3) << m_lng_evap_flux->min(0) << "  max=" 
                       << m_lng_evap_flux->max(0) << "  kg/m^2/s\n";
        amrex::Print() << "[LNG DEBUG3]   lng_vapor_conc   min=" << std::scientific 
                       << std::setprecision(3) << m_lng_vapor_conc->min(0) << "  max=" 
                       << m_lng_vapor_conc->max(0) << "  kg/m^3\n";
    }
}

void LNGLayer::apply_to_cc_source(amrex::MultiFab& cc_source,
                                  const amrex::MultiFab& z_phys_cc,
                                  const amrex::Geometry& geom_atm)
{
    // Phase 1 stub: no atmosphere coupling yet
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
    
    if (m_params.lng_debug) {
        // Append CSV row with all zeros (no physics in Phase 1)
        append_lng_stats(nstep, cur_time, m_params.lng_diag_file,
                         m_lng_evap_flux.get(), m_lng_tsfc.get(), m_lng_vapor_conc.get());
        
        amrex::Print() << "[LNG DEBUG] write_output step=" << nstep 
                       << "  time=" << std::scientific << std::setprecision(3) << cur_time 
                       << "  total_pool_mass=0.000 kg  total_vapor_mass=0.000 kg\n";
    }
}

#include <iomanip>
