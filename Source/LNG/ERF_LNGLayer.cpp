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
#include "ERF_LNGWindExtract.H"
#include "ERF_LNGAtmReturn.H"
#include "ERF_LNGGravityCurrent.H"
#include "ERF_LNGFlammability.H"
#include "ERF.H"
#include "ERF_IndexDefines.H"
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
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

    // ATM-grid flux on the ATM k=0 slab — same BoxArray as cc_source, matching Dust pattern.
    // Must NOT be coarsened: const_array(mfi) must be valid when iterating over cc_source.
    {
        amrex::BoxArray ba_atm = erf.boxArray(0);
        amrex::Vector<amrex::Box> bl;
        for (int b = 0; b < ba_atm.size(); ++b) {
            amrex::Box bx = ba_atm[b];
            bx.setSmall(2, 0);
            bx.setBig(2, 0);
            bl.push_back(bx);
        }
        amrex::BoxArray ba2d(amrex::BoxList(std::move(bl)));
        m_lng_flux_atm = std::make_unique<amrex::MultiFab>(
            ba2d, erf.DistributionMap(0), ncomp, amrex::IntVect(1,1,0));
    }

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
    m_lng_tsfc->setVal(params.test_surf_temp_K);
    m_lng_pblh->setVal(1000.0);
    m_lng_conc_sfc->setVal(0.0);
    m_lng_lfl_mask->setVal(0.0);
    m_lng_ufl_mask->setVal(0.0);
    
    // Phase 5: gravity current state
    m_lng_gc_h       = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_gc_u       = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_gc_v       = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_gc_ri_flag = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_gc_h->setVal(0.0);
    m_lng_gc_u->setVal(0.0);
    m_lng_gc_v->setVal(0.0);
    m_lng_gc_ri_flag->setVal(0.0);  // 0 = GC active initially
    
    // Set initial pool region: circle of radius sqrt(area/π) at domain center
    amrex::Real pool_radius = std::sqrt(params.pool_area_m2 / M_PI);
    const auto& geom_lng = m_lg.geom;
    const auto& prob_domain = geom_lng.ProbDomain();
    amrex::Real pool_center_x = 0.5 * (prob_domain.lo(0) + prob_domain.hi(0));
    amrex::Real pool_center_y = 0.5 * (prob_domain.lo(1) + prob_domain.hi(1));
    
    // Fill pool_mask and pool_depth
    // Use effective_radius = max(pool_radius, half-diagonal) so at least 1 cell is always seeded
    const auto& dx_lng = geom_lng.CellSize();
    amrex::Real effective_radius = amrex::max(pool_radius,
                                              0.5 * std::sqrt(dx_lng[0]*dx_lng[0] + dx_lng[1]*dx_lng[1]));

    for (amrex::MFIter mfi(*m_lng_pool_mask); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.validbox();
        auto pool_mask_arr  = (*m_lng_pool_mask)[mfi].array();
        auto pool_depth_arr = (*m_lng_pool_depth)[mfi].array();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real x = geom_lng.ProbLo(0) + (i + 0.5) * dx_lng[0];
            amrex::Real y = geom_lng.ProbLo(1) + (j + 0.5) * dx_lng[1];
            amrex::Real r = std::sqrt((x - pool_center_x)*(x - pool_center_x) +
                                      (y - pool_center_y)*(y - pool_center_y));
            if (r <= effective_radius) {
                pool_mask_arr(i, j, k)  = 1.0;
                pool_depth_arr(i, j, k) = params.pool_depth_init_m;
            } else {
                pool_mask_arr(i, j, k)  = 0.0;
                pool_depth_arr(i, j, k) = 0.0;
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
    // Use RhoLNG_comp = RhoScalar_comp + 1 (same pattern as Dust)
    m_lng_scalar_comp = RhoLNG_comp;
     
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
                       << "LFL=" << params.lfl_vol_fraction * 100.0 << "%  "
                       << "UFL=" << params.ufl_vol_fraction * 100.0 << "%\n"
                       << "[LNG DEBUG] Phase 1: grid_ratio=" << params.grid_ratio << "  "
                       << "feedback=" << params.atm_feedback << "  "
                       << "verbose=" << params.verbose << "  "
                       << "debug=" << (params.lng_debug ? "ON" : "OFF") << "\n"
                       << "[LNG DEBUG] Phase 1: LNGGrid created " << m_lg.ba.size() 
                       << " boxes, grid_ratio=" << params.grid_ratio << "\n"
                       << "[LNG DEBUG] Phase 1: MultiFabs allocated (pool_depth, pool_mask, evap_flux, "
                       << "latent_flux, vapor_conc, flux_atm, wind_ref, ustar, tsfc, pblh, conc_sfc, "
                       << "lfl_mask, ufl_mask) ncomp=1\n"
                       << "[LNG DEBUG] Phase 5: gravity current MultiFabs allocated"
                       << " (gc_h, gc_u, gc_v, gc_ri_flag)\n"
                       << "[LNG DEBUG] Phase 5:   enable_gravity_current=" << params.enable_gravity_current
                       << "  Cd=" << params.gc_drag_coeff
                       << "  Ri_crit=" << params.gc_ri_crit << "\n"
                       << "[LNG DEBUG] Phase 5:   g_prime_est="
                       << 9.81 * (params.rho_vapor_ref - params.rho_air) / params.rho_air
                       << " m/s^2  (g*(rho_v - rho_a)/rho_a)\n"
                       << "[LNG DEBUG] Phase 3: lng_scalar_comp=" << m_lng_scalar_comp
                       << " (RhoLNG_comp)\n";
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
                       class SurfaceLayer* surface_layer,
                       const amrex::MultiFab* xvel_mf,
                       const amrex::MultiFab* yvel_mf,
                       const amrex::MultiFab* zvel_mf,
                       const amrex::MultiFab* z_phys_cc_mf,
                       const amrex::MultiFab* S_cons,
                       const amrex::Geometry* geom_atm,
                       int nz)
{
    // Phase 2: Full evaporation physics sequence
    
    // Cache atmospheric pointers for Phase 4 extraction
    m_surface_layer_ptr = surface_layer;
    m_S_cons_ptr        = S_cons;
    m_geom_atm_ptr      = geom_atm;
    
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
    
    // Step C: Extract ATM fields or fall back to placeholders (Phase 4)
    bool have_atm = (xvel_mf && yvel_mf && z_phys_cc_mf && nz > 0);

    if (have_atm) {
        // Extract u* from SurfaceLayer (passed through advance() signature)
        // SurfaceLayer is accessible via ERF.cpp — see DustLayer pattern for wiring
        if (m_surface_layer_ptr && m_surface_layer_ptr->get_u_star(0)) {
            fill_lng_ustar_from_surface_layer(
                *m_lng_ustar, *m_surface_layer_ptr->get_u_star(0), m_lg,
                params.lng_debug);
        }

        // Extract wind at zref via vertical interpolation
        fill_lng_wind_from_interpolation(
            *m_lng_wind_ref, *xvel_mf, *yvel_mf, *z_phys_cc_mf,
            m_lg, params.zref, nz, params.lng_debug);

        // Extract T_sfc from SurfaceLayer
        if (m_surface_layer_ptr && m_surface_layer_ptr->get_t_surf(0)) {
            fill_lng_scalar_from_atm(
                *m_lng_tsfc, *m_surface_layer_ptr->get_t_surf(0), m_lg,
                params.lng_debug, "T_sfc");
        }

        // Extract PBLH from SurfaceLayer
        if (m_surface_layer_ptr && m_surface_layer_ptr->get_pblh(0)) {
            fill_lng_scalar_from_atm(
                *m_lng_pblh, *m_surface_layer_ptr->get_pblh(0), m_lg,
                params.lng_debug, "PBLH");
        }

        if (params.lng_debug)
            amrex::Print() << "[LNG DEBUG] Phase 4: live ATM extraction active"
                           << "  u*_max=" << m_lng_ustar->max(0)
                           << " m/s  u_ref_max=" << m_lng_wind_ref->max(0)
                           << " m/s  PBLH_max=" << m_lng_pblh->max(0) << " m\n";
    } else {
        // Placeholder path — active when no ATM coupling yet
        m_lng_ustar->setVal(params.test_ustar);
        m_lng_tsfc->setVal(params.test_surf_temp_K);
        m_lng_wind_ref->setVal(params.test_wind_speed);

        if (params.lng_debug)
            amrex::Print() << "[LNG DEBUG] Phase 4: placeholder path"
                           << "  test_ustar=" << params.test_ustar
                           << " m/s  test_T_sfc=" << params.test_surf_temp_K
                           << " K  test_wind=" << params.test_wind_speed << " m/s\n";
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
    
    // Step G2: Advance gravity current (Phase 5)
    if (params.enable_gravity_current && dt > 0.0) {
        advance_gravity_current(*m_lng_gc_h, *m_lng_gc_u, *m_lng_gc_v,
                                *m_lng_gc_ri_flag,
                                *m_lng_evap_flux, *m_lng_ustar,
                                m_lg.geom,
                                params.rho_vapor_ref, params.rho_air,
                                params.gc_drag_coeff, dt,
                                params.lng_debug);
        
        if (params.lng_debug) {
            amrex::Real h_max  = m_lng_gc_h->max(0);
            amrex::Real u_max  = m_lng_gc_u->max(0);
            amrex::Real ri_sum = m_lng_gc_ri_flag->sum(0);
            amrex::Long gc_cells = m_lng_gc_h->size() - (long)ri_sum;
            amrex::Print() << "[LNG DEBUG] Phase 5: step=" << m_step
                           << "  gc_h_max=" << h_max << " m"
                           << "  gc_u_max=" << u_max << " m/s"
                           << "  gc_active_cells=" << gc_cells
                           << "  mixed_cells=" << (long)ri_sum << "\n";
        }
    }
    
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
        if (params.enable_gravity_current) {
            if (m_lng_gc_h->contains_nan(0))     nan_found = true;
            if (m_lng_gc_u->contains_nan(0))     nan_found = true;
            if (m_lng_gc_v->contains_nan(0))     nan_found = true;
        }
        
        if (nan_found) {
            amrex::Abort("[LNG] NaN detected in LNG MultiFab at step " + std::to_string(m_step));
        } else {
            amrex::Print() << "[LNG DEBUG] NaN check PASSED step=" << m_step << "\n";
        }
    }
    
    // Step J: Extract return fields from 3D solver (Phase 4)
    // Called with the conserved state from the CURRENT timestep.
    // Fills lng_conc_sfc for future loading feedback.
    if (have_atm && m_S_cons_ptr && m_geom_atm_ptr) {
        fill_lng_conc_from_atm(*m_lng_conc_sfc, *m_S_cons_ptr,
                                m_lng_scalar_comp, *m_geom_atm_ptr, m_lg.grid_ratio);
        if (params.lng_debug)
            amrex::Print() << "[LNG DEBUG] Phase 5: extract_atm_return_fields step=" << m_step
                           << "  conc_sfc_max=" << m_lng_conc_sfc->max(0)
                           << " kg/m^3  conc_sfc_sum=" << m_lng_conc_sfc->sum(0) << "\n";
    }
    
    // Step J2: Compute flammability diagnostics (Phase 5)
    compute_flammability_diagnostics(dt, m_time, m_step);
    
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

    // Step 2: coarsen evap flux from LNG fine grid → ATM k=0 slab
    // m_lng_flux_atm is on ATM BoxArray (k=0 slab) — same as cc_source x,y layout.
    // coarsen_lng_flux_to_atm averages m_lng_evap_flux down by grid_ratio.
    coarsen_lng_flux_to_atm(*m_lng_flux_atm, *m_lng_evap_flux,
                             m_lg.geom, geom_atm, m_lg.grid_ratio);

    // Step 3: inject into cc_source at k=0
    // m_lng_flux_atm now has matching BoxArray to cc_source — const_array(mfi) is valid.
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
    // Phase 5: Extract concentration from atmosphere and map to LNG grid
    if (!m_lng_conc_sfc) return;
    
    fill_lng_conc_from_atm(*m_lng_conc_sfc, S_new_cons,
                            m_lng_scalar_comp, geom_atm, m_lg.grid_ratio);

    if (m_params.lng_debug) {
        amrex::Real conc_max = m_lng_conc_sfc->max(0);
        amrex::Real conc_sum = m_lng_conc_sfc->sum(0);
        amrex::Print() << "[LNG DEBUG] Phase 5: extract_atm_return_fields step=" << m_step
                       << "  conc_sfc_max=" << conc_max << " kg/m^3"
                       << "  conc_sfc_sum=" << conc_sum << "\n";
    }
}

void LNGLayer::compute_flammability_diagnostics(amrex::Real dt, amrex::Real cur_time, int nstep)
{
    // Phase 5: Compute flammability zones from concentration field
    if (!m_lng_conc_sfc) return;
    if (!m_params.track_flammability) return;

    compute_flammability_masks(*m_lng_lfl_mask, *m_lng_ufl_mask,
                               *m_lng_conc_sfc,
                               m_params.rho_vapor_ref, m_params.mol_weight_LNG,
                               m_params.lfl_vol_fraction, m_params.ufl_vol_fraction);

    m_lfl_area = compute_lfl_area(*m_lng_lfl_mask, m_lg.geom);
    m_ufl_area = compute_ufl_area(*m_lng_ufl_mask, m_lg.geom);

    if (m_params.lng_debug) {
        amrex::Real conc_max = m_lng_conc_sfc->max(0);
        amrex::Real vol_frac_max = 0.0;
        if (m_params.rho_vapor_ref > 1.0e-10) {
            vol_frac_max = (conc_max / m_params.rho_vapor_ref) * 
                          (28.97 / m_params.mol_weight_LNG);
        }
        amrex::Print() << "[LNG DEBUG] Phase 5: flammability step=" << nstep
                       << "  lfl_area=" << m_lfl_area << " m^2"
                       << "  ufl_area=" << m_ufl_area << " m^2"
                       << "  conc_sfc_max=" << conc_max << " kg/m^3"
                       << "  vol_frac_max=" << vol_frac_max << "\n";
    }
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