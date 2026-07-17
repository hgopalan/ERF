/**
 * @file ERF_LNGLayer.cpp
 * @brief LNG container class implementation
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
#include "ERF_LNGPlotfile.H"
#include "ERF_LNGReceptorOutput.H"
#include "ERF_LNGRegulatory.H"
#include "ERF.H"
#include "ERF_IndexDefines.H"
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <cmath>
#include <iomanip>

void LNGLayer::initialize(const ERF& erf, const LNGParams& params)
{
    m_params = params;

    check_lng_prerequisites(params, erf.Geom(0), erf.boxArray(0));

    m_lg.build(erf.boxArray(0), erf.DistributionMap(0), erf.Geom(0), params.grid_ratio);

    const int ncomp  = 1;
    const int nghost = 1;
    m_lng_pool_depth    = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_pool_mask     = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_evap_flux     = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_latent_flux   = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_vapor_conc    = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);

    // ATM-grid flux on the ATM k=0 slab — same BoxArray as cc_source
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

    // Wind field (2 components: u, v) — on LNG grid
    m_lng_wind_ref = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, 2, nghost);

    // Scalars — on LNG grid
    m_lng_ustar    = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_tsfc     = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_pblh     = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_conc_sfc = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_lfl_mask = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_ufl_mask = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);

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

    // Phase 5: gravity current state — on LNG grid
    m_lng_gc_h       = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_gc_u       = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_gc_v       = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_gc_ri_flag = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_gc_h->setVal(0.0);
    m_lng_gc_u->setVal(0.0);
    m_lng_gc_v->setVal(0.0);
    m_lng_gc_ri_flag->setVal(0.0);

    // Phase 7: regulatory diagnostics
    m_lng_conc_1h_avg = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_exceed_flag = std::make_unique<amrex::MultiFab>(m_lg.ba, m_lg.dm, ncomp, nghost);
    m_lng_conc_1h_avg->setVal(0.0);
    m_lng_exceed_flag->setVal(0.0);

    // ── ATM-resolution copies for wind extraction ─────────────────────────
    // These live on the ATM BoxArray (same DM as ATM) so ParallelCopy from
    // xvel/yvel/z_phys_cc is always MPI-safe regardless of grid_ratio.
    // fill_lng_wind_from_interpolation reads these, then maps i_a=i_l/C.
    {
        const amrex::BoxArray& ba_atm = erf.boxArray(0);
        const amrex::DistributionMapping& dm_atm = erf.DistributionMap(0);
        int nz = erf.Geom(0).Domain().length(2);
        // xvel is face-centred: one extra x-face per box
        amrex::BoxArray ba_xface = amrex::convert(ba_atm, amrex::IntVect(1,0,0));
        amrex::BoxArray ba_yface = amrex::convert(ba_atm, amrex::IntVect(0,1,0));
        m_xvel_atm = std::make_unique<amrex::MultiFab>(ba_xface, dm_atm, 1, 1);
        m_yvel_atm = std::make_unique<amrex::MultiFab>(ba_yface, dm_atm, 1, 1);
        m_zphys_atm = std::make_unique<amrex::MultiFab>(ba_atm,  dm_atm, 1, 1);
        m_xvel_atm->setVal(0.0);
        m_yvel_atm->setVal(0.0);
        m_zphys_atm->setVal(0.0);
    }

    // Set initial pool region
    amrex::Real pool_radius = std::sqrt(params.pool_area_m2 / M_PI);
    const auto& geom_lng    = m_lg.geom;
    const auto& prob_domain = geom_lng.ProbDomain();
    amrex::Real pool_center_x = 0.5 * (prob_domain.lo(0) + prob_domain.hi(0));
    amrex::Real pool_center_y = 0.5 * (prob_domain.lo(1) + prob_domain.hi(1));
    const auto& dx_lng = geom_lng.CellSize();
    amrex::Real effective_radius = amrex::max(pool_radius,
                                              0.5*std::sqrt(dx_lng[0]*dx_lng[0]+dx_lng[1]*dx_lng[1]));

    for (amrex::MFIter mfi(*m_lng_pool_mask); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.validbox();
        auto pool_mask_arr  = (*m_lng_pool_mask)[mfi].array();
        auto pool_depth_arr = (*m_lng_pool_depth)[mfi].array();
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real x = geom_lng.ProbLo(0) + (i+0.5)*dx_lng[0];
            amrex::Real y = geom_lng.ProbLo(1) + (j+0.5)*dx_lng[1];
            amrex::Real r = std::sqrt((x-pool_center_x)*(x-pool_center_x)+
                                      (y-pool_center_y)*(y-pool_center_y));
            if (r <= effective_radius) {
                pool_mask_arr(i,j,k)  = 1.0;
                pool_depth_arr(i,j,k) = params.pool_depth_init_m;
            } else {
                pool_mask_arr(i,j,k)  = 0.0;
                pool_depth_arr(i,j,k) = 0.0;
            }
        });
    }

    m_lg_z0 = params.z0_lng;
    if (m_pool_cx < 0.0 || m_pool_cy < 0.0) {
        m_pool_cx = 0.5*(prob_domain.lo(0)+prob_domain.hi(0));
        m_pool_cy = 0.5*(prob_domain.lo(1)+prob_domain.hi(1));
    }
    m_lng_ustar->setVal(params.test_ustar);
    m_lng_tsfc->setVal(params.test_surf_temp_K);
    m_lng_scalar_comp = RhoLNG_comp;

    amrex::Real pool_mass_init = compute_pool_mass(*m_lng_pool_depth, geom_lng, params.rho_LNG);
    amrex::Real pool_area_init = compute_pool_area(*m_lng_pool_mask,  geom_lng);

    if (params.lng_debug) {
        amrex::Print() << "[LNG] ===== ERF-LNG Phase 1 initialized =====\n"
                       << "[LNG DEBUG] Phase 1: pool_centre=(" << m_pool_cx << ", " << m_pool_cy << ") m  "
                       << "area=" << pool_area_init << " m^2  depth=" << params.pool_depth_init_m << " m\n"
                       << "[LNG DEBUG] Phase 1: mol_weight_LNG=" << params.mol_weight_LNG << " g/mol  "
                       << "LFL=" << params.lfl_vol_fraction*100.0 << "%  UFL=" << params.ufl_vol_fraction*100.0 << "%\n"
                       << "[LNG DEBUG] Phase 1: grid_ratio=" << params.grid_ratio
                       << "  feedback=" << params.atm_feedback
                       << "  verbose=" << params.verbose
                       << "  debug=" << (params.lng_debug?"ON":"OFF") << "\n"
                       << "[LNG DEBUG] Phase 1: LNGGrid created " << m_lg.ba.size()
                       << " boxes, grid_ratio=" << params.grid_ratio << "\n"
                       << "[LNG DEBUG] Phase 1: MultiFabs allocated (pool_depth, pool_mask, evap_flux, "
                       << "latent_flux, vapor_conc, flux_atm, wind_ref, ustar, tsfc, pblh, conc_sfc, "
                       << "lfl_mask, ufl_mask) ncomp=1\n"
                       << "[LNG DEBUG] Phase 5: gravity current MultiFabs allocated (gc_h, gc_u, gc_v, gc_ri_flag)\n"
                       << "[LNG DEBUG] Phase 5:   enable_gravity_current=" << params.enable_gravity_current
                       << "  Cd=" << params.gc_drag_coeff << "  Ri_crit=" << params.gc_ri_crit << "\n"
                       << "[LNG DEBUG] Phase 5:   g_prime_est="
                       << 9.81*(params.rho_vapor_ref-params.rho_air)/params.rho_air
                       << " m/s^2  (g*(rho_v - rho_a)/rho_a)\n"
                       << "[LNG DEBUG] Phase 3: lng_scalar_comp=" << m_lng_scalar_comp << " (RhoLNG_comp)\n";
    }

    write_lng_stats_header(params.lng_diag_file);
    if (params.lng_debug)
        amrex::Print() << "[LNG DEBUG] Phase 1: lng_diag.csv header written\n";

    // Phase 6: create receptor CSV header files
    for (int r = 0; r < (int)m_params.lng_receptor_names.size(); ++r) {
        write_lng_receptor_header(
            "lng_receptor_" + m_params.lng_receptor_names[r] + ".csv",
            m_params.lng_receptor_names[r],
            m_params.lng_receptor_x[r], m_params.lng_receptor_y[r]);
    }
    if (m_params.lng_debug && !m_params.lng_receptor_names.empty())
        amrex::Print() << "[LNG DEBUG] Phase 6: " << m_params.lng_receptor_names.size()
                       << " receptor file(s) initialized\n";

    // Phase 7: write regulatory CSV header
    write_lng_regulatory_header(m_params.lng_regulatory_file);

    if (m_params.lng_debug)
        amrex::Print() << "[LNG DEBUG] Phase 7: regulatory diagnostics initialized\n"
                       << "[LNG DEBUG] Phase 7:   nfpa59a_exclusion_conc="
                       << m_params.nfpa59a_exclusion_conc << " vol/vol (1/2 LFL)\n"
                       << "[LNG DEBUG] Phase 7:   regulatory_file=" << m_params.lng_regulatory_file << "\n";

    if (params.lng_debug || params.verbose >= 1) {
        amrex::Print() << "[LNG DEBUG] Phase 2: pool evaporation model initialized\n"
                       << "[LNG DEBUG] Phase 2:   pool_centre=(" << m_pool_cx << ", " << m_pool_cy << ") m\n"
                       << "[LNG DEBUG] Phase 2:   pool_area_init=" << pool_area_init << " m^2  "
                       << "pool_depth_init=" << params.pool_depth_init_m << " m\n"
                       << "[LNG DEBUG] Phase 2:   pool_mass_init=" << pool_mass_init << " kg\n"
                       << "[LNG DEBUG] Phase 2:   rho_LNG=" << params.rho_LNG << " kg/m^3  "
                       << "Hv=" << params.Hv_LNG << " J/kg  rho_vapor_ref=" << params.rho_vapor_ref << " kg/m^3\n"
                       << "[LNG DEBUG] Phase 2:   test_ustar=" << params.test_ustar << " m/s  "
                       << "test_surf_temp=" << params.test_surf_temp_K << " K\n"
                       << "[LNG DEBUG] Phase 2:   z0_lng=" << m_lg_z0 << " m  zref=" << params.zref << " m\n"
                       << "[LNG DEBUG] Phase 2:   evap model: k_mass = u* * kappa / (Sc^(2/3) * ln(zref/z0))\n";
    }

    if (params.lng_debug) {
        int pool_cells = amrex::ReduceSum(*m_lng_pool_mask, 0,
            [=] (amrex::Box const& bx, amrex::Array4<amrex::Real const> const& arr) -> int {
                int count = 0;
                amrex::Loop(bx, [&](amrex::IntVect const& iv) { if (arr(iv)>0.5) ++count; });
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
    m_surface_layer_ptr = surface_layer;
    m_S_cons_ptr        = S_cons;
    m_geom_atm_ptr      = geom_atm;

    ++m_step;
    m_time += dt;

    if (params.lng_debug) {
        amrex::Real pool_mass = compute_pool_mass(*m_lng_pool_depth, m_lg.geom, params.rho_LNG);
        amrex::Real ef_max    = m_lng_evap_flux->max(0);
        amrex::Real vc_max    = m_lng_vapor_conc->max(0);
        amrex::Print() << "[LNG DEBUG] advance: step=" << m_step
                       << "  time=" << std::scientific << std::setprecision(3) << m_time << " s"
                       << "  dt=" << dt << " s"
                       << "  pool_mass=" << pool_mass << " kg"
                       << "  evap_flux_max=" << ef_max << " kg/m^2/s"
                       << "  vapor_conc_max=" << vc_max << " kg/m^3\n";
    }

    // ── Phase 4: ATM field extraction ────────────────────────────────────────
    // KEY FIX: copy xvel/yvel/z_phys_cc into ATM-BoxArray-local MFs first via
    // ParallelCopy so that fill_lng_wind_from_interpolation never accesses data
    // owned by a different MPI rank.
    bool have_atm = (xvel_mf && yvel_mf && z_phys_cc_mf && nz > 0);

    if (have_atm) {
        // ParallelCopy: MPI-safe copy into rank-local ATM-grid MFs
        const amrex::Periodicity& per = (geom_atm ? geom_atm->periodicity()
                                                   : amrex::Periodicity::NonPeriodic());
        m_xvel_atm->ParallelCopy(*xvel_mf,   0, 0, 1, 0, 1, per);
        m_yvel_atm->ParallelCopy(*yvel_mf,   0, 0, 1, 0, 1, per);
        m_zphys_atm->ParallelCopy(*z_phys_cc_mf, 0, 0, 1, 0, 1, per);
        m_xvel_atm->FillBoundary(per);
        m_yvel_atm->FillBoundary(per);
        m_zphys_atm->FillBoundary(per);

        if (m_surface_layer_ptr && m_surface_layer_ptr->get_u_star(0))
            fill_lng_ustar_from_surface_layer(
                *m_lng_ustar, *m_surface_layer_ptr->get_u_star(0), m_lg, params.lng_debug);

        // Now safe: m_xvel_atm/m_yvel_atm/m_zphys_atm live on ATM BoxArray
        fill_lng_wind_from_interpolation(
            *m_lng_wind_ref, *m_xvel_atm, *m_yvel_atm, *m_zphys_atm,
            m_lg, params.zref, nz, params.lng_debug);

        if (m_surface_layer_ptr && m_surface_layer_ptr->get_t_surf(0))
            fill_lng_scalar_from_atm(
                *m_lng_tsfc, *m_surface_layer_ptr->get_t_surf(0), m_lg,
                params.lng_debug, "T_sfc");

        if (m_surface_layer_ptr && m_surface_layer_ptr->get_pblh(0))
            fill_lng_scalar_from_atm(
                *m_lng_pblh, *m_surface_layer_ptr->get_pblh(0), m_lg,
                params.lng_debug, "PBLH");

        if (params.lng_debug)
            amrex::Print() << "[LNG DEBUG] Phase 4: live ATM extraction active"
                           << "  u*_max=" << m_lng_ustar->max(0)
                           << " m/s  u_ref_max=" << m_lng_wind_ref->max(0)
                           << " m/s  PBLH_max=" << m_lng_pblh->max(0) << " m\n";
    } else {
        m_lng_ustar->setVal(params.test_ustar);
        m_lng_tsfc->setVal(params.test_surf_temp_K);
        m_lng_wind_ref->setVal(params.test_wind_speed);
        if (params.lng_debug)
            amrex::Print() << "[LNG DEBUG] Phase 4: placeholder path"
                           << "  test_ustar=" << params.test_ustar
                           << " m/s  test_T_sfc=" << params.test_surf_temp_K
                           << " K  test_wind=" << params.test_wind_speed << " m/s\n";
    }

    // ── Phase 2: pool physics ────────────────────────────────────────────────
    if (params.spill_rate_kg_s > 0.0 && dt > 0.0) {
        apply_spill_source(*m_lng_pool_depth, m_lg.geom,
                           params.spill_rate_kg_s, params.rho_LNG,
                           params.pool_area_m2, m_pool_cx, m_pool_cy, dt);
        if (params.lng_debug)
            amrex::Print() << "[LNG DEBUG] Phase 2: spill source applied  rate="
                           << params.spill_rate_kg_s << " kg/s  dt=" << dt << " s\n";
    }

    compute_lng_evap_flux(*m_lng_evap_flux, *m_lng_latent_flux,
                          *m_lng_pool_mask, *m_lng_ustar,
                          params.zref, m_lg_z0,
                          params.rho_vapor_ref, params.Hv_LNG,
                          params.lng_debug);

    if (dt > 0.0)
        deplete_pool_from_evaporation(*m_lng_pool_depth, *m_lng_evap_flux,
                                      m_lg.geom, params.rho_LNG, dt, params.lng_debug);

    update_pool_mask(*m_lng_pool_mask, *m_lng_pool_depth, m_lg.geom);

    // ── Phase 5: gravity current ─────────────────────────────────────────────
    if (params.enable_gravity_current && dt > 0.0) {
        advance_gravity_current(*m_lng_gc_h, *m_lng_gc_u, *m_lng_gc_v,
                                *m_lng_gc_ri_flag,
                                *m_lng_evap_flux, *m_lng_ustar,
                                *m_lng_pool_mask,
                                *m_lng_pool_depth,
                                m_lg.geom,
                                params.rho_vapor_ref, params.rho_air,
                                params.gc_drag_coeff, dt,
                                params.lng_debug);

        if (params.lng_debug) {
            amrex::Real h_max  = m_lng_gc_h->max(0);
            amrex::Real u_max  = m_lng_gc_u->max(0);
            amrex::Real ri_sum = m_lng_gc_ri_flag->sum(0);
            amrex::Long total_gc_cells = static_cast<amrex::Long>(m_lng_gc_h->boxArray().numPts());
            amrex::Long gc_cells = total_gc_cells - static_cast<amrex::Long>(ri_sum);
            amrex::Print() << "[LNG DEBUG] Phase 5: step=" << m_step
                           << "  gc_h_max=" << h_max << " m"
                           << "  gc_u_max=" << u_max << " m/s"
                           << "  gc_active_cells=" << gc_cells
                           << "  mixed_cells=" << (long)ri_sum << "\n";
        }
    }

    // ── Mass budget ──────────────────────────────────────────────────────────
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

    // ── NaN check ────────────────────────────────────────────────────────────
    if (params.lng_debug) {
        bool nan_found = false;
        if (m_lng_pool_depth->contains_nan(0))  nan_found = true;
        if (m_lng_pool_mask->contains_nan(0))   nan_found = true;
        if (m_lng_evap_flux->contains_nan(0))   nan_found = true;
        if (m_lng_latent_flux->contains_nan(0)) nan_found = true;
        if (m_lng_vapor_conc->contains_nan(0))  nan_found = true;
        if (m_lng_ustar->contains_nan(0))       nan_found = true;
        if (m_lng_tsfc->contains_nan(0))        nan_found = true;
        if (params.enable_gravity_current) {
            if (m_lng_gc_h->contains_nan(0))    nan_found = true;
            if (m_lng_gc_u->contains_nan(0))    nan_found = true;
            if (m_lng_gc_v->contains_nan(0))    nan_found = true;
        }
        if (m_lng_conc_1h_avg && m_lng_conc_1h_avg->contains_nan(0)) nan_found = true;
        if (m_lng_exceed_flag  && m_lng_exceed_flag->contains_nan(0))  nan_found = true;
        if (nan_found)
            amrex::Abort("[LNG] NaN detected in LNG MultiFab at step " + std::to_string(m_step));
        else
            amrex::Print() << "[LNG DEBUG] NaN check PASSED step=" << m_step << "\n";
    }

    // ── Phase 5: extract ATM return fields ───────────────────────────────────
    if (have_atm && m_S_cons_ptr && m_geom_atm_ptr) {
        fill_lng_conc_from_atm(*m_lng_conc_sfc, *m_S_cons_ptr,
                               m_lng_scalar_comp, *m_geom_atm_ptr, m_lg.grid_ratio);
        if (params.lng_debug)
            amrex::Print() << "[LNG DEBUG] Phase 5: extract_atm_return_fields step=" << m_step
                           << "  conc_sfc_max=" << m_lng_conc_sfc->max(0)
                           << " kg/m^3  conc_sfc_sum=" << m_lng_conc_sfc->sum(0) << "\n";
    }

    compute_flammability_diagnostics(dt, m_time, m_step);

    // Phase 6: receptor point sampling
    if (m_lng_conc_sfc && !m_params.lng_receptor_names.empty()) {
        for (int r = 0; r < (int)m_params.lng_receptor_names.size(); ++r) {
            append_receptor_sample(m_step, m_time,
                "lng_receptor_" + m_params.lng_receptor_names[r] + ".csv",
                m_params.lng_receptor_names[r],
                m_params.lng_receptor_x[r], m_params.lng_receptor_y[r],
                *m_lng_conc_sfc, m_lg.geom,
                m_params.rho_vapor_ref, m_params.mol_weight_LNG,
                m_params.lfl_vol_fraction);
        }
        if (m_params.lng_debug)
            amrex::Print() << "[LNG DEBUG] Phase 6: receptor sampling step=" << m_step
                           << "  n_receptors=" << m_params.lng_receptor_names.size() << "\n";
    }

    // ── Phase 7: regulatory compliance ───────────────────────────────────────
    if (m_lng_conc_sfc && m_lng_conc_1h_avg && m_lng_exceed_flag) {
            // Update 1-hour running average
            //update_lng_1h_average(*m_lng_conc_1h_avg, *m_lng_conc_sfc, dt);
            update_lng_1h_average(*m_lng_conc_1h_avg, *m_lng_conc_sfc, m_lg.geom, dt);
            // Compute NFPA exceedance (uses nfpa59a_exclusion_conc = 1/2 LFL)
            compute_lng_exceedance(*m_lng_exceed_flag, *m_lng_conc_1h_avg,m_lg.geom,
                                    m_params.rho_vapor_ref, m_params.mol_weight_LNG,
                                    m_params.nfpa59a_exclusion_conc);

            // Estimate exclusion zone radius
            m_exclusion_radius_m = compute_exclusion_zone_radius(
                *m_lng_exceed_flag, m_lg.geom, m_pool_cx, m_pool_cy);

            if (m_params.lng_debug)
                amrex::Print() << "[LNG DEBUG] Phase 7: step=" << m_step
                               << "  exclusion_radius=" << m_exclusion_radius_m << " m"
                               << "  conc_1h_max=" << m_lng_conc_1h_avg->max(0) << " kg/m^3"
                               << "  n_exceed=" << (long)m_lng_exceed_flag->sum(0) << "\n";
    }

    if (params.verbose >= 3) {
        amrex::Print() << "[LNG DEBUG3] step=" << m_step << "\n"
                       << "[LNG DEBUG3]   lng_pool_depth   min=" << std::scientific
                       << std::setprecision(3) << m_lng_pool_depth->min(0) << "  max="
                       << m_lng_pool_depth->max(0) << "  m\n"
                       << "[LNG DEBUG3]   lng_pool_mask    min=" << m_lng_pool_mask->min(0)
                       << "  max=" << m_lng_pool_mask->max(0) << "\n"
                       << "[LNG DEBUG3]   lng_evap_flux    min=" << m_lng_evap_flux->min(0)
                       << "  max=" << m_lng_evap_flux->max(0) << "  kg/m^2/s\n"
                       << "[LNG DEBUG3]   lng_latent_flux  min=" << m_lng_latent_flux->min(0)
                       << "  max=" << m_lng_latent_flux->max(0) << "  W/m^2\n"
                       << "[LNG DEBUG3]   lng_ustar        min=" << m_lng_ustar->min(0)
                       << "  max=" << m_lng_ustar->max(0) << "  m/s\n"
                       << "[LNG DEBUG3]   lng_tsfc         min=" << m_lng_tsfc->min(0)
                       << "  max=" << m_lng_tsfc->max(0) << "  K\n";
    }
}

void LNGLayer::apply_to_cc_source(amrex::MultiFab& cc_source,
                                  const amrex::MultiFab& z_phys_cc,
                                  const amrex::Geometry& geom_atm)
{
    if (!m_lng_flux_atm) return;
    if (m_params.atm_feedback <= 0.0) return;

    m_lng_flux_atm->setVal(0.0);

    coarsen_lng_flux_to_atm(*m_lng_flux_atm, *m_lng_evap_flux,
                             m_lg.geom, geom_atm, m_lg.grid_ratio);

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
    if (!m_lng_conc_sfc) return;
    fill_lng_conc_from_atm(*m_lng_conc_sfc, S_new_cons,
                            m_lng_scalar_comp, geom_atm, m_lg.grid_ratio);
    if (m_params.lng_debug) {
        amrex::Print() << "[LNG DEBUG] Phase 5: extract_atm_return_fields step=" << m_step
                       << "  conc_sfc_max=" << m_lng_conc_sfc->max(0)
                       << " kg/m^3  conc_sfc_sum=" << m_lng_conc_sfc->sum(0) << "\n";
    }
}

void LNGLayer::compute_flammability_diagnostics(amrex::Real dt, amrex::Real cur_time, int nstep)
{
    if (!m_lng_conc_sfc) return;
    if (!m_params.track_flammability) return;

    compute_flammability_masks(*m_lng_lfl_mask, *m_lng_ufl_mask,
                               *m_lng_conc_sfc,
                               m_params.rho_vapor_ref, m_params.mol_weight_LNG,
                               m_params.lfl_vol_fraction, m_params.ufl_vol_fraction);

    m_lfl_area = compute_lfl_area(*m_lng_lfl_mask, m_lg.geom);
    m_ufl_area = compute_ufl_area(*m_lng_ufl_mask, m_lg.geom);

    if (m_params.lng_debug) {
        amrex::Real conc_max    = m_lng_conc_sfc->max(0);
        amrex::Real vol_frac_max = 0.0;
        if (m_params.rho_vapor_ref > 1.0e-10)
            vol_frac_max = (conc_max / m_params.rho_vapor_ref) * (28.97 / m_params.mol_weight_LNG);
        amrex::Print() << "[LNG DEBUG] Phase 5: flammability step=" << nstep
                       << "  lfl_area=" << m_lfl_area << " m^2"
                       << "  ufl_area=" << m_ufl_area << " m^2"
                       << "  conc_sfc_max=" << conc_max << " kg/m^3"
                       << "  vol_frac_max=" << vol_frac_max << "\n";
    }
}

void LNGLayer::write_output(int nstep, double cur_time, bool is_final)
{
    // Guard against duplicate calls at the same step
    if (nstep == m_last_output_step) return;
    m_last_output_step = nstep;

    // Phase 6: plotfile output
    bool write_plt = false;
    if (m_params.lng_plot_int > 0) {
        write_plt = (nstep % m_params.lng_plot_int == 0);
    }
    if (is_final) {
        write_plt = true;
    }

    if (write_plt) {
        WriteLNGPlotfile(m_params.lng_plot_prefix, *this, cur_time, nstep);
        if (m_params.lng_debug)
            amrex::Print() << "[LNG DEBUG] Phase 6: plotfile written step=" << nstep
                           << "  is_final=" << is_final << "\n";
    }

    // MPI rule (LNG_MPI_SKILLS.md B1): all reductions before IOProcessor guard
    // lfl_area and ufl_area are scalar reals computed in advance() — no new reduction needed here
    append_lng_stats_phase2(nstep, cur_time, m_params.lng_diag_file,
                            m_lng_pool_depth.get(), m_lng_pool_mask.get(),
                            m_lng_evap_flux.get(), m_lng_vapor_conc.get(),
                            m_lg.geom, m_params.rho_LNG,
                            m_lfl_area, m_ufl_area);

    // Phase 7: regulatory CSV (always write, not gated on lng_debug — Rule A4)
    if (m_lng_conc_1h_avg && m_lng_exceed_flag) {
        append_lng_regulatory_row(nstep, cur_time,
                                  m_params.lng_regulatory_file,
                                  m_exclusion_radius_m,
                                  *m_lng_conc_1h_avg,
                                  *m_lng_exceed_flag);
    }

    if (m_params.lng_debug) {
        amrex::Real pool_mass = compute_pool_mass(*m_lng_pool_depth, m_lg.geom, m_params.rho_LNG);
        amrex::Print() << "[LNG DEBUG] write_output step=" << nstep
                       << "  time=" << std::scientific << std::setprecision(3) << cur_time
                       << "  pool_mass=" << pool_mass << " kg\n";
    }
}