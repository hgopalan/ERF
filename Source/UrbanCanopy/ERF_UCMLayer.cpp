/**
 * @file ERF_UCMLayer.cpp
 * @brief Implementation of UCMLayer physics driver for facet SEB and conduction
 *
 * Phase 1.3 simplified implementation of SLUCM integration:
 * 1. Extract forcing (u*, wind, T, q, SW/LW)
 * 2. Solve facet SEB via Newton iteration
 * 3. Advance slab conduction
 * 4. Compute and store fluxes
 *
 * References:
 *  - Kusaka et al. (2001)
 *  - Chen et al. (2011)
 *  - WRF phys/module_sf_urban.F
 */

#include <UrbanCanopy/ERF_UCMLayer.H>
#include <UrbanCanopy/ERF_UCMSlabConduction.H>
#include <UrbanCanopy/ERF_UCMSEBSolver.H>
#include <UrbanCanopy/ERF_UCMAllocate.H>
#include <UrbanCanopy/ERF_UCMShadowing.H>
#include <UrbanCanopy/ERF_UCMStabilityCorrection.H>
#include <UrbanCanopy/ERF_UCMRadiationForcing.H>
#include <UrbanCanopy/ERF_UCMViewFactors.H>
#include <UrbanCanopy/ERF_UCMRadiationExtraction.H>
#include <UrbanCanopy/ERF_UCMRadiosity.H>
#include <UrbanCanopy/ERF_UCMRadiosityLW.H>
#include <UrbanCanopy/ERF_UCMHVAC.H>
#include <UrbanCanopy/ERF_UCMHVACReader.H>
#include <UrbanCanopy/ERF_UCMOccupancyReader.H>
#include <AMReX_ParallelDescriptor.H>
#include <ERF_Constants.H>
#include <cmath>

// ============================================================================
// Constructors
// ============================================================================

UCMLayer::UCMLayer(const UCMParams& params, int lev)
    : m_params(params), m_warn_radiation_placeholder_printed(false)
{
    // Phase 1.1: enforce anchor_level == 0
    if (lev != params.anchor_level) {
        std::string msg = std::string("[UCM] UCMLayer constructed at level ")
                        + std::to_string(lev) + " but params.anchor_level = "
                        + std::to_string(params.anchor_level)
                        + ". Phase 1.3 supports only anchor_level=0.";
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false, msg.c_str());
    }
}

// ============================================================================
// Main advance function
// ============================================================================

void UCMLayer::advance(UCMFields& fields,
                       UCMForcing& forcing,
                       const UCMGrid& ucm_grid,
                       const amrex::MultiFab& atm_u_star,
                       const amrex::MultiFab& atm_t_star,
                       const amrex::MultiFab& /*atm_q_star*/,
                       const amrex::MultiFab& xvel,
                       const amrex::MultiFab& yvel,
                       const amrex::MultiFab& z_phys_cc,
                       const amrex::MultiFab& T_atm_lowest,
                       const amrex::MultiFab& q_atm_lowest,
                       const amrex::MultiFab& atm_olen,
                       const amrex::Geometry& geom_atm,
                       amrex::Real time,
                       amrex::Real dt,
                       int nz_atm,
                       int lev)
{
    // ========================================================================
    // Initialize output flux fields and debug print refined ATM inputs
    // ========================================================================

    fields.H_sensible->setVal(0.0);
    fields.LE_latent->setVal(0.0);

    // Phase 2.3: zero-init facet-split fluxes
    fields.H_road->setVal(0.0);
    fields.H_wall->setVal(0.0);
    fields.H_roof->setVal(0.0);
    // Phase 3.5A-hotfix2: zero-init ATM injection fluxes
    fields.H_road_atm->setVal(0.0);
    fields.H_wall_atm->setVal(0.0);
    fields.H_roof_atm->setVal(0.0);

    // Phase 3.5a-hotfix3: T_skin persistence diagnostic (tracks slab conduction leak)
    if (m_params.ucm_debug) {
        amrex::Real T_roof_min  = fields.T_skin_roof->min(0, 0);
        amrex::Real T_roof_max  = fields.T_skin_roof->max(0, 0);
        amrex::Real T_wall_min  = fields.T_skin_wall->min(0, 0);
        amrex::Real T_wall_max  = fields.T_skin_wall->max(0, 0);
        amrex::Real T_road_min  = fields.T_skin_road->min(0, 0);
        amrex::Real T_road_max  = fields.T_skin_road->max(0, 0);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A-hotfix3][entry] T_skin_roof=[" << T_roof_min
                           << "," << T_roof_max << "] K; T_skin_wall=[" << T_wall_min
                           << "," << T_wall_max << "] K; T_skin_road=[" << T_road_min
                           << "," << T_road_max << "] K\n";
        }
    }

    // Phase 3.5A-diag: T_slab state at UCMLayer::advance() entry.
    if (m_params.ucm_debug) {
        amrex::Real T1_roof_min = fields.T_slab_roof->min(0, 0);
        amrex::Real T1_roof_max = fields.T_slab_roof->max(0, 0);
        amrex::Real T1_wall_min = fields.T_slab_wall->min(0, 0);
        amrex::Real T1_wall_max = fields.T_slab_wall->max(0, 0);
        amrex::Real T1_road_min = fields.T_slab_road->min(0, 0);
        amrex::Real T1_road_max = fields.T_slab_road->max(0, 0);
        const int nlyr = fields.T_slab_roof->nComp() - 1;
        amrex::Real TN_roof_min = fields.T_slab_roof->min(nlyr, 0);
        amrex::Real TN_roof_max = fields.T_slab_roof->max(nlyr, 0);
        amrex::Real TN_wall_min = fields.T_slab_wall->min(nlyr, 0);
        amrex::Real TN_wall_max = fields.T_slab_wall->max(nlyr, 0);
        amrex::Real TN_road_min = fields.T_slab_road->min(nlyr, 0);
        amrex::Real TN_road_max = fields.T_slab_road->max(nlyr, 0);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A-diag][slab-top-ENTRY] "
                           << "T_slab_roof[0]=[" << T1_roof_min << "," << T1_roof_max << "] K "
                           << "T_slab_wall[0]=[" << T1_wall_min << "," << T1_wall_max << "] K "
                           << "T_slab_road[0]=[" << T1_road_min << "," << T1_road_max << "] K\n"
                           << "[UCM][3.5A-diag][slab-deep-ENTRY] "
                           << "T_slab_roof[" << nlyr << "]=[" << TN_roof_min << "," << TN_roof_max << "] K "
                           << "T_slab_wall[" << nlyr << "]=[" << TN_wall_min << "," << TN_wall_max << "] K "
                           << "T_slab_road[" << nlyr << "]=[" << TN_road_min << "," << TN_road_max << "] K\n";
        }
    }

    // Debug: per-step ATM forcing summary on UCM grid
    if (m_params.ucm_debug) {
        amrex::Real ust_min = atm_u_star.min(0, 0), ust_max = atm_u_star.max(0, 0);
        amrex::Real tst_min = atm_t_star.min(0, 0), tst_max = atm_t_star.max(0, 0);
        amrex::Real Tat_min = T_atm_lowest.min(0, 0), Tat_max = T_atm_lowest.max(0, 0);
        amrex::Real Uat_min = xvel.min(0, 0), Uat_max = xvel.max(0, 0);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.3][UCMLayer::advance] ATM forcing on UCM grid:"
                           << " u_star=[" << ust_min << "," << ust_max << "] m/s"
                           << " t_star=[" << tst_min << "," << tst_max << "] K"
                           << " T_atm=["  << Tat_min << "," << Tat_max << "] K"
                           << " U_atm=["  << Uat_min << "," << Uat_max << "] m/s\n";
        }
    }

    // Phase 2.2: One-time banner
    static bool banner_printed = false;
    if (!banner_printed && m_params.ucm_debug) {
        banner_printed = true;
        amrex::Real H_min = fields.H_bldg->min(0, 0);
        amrex::Real H_max = fields.H_bldg->max(0, 0);
        amrex::Real alb_min = fields.albedo_roof->min(0, 0);
        amrex::Real alb_max = fields.albedo_roof->max(0, 0);
        amrex::Real k_min = fields.k_therm_roof->min(0, 0);
        amrex::Real k_max = fields.k_therm_roof->max(0, 0);
        amrex::Real z0_min = fields.z0_ucm->min(0, 0);
        amrex::Real z0_max = fields.z0_ucm->max(0, 0);
        amrex::Real d_min = fields.d_disp_ucm->min(0, 0);
        amrex::Real d_max = fields.d_disp_ucm->max(0, 0);

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n[UCM][2.2][BANNER] Phase 2.2 per-cell wiring active:\n"
                           << "  H_bldg      min=" << H_min
                           << " max=" << H_max << " m\n"
                           << "  albedo_roof min=" << alb_min
                           << " max=" << alb_max << "\n"
                           << "  k_therm_roof min=" << k_min
                           << " max=" << k_max << " W/m/K\n"
                           << "  z0          min=" << z0_min
                           << " max=" << z0_max << " m\n"
                           << "  d_disp      min=" << d_min
                           << " max=" << d_max << " m\n\n";
        }
    }

    // ========================================================================
    // Step 1: Allocate and extract forcing (u*, wind, T, q, SW/LW)
    // ========================================================================

    if (!forcing.all_allocated()) {
        forcing.u_star = std::make_unique<amrex::MultiFab>(
            ucm_grid.ba, ucm_grid.dm, 1, amrex::IntVect(1, 1, 0));
        forcing.wind_ref = std::make_unique<amrex::MultiFab>(
            ucm_grid.ba, ucm_grid.dm, 2, amrex::IntVect(1, 1, 0));
        forcing.T_atm_ref = std::make_unique<amrex::MultiFab>(
            ucm_grid.ba, ucm_grid.dm, 1, amrex::IntVect(1, 1, 0));
        forcing.q_atm_ref = std::make_unique<amrex::MultiFab>(
            ucm_grid.ba, ucm_grid.dm, 1, amrex::IntVect(1, 1, 0));
        forcing.SW_down = std::make_unique<amrex::MultiFab>(
            ucm_grid.ba, ucm_grid.dm, 1, amrex::IntVect(1, 1, 0));
        forcing.LW_down = std::make_unique<amrex::MultiFab>(
            ucm_grid.ba, ucm_grid.dm, 1, amrex::IntVect(1, 1, 0));
    }

    fill_ucm_ustar_from_surface_layer(*forcing.u_star, atm_u_star, ucm_grid, lev);

    fill_ucm_wind_from_interpolation(*forcing.wind_ref, xvel, yvel, z_phys_cc,
                                     *fields.H_bldg, *fields.z0_ucm, *fields.d_disp_ucm,
                                     m_params.zref, ucm_grid, nz_atm, lev);

    fill_ucm_scalar_from_atm(*forcing.T_atm_ref, T_atm_lowest, ucm_grid, geom_atm, lev, 0);

    if (q_atm_lowest.boxArray().size() > 0) {
        fill_ucm_scalar_from_atm(*forcing.q_atm_ref, q_atm_lowest, ucm_grid, geom_atm, lev, 0);
    }

    if (m_params.ucm_debug) {
        amrex::Real ustar_min = forcing.u_star->min(0), ustar_max = forcing.u_star->max(0);
        amrex::Real T_min     = forcing.T_atm_ref->min(0), T_max = forcing.T_atm_ref->max(0);
        amrex::Real U_min     = forcing.wind_ref->min(0), U_max = forcing.wind_ref->max(0);
        amrex::Real V_min     = forcing.wind_ref->min(1), V_max = forcing.wind_ref->max(1);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.3][UCMLayer::advance] extraction results:\n";
            amrex::Print() << "  u_star=[" << ustar_min << "," << ustar_max << "] m/s"
                           << "  T_ref=[" << T_min << "," << T_max << "] K"
                           << "  U=[" << U_min << "," << U_max << "]"
                           << "  V=[" << V_min << "," << V_max << "] m/s\n";
        }
    }

    // ========================================================================
    // Step 2: Fill radiation (analytical placeholder, Phase 1.3)
    // ========================================================================

    forcing.LW_down->setVal(350.0);

    amrex::Real phase = 2.0*M_PI*time/86400.0 - 0.5*M_PI;
    amrex::Real sw_val = 800.0 * std::max(0.0, std::cos(phase));
    forcing.SW_down->setVal(sw_val);

    if (!m_warn_radiation_placeholder_printed &&
       m_params.radiation_source == RadiationSource::Analytic &&
       m_params.cloud_source == CloudSource::None)
    {
       if (amrex::ParallelDescriptor::IOProcessor()) {
           amrex::Print() << "[UCM][1.3][WARNING] Radiation (SW/LW) filled analytically. "
                         << "Phase 4.3 placeholder allows switching to erf radiation source, "
                         << "but real extraction is deferred until RRTMG interface stabilizes.\n";
       }
       m_warn_radiation_placeholder_printed = true;
    }

    // ========================================================================
    // Step 2.4: Sky view factors + multi-facet view factors
    // ========================================================================
    compute_sky_view_factors(*fields.SVF_wall, *fields.SVF_road, *fields.SVF_roof,
                             *fields.H_bldg, *fields.W_road, *fields.is_urban,
                             lev, m_params.ucm_debug);

    static bool view_factors_computed = false;
    if (!view_factors_computed) {
        view_factors_computed = true;
        compute_view_factors(*fields.F_wall_sky,
                             *fields.F_wall_wall,
                             *fields.F_wall_road,
                             *fields.F_road_sky,
                             *fields.F_road_wall,
                             *fields.F_roof_sky,
                             *fields.H_bldg,
                             *fields.W_road,
                             *fields.is_urban,
                             lev,
                             m_params.ucm_debug);
    }

    // ========================================================================
    // Step 3: Solve facet SEB and advance slab conduction
    // ========================================================================

    const amrex::Real sigma_sb = 5.670374419e-8;
    amrex::ignore_unused(sigma_sb);
    const amrex::Real rho_cp   = 1.2 * Cp_d;
    const amrex::Real Ch_roof  = m_params.Ch_roof;
    const amrex::Real Ch_wall  = m_params.Ch_wall;
    const amrex::Real Ch_road  = m_params.Ch_road;
    const amrex::Real dz_slab  = m_params.slab_dz;
    const int         max_iter = m_params.newton_max_iter;
    const amrex::Real tol_K    = m_params.newton_tol_K;

    amrex::Real SW_down = 0.0;
    amrex::Real LW_down = 350.0;

    if (m_params.radiation_source == RadiationSource::Erf) {
        amrex::Real time_s_local = m_params.solar_time_start_s + time;
        extract_radiation_from_erf(SW_down, LW_down, time_s_local);

        if (SW_down < 0.0 || LW_down < 0.0) {
            amrex::Abort(
                "[UCM][4.3] extract_radiation_from_erf returned sentinel "
                "(SW=" + std::to_string(SW_down) +
                ", LW=" + std::to_string(LW_down) +
                "). Phase 4.3 real extraction is not yet implemented. "
                "Set erf.ucm.radiation_source=analytic.");
        }

        if (SW_down < 0.0 || SW_down > 1500.0 || LW_down < 100.0 || LW_down > 600.0) {
            if (amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print()
                    << "[UCM][4.3][WARNING] extract_radiation_from_erf returned "
                    << "out-of-range values: SW=" << SW_down
                    << " W/m² (expected [0, 1500]), LW=" << LW_down
                    << " W/m² (expected [100, 600]). Check unit conversion and "
                    << "MultiFab access in the extraction path.\n";
            }
        }
    }
    else if (m_params.use_prescribed_radiation) {
        amrex::Real time_s_local = m_params.solar_time_start_s + time;
        amrex::Real lat_rad = m_params.lat_deg * (3.14159265358979323846 / 180.0);
        amrex::Real lon_rad = m_params.lon_deg * (3.14159265358979323846 / 180.0);
        amrex::ignore_unused(lon_rad);

        amrex::Real cos_zenith = solar_zenith_angle(time_s_local, lat_rad, lon_rad,
                                                    m_params.julian_day);

        amrex::Real SW_clear = clear_sky_SW_down(cos_zenith,
                                                  m_params.solar_constant,
                                                  m_params.sw_transmission);

        amrex::Real T_atm_min    = T_atm_lowest.min(0, 0);
        amrex::Real T_atm_max    = T_atm_lowest.max(0, 0);
        amrex::Real T_atm_approx = 0.5 * (T_atm_min + T_atm_max);

        amrex::Real LW_clear = gray_sky_LW_down(T_atm_approx, m_params.sky_emissivity);

        amrex::Real cf = 0.0;
        if (m_params.cloud_source == CloudSource::Constant) {
            cf = m_params.cloud_constant_fraction;
        } else if (m_params.cloud_source == CloudSource::Csv) {
            if (!m_cloud_csv_load_attempted) {
                m_cloud_csv_load_attempted = true;
                try {
                    m_cloud_csv_reader = std::make_unique<UCMCloudCSVReader>(
                        m_params.cloud_csv_path);
                } catch (...) {
                    if (amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "[UCM][4.2][ERROR] Failed to load cloud CSV '"
                                       << m_params.cloud_csv_path
                                       << "'; falling back to cf=0.\n";
                    }
                    m_cloud_csv_reader.reset();
                }
            }
            if (m_cloud_csv_reader) {
                cf = m_cloud_csv_reader->get_cloud_fraction_at(
                    time_s_local, m_params.ucm_debug);
            }
        }

        SW_down = cloud_attenuated_SW_down(SW_clear, cf,
                                            m_params.cloud_sw_a,
                                            m_params.cloud_sw_b);
        LW_down = cloud_enhanced_LW_down(LW_clear, cf, T_atm_approx);

        if (m_params.ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][4.2][radiation-cloud]"
                           << " sim_time_s=" << time_s_local
                           << " SW_down_clear=" << SW_clear
                           << " SW_down_cloudy=" << SW_down
                           << " LW_down_clear=" << LW_clear
                           << " LW_down_cloudy=" << LW_down
                           << " cloud_fraction=" << cf << "\n";
        }
    }

    const int         N_layers = m_params.slab_N_layers;
    const amrex::Real slab_L   = m_params.slab_L;
    const amrex::Real T_deep   = m_params.slab_T_deep;

    amrex::Long n_clamped_roof  = 0;
    amrex::Long n_clamped_wall  = 0;
    amrex::Long n_clamped_road  = 0;
    amrex::Long n_diverged_roof = 0;
    amrex::Long n_diverged_wall = 0;
    amrex::Long n_diverged_road = 0;

    amrex::Long* p_clamped_roof  = &n_clamped_roof;
    amrex::Long* p_diverged_roof = &n_diverged_roof;
    amrex::Long* p_clamped_wall  = &n_clamped_wall;
    amrex::Long* p_diverged_wall = &n_diverged_wall;
    amrex::Long* p_clamped_road  = &n_clamped_road;
    amrex::Long* p_diverged_road = &n_diverged_road;

    amrex::MultiFab newton_diag_roof(fields.T_skin_roof->boxArray(),
                                     fields.T_skin_roof->DistributionMap(), 8, 0);
    amrex::MultiFab newton_diag_wall(fields.T_skin_wall->boxArray(),
                                     fields.T_skin_wall->DistributionMap(), 8, 0);
    amrex::MultiFab newton_diag_road(fields.T_skin_road->boxArray(),
                                     fields.T_skin_road->DistributionMap(), 8, 0);
    newton_diag_roof.setVal(0.0);
    newton_diag_wall.setVal(0.0);
    newton_diag_road.setVal(0.0);

    for (amrex::MFIter mfi(*fields.T_skin_roof, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();

        auto const is_urb   = fields.is_urban->const_array(mfi);
        auto const svf_wall = fields.SVF_wall->const_array(mfi);
        auto const svf_road = fields.SVF_road->const_array(mfi);
        auto const alb_rf   = fields.albedo_roof->const_array(mfi);
        auto const alb_wl   = fields.albedo_wall->const_array(mfi);
        auto const alb_rd   = fields.albedo_road->const_array(mfi);
        auto const eps_rf   = fields.emissivity_roof->const_array(mfi);
        auto const eps_wl   = fields.emissivity_wall->const_array(mfi);
        auto const eps_rd   = fields.emissivity_road->const_array(mfi);
        auto const k_rf     = fields.k_therm_roof->const_array(mfi);
        auto const k_wl     = fields.k_therm_wall->const_array(mfi);
        auto const k_rd     = fields.k_therm_road->const_array(mfi);
        auto const U_a      = forcing.wind_ref->const_array(mfi);
        auto const T_can_a  = fields.T_canyon_air->const_array(mfi);

        auto const T1_rf    = fields.T_slab_roof->const_array(mfi);
        auto const T1_wl    = fields.T_slab_wall->const_array(mfi);
        auto const T1_rd    = fields.T_slab_road->const_array(mfi);

        auto Tskin_rf = fields.T_skin_roof->array(mfi);
        auto Tskin_wl = fields.T_skin_wall->array(mfi);
        auto Tskin_rd = fields.T_skin_road->array(mfi);
        auto h_roof_a = fields.H_roof->array(mfi);
        auto h_wall_a = fields.H_wall->array(mfi);
        auto h_road_a = fields.H_road->array(mfi);

        auto diag_rf = newton_diag_roof.array(mfi);
        auto diag_wl = newton_diag_wall.array(mfi);
        auto diag_rd = newton_diag_road.array(mfi);

        auto const Fww_a = fields.F_wall_wall->const_array(mfi);
        auto const Fwr_a = fields.F_wall_road->const_array(mfi);
        auto const Frw_a = fields.F_road_wall->const_array(mfi);

        auto const Fws_a = fields.F_wall_sky->const_array(mfi);
        auto const Frs_a = fields.F_road_sky->const_array(mfi);

        auto const T_wall_prev_a = fields.T_skin_wall->const_array(mfi);
        auto const T_road_prev_a = fields.T_skin_road->const_array(mfi);
        auto const eps_wl_a = fields.emissivity_wall->const_array(mfi);
        auto const eps_rd_a = fields.emissivity_road->const_array(mfi);

        const bool radiosity_mode_is_multi = (m_params.radiosity_mode == RadiosityMode::Multi);
        const bool lw_radiosity_mode_is_multi = (m_params.lw_radiosity_mode == LWRadiosityMode::MultiLagged);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) noexcept {
            if (is_urb(i,j,0) == 0) return;

            amrex::Real U = amrex::max(std::sqrt(U_a(i,j,0)*U_a(i,j,0)), 0.01);
            amrex::Real T_can = T_can_a(i,j,0);

            amrex::Real SW_roof = SW_down;
            amrex::Real SW_wall, SW_road;
            const amrex::Real SW0_wall = SW_down * svf_wall(i,j,0);
            const amrex::Real SW0_road = SW_down * svf_road(i,j,0);
            if (radiosity_mode_is_multi) {
                sw_radiosity_multi_bounce(SW0_wall, SW0_road,
                                          alb_wl(i,j,0), alb_rd(i,j,0),
                                          Fww_a(i,j,0), Fwr_a(i,j,0), Frw_a(i,j,0),
                                          SW_wall, SW_road);
            } else {
                SW_wall = SW0_wall;
                SW_road = SW0_road;
            }

            amrex::Real LW_wall_in, LW_road_in;
            const amrex::Real LW_roof_in = LW_down;

            if (lw_radiosity_mode_is_multi) {
                const amrex::Real T_sky = std::pow(LW_down / UCM_SIGMA_SB, amrex::Real(0.25));

                lw_radiosity_multi_bounce(
                    T_wall_prev_a(i,j,0), T_road_prev_a(i,j,0), T_sky,
                    eps_wl_a(i,j,0), eps_rd_a(i,j,0),
                    Fww_a(i,j,0), Fwr_a(i,j,0), Frw_a(i,j,0),
                    Fws_a(i,j,0), Frs_a(i,j,0),
                    LW_wall_in, LW_road_in
                );
                LW_wall_in += (1.0 - Fws_a(i,j,0)) * UCM_SIGMA_SB * std::pow(T_wall_prev_a(i,j,0), 4);
                LW_road_in += (1.0 - Frs_a(i,j,0)) * UCM_SIGMA_SB * std::pow(T_road_prev_a(i,j,0), 4);

            } else {
                constexpr amrex::Real sigma_sb_local = 5.670374419e-8;
                const amrex::Real T_can_val = T_can_a(i,j,0);
                const amrex::Real T_can4 = T_can_val*T_can_val*T_can_val*T_can_val;
                const amrex::Real T_skin_prev_wl = Tskin_wl(i,j,0);
                const amrex::Real T_skin_prev_rd = Tskin_rd(i,j,0);
                const amrex::Real T_skin4_wl = T_skin_prev_wl*T_skin_prev_wl*T_skin_prev_wl*T_skin_prev_wl;
                const amrex::Real T_skin4_rd = T_skin_prev_rd*T_skin_prev_rd*T_skin_prev_rd*T_skin_prev_rd;

                const amrex::Real svf_w = svf_wall(i,j,0);
                const amrex::Real svf_r = svf_road(i,j,0);

                LW_wall_in = svf_w * LW_down
                           + (1.0 - svf_w) * sigma_sb_local * T_can4
                           + (1.0 - svf_w) * sigma_sb_local * T_skin4_wl;
                LW_road_in = svf_r * LW_down
                           + (1.0 - svf_r) * sigma_sb_local * T_can4
                           + (1.0 - svf_r) * sigma_sb_local * T_skin4_rd;
            }

            const amrex::Real LW_roof_eff = LW_roof_in;
            const amrex::Real LW_wall_eff = LW_wall_in;
            const amrex::Real LW_road_eff = LW_road_in;

            amrex::Real H_rf, H_wl, H_rd;

            // ROOF
            {
                amrex::Real T_unclamped, residual, SW_abs, LW_abs, H_sens, G_cond;
                int n_iter;
                solve_facet_seb_with_diag(
                    Tskin_rf(i,j,0), T1_rf(i,j,0), T_can,
                    SW_roof, LW_roof_eff,
                    alb_rf(i,j,0), eps_rf(i,j,0),
                    k_rf(i,j,0), dz_slab,
                    Ch_roof, U, rho_cp, max_iter, tol_K,
                    Tskin_rf(i,j,0), H_rf,
                    T_unclamped, n_iter, residual,
                    SW_abs, LW_abs, H_sens, G_cond);

                diag_rf(i,j,0,0) = Tskin_rf(i,j,0);
                diag_rf(i,j,0,1) = T_unclamped;
                diag_rf(i,j,0,2) = residual;
                diag_rf(i,j,0,3) = static_cast<amrex::Real>(n_iter);
                diag_rf(i,j,0,4) = SW_abs;
                diag_rf(i,j,0,5) = LW_abs;
                diag_rf(i,j,0,6) = H_sens;
                diag_rf(i,j,0,7) = G_cond;

                constexpr amrex::Real T_min_K = 260.0;
                constexpr amrex::Real T_clamp_tol = 0.01;
                if (std::abs(Tskin_rf(i,j,0) - T_min_K) < T_clamp_tol && T_unclamped < T_min_K) {
                    amrex::Gpu::Atomic::AddNoRet(p_clamped_roof, amrex::Long(1));
                }
                if (n_iter >= max_iter && residual > tol_K) {
                    amrex::Gpu::Atomic::AddNoRet(p_diverged_roof, amrex::Long(1));
                }
            }

            // WALL
            {
                amrex::Real T_unclamped, residual, SW_abs, LW_abs, H_sens, G_cond;
                int n_iter;
                solve_facet_seb_with_diag(
                    Tskin_wl(i,j,0), T1_wl(i,j,0), T_can,
                    SW_wall, LW_wall_eff,
                    alb_wl(i,j,0), eps_wl(i,j,0),
                    k_wl(i,j,0), dz_slab,
                    Ch_wall, U, rho_cp, max_iter, tol_K,
                    Tskin_wl(i,j,0), H_wl,
                    T_unclamped, n_iter, residual,
                    SW_abs, LW_abs, H_sens, G_cond);

                diag_wl(i,j,0,0) = Tskin_wl(i,j,0);
                diag_wl(i,j,0,1) = T_unclamped;
                diag_wl(i,j,0,2) = residual;
                diag_wl(i,j,0,3) = static_cast<amrex::Real>(n_iter);
                diag_wl(i,j,0,4) = SW_abs;
                diag_wl(i,j,0,5) = LW_abs;
                diag_wl(i,j,0,6) = H_sens;
                diag_wl(i,j,0,7) = G_cond;

                constexpr amrex::Real T_min_K = 260.0;
                constexpr amrex::Real T_clamp_tol = 0.01;
                if (std::abs(Tskin_wl(i,j,0) - T_min_K) < T_clamp_tol && T_unclamped < T_min_K) {
                    amrex::Gpu::Atomic::AddNoRet(p_clamped_wall, amrex::Long(1));
                }
                if (n_iter >= max_iter && residual > tol_K) {
                    amrex::Gpu::Atomic::AddNoRet(p_diverged_wall, amrex::Long(1));
                }
            }

            // ROAD
            {
                amrex::Real T_unclamped, residual, SW_abs, LW_abs, H_sens, G_cond;
                int n_iter;
                solve_facet_seb_with_diag(
                    Tskin_rd(i,j,0), T1_rd(i,j,0), T_can,
                    SW_road, LW_road_eff,
                    alb_rd(i,j,0), eps_rd(i,j,0),
                    k_rd(i,j,0), dz_slab,
                    Ch_road, U, rho_cp, max_iter, tol_K,
                    Tskin_rd(i,j,0), H_rd,
                    T_unclamped, n_iter, residual,
                    SW_abs, LW_abs, H_sens, G_cond);

                diag_rd(i,j,0,0) = Tskin_rd(i,j,0);
                diag_rd(i,j,0,1) = T_unclamped;
                diag_rd(i,j,0,2) = residual;
                diag_rd(i,j,0,3) = static_cast<amrex::Real>(n_iter);
                diag_rd(i,j,0,4) = SW_abs;
                diag_rd(i,j,0,5) = LW_abs;
                diag_rd(i,j,0,6) = H_sens;
                diag_rd(i,j,0,7) = G_cond;

                constexpr amrex::Real T_min_K = 260.0;
                constexpr amrex::Real T_clamp_tol = 0.01;
                if (std::abs(Tskin_rd(i,j,0) - T_min_K) < T_clamp_tol && T_unclamped < T_min_K) {
                    amrex::Gpu::Atomic::AddNoRet(p_clamped_road, amrex::Long(1));
                }
                if (n_iter >= max_iter && residual > tol_K) {
                    amrex::Gpu::Atomic::AddNoRet(p_diverged_road, amrex::Long(1));
                }
            }

            h_roof_a(i,j,0) = H_rf;
            h_wall_a(i,j,0) = H_wl;
            h_road_a(i,j,0) = H_rd;
        });
    }

    // Phase 3.5a-hotfix: MPI reduction and diagnostic output
    {
        amrex::ParallelDescriptor::ReduceLongSum(&n_clamped_roof, 1);
        amrex::ParallelDescriptor::ReduceLongSum(&n_clamped_wall, 1);
        amrex::ParallelDescriptor::ReduceLongSum(&n_clamped_road, 1);
        amrex::ParallelDescriptor::ReduceLongSum(&n_diverged_roof, 1);
        amrex::ParallelDescriptor::ReduceLongSum(&n_diverged_wall, 1);
        amrex::ParallelDescriptor::ReduceLongSum(&n_diverged_road, 1);

        if (m_params.ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A-hotfix][clamp-count] time=" << time
                           << "\n  Clamped to T_skin_min=260K:  roof=" << n_clamped_roof
                           << "  wall=" << n_clamped_wall
                           << "  road=" << n_clamped_road
                           << "\n  Newton diverged (hit max_iter): roof=" << n_diverged_roof
                           << "  wall=" << n_diverged_wall
                           << "  road=" << n_diverged_road
                           << "\n";
        }

        if (m_params.ucm_debug && m_params.newton_trace_ncells > 0 &&
            (n_clamped_roof > 0 || n_clamped_wall > 0 || n_clamped_road > 0)) {
            int n_traced = 0;
            const int n_trace_max = m_params.newton_trace_ncells;
            constexpr amrex::Real T_min_K = 260.0;
            constexpr amrex::Real T_clamp_tol = 0.01;

            for (amrex::MFIter mfi(newton_diag_roof); mfi.isValid() && n_traced < n_trace_max; ++mfi) {
                auto const diag = newton_diag_roof.const_array(mfi);
                auto const is_urb = fields.is_urban->const_array(mfi);
                const amrex::Box& bx = mfi.validbox();
                for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                    for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                        if (is_urb(i,j,0) > 0 && std::abs(diag(i,j,0,0) - T_min_K) < T_clamp_tol &&
                            diag(i,j,0,1) < T_min_K) {
                            if (amrex::ParallelDescriptor::IOProcessor()) {
                                amrex::Print() << "[UCM][3.5A-hotfix][clamp-trace] ROOF cell (i=" << i << ", j=" << j << ")"
                                               << "\n  T_initial=" << diag(i,j,0,1)
                                               << "  T_final=" << diag(i,j,0,0)
                                               << "  residual=" << diag(i,j,0,2)
                                               << "  n_iter=" << static_cast<int>(diag(i,j,0,3))
                                               << "\n  SW_abs=" << diag(i,j,0,4) << " W/m2"
                                               << "  LW_abs=" << diag(i,j,0,5) << " W/m2"
                                               << "  H_sens=" << diag(i,j,0,6) << " W/m2"
                                               << "  G_cond=" << diag(i,j,0,7) << " W/m2"
                                               << "\n  Balance: SW+LW-H-G = "
                                               << (diag(i,j,0,4) + diag(i,j,0,5) - diag(i,j,0,6) - diag(i,j,0,7))
                                               << " W/m2 (should be ~0 for converged, negative = losing heat)"
                                               << "\n";
                            }
                            n_traced++;
                            if (n_traced >= n_trace_max) break;
                        }
                    }
                    if (n_traced >= n_trace_max) break;
                }
            }

            for (amrex::MFIter mfi(newton_diag_wall); mfi.isValid() && n_traced < n_trace_max; ++mfi) {
                auto const diag = newton_diag_wall.const_array(mfi);
                auto const is_urb = fields.is_urban->const_array(mfi);
                const amrex::Box& bx = mfi.validbox();
                for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                    for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                        if (is_urb(i,j,0) > 0 && std::abs(diag(i,j,0,0) - T_min_K) < T_clamp_tol &&
                            diag(i,j,0,1) < T_min_K) {
                            if (amrex::ParallelDescriptor::IOProcessor()) {
                                amrex::Print() << "[UCM][3.5A-hotfix][clamp-trace] WALL cell (i=" << i << ", j=" << j << ")"
                                               << "\n  T_initial=" << diag(i,j,0,1)
                                               << "  T_final=" << diag(i,j,0,0)
                                               << "  residual=" << diag(i,j,0,2)
                                               << "  n_iter=" << static_cast<int>(diag(i,j,0,3))
                                               << "\n  SW_abs=" << diag(i,j,0,4) << " W/m2"
                                               << "  LW_abs=" << diag(i,j,0,5) << " W/m2"
                                               << "  H_sens=" << diag(i,j,0,6) << " W/m2"
                                               << "  G_cond=" << diag(i,j,0,7) << " W/m2"
                                               << "\n  Balance: SW+LW-H-G = "
                                               << (diag(i,j,0,4) + diag(i,j,0,5) - diag(i,j,0,6) - diag(i,j,0,7))
                                               << " W/m2 (should be ~0 for converged, negative = losing heat)"
                                               << "\n";
                            }
                            n_traced++;
                            if (n_traced >= n_trace_max) break;
                        }
                    }
                    if (n_traced >= n_trace_max) break;
                }
            }

            for (amrex::MFIter mfi(newton_diag_road); mfi.isValid() && n_traced < n_trace_max; ++mfi) {
                auto const diag = newton_diag_road.const_array(mfi);
                auto const is_urb = fields.is_urban->const_array(mfi);
                const amrex::Box& bx = mfi.validbox();
                for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                    for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                        if (is_urb(i,j,0) > 0 && std::abs(diag(i,j,0,0) - T_min_K) < T_clamp_tol &&
                            diag(i,j,0,1) < T_min_K) {
                            if (amrex::ParallelDescriptor::IOProcessor()) {
                                amrex::Print() << "[UCM][3.5A-hotfix][clamp-trace] ROAD cell (i=" << i << ", j=" << j << ")"
                                               << "\n  T_initial=" << diag(i,j,0,1)
                                               << "  T_final=" << diag(i,j,0,0)
                                               << "  residual=" << diag(i,j,0,2)
                                               << "  n_iter=" << static_cast<int>(diag(i,j,0,3))
                                               << "\n  SW_abs=" << diag(i,j,0,4) << " W/m2"
                                               << "  LW_abs=" << diag(i,j,0,5) << " W/m2"
                                               << "  H_sens=" << diag(i,j,0,6) << " W/m2"
                                               << "  G_cond=" << diag(i,j,0,7) << " W/m2"
                                               << "\n  Balance: SW+LW-H-G = "
                                               << (diag(i,j,0,4) + diag(i,j,0,5) - diag(i,j,0,6) - diag(i,j,0,7))
                                               << " W/m2 (should be ~0 for converged, negative = losing heat)"
                                               << "\n";
                            }
                            n_traced++;
                            if (n_traced >= n_trace_max) break;
                        }
                    }
                    if (n_traced >= n_trace_max) break;
                }
            }
        }
    }

    // Phase 5.1b: Per-step debug banner for radiosity
    amrex::Real Fwr_min = 0.0, Fwr_max = 0.0;
    if (m_params.ucm_debug && m_params.radiosity_mode == RadiosityMode::Multi) {
        Fwr_min = fields.F_wall_road->min(0, 0);
        Fwr_max = fields.F_wall_road->max(0, 0);
    }
    if (m_params.ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
        const char* mode_str = (m_params.radiosity_mode == RadiosityMode::Multi) ? "multi" : "single";
        amrex::Print() << "[UCM][5.1b][radiosity] mode=" << mode_str
                       << " alpha_wall=" << m_params.albedo_wall
                       << " alpha_road=" << m_params.albedo_road;
        if (m_params.radiosity_mode == RadiosityMode::Multi) {
            amrex::Print() << " F_wall_road=[" << Fwr_min << ", " << Fwr_max << "]";
        }
        amrex::Print() << "\n";
    }

    // Phase 5.1c: Per-step debug banner for LW radiosity
    amrex::Real Tsky_min = 0.0, Tsky_max = 0.0;
    if (m_params.ucm_debug && m_params.lw_radiosity_mode == LWRadiosityMode::MultiLagged) {
        const amrex::Real LW_min = forcing.LW_down->min(0, 0);
        const amrex::Real LW_max = forcing.LW_down->max(0, 0);
        Tsky_min = std::pow(LW_min / UCM_SIGMA_SB, amrex::Real(0.25));
        Tsky_max = std::pow(LW_max / UCM_SIGMA_SB, amrex::Real(0.25));
    }
    if (m_params.ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
        const char* mode_str = (m_params.lw_radiosity_mode == LWRadiosityMode::MultiLagged)
                               ? "multi-lagged" : "single";
        amrex::Print() << "[UCM][5.1c][lw-radiosity] mode=" << mode_str
                       << " eps_wall=" << m_params.emissivity_wall
                       << " eps_road=" << m_params.emissivity_road;
        if (m_params.lw_radiosity_mode == LWRadiosityMode::MultiLagged) {
            amrex::Print() << " T_sky=[" << Tsky_min << ", " << Tsky_max << "]";
        }
        amrex::Print() << "\n";
    }

    // Phase 3.5A: Advance slab conduction using SEB-derived H as surface BC

    for (amrex::MFIter mfi(*fields.T_slab_roof, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        advance_slab_conduction_mfi(
            fields.T_slab_roof->array(mfi),
            fields.H_roof->array(mfi),
            fields.k_therm_roof->array(mfi),
            fields.rho_cp_roof->array(mfi),
            fields.is_urban->array(mfi),
            bx, dt, N_layers, slab_L, T_deep);
    }

    for (amrex::MFIter mfi(*fields.T_slab_wall, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        advance_slab_conduction_mfi(
            fields.T_slab_wall->array(mfi),
            fields.H_wall->array(mfi),
            fields.k_therm_wall->array(mfi),
            fields.rho_cp_wall->array(mfi),
            fields.is_urban->array(mfi),
            bx, dt, N_layers, slab_L, T_deep);
    }

    for (amrex::MFIter mfi(*fields.T_slab_road, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        advance_slab_conduction_mfi(
            fields.T_slab_road->array(mfi),
            fields.H_road->array(mfi),
            fields.k_therm_road->array(mfi),
            fields.rho_cp_road->array(mfi),
            fields.is_urban->array(mfi),
            bx, dt, N_layers, slab_L, T_deep);
    }

    if (m_params.ucm_debug) {
        amrex::Real T1_roof_min = fields.T_slab_roof->min(0, 0);
        amrex::Real T1_roof_max = fields.T_slab_roof->max(0, 0);
        amrex::Real T1_wall_min = fields.T_slab_wall->min(0, 0);
        amrex::Real T1_wall_max = fields.T_slab_wall->max(0, 0);
        amrex::Real T1_road_min = fields.T_slab_road->min(0, 0);
        amrex::Real T1_road_max = fields.T_slab_road->max(0, 0);
        amrex::Real Hr_min = fields.H_roof->min(0, 0);
        amrex::Real Hr_max = fields.H_roof->max(0, 0);
        amrex::Real Hw_min = fields.H_wall->min(0, 0);
        amrex::Real Hw_max = fields.H_wall->max(0, 0);
        amrex::Real Hd_min = fields.H_road->min(0, 0);
        amrex::Real Hd_max = fields.H_road->max(0, 0);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A-diag][slab-top-AFTER-CONDUCTION] "
                           << "T_slab_roof[0]=[" << T1_roof_min << "," << T1_roof_max << "] K "
                           << "T_slab_wall[0]=[" << T1_wall_min << "," << T1_wall_max << "] K "
                           << "T_slab_road[0]=[" << T1_road_min << "," << T1_road_max << "] K\n"
                           << "[UCM][3.5A-diag][H-into-slab] "
                           << "H_roof=[" << Hr_min << "," << Hr_max << "] W/m2 "
                           << "H_wall=[" << Hw_min << "," << Hw_max << "] W/m2 "
                           << "H_road=[" << Hd_min << "," << Hd_max << "] W/m2\n";
        }
    }

    // Phase 3.5A: Update canyon-air temperature using Newton-computed H
    for (amrex::MFIter mfi(*fields.T_canyon_air, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto const is_urb   = fields.is_urban->const_array(mfi);
        auto const H_rd     = fields.H_road->const_array(mfi);
        auto const H_wl     = fields.H_wall->const_array(mfi);
        auto const H_bldg   = fields.H_bldg->const_array(mfi);
        auto const W_road   = fields.W_road->const_array(mfi);
        auto const T_atm    = forcing.T_atm_ref->const_array(mfi);
        auto T_canyon_a     = fields.T_canyon_air->array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) noexcept {
            if (is_urb(i,j,0) == 0) return;

            amrex::Real Hb  = H_bldg(i,j,0);
            amrex::Real W   = amrex::max(W_road(i,j,0), amrex::Real(1.0e-6));
            amrex::Real HoW = Hb / W;
            amrex::ignore_unused(HoW);

            constexpr amrex::Real rho_cp_c   = 1.2 * 1005.0;
            constexpr amrex::Real U_canyon    = 2.0;
            constexpr amrex::Real Ch_c        = 0.01;
            constexpr amrex::Real max_dT_per_step = 2.0;

            amrex::Real conductance = amrex::max(rho_cp_c * Ch_c * U_canyon,
                                                  amrex::Real(1.0e-6));

            amrex::Real H_canyon_depth = 0.5 * Hb;
            amrex::Real thermal_mass = rho_cp_c * amrex::max(H_canyon_depth, amrex::Real(1.0));

            amrex::Real H_net = H_rd(i,j,0) + H_wl(i,j,0)
                              - conductance * (T_canyon_a(i,j,0) - T_atm(i,j,0));

            amrex::Real dT = H_net * dt / thermal_mass;

            if (dT >  max_dT_per_step) dT =  max_dT_per_step;
            if (dT < -max_dT_per_step) dT = -max_dT_per_step;

            amrex::Real T_canyon_new = T_canyon_a(i,j,0) + dT;

            T_canyon_new = amrex::max(amrex::min(T_canyon_new, amrex::Real(380.0)),
                                       amrex::Real(200.0));
            T_canyon_a(i,j,0) = T_canyon_new;
        });
    }

    // ========================================================================
    // Phase 5.2: HVAC waste heat (Contract #21 / #22)
    //
    // Scalar-hoisted single-profile lookup — no std::vector or std::string
    // captured into the ParallelFor lambda (Contract #22, GPU safety).
    //
    // Phase 5.2-hotfix: All AMReX collective reductions (min/max) called
    // OUTSIDE the IOProcessor() guard per PR #201 Bug #9 rule. Guarding a
    // collective behind IOProcessor causes deadlock (rank 0 executes the
    // collective; rank 1 skips it; subsequent collectives desynchronize).
    //
    // TODO (Phase 5.2 followup): cache HVAC/occupancy CSVs at UCMLayer
    // construction time rather than re-reading every timestep.
    // ========================================================================
    const bool hvac_on = (m_params.hvac_mode == HVACMode::Simple);

    if (hvac_on) {
        const amrex::Real hyst_K = m_params.hvac_hysteresis_K;
        const amrex::Real cop_default = m_params.hvac_cop_default;
        const amrex::Real setpt_default = m_params.hvac_setpoint_default_K;

        const amrex::Real solar_time_local = m_params.solar_time_start_s + time;
        const int hour_of_day = static_cast<int>(std::fmod(solar_time_local / 3600.0, 24.0));

        // Load CSVs (once per call; TODO: cache at UCMLayer construction)
        UCMHVACReader hvac_reader(m_params.hvac_csv_path);
        UCMOccupancyReader occ_reader(m_params.occupancy_csv_path);
        const auto& hvac_profiles = hvac_reader.get_all_profiles();
        const auto& occ_profiles = occ_reader.get_all_profiles();

        // Contract #22: NO std::vector, std::string, or heap-allocated STL
        // containers may be captured by value into an amrex::ParallelFor lambda.
        // Pre-resolve profile 0's values on HOST as scalars.
        // Multi-profile per-cell dispatch is deferred to Phase 6.2b (BEM-lite)
        // via amrex::Gpu::DeviceVector.
        amrex::Real T_setpt_resolved = setpt_default;
        amrex::Real cop_resolved = cop_default;
        int occ_id_resolved = 0;
        if (hvac_profiles.size() > 0) {
            T_setpt_resolved = hvac_profiles[0].t_setpoint_K;
            cop_resolved = hvac_profiles[0].cop;
            occ_id_resolved = hvac_profiles[0].occupancy_profile_id;
        }

        amrex::Real f_occ_resolved = 1.0;
        for (const auto& p : occ_profiles) {
            if (p.id == occ_id_resolved) {
                if (hour_of_day >= 0 && hour_of_day < 24) {
                    f_occ_resolved = p.hourly_frac[hour_of_day];
                }
                break;
            }
        }

        // Phase 5.2-hotfix: sync ranks after file I/O to prevent step-2
        // collective deadlock from rank drift.
        amrex::ParallelDescriptor::Barrier();

        if (m_params.ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][5.2-diag-resolved] T_setpt=" << T_setpt_resolved
                           << " cop=" << cop_resolved
                           << " occ_id=" << occ_id_resolved
                           << " f_occ=" << f_occ_resolved
                           << " hour=" << hour_of_day << "\n";
        }

        for (amrex::MFIter mfi(*fields.T_canyon_air, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto const H_wl_a = fields.H_wall->const_array(mfi);
            auto const H_rf_a = fields.H_roof->const_array(mfi);
            auto const T_can_a = fields.T_canyon_air->const_array(mfi);
            auto const is_urb_a = fields.is_urban->const_array(mfi);
            auto AH_a = fields.AH->array(mfi);
            auto Q_diag_a = fields.Q_HVAC_diag->array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) noexcept {
                Q_diag_a(i,j,0) = 0.0;
                if (is_urb_a(i,j,0) == 0) return;

                const amrex::Real Q_HVAC = compute_hvac_waste_heat(
                    H_wl_a(i,j,0), H_rf_a(i,j,0),
                    T_can_a(i,j,0), T_setpt_resolved,
                    hyst_K, cop_resolved, f_occ_resolved);

                AH_a(i,j,0) += Q_HVAC;
                Q_diag_a(i,j,0) = Q_HVAC;
            });
        }

        // Phase 5.2-hotfix: collectives OUTSIDE IOProcessor guard (PR #201 Bug #9)
        if (m_params.ucm_debug) {
            amrex::Real Q_HVAC_min = fields.Q_HVAC_diag->min(0, 0);
            amrex::Real Q_HVAC_max = fields.Q_HVAC_diag->max(0, 0);
            if (amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "[UCM][5.2][hvac] mode=simple hour=" << hour_of_day
                               << " Q_HVAC=[" << Q_HVAC_min << ", " << Q_HVAC_max << "] W/m²\n";
            }
        }
    }

    // Phase 2.3: Compute anthropogenic heat
    compute_anthropogenic_heat(*fields.AH, *fields.ah_profile_id, *fields.is_urban,
                              *fields.AH_Wm2_ucm,
                              m_params, time, lev);

    // ------------------------------------------------------------------------
    // Phase 2.3 facet-split sensible heat flux (Phase 2.3.1 physics fix)
    // ------------------------------------------------------------------------

    const amrex::Real Cp = Cp_d;
    const amrex::Real rho_ref = 1.2;
    const amrex::Real zref = 2.0;

    for (amrex::MFIter mfi(*forcing.u_star, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi)
    {
       const amrex::Box& bx = mfi.tilebox();

       auto const plan_a   = fields.plan_area_frac->const_array(mfi);
       auto const Hbldg_a  = fields.H_bldg->const_array(mfi);
       auto const Wrd_a    = fields.W_road->const_array(mfi);
       auto const Wrf_a    = fields.W_roof->const_array(mfi);
       auto const ah_a     = fields.AH->const_array(mfi);
       auto const is_urb_a = fields.is_urban->const_array(mfi);
       auto const u_star_a = forcing.u_star->const_array(mfi);
       auto const t_star_a = atm_t_star.const_array(mfi);
       auto const olen_a   = atm_olen.const_array(mfi);

       auto       h_road_atm_a = fields.H_road_atm->array(mfi);
       auto       h_wall_atm_a = fields.H_wall_atm->array(mfi);
       auto       h_roof_atm_a = fields.H_roof_atm->array(mfi);
       auto       h_sens_a = fields.H_sensible->array(mfi);

       const bool use_stab_corr = m_params.use_stability_correction;
       const amrex::Real zeta_max_stable = m_params.zeta_max_stable;
       const amrex::Real zeta_min_unstable = m_params.zeta_min_unstable;

       amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) noexcept {
           if (is_urb_a(i,j,0) == 0) {
               h_road_atm_a(i,j,0) = 0.0;
               h_wall_atm_a(i,j,0) = 0.0;
               h_roof_atm_a(i,j,0) = 0.0;
               h_sens_a(i,j,0) = 0.0;
               return;
           }

           const amrex::Real pf   = plan_a(i,j,0);
           const amrex::Real Hb   = Hbldg_a(i,j,0);
           const amrex::Real Wsum = amrex::max(Wrd_a(i,j,0) + Wrf_a(i,j,0), 1.0e-6);

           const amrex::Real f_road = 1.0 - pf;
           const amrex::Real f_roof = pf;

           const amrex::Real u_star = u_star_a(i,j,0);
           const amrex::Real t_star = t_star_a(i,j,0);
           amrex::Real H_base = rho_ref * Cp * u_star * t_star;
           const amrex::Real AH_val = ah_a(i,j,0);

           if (use_stab_corr) {
               const amrex::Real olen = olen_a(i,j,0);
               H_base = compute_ch_stability_correction(H_base, olen, zref,
                                                        zeta_max_stable, zeta_min_unstable);
           }

           amrex::Real Hr = f_road * H_base;
           amrex::Real Hw = 2.0 * pf * Hb / Wsum * H_base;
           amrex::Real Hf = f_roof * H_base;

           if (!amrex::Math::isfinite(Hr)) Hr = 0.0;
           if (!amrex::Math::isfinite(Hw)) Hw = 0.0;
           if (!amrex::Math::isfinite(Hf)) Hf = 0.0;

           Hr = amrex::max(-1500.0, amrex::min(1500.0, Hr));
           Hw = amrex::max(-1500.0, amrex::min(1500.0, Hw));
           Hf = amrex::max(-1500.0, amrex::min(1500.0, Hf));

           Hf += AH_val;

           h_road_atm_a(i,j,0) = Hr;
           h_wall_atm_a(i,j,0) = Hw;
           h_roof_atm_a(i,j,0) = Hf;

           const amrex::Real H_lumped = Hr + Hw + Hf;

           h_sens_a(i,j,0) = H_lumped;
       });
    }

    fields.LE_latent->setVal(0.0);

    // ========================================================================
    // Step 4: Debug trace
    // ========================================================================

    if (m_params.ucm_debug) {
        amrex::Real T_roof_min  = fields.T_skin_roof->min(0, 0);
        amrex::Real T_roof_max  = fields.T_skin_roof->max(0, 0);
        amrex::Real T_wall_min  = fields.T_skin_wall->min(0, 0);
        amrex::Real T_wall_max  = fields.T_skin_wall->max(0, 0);
        amrex::Real T_road_min  = fields.T_skin_road->min(0, 0);
        amrex::Real T_road_max  = fields.T_skin_road->max(0, 0);
        amrex::Real T_can_min   = fields.T_canyon_air->min(0, 0);
        amrex::Real T_can_max   = fields.T_canyon_air->max(0, 0);
        amrex::Real H_sens_min  = fields.H_sensible->min(0, 0);
        amrex::Real H_sens_max  = fields.H_sensible->max(0, 0);
        amrex::Real H_roof_min  = fields.H_roof->min(0, 0);
        amrex::Real H_roof_max  = fields.H_roof->max(0, 0);
        amrex::Real H_road_min  = fields.H_road->min(0, 0);
        amrex::Real H_road_max  = fields.H_road->max(0, 0);
        amrex::Real H_wall_min  = fields.H_wall->min(0, 0);
        amrex::Real H_wall_max  = fields.H_wall->max(0, 0);
        amrex::Real AH_min      = fields.AH->min(0, 0);
        amrex::Real AH_max      = fields.AH->max(0, 0);

        amrex::Real H_roof_atm_min = fields.H_roof_atm->min(0, 0);
        amrex::Real H_roof_atm_max = fields.H_roof_atm->max(0, 0);
        amrex::Real H_road_atm_min = fields.H_road_atm->min(0, 0);
        amrex::Real H_road_atm_max = fields.H_road_atm->max(0, 0);
        amrex::Real H_wall_atm_min = fields.H_wall_atm->min(0, 0);
        amrex::Real H_wall_atm_max = fields.H_wall_atm->max(0, 0);

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A][SEB] dt=" << dt << "s, sim_time=" << time << "s\n";
            amrex::Print() << "  T_skin_roof=[" << T_roof_min << "," << T_roof_max << "] K\n";
            amrex::Print() << "  T_skin_wall=[" << T_wall_min << "," << T_wall_max << "] K\n";
            amrex::Print() << "  T_skin_road=[" << T_road_min << "," << T_road_max << "] K\n";
            amrex::Print() << "  T_canyon_air=[" << T_can_min << "," << T_can_max << "] K\n";
            amrex::Print() << "  H_roof min=" << H_roof_min << " max=" << H_roof_max << " W/m2\n";
            amrex::Print() << "  H_wall min=" << H_wall_min << " max=" << H_wall_max << " W/m2\n";
            amrex::Print() << "  H_road min=" << H_road_min << " max=" << H_road_max << " W/m2\n";
            amrex::Print() << "  AH min=" << AH_min << " max=" << AH_max << " W/m2\n";
            amrex::Print() << "  H_sensible min=" << H_sens_min << " max=" << H_sens_max << " W/m2"
                          << " (= H_road+H_wall+H_roof; AH already in H_roof)\n";
        }

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A-hotfix2][consistency] time=" << time << " s\n"
                          << "  Newton  H_roof min=" << H_roof_min << " max=" << H_roof_max << " W/m2  (drives slab conduction)\n"
                          << "  MOST    H_roof min=" << H_roof_atm_min << " max=" << H_roof_atm_max << " W/m2  (drives ATM injection)\n"
                          << "  Newton  H_road min=" << H_road_min << " max=" << H_road_max << " W/m2\n"
                          << "  MOST    H_road min=" << H_road_atm_min << " max=" << H_road_atm_max << " W/m2\n"
                          << "  Newton  H_wall min=" << H_wall_min << " max=" << H_wall_max << " W/m2\n"
                          << "  MOST    H_wall min=" << H_wall_atm_min << " max=" << H_wall_atm_max << " W/m2\n";
        }

        amrex::Real tat_min = T_atm_lowest.min(0, 0), tat_max = T_atm_lowest.max(0, 0);
        amrex::Real uat_min = xvel.min(0, 0), uat_max = xvel.max(0, 0);
        amrex::Real vat_min = yvel.min(0, 0), vat_max = yvel.max(0, 0);

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.2][SEB-inputs] step: "
                          << "T_atm_ucm min=" << tat_min << " max=" << tat_max << " K\n"
                          << "        U_atm_ucm min=" << uat_min << " max=" << uat_max << " m/s\n"
                          << "        V_atm_ucm min=" << vat_min << " max=" << vat_max << " m/s\n"
                          << "        H_road min=" << H_road_min << " max=" << H_road_max << " W/m2\n"
                          << "        H_wall min=" << H_wall_min << " max=" << H_wall_max << " W/m2\n"
                          << "        H_roof min=" << H_roof_min << " max=" << H_roof_max << " W/m2\n"
                          << "        H_sensible min=" << H_sens_min << " max=" << H_sens_max << " W/m2\n";
        }
    }

    static bool pblh_guard_printed = false;
    if (!pblh_guard_printed && m_params.ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
        pblh_guard_printed = true;
        amrex::Print() << "[UCM][3.3][pblh-guard] PBLH dependency check: CLEAN\n"
                       << "  Stability inputs: u_star, t_star, q_star only. No PBLH consumed.\n";
    }
}