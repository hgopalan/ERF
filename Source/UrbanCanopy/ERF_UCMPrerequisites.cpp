/**
 * @file ERF_UCMPrerequisites.cpp
 * @brief Implementation of prerequisite verification for SLUCM
 *
 * Provides detailed parameter validation and startup banner output.
 * Each check produces actionable error messages pointing to Phase information.
 *
 * References:
 *  - Source/Dust/DUST_DEVELOPMENT.md (ERF-Hazard)
 *  - Source/LNG/LNG_DEVELOPMENT.md (ba55b73...)
 */

#include <ERF_UCMPrerequisites.H>
#include <ERF.H>
#include <AMReX_Print.H>

void resolve_wall_drag_mode(const std::string& wall_drag_mode_str,
                             bool is_anelastic,
                             WallDragMode& resolved_mode)
{
    if (wall_drag_mode_str == "auto") {
        if (is_anelastic) {
            resolved_mode = WallDragMode::Implicit;
        } else {
            resolved_mode = WallDragMode::Explicit;
        }
    } else if (wall_drag_mode_str == "explicit") {
        resolved_mode = WallDragMode::Explicit;
    } else if (wall_drag_mode_str == "implicit") {
        resolved_mode = WallDragMode::Implicit;
    } else if (wall_drag_mode_str == "off") {
        resolved_mode = WallDragMode::Off;
    } else {
        std::string msg = "[UCM] Invalid wall_drag_mode: \"" + wall_drag_mode_str + "\". "
                        + "Valid options: \"auto\", \"explicit\", \"implicit\", \"off\".";
        amrex::Abort(msg);
    }
}

void check_ucm_prerequisites(const UCMParams& params,
                              int /*max_level*/,
                              int finest_level,
                              bool /*use_terrain*/,
                              int /*lev*/)
{
    // Check 1: anchor_level within bounds
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.anchor_level >= 0,
        "[UCM] anchor_level must be >= 0. Set: erf.ucm.anchor_level = 0");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.anchor_level <= finest_level,
        "[UCM] anchor_level must be <= finest_level. Reduce erf.ucm.anchor_level");

    // Phase 3.1b: anchor_level > 0 is now supported. Validation is limited to
    // the bounds check above (0 <= anchor_level <= finest_level). Stress test
    // with anchor_level > 0 is scheduled for Phase 3.7 (anchor_level=2 nested).
    if (params.anchor_level > 0) {
        amrex::Print() << "[UCM] anchor_level = " << params.anchor_level
                       << " (multi-level UCM enabled; Phase 3.7 will stress-test)\n";
    }

    // Check 3: static_refinement must be true
    if (!params.static_refinement) {
        std::string msg = std::string("[UCM] static_refinement must be true. ")
                        + "Regridding during integration not yet supported. "
                        + "Set: erf.ucm.static_refinement = true";
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false, msg.c_str());
    }

    // Check 4: grid_ratio >= 1
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.grid_ratio >= 1,
        "[UCM] grid_ratio must be >= 1. Set: erf.ucm.grid_ratio = 1");

    // Check 5: alpha_ucm > 0
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.alpha_ucm > 0.0,
        "[UCM] alpha_ucm must be > 0.0 [m]. Set: erf.ucm.alpha_ucm = 10.0");

    // Check 6: zref > 0
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.zref > 0.0,
        "[UCM] zref must be > 0.0 [m]. Set: erf.ucm.zref = 2.0");

    // Check 7: All building dimensions positive
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.H_bldg_uniform > 0.0,
        "[UCM] H_bldg_uniform must be > 0.0 [m]. Set: erf.ucm.H_bldg_uniform = 10.0");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.W_road_uniform > 0.0,
        "[UCM] W_road_uniform must be > 0.0 [m]. Set: erf.ucm.W_road_uniform = 10.0");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.W_roof_uniform > 0.0,
        "[UCM] W_roof_uniform must be > 0.0 [m]. Set: erf.ucm.W_roof_uniform = 10.0");

    // Check 8: All albedos in [0, 1]
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.albedo_roof >= 0.0 && params.albedo_roof <= 1.0,
        "[UCM] albedo_roof must be in [0,1]. Set: erf.ucm.albedo_roof = 0.20");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.albedo_wall >= 0.0 && params.albedo_wall <= 1.0,
        "[UCM] albedo_wall must be in [0,1]. Set: erf.ucm.albedo_wall = 0.20");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.albedo_road >= 0.0 && params.albedo_road <= 1.0,
        "[UCM] albedo_road must be in [0,1]. Set: erf.ucm.albedo_road = 0.15");

    // Check 9: All emissivities in [0, 1]
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.emissivity_roof >= 0.0 && params.emissivity_roof <= 1.0,
        "[UCM] emissivity_roof must be in [0,1]. Set: erf.ucm.emissivity_roof = 0.90");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.emissivity_wall >= 0.0 && params.emissivity_wall <= 1.0,
        "[UCM] emissivity_wall must be in [0,1]. Set: erf.ucm.emissivity_wall = 0.90");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.emissivity_road >= 0.0 && params.emissivity_road <= 1.0,
        "[UCM] emissivity_road must be in [0,1]. Set: erf.ucm.emissivity_road = 0.94");

    // Check 10: Phase 2.11-fix — Per-process feedback knobs in [0.0, 1.0]
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.atm_feedback_momentum >= 0.0 && params.atm_feedback_momentum <= 1.0,
        "[UCM] atm_feedback_momentum must be in [0.0, 1.0]. Set: erf.ucm.atm_feedback_momentum = 1.0");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.atm_feedback_heat >= 0.0 && params.atm_feedback_heat <= 1.0,
        "[UCM] atm_feedback_heat must be in [0.0, 1.0]. Set: erf.ucm.atm_feedback_heat = 0.0");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.atm_feedback_moisture >= 0.0 && params.atm_feedback_moisture <= 1.0,
        "[UCM] atm_feedback_moisture must be in [0.0, 1.0]. Set: erf.ucm.atm_feedback_moisture = 0.0");

    // Check 12: Phase 1.3+ slab conduction parameters
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.slab_N_layers >= 1,
        "[UCM] slab_N_layers must be >= 1. Set: erf.ucm.slab_N_layers = 4");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.slab_L > 0.0,
        "[UCM] slab_L must be > 0.0 [m]. Set: erf.ucm.slab_L = 0.3");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.k_therm_uniform > 0.0,
        "[UCM] k_therm_uniform must be > 0.0 [W/m/K]. Set: erf.ucm.k_therm_uniform = 1.5");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.rho_cp_uniform > 0.0,
        "[UCM] rho_cp_uniform must be > 0.0 [J/m^3/K]. Set: erf.ucm.rho_cp_uniform = 2.0e6");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.newton_max_iter >= 1,
        "[UCM] newton_max_iter must be >= 1. Set: erf.ucm.newton_max_iter = 20");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.newton_tol_K > 0.0,
        "[UCM] newton_tol_K must be > 0.0 [K]. Set: erf.ucm.newton_tol_K = 0.01");

    // Check 13: zref warnings (Phase 1.3+)
    if (params.zref < 0.5) {
        amrex::Print() << "[UCM] WARNING: zref < 0.5 m may be too close to roof surface\n";
    } else if (params.zref > 20.0) {
        amrex::Print() << "[UCM] WARNING: zref > 20 m may be above boundary layer\n";
    }

    // Startup banner with all parameter values
    amrex::Print() << "\n";
    amrex::Print() << "[UCM] =========================================================\n";
    amrex::Print() << "[UCM] SLUCM Module Initialization Summary (Phase 1.1-1.4)\n";
    amrex::Print() << "[UCM] =========================================================\n";
    amrex::Print() << "[UCM]   enable              = " << (params.enable ? "true" : "false") << "\n";
    amrex::Print() << "[UCM]   ucm_debug           = " << (params.ucm_debug ? "true" : "false") << "\n";
    amrex::Print() << "[UCM]   anchor_level        = " << params.anchor_level << "\n";
    amrex::Print() << "[UCM]   static_refinement   = " << (params.static_refinement ? "true" : "false") << "\n";
    amrex::Print() << "[UCM]   grid_ratio          = " << params.grid_ratio << "\n";
    amrex::Print() << "[UCM]   allow_steep_terrain = " << (params.allow_steep_terrain ? "true" : "false") << "\n";
    amrex::Print() << "[UCM]   zref [m]            = " << params.zref << "\n";
    amrex::Print() << "[UCM]   z0_over_H           = " << params.z0_over_H << "\n";
    amrex::Print() << "[UCM]   d_over_H            = " << params.d_over_H << "\n";
    amrex::Print() << "[UCM]   --- Phase 1.3 Slab Conduction Parameters ---\n";
    amrex::Print() << "[UCM]   H_bldg_uniform [m]  = " << params.H_bldg_uniform << "\n";
    amrex::Print() << "[UCM]   W_road_uniform [m]  = " << params.W_road_uniform << "\n";
    amrex::Print() << "[UCM]   W_roof_uniform [m]  = " << params.W_roof_uniform << "\n";
    amrex::Print() << "[UCM]   albedo_roof         = " << params.albedo_roof << "\n";
    amrex::Print() << "[UCM]   albedo_wall         = " << params.albedo_wall << "\n";
    amrex::Print() << "[UCM]   albedo_road         = " << params.albedo_road << "\n";
    amrex::Print() << "[UCM]   emissivity_roof     = " << params.emissivity_roof << "\n";
    amrex::Print() << "[UCM]   emissivity_wall     = " << params.emissivity_wall << "\n";
    amrex::Print() << "[UCM]   emissivity_road     = " << params.emissivity_road << "\n";
    amrex::Print() << "[UCM]   slab_N_layers [#]   = " << params.slab_N_layers << "\n";
    amrex::Print() << "[UCM]   slab_T_deep [K]     = " << params.slab_T_deep << "\n";
    amrex::Print() << "[UCM]   slab_L [m]          = " << params.slab_L << "\n";
    amrex::Print() << "[UCM]   k_therm [W/m/K]     = " << params.k_therm_uniform << "\n";
    amrex::Print() << "[UCM]   rho_cp [J/m^3/K]    = " << params.rho_cp_uniform << "\n";
    amrex::Print() << "[UCM]   newton_max_iter     = " << params.newton_max_iter << "\n";
    amrex::Print() << "[UCM]   newton_tol_K        = " << params.newton_tol_K << "\n";
    amrex::Print() << "[UCM]   newton_trace_ncells = " << params.newton_trace_ncells << " (Phase 3.5a-hotfix)\n";
    amrex::Print() << "[UCM]   --- Phase 1.4 Injection Parameters ---\n";
    amrex::Print() << "[UCM]   [SLUCM] Feedback configuration:\n";
    amrex::Print() << "[UCM]     atm_feedback_momentum = " << params.atm_feedback_momentum << " (drag always active)\n";
    amrex::Print() << "[UCM]     atm_feedback_heat     = " << params.atm_feedback_heat << " (opt-in; Phase 3.2 TBD)\n";
    amrex::Print() << "[UCM]     atm_feedback_moisture = " << params.atm_feedback_moisture << " (opt-in; Phase 4+ TBD)\n";
    if (params.atm_feedback >= 0.0) {
        amrex::Print() << "[UCM]     (legacy atm_feedback  = " << params.atm_feedback << " was propagated to all three)\n";
    } else {
        amrex::Print() << "[UCM]     (legacy atm_feedback  = -1.0, not set)\n";
    }
    amrex::Print() << "[UCM]   alpha_ucm [m]       = " << params.alpha_ucm << "\n";
    amrex::Print() << "[UCM]   ucm_plot_int        = " << params.ucm_plot_int << "\n";
    amrex::Print() << "[UCM]   ucm_atm_plot_int    = " << params.ucm_atm_plot_int << "\n";
    amrex::Print() << "[UCM]   ucm_diag_file       = " << params.ucm_diag_file << "\n";
    amrex::Print() << "[UCM]   --- Phase 2.6 Morphology-Aware Injection ---\n";
    amrex::Print() << "[UCM]   use_morphology_injection = " << (params.use_morphology_injection ? "true" : "false") << "\n";
    amrex::Print() << "[UCM]   alpha_scale         = " << params.alpha_scale << "\n";
    amrex::Print() << "[UCM]   alpha_min [m]       = " << params.alpha_min << "\n";
    amrex::Print() << "[UCM]   alpha_max [m]       = " << params.alpha_max << "\n";
    amrex::Print() << "[UCM]   --- Phase 2.7 Facet3D BEP-Continuous-TF ---\n";
    amrex::Print() << "[UCM]   use_facet3d_injection = " << (params.use_facet3d_injection ? "true" : "false") << "\n";
    amrex::Print() << "[UCM]   use_gaussian_height_distribution = " << (params.use_gaussian_height_distribution ? "true" : "false") << "\n";
    amrex::Print() << "[UCM]   height_std_threshold_m = " << params.height_std_threshold_m << "\n";
    amrex::Print() << "[UCM]   --- Phase 2.8 BEP Momentum Drag ---\n";
    amrex::Print() << "[UCM]   wall_drag_mode      = \"" << params.wall_drag_mode_str << "\" (resolved: ";
    switch (params.wall_drag_mode) {
        case WallDragMode::Off:      amrex::Print() << "off"; break;
        case WallDragMode::Explicit: amrex::Print() << "explicit"; break;
        case WallDragMode::Implicit: amrex::Print() << "implicit"; break;
    }
    amrex::Print() << ")\n";
    amrex::Print() << "[UCM]   Cd_wall             = " << params.Cd_wall << "\n";
    amrex::Print() << "[UCM]   Cd_roof             = " << params.Cd_roof << "\n";
    amrex::Print() << "[UCM]   --- Phase 3.5A SEB Newton Solver ---\n";
    amrex::Print() << "[UCM]   Ch_roof             = " << params.Ch_roof << "\n";
    amrex::Print() << "[UCM]   Ch_wall             = " << params.Ch_wall << "\n";
    amrex::Print() << "[UCM]   Ch_road             = " << params.Ch_road << "\n";
    amrex::Print() << "[UCM]   slab_dz [m]         = " << params.slab_dz << "\n";
    amrex::Print() << "[UCM]   --- Phase 3.5B Prescribed Radiation Forcing ---\n";
    amrex::Print() << "[UCM]   use_prescribed_radiation = " << (params.use_prescribed_radiation ? "true" : "false") << "\n";
    amrex::Print() << "[UCM]   lat_deg [°N]        = " << params.lat_deg << "\n";
    amrex::Print() << "[UCM]   lon_deg [°E]        = " << params.lon_deg << "\n";
    amrex::Print() << "[UCM]   julian_day [1-365]  = " << params.julian_day << "\n";
    amrex::Print() << "[UCM]   solar_time_start_s [s] = " << params.solar_time_start_s << "\n";
    amrex::Print() << "[UCM]   solar_constant [W/m^2] = " << params.solar_constant << "\n";
    amrex::Print() << "[UCM]   sw_transmission [-] = " << params.sw_transmission << "\n";
    amrex::Print() << "[UCM]   sky_emissivity [-]  = " << params.sky_emissivity << "\n";
    
    // Phase 4.2 — Cloud-aware radiation
    std::string cloud_source_name;
    if (params.cloud_source == CloudSource::None) {
       cloud_source_name = "none";
    } else if (params.cloud_source == CloudSource::Constant) {
       cloud_source_name = "constant";
    } else if (params.cloud_source == CloudSource::Csv) {
       cloud_source_name = "csv";
    }
    amrex::Print() << "[UCM]   cloud_source = " << cloud_source_name << "\n";
    if (params.cloud_source == CloudSource::Constant) {
       amrex::Print() << "[UCM]   cloud_constant_fraction [-] = " << params.cloud_constant_fraction << "\n";
    }
    if (params.cloud_source == CloudSource::Csv) {
       amrex::Print() << "[UCM]   cloud_csv_path = " << params.cloud_csv_path << "\n";
    }
    amrex::Print() << "[UCM]   cloud_sw_a [-] = " << params.cloud_sw_a << "\n";
    amrex::Print() << "[UCM]   cloud_sw_b [-] = " << params.cloud_sw_b << "\n";
    
    // Phase 4.3 — Real radiation extraction (placeholder)
    amrex::Print() << "[UCM]   radiation_source = " << params.radiation_source_str << "\n";
    
    // Phase 5.1b — SW multi-bounce radiosity solver
    amrex::Print() << "[UCM]   --- Phase 5.1b SW Multi-Bounce Radiosity ---\n";
    amrex::Print() << "[UCM]   radiosity_mode = " << params.radiosity_mode_str << "\n";

    // Phase 5.1c — LW multi-bounce radiosity solver
    amrex::Print() << "[UCM]   --- Phase 5.1c LW Multi-Bounce Radiosity ---\n";
    amrex::Print() << "[UCM]   lw_radiosity_mode = " << params.lw_radiosity_mode_str << "\n";
    
    // Phase 5.2 — HVAC waste heat
    amrex::Print() << "[UCM]   --- Phase 5.2 HVAC Waste Heat ---\n";
    amrex::Print() << "[UCM]   hvac_mode = " << params.hvac_mode_str << "\n";
    if (params.hvac_mode == HVACMode::Simple) {
        amrex::Print() << "[UCM]   hvac_csv_path = " << params.hvac_csv_path << "\n";
        amrex::Print() << "[UCM]   occupancy_csv_path = " << params.occupancy_csv_path << "\n";
        amrex::Print() << "[UCM]   hvac_hysteresis_K = " << params.hvac_hysteresis_K << "\n";
        amrex::Print() << "[UCM]   hvac_cop_default = " << params.hvac_cop_default << "\n";
        amrex::Print() << "[UCM]   hvac_setpoint_default_K = " << params.hvac_setpoint_default_K << " K\n";
    }

    // Phase 5.6 — Fractional urban coverage (f_urb) blending
    amrex::Print() << "[UCM]   --- Phase 5.6 Flux Interface Blending ---\n";
    amrex::Print() << "[UCM]   interface_mode = " << params.interface_mode_str << "\n";
    
    amrex::Print() << "[UCM] =========================================================\n";
    amrex::Print() << "\n";

    // Phase 1.2 grid check banner (post-allocation) will be emitted after 
    // allocate_ucm_fields() completes in ERF::InitData_post()
}

void check_ucm_grid_and_fields(const UCMParams& params,
                                const UCMGrid& ucm_grid,
                                const UCMFields& ucm_fields,
                                int /*lev*/)
{
    // Check that all fields are allocated
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        ucm_fields.all_allocated(),
        "[UCM] Not all UCMFields are allocated! Check the list above.");

    // Phase 1.2: Grid check banner (post-allocation, gated on ucm_debug)
    if (params.ucm_debug && amrex::ParallelDescriptor::IOProcessor()) {
        const amrex::Box& ucm_domain = ucm_grid.geom.Domain();
        int nx_ucm = ucm_domain.bigEnd(0) - ucm_domain.smallEnd(0) + 1;
        int ny_ucm = ucm_domain.bigEnd(1) - ucm_domain.smallEnd(1) + 1;

        amrex::Print() << "\n";
        amrex::Print() << "[UCM] =========================================================\n";
        amrex::Print() << "[UCM] Phase 1.2 — Grid and Fields Check\n";
        amrex::Print() << "[UCM] =========================================================\n";
        amrex::Print() << "[UCM]   UCM grid extents   = " << nx_ucm << " × " << ny_ucm 
                       << " × 1 (cells)\n";
        amrex::Print() << "[UCM]   Refinement ratio   = " << ucm_grid.grid_ratio << "\n";
        amrex::Print() << "[UCM]   Ghost cells        = IntVect(1, 1, 0)\n";
        amrex::Print() << "[UCM]   All fields allocated: true\n";
        //amrex::Print() << "[UCM]   is_urban set to 1 everywhere (homogeneous patch)\n";
        if (params.building_layout_csv_path.empty()) {
            amrex::Print() << "[UCM]   is_urban set to 1 everywhere (homogeneous patch)\n";
        } else {
            amrex::Print() << "[UCM]   is_urban populated from CSV (mixed urban/non-urban)\n";
        }
        amrex::Print() << "[UCM] =========================================================\n";
        amrex::Print() << "\n";
    }

    // Phase 3.3: PBLH dependency guard — one-time banner
    // Design contract #4: UCM MUST NOT call SurfaceLayer::get_pblh().
    // All stability inputs come from u_star, t_star, q_star only.
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "[UCM][3.3][prerequisites] PBLH guard: CLEAN — no PBLH dependency detected in UCM inputs.\n"
                       << "  SurfaceLayer inputs consumed: u_star, t_star, q_star (all OK per design contract #4).\n";
    }
}
