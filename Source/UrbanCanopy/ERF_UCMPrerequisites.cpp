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

void check_ucm_prerequisites(const UCMParams& params,
                              int max_level,
                              int finest_level,
                              bool use_terrain,
                              int lev)
{
    // Check 1: anchor_level within bounds
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.anchor_level >= 0,
        "[UCM] anchor_level must be >= 0. Set: erf.ucm.anchor_level = 0");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.anchor_level <= finest_level,
        "[UCM] anchor_level must be <= finest_level. Reduce erf.ucm.anchor_level");

    // Check 2: Phase 1.1 constraint - anchor_level must be 0
    if (params.anchor_level > 0) {
        std::string msg = std::string("[UCM] anchor_level > 0 not supported in Phase 1.1. ")
                        + "Set: erf.ucm.anchor_level = 0 or wait for Phase 3.1 multi-level support. "
                        + "See: Source/UrbanCanopy/UCM_DEVELOPMENT.md";
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false, msg.c_str());
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

    // Check 10: Phase 1.4 — atm_feedback in [0.0, 1.0]
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        params.atm_feedback >= 0.0 && params.atm_feedback <= 1.0,
        "[UCM] atm_feedback must be in [0.0, 1.0]. Phase 1.4: 0=one-way (compute but don't inject), 1=full injection.");

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
    amrex::Print() << "[UCM]   --- Phase 1.4 Injection Parameters ---\n";
    amrex::Print() << "[UCM]   atm_feedback        = " << params.atm_feedback << " [0,1]\n";
    amrex::Print() << "[UCM]   alpha_ucm [m]       = " << params.alpha_ucm << "\n";
    amrex::Print() << "[UCM]   ucm_plot_int        = " << params.ucm_plot_int << "\n";
    amrex::Print() << "[UCM]   ucm_diag_file       = " << params.ucm_diag_file << "\n";
    amrex::Print() << "[UCM] =========================================================\n";
    amrex::Print() << "\n";

    // Phase 1.2 grid check banner (post-allocation) will be emitted after 
    // allocate_ucm_fields() completes in ERF::InitData_post()
}

void check_ucm_grid_and_fields(const UCMParams& params,
                                const UCMGrid& ucm_grid,
                                const UCMFields& ucm_fields,
                                int lev)
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
        amrex::Print() << "[UCM]   is_urban set to 1 everywhere (homogeneous patch)\n";
        amrex::Print() << "[UCM] =========================================================\n";
        amrex::Print() << "\n";
    }
}
