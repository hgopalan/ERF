/**
 * @file ERF_UCMParams.cpp
 * @brief Implementation of SLUCM parameter reading
 *
 * Reads all SLUCM parameters from the "erf.ucm.*" ParmParse namespace.
 * Each parameter has a default value suitable for WRF-style neutral ABL simulations.
 *
 * References:
 *  - Source/Dust/ERF_DustParams.cpp
 *  - AMReX ParmParse documentation
 */

#include <ERF_UCMParams.H>
#include <AMReX_ParmParse.H>

void UCMParams::read_from_parmparse()
{
    amrex::ParmParse pp("erf.ucm");

    // Section 1: Module control flags
    pp.query("enable", enable);
    pp.query("ucm_debug", ucm_debug);

    // Section 2: AMR and grid parameters
    pp.query("anchor_level", anchor_level);
    pp.query("static_refinement", static_refinement);
    pp.query("grid_ratio", grid_ratio);
    pp.query("allow_steep_terrain", allow_steep_terrain);

    // Section 3: Atmosphere coupling
    // Phase 2.11-fix: Try to read per-process knobs first, then legacy scalar for backward compat.
    pp.query("atm_feedback_momentum", atm_feedback_momentum);
    pp.query("atm_feedback_heat", atm_feedback_heat);
    pp.query("atm_feedback_moisture", atm_feedback_moisture);

    // Also try to read legacy scalar (sentinel -1.0 = not set).
    pp.query("atm_feedback", atm_feedback);

    // Resolution logic:
    // If legacy was explicitly set (>= 0) and no new knobs were set yet, propagate legacy to all three.
    if (atm_feedback >= 0.0 && atm_feedback_momentum == 1.0 && atm_feedback_heat == 0.0 && atm_feedback_moisture == 0.0) {
        // Only propagate if the new knobs are still at their defaults (not user-set).
        // Since defaults are distinct (1.0, 0.0, 0.0), if all are default and legacy is set,
        // we can safely assume legacy was the intent.
        atm_feedback_momentum = atm_feedback;
        atm_feedback_heat     = atm_feedback;
        atm_feedback_moisture = atm_feedback;
    } else if (atm_feedback >= 0.0 && (atm_feedback_momentum != 1.0 || atm_feedback_heat != 0.0 || atm_feedback_moisture != 0.0)) {
        // Both legacy and new knobs set: warn and let new knobs win.
        amrex::Print() << "[UCM][2.11] WARNING: Both legacy atm_feedback and per-process knobs set!\n"
                       << "  Using per-process values: momentum=" << atm_feedback_momentum
                       << ", heat=" << atm_feedback_heat
                       << ", moisture=" << atm_feedback_moisture << "\n"
                       << "  (Legacy atm_feedback=" << atm_feedback << " is ignored.)\n";
    }

    pp.query("zref", zref);
    pp.query("z0_over_H", z0_over_H);
    pp.query("d_over_H", d_over_H);

    // Section 4: Vertical structure
    pp.query("alpha_ucm", alpha_ucm);

    // Section 4.1: Phase 2.6 — Morphology-aware injection parameters
    pp.query("alpha_scale", alpha_scale);
    pp.query("alpha_min", alpha_min);
    pp.query("alpha_max", alpha_max);
    pp.query("use_morphology_injection", use_morphology_injection);

    // Section 4.2: Phase 2.7 — Facet3D BEP-style geometric injection parameters
    pp.query("use_facet3d_injection", use_facet3d_injection);
    pp.query("use_gaussian_height_distribution", use_gaussian_height_distribution);
    pp.query("height_std_threshold_m", height_std_threshold_m);

    // Section 4.3: Phase 2.8 — BEP-style momentum drag parameters
    pp.query("wall_drag_mode", wall_drag_mode_str);
    pp.query("Cd_wall", Cd_wall);
    pp.query("Cd_roof", Cd_roof);

    pp.query("slab_N_layers", slab_N_layers);
    pp.query("slab_T_deep", slab_T_deep);
    pp.query("slab_L", slab_L);
    pp.query("k_therm_uniform", k_therm_uniform);
    pp.query("rho_cp_uniform", rho_cp_uniform);
    pp.query("newton_max_iter", newton_max_iter);
    pp.query("newton_tol_K", newton_tol_K);
    pp.query("newton_trace_ncells", newton_trace_ncells);

    // Section 5: Building morphology (homogeneous)
    pp.query("H_bldg_uniform", H_bldg_uniform);
    pp.query("W_road_uniform", W_road_uniform);
    pp.query("W_roof_uniform", W_roof_uniform);

    // Section 6: Radiative properties (shortwave)
    pp.query("albedo_roof", albedo_roof);
    pp.query("albedo_wall", albedo_wall);
    pp.query("albedo_road", albedo_road);

    // Section 7: Radiative properties (longwave)
    pp.query("emissivity_roof", emissivity_roof);
    pp.query("emissivity_wall", emissivity_wall);
    pp.query("emissivity_road", emissivity_road);

    // Section 8: Output and diagnostics
    pp.query("ucm_plot_int", ucm_plot_int);
    pp.query("ucm_atm_plot_int", ucm_atm_plot_int);
    pp.query("ucm_diag_file", ucm_diag_file);

    // Section 9: CSV readers for heterogeneous inputs (Phase 2.1)
    pp.query("building_layout_csv_path", building_layout_csv_path);
    pp.query("material_library_csv_path", material_library_csv_path);

    // Section 9b: Phase 2.3 — Facet-split sensible heat and anthropogenic heat
    pp.query("plan_area_frac_uniform", plan_area_frac_uniform);
    pp.query("AH_uniform_Wm2", AH_uniform_Wm2);
    pp.query("AH_daytime_peak", AH_daytime_peak);
    pp.query("AH_profile_type_default", AH_profile_type_default);

    // Section 9a: Test placeholders
    pp.query("test_ustar", test_ustar);
    pp.query("test_surf_temp_K", test_surf_temp_K);

    // Section 10: Phase 3.4 — Stability-aware canyon-atmosphere exchange
    pp.query("use_stability_correction", use_stability_correction);
    pp.query("zeta_max_stable", zeta_max_stable);
    pp.query("zeta_min_unstable", zeta_min_unstable);

    // Section 11: Phase 3.5A — SEB Newton solver exchange coefficients
    pp.query("Ch_roof", Ch_roof);
    pp.query("Ch_wall", Ch_wall);
    pp.query("Ch_road", Ch_road);
    pp.query("slab_dz", slab_dz);
    pp.query("T_skin_init_K", T_skin_init_K);
    pp.query("T_canyon_init_K", T_canyon_init_K);

    // Section 12: Phase 3.5B — Prescribed diurnal SW/LW radiation forcing
    pp.query("use_prescribed_radiation", use_prescribed_radiation);
    pp.query("lat_deg", lat_deg);
    pp.query("lon_deg", lon_deg);
    pp.query("julian_day", julian_day);
    pp.query("solar_time_start_s", solar_time_start_s);
    pp.query("solar_constant", solar_constant);
    pp.query("sw_transmission", sw_transmission);
    pp.query("sky_emissivity", sky_emissivity);

    // Section 13: Phase 4.2 — Cloud-aware analytical radiation forcing
    pp.query("cloud_source", cloud_source_str);
    pp.query("cloud_constant_fraction", cloud_constant_fraction);
    pp.query("cloud_csv_path", cloud_csv_path);
    pp.query("cloud_sw_a", cloud_sw_a);
    pp.query("cloud_sw_b", cloud_sw_b);

    // Resolve cloud_source_str to enum
    if (cloud_source_str == "none") {
        cloud_source = CloudSource::None;
    } else if (cloud_source_str == "constant") {
        cloud_source = CloudSource::Constant;
    } else if (cloud_source_str == "csv") {
        cloud_source = CloudSource::Csv;
    } else {
        amrex::Error("erf.ucm.cloud_source must be 'none', 'constant', or 'csv'");
    }

    // Section 14: Phase 4.3 — Real radiation extraction (placeholder)
    pp.query("radiation_source", radiation_source_str);
    if (radiation_source_str == "analytic") {
       radiation_source = RadiationSource::Analytic;
    } else if (radiation_source_str == "erf") {
       radiation_source = RadiationSource::Erf;
       // Layer 1 (startup validation): Verify ERF has an active radiation solver
       amrex::ParmParse pp_erf("erf");
       std::string rad_model = "None";
       pp_erf.query("radiation_model", rad_model);
       if (rad_model == "None" || rad_model == "none" || rad_model.empty()) {
           amrex::Abort(
               "[UCM][4.3] erf.ucm.radiation_source=erf requires an active ERF "
               "radiation solver, but erf.radiation_model is '" + rad_model + "'. "
               "Either enable ERF radiation (e.g., erf.radiation_model=RRTMGP) or "
               "set erf.ucm.radiation_source=analytic.");
       }
    } else {
       amrex::Abort("[UCM][4.3] Invalid erf.ucm.radiation_source '" +
                    radiation_source_str +
                    "'; expected 'analytic' or 'erf'.");
    }

    // Section 15: Phase 5.1b — SW multi-bounce radiosity solver
    pp.query("radiosity_mode", radiosity_mode_str);
    if (radiosity_mode_str == "single") {
        radiosity_mode = RadiosityMode::Single;
    } else if (radiosity_mode_str == "multi") {
        radiosity_mode = RadiosityMode::Multi;
    } else {
        amrex::Abort("[UCM][5.1b] Invalid erf.ucm.radiosity_mode; expected 'single' or 'multi'.");
    }

    // Section 16: Phase 5.1c — LW multi-bounce radiosity (Contracts #19, #20)
    pp.query("lw_radiosity_mode", lw_radiosity_mode_str);
    if (lw_radiosity_mode_str == "single") {
        lw_radiosity_mode = LWRadiosityMode::Single;
    } else if (lw_radiosity_mode_str == "multi-lagged") {
        lw_radiosity_mode = LWRadiosityMode::MultiLagged;
    } else {
        amrex::Abort("[UCM][5.1c] Invalid erf.ucm.lw_radiosity_mode '" +
                     lw_radiosity_mode_str +
                     "'; expected 'single' or 'multi-lagged'.");
    }

    // Phase 5.2: HVAC waste heat mode
    pp.query("hvac_mode", hvac_mode_str);
    if (hvac_mode_str == "off") {
        hvac_mode = HVACMode::Off;
    } else if (hvac_mode_str == "simple") {
        hvac_mode = HVACMode::Simple;
    } else {
        amrex::Abort("[UCM][5.2] Invalid erf.ucm.hvac_mode '" + hvac_mode_str +
                     "'; expected 'off' or 'simple'.");
    }
    pp.query("hvac_csv_path", hvac_csv_path);
    pp.query("occupancy_csv_path", occupancy_csv_path);
    pp.query("hvac_hysteresis_K", hvac_hysteresis_K);
    pp.query("hvac_cop_default", hvac_cop_default);
    pp.query("hvac_setpoint_default_K", hvac_setpoint_default_K);
    pp.query("hvac_cop_degradation_per_K", hvac_cop_degradation_per_K);  // Phase 5.5

    // Validation: if hvac_mode == Simple, both CSV paths must be non-empty
    if (hvac_mode == HVACMode::Simple) {
        if (hvac_csv_path.empty()) {
            amrex::Abort("[UCM][5.2] hvac_mode='simple' requires erf.ucm.hvac_csv_path to be specified. "
                        "See UCM_DEVELOPMENT.md for CSV format (D3).");
        }
        if (occupancy_csv_path.empty()) {
            amrex::Abort("[UCM][5.2] hvac_mode='simple' requires erf.ucm.occupancy_csv_path to be specified. "
                        "See UCM_DEVELOPMENT.md for CSV format (D4).");
        }
    }

    // Phase 5.3: Cool roof mode (CSV knob documentation)
    pp.query("cool_roof_mode", cool_roof_mode_str);
    if (cool_roof_mode_str == "off") {
        cool_roof_mode = CoolRoofMode::Off;
    } else if (cool_roof_mode_str == "recipe-only") {
        cool_roof_mode = CoolRoofMode::RecipeOnly;
    } else {
        amrex::Abort("[UCM][5.3] Invalid erf.ucm.cool_roof_mode '" + cool_roof_mode_str +
                     "'; expected 'off' or 'recipe-only'.");
    }

    // Phase 5.3: Green roof mode (soil conduction + latent heat)
    pp.query("green_roof_mode", green_roof_mode_str);
    if (green_roof_mode_str == "off") {
        green_roof_mode = GreenRoofMode::Off;
    } else if (green_roof_mode_str == "simple") {
        green_roof_mode = GreenRoofMode::Simple;
    } else {
        amrex::Abort("[UCM][5.3] Invalid erf.ucm.green_roof_mode '" + green_roof_mode_str +
                     "'; expected 'off' or 'simple'.");
    }
    pp.query("green_roof_r_stomatal_s_per_m", green_roof_r_stomatal_s_per_m);
    pp.query("green_roof_thickness_m", green_roof_thickness_m);
    pp.query("green_roof_k_therm", green_roof_k_therm);
    pp.query("green_roof_rho_cp", green_roof_rho_cp);
    pp.query("green_roof_soil_capacity_m", green_roof_soil_capacity_m);

    // Phase 5.3: Permeable road mode (soil layer + moisture bucket)
    pp.query("permeable_road_mode", permeable_road_mode_str);
    if (permeable_road_mode_str == "off") {
        permeable_road_mode = PermeableRoadMode::Off;
    } else if (permeable_road_mode_str == "simple") {
        permeable_road_mode = PermeableRoadMode::Simple;
    } else {
        amrex::Abort("[UCM][5.3] Invalid erf.ucm.permeable_road_mode '" + permeable_road_mode_str +
                     "'; expected 'off' or 'simple'.");
    }
    pp.query("permeable_road_thickness_m", permeable_road_thickness_m);
    pp.query("permeable_road_k_therm", permeable_road_k_therm);
    pp.query("permeable_road_rho_cp", permeable_road_rho_cp);
    pp.query("permeable_road_soil_capacity_m", permeable_road_soil_capacity_m);

    // Phase 5.6: Interface mode (f_urb blending)
    pp.query("interface_mode", interface_mode_str);
    if (interface_mode_str == "binary") {
       interface_mode = InterfaceMode::Binary;
    } else if (interface_mode_str == "blended") {
       interface_mode = InterfaceMode::Blended;
    } else {
       amrex::Abort("[UCM][5.6] Invalid erf.ucm.interface_mode '" + interface_mode_str +
                    "'; expected 'binary' or 'blended'.");
    }

    // Phase 6.1: Tree canopy drag
    pp.query("tree_drag_mode", tree_drag_mode_str);
    pp.query("tree_layout_csv_path", tree_layout_csv_path);
    pp.query("Cd_leaf_default", Cd_leaf_default);
    if (tree_drag_mode_str == "off") {
        tree_drag_mode = TreeDragMode::Off;
    } else if (tree_drag_mode_str == "explicit") {
        tree_drag_mode = TreeDragMode::Explicit;
    } else {
        amrex::Abort("[UCM][6.1] Invalid erf.ucm.tree_drag_mode '" + tree_drag_mode_str +
                     "'; expected 'off' or 'explicit'.");
    }
    if (tree_drag_mode == TreeDragMode::Explicit && tree_layout_csv_path.empty()) {
        amrex::Abort("[UCM][6.1] tree_drag_mode=explicit requires "
                     "erf.ucm.tree_layout_csv_path (Contract #25).");
    }

    // Phase 6.2a: Tree Beer-Lambert SW attenuation
    pp.query("tree_rad_mode", tree_rad_mode_str);
    pp.query("k_ext_tree",    k_ext_tree);
    if (tree_rad_mode_str == "off") {
        tree_rad_mode = TreeRadMode::Off;
    } else if (tree_rad_mode_str == "beer_lambert") {
        tree_rad_mode = TreeRadMode::BeerLambert;
    } else {
        amrex::Abort("[UCM][6.2a] Invalid erf.ucm.tree_rad_mode '"
                     + tree_rad_mode_str + "'; expected 'off' or 'beer_lambert'.");
    }
    if (k_ext_tree < 0.0 || k_ext_tree > 5.0) {
        amrex::Abort("[UCM][6.2a] erf.ucm.k_ext_tree out of [0, 5] range.");
    }

    // Phase 6.2b: Crown SEB facet (4-var Newton solver)
    pp.query("seb_mode", seb_mode_str);
    pp.query("Ch_leaf", Ch_leaf);
    pp.query("eps_leaf", eps_leaf);
    if      (seb_mode_str == "3var") seb_mode = SEBMode::ThreeVar;
    else if (seb_mode_str == "4var") seb_mode = SEBMode::FourVar;
    else amrex::Abort("[UCM][6.2b] Invalid erf.ucm.seb_mode '" + seb_mode_str
                     + "'; expected '3var' or '4var'.");
    if (Ch_leaf <= 0.0 || Ch_leaf > 1.0)
       amrex::Abort("[UCM][6.2b] Ch_leaf out of (0,1] range.");
    if (eps_leaf <= 0.0 || eps_leaf > 1.0)
       amrex::Abort("[UCM][6.2b] eps_leaf out of (0,1] range.");

    // Phase 6.2b hotfix3: facet-specific crown view factors
    // Roof gets ZERO (crown is geometrically below roof plane — no LW hits roof).
    // Wall and road each have their own view factor (road sees more of crown than wall).
    pp.query("crown_view_factor_wall", crown_view_factor_wall);
    pp.query("crown_view_factor_road", crown_view_factor_road);
    if (crown_view_factor_wall < 0.0 || crown_view_factor_wall > 1.0)
       amrex::Abort("[UCM][6.2b] crown_view_factor_wall out of [0,1] range.");
    if (crown_view_factor_road < 0.0 || crown_view_factor_road > 1.0)
       amrex::Abort("[UCM][6.2b] crown_view_factor_road out of [0,1] range.");

    // Phase 6.2b hotfix3: fallback crown area fraction (used only if fields.crown_area_frac not populated)
    pp.query("crown_area_frac_default", crown_area_frac_default);
    if (crown_area_frac_default < 0.0 || crown_area_frac_default > 1.0)
       amrex::Abort("[UCM][6.2b] crown_area_frac_default out of [0,1] range.");
}