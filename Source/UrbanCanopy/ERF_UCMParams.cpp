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
}
