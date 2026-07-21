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

void UCMParams::read_from_parmparse(int lev)
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
    pp.query("atm_feedback", atm_feedback);
    pp.query("zref", zref);

    // Section 4: Vertical structure
    pp.query("alpha_ucm", alpha_ucm);

    // Section 4b: Slab conduction solver parameters (Phase 1.3+)
    pp.query("slab_N_layers", slab_N_layers);
    pp.query("slab_T_deep", slab_T_deep);
    pp.query("slab_L", slab_L);
    pp.query("k_therm_uniform", k_therm_uniform);
    pp.query("rho_cp_uniform", rho_cp_uniform);
    pp.query("newton_max_iter", newton_max_iter);
    pp.query("newton_tol_K", newton_tol_K);

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
    pp.query("ucm_diag_file", ucm_diag_file);

    // Section 9: Test placeholders
    pp.query("test_ustar", test_ustar);
    pp.query("test_surf_temp_K", test_surf_temp_K);
}
