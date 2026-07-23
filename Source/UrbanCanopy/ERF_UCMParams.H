/**
 * @file ERF_UCMParams.H
 * @brief Parameter struct for the ERF-SLUCM module
 *
 * Defines all configuration parameters for the Single-Layer Urban Canopy Model
 * (SLUCM) integrated within the ERF mesoscale atmospheric model. Parameters are
 * read from the "erf.ucm.*" ParmParse namespace at initialization.
 *
 * ## Cross-Cutting Design Contracts (applies to all phases)
 *
 * **Contract 1 — `lev`-aware API from day 1:**
 * All UCM functions take `int lev` as an argument (default 0). No hardcoded `0`
 * may appear in any file for level indexing. UCM MultiFabs are Vector<unique_ptr>
 * sized to finest_level+1; only anchor_level is allocated in Phase 1.1.
 *
 * **Contract 2 — Anchor level and static refinement:**
 * `anchor_level` (int, default 0): Single AMR level UCM runs on.
 * `static_refinement` (bool, default true): Required true; errors if regrid occurs.
 * In Phase 1.1, only anchor_level=0 is allowed; higher values gated for Phase 3.1.
 *
 * **Contract 3 — Terrain-following coordinates:**
 * All vertical operations use `z_phys_cc(i,j,k) - z_phys_cc(i,j,klo)` for
 * height-above-surface. Wind extraction uses `z_target = z_phys_cc(i,j,klo) + 
 * H_bldg(i,j) + zref`. No hardcoded z-coordinates permitted.
 * Terrain slope > 20° → warning; > 30° → error (unless allow_steep_terrain=true).
 *
 * **Contract 4 — Zero PBLH dependency:**
 * UCM MUST NOT call `SurfaceLayer::get_pblh(lev)`. All stability info comes from
 * u*, theta*, q*, and Obukhov length L (populated by MRF/YSU/MYNN2.5).
 *
 * **Contract 5 — `is_urban(i,j)` mask exclusivity:**
 * `ucm_is_urban` iMultiFab declared on ERF (allocated Phase 1.2). No LSM/MOST
 * bypass hooks wired yet; Phase 4.1 integrates.
 *
 * **Contract 6 — Build and file conventions:**
 * Every .cpp in Source/UrbanCanopy/ registered in BOTH Make.package AND CMake/BuildERFExe.cmake.
 * All MultiFabs use IntVect(1,1,0) ghost cells (2D slab). Header-only files in
 * Make.package CEXE_headers only. ParallelFor headers only included where used.
 *
 * References:
 *   - Source/Dust/DUST_DEVELOPMENT.md (ERF-Hazard branch)
 *   - Source/LNG/LNG_DEVELOPMENT.md (commit ba55b73...)
 *   - Source/LNG/LNG_MPI_SKILLS.md (commit ba55b73...)
 */

#ifndef ERF_UCM_PARAMS_H
#define ERF_UCM_PARAMS_H

#include <AMReX_REAL.H>
#include <string>

/**
 * @enum WallDragMode
 * @brief Resolved wall drag application mode for momentum coupling
 *
 * Determined at initialization based on erf.ucm.wall_drag_mode and solver type.
 * Phase 2.8: Compressible uses explicit; anelastic uses implicit (stub for validation deferred to 2.8b).
 */
enum class WallDragMode
{
    Off = 0,       ///< Drag disabled (backward-compatible default)
    Explicit = 1,  ///< Apply drag directly to momentum RHS (compressible)
    Implicit = 2   ///< Post-projection multiplicative correction (anelastic stub)
};

/**
 * @struct UCMParams
 * @brief Container for all SLUCM configuration parameters read from ParmParse
 *
 * Organized by functional category for clarity and maintenance. All parameters
 * have default values suitable for a WRF-style neutral ABL homogeneous baseline.
 * 
 * Default configuration represents a simplified urban domain with uniform 10 m
 * buildings, 10 m roads/roofs, and wavelength-independent (gray) radiative properties.
 */
struct UCMParams
{
    // =========================================================================
    // Section 1: Module control flags
    // =========================================================================

    bool    enable             = false;     ///< Enable SLUCM module [Phase 1]
    bool    ucm_debug          = false;     ///< Per-step debug output [Phase 1]

    // =========================================================================
    // Section 2: AMR and grid parameters
    // =========================================================================

    int     anchor_level       = 0;         ///< AMR level at which UCM runs [Phase 1.1 only 0]
    bool    static_refinement  = true;      ///< Static refinement required (no regrid on anchor_level)
    int     grid_ratio         = 1;         ///< UCM grid refinement vs ATM anchor level [unitless]
    bool    allow_steep_terrain = false;    ///< Permit terrain slope > 30° [Phase 1.1 gated false]

    // =========================================================================
    // Section 3: Atmosphere coupling
    // =========================================================================

    amrex::Real atm_feedback   = 0.0;       ///< Coupling strength [0,1] [Phase 1.1 must be 0.0]
    amrex::Real zref           = 2.0;       ///< Wind extraction ref height above roof [m]
    amrex::Real z0_over_H      = 0.1;       ///< z0 as fraction of building height (MacDonald 1998) [-] [Phase 2.2]
    amrex::Real d_over_H       = 0.7;       ///< d as fraction of building height (WRF SLUCM) [-] [Phase 2.2]

    // =========================================================================
    // Section 4: Vertical structure (slab conduction, Phase 1.3+)
    // =========================================================================

    amrex::Real alpha_ucm      = 10.0;      ///< Exponential injection e-folding depth [m]

    // =========================================================================
    // Section 4.1: Phase 2.6 — Morphology-aware injection parameters
    // =========================================================================

    amrex::Real alpha_scale              = 1.5;       ///< Multiplier: alpha_ij = alpha_scale * H_bldg_mean (per-cell e-folding) [-]
    amrex::Real alpha_min                = 1.0;       ///< Minimum e-folding depth [m] (avoid divide-by-zero for non-urban)
    amrex::Real alpha_max                = 50.0;      ///< Maximum e-folding depth [m] (cap for tall towers)
    bool        use_morphology_injection  = true;      ///< Phase 2.6: morphology-aware split injection; false = fall back to uniform alpha_ucm

    // =========================================================================
    // Section 4.2: Phase 2.7 — Facet3D BEP-style geometric injection parameters
    // =========================================================================

    bool        use_facet3d_injection              = true;       ///< Phase 2.7: enable facet-3D geometric injection with BEP-style overlap; false = fall back to Phase 2.6
    bool        use_gaussian_height_distribution  = false;      ///< Phase 2.7: use continuous Gaussian PDF for wall height dist (true) vs sharp 1-class BEP (false)
    amrex::Real height_std_threshold_m            = 0.1;        ///< Phase 2.7: below this threshold, Gaussian mode falls back to sharp [m]

    // =========================================================================
    // Section 4.3: Phase 2.8 — BEP-style momentum drag parameters
    // =========================================================================

    std::string wall_drag_mode_str = "auto";   ///< Phase 2.8: "auto" | "explicit" | "implicit" | "off" [string]
    WallDragMode wall_drag_mode = WallDragMode::Off;  ///< Phase 2.8: resolved mode (auto-determined at init)
    amrex::Real Cd_wall          = 0.4;        ///< Phase 2.8: BEP wall drag coefficient (Martilli 2002) [-]
    amrex::Real Cd_roof          = 0.15;       ///< Phase 2.8: BEP roof drag coefficient [-]

    int     slab_N_layers      = 4;         ///< Number of soil/material layers in slab [-]
    amrex::Real slab_T_deep    = 287.0;     ///< Deep (bottom BC) temperature [K]
    amrex::Real slab_L         = 0.3;       ///< Total slab thickness [m]
    amrex::Real k_therm_uniform = 1.5;      ///< Uniform thermal conductivity [W/m/K]
    amrex::Real rho_cp_uniform = 2.0e6;     ///< Uniform volumetric heat capacity [J/m^3/K]
    int     newton_max_iter    = 20;        ///< Max Newton iterations for facet SEB [-]
    amrex::Real newton_tol_K   = 0.01;      ///< Temperature convergence tolerance [K]

    // =========================================================================
    // Section 5: Building morphology (homogeneous, overridden by CSV in Phase 2)
    // =========================================================================

    amrex::Real H_bldg_uniform = 10.0;      ///< Uniform building height [m]
    amrex::Real W_road_uniform = 10.0;      ///< Uniform road width [m]
    amrex::Real W_roof_uniform = 10.0;      ///< Uniform roof width [m]

    // =========================================================================
    // Section 6: Radiative properties (shortwave albedo, Phase 1.4+ diagnostics)
    // =========================================================================

    amrex::Real albedo_roof    = 0.20;      ///< Homogeneous roof SW albedo [-]
    amrex::Real albedo_wall    = 0.20;      ///< Homogeneous wall SW albedo [-]
    amrex::Real albedo_road    = 0.15;      ///< Homogeneous road SW albedo [-]

    // =========================================================================
    // Section 7: Radiative properties (longwave emissivity)
    // =========================================================================

    amrex::Real emissivity_roof = 0.90;     ///< Homogeneous roof LW emissivity [-]
    amrex::Real emissivity_wall = 0.90;     ///< Homogeneous wall LW emissivity [-]
    amrex::Real emissivity_road = 0.94;     ///< Homogeneous road LW emissivity [-]

    // =========================================================================
    // Section 8: Output and diagnostics
    // =========================================================================

    int     ucm_plot_int       = -1;        ///< Plotfile interval [steps] (-1 disabled; Phase 1.4)
    int     ucm_atm_plot_int   = -1;        ///< ATM-grid plotfile interval [steps] (-1 disabled; Phase 2.5)
    std::string ucm_diag_file  = "ucm_diag.dat"; ///< Diagnostics CSV filename

    // =========================================================================
    // Section 9: CSV readers for heterogeneous inputs (Phase 2.1)
    // =========================================================================

    std::string building_layout_csv_path = "";  ///< Path to building-layout.csv (empty = homogeneous)
    std::string material_library_csv_path = ""; ///< Path to materials.csv (empty = homogeneous)

    // =========================================================================
    // Section 9b: Phase 2.3 — Facet-split sensible heat and anthropogenic heat
    // =========================================================================

    amrex::Real plan_area_frac_uniform = 0.5;   ///< Default plan-area fraction [-]
    amrex::Real AH_uniform_Wm2         = 0.0;   ///< Uniform anthropogenic heat [W/m^2]
    amrex::Real AH_daytime_peak        = 20.0;  ///< Peak of diurnal AH [W/m^2]
    int         AH_profile_type_default = 0;    ///< 0 = uniform, 1 = diurnal cosine

    // =========================================================================
    // Section 9a: Test placeholders (Phase 1.1 only; replaced Phase 1.3+)
    // =========================================================================

    amrex::Real test_ustar     = 0.0;       ///< Placeholder u* before Phase 1.3 [m/s]
    amrex::Real test_surf_temp_K = 293.15;  ///< Placeholder T_sfc before Phase 1.3 [K]

    // =========================================================================
    // Public methods
    // =========================================================================

    /**
     * @brief Read all parameters from ParmParse namespace "erf.ucm"
     *
     * Called during ERF initialization to populate this struct from input file.
     * All parameter names follow pattern "erf.ucm.name". ParmParse::query() is
     * used to preserve defaults if a parameter is not specified.
     *
     * @param[in] lev  AMR level (future use for level-specific params; default 0)
     */
    void read_from_parmparse(int lev = 0);
};

#endif // ERF_UCM_PARAMS_H
