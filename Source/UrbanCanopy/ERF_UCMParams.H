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
 * @enum CloudSource
 * @brief Cloud attenuation source for analytical radiation (Phase 4.2)
 *
 * Determines how cloud cover is supplied to the SW/LW attenuation formulae.
 * Phase 3.5b clear-sky (zero cloud) is the backward-compatible default.
 */
enum class CloudSource
{
    None = 0,      ///< Phase 3.5b clear-sky (zero cloud, default)
    Constant = 1,  ///< Single scalar cloud fraction held for the whole run
    Csv = 2        ///< Hourly cloud fraction read from CSV time series
};

/**
 * @enum RadiationSource
 * @brief Radiation source for UCM SEB closure (Phase 4.3)
 *
 * Selects whether SW_down and LW_down are computed via analytic formulae
 * (Phase 3.5b + Phase 4.2 cloud attenuation) or extracted from ERF's radiation
 * solver (Phase 4.3, currently a placeholder returning sentinel values).
 */
enum class RadiationSource
{
    Analytic = 0,  ///< Phase 3.5b analytic clear-sky + Phase 4.2 cloud (default)
    Erf = 1        ///< Extract from ERF radiation module (Phase 4.3 — placeholder)
};

/**
 * @enum RadiosityMode
 * @brief SW multi-bounce radiosity solver mode (Phase 5.1b)
 *
 * Selects whether SW reflection between wall_A, wall_B, and road is computed
 * via infinite-bounce 2×2 Cramer solver (multi) or single-bounce SVF weighting (single).
 * Roof unshaded; LW multi-bounce deferred to Phase 5.1c.
 */
enum class RadiosityMode
{
    Single = 0,  ///< Phase 2.4 single-bounce SVF weighting (backward compatible)
    Multi = 1    ///< Phase 5.1b 2×2 Cramer multi-bounce solver (default)
};

/**
 * @enum LWRadiosityMode
 * @brief LW radiation distribution mode (Phase 5.1c)
 *
 * Selects whether LW absorbed on wall and road facets uses single-bounce SVF
 * weighting (legacy Phase 3.5b path) or 2×2 lagged-T_skin radiosity
 * (Phase 5.1c). Multi-lagged uses previous-timestep T_skin in the RHS,
 * decoupling from the Newton iteration entirely. Fully coupled
 * radiosity-inside-Newton is out of scope (see Contract #19).
 */
enum class LWRadiosityMode
{
    Single = 0,       ///< Phase 3.5b single-bounce SVF (backward-compat default)
    MultiLagged = 1   ///< Phase 5.1c 2×2 lagged-T_skin radiosity multi-bounce
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

    // Phase 2.11-fix: Split atm_feedback into per-process knobs for physical correctness.
    // Momentum drag is always active (buildings always block wind).
    // Heat/moisture feedback is opt-in (TBD in later phases).
    amrex::Real atm_feedback_momentum = 1.0;  ///< Momentum drag feedback strength [0,1] [Phase 2.11]
    amrex::Real atm_feedback_heat     = 0.0;  ///< Heat injection feedback strength [0,1] [Phase 2.11]
    amrex::Real atm_feedback_moisture = 0.0;  ///< Moisture injection feedback strength [0,1] [Phase 2.11]

    // Legacy scalar. Retained for backward compatibility.
    // Sentinel value -1.0 means "not set by user in input file".
    // When set explicitly by user (>= 0), initializes all three per-process knobs above.
    amrex::Real atm_feedback   = -1.0;       ///< Legacy: coupling strength [0,1] (Phase 1.1-2.10)

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
    int     newton_trace_ncells = 5;        ///< Phase 3.5a-hotfix: max clamped cells to trace per step (0=disable) [-]

    // =========================================================================
    // Section 5: Building morphology (homogeneous, overridden by CSV in Phase 2)
    // =========================================================================

    amrex::Real H_bldg_uniform = 10.0;      ///< Uniform building height [m]
    amrex::Real W_road_uniform = 10.0;      ///< Uniform road width [m] (non-zero default allows CSV-driven canonicals to bypass assertion)
    amrex::Real W_roof_uniform = 10.0;      ///< Uniform roof width [m] (non-zero default allows CSV-driven canonicals to bypass assertion)

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
    // Section 10: Phase 3.4 — Stability-aware canyon-atmosphere exchange
    // =========================================================================

    bool use_stability_correction = false;  ///< Phase 3.4: Enable Obukhov-corrected heat exchange coefficient
    amrex::Real zeta_max_stable   = 2.0;   ///< Phase 3.4: Clip zeta > 0 at this value (prevent Ch → 0) [-]
    amrex::Real zeta_min_unstable = -5.0;  ///< Phase 3.4: Clip zeta < 0 at this value (prevent Ch blow-up) [-]

    // =========================================================================
    // Section 11: Phase 3.5A — SEB Newton solver exchange coefficients
    // =========================================================================

    amrex::Real Ch_roof = 0.01;      ///< Roof heat exchange coefficient [-]
    amrex::Real Ch_wall = 0.01;      ///< Wall heat exchange coefficient [-]
    amrex::Real Ch_road = 0.01;      ///< Road heat exchange coefficient [-]
    amrex::Real slab_dz  = 0.05;     ///< Top slab layer half-thickness [m]
    amrex::Real T_skin_init_K = 300.0;    ///< Initial facet skin temperature [K]
    amrex::Real T_canyon_init_K = 295.0;  ///< Initial canyon air temperature [K]

    // =========================================================================
    // Section 12: Phase 3.5B — Prescribed diurnal SW/LW radiation forcing
    // =========================================================================
    // Bridge analytic radiation formulae to unblock SEB closure until Phase 4.2.
    // At Phase 4.2, SW_down/LW_down will come from RRTM/RRTMG module instead.

    bool        use_prescribed_radiation = true;   ///< Phase 3.5b: Enable analytic SW/LW; false = wait for Phase 4.2
    amrex::Real lat_deg           = 42.36;         ///< Domain center latitude [°N] (Boston default)
    amrex::Real lon_deg           = -71.06;        ///< Domain center longitude [°E] (Boston default)
    amrex::Real julian_day        = 172.0;         ///< Day of year [1–365] (172 = summer solstice)
    amrex::Real solar_time_start_s = 21600.0;      ///< Local Solar Time at t=0 [s] (default 06:00 LST)
    amrex::Real solar_constant    = 1361.0;        ///< Top-of-atmosphere solar constant [W/m²]
    amrex::Real sw_transmission   = 0.7;           ///< Clear-sky bulk optical depth [-] (0.5–0.8)
    amrex::Real sky_emissivity    = 0.83;          ///< Clear-sky LW emissivity [-] (0.70–0.95)

    // =========================================================================
    // Section 13: Phase 4.2 — Cloud-aware analytical radiation forcing
    // =========================================================================
    // Extends Phase 3.5b with cloud fraction attenuation applied to SW/LW.
    // Cloud fraction source is selectable: none (clear-sky, default), constant scalar,
    // or CSV time series. Backward compatible: ucm.cloud_source=none reproduces Phase 3.5b
    // bit-identically.

    std::string cloud_source_str = "none";     ///< Phase 4.2: "none" | "constant" | "csv" [string]
    CloudSource cloud_source = CloudSource::None;  ///< Phase 4.2: cloud attenuation source (auto-determined at init)
    amrex::Real cloud_constant_fraction = 0.0;    ///< Phase 4.2: constant cloud fraction [0–1] if source=constant
    std::string cloud_csv_path = "";              ///< Phase 4.2: path to cloud CSV if source=csv
    amrex::Real cloud_sw_a = 0.75;                ///< Phase 4.2: Kasten & Czeplak SW attenuation coeff a [-]
    amrex::Real cloud_sw_b = 3.4;                 ///< Phase 4.2: Kasten & Czeplak SW attenuation exponent b [-]

    // =========================================================================
    // Section 14: Phase 4.3 — Real radiation extraction (placeholder)
    // =========================================================================
    // Selects SW/LW source for SEB. Default 'analytic' preserves Phase 3.5b +
    // Phase 4.2 behavior. 'erf' path is a placeholder: it wires the interface
    // through UCMLayer but the extraction stub returns sentinel values with a
    // one-time warning. Full RRTMG plumbing is deferred until the ERF core
    // radiation interface stabilizes.

    std::string radiation_source_str = "analytic";       ///< Phase 4.3: "analytic" | "erf"
    RadiationSource radiation_source = RadiationSource::Analytic;  ///< Phase 4.3: resolved enum

    // =========================================================================
    // Section 15: Phase 5.1b — SW multi-bounce radiosity solver
    // =========================================================================
    // Selects whether SW reflection between walls and road is computed via
    // infinite-bounce 2×2 Cramer solver (multi, default) or single-bounce
    // SVF weighting (single, backward compatible). Roof unshaded (F_roof_sky=1).
    // LW multi-bounce deferred to Phase 5.1c.

    std::string radiosity_mode_str = "multi";     ///< Phase 5.1b: "single" | "multi" [string]
    RadiosityMode radiosity_mode = RadiosityMode::Multi;  ///< Phase 5.1b: resolved enum

    // =========================================================================
    // Section 16: Phase 5.1c — LW multi-bounce radiosity (Contracts #19, #20)
    // =========================================================================
    // Multi-bounce LW reflection between wall_A, wall_B, and road via 2×2
    // radiosity solve using previous-timestep T_skin (lagged approach).
    // Roof LW unchanged (single-bounce). Defaults to 'single' for backward
    // compatibility — user opts into 'multi-lagged' explicitly.

    std::string lw_radiosity_mode_str = "single";           ///< "single" | "multi-lagged"
    LWRadiosityMode lw_radiosity_mode = LWRadiosityMode::Single;

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
    */
    void read_from_parmparse();
};

#endif // ERF_UCM_PARAMS_H
