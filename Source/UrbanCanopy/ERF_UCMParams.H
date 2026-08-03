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
 * @enum HVACMode
 * @brief HVAC waste heat computation mode (Phase 5.2)
 *
 * Selects whether HVAC waste heat is added to the anthropogenic heat
 * injection. Simple mode uses COP-based rejection of the residual
 * slab-conduction cooling load.
 */
enum class HVACMode
{
    Off = 0,      ///< No HVAC waste heat (backward-compat default)
    Simple = 1    ///< COP-based rejection with setpoint gate
};

/**
 * @enum HVACRejectionFacet
 * @brief HVAC waste heat rejection destination (Phase 5.5)
 *
 * Selects where HVAC waste heat is injected into the urban environment.
 * Sensible heat is distributed to the selected facet(s); latent heat
 * always goes to LE_latent (canyon moisture budget).
 */
enum class HVACRejectionFacet
{
    Roof = 0,         ///< Waste heat at roof only (Phase 5.4 default behavior)
    Road = 1,         ///< Waste heat at street-level (ground-source or packaged units)
    Distributed = 2   ///< Waste heat split across roof, walls, and road (weighted by area)
};

/**
 * @enum CoolRoofMode
 * @brief Cool roof high-albedo/high-emissivity surface mode (Phase 5.3)
 *
 * Cool roofs are fully representable via existing per-cell `albedo_roof` and
 * `emissivity_roof` fields. This enum is a documentation placeholder for the
 * CSV knob recipe. No new physics code is required; users set albedo_roof ~0.85
 * and emissivity_roof ~0.9 per cell via building_layout.csv.
 */
enum class CoolRoofMode
{
    Off = 0,      ///< No cool roof recipe (backward-compat default)
    RecipeOnly = 1 ///< Documentation mode; no physics change (values via CSV knob)
};

/**
 * @enum GreenRoofMode
 * @brief Green roof soil + vegetation layer mode (Phase 5.3)
 *
 * Green roofs add a vegetated soil layer on top of existing roof slab.
 * Reuses slab conduction TDMA (with green-roof-specific k_therm, rho_cp, thickness)
 * and latent heat pathway (LE_green → LE_latent). Bulk stomatal resistance model
 * for evapotranspiration (Ball-Berry / Jarvis deferred to Phase 6.3).
 */
enum class GreenRoofMode
{
    Off = 0,      ///< No green roof (backward-compat default)
    Simple = 1    ///< Soil conduction + bulk stomatal resistance
};

/**
 * @enum PermeableRoadMode
 * @brief Permeable pavement soil + moisture bucket mode (Phase 5.3)
 *
 * Permeable pavements replace conventional road surface with soil layer.
 * Reuses slab conduction TDMA (permeable-pavement-specific k_therm, rho_cp)
 * and latent heat pathway (LE_perm → LE_latent). Soil moisture bucket
 * depleted by evaporation; initially filled from per-cell CSV.
 */
enum class PermeableRoadMode
{
    Off = 0,      ///< No permeable pavement (backward-compat default)
    Simple = 1    ///< Soil conduction + moisture bucket + evaporation
};

/**
 * @enum InterfaceMode
 * @brief Flux interface mode for urban/non-urban blending (Phase 5.6)
 *
 * Determines whether MOST and UCM fluxes are gated by a binary is_urban mask
 * (Phase 4.1 sharp interface) or blended using fractional urban coverage f_urb.
 */
enum class InterfaceMode
{
    Binary = 0,   ///< Phase 4.1 sharp mask (default): MOST where is_urban=0, UCM where is_urban=1
    Blended = 1   ///< Phase 5.6 f_urb blending: MOST scaled by (1-f_urb), UCM scaled by f_urb
};

/**
 * @enum TreeDragMode
 * @brief Tree canopy drag application mode for momentum coupling (Phase 6.1)
 *
 * Determines whether tree canopy drag is applied to atmospheric momentum.
 * Phase 6.1 supports: Off (default, bit-identical to pre-Phase-6.1),
 * and Explicit (apply drag directly to ATM momentum RHS).
 */
enum class TreeDragMode
{
    Off = 0,       ///< Phase 6.1 default: no tree drag (bit-identical to pre-Phase-6.1 output)
    Explicit = 1   ///< Phase 6.1: apply tree drag directly to xmom_src and ymom_src
};

/**
 * @enum TreeRadMode
 * @brief Tree canopy radiation attenuation mode for SW coupling (Phase 6.2a)
 *
 * Determines whether Beer-Lambert SW attenuation through tree crown is applied.
 * Phase 6.2a supports: Off (default, bit-identical to pre-Phase-6.2a),
 * and BeerLambert (apply SW attenuation to facet SEB inputs).
 */
enum class TreeRadMode
{
    Off = 0,           ///< Phase 6.2a default: no tree radiation (bit-identical to pre-Phase-6.2a output)
    BeerLambert = 1    ///< Phase 6.2a: apply Beer-Lambert SW attenuation to roof/wall/road SEB inputs
};

/**
 * @enum SEBMode
 * @brief Surface Energy Balance solver mode for urban facets (Phase 6.2b)
 *
 * Determines whether to use the legacy 3-facet Newton solver (roof/wall/road) or
 * the new 4-facet solver that adds crown as a separate prognostic temperature.
 * Phase 6.2b supports: ThreeVar (default, bit-identical to pre-Phase-6.2b),
 * and FourVar (new crown facet with 4×4 Jacobian solver).
 */
enum class SEBMode
{
    ThreeVar = 0,      ///< Phase 6.2b default: 3-facet legacy solver (bit-identical to pre-Phase-6.2b output)
    FourVar = 1        ///< Phase 6.2b: 4-facet solver with prognostic T_crown
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
    // Section 17: Phase 5.2 — AC waste heat & building energy (Contract #21)
    // =========================================================================
    // HVAC waste heat coupled to SEB via cooling-load proportionality.
    // Reuses Phase 2.3 AH injection path. New CSV: hvac.csv (per-profile
    // COP, setpoint, occupancy). Default 'off' for backward-compat.

    std::string hvac_mode_str = "off";               ///< "off" | "simple"
    HVACMode hvac_mode = HVACMode::Off;
    std::string hvac_csv_path = "";                  ///< path to hvac.csv
    std::string occupancy_csv_path = "";             ///< path to occupancy.csv
    amrex::Real hvac_hysteresis_K = 2.0;             ///< setpoint hysteresis
    amrex::Real hvac_cop_default = 3.0;              ///< fallback COP if CSV missing
    amrex::Real hvac_setpoint_default_K = 297.15;    ///< 24 °C default
    amrex::Real hvac_cop_degradation_per_K = 0.03;   ///< Phase 5.5: COP reduction per K above T_rated [-/K]

    // =========================================================================
    // Section 18: Phase 5.3 — Green roofs, cool roofs, permeable pavements (Contract #21 extended)
    // =========================================================================
    // Green roof: soil layer on roof, reuses slab conduction + LE_latent path.
    // Permeable pavement: soil-water bucket on road, modified conduction.
    // Cool roof: CSV knob only via albedo/emissivity (no new code).
    // All default OFF for backward-compat bit-identity.

    std::string cool_roof_mode_str = "off";          ///< "off" | "recipe-only" (documentation mode)
    CoolRoofMode cool_roof_mode = CoolRoofMode::Off;

    std::string green_roof_mode_str = "off";         ///< "off" | "simple"
    GreenRoofMode green_roof_mode = GreenRoofMode::Off;
    amrex::Real green_roof_r_stomatal_s_per_m = 100.0;  ///< bulk stomatal resistance [s/m]
    amrex::Real green_roof_thickness_m = 0.1;        ///< soil layer depth [m]
    amrex::Real green_roof_k_therm = 0.5;            ///< thermal conductivity [W/m/K]
    amrex::Real green_roof_rho_cp = 1.8e6;           ///< volumetric heat capacity [J/m³/K]
    amrex::Real green_roof_soil_capacity_m = 0.10;   ///< water bucket capacity [m H2O]

    std::string permeable_road_mode_str = "off";     ///< "off" | "simple"
    PermeableRoadMode permeable_road_mode = PermeableRoadMode::Off;
    amrex::Real permeable_road_thickness_m = 0.2;    ///< soil layer depth [m]
    amrex::Real permeable_road_k_therm = 0.8;        ///< thermal conductivity [W/m/K]
    amrex::Real permeable_road_rho_cp = 1.6e6;       ///< volumetric heat capacity [J/m³/K]
    amrex::Real permeable_road_soil_capacity_m = 0.20; ///< water bucket capacity [m H2O]

    // =========================================================================
    // Section 19: Phase 5.6 — Fractional urban coverage (f_urb) blending (Contract #10 revised)
    // =========================================================================
    // Controls how MOST and UCM fluxes are combined at the ATM/UCM interface.
    // 'binary' (default, Phase 4.1): Sharp mask—MOST at is_urban=0, UCM at is_urban=1.
    // 'blended' (Phase 5.6): Fractional blending—MOST scaled by (1-f_urb), UCM by f_urb.
    // Both f_urb_atm and is_urban_atm are pre-computed; mode selects which is used.

    std::string interface_mode_str = "binary";       ///< "binary" | "blended"
    InterfaceMode interface_mode = InterfaceMode::Binary;

    // =========================================================================
    // Section 20: Phase 6.1 — Tree canopy layout + drag
    // =========================================================================
    // Tree morphology is stored as 2D UCM MultiFabs (H_tree, H_crown_base, LAD_bulk,
    // crown_area_frac, Cd_leaf, is_tree). ATM-grid aggregates (m_ucm_is_tree_atm, etc.)
    // are ERF class members allocated in the same site as m_ucm_is_urban_atm.
    // Tree drag is gated by tree_drag_mode; default 'off' ensures bit-identity when disabled.

    std::string tree_drag_mode_str = "off";          ///< Phase 6.1: "off" | "explicit"
    TreeDragMode tree_drag_mode = TreeDragMode::Off;
    std::string tree_layout_csv_path = "";           ///< Phase 6.1: path to tree_layout.csv (required if tree_drag_mode=explicit)
    amrex::Real Cd_leaf_default   = 0.2;             ///< Phase 6.1: default leaf drag coefficient [0.1–0.3]; Katul et al. 2004 [-]
    amrex::Real LAD_max_bound     = 5.0;             ///< Phase 6.1: sanity bound for LAD [m^2/m^3]
    amrex::Real H_tree_max_bound  = 40.0;            ///< Phase 6.1: sanity bound for tree height [m]

    // =========================================================================
    // Section 19: Phase 6.2a — Tree radiation (Beer-Lambert SW attenuation)
    // =========================================================================
    // Beer-Lambert attenuation of SW through tree crown; applied to SEB facet inputs.
    // Tree radiation is gated by tree_rad_mode; default 'off' ensures bit-identity
    // when disabled. Contract #28: double-gate: if tree_drag_mode=off, is_tree is
    // zero everywhere, so attenuation is no-op even if tree_rad_mode=beer_lambert.

    std::string tree_rad_mode_str = "off";           ///< Phase 6.2a: "off" | "beer_lambert"
    TreeRadMode tree_rad_mode = TreeRadMode::Off;
    amrex::Real k_ext_tree = 0.5;                    ///< Phase 6.2a: Beer-Lambert extinction coefficient (Campbell & Norman 1998) [-]

    // =========================================================================
    // Section 20: Phase 6.2b — Crown SEB facet (4-var Newton)
    // =========================================================================
    // Crown as a separate 4th facet in surface energy balance. Semi-implicit coupling
    // to existing roof/wall/road facets. Default 3-var mode (backward-compat, bit-identical).
    // T_crown MF allocated ONLY in 4-var mode. Crown physics: SW absorption + 2-sided LW.
    // Contract #29: 3-var solver unchanged; 4-var is separate standalone solver.
    // Contract #30: T_crown MF allocated ONLY in 4-var mode; nullptr in 3-var.
    // Contract #31: Semi-implicit lagged T_crown coupling to existing facets.
    // Contract #32: No LW attenuation through crown medium in 6.2b (deferred to 6.2c).

    SEBMode      seb_mode = SEBMode::ThreeVar;
    std::string  seb_mode_str = "3var";
    amrex::Real  Ch_leaf = 0.02;                     ///< Phase 6.2b: leaf drag coefficient [-], typical 0.01–0.03
    amrex::Real  eps_leaf = 0.96;                    ///< Phase 6.2b: leaf longwave emissivity [-]
    amrex::Real  crown_view_factor = 0.3;            ///< Phase 6.2b: crown→canyon geometric view factor [-]

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
