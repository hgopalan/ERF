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

    // =========================================================================
    // Section 4: Vertical structure (slab conduction, Phase 1.3+)
    // =========================================================================

    amrex::Real alpha_ucm      = 10.0;      ///< Exponential injection e-folding depth [m]

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
    std::string ucm_diag_file  = "ucm_diag.dat"; ///< Diagnostics CSV filename

    // =========================================================================
    // Section 9: Test placeholders (Phase 1.1 only; replaced Phase 1.3+)
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
