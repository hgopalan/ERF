/**
 * @file ERF_UCMFields.H
 * @brief SLUCM MultiFab field container
 *
 * Declares all per-cell URBPARM parameters and state variables as MultiFabs
 * on the UCM grid. Phase 1.2 populates homogeneous fields from ParmParse.
 * Phase 2.1 reads per-cell heterogeneous values from CSV.
 *
 * All MultiFabs are allocated with ghost cells IntVect(1, 1, 0) for efficient
 * inter-rank communication during coupling and flux calculations.
 *
 * References:
 *  - Source/Dust/ERF_DustLayer.H
 *  - Source/LNG/LNG*.H (field container pattern)
 */

#ifndef ERF_UCM_FIELDS_H
#define ERF_UCM_FIELDS_H

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <memory>

/**
 * @struct UCMFields
 * @brief Container for all SLUCM 2D MultiFab fields on the UCM grid
 *
 * All fields are allocated on the UCM BoxArray with ghost cells IntVect(1,1,0)
 * and ncomp=1 (or ncomp=N_layers for slab temperature fields). Homogeneous fill 
 * in Phase 1.2; heterogeneous CSV fill Phase 2.1.
 *
 * Phase 1.2 allocates and fills all fields with uniform values from UCMParams.
 * Future phases will populate spatially-varying morphology and state.
 */
struct UCMFields
{
    // =========================================================================
    // Building morphology (uniform in Phase 1.2, heterogeneous Phase 2.1+)
    // =========================================================================

    /// Uniform building height [m]. Determines canyon depth and wind profile.
    std::unique_ptr<amrex::MultiFab>  H_bldg;

    /// Road width (canyon street width) [m]. Determines floor area and albedo weighting.
    std::unique_ptr<amrex::MultiFab>  W_road;

    /// Roof width (building footprint) [m]. Determines roof area and albedo weighting.
    std::unique_ptr<amrex::MultiFab>  W_roof;

    // =========================================================================
    // Shortwave albedo (solar reflectance)
    // =========================================================================

    /// Roof shortwave albedo [-]. Wavelength-independent gray surface (Phase 1.4+).
    std::unique_ptr<amrex::MultiFab>  albedo_roof;

    /// Wall shortwave albedo [-]. Canyon-facing vertical surface.
    std::unique_ptr<amrex::MultiFab>  albedo_wall;

    /// Road shortwave albedo [-]. Canyon-floor surface.
    std::unique_ptr<amrex::MultiFab>  albedo_road;

    // =========================================================================
    // Longwave emissivity (thermal radiation)
    // =========================================================================

    /// Roof longwave emissivity [-]. Typical urban: 0.85-0.95.
    std::unique_ptr<amrex::MultiFab>  emissivity_roof;

    /// Wall longwave emissivity [-]. Typical urban: 0.85-0.95.
    std::unique_ptr<amrex::MultiFab>  emissivity_wall;

    /// Road longwave emissivity [-]. Asphalt/concrete: 0.90-0.95.
    std::unique_ptr<amrex::MultiFab>  emissivity_road;

    // =========================================================================
    // Surface temperatures (allocated Phase 1.2, filled Phase 1.3 SEB)
    // =========================================================================

    /// Roof skin temperature [K]. Boundary condition for slab conduction.
    /// Phase 1.2: initialized to test_surf_temp_K; Phase 1.3: from energy balance.
    std::unique_ptr<amrex::MultiFab>  T_skin_roof;

    /// Wall skin temperature [K]. Canyon vertical surface temperature.
    /// Phase 1.2: initialized to test_surf_temp_K; Phase 1.3: from energy balance.
    std::unique_ptr<amrex::MultiFab>  T_skin_wall;

    /// Road skin temperature [K]. Canyon floor temperature.
    /// Phase 1.2: initialized to test_surf_temp_K; Phase 1.3: from energy balance.
    std::unique_ptr<amrex::MultiFab>  T_skin_road;

    /// Canyon air temperature [K]. Prognostic variable driven by SEB and mixing.
    /// Phase 1.2: initialized to test_surf_temp_K; Phase 1.3: time-integrated.
    std::unique_ptr<amrex::MultiFab>  T_canyon_air;

    // =========================================================================
    // Slab temperature profiles (Phase 3.5A NEW; multi-layer thermal conduction)
    // =========================================================================

    /// Roof slab temperature profile [K]. ncomp = N_layers.
    /// Phase 3.5A: multi-layer representation (comp 0 = top/surface, comp N_layers-1 = deep).
    /// Boundary conditions: T_skin_roof at comp=0, T_deep at comp=N_layers-1.
    std::unique_ptr<amrex::MultiFab>  T_slab_roof;

    /// Wall slab temperature profile [K]. ncomp = N_layers.
    /// Phase 3.5A: multi-layer representation.
    std::unique_ptr<amrex::MultiFab>  T_slab_wall;

    /// Road slab temperature profile [K]. ncomp = N_layers.
    /// Phase 3.5A: multi-layer representation.
    std::unique_ptr<amrex::MultiFab>  T_slab_road;

    // =========================================================================
    // Sensible and latent heat fluxes (Phase 1.3+ SEB products)
    // =========================================================================

    /// Net sensible heat flux from canyon to atmosphere [W/m²].
    /// Phase 1.2: initialized to 0.0; Phase 1.3: from SEB.
    std::unique_ptr<amrex::MultiFab>  H_sensible;

    /// Net latent heat flux from canyon to atmosphere [W/m²].
    /// Phase 1.2: initialized to 0.0; Phase 1.3: from evaporation/transpiration.
    std::unique_ptr<amrex::MultiFab>  LE_latent;

    // =========================================================================
    // Urban mask (Phase 1.2 NEW; LSM/MOST bypass Phase 4.1)
    // =========================================================================

    /// Urban/non-urban mask [0 or 1]. 1 = UCM owns this cell (full urban patch).
    /// 0 = LSM/MOST owns this cell (non-urban, Phase 2.1+).
    /// Phase 1.2: set to 1 everywhere (homogeneous urban patch).
    /// Phase 4.1: heterogeneous LSM/MOST bypass wired.
    std::unique_ptr<amrex::iMultiFab> is_urban;

    // =========================================================================
    // Per-cell material IDs (Phase 2.1 NEW; index into UCMMaterialRegistry)
    // =========================================================================

    /// Material ID for roof surface (index into UCMMaterialRegistry).
    /// Phase 2.1: filled from building_layout.csv roof_mat_id field.
    /// Phase 1.2 (homogeneous fallback): set to default material ID (typically 0 or 1).
    std::unique_ptr<amrex::iMultiFab> mat_id_roof;

    /// Material ID for wall surface (index into UCMMaterialRegistry).
    /// Phase 2.1: filled from building_layout.csv wall_mat_id field.
    /// Phase 1.2 (homogeneous fallback): set to default material ID.
    std::unique_ptr<amrex::iMultiFab> mat_id_wall;

    /// Material ID for road surface (index into UCMMaterialRegistry).
    /// Phase 2.1: filled from building_layout.csv road_mat_id field.
    /// Phase 1.2 (homogeneous fallback): set to default material ID.
    std::unique_ptr<amrex::iMultiFab> mat_id_road;

    // =========================================================================
    // Per-cell thermal properties (Phase 2.2 NEW; material-dependent)
    // =========================================================================

    /// Roof thermal conductivity [W/m/K]. Phase 2.2: filled from material registry.
    /// Phase 1.2 (homogeneous fallback): set to k_therm_uniform.
    std::unique_ptr<amrex::MultiFab> k_therm_roof;

    /// Wall thermal conductivity [W/m/K]. Phase 2.2: filled from material registry.
    /// Phase 1.2 (homogeneous fallback): set to k_therm_uniform.
    std::unique_ptr<amrex::MultiFab> k_therm_wall;

    /// Road thermal conductivity [W/m/K]. Phase 2.2: filled from material registry.
    /// Phase 1.2 (homogeneous fallback): set to k_therm_uniform.
    std::unique_ptr<amrex::MultiFab> k_therm_road;

    /// Roof volumetric heat capacity [J/m³/K]. Phase 2.2: from material registry.
    /// Phase 1.2 (homogeneous fallback): set to rho_cp_uniform.
    std::unique_ptr<amrex::MultiFab> rho_cp_roof;

    /// Wall volumetric heat capacity [J/m³/K]. Phase 2.2: from material registry.
    /// Phase 1.2 (homogeneous fallback): set to rho_cp_uniform.
    std::unique_ptr<amrex::MultiFab> rho_cp_wall;

    /// Road volumetric heat capacity [J/m³/K]. Phase 2.2: from material registry.
    /// Phase 1.2 (homogeneous fallback): set to rho_cp_uniform.
    std::unique_ptr<amrex::MultiFab> rho_cp_road;

    /// Roof slab thickness [m]. Phase 2.2: from material registry.
    /// Phase 1.2 (homogeneous fallback): set to slab_L.
    std::unique_ptr<amrex::MultiFab> slab_L_roof;

    /// Wall slab thickness [m]. Phase 2.2: from material registry.
    /// Phase 1.2 (homogeneous fallback): set to slab_L.
    std::unique_ptr<amrex::MultiFab> slab_L_wall;

    /// Road slab thickness [m]. Phase 2.2: from material registry.
    /// Phase 1.2 (homogeneous fallback): set to slab_L.
    std::unique_ptr<amrex::MultiFab> slab_L_road;

    /// Aerodynamic roughness length [m]. Phase 2.2: from z0 = z0_over_H * H_bldg.
    /// Phase 1.2 (homogeneous fallback): set to 0.1 m (MOST default).
    std::unique_ptr<amrex::MultiFab> z0_ucm;

    /// Displacement height [m]. Phase 2.2: from d = d_over_H * H_bldg.
    /// Phase 1.2 (homogeneous fallback): set to 0.0 m.
    std::unique_ptr<amrex::MultiFab> d_disp_ucm;

    // =========================================================================
    // Phase 2.3: Facet-split sensible heat and anthropogenic heat
    // =========================================================================

    /// Road sensible heat flux [W/m²]. Phase 2.3: facet split from H_sensible.
    std::unique_ptr<amrex::MultiFab> H_road;

    /// Wall sensible heat flux [W/m²]. Phase 2.3: facet split from H_sensible.
    std::unique_ptr<amrex::MultiFab> H_wall;

    /// Roof sensible heat flux [W/m²]. Phase 2.3: facet split from H_sensible.
    /// Phase 3.5A-hotfix2: Newton-derived for slab conduction consistency.
    std::unique_ptr<amrex::MultiFab> H_roof;

    /// Phase 3.5A-hotfix2: Atmospheric injection heat fluxes (MOST-derived).
    /// These are used for ATM injection and are computed from MOST-derived u_star*t_star.
    /// Separate from H_roof/wall/road which are Newton-derived for slab conduction.
    /// Roof sensible heat flux for ATM injection [W/m²]. Phase 3.5A-hotfix2: MOST-derived.
    std::unique_ptr<amrex::MultiFab> H_roof_atm;

    /// Road sensible heat flux for ATM injection [W/m²]. Phase 3.5A-hotfix2: MOST-derived.
    std::unique_ptr<amrex::MultiFab> H_road_atm;

    /// Wall sensible heat flux for ATM injection [W/m²]. Phase 3.5A-hotfix2: MOST-derived.
    std::unique_ptr<amrex::MultiFab> H_wall_atm;

    /// Anthropogenic heat flux [W/m²]. Phase 2.3: added to roof; distributed by Profile.
    std::unique_ptr<amrex::MultiFab> AH;

    /// Per-cell anthropogenic heat override [W/m²]. Phase 2.9: from CSV; 0 → use ParmParse fallback.
    std::unique_ptr<amrex::MultiFab> AH_Wm2_ucm;

    /// Plan-area fraction (building footprint / grid cell area) [-]. Phase 2.3: needed for facet split.
    std::unique_ptr<amrex::MultiFab> plan_area_frac;

    /// Anthropogenic heat profile selector [0=uniform, 1=diurnal]. Phase 2.3: per-cell from CSV.
    std::unique_ptr<amrex::iMultiFab> ah_profile_id;

    // =========================================================================
    // Phase 2.4: Sky view factors (shadowing model)
    // =========================================================================

    /// Sky view factor for wall surface [-]. Phase 2.4: from canyon aspect ratio (Kusaka 2001).
    /// Reduces incoming shortwave on canyon walls due to building self-shading.
    /// Computed from H_bldg / max(W_road, 1.0e-6); typical range: 0.3-0.8.
    std::unique_ptr<amrex::MultiFab> SVF_wall;

    /// Sky view factor for road surface [-]. Phase 2.4: from canyon aspect ratio (Kusaka 2001).
    /// Reduces incoming shortwave on canyon floor due to building overhead obstruction.
    /// Computed from H_bldg / max(W_road, 1.0e-6); typical range: 0.3-0.8.
    std::unique_ptr<amrex::MultiFab> SVF_road;

    /// Sky view factor for roof surface [-]. Phase 2.4: always 1.0 (unshaded from above).
    /// Included for symmetry; not used in SEB (roof always sees full SW_down).
    std::unique_ptr<amrex::MultiFab> SVF_roof;

    // =========================================================================
    // Public methods
    // =========================================================================

    /**
     * @brief Check that all MultiFabs are allocated
     *
     * Returns true if every field pointer is non-null (all allocated).
     * Returns false and prints diagnostic messages if any pointer is null.
     *
     * Called after allocate_ucm_fields() and in prerequisites checks.
     *
     * @return true if all fields allocated; false if any is missing
     */
    bool all_allocated() const;
};

#endif // ERF_UCM_FIELDS_H