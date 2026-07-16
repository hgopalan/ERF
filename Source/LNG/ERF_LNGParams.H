/**
 * @file ERF_LNGParams.H
 * @brief LNG hazardous gas dispersion module parameters
 * @details
 * This header defines the LNGParams struct, which holds all configuration parameters
 * for the ERF-LNG module (liquefied natural gas spill and vaporization modeling).
 * All parameters are read from the ParmParse namespace `erf.lng` during initialization.
 * 
 * The parameters span eight implementation phases:
 * - Phase 1: Enable/disable, debug flags, LNG properties (boiling point, density, mol. weight)
 * - Phase 2: Evaporation thermodynamics and pool spreading
 * - Phase 3: Atmospheric feedback and coupling
 * - Phase 4: Wind field extraction and PBL interaction
 * - Phase 5: Flammability zone tracking
 * - Phase 6: Concentration output and receptor sampling
 * - Phase 7: Regulatory threshold post-processing (NFPA 59A)
 * - Phase 8: Spill schedule and inventory tracking
 * 
 * @note Phase 1 is a zero-physics stub; all parameters are registered but only
 *       subset are actively used (enable, lng_debug, verbose, and pool geometry).
 *       Subsequent phases fill in physical algorithms.
 * 
 * @see Source/Dust/ERF_DustParams.H — Dust module parameter pattern (reference)
 * @see NFPA 59A — Recommended Practice for the Siting of Liquefied Natural Gas Facilities
 * @see Koopman, R.P., 1982: "Burro LNG spill test series final report" NTIS
 */

#ifndef ERF_LNG_PARAMS_H
#define ERF_LNG_PARAMS_H

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <string>

/**
 * @struct LNGParams
 * @brief Configuration container for LNG module
 * @details
 * Holds all user-configurable parameters for LNG dispersion modeling.
 * Parameters are read from the ParmParse namespace `erf.lng.*` in the constructor.
 * All fields have default values suitable for a 100 m^2 spill of typical LNG.
 */
struct LNGParams {
    /**
     * @name Enable/Debug Control
     * @{
     */
    
    /// Enable the LNG module (true = active, false = no LNG simulation)
    /// @details Allows toggling module on/off without code changes.
    ///          When false, zero overhead is incurred in the main ERF loop.
    bool enable = false;
    
    /// Write detailed per-timestep debug fields to lng_debug.cpp
    /// @details If true, activates verbose per-step debug output and CSV row writes.
    ///          Also triggers NaN checking after every advance() call.
    bool lng_debug = false;
    
    /// Verbosity level for LNG module output
    /// @details
    ///   - 0: silent (only errors/aborts printed)
    ///   - 1: initialization summary (pool area, spill rate, composition, grid info)
    ///   - 2: per-step one-liner (pool cells, max evaporation flux, max vapor concentration)
    ///   - 3: per-step MultiFab min/max for all fields (verbose=2 + detailed field statistics)
    int verbose = 0;
    
    /** @} */
    
    /**
     * @name LNG Physical Properties
     * @{
     */
    
    /// Effective molecular weight of LNG mixture [g/mol]
    /// @details Typical LNG is primarily CH4 (16.04 g/mol) with ~10% ethane (30.07 g/mol).
    ///          Default 17.4 g/mol is Burro/Coyote test LNG blend.
    ///          Phase 2: used in Clausius-Clapeyron and vapor density calculations.
    amrex::Real mol_weight_LNG = 17.4;
    
    /// Molar fraction of CH4 in LNG [-]
    /// @details Typically 0.85–0.95 for pipeline LNG; Burro used ~0.90.
    ///          Used for molecular weight averaging (Phase 2) and composition reporting.
    amrex::Real ch4_mole_fraction = 0.90;
    
    /// Molar fraction of C2H6 (ethane) in LNG [-]
    /// @details Typically 0.02–0.10; heavier components increase evaporation enthalpy slightly.
    amrex::Real c2h6_mole_fraction = 0.08;
    
    /// Molar fraction of N2 (nitrogen) in LNG [-]
    /// @details Inert diluent; typically 0.00–0.05. Remainder to sum ≈ 1.0 is other species.
    amrex::Real n2_mole_fraction = 0.02;
    
    /// Liquid LNG density at normal boiling point and 1 atm [kg/m^3]
    /// @details Typical range 420–450 kg/m^3. Burro 424 kg/m^3.
    ///          Phase 2: scales pool volume to mass and spreads pool radially.
    amrex::Real rho_LNG = 425.0;
    
    /// Normal boiling point of LNG at 1 atm [K]
    /// @details Approximate average of CH4 (111.6 K) and C2H6 (184.6 K).
    ///          Burro measured 111.7 K at 1.0 bar. Used as reference for evaporation thermodynamics.
    amrex::Real lng_boil_temp_K = 111.7;
    
    /// Latent heat of vaporization of LNG [J/kg]
    /// @details CH4 at 111.6 K: ~510 kJ/kg. Phase 2: scales heat requirement for evaporation.
    ///          Also determines sensible heat needed to warm vapor from pool to air temperature.
    amrex::Real Hv_LNG = 509000.0;
    
    /// Density of LNG vapor at boiling point and 1 atm [kg/m^3]
    /// @details Burro measurement ~1.76 kg/m^3 at saturation. Phase 3: used for mixture density.
    amrex::Real rho_vapor_ref = 1.76;
    
    /// Specific heat capacity of liquid LNG [J/kg/K]
    /// @details Typical ~3.5 kJ/kg/K. Phase 2: sensible heat from ground/rain cooling.
    amrex::Real Cp_LNG = 3500.0;
    
    /** @} */
    
    /**
     * @name Flammability Thresholds
     * @{
     */
    
    /// Lower flammability limit (LFL) of LNG vapor in air [vol/vol]
    /// @details NFPA 59A standard: 5.0 vol% for natural gas. 
    ///          Phase 5: concentration > LFL defines inner flammability zone.
    amrex::Real lfl_vol_fraction = 0.05;
    
    /// Upper flammability limit (UFL) of LNG vapor in air [vol/vol]
    /// @details NFPA 59A standard: 15 vol% for natural gas.
    ///          Phase 5: concentration between LFL and UFL is flammable.
    amrex::Real ufl_vol_fraction = 0.15;
    
    /// Enable computation of flammable concentration map [boolean]
    /// @details If true, phase 5+ tracks cells where conc. is in [LFL, UFL] range
    ///          and writes flammability zone area to diagnostics.
    bool track_flammability = false;
    
    /** @} */
    
    /**
     * @name Phase 5: Gravity Current & Richardson Transition
     * @{
     */
    
    /// Enable 2D shallow-water gravity current PDEs [boolean]
    /// @details If true, Phase 5 solves for depth-averaged velocity and depth on 2D LNG grid.
    ///          Richardson number Ri = g'*h / u*^2 controls 2D→3D handoff to ERF solver.
    ///          Phase 5: core dense gas near-field model (replaces DEGADIS).
    bool enable_gravity_current = false;
    
    /// Surface drag coefficient for gravity current [-]
    /// @details Typically 1e-3 to 5e-3. Default 2e-3 from Webber & Brighton (1987).
    ///          Used in momentum equations: -Cd*u*|u|. Phase 5: tuned for LNG.
    amrex::Real gc_drag_coeff = 2.0e-3;
    
    /// Richardson number transition threshold [-]
    /// @details When Ri exceeds this, gravity current regime is active.
    ///          When Ri < threshold, transition to well-mixed (3D ERF dispersion).
    ///          Default 0.25 from Benjamin (1968). Phase 5: used for automatic handoff.
    amrex::Real gc_ri_crit = 0.25;
    
    /** @} */
    
    /**
     * @name Pool Geometry and Release Schedule
     * @{
     */
    
    /// Initial area of LNG pool [m^2]
    /// @details Typical spill sizes 10–10,000 m^2. Default 100 m^2 is modest test case.
    ///          Phase 1: used to set initial pool_mask. Converted to radius = sqrt(area/π).
    amrex::Real pool_area_m2 = 100.0;
    
    /// Constant LNG release rate [kg/s]
    /// @details If > 0, a continuous source at this rate. If 0, no release (spill event only).
    ///          Phase 8: time-dependent release read from spill_schedule_file instead.
    amrex::Real spill_rate_kg_s = 0.0;
    
    /// CSV file specifying time-dependent spill release schedule [string]
    /// @details Format: time_s cx cy radius_m rate_kg_s (one row per time). Empty = disabled.
    ///          Phase 8: overrides pool_area_m2 and spill_rate_kg_s if provided.
    std::string spill_schedule_file = "";
    
    /// Initial depth of LNG pool [m]
    /// @details Typical range 0.01–0.5 m for spreading scenarios. Phase 1: used as initial condition.
    ///          Phase 2: updated as pool spreads and evaporates.
    amrex::Real pool_depth_init_m = 0.01;
    
    /** @} */
    
    /**
     * @name Grid and Coupling Configuration
     * @{
     */
    
    /// Refinement ratio of LNG grid relative to atmospheric level-0 grid [-]
    /// @details E.g., grid_ratio=4 creates 4×4 LNG cells per ATM cell in x,y.
    ///          Used to resolve thin LNG pool. Phase 1: validates divisibility.
    ///          Phase 3: couples via coarsening (average_down).
    int grid_ratio = 1;
    
    /// Coupling strength for atmospheric feedback [0, 1]
    /// @details 0 = LNG is passive (advected only), 1 = full two-way feedback.
    ///          Phase 3: scales latent heat injection to ATM sensible heat source.
    amrex::Real atm_feedback = 1.0;
    
    /// Reference height for wind extraction [m]
    /// @details Typical meteorological reference: 10 m. 
    ///          Phase 4: wind at zref is interpolated from ATM field and used for evaporation.
    amrex::Real zref = 10.0;
    
    /// Aerodynamic roughness length over LNG pool [m]
    /// @details Used in mass transfer coefficient calculation: k_mass ~ u* / ln(z_ref/z0).
    ///          Brighton (1990): smooth liquid surfaces ~0.01 m. Typical range 0.001–0.1 m.
    ///          Phase 2: directly affects evaporation flux magnitude.
    amrex::Real z0_lng = 0.01;
    
    /** @} */
    
    /**
     * @name Placeholder Test Fields (Phase 1 only)
     * @{
     */
    
    /// Placeholder 10-meter wind speed [m/s]
    /// @details Used in Phase 1 stubs; replaced by ATM wind extraction in Phase 4.
    amrex::Real test_wind_speed = 3.0;
    
    /// Placeholder surface temperature [K]
    /// @details Used in Phase 1 stubs; replaced by ATM surface layer data in Phase 3.
    amrex::Real test_surf_temp_K = 293.15;
    
    /// Placeholder friction velocity [m/s]
    /// @details Used in Phase 1 stubs; replaced by ATM surface layer u* in Phase 4.
    amrex::Real test_ustar = 0.0;
    
    /** @} */
    
    /**
     * @name Atmospheric Reference Values
     * @{
     */
    
    /// Air density at standard atmosphere [kg/m^3]
    /// @details 1.225 kg/m^3 at 15°C, 1 atm. Used for buoyancy and Froude number (Phase 2).
    amrex::Real rho_air = 1.225;
    
    /** @} */
    
    /**
     * @name Output and Diagnostics
     * @{
     */
    
    /// Plotfile output interval [steps]
    /// @details If > 0, write LNG plotfiles every N steps. -1 = disabled (Phase 1).
    ///          Phase 6: implements actual VisMF plotfile writes.
    int lng_plot_int = -1;
    
    /// Prefix for LNG plotfile names [string]
    /// @details E.g., "plt_lng_" produces "plt_lng_00000", "plt_lng_00001", etc.
    std::string lng_plot_prefix = "plt_lng_";
    
    /// CSV filename for time-series diagnostics [string]
    /// @details Written to run directory. Columns: step, time, evaporation_rate, T_min, conc_max, etc.
    ///          Phase 1: header only. Phases 2+: populated with actual diagnostics.
    std::string lng_diag_file = "lng_diag.csv";
    
    /// CSV file for receptor point concentration sampling [string]
    /// @details Format: name x y z (one row per receptor). Empty = disabled (Phase 6).
    std::string lng_receptor_file = "";
    
    /** @} */
    
    /**
     * @name Phase 6: Receptor Sampling
     * @{
     */
    
    /// Receptor point names [vector of strings]
    /// @details Human-readable identifiers (e.g., "center", "downwind", "northeast").
    ///          Phase 6: one CSV file per receptor with time-series concentration samples.
    amrex::Vector<std::string> lng_receptor_names;
    
    /// Receptor point x-coordinates [m, vector of reals]
    /// @details Parallel to lng_receptor_names. Must have same length.
    amrex::Vector<amrex::Real> lng_receptor_x;
    
    /// Receptor point y-coordinates [m, vector of reals]
    /// @details Parallel to lng_receptor_names. Must have same length.
    amrex::Vector<amrex::Real> lng_receptor_y;
    
    /** @} */
    
    /**
     * @name Regulatory Thresholds
     * @{
     */
    
    /// NFPA 59A exclusion zone concentration (1/2 LFL) [vol/vol]
    /// @details Standard exclusion radius defined by concentration = 1/2 LFL (2.5 vol%).
    ///          Phase 7: used to report exclusion zone area and radius to regulatory output.
    amrex::Real nfpa59a_exclusion_conc = 0.025;
    
    /// CSV filename for regulatory threshold summary [string]
    /// @details Written by Phase 7. Columns: time, exclusion_radius, LFL_area, UFL_area, etc.
    std::string lng_regulatory_file = "lng_regulatory.csv";
    
    /** @} */
    
    /**
     * @brief Constructor: reads all parameters from ParmParse namespace `erf.lng`
     * @details
     * Invokes `ParmParse pp("erf.lng")` and calls `pp.query()` for every parameter.
     * All parameters retain their default values if not specified in the input file.
     * This ensures forward/backward compatibility as phases add new parameters.
     */
    LNGParams () {
        amrex::ParmParse pp("erf.lng");
        
        // Enable/Debug
        pp.query("enable", enable);
        pp.query("lng_debug", lng_debug);
        pp.query("verbose", verbose);
        
        // Physical properties
        pp.query("mol_weight_LNG", mol_weight_LNG);
        pp.query("ch4_mole_fraction", ch4_mole_fraction);
        pp.query("c2h6_mole_fraction", c2h6_mole_fraction);
        pp.query("n2_mole_fraction", n2_mole_fraction);
        pp.query("rho_LNG", rho_LNG);
        pp.query("lng_boil_temp_K", lng_boil_temp_K);
        pp.query("Hv_LNG", Hv_LNG);
        pp.query("rho_vapor_ref", rho_vapor_ref);
        pp.query("Cp_LNG", Cp_LNG);
        
        // Flammability
        pp.query("lfl_vol_fraction", lfl_vol_fraction);
        pp.query("ufl_vol_fraction", ufl_vol_fraction);
        pp.query("track_flammability", track_flammability);
        
        // Phase 5: Gravity current & Richardson transition
        pp.query("enable_gravity_current", enable_gravity_current);
        pp.query("gc_drag_coeff", gc_drag_coeff);
        pp.query("gc_ri_crit", gc_ri_crit);
        
        // Pool geometry and release
        pp.query("pool_area_m2", pool_area_m2);
        pp.query("spill_rate_kg_s", spill_rate_kg_s);
        pp.query("spill_schedule_file", spill_schedule_file);
        pp.query("pool_depth_init_m", pool_depth_init_m);
        
        // Grid and coupling
        pp.query("grid_ratio", grid_ratio);
        pp.query("atm_feedback", atm_feedback);
        pp.query("zref", zref);
        pp.query("z0_lng", z0_lng);
        
        // Test placeholders
        pp.query("test_wind_speed", test_wind_speed);
        pp.query("test_surf_temp_K", test_surf_temp_K);
        pp.query("test_ustar", test_ustar);
        
        // Atmospheric reference
        pp.query("rho_air", rho_air);
        
        // Output
         pp.query("lng_plot_int", lng_plot_int);
        pp.query("lng_plot_prefix", lng_plot_prefix);
        pp.query("lng_diag_file", lng_diag_file);
        pp.query("lng_receptor_file", lng_receptor_file);
        
        // Phase 6: Receptor sampling
        pp.queryarr("lng_receptor_names", lng_receptor_names);
        pp.queryarr("lng_receptor_x", lng_receptor_x);
        pp.queryarr("lng_receptor_y", lng_receptor_y);
        
        // Regulatory
        pp.query("nfpa59a_exclusion_conc", nfpa59a_exclusion_conc);
        pp.query("lng_regulatory_file", lng_regulatory_file);
    }
};

#endif /* ERF_LNG_PARAMS_H */
