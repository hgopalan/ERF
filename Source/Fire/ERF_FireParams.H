#ifndef ERF_FIRE_PARAMS_H
#define ERF_FIRE_PARAMS_H

#include <AMReX_ParmParse.H>
#include <ERF_FireBreak.H>
#include <ERF_FireAcceleration.H>
#include <string>
#include <vector>

/**
 * @file ERF_FireParams.H
 * @brief Fire simulation parameters container
 *
 * Loads fire configuration from ParmParse "erf.fire" prefix at runtime.
 * Stores all tunable parameters for the wildfire spread module including
 * fuel model selection, moisture conditions, wind reference height, and
 * wind adjustment factors.
 */

struct FireParams
{
    bool              enable              = false;       ///< Enable fire module
    bool              fire_debug          = false;       ///< Enable debug output for fire calculations
    int               grid_ratio          = 5;           ///< Fire grid refinement factor (C)
    int               fuel_model_id       = 1;           ///< Anderson FBFM13 model (1-13)
    amrex::Real       moisture_1hr        = 0.08;        ///< 1-hour fuel moisture [fraction]
    amrex::Real       moisture_10hr       = 0.08;        ///< 10-hour fuel moisture [fraction]
    amrex::Real       moisture_100hr      = 0.10;        ///< 100-hour fuel moisture [fraction]
    amrex::Real       ignition_x          = 0.0;         ///< Ignition center x [m]
    amrex::Real       ignition_y          = 0.0;         ///< Ignition center y [m]
    amrex::Real       ignition_r          = 20.0;        ///< Ignition radius [m]
    amrex::Real       wind_ref_ht         = 6.1;         ///< MOST wind reference height [m]
    // Phase 13A: Per-fuel wind height
    bool              use_per_fuel_wind_ht = false; ///< Use per-fuel fcwh table for wind height
    amrex::Real       waf_fcz0_scale       = 1.0;   ///< Scale factor on fcz0 roughness for WAF [dimensionless]
    bool              use_waf             = true;        ///< Apply Wind Adjustment Factor
    std::string       waf_formula         = "andrews";   ///< "andrews" or "behaviorplus"
    bool              use_wind_limit      = true;        ///< Apply MEWS cap to wind
    bool              use_terrain_wind    = true;        ///< Apply terrain corrections
    amrex::Real       k_ridge             = 1.5;         ///< Ridge speed-up factor
    amrex::Real       k_shelter           = 0.6;         ///< Sheltered wind factor
    amrex::Real       k_valley            = 0.8;         ///< Valley channeling factor
    amrex::Real       k_deflect           = 0.3;         ///< Wind deflection factor
    
    // Fuel moisture control (Phase 4)
    bool              moisture_dynamic    = true;        ///< Advance moisture each step via ODEs
    amrex::Real       precip_rate_mm_hr   = 0.0;         ///< Uniform precipitation rate [mm/hr]
    amrex::Real       moisture_live       = 0.60;        ///< Live fuel moisture [fraction]
    
    // Moisture of extinction (Phase 4)
    bool              use_dynamic_mext    = true;        ///< Compute M_x from fuel SAV
    
    // Heat flux parameters (Phase 5)
    amrex::Real       tau_residence_s     = 0.0;         ///< Residence time [s]; 0 = derive from fuel SAV
    amrex::Real       f_c_min             = 0.2;         ///< Minimum combustion fraction
    amrex::Real       f_c_max             = 0.9;         ///< Maximum combustion fraction
    
    // FARSITE parameters
    amrex::Real       farsite_phi_threshold   = 0.1;  ///< front detection level in phi
    int               farsite_use_anderson_lw = 1;       ///< derive coeff_a/b/c from Anderson L/W
    amrex::Real       farsite_coeff_a         = 0.5;  ///< Richards head coefficient
    amrex::Real       farsite_coeff_b         = 0.25; ///< Richards flank coefficient
    amrex::Real       farsite_coeff_c         = 0.1;  ///< Richards backing coefficient
    amrex::Real       farsite_gaussian_sigma  = -1.0; ///< phi stamping radius [m]
    amrex::Real       farsite_cfl_fire        = 0.5;  ///< fire CFL number

    // Level-set propagation parameters
    std::string       propagation_method      = "farsite";   ///< "farsite" (default, Lagrangian) or "levelset" (PDE-based)
    amrex::Real       levelset_cfl            = 0.4;   ///< CFL for level-set subcycle
    amrex::Real       levelset_eps_visc       = 0.4;   ///< Artificial viscosity coefficient ε
    int               levelset_reinit_every   = 5;     ///< Reinitialize φ every N fire subcycles
    int               levelset_reinit_iters   = 10;    ///< Sussman reinitialization iterations
    amrex::Real       levelset_reinit_band_m  = -1.0;  ///< Band half-width L [m] where |phi| reaches 1; <=0 selects 3*min(dx,dy)
    amrex::Real       levelset_reinit_dtau    = -1.0;  ///< Reinitialization pseudo-dt (<0 = auto = 0.5*min(dx,dy))
    // Extrapolate the level set into non-burnable cells inside every stencil
    // (gradient, Laplacian, front normal, reinitialisation), so a masked wall
    // is a zero-gradient boundary of the distance function rather than a
    // front. Without it the cells along a wall burn down at the rate the
    // one-sided difference into the wall dictates and the flank along it runs
    // about 20% fast. Off by default; needs a mask source to do anything.
    bool              levelset_wall_extrapolate = false;

    // Terrain file for fine-grid terrain data
    std::string       terrain_file_name       = "";   ///< Path to terrain file for fine-grid interpolation
    
    // CSV output for fire statistics (burned area, perimeter, etc.)
    bool              write_fire_stats_csv    = true; ///< Write burned area and perimeter CSV
    std::string       fire_stats_csv_file     = "fire_stats.csv"; ///< Fire statistics output filename
    
    // Phase 6: fire-atmosphere coupling parameters
    amrex::Real       heat_flux_alfg          = 45.0;  ///< e-folding height for sensible heat distribution [m]
    amrex::Real       fire_atm_feedback       = 1.0;   ///< multiplier on fire flux (0 = one-way, 1 = full)
    // Coupling mode between fire and atmosphere.
    //   "passive"      — fire heat computed but not injected into atmosphere
    //   "lagged"       — lagged coupling: fire heat from step n enters atmosphere
    //                    at step n+1 via RK source term (default, Phase 6 behaviour)
    //   "synchronous"  — synchronous coupling: fire advances using post-dycore wind
    //                    (vars_new), fire heat still enters atmosphere at step n+1
    std::string       coupling_type           = "lagged";
    bool              inject_latent           = true;  ///< inject latent heat alongside sensible (WRF-Fire parity)
    // How the fire tendency enters the cell source after make_sources has
    // rebuilt it for the Runge-Kutta stage. "overwrite" (default, legacy)
    // replaces the potential-temperature and vapour slots, which discards any
    // other source in them (immersed-forcing scalar relaxation, radiation,
    // Rayleigh damping). "add" accumulates into the freshly rebuilt source,
    // once per stage, so nothing is lost and nothing is double counted.
    std::string       source_mode             = "overwrite";
    // Place the injected heat only in the open part of each atmospheric column:
    // below the sampled roof height a cell's weight is scaled by its open
    // fraction (one minus the area covered by non-burnable structure cells),
    // and the column's total is renormalised so the injected energy is
    // unchanged. Needs erf.fire.structures.enable. Off by default.
    bool              heat_open_fraction      = false;
    // Tendency form. false (default): the energy-consistent
    // d(rho theta)/dt = -(1/Cp) dQ/dz, with which the coupling's energy
    // diagnostic reads 1 - exp(-z_top/alfg). true: the historical form with
    // an extra density factor, -rho (1/Cp) dQ/dz, which injects rho times the
    // fire flux (about 10-15% too much at sea level); kept to reproduce
    // coupled results from before 2026-09-03.
    bool              heat_tendency_density   = false;

    // Phase 8: Albini (1983) ember spotting parameters
    struct SpottingParams {
        bool              enable              = false;   ///< Master on/off switch
        amrex::Real       I_B_min             = 100.0;   ///< Minimum Byram intensity to launch [kW/m]
        amrex::Real       P_base              = 0.01;    ///< Base launch probability per front cell per step
        amrex::Real       terminal_velocity   = 0.5;     ///< Firebrand terminal fall velocity [m/s]
        int               n_traj_steps        = 20;      ///< Trajectory integration sub-steps
        amrex::Real       spot_radius         = 10.0;    ///< Ignition disk radius at landing [m]
        amrex::Real       P_catch             = 1.0;     ///< Catching probability at landing [0–1]
        amrex::Real       reentry_fuel_thresh = 0.05;    ///< Minimum residual fuel fraction for re-entry,
                                                       ///< as a fraction of the cell's initial load
        int               random_seed         = 0;       ///< 0 = time-based seed, >0 = fixed seed
        int               spotting_interval   = 1;       ///< Apply spotting every N fire subcycles
        std::string       fuel_system         = "13";    ///< "13" (FBFM13) or "40" (Scott-Burgan 40)
        // Which burned cells may launch a brand. "burned" (default, legacy):
        // every cell behind the front, so the brand count scales with the
        // burned area and P_base has to be retuned per domain. "front": only
        // burned cells with unburned fuel within front_band cells, the
        // fireline Albini's model assumes, so the count scales with its
        // length.
        std::string       launch_from         = "burned";
        int               front_band          = 2;       ///< "front": band width [fire cells]
    } spotting;

    // Phase 9: Crown fire initiation parameters.
    // Van Wagner (1977) surface-to-crown criterion + Cruz et al. (2005) active crown ROS.
    // Applicable to conifer forest stands. Not appropriate for grass or shrub fuels.
    struct CrownFireParams {
        bool          enable          = false;   ///< Enable crown fire initiation
        amrex::Real   canopy_base_ht  = 5.0;    ///< Canopy base height C_BH [m]
        amrex::Real   canopy_bulk_den = 0.2;    ///< Canopy bulk density C_BD [kg/m³]
        amrex::Real   canopy_depth    = 10.0;   ///< Canopy depth [m]
        amrex::Real   foliar_moisture = 1.0;    ///< Foliar moisture M_fold [fraction]
        amrex::Real   M_c             = 0.30;   ///< Critical foliar moisture [fraction] (Van Wagner 1977)
        amrex::Real   h_crown_BTU_lb  = 8000.0; ///< Canopy heat content [BTU/lb]
        amrex::Real   wind_10m_kmh    = 0.0;    ///< 10-m wind speed [km/h]; 0 = derive from fire_wind_eff
        std::string   ros_model       = "cruz"; ///< Crown ROS model: "cruz", "rothermel1991", "van_wagner_proxy"
        bool          use_passive_blend = false; ///< Use Van Wagner passive-to-active ROS blend
    } crown;

    // Phase 9: Flame temperature diagnostic
    std::string       flame_temp_method   = "byram_radiant"; ///< "byram_radiant", "mcalpine_heat", "nelson_emissivity"
    amrex::Real       flame_temp_T_amb    = 300.0;           ///< Ambient temperature for flame temp formula [K]
    
    // Phase 9: Flame tilt angle diagnostic
    bool              compute_flame_tilt  = false;           ///< Compute flame tilt angle diagnostic [degrees]
    amrex::Real       flame_tilt_T_amb    = 300.0;           ///< Ambient temperature for flame tilt [K]
    amrex::Real       flame_tilt_rho_air  = 1.2;             ///< Air density for flame tilt [kg/m³]

    // Phase 10: Spatial fuel map parameters.
    struct FuelMapParams {
        std::string   fuel_map_file       = "";        ///< Path to fuel map file; empty = uniform
        std::string   fuel_map_format     = "ascii";   ///< "ascii" or "lcp"
        amrex::Real   blending_fraction   = 0.0;       ///< ROS blending at fuel boundaries [0-1]
        /// Fuel codes that are non-burnable (e.g. 0 for nodata, 91-99 in the
        /// Scott-Burgan set). Empty by default, in which case an unknown code
        /// falls through to fuel model 1 in the Rothermel table as before.
        std::vector<int> nonburnable_codes;
    } fuel_map;

    // Phase 11: Multi-ignition schedule parameters.
    // When ignition_schedule_file is non-empty, a CSV schedule of timed ignition
    // events is loaded and applied during advance(). The primary ignition from
    // ignition_x/y/r is always applied at t=0 regardless of this setting.
    //
    // CSV format: time_s  cx  cy  radius  [source_type]  [priority]
    //   source_type: "sphere" (default) or "polyline"
    //   priority: integer; higher priority events fire first within the same timestep
    //   Comment lines start with # or !
    //
    // Polygon ignition (initial perimeter from a vertex file):
    //   polygon_file: path to CSV of polygon vertices (x y per line)
    //   polygon_type: "polygon" (closed) or "polyline" (open line fire)
    //   polyline_width: half-width of line ignition zone [m]
    struct IgnitionParams {
        std::string   ignition_schedule_file  = "";         ///< Path to ignition schedule CSV; empty = disabled
        std::string   polygon_file            = "";         ///< Path to polygon vertex CSV; empty = disabled
        std::string   polygon_type            = "polygon";  ///< "polygon" or "polyline"
        amrex::Real   polyline_width          = 10.0;       ///< Half-width of polyline ignition zone [m]
    } ignition;

    // Phase 12: Fire acceleration at startup.
    // Small fires do not immediately achieve their quasi-steady-state (QSS) ROS.
    // Two models are available, selected by use_temporal:
    //
    //   Size-based (use_temporal = false, default):
    //     alpha = 1 - exp(-r_fire / L_acc)
    //     R_effective = alpha * R_Rothermel
    //   where r_fire is the effective radius from burned area and L_acc is the
    //   acceleration length scale [m].
    //   Reference: Rothermel (1983) GTR INT-143.
    //
    //   Temporal (use_temporal = true):
    //     R(t) = R_E * (1 - exp(-A * t))
    //   where t is elapsed time since ignition [s] and A [1/min] is the
    //   acceleration constant, selected based on perimeter length relative to
    //   perim_limit (point vs line fire).
    //   Reference: McAlpine & Wakimoto (1991) Forest Science 37(5).
    //
    // Disabled by default (enable = false). When disabled, no MultiFab is
    // allocated and the call in advance() returns immediately.
    AccelerationParams accel;

    // Evaluate the rate of spread along the local front normal on the level-set
    // path, instead of handing it a single scalar rate that it then applies in
    // every direction. The wind and slope are projected onto the normal and the
    // selected ROS model is evaluated with the projected scalars, so head, flank
    // and backing spread come from the model itself.
    //
    // Off by default: with it off every model keeps the isotropic behaviour it
    // has always had on that path. No effect on the FARSITE path, which gets its
    // directionality from the Anderson ellipse. Balbi has its own switch,
    // erf.fire.balbi.directional, which additionally carries that model's
    // per-cell moisture, surface temperature and heat-flux couplings; either
    // flag enables direction-dependent spread when ros_model = "balbi".
    bool              directional_ros     = false;

    // Horizontal interpolation of the atmospheric wind onto the fire grid:
    // 0 = nearest atmospheric column (every fire cell in a column shares one
    // wind vector), 1 = bilinear between the four surrounding columns.
    // Set through erf.fire.wind_interp = "nearest" | "bilinear".
    int               wind_interp         = 1;

    // Phase 13B: Rate-of-spread model selection and parameters
    
    // Phase 13B: Rate-of-spread model selection.
    // "rothermel"    — Rothermel (1972) single-class model (default, Phases 1-12)
    // "macarthur"    — MacArthur (1966) Australian formula R = 0.18*exp(0.8424*U)
    // "balbi"        — Balbi (2009) physical model R = A*(1+sin α − cos α)
    // "cheney_gould" — Cheney & Gould (1998) empirical grassland model
    std::string       ros_model = "rothermel";  ///< ROS model selector

    // Phase 13B: Balbi (2009) physical fire spread model parameters.
    // Only used when ros_model = "balbi".
    // Defaults from Balbi (2009) Table 1.
    // The struct is copied to the device and evaluated inside GPU kernels, so
    // every member has to stay trivially copyable (no std::string here; the
    // "formulation" input string is mapped onto the integer flag below).
    struct BalbiParams {
        amrex::Real T_a      = 300.0;    ///< Ambient temperature [K]
        amrex::Real T_f      = 1000.0;   ///< Mean flame temperature [K]
        amrex::Real T_i      = 600.0;    ///< Ignition temperature [K]
        amrex::Real delta_H  = 2.26e6;   ///< Latent heat of vaporisation [J/kg]
        amrex::Real C_pf     = 1800.0;   ///< Specific heat of dry fuel [J/(kg·K)]
        amrex::Real r_00     = 2.5e-4;   ///< Radiation length scale [m]
        amrex::Real tau_0    = 75591.0;  ///< Residence time coefficient [s/m]

        // Formulation selector: 0 = Balbi (2009) steady explicit form (default),
        // 1 = Balbi (2020) convective-radiative implicit form.
        // Set through erf.fire.balbi.formulation = "2009" | "2020".
        int         formulation = 0;

        // Balbi (2020) constants. Defaults follow the nomenclature of
        // Balbi et al. (2020) as implemented in ForeFire's Balbi2020.
        amrex::Real chi_0    = 0.3;      ///< Radiant fraction [-]
        amrex::Real C_pa     = 1150.0;   ///< Specific heat of air [J/(kg·K)]
        amrex::Real C_pw     = 4180.0;   ///< Specific heat of water [J/(kg·K)]
        amrex::Real T_vap    = 373.0;    ///< Water vaporisation temperature [K]
        amrex::Real K1       = 130.0;    ///< Convective decay coefficient [-]
        amrex::Real st       = 17.0;     ///< Stoichiometric coefficient [-]
        amrex::Real rho_a    = 1.2;      ///< Air density [kg/m³]
        amrex::Real sigma_B  = 5.6e-8;   ///< Stefan-Boltzmann constant [W/(m²·K⁴)]
        int         max_iter = 40;       ///< Max root-solve iterations (2020 only)
        amrex::Real tol      = 1.0e-4;   ///< Root-solve bracket width [m/s] (2020 only)

        // Direction-dependent spread on the level-set path: evaluate the ROS
        // along the local front normal instead of using the wind and slope
        // magnitudes. No effect on the FARSITE path, which gets its
        // directionality from the ellipse.
        bool directional = false;

        // Take the ambient temperature per cell from fire_surface_temp instead
        // of the fixed balbi.T_a.
        bool use_surface_temp = false;

        // Derive the mean flame temperature from the combustion energy,
        // T_f = T_a + delta_H_combustion (1 - chi_0) / (C_pa (st + 1)),
        // instead of using the fixed balbi.T_f. Balbi (2009) form only; the
        // 2020 form always computes its own flame temperature.
        bool flame_temp_from_combustion = false;

        // Take the fuel moisture per cell from the Phase 4 moisture ODE state
        // (fire_fuel_mc component 0) instead of the domain average. Requires
        // erf.fire.moisture_dynamic = true.
        bool use_cell_moisture = false;

        // Zero the rate of spread once the fuel moisture reaches the fuel
        // model's moisture of extinction. Without this the model has no
        // extinction limit and a wet fuel bed still spreads.
        bool use_moisture_extinction = false;

        // Fraction of the live herbaceous load that has cured and is carried
        // as dead fine fuel. Only the Balbi (2020) form uses a fuel load, so
        // this has no effect under the 2009 formulation.
        amrex::Real herb_curing = 0.0;

        // Which wind the model consumes: 0 = the midflame wind after the
        // Wind Adjustment Factor and terrain corrections (fire_wind_eff,
        // default, what every other ROS model uses), 1 = the reference-height
        // wind before those corrections (fire_wind_ref). Balbi normalises the
        // wind by its own vertical velocity scale rather than by a midflame
        // reduction, so the WAF is arguably double counting.
        // Set through erf.fire.balbi.wind_source = "midflame" | "reference".
        int wind_source = 0;

        // Augment the buoyancy velocity with the fire-induced component built
        // from the surface heat flux:
        //   v_b_Q   = k_upward sqrt(g Q H_ref / (rho_a C_pa T_a))
        //   v_b_eff = sqrt(v_b^2 + v_b_Q^2)
        // The heat flux lags the ROS by one fire step.
        bool        heat_flux_coupling = false;
        amrex::Real k_upward           = 1.0;   ///< Scale on the induced velocity [-]
        amrex::Real hf_ref_height      = 10.0;  ///< Reference height for v_b_Q [m]
    } balbi;

    // Phase 13B: Cheney-Gould (1998) grassland model parameters.
    // Only used when ros_model = "cheney_gould".
    struct CheneyGouldParams {
        amrex::Real moisture = 10.0;  ///< Dead fine fuel moisture [%]
        amrex::Real curing   = 1.0;   ///< Degree of curing [0-1]; 1 = fully cured
    } cheney_gould;

    // Phase 15: BEHAVE multi-class Rothermel parameters.
    // Only used when ros_model = "behave".
    // Controls the dynamic live-to-dead load transfer threshold.
    // All other fuel properties come from FuelModelParams via get_anderson_fuel_params().
    struct BehaveParams {
        amrex::Real dynamic_transfer_lo  = 0.30;  ///< Live herba moisture [fraction] below which full transfer occurs
        amrex::Real dynamic_transfer_hi  = 1.20;  ///< Live herba moisture [fraction] above which no transfer occurs
    } behave;

    // Hybrid rate of spread: two of the models above evaluated on every cell
    // and blended with a per-cell weight in [0,1], where 0 takes the primary
    // model and 1 the secondary. Only used when ros_model = "hybrid".
    // Selectors: "region" and "fuel" set a static weight at initialisation,
    // "structure" a static weight from distance to a building footprint, and
    // "wind" a weight rebuilt every fire step from the effective wind speed.
    // Works on the isotropic path and, with directional_ros, on the
    // direction-dependent level-set path.
    struct HybridParams {
        std::string      primary   = "rothermel";  ///< Model where the weight is 0
        std::string      secondary = "balbi";      ///< Model where the weight is 1
        std::string      selector  = "region";     ///< "region" | "fuel" | "structure" | "wind"
        amrex::Real      region_x_lo = 0.0;        ///< selector = region: rectangle [m]
        amrex::Real      region_y_lo = 0.0;
        amrex::Real      region_x_hi = 0.0;
        amrex::Real      region_y_hi = 0.0;
        std::vector<int> secondary_fuel_codes;     ///< selector = fuel: codes that take the secondary
        std::string      structure_file;           ///< selector = structure: building heightmap; empty = erf.buildings_file_name
        amrex::Real      structure_min_height = 0.5; ///< selector = structure: height above which a cell is a structure [m]
        amrex::Real      structure_distance = 10.0; ///< selector = structure: cells within this distance take the secondary [m]
        amrex::Real      wind_lo = 8.0;            ///< selector = wind: weight 0 at and below this speed [m/s]
        amrex::Real      wind_hi = 12.0;           ///< selector = wind: weight 1 at and above this speed [m/s]
        amrex::Real      blend_width = 0.0;        ///< selector = region or structure: linear ramp across the edge [m]
    } hybrid;

    // Arrival-time probes: flat list of x y pairs [m]. Each probe reports the
    // fire arrival time at its fire cell once, on the step it first burns.
    std::vector<amrex::Real> probes;

    // Per-structure exposure diagnostics (needs structures.enable). Each
    // connected footprint of the heightmap gets an id, and every `interval`
    // fire steps one line per structure is appended to `file`: the first and
    // last arrival of the front in the ring of burnable cells within `ring`
    // fire cells of the footprint, the fraction of that ring burned, the peak
    // fireline intensity and the accumulated heat load there, and the number
    // of embers that landed on the footprint. Off by default.
    struct ExposureParams {
        bool        enable   = false;
        std::string file     = "fire_exposure.csv";
        int         interval = 100;   ///< fire steps between rows
        int         ring     = 1;     ///< wall band width [fire cells]
    } exposure;

    // Structures on the fire grid: a building heightmap in the ERF terrain
    // text format sampled onto fire cell centres by nearest point. With
    // enable = true every cell whose sampled height exceeds min_height is
    // non-burnable: zero rate of spread, the level set held outside the
    // burned region, FARSITE and spotting landings rejected, no fuel and no
    // heat. Off by default; the hybrid "structure" selector reads the same
    // file but by itself does not stop the fire.
    struct StructureParams {
        bool        enable     = false;   ///< Treat structure cells as non-burnable
        std::string file;                 ///< Heightmap; empty = hybrid.structure_file, then erf.buildings_file_name
        amrex::Real min_height = 0.5;     ///< Sampled height above which a cell is a structure [m]
        // Bilinear wind weights renormalised over the atmospheric columns
        // that are open at the fire cell's wind height: a column whose
        // sampled roof is above that height contributes with its open
        // fraction. With immersed-forcing buildings the columns inside a
        // footprint carry the relaxed in-building velocity, which otherwise
        // leaks into the wind of the fire cells around it. Off by default;
        // no effect with wind_interp = "nearest".
        bool        wind_open_columns = false;
    } structures;

    // Firebreak cells stamped at initialization are also held in the
    // non-burnable mask when true, so the level-set path cannot burn back
    // into them after reinitialisation. Off by default (legacy behaviour).
    bool firebreak_use_mask = false;

    // Evaluate the Rothermel coefficients per cell from the spatial fuel map
    // instead of the domain fuel_model_id. Off by default: the Rothermel
    // kernel then spreads with the domain fuel everywhere, as it always has,
    // and only the heat flux and the Balbi kernels see the map.
    bool rothermel_per_fuel = false;

    // Phase 10: Firebreak barriers (up to 10, indexed by N=0..9)
    int n_firebreaks = 0;
    std::vector<FireBreak> firebreaks;

    // Phase 4: smoke tracer
    bool  smoke_enable           = false;
    amrex::Real  smoke_emission_factor  = 0.02;       // kg smoke / kg fuel [-]
    amrex::Real  smoke_heat_of_comb     = 18700000.0; // J/kg fuel

    /**
     * @brief Construct FireParams from ParmParse
     *
     * Reads from "erf.fire.*" parameters. Unspecified values use defaults above.
     */
    FireParams()
    {
       amrex::ParmParse pp("erf.fire");
       pp.query("enable", enable);
       pp.query("fire_debug", fire_debug);
       pp.query("grid_ratio", grid_ratio);
       pp.query("fuel_model_id", fuel_model_id);
       pp.query("moisture_1hr", moisture_1hr);
       pp.query("moisture_10hr", moisture_10hr);
       pp.query("moisture_100hr", moisture_100hr);
       pp.query("ignition_x", ignition_x);
       pp.query("ignition_y", ignition_y);
       pp.query("ignition_r", ignition_r);
       pp.query("wind_ref_ht", wind_ref_ht);
       // Phase 13A: Per-fuel wind height
       pp.query("use_per_fuel_wind_ht", use_per_fuel_wind_ht);
       pp.query("waf_fcz0_scale", waf_fcz0_scale);
       pp.query("use_waf", use_waf);
       pp.query("waf_formula", waf_formula);
       pp.query("use_wind_limit", use_wind_limit);
       pp.query("use_terrain_wind", use_terrain_wind);
       pp.query("k_ridge", k_ridge);
       pp.query("k_shelter", k_shelter);
       pp.query("k_valley", k_valley);
       pp.query("k_deflect", k_deflect);
        
       // Fuel moisture control (Phase 4)
       pp.query("moisture_dynamic", moisture_dynamic);
       pp.query("precip_rate_mm_hr", precip_rate_mm_hr);
       pp.query("moisture_live", moisture_live);
        
       // Moisture of extinction (Phase 4)
       pp.query("use_dynamic_mext", use_dynamic_mext);
          
       // Heat flux parameters (Phase 5)
       pp.query("tau_residence_s", tau_residence_s);
       pp.query("f_c_min", f_c_min);
       pp.query("f_c_max", f_c_max);

       // FARSITE parameters
       pp.query("farsite.phi_threshold", farsite_phi_threshold);
       pp.query("farsite.use_anderson_lw", farsite_use_anderson_lw);
       pp.query("farsite.coeff_a", farsite_coeff_a);
       pp.query("farsite.coeff_b", farsite_coeff_b);
       pp.query("farsite.coeff_c", farsite_coeff_c);
       pp.query("farsite.gaussian_sigma", farsite_gaussian_sigma);
       pp.query("farsite.cfl_fire", farsite_cfl_fire);

       // Level-set propagation parameters
       pp.query("propagation_method", propagation_method);
       pp.query("levelset.cfl", levelset_cfl);
       pp.query("levelset.eps_visc", levelset_eps_visc);
       pp.query("levelset.reinit_every", levelset_reinit_every);
       pp.query("levelset.reinit_iters", levelset_reinit_iters);
       pp.query("levelset.reinit_band_m", levelset_reinit_band_m);
       pp.query("levelset.reinit_dtau", levelset_reinit_dtau);
       pp.query("levelset.wall_extrapolate", levelset_wall_extrapolate);

       // Terrain file
       amrex::ParmParse pp_erf("erf");
       pp_erf.query("terrain_file_name", terrain_file_name);
        
       // CSV output parameters
       pp.query("write_fire_stats_csv", write_fire_stats_csv);
       pp.query("fire_stats_csv_file", fire_stats_csv_file);
        
       // Phase 6: fire-atmosphere coupling parameters
       pp.query("heat_flux_alfg", heat_flux_alfg);
       pp.query("fire_atm_feedback", fire_atm_feedback);
       pp.query("coupling_type", coupling_type);
       pp.query("inject_latent", inject_latent);
       pp.query("source_mode", source_mode);
       if (source_mode != "overwrite" && source_mode != "add") {
           amrex::Abort("erf.fire.source_mode must be \"overwrite\" or \"add\"");
       }
       pp.query("heat_open_fraction", heat_open_fraction);
       pp.query("heat_tendency_density", heat_tendency_density);

       // Phase 4: smoke tracer parameters
       pp.query("smoke_enable", smoke_enable);
       pp.query("smoke_emission_factor", smoke_emission_factor);
       pp.query("smoke_heat_of_comb", smoke_heat_of_comb);

       // Phase 8: Albini (1983) ember spotting parameters
       auto& sp = spotting;
       pp.query("spotting.enable",              sp.enable);
       pp.query("spotting.I_B_min",             sp.I_B_min);
       pp.query("spotting.P_base",              sp.P_base);
       pp.query("spotting.terminal_velocity",   sp.terminal_velocity);
       pp.query("spotting.n_traj_steps",        sp.n_traj_steps);
       pp.query("spotting.spot_radius",         sp.spot_radius);
       pp.query("spotting.launch_from",         sp.launch_from);
       pp.query("spotting.front_band",          sp.front_band);
       if (sp.launch_from != "burned" && sp.launch_from != "front") {
           amrex::Abort("erf.fire.spotting.launch_from must be \"burned\" or \"front\"");
       }
       if (sp.front_band < 1) { amrex::Abort("erf.fire.spotting.front_band must be >= 1"); }
       pp.query("spotting.P_catch",             sp.P_catch);
       pp.query("spotting.reentry_fuel_thresh", sp.reentry_fuel_thresh);
       pp.query("spotting.random_seed",         sp.random_seed);
       pp.query("spotting.spotting_interval",   sp.spotting_interval);
       pp.query("spotting.fuel_system",         sp.fuel_system);

       // Phase 9: Crown fire initiation parameters
       auto& cr = crown;
       pp.query("crown.enable",          cr.enable);
       pp.query("crown.canopy_base_ht",  cr.canopy_base_ht);
       pp.query("crown.canopy_bulk_den", cr.canopy_bulk_den);
       pp.query("crown.canopy_depth",    cr.canopy_depth);
       pp.query("crown.foliar_moisture", cr.foliar_moisture);
       pp.query("crown.M_c",             cr.M_c);
       pp.query("crown.h_crown_BTU_lb",  cr.h_crown_BTU_lb);
       pp.query("crown.wind_10m_kmh",    cr.wind_10m_kmh);
       pp.query("crown.ros_model",       cr.ros_model);
       pp.query("crown.use_passive_blend", cr.use_passive_blend);

       // Phase 9: Flame temperature diagnostic
       pp.query("flame_temp_method", flame_temp_method);
       pp.query("flame_temp_T_amb",  flame_temp_T_amb);
         
       // Phase 9: Flame tilt angle diagnostic
       pp.query("compute_flame_tilt", compute_flame_tilt);
       pp.query("flame_tilt_T_amb",   flame_tilt_T_amb);
       pp.query("flame_tilt_rho_air", flame_tilt_rho_air);

       // Phase 10: fuel map
       auto& fm = fuel_map;
       pp.query("fuel_map.file",               fm.fuel_map_file);
       pp.query("fuel_map.format",             fm.fuel_map_format);
       pp.query("fuel_map.blending_fraction",  fm.blending_fraction);

       // Phase 11: multi-ignition schedule and polygon ignition
       auto& ig = ignition;
       pp.query("ignition.schedule_file", ig.ignition_schedule_file);
       pp.query("ignition.polygon_file",  ig.polygon_file);
       pp.query("ignition.polygon_type",  ig.polygon_type);
       pp.query("ignition.polyline_width", ig.polyline_width);

       // Phase 12: fire acceleration
       auto& ac = accel;
       pp.query("accel.enable",          ac.enable);
       pp.query("accel.use_temporal",    ac.use_temporal);
       pp.query("accel.L_acc",           ac.L_acc);
       pp.query("accel.A_point",         ac.A_point);
       pp.query("accel.A_line",          ac.A_line);
       pp.query("accel.perim_limit",     ac.perim_limit);
       pp.query("accel.enable_wind_lag", ac.enable_wind_lag);
       pp.query("accel.tau_wind",        ac.tau_wind);

       pp.query("directional_ros", directional_ros);

       {
           std::string wind_interp_str = (wind_interp == 0) ? "nearest" : "bilinear";
           pp.query("wind_interp", wind_interp_str);
           if (wind_interp_str == "bilinear") {
               wind_interp = 1;
           } else if (wind_interp_str == "nearest") {
               wind_interp = 0;
           } else {
               amrex::Abort("erf.fire.wind_interp must be \"bilinear\" or \"nearest\"");
           }
       }

       // Phase 13B: ROS model selection
       pp.query("ros_model", ros_model);
       pp.query("rothermel_per_fuel", rothermel_per_fuel);

       // Hybrid ROS
       {
           auto& hy = hybrid;
           pp.query("hybrid.primary",   hy.primary);
           pp.query("hybrid.secondary", hy.secondary);
           pp.query("hybrid.selector",  hy.selector);
           std::vector<amrex::Real> region;
           if (pp.queryarr("hybrid.region", region)) {
               if (region.size() != 4) {
                   amrex::Abort("erf.fire.hybrid.region needs four values: x_lo y_lo x_hi y_hi");
               }
               hy.region_x_lo = region[0];
               hy.region_y_lo = region[1];
               hy.region_x_hi = region[2];
               hy.region_y_hi = region[3];
           }
           pp.queryarr("hybrid.secondary_fuel_codes", hy.secondary_fuel_codes);
           pp.query("hybrid.structure_file",       hy.structure_file);
           pp.query("hybrid.structure_min_height", hy.structure_min_height);
           pp.query("hybrid.structure_distance",   hy.structure_distance);
           pp.query("hybrid.wind_lo",            hy.wind_lo);
           pp.query("hybrid.wind_hi",            hy.wind_hi);
           pp.query("hybrid.blend_width",        hy.blend_width);

           if (ros_model == "hybrid") {
               auto known = [](const std::string& m) {
                   return m == "rothermel" || m == "balbi" || m == "behave" ||
                          m == "macarthur" || m == "cheney_gould";
               };
               if (!known(hy.primary) || !known(hy.secondary)) {
                   amrex::Abort("erf.fire.hybrid.primary/secondary must each be one of "
                                "rothermel, balbi, behave, macarthur, cheney_gould");
               }
               if (hy.primary == hy.secondary) {
                   amrex::Abort("erf.fire.hybrid.primary and secondary must differ");
               }
               if (hy.selector == "region") {
                   if (!pp.contains("hybrid.region")) {
                       amrex::Abort("erf.fire.hybrid.selector = region needs erf.fire.hybrid.region");
                   }
                   if (hy.region_x_hi < hy.region_x_lo || hy.region_y_hi < hy.region_y_lo) {
                       amrex::Abort("erf.fire.hybrid.region must satisfy x_lo <= x_hi and y_lo <= y_hi");
                   }
               } else if (hy.selector == "fuel") {
                   if (hy.secondary_fuel_codes.empty()) {
                       amrex::Abort("erf.fire.hybrid.selector = fuel needs erf.fire.hybrid.secondary_fuel_codes");
                   }
               } else if (hy.selector == "structure") {
                   if (hy.structure_file.empty()) {
                       amrex::ParmParse pp_erf("erf");
                       pp_erf.query("buildings_file_name", hy.structure_file);
                   }
                   if (hy.structure_file.empty()) {
                       amrex::Abort("erf.fire.hybrid.selector = structure needs "
                                    "erf.fire.hybrid.structure_file or erf.buildings_file_name");
                   }
                   if (hy.structure_distance < 0.0) {
                       amrex::Abort("erf.fire.hybrid.structure_distance must be >= 0");
                   }
               } else if (hy.selector == "wind") {
                   if (!(hy.wind_hi > hy.wind_lo)) {
                       amrex::Abort("erf.fire.hybrid.wind_hi must exceed erf.fire.hybrid.wind_lo");
                   }
               } else {
                   amrex::Abort("erf.fire.hybrid.selector must be region, fuel, structure or wind");
               }
               if (hy.blend_width < 0.0) {
                   amrex::Abort("erf.fire.hybrid.blend_width must be >= 0");
               }
           }
       }

       pp.queryarr("probes", probes);
       if (probes.size() % 2 != 0) {
           amrex::Abort("erf.fire.probes must be a flat list of x y pairs");
       }
       pp.query("exposure.enable",   exposure.enable);
       pp.query("exposure.file",     exposure.file);
       pp.query("exposure.interval", exposure.interval);
       pp.query("exposure.ring",     exposure.ring);
       // Checked against structures.enable below, once that block is read.

       // Non-burnable mask: structures, fuel codes, firebreaks
       pp.query("structures.enable",     structures.enable);
       pp.query("structures.file",       structures.file);
       pp.query("structures.min_height", structures.min_height);
       pp.query("structures.wind_open_columns", structures.wind_open_columns);
       pp.queryarr("fuel_map.nonburnable_codes", fuel_map.nonburnable_codes);
       pp.query("firebreak.use_mask", firebreak_use_mask);
       if (structures.enable) {
           if (structures.file.empty()) { structures.file = hybrid.structure_file; }
           if (structures.file.empty()) {
               amrex::ParmParse pp_erf("erf");
               pp_erf.query("buildings_file_name", structures.file);
           }
           if (structures.file.empty()) {
               amrex::Abort("erf.fire.structures.enable needs erf.fire.structures.file "
                            "or erf.buildings_file_name");
           }
           if (structures.min_height < 0.0) {
               amrex::Abort("erf.fire.structures.min_height must be >= 0");
           }
       }
       if (structures.wind_open_columns && !structures.enable) {
           amrex::Abort("erf.fire.structures.wind_open_columns needs erf.fire.structures.enable");
       }
       if (exposure.enable) {
           if (!structures.enable) {
               amrex::Abort("erf.fire.exposure.enable needs erf.fire.structures.enable");
           }
           if (exposure.interval < 1) { amrex::Abort("erf.fire.exposure.interval must be >= 1"); }
           if (exposure.ring < 1)     { amrex::Abort("erf.fire.exposure.ring must be >= 1"); }
       }
       if (heat_open_fraction && !structures.enable) {
           amrex::Abort("erf.fire.heat_open_fraction needs erf.fire.structures.enable");
       }

       // Balbi parameters
       auto& bl = balbi;
       pp.query("balbi.T_a",     bl.T_a);
       pp.query("balbi.T_f",     bl.T_f);
       pp.query("balbi.T_i",     bl.T_i);
       pp.query("balbi.delta_H", bl.delta_H);
       pp.query("balbi.C_pf",    bl.C_pf);
       pp.query("balbi.r_00",    bl.r_00);
       pp.query("balbi.tau_0",   bl.tau_0);

       // Balbi formulation selector, mapped onto the device-side integer flag
       {
           std::string balbi_formulation = (bl.formulation == 1) ? "2020" : "2009";
           pp.query("balbi.formulation", balbi_formulation);
           if (balbi_formulation == "2020") {
               bl.formulation = 1;
           } else if (balbi_formulation == "2009") {
               bl.formulation = 0;
           } else {
               amrex::Abort("erf.fire.balbi.formulation must be \"2009\" or \"2020\"");
           }
       }

       // Balbi (2020) constants and solver controls
       // Balbi et al. (2020) fit r00 = 2.5e-5, while the 2009 radiation length
       // scale is 2.5e-4 m. Move the default with the formulation unless the
       // user has set it explicitly.
       if (bl.formulation == 1 && !pp.contains("balbi.r_00")) {
           bl.r_00 = 2.5e-5;
       }

       pp.query("balbi.chi_0",    bl.chi_0);
       pp.query("balbi.C_pa",     bl.C_pa);
       pp.query("balbi.C_pw",     bl.C_pw);
       pp.query("balbi.T_vap",    bl.T_vap);
       pp.query("balbi.K1",       bl.K1);
       pp.query("balbi.st",       bl.st);
       pp.query("balbi.rho_a",    bl.rho_a);
       pp.query("balbi.sigma_B",  bl.sigma_B);
       pp.query("balbi.max_iter", bl.max_iter);
       pp.query("balbi.tol",      bl.tol);

       // Balbi coupling options
       pp.query("balbi.directional",                bl.directional);
       pp.query("balbi.use_surface_temp",           bl.use_surface_temp);
       pp.query("balbi.flame_temp_from_combustion", bl.flame_temp_from_combustion);
       pp.query("balbi.use_cell_moisture",          bl.use_cell_moisture);
       pp.query("balbi.use_moisture_extinction",    bl.use_moisture_extinction);
       pp.query("balbi.herb_curing",                bl.herb_curing);
       pp.query("balbi.heat_flux_coupling",         bl.heat_flux_coupling);

       {
           std::string balbi_wind_source = (bl.wind_source == 1) ? "reference" : "midflame";
           pp.query("balbi.wind_source", balbi_wind_source);
           if (balbi_wind_source == "reference") {
               bl.wind_source = 1;
           } else if (balbi_wind_source == "midflame") {
               bl.wind_source = 0;
           } else {
               amrex::Abort("erf.fire.balbi.wind_source must be "
                            "\"midflame\" or \"reference\"");
           }
       }
       pp.query("balbi.k_upward",                   bl.k_upward);
       pp.query("balbi.hf_ref_height",              bl.hf_ref_height);

       // Cheney-Gould parameters
       auto& cg = cheney_gould;
       pp.query("cheney_gould.moisture", cg.moisture);
       pp.query("cheney_gould.curing",   cg.curing);

       // Phase 15: BEHAVE parameters
       auto& bh = behave;
       pp.query("behave.dynamic_transfer_lo", bh.dynamic_transfer_lo);
       pp.query("behave.dynamic_transfer_hi", bh.dynamic_transfer_hi);

       // Phase 10: firebreaks (indexed, stop at first missing)
       for (int nb = 0; nb < 10; ++nb) {
           std::string type_key = "firebreak." + std::to_string(nb) + ".type";
           std::string type_val;
           if (!pp.query(type_key.c_str(), type_val)) break;
           FireBreak fb;
           fb.type = type_val;
           std::string pfx = "firebreak." + std::to_string(nb) + ".";
           pp.query((pfx + "x_lo").c_str(),   fb.x_lo);
           pp.query((pfx + "y_lo").c_str(),   fb.y_lo);
           pp.query((pfx + "x_hi").c_str(),   fb.x_hi);
           pp.query((pfx + "y_hi").c_str(),   fb.y_hi);
           pp.query((pfx + "cx").c_str(),     fb.cx);
           pp.query((pfx + "cy").c_str(),     fb.cy);
           pp.query((pfx + "radius").c_str(), fb.radius);
           firebreaks.push_back(fb);
       }
       n_firebreaks = static_cast<int>(firebreaks.size());
    }

    /// True when ros_model = "hybrid".
    bool is_hybrid() const { return ros_model == "hybrid"; }

    /**
     * @brief Whether a rate-of-spread model is evaluated in this run.
     *
     * Covers the single-model case and both members of a hybrid pair.
     * "rothermel" is the fallback for any unrecognised ros_model string,
     * matching the dispatch in FireLayer.
     */
    bool uses_model(const std::string& m) const
    {
        if (is_hybrid()) { return hybrid.primary == m || hybrid.secondary == m; }
        if (m == "rothermel") {
            return ros_model != "balbi" && ros_model != "behave" &&
                   ros_model != "macarthur" && ros_model != "cheney_gould";
        }
        return ros_model == m;
    }

    // Convenience accessors for coupling mode
    bool is_passive()     const { return coupling_type == "passive"; }
    bool is_lagged()      const { return coupling_type == "lagged"; }
    bool is_synchronous() const { return coupling_type == "synchronous"; }
    bool injects_flux()   const { return is_lagged() || is_synchronous(); }
};

#endif
