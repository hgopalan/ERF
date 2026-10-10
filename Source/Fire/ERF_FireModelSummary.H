#ifndef ERF_FIRE_MODEL_SUMMARY_H
#define ERF_FIRE_MODEL_SUMMARY_H

/**
 * @file ERF_FireModelSummary.H
 * @brief The model or option the run uses for every piece of fire physics,
 *        printed once at start-up, before the first step, whatever
 *        erf.fire.fire_debug says.
 *
 * Each line names the physics, the erf.fire key that selects it, the value
 * the run uses and whether the deck set it or took the default. An option is
 * listed only when it acts in this run: the sub-options of a model only when
 * that model is in use (the Cheney-Gould pasture with ros_model =
 * cheney_gould, the level-set gradient with propagation_method = levelset),
 * the heat-injection options only when heat reaches the atmosphere, and so
 * on. A value the code forced is marked (directional_ros off under
 * levelset.ellipse). Every key listed must be one FireParams always reads:
 * ParmParse::contains counts as a use, so a key read only conditionally
 * would hide from the unused-input report.
 */

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "ERF_FireParams.H"
#include "ERF_FireAcceleration.H"
#include "ERF_FarsiteEllipse.H"
#include "ERF_BehaveModel.H"
#include "ERF_CheneyGouldModel.H"
#include "ERF_Rothermel.H"

/// One line of the summary: the physics, its erf.fire key and the value used
struct FireModelLine {
    std::string physics;
    std::string key;
    std::string value;
};

namespace fire_model_summary_detail {
inline std::string on_off (bool b) { return b ? "true" : "false"; }
inline std::string real_str (amrex::Real v)
{
    std::ostringstream s; s << v; return s.str();
}
} // namespace fire_model_summary_detail

/**
 * @brief The model or option of every piece of fire physics this run uses.
 *
 * Integer-coded selectors are written with the input word that selects them,
 * as FireParams reads it.
 */
inline std::vector<FireModelLine> fire_model_summary (const FireParams& fp)
{
    using namespace fire_model_summary_detail;
    std::vector<FireModelLine> L;
    auto add = [&L] (const char* physics, const char* key, const std::string& value) {
        L.push_back(FireModelLine{physics, key, value});
    };
    const bool levelset = fp.propagation_method == "levelset";

    // ---- rate of spread
    add("Rate of spread", "ros_model", fp.ros_model);
    if (fp.is_hybrid()) {
        add("  hybrid primary", "hybrid.primary", fp.hybrid.primary);
        add("  hybrid secondary", "hybrid.secondary", fp.hybrid.secondary);
        add("  hybrid selector", "hybrid.selector", fp.hybrid.selector);
    }
    if (fp.uses_model("rothermel")) {
        add("  Rothermel reaction velocity", "reaction_velocity_formula", fp.reaction_velocity_formula);
        add("  Rothermel WRF moisture deflation", "wrf_bmst_compat", on_off(fp.wrf_bmst_compat));
        add("  Rothermel per-fuel coefficients", "rothermel_per_fuel", on_off(fp.rothermel_per_fuel));
        if (fp.moisture_dynamic) {
            add("  Rothermel per-cell moisture", "rothermel_cell_moisture", on_off(fp.rothermel_cell_moisture));
        }
    }
    if (fp.uses_model("rothermel") || fp.uses_model("behave")) {
        add("  midflame wind limit", "use_wind_limit", on_off(fp.use_wind_limit));
        if (fp.use_wind_limit) {
            add("  wind limit form", "wind_limit",
                fp.wind_limit_mode == fire_wind_limit::fuel_class ? "fuel_class" : "rothermel");
        }
    }
    if (fp.uses_model("behave")) {
        add("  BEHAVE net load", "behave.net_load",
            fp.behave.net_load_mode == behave_net_load::sum ? "sum" : "weighted");
    }
    if (fp.uses_model("cheney_gould")) {
        add("  Cheney-Gould pasture", "cheney_gould.pasture",
            fp.cheney_gould.pasture == cheney_gould::pasture_grazed ? "grazed" : "natural");
        add("  Cheney-Gould curing curve", "cheney_gould.curing_curve",
            fp.cheney_gould.curing_curve == cheney_gould::curing_cheney1998 ? "cheney1998" : "cruz2015");
        add("  Cheney-Gould wind", "cheney_gould.wind_source",
            fp.cheney_gould.wind_source == 1 ? "midflame" : "reference");
    }
    if (fp.uses_model("fbp")) {
        add("  FBP fuel type", "fbp.fuel_type", fp.fbp.fuel_type);
        add("  FBP wind", "fbp.wind_source", fp.fbp.wind_source == 1 ? "midflame" : "reference");
        add("  FBP slope", "fbp.use_slope", on_off(fp.fbp.use_slope));
    }
    if (fp.uses_model("macarthur")) {
        add("  McArthur rate cap [m/s]", "macarthur.ros_max", real_str(fp.macarthur_ros_max));
    }
    if (fp.uses_model("balbi")) {
        add("  Balbi formulation", "balbi.formulation", fp.balbi.formulation == 1 ? "2020" : "2009");
        add("  Balbi wind", "balbi.wind_source", fp.balbi.wind_source == 1 ? "reference" : "midflame");
        // the Balbi front is directional with either switch (FireLayer's balbi_directional)
        const bool balbi_dir = fp.balbi.directional || (levelset && fp.directional_ros);
        add("  Balbi directional", "balbi.directional",
            on_off(balbi_dir) + (balbi_dir && !fp.balbi.directional ? std::string(" (through directional_ros)") : std::string()));
        add("  Balbi heat-flux coupling", "balbi.heat_flux_coupling", on_off(fp.balbi.heat_flux_coupling));
        add("  Balbi surface temperature", "balbi.use_surface_temp", on_off(fp.balbi.use_surface_temp));
        add("  Balbi flame temperature from combustion", "balbi.flame_temp_from_combustion",
            on_off(fp.balbi.flame_temp_from_combustion));
        add("  Balbi per-cell moisture", "balbi.use_cell_moisture", on_off(fp.balbi.use_cell_moisture));
        add("  Balbi moisture of extinction", "balbi.use_moisture_extinction", on_off(fp.balbi.use_moisture_extinction));
    }
    if (fp.uses_model("prescribed")) {
        add("  prescribed rate [m/s]", "prescribed.ros", real_str(fp.prescribed.ros));
        add("  prescribed rate by fuel", "prescribed.by_fuel",
            fp.prescribed.by_fuel.empty() ? std::string("none")
                                          : std::to_string(fp.prescribed.by_fuel.size() / 2) + " fuel codes");
    }

    // ---- front propagation
    add("Front propagation", "propagation_method", fp.propagation_method);
    if (levelset) {
        add("  level-set gradient", "levelset.gradient", fp.levelset_gradient);
        add("  level-set reinitialization", "levelset.reinit_scheme", fp.levelset_reinit_scheme);
        add("  level-set artificial viscosity", "levelset.eps_visc", real_str(fp.levelset_eps_visc));
        add("  level-set wall extrapolation", "levelset.wall_extrapolate", on_off(fp.levelset_wall_extrapolate));
        add("  level-set ellipse", "levelset.ellipse", on_off(fp.levelset_ellipse));
        // levelset.ellipse and ros_model = prescribed switch the directional rate off
        const std::string forced = fp.levelset_ellipse ? " (off: levelset.ellipse)"
                                 : (fp.ros_model == "prescribed" ? " (off: ros_model = prescribed)" : "");
        add("  directional rate of spread", "directional_ros", on_off(fp.directional_ros) + forced);
        if (fp.directional_ros) {
            add("  directional shape", "directional_shape", fp.directional_shape == 1 ? "ellipse" : "projection");
            add("  directional wind coupling", "directional_wind_coupling",
                fp.directional_wind_coupling == 1 ? "advective" : "projection");
            if (fp.directional_shape == 1) {
                add("  directional length-to-width", "directional_ellipse_lw",
                    fp.directional_ellipse_lw == 1 ? "anderson" : "model");
            }
            add("  split Hamiltonian", "directional_split_hamiltonian", on_off(fp.directional_split_hamiltonian));
        }
    } else {
        add("  FARSITE shape", "farsite.shape",
            fp.farsite_shape == farsite_shape::rectangle ? "rectangle" : "ellipse");
        add("  FARSITE length-to-width from Anderson", "farsite.use_anderson_lw", on_off(fp.farsite_use_anderson_lw != 0));
        add("  FARSITE front update", "farsite.front_update",
            fp.farsite_front_update == farsite_front::legacy ? "legacy" : "front_cell");
    }

    // ---- wind
    add("Prescribed wind", "prescribed_wind", on_off(fp.prescribed_wind));
    if (!fp.prescribed_wind) {
        add("Wind interpolation", "wind_interp", fp.wind_interp == 0 ? "nearest" : "bilinear");
        add("Wind height [m]", fp.use_per_fuel_wind_ht ? "use_per_fuel_wind_ht" : "wind_ref_ht",
            fp.use_per_fuel_wind_ht ? std::string("true (6.096)") : real_str(fp.wind_ref_ht));
        add("Wind sampling height [m]", "wind_sample_ht",
            fp.wind_sample_ht > 0.0 ? real_str(fp.wind_sample_ht) : std::string("0 (at the wind height)"));
        add("Wind below the first cell", "wind_below_first_cell", fp.wind_below_first_log ? "log" : "clamp");
    }
    // the roughness of the two-height resampling, the log wind below the first
    // cell and the ember drift
    if (!fp.prescribed_wind && (fp.wind_sample_ht > 0.0 || fp.wind_below_first_log) ) {
        add("  log-profile roughness [m]", "wind_sample_z0", real_str(fp.wind_sample_z0));
    } else if (fp.spotting.enable) {
        add("Ember-drift roughness [m]", "wind_sample_z0", real_str(fp.wind_sample_z0));
    }
    add("Wind adjustment factor", "use_waf", on_off(fp.use_waf));
    if (fp.use_waf) { add("  WAF formula", "waf_formula", fp.waf_formula); }
    add("Terrain wind factors", "use_terrain_wind", on_off(fp.use_terrain_wind));
    add("Fire terrain file", "terrain_file_name",
        fp.terrain_file_name.empty() ? std::string("none") : fp.terrain_file_name);

    // ---- fuel and moisture
    add("Fuel model", "fuel_model_id", std::to_string(fp.fuel_model_id));
    add("Fuel set", "fuel_map.fuel_set", fp.fuel_map.fuel_set);
    if (fp.fuel_map.fuel_set == "scott_burgan40") {
        add("  Scott-Burgan to Anderson crosswalk", "fuel_map.sb40_crosswalk", on_off(fp.fuel_map.sb40_crosswalk));
    }
    add("Fuel map", "fuel_map.file",
        fp.fuel_map.fuel_map_file.empty() ? std::string("none (uniform fuel)") : fp.fuel_map.fuel_map_file);
    if (!fp.fuel_map.fuel_map_file.empty()) {
        add("  initial load from the map", "fuel_map.load_from_map", on_off(fp.fuel_map.load_from_map));
        add("  boundary blending fraction", "fuel_map.blending_fraction", real_str(fp.fuel_map.blending_fraction));
    }
    add("Burnout", "burnout_model", fp.burnout_model);
    if (fp.burnout_model == "residence") {
        add("  residence time [s]", "tau_residence_s",
            fp.tau_residence_s > 0.0 ? real_str(fp.tau_residence_s) : std::string("0 (from the fuel SAV)"));
    } else if (fp.burnout_model == "sfire") {
        add("  burn time per e-fold", "burnout_time_to_efold", real_str(fp.burnout_time_to_efold));
        add("  burn times [s]", "burnout_times_s",
            fp.burnout_times_s.empty() ? std::string("the fuel table") : std::string("given per Anderson model"));
    }
    add("Dynamic fuel moisture", "moisture_dynamic", on_off(fp.moisture_dynamic));
    if (fp.moisture_dynamic) {
        add("  dead moisture model", "moisture_model", fp.moisture_model);
        add("  equilibrium moisture", "emc_model", fp.emc_model);
        add("  precipitation", "precip_source", fp.precip_source);
        add("  live fuel moisture model", "moisture_live_model", fp.moisture_live_model);
    } else {
        add("Live fuel moisture [-]", "moisture_live", real_str(fp.moisture_live));
    }

    // ---- ignition
    add("Ignition disc radius [m]", "ignition_r",
        fp.ignition_r > 0.0 ? real_str(fp.ignition_r) : std::string("0 (no disc)"));
    add("Ignition polygon", "ignition.polygon_file",
        fp.ignition.has_polygon() ? std::to_string(fp.ignition.polygon_files.size()) + " file(s)" : std::string("none"));
    add("Ignition schedule", "ignition.schedule_file",
        fp.ignition.ignition_schedule_file.empty() ? std::string("none") : fp.ignition.ignition_schedule_file);
    add("Threshold ignition", "ignition.threshold_enable", on_off(fp.ignition.threshold_enable));

    // ---- coupling to the atmosphere
    add("Atmosphere coupling", "coupling_type", fp.coupling_type);
    if (fp.injects_flux()) {
        add("  feedback multiplier", "fire_atm_feedback", real_str(fp.fire_atm_feedback));
        if (fp.fire_atm_feedback > 0.0) {
            add("  heat e-folding depth [m]", "heat_flux_alfg", real_str(fp.heat_flux_alfg));
            add("  heat partition", "heat_flux_partition", fp.heat_flux_partition);
            add("  source mode", "source_mode", fp.source_mode);
            add("  latent heat", "inject_latent", on_off(fp.inject_latent));
            add("  Exner factor in the tendency", "heat_tendency_exner", on_off(fp.heat_tendency_exner));
            add("  density in the tendency", "heat_tendency_density", on_off(fp.heat_tendency_density));
            add("  heat through open columns", "heat_open_fraction", on_off(fp.heat_open_fraction));
        }
    }
    if (fp.prescribed_heat.flux > 0.0) {
        add("Prescribed heat flux [W/m2]", "prescribed_heat.flux", real_str(fp.prescribed_heat.flux));
    }
    add("Flame temperature (diagnostic)", "flame_temp_method", fp.flame_temp_method);

    // ---- other physics
    add("Fire acceleration", "accel.enable", on_off(fp.accel.enable));
    if (fp.accel.enable) {
        add("  acceleration form", "accel.use_temporal", fp.accel.use_temporal ? "temporal" : "size_based");
        if (fp.accel.use_temporal) {
            add("  acceleration clock", "accel.clock", fp.accel.clock == accel_clock::front ? "front" : "legacy");
            add("  wind lag", "accel.enable_wind_lag", on_off(fp.accel.enable_wind_lag));
        }
    }
    add("Spotting", "spotting.enable", on_off(fp.spotting.enable));
    if (fp.spotting.enable) {
        add("  spotting fuel system", "spotting.fuel_system", fp.spotting.fuel_system);
        add("  brands launched from", "spotting.launch_from", fp.spotting.launch_from);
        add("  spotting every N subcycles", "spotting.spotting_interval", std::to_string(fp.spotting.spotting_interval));
    }
    add("Crown fire", "crown.enable", on_off(fp.crown.enable));
    if (fp.crown.enable) {
        add("  crown rate of spread", "crown.ros_model", fp.crown.ros_model);
        add("  passive-to-active blend", "crown.use_passive_blend", on_off(fp.crown.use_passive_blend));
        add("  crown 10 m wind [km/h]", "crown.wind_10m_kmh",
            fp.crown.wind_10m_kmh > 0.0 ? real_str(fp.crown.wind_10m_kmh) : std::string("0 (from the model wind)"));
    }
    add("Smoke", "smoke_enable", on_off(fp.smoke_enable));
    if (fp.smoke_enable) { add("  smoke heat from the fuel", "smoke_heat_from_fuel", on_off(fp.smoke_heat_from_fuel)); }
    add("Structures", "structures.enable", on_off(fp.structures.enable));
    if (fp.structures.enable) {
        add("  wind through open columns", "structures.wind_open_columns", on_off(fp.structures.wind_open_columns));
        add("  structure ignition", "structures.ignition.enable", on_off(fp.structures.ignition.enable));
    }
    add("Suppression", "suppression.enable", on_off(fp.suppression.enable));
    return L;
}

/**
 * @brief Print the summary on the I/O rank: one line per physics, with
 *        "(set)" for a key the deck gives and "(default)" otherwise.
 */
inline void print_fire_model_summary (const FireParams& fp)
{
    amrex::ParmParse pp("erf.fire");
    amrex::Print() << "[FIRE] Models (erf.fire key = value; set in the inputs, or the default):\n";
    for (const auto& m : fire_model_summary(fp)) {
        std::ostringstream line;
        line << "[FIRE]   " << std::left << std::setw(36) << m.physics
             << m.key << " = " << m.value << (pp.contains(m.key.c_str()) ? "  (set)" : "  (default)");
        amrex::Print() << line.str() << "\n";
    }
}

#endif // ERF_FIRE_MODEL_SUMMARY_H
