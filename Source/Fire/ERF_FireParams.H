/**
 * \file ERF_FireParams.H
 *
 * \brief Fire module configuration parameters.
 *
 * Stores all fire simulation parameters read from ParmParse at startup.
 * This struct is populated once from `erf.fire.*` prefixed parameters.
 *
 * References:
 * - Rothermel, R.C. (1972), A mathematical model for predicting fire spread
 *   in wildland fuels, USDA Forest Service Research Paper INT-115.
 * - Anderson, H.E. (1982), Aids to determining fuel models for estimating fire behavior,
 *   USDA Forest Service General Technical Report INT-122.
 *
 * Ported concepts from: hgopalan/wildfire_levelset/src/constants.H
 */

#ifndef ERF_FIRE_PARAMS_H
#define ERF_FIRE_PARAMS_H

#include <AMReX_ParmParse.H>
#include <string>

/**
 * \struct FireParams
 *
 * Fire simulation parameters. All members are populated from ParmParse
 * with the prefix "erf.fire" at ERF startup.
 */
struct FireParams
{
    bool         enable = false;              ///< Fire module enabled (erf.fire.enable)
    int          grid_ratio = 5;              ///< Fire grid refinement factor C (x,y only)
    int          fuel_model_id = 1;           ///< Anderson FBFM13 model ID (1-13)
    amrex::Real  moisture_1hr = 0.08;         ///< 1-hour fuel moisture fraction
    amrex::Real  moisture_10hr = 0.08;        ///< 10-hour fuel moisture fraction
    amrex::Real  moisture_100hr = 0.10;       ///< 100-hour fuel moisture fraction
    amrex::Real  ignition_x = 0.0;            ///< Ignition center X coordinate [m]
    amrex::Real  ignition_y = 0.0;            ///< Ignition center Y coordinate [m]
    amrex::Real  ignition_r = 20.0;           ///< Ignition radius [m]
    amrex::Real  wind_ref_ht = 6.1;           ///< Wind reference height [m] (20-ft)
    bool         use_waf = true;              ///< Apply Wind Adjustment Factor
    std::string  waf_formula = "andrews";     ///< WAF formula: "andrews" or "behaviorplus"
    bool         use_wind_limit = true;       ///< Apply MEWS cap on ROS
    bool         use_terrain_wind = true;     ///< Apply FARSITE terrain wind corrections
    amrex::Real  k_ridge = 1.5;               ///< FARSITE ridge enhancement factor
    amrex::Real  k_shelter = 0.6;             ///< FARSITE lee-side shelter factor
    amrex::Real  k_valley = 0.8;              ///< FARSITE valley channeling factor
    amrex::Real  k_deflect = 0.3;             ///< FARSITE directional deflection factor

    /**
     * \brief Initialize from ParmParse with prefix "erf.fire"
     */
    explicit FireParams(const amrex::ParmParse& pp_fire)
    {
        pp_fire.query("enable", enable);
        pp_fire.query("grid_ratio", grid_ratio);
        pp_fire.query("fuel_model_id", fuel_model_id);
        pp_fire.query("moisture_1hr", moisture_1hr);
        pp_fire.query("moisture_10hr", moisture_10hr);
        pp_fire.query("moisture_100hr", moisture_100hr);
        pp_fire.query("ignition_x", ignition_x);
        pp_fire.query("ignition_y", ignition_y);
        pp_fire.query("ignition_r", ignition_r);
        pp_fire.query("wind_ref_ht", wind_ref_ht);
        pp_fire.query("use_waf", use_waf);
        pp_fire.query("waf_formula", waf_formula);
        pp_fire.query("use_wind_limit", use_wind_limit);
        pp_fire.query("use_terrain_wind", use_terrain_wind);
        pp_fire.query("k_ridge", k_ridge);
        pp_fire.query("k_shelter", k_shelter);
        pp_fire.query("k_valley", k_valley);
        pp_fire.query("k_deflect", k_deflect);
    }

    /**
     * \brief Default constructor for no-fire case.
     */
    FireParams() = default;
};

#endif // ERF_FIRE_PARAMS_H
