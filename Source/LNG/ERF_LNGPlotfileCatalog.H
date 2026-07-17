/**
 * @file ERF_LNGPlotfileCatalog.H
 * @brief LNG plotfile variable catalog (Phase 6)
 * @details
 * Header-only definition of all 17 LNG output variables and their ordering.
 * The order here MUST exactly match the copy_if() indices in ERF_LNGPlotfile.cpp.
 * This catalog is used by WriteLNGPlotfile() to populate the plotfile header
 * and determine variable ordering in the output.
 * 
 * Pattern: Source/Dust/ERF_DustPlotfileCatalog.H
 */

#ifndef ERF_LNG_PLOTFILE_CATALOG_H
#define ERF_LNG_PLOTFILE_CATALOG_H

#ifdef ERF_USE_LNG

#include <vector>
#include <string>

/**
 * @brief Get ordered list of LNG plotfile variable names
* @return Vector of 19 variable name strings
 * @details
 * Order:
 *   0: lng_pool_depth       [m]
 *   1: lng_pool_mask        [0/1]
 *   2: lng_evap_flux        [kg/m^2/s]
 *   3: lng_latent_flux      [W/m^2]
 *   4: lng_vapor_conc       [kg/m^3]
 *   5: lng_ustar            [m/s]
 *   6: lng_tsfc             [K]
 *   7: lng_pblh             [m]
 *   8: lng_conc_sfc         [kg/m^3]
 *   9: lng_lfl_mask         [0/1]
 *  10: lng_ufl_mask         [0/1]
 *  11: lng_wind_u           [m/s]
 *  12: lng_wind_v           [m/s]
 *  13: lng_gc_h             [m]
 *  14: lng_gc_u             [m/s]
 *  15: lng_gc_v             [m/s]
 *  16: lng_gc_ri_flag       [0/1]
*  17: lng_conc_1h_avg      [kg/m^3]  (Phase 7)
*  18: lng_exceed_flag      [0/1]     (Phase 7)
*/
inline std::vector<std::string> lng_plotfile_var_names()
{
   return {
        "lng_pool_depth",    // 0  [m]
        "lng_pool_mask",     // 1  [0/1]
        "lng_evap_flux",     // 2  [kg/m^2/s]
        "lng_latent_flux",   // 3  [W/m^2]
        "lng_vapor_conc",    // 4  [kg/m^3]
        "lng_ustar",         // 5  [m/s]
        "lng_tsfc",          // 6  [K]
        "lng_pblh",          // 7  [m]
        "lng_conc_sfc",      // 8  [kg/m^3]
        "lng_lfl_mask",      // 9  [0/1]
        "lng_ufl_mask",      // 10 [0/1]
        "lng_wind_u",        // 11 [m/s]
        "lng_wind_v",        // 12 [m/s]
        "lng_gc_h",          // 13 [m]
        "lng_gc_u",          // 14 [m/s]
        "lng_gc_v",          // 15 [m/s]
        "lng_gc_ri_flag",    // 16 [0/1]
        "lng_conc_1h_avg",   // 17 [kg/m^3]
        "lng_exceed_flag"    // 18 [0/1]
    };
}

/**
 * @brief Get number of LNG plotfile components
 * @return 19 (number of output variables)
 */
inline int lng_plotfile_ncomp()
{
    return 19;
}

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_PLOTFILE_CATALOG_H */
