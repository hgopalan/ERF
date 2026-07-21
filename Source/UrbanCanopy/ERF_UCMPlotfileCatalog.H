/**
 * @file ERF_UCMPlotfileCatalog.H
 * @brief Component catalog for UCM plotfile output
 *
 * Enumerates all MultiFab components written to plt_ucm_NNNNN files
 * on the native UCM grid. Used by both write and read routines.
 *
 * References:
 *  - Source/Dust/ERF_DustPlotfileCatalog.H
 */

#ifndef ERF_UCM_PLOTFILE_CATALOG_H
#define ERF_UCM_PLOTFILE_CATALOG_H

/**
 * @enum UCMPlotfileComponents
 * @brief Component indices for UCM plotfile output
 *
 * All fields are on the native UCM 2D slab grid.
 * Total ncomp = 19 in Phase 1.4.
 */
enum UCMPlotfileComponents {
    // Morphology (1–3)
    UCMPlot_H_bldg           = 0,   ///< Building height [m]
    UCMPlot_W_road           = 1,   ///< Road width (canyon street) [m]
    UCMPlot_W_roof           = 2,   ///< Roof width (building footprint) [m]

    // Shortwave albedo (3–5)
    UCMPlot_albedo_roof      = 3,   ///< Roof SW albedo [-]
    UCMPlot_albedo_wall      = 4,   ///< Wall SW albedo [-]
    UCMPlot_albedo_road      = 5,   ///< Road SW albedo [-]

    // Longwave emissivity (6–8)
    UCMPlot_emissivity_roof  = 6,   ///< Roof LW emissivity [-]
    UCMPlot_emissivity_wall  = 7,   ///< Wall LW emissivity [-]
    UCMPlot_emissivity_road  = 8,   ///< Road LW emissivity [-]

    // Surface temperatures (9–11)
    UCMPlot_T_skin_roof      = 9,   ///< Roof skin temperature [K]
    UCMPlot_T_skin_wall      = 10,  ///< Wall skin temperature [K]
    UCMPlot_T_skin_road      = 11,  ///< Road skin temperature [K]

    // Canyon state (12)
    UCMPlot_T_canyon_air     = 12,  ///< Canyon-air temperature [K]

    // Heat fluxes (13–14)
    UCMPlot_H_sensible       = 13,  ///< Sensible heat flux [W/m²]
    UCMPlot_LE_latent        = 14,  ///< Latent heat flux [W/m²] (zero in Phase 1.4)

    // Mask (15)
    UCMPlot_is_urban         = 15,  ///< Urban mask (0/1) [-]

    // Total
    UCMPlot_ncomp            = 16   ///< Total number of components
};

/**
 * @brief Return component name for plotfile header
 *
 * Used to label each component in the plotfile ASCII header.
 *
 * @param[in] comp  Component index (UCMPlotfileComponents)
 * @return          String name (e.g., "H_bldg", "T_skin_roof")
 */
inline const char* UCMPlotfileComponentName(int comp)
{
    switch (comp) {
        case UCMPlot_H_bldg:           return "H_bldg";
        case UCMPlot_W_road:           return "W_road";
        case UCMPlot_W_roof:           return "W_roof";
        case UCMPlot_albedo_roof:      return "albedo_roof";
        case UCMPlot_albedo_wall:      return "albedo_wall";
        case UCMPlot_albedo_road:      return "albedo_road";
        case UCMPlot_emissivity_roof:  return "emissivity_roof";
        case UCMPlot_emissivity_wall:  return "emissivity_wall";
        case UCMPlot_emissivity_road:  return "emissivity_road";
        case UCMPlot_T_skin_roof:      return "T_skin_roof";
        case UCMPlot_T_skin_wall:      return "T_skin_wall";
        case UCMPlot_T_skin_road:      return "T_skin_road";
        case UCMPlot_T_canyon_air:     return "T_canyon_air";
        case UCMPlot_H_sensible:       return "H_sensible";
        case UCMPlot_LE_latent:        return "LE_latent";
        case UCMPlot_is_urban:         return "is_urban";
        default:                       return "unknown";
    }
}

#endif  // ERF_UCM_PLOTFILE_CATALOG_H
