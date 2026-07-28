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
 * Total ncomp = 25 in Phase 5.1a.
 */
enum UCMPlotfileComponents {
    // Morphology (0–2)
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

    // Phase 2.4: Sky view factors (single-bounce shadowing) (16–18)
    UCMPlot_SVF_wall         = 16,  ///< Wall sky view factor [-]
    UCMPlot_SVF_road         = 17,  ///< Road sky view factor [-]
    UCMPlot_SVF_roof         = 18,  ///< Roof sky view factor [-] (always 1.0)

    // Phase 5.1a: Multi-facet view factors (Hottel crossed-string geometry) (19–24)
    // Consumed by Phase 5.1b radiosity solver; unused at Phase 5.1a but plotted
    // for validation and debugging (per Contract #16).
    UCMPlot_F_wall_sky       = 19,  ///< Wall→sky view factor [-]
    UCMPlot_F_wall_wall      = 20,  ///< Wall→opposite-wall view factor [-]
    UCMPlot_F_wall_road      = 21,  ///< Wall→road view factor [-]
    UCMPlot_F_road_sky       = 22,  ///< Road→sky view factor [-]
    UCMPlot_F_road_wall      = 23,  ///< Road→one-wall view factor [-]
    UCMPlot_F_roof_sky       = 24,  ///< Roof→sky view factor [-] (always 1.0)

    // Total (Phase 5.1a)
    UCMPlot_ncomp            = 25   ///< Total number of components (Phase 5.1a)
};

/**
 * @enum ATMPlotfileComponents
 * @brief Component indices for ATM-grid aggregates in UCM plotfile (Phase 2.5)
 *
 * These are computed per ATM cell from the UCM 2D slab.
 * Total ncomp = 5 in Phase 2.5.
 * Written to a separate section of the plotfile or separate file if needed.
 */
enum ATMPlotfileComponents {
    ATMPlot_f_urb_atm         = 0,   ///< Urban fraction per ATM cell [0,1] (Phase 2.5)
    ATMPlot_H_bldg_mean_atm   = 1,   ///< Mean building height per ATM cell [m] (Phase 2.5)
    ATMPlot_H_bldg_std_atm    = 2,   ///< Std dev building height per ATM cell [m] (Phase 2.5)
    ATMPlot_lambda_p_atm      = 3,   ///< Mean plan-area density per ATM cell [-] (Phase 2.5)
    ATMPlot_lambda_f_atm      = 4,   ///< Mean frontal-area density per ATM cell [-] (Phase 2.5)

    // Total (Phase 2.5)
    ATMPlot_ncomp             = 5    ///< Total number of ATM-grid components (Phase 2.5)
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
        case UCMPlot_SVF_wall:         return "SVF_wall";
        case UCMPlot_SVF_road:         return "SVF_road";
        case UCMPlot_SVF_roof:         return "SVF_roof";
        case UCMPlot_F_wall_sky:       return "F_wall_sky";
        case UCMPlot_F_wall_wall:      return "F_wall_wall";
        case UCMPlot_F_wall_road:      return "F_wall_road";
        case UCMPlot_F_road_sky:       return "F_road_sky";
        case UCMPlot_F_road_wall:      return "F_road_wall";
        case UCMPlot_F_roof_sky:       return "F_roof_sky";
        default:                       return "unknown";
    }
}

#endif  // ERF_UCM_PLOTFILE_CATALOG_H
