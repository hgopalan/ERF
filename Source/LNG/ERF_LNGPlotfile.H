/**
 * @file ERF_LNGPlotfile.H
 * @brief LNG plotfile output (Phase 6+)
 * @details
 * Header-only stubs for writing LNG fields to plotfiles.
 * These will be implemented in Phase 6 when visualization output is added.
 * 
 * @see Source/Dust/ERF_DustPlotfile.H — Dust module plotfile pattern (reference)
 */

#ifndef ERF_LNG_PLOTFILE_H
#define ERF_LNG_PLOTFILE_H

#ifdef ERF_USE_LNG

#include <string>

class LNGLayer;

/**
 * @brief Write LNG state to plotfile
 * @param[in] plotfile_prefix    Prefix for plotfile name (e.g., "plt_lng_")
 * @param[in] lng_layer          LNG layer containing all fields
 * @param[in] time               Simulation time [s]
 * @param[in] step               Timestep number
 * @details
 * Phase 1: stub (does nothing).
 * Phase 6: will write all LNG MultiFabs to native LNG grid plotfile using VisMF.
 *          Each plotfile will be named plotfile_prefix + NNNNN (e.g., plt_lng_00000).
 */
inline void write_lng_plotfile(const std::string& plotfile_prefix,
                               const LNGLayer&    lng_layer,
                               double             time,
                               int                step)
{
    // Phase 1: stub
}

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_PLOTFILE_H */
