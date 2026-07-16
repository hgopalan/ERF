/**
 * @file ERF_LNGPlotfile.H
 * @brief LNG plotfile output (Phase 6+)
 * @details
 * Declares functions for writing LNG state to native 2D grid plotfiles using VisMF.
 * Implements the MPI-safe plotfile pattern from Source/Dust/ERF_DustPlotfile.cpp.
 * 
 * Pattern: Source/Dust/ERF_DustPlotfile.H
 */

#ifndef ERF_LNG_PLOTFILE_H
#define ERF_LNG_PLOTFILE_H

#ifdef ERF_USE_LNG

#include <string>

class LNGLayer;

/**
 * @brief Write LNG state to plotfile on native 2D grid
 * @param[in] plotfile_prefix    Prefix for plotfile name (e.g., "plt_lng_")
 * @param[in] lng_layer          LNG layer containing all fields
 * @param[in] time               Simulation time [s]
 * @param[in] step               Timestep number
 * @details
 * Phase 6: writes all 17 LNG MultiFabs to native LNG grid plotfile using VisMF.
 * Each plotfile is named plotfile_prefix + NNNNN (e.g., plt_lng_00005).
 * Creates a hierarchical directory structure:
 *   plotfile_prefix_NNNNN/Header           (AMReX header)
 *   plotfile_prefix_NNNNN/Level_0/Cell    (binary VisMF data)
 *   plotfile_prefix_NNNNN/LNGMetadata.json (JSON sidecar with metadata)
 * 
 * MPI rules: ALL ranks call VisMF::Write (MPI-collective). IOProcessor writes
 * header and JSON. Barriers protect directory creation and MPI writes.
 * See LNG_MPI_SKILLS.md Rule B1 (IOProcessor guard order).
 */
void WriteLNGPlotfile(const std::string& plotfile_prefix,
                       const LNGLayer& lng_layer,
                       double time, int step);

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_PLOTFILE_H */
