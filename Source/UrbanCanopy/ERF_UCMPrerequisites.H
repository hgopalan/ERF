/**
 * @file ERF_UCMPrerequisites.H
 * @brief Prerequisite verification for the ERF-SLUCM module
 *
 * Declares a function that verifies the ERF configuration meets all
 * requirements for SLUCM initialization, performed once at startup before
 * SLUCM data allocation.
 *
 * Prerequisites include:
 *  - AMR level constraints (anchor_level must be within [0, finest_level])
 *  - Parameter validity ranges (grid spacing, coupling strength, angles)
 *  - Grid decomposition constraints (no z-direction splitting)
 *  - Terrain availability if use_terrain is enabled
 *  - Buildable morphology parameters (all positive dimensions)
 *
 * References:
 *  - Source/Dust/DUST_DEVELOPMENT.md (ERF-Hazard)
 *  - Source/LNG/LNG_DEVELOPMENT.md (ba55b73...)
 *  - Source/LNG/LNG_MPI_SKILLS.md (ba55b73...)
 */

#ifndef ERF_UCM_PREREQUISITES_H
#define ERF_UCM_PREREQUISITES_H

#include <ERF_UCMParams.H>

// Forward declarations
struct UCMGrid;
struct UCMFields;

/**
 * @brief Verify ERF configuration requirements for the SLUCM module
 *
 * Called once at startup before SLUCM initialization. Aborts with clear,
 * actionable diagnostic messages if any requirement is not met.
 *
 * Checks performed (Phase 1.1 only):
 *  1. anchor_level >= 0 and anchor_level <= finest_level
 *  2. anchor_level == 0 (higher values gated for Phase 3.1)
 *  3. static_refinement == true (false not supported)
 *  4. grid_ratio >= 1
 *  5. alpha_ucm > 0.0
 *  6. zref > 0.0
 *  7. All dimensions (H_bldg_uniform, W_road_uniform, W_roof_uniform) > 0.0
 *  8. All albedos in [0, 1]; all emissivities in [0, 1]
 *  9. atm_feedback == 0.0 (one-way only in Phase 1; Phase 3.2 relaxes this)
 * 10. Terrain mode active message if use_terrain == true
 * 11. Grid decomposition: no z-direction splitting (all ranks own full z-extent)
 * 12. Print startup banner with all parameter values
 *
 * @param[in] params       SLUCM configuration parameters
 * @param[in] max_level    Maximum AMR level (rarely used; kept for generality)
 * @param[in] finest_level Finest AMR level actually in use
 * @param[in] use_terrain  Whether terrain-following coordinates are enabled
 * @param[in] lev          AMR level to check constraints for (default 0)
 * @throws                 Aborts via AMREX_ALWAYS_ASSERT_WITH_MESSAGE on failure
 */
void check_ucm_prerequisites(const UCMParams& params,
                              int max_level,
                              int finest_level,
                              bool use_terrain,
                              int lev);

/**
 * @brief Resolve wall drag mode based on user input and solver configuration
 *
 * Phase 2.8: auto-resolves wall_drag_mode to explicit (compressible) or implicit
 * (anelastic) based on solverChoice and the requested mode string.
 *
 * @param[in] wall_drag_mode_str User-specified mode string from input
 * @param[in] is_anelastic       True if anelastic mode, false for compressible
 * @param[out] resolved_mode     Resolved WallDragMode enum
 *
 * @return "auto" → explicit (compressible) or implicit (anelastic)
 *         "explicit", "implicit", "off" → as specified
 *         Invalid string → error abort
 */
void resolve_wall_drag_mode(const std::string& wall_drag_mode_str,
                             bool is_anelastic,
                             WallDragMode& resolved_mode);

/**
 * @brief Verify UCM grid and fields are properly allocated (Phase 1.2 post-allocation check)
 *
 * Called after allocate_ucm_fields() to verify all MultiFabs were successfully
 * allocated and to print the Phase 1.2 grid verification banner.
 *
 * @param[in] params      SLUCM configuration parameters
 * @param[in] ucm_grid    Constructed UCM grid (checked for consistency)
 * @param[in] ucm_fields  Allocated UCM MultiFab fields (checked for completeness)
 * @param[in] lev         AMR level (default 0)
 * @throws                Aborts if any field is not allocated
 */
void check_ucm_grid_and_fields(const UCMParams& params,
                                const UCMGrid& ucm_grid,
                                const UCMFields& ucm_fields,
                                int lev);

#endif // ERF_UCM_PREREQUISITES_H
