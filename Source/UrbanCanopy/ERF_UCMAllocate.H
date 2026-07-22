/**
 * @file ERF_UCMAllocate.H
 * @brief SLUCM MultiFab allocation and initialization interface
 *
 * Provides functions to allocate all MultiFab fields on the UCM grid and
 * populate them with initial values (homogeneous in Phase 1.2, heterogeneous
 * CSV-read in Phase 2.1+).
 *
 * References:
 *  - Source/Dust/ERF_DustLayer.H (allocation pattern)
 *  - Source/LNG/ERF_LNGLayer.H (field setup pattern)
 */

#ifndef ERF_UCM_ALLOCATE_H
#define ERF_UCM_ALLOCATE_H

#include <ERF_UCMFields.H>
#include <ERF_UCMGrid.H>
#include <ERF_UCMParams.H>

/**
 * @brief Allocate all UCMFields MultiFabs on the UCM grid
 *
 * Allocates every field in the UCMFields struct on the provided UCM BoxArray
 * and DistributionMapping. All MultiFabs are created with:
 *  - ncomp = 1 (scalar field per cell)
 *  - ngrow = IntVect(1, 1, 0) (1 ghost cell in x,y; 0 in z per Phase 1.2 contract)
 *  - Initialized to zero
 *
 * Debug output is emitted per-MultiFab when params.ucm_debug is true,
 * prefixed as [UCM][1.2][allocate_ucm_fields].
 *
 * @param[out] fields       UCMFields struct to populate
 * @param[in] ucm_grid      UCM grid (contains ba, dm for allocation)
 * @param[in] params        UCM parameters (for debug output control)
 * @param[in] lev           AMR level (default 0; Phase 1.1 only uses lev=0)
 *
 * @note Postcondition: all_allocated() == true
 * @note All MPI ranks participate in allocation (collective operation)
 */
void allocate_ucm_fields(UCMFields& fields,
                         const UCMGrid& ucm_grid,
                         const UCMParams& params,
                         int lev = 0);

/**
 * @brief Fill all UCMFields with heterogeneous values from CSV data
 *
 * Phase 2.1 heterogeneous fill pattern:
 *  - Maps each (i_ucm, j_ucm) to ATM coordinates (i_atm, j_atm) using grid_ratio
 *  - Looks up corresponding UCMBuildingRow from building_layout_csv
 *  - Retrieves material properties from material_library_csv via mat_ids
 *  - Writes H_bldg, W_road, W_roof, albedo_*, emissivity_*, is_urban to MultiFabs
 *
 * Debug output emitted per-unique-mat_id when params.ucm_debug is true,
 * prefixed as [UCM][2.1][fill_ucm_fields_from_csv].
 *
 * @param[in,out] fields            UCMFields struct to fill
 * @param[in] ucm_grid              UCM grid (needed for domain information)
 * @param[in] building_reader       Building layout reader (broadcast data)
 * @param[in] material_registry     Material library registry (broadcast data)
 * @param[in] grid_ratio            UCM-to-ATM refinement ratio
 * @param[in] lev                   AMR level (default 0; Phase 1.1 only uses lev=0)
 * @param[in] ucm_debug             Debug flag for output
 *
 * @precondition all_allocated() == true (fields already allocated)
 * @precondition building_reader.read_and_broadcast() completed
 * @precondition material_registry.load_and_broadcast() completed
 * @postcondition all MultiFabs contain heterogeneous values from CSV
 * @throws Aborts if building row lookup fails or material lookup fails
 */
void fill_ucm_fields_from_csv(UCMFields& fields,
                              const UCMGrid& ucm_grid,
                              const UCMBuildingLayoutReader& building_reader,
                              const UCMMaterialRegistry& material_registry,
                              int grid_ratio,
                              int lev = 0,
                              bool ucm_debug = false);

/**
 * @brief Fill all UCMFields with homogeneous values from ParmParse
 *
 * Sets every field to its uniform value from UCMParams. Called once after
 * allocate_ucm_fields() to establish initial homogeneous urban patch.
 *
 * Phase 1.2 fill pattern (homogeneous):
 *  - H_bldg, W_road, W_roof ← params.H_bldg_uniform, etc.
 *  - albedo_roof, albedo_wall, albedo_road ← params.albedo_*
 *  - emissivity_roof, emissivity_wall, emissivity_road ← params.emissivity_*
 *  - T_skin_roof, T_skin_wall, T_skin_road ← params.test_surf_temp_K
 *  - T_canyon_air ← params.test_surf_temp_K
 *  - H_sensible, LE_latent ← 0.0
 *  - is_urban ← 1 (everywhere — fully urban patch)
 *
 * Phase 2.1+ heterogeneous fill (NOT Phase 1.2):
 *  - To be implemented: read per-cell values from CSV and scatter to fields
 *
 * Debug output emitted per-field when params.ucm_debug is true,
 * prefixed as [UCM][1.2][fill_ucm_fields_homogeneous].
 *
 * @param[in,out] fields    UCMFields struct to fill
 * @param[in] params        UCM parameters (values and debug flag)
 * @param[in] lev           AMR level (default 0; Phase 1.1 only uses lev=0)
 *
 * @precondition all_allocated() == true (fields already allocated)
 * @postcondition all MultiFabs contain uniform values from params
 */
void fill_ucm_fields_homogeneous(UCMFields& fields,
                                  const UCMParams& params,
                                  int lev = 0);

#endif // ERF_UCM_ALLOCATE_H
