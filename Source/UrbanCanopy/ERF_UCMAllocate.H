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
#include <UrbanCanopy/ERF_UCMBuildingLayoutReader.H>
#include <UrbanCanopy/ERF_UCMMaterialRegistry.H>


/**
 * @brief Allocate all UCMFields MultiFabs on the UCM grid
 *
 * Allocates every field in the UCMFields struct on the provided UCM BoxArray
 * and DistributionMapping. All MultiFabs are created with:
 *  - ncomp = 1 (scalar field per cell, except T_slab_* which have ncomp=N_layers)
 *  - ngrow = IntVect(1, 1, 0) (1 ghost cell in x,y; 0 in z per Phase 1.2 contract)
 *  - Initialized to zero
 *
 * Debug output is emitted per-MultiFab when params.ucm_debug is true,
 * prefixed as [UCM][1.2][allocate_ucm_fields].
 *
 * @param[out] fields       UCMFields struct to populate
 * @param[in] ucm_grid      UCM grid (contains ba, dm for allocation)
 * @param[in] params        UCM parameters (for debug output control and slab layer count)
 * @param[in] lev           AMR level (default 0; Phase 1.1 only uses lev=0)
 *
 * @note Postcondition: all_allocated() == true
 * @note All MPI ranks participate in allocation (collective operation)
 */
void allocate_ucm_fields(UCMFields& fields,
                         const UCMGrid& ucm_grid,
                         const UCMParams& params,
                         int lev);

/**
 * @brief Fill all UCMFields with heterogeneous values from CSV data
 *
 * Phase 2.1 heterogeneous fill pattern:
 *  - Maps each (i_ucm, j_ucm) to its corresponding UCMBuildingRow
 *  - Retrieves material properties from material_library_csv via mat_ids
 *  - Writes H_bldg, W_road, W_roof, albedo_*, emissivity_*, is_urban to MultiFabs
 *  - Populates thermal properties and slab temperatures from material registry
 *
 * Debug output emitted when ucm_debug is true,
 * prefixed as [UCM][2.1][fill_ucm_fields_from_csv].
 *
 * @param[in,out] fields            UCMFields struct to fill
 * @param[in] ucm_grid              UCM grid (needed for domain information)
 * @param[in] building_reader       Building layout reader (broadcast data)
 * @param[in] material_registry     Material library registry (broadcast data)
 * @param[in] params                UCM parameters (temperature initializers, debug flag)
 * @param[in] grid_ratio            UCM-to-ATM refinement ratio (deprecated; kept for ABI stability)
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
                              const UCMParams& params,
                              int grid_ratio,
                              int lev,
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
 *  - T_skin_roof, T_skin_wall, T_skin_road ← params.T_skin_init_K
 *  - T_canyon_air ← params.T_canyon_init_K
 *  - T_slab_roof, T_slab_wall, T_slab_road ← params.T_skin_init_K (all layers)
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
                                  int lev);

/**
 * @brief Fill z0_ucm and d_disp_ucm from building height and morphology ratios
 *
 * Phase 2.2 new function:
 *  - z0_ucm(i,j) = z0_over_H * H_bldg(i,j) for urban cells
 *  - d_disp_ucm(i,j) = d_over_H * H_bldg(i,j) for urban cells
 *  - Non-urban cells: z0_ucm = 0.1 m (MOST default), d_disp_ucm = 0.0 m
 *
 * Called AFTER fill_ucm_fields_homogeneous or fill_ucm_fields_from_csv.
 *
 * @param[in,out] fields    UCMFields struct to fill
 * @param[in] params        UCM parameters (z0_over_H, d_over_H, debug flag)
 * @param[in] lev           AMR level (default 0; Phase 1.1 only uses lev=0)
 *
 * @precondition all_allocated() == true (fields already allocated)
 * @precondition H_bldg and is_urban already filled
 * @postcondition z0_ucm and d_disp_ucm contain per-cell values
 */
void fill_ucm_z0_and_disp(UCMFields& fields,
                          const UCMParams& params,
                          int lev);

/**
 * @brief Compute anthropogenic heat (AH) from profile type and time of day
 *
 * Phase 2.3 new function:
 *  - For urban cells (is_urban == 1):
 *    - Profile type 0 (uniform): AH = AH_uniform_Wm2 [constant]
 *    - Profile type 1 (diurnal): AH = AH_daytime_peak * max(0, cos(phase)) [W/m²]
 *      where phase = 2π * time / 86400 - π/2
 *  - Phase 2.9: per-cell override from AH_Wm2_ucm if > 0 (no diurnal scaling)
 *  - For non-urban cells: AH = 0
 *
 * Called once per timestep in UCMLayer::advance() BEFORE the SEB kernel.
 *
 * @param[out] AH_out           Anthropogenic heat MultiFab to fill [W/m²]
 * @param[in] ah_profile_id     Per-cell profile type (0=uniform, 1=diurnal)
 * @param[in] is_urban          Urban mask (0 or 1)
 * @param[in] AH_Wm2_ucm        Per-cell AH override [W/m²] (Phase 2.9); 0 → use ParmParse fallback
 * @param[in] params            UCM parameters (AH_uniform_Wm2, AH_daytime_peak, debug flag)
 * @param[in] time              Current simulation time [seconds since start]
 * @param[in] lev               AMR level (default 0; Phase 1.1 only uses lev=0)
 *
 * @precondition AH_out allocated
 * @precondition ah_profile_id, is_urban, and AH_Wm2_ucm already populated
 * @postcondition AH_out contains per-cell anthropogenic heat [W/m²]
 */
void compute_anthropogenic_heat(amrex::MultiFab&        AH_out,
                                const amrex::iMultiFab& ah_profile_id,
                                const amrex::iMultiFab& is_urban,
                                const amrex::MultiFab&  AH_Wm2_ucm,
                                const UCMParams&        params,
                                amrex::Real             time,
                                int                     lev);

#endif // ERF_UCM_ALLOCATE_H