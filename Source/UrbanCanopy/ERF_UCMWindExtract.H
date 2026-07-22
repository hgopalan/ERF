/**
 * @file ERF_UCMWindExtract.H
 * @brief Declarations for terrain-aware wind and scalar extraction from ATM to UCM
 *
 * Three extraction functions mirror the Dust module pattern (ERF_DustWindExtract.H):
 * 1. `fill_ucm_ustar_from_surface_layer` — Copy u* from SurfaceLayer
 * 2. `fill_ucm_wind_from_interpolation` — Log-law interpolation to z_target
 * 3. `fill_ucm_scalar_from_atm` — Copy scalars (T, q) from ATM lowest level
 *
 * All functions are terrain-aware: wind target height is computed as
 * z_target = z_phys_cc(i,j,klo) + H_bldg(i,j) + zref, where:
 * - z_phys_cc(i,j,klo) = physical height of ATM lowest cell center
 * - H_bldg(i,j) = building height (from UCMFields)
 * - zref = reference height above roof (from UCMParams)
 *
 * **Radiation Placeholder Policy (Phase 1.3):**
 * SW/LW radiation coupling not yet available (Phase 4.2). This file declares
 * a stub struct UCMForcing to hold forcing arrays. SW/LW are filled analytically
 * in ERF_UCMLayer.cpp with a diurnal cycle. A one-time warning is emitted.
 *
 * References:
 *  - Source/Dust/ERF_DustWindExtract.H (structural pattern)
 *  - Source/LNG/LNG_DEVELOPMENT.md (ATM coupling examples)
 *  - WRF: phys/module_sf_urban.F (Kusaka 2001, Chen 2011)
 */

#ifndef ERF_UCM_WIND_EXTRACT_H
#define ERF_UCM_WIND_EXTRACT_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <memory>

// Forward declarations
struct UCMGrid;
struct UCMParams;

/**
 * @struct UCMForcing
 * @brief Container for atmospheric forcing data extracted into UCM
 *
 * All fields are 2D MultiFabs on the UCM grid with components as follows:
 * - u_star: 1 component (m/s)
 * - wind_ref: 2 components (u, v) at z_target [m/s]
 * - T_atm_ref: 1 component [K]
 * - q_atm_ref: 1 component [kg/kg], may be empty if moisture not available
 * - SW_down: 1 component [W/m²], placeholder analytical fill in Phase 1.3
 * - LW_down: 1 component [W/m²], placeholder analytical fill in Phase 1.3
 *
 * All allocated with ghost cells IntVect(1,1,0) for efficient MPI communication.
 * Phase 1.3: q_atm_ref and LW_down may be empty (null-safe). Phase 4.2 populates
 * SW/LW from radiation solver.
 */
struct UCMForcing
{
    /// Friction velocity from SurfaceLayer [m/s]
    std::unique_ptr<amrex::MultiFab> u_star;

    /// Wind components (u, v) at z_target (2 components) [m/s]
    /// Computed via log-law interpolation from ATM lowest level
    std::unique_ptr<amrex::MultiFab> wind_ref;

    /// Atmospheric temperature at z_target [K]
    /// Extracted from ATM lowest cell; scaling for height via potential temperature
    /// not done yet (Phase 1.4+); currently copy of lowest level
    std::unique_ptr<amrex::MultiFab> T_atm_ref;

    /// Atmospheric water vapor [kg/kg]
    /// Extracted from ATM lowest cell. May be empty (null pointer) if moisture
    /// not available in simulation. Phase 1.3: null-safe operations required.
    std::unique_ptr<amrex::MultiFab> q_atm_ref;

    /// Downwelling shortwave radiation [W/m²]
    /// Phase 1.3: filled analytically with diurnal cycle
    /// Phase 4.2: replaced with radiation scheme extraction
    std::unique_ptr<amrex::MultiFab> SW_down;

    /// Downwelling longwave radiation [W/m²]
    /// Phase 1.3: filled analytically with constant 350 W/m²
    /// Phase 4.2: replaced with radiation scheme extraction
    std::unique_ptr<amrex::MultiFab> LW_down;

    /**
     * @brief Check that all allocated forcing arrays are present (null-safe)
     *
     * Returns true if at minimum u_star, wind_ref, T_atm_ref, SW_down, LW_down
     * are allocated. q_atm_ref may be null (accepted).
     *
     * @return true if minimal forcing set is allocated; false otherwise
     */
    bool all_allocated() const;

    /**
     * @brief Deallocate all forcing arrays
     */
    void clear();
};

/**
 * @brief Extract friction velocity u* from SurfaceLayer into UCM
 *
 * Simple copy of SurfaceLayer u* (computed by MRF, YSU, MYNN2.5) onto UCM grid.
 * No interpolation; bilinear mapping from ATM to UCM cells.
 *
 * **Debug Trace (if ucm_debug=true):**
 * Emits [UCM][1.3][fill_ucm_ustar_from_surface_layer] min/max u*.
 *
 * @param[out] ucm_ustar      Target UCM u* MultiFab [m/s]
 * @param[in]  atm_u_star     Source ATM u* MultiFab from SurfaceLayer [m/s]
 * @param[in]  ucm_grid       UCM grid geometry and layout
 * @param[in]  lev            AMR level (default 0)
 */
void fill_ucm_ustar_from_surface_layer(amrex::MultiFab& ucm_ustar,
                                       const amrex::MultiFab& atm_u_star,
                                       const UCMGrid& ucm_grid,
                                       int lev = 0);

/**
 * @brief Interpolate ATM wind to z_target via log-law profile with per-cell z0 and displacement
 *
 * Extracts horizontal wind (u, v) from ATM at lowest level and interpolates
 * vertically to target height using Monin-Obukhov similarity (log-law under
 * neutral conditions, or with Obukhov length L when available).
 *
 * **Target Height (terrain-aware):**
 * \f[
 *   z_{\text{target}} = z_{\text{phys\_cc}}(i,j,klo) + H_{\text{bldg}}(i,j) + z_{\text{ref}}
 * \f]
 *
 * where klo is the lowest ATM cell index (typically klo=0 for non-terrain,
 * or adjusted for terrain-following coords).
 *
 * **Interpolation (neutral, log-law with displacement height):**
 * \f[
 *   U(z) = \frac{u_*}{\kappa} \ln\left(\frac{z - d + z_0}{z_0}\right)
 * \f]
 *
 * Roughness length z_0 and displacement height d are now per-cell (Phase 2.2);
 * they come from z0_ucm and d_disp_ucm MultiFabs.
 * Obukhov-length stability correction deferred to Phase 1.4.
 *
 * **Debug Trace (if ucm_debug=true):**
 * Emits [UCM][1.3][fill_ucm_wind_from_interpolation] z_target at (0,0);
 * min/max |wind|.
 *
 * @param[out] ucm_wind_ref   Target UCM wind MultiFab (2 components: u, v) [m/s]
 * @param[in]  xvel           ATM u-velocity [m/s]
 * @param[in]  yvel           ATM v-velocity [m/s]
 * @param[in]  z_phys_cc      ATM physical z at cell centers [m]
 * @param[in]  H_bldg         Building height from UCMFields [m]
 * @param[in]  z0_ucm         Per-cell aerodynamic roughness [m] (Phase 2.2)
 * @param[in]  d_disp_ucm     Per-cell displacement height [m] (Phase 2.2)
 * @param[in]  zref           Reference height above roof from UCMParams [m]
 * @param[in]  ucm_grid       UCM grid geometry and layout
 * @param[in]  nz_atm         Number of vertical ATM levels
 * @param[in]  lev            AMR level (default 0)
 */
void fill_ucm_wind_from_interpolation(amrex::MultiFab& ucm_wind_ref,
                                     const amrex::MultiFab& xvel,
                                     const amrex::MultiFab& yvel,
                                     const amrex::MultiFab& z_phys_cc,
                                     const amrex::MultiFab& H_bldg,
                                     const amrex::MultiFab& z0_ucm,
                                     const amrex::MultiFab& d_disp_ucm,
                                     amrex::Real zref,
                                     const UCMGrid& ucm_grid,
                                     int nz_atm,
                                     int lev = 0);

/**
 * @brief Extract atmospheric scalars (T, q) from lowest ATM level to UCM
 *
 * Copies scalar fields from ATM lowest cell level to UCM grid via
 * interpolation/averaging. Supports generic scalar comp index for flexibility.
 *
 * **Phase 1.3 Usage:**
 * - Temperature: comp=0 (assume state_comp.Temp = 0)
 * - Water vapor: comp=1 (if available; may be empty)
 * - Other: general framework (Phase 2+)
 *
 * **Null-Safety:**
 * If atm_scalar is empty (e.g., no moisture in dry simulations), this function
 * should gracefully return or fill with default values (Phase 1.3 TBD in caller).
 *
 * **Debug Trace (if ucm_debug=true):**
 * Emits [UCM][1.3][fill_ucm_scalar_from_atm] min/max scalar.
 *
 * @param[out] ucm_scalar    Target UCM scalar MultiFab
 * @param[in]  atm_scalar    Source ATM scalar MultiFab (e.g., T, q)
 * @param[in]  ucm_grid      UCM grid geometry and layout
 * @param[in]  geom_atm      ATM Geometry (for coordinate transformations)
 * @param[in]  comp          Component index in atm_scalar (default 0)
 * @param[in]  lev           AMR level (default 0)
 */
void fill_ucm_scalar_from_atm(amrex::MultiFab& ucm_scalar,
                              const amrex::MultiFab& atm_scalar,
                              const UCMGrid& ucm_grid,
                              const amrex::Geometry& geom_atm,
                              int comp = 0,
                              int lev = 0);

#endif  // ERF_UCM_WIND_EXTRACT_H
