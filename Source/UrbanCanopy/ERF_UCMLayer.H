/**
 * @file ERF_UCMLayer.H
 * @brief Physics driver class for SLUCM facet SEB and slab conduction
 *
 * UCMLayer orchestrates the per-timestep sequence for the Single-Layer Urban
 * Canopy Model:
 * 1. Extract atmospheric forcing (u*, wind, T, q, SW/LW) into UCMForcing
 * 2. Compute shortwave/longwave radiation (analytical placeholder Phase 1.3)
 * 3. Solve facet SEB (roof/wall/road) via Newton iteration
 * 4. Advance slab conduction (implicit Euler TDMA) per facet
 * 5. Compute net sensible and latent heat fluxes
 * 6. Update canyon-air temperature budget
 *
 * **Per-Facet SEB (Kusaka 2001, Chen 2011):**
 * \f[
 *   R_n - H - LE - G = 0
 * \f]
 *
 * where:
 * - \f$ R_n = (1-\alpha) SW_{\text{abs}} + \varepsilon(LW_{\text{down}} - \sigma T^4) \f$ [W/m²]
 * - \f$ H = \rho c_p C_h |U| (T_{\text{skin}} - T_{\text{ref}}) \f$ [W/m²]
 * - \f$ LE = 0 \f$ [W/m²] (latent heat; Phase 1.3 placeholder)
 * - \f$ G = k_{\text{th}} \frac{T_{\text{skin}} - T_1}{\Delta z / 2} \f$ [W/m²]
 *
 * **Canyon-Air Budget (simplified single-iteration, Kusaka eq. 21):**
 * \f[
 *   T_{\text{canyon}} = \frac{
 *     \frac{H_{\text{road}}}{\rho c_p C_h^{\text{road}}} + 2\frac{H}{W} \frac{H_{\text{wall}}}{\rho c_p C_h^{\text{wall}}} + T_{\text{atm}}
 *   }{1 + 1/C_h^{\text{road}} + 2H/W \cdot 1/C_h^{\text{wall}}}
 * \f]
 *
 * **Debug Output (Phase 1.3 mandatory):**
 * Each call to advance() emits one [UCM][1.3][UCMLayer::advance] trace with:
 * - dt, sim time (seconds)
 * - Summary: min/max skin T (roof/wall/road), canyon-air T, max Newton iters
 *
 * References:
 *  - Kusaka et al. (2001). "A Simple Single-Layer Urban Canopy Model..."
 *  - Chen et al. (2011). "Coupling an Advanced Land Surface-Hydrology Model..."
 *  - WRF phys/module_sf_urban.F
 */

#ifndef ERF_UCM_LAYER_H
#define ERF_UCM_LAYER_H

#include <UrbanCanopy/ERF_UCMParams.H>
#include <UrbanCanopy/ERF_UCMFields.H>
#include <UrbanCanopy/ERF_UCMGrid.H>
#include <UrbanCanopy/ERF_UCMWindExtract.H>
#include <AMReX_Geometry.H>

/**
 * @class UCMLayer
 * @brief Physics integrator for SLUCM facet energy balance and conduction
 *
 * Instantiated once per AMR level (currently only level 0 in Phase 1.3).
 * Holds references to UCMParams and methods to drive per-timestep SLUCM
 * computations.
 *
 * **Per-Cell Operation:**
 * Each (i, j) UCM cell is guarded by `is_urban(i, j, 0) == 1`. If not urban,
 * the cell is skipped (Phase 1.3: all-urban; Phase 4.1: heterogeneous).
 *
 * **Thread Safety:**
 * ParallelFor kernels are GPU-safe. No shared state across cells.
 */
class UCMLayer
{
public:
    /**
     * @brief Construct UCMLayer for a given AMR level
     *
     * Stores a copy of parameters and verifies that the level is valid
     * (Phase 1.1: only level 0 permitted).
     *
     * @param[in] params  SLUCM configuration (copied)
     * @param[in] lev     AMR level
     */
    UCMLayer(const UCMParams& params, int lev);

    /**
     * @brief Advance SLUCM one timestep (main physics driver)
     *
     * Orchestrates the complete per-step sequence:
     * 1. Extract atmospheric forcing (u*, wind, T, q) → UCMForcing
     * 2. Fill radiation fields (analytical placeholder in Phase 1.3)
     * 3. Solve facet SEB → skin temperatures (roof/wall/road)
     * 4. Advance slab conduction → interior layer temperatures
     * 5. Compute sensible/latent fluxes → UCMFields
     * 6. Update canyon-air temperature
     *
     * **Input MultiFabs (all on ATM grid at lev):**
     * - atm_u_star, atm_t_star, atm_q_star: SurfaceLayer diagnostics [m/s, K, kg/kg]
     * - xvel, yvel: ATM wind components [m/s]
     * - z_phys_cc: ATM physical height at cell centers [m]
     * - T_atm_lowest, q_atm_lowest: ATM state at lowest level [K, kg/kg]
     * - geom_atm: ATM Geometry for coordinate transforms
     *
     * **Modified Output (UCMFields):**
     * - T_skin_roof, T_skin_wall, T_skin_road [K]
     * - T_canyon_air [K]
     * - H_sensible, LE_latent [W/m²]
     *
     * **Parameters from UCMParams:**
     * - slab_N_layers, slab_T_deep, slab_L, k_therm_uniform, rho_cp_uniform
     * - newton_max_iter, newton_tol_K
     * - zref, H_bldg_uniform, W_road_uniform, W_roof_uniform
     * - albedo_*, emissivity_*
     *
     * **Null-Safety:**
     * atm_q_star and q_atm_lowest may be empty (dry simulations). All operations
     * check for null pointers before dereferencing.
     *
     * **Debug Output (if ucm_debug=true):**
     * Emits [UCM][1.3][UCMLayer::advance] at end of call, once per IO rank.
     *
     * **Guaranteed Behavior:**
     * - is_urban mask enforced: cells with is_urban==0 skipped
     * - All skin temperatures bounded in [250, 330] K (sanity check)
     * - Canyon air temp computed from road+wall SEB budgets
     * - H_sensible, LE_latent stored in UCMFields for later output
     *
     * @param[in,out] fields       UCMFields to be updated (skin T, canyon T, fluxes)
     * @param[in,out] forcing      UCMForcing container (populated during call)
     * @param[in]     ucm_grid     UCM grid (BoxArray, Geometry, refinement ratio)
     * @param[in]     atm_u_star   ATM friction velocity from SurfaceLayer [m/s]
     * @param[in]     atm_t_star   ATM potential temp scale from SurfaceLayer [K]
     * @param[in]     atm_q_star   ATM humidity scale from SurfaceLayer [kg/kg]
     *                             (may be empty)
     * @param[in]     xvel         ATM u-component [m/s]
     * @param[in]     yvel         ATM v-component [m/s]
     * @param[in]     z_phys_cc    ATM physical height at cell centers [m]
     * @param[in]     T_atm_lowest ATM temperature at lowest level [K]
     * @param[in]     q_atm_lowest ATM water vapor at lowest level [kg/kg]
     *                             (may be empty)
     * @param[in]     geom_atm     ATM Geometry (for coord transforms, ref level)
     * @param[in]     time         Simulation time elapsed [s]
     * @param[in]     dt           Timestep size [s]
     * @param[in]     nz_atm       Number of vertical ATM levels (for bounds checks)
     * @param[in]     lev          AMR level (default 0)
     *
     * @return void
     *
     * **Thread Safety:**
     * ParallelFor kernels are GPU-safe. No static data mutated. Safe to call
     * from multiple AMR levels (if Phase 3.1 enables multi-level).
     */
    void advance(UCMFields& fields,
                 UCMForcing& forcing,
                 const UCMGrid& ucm_grid,
                 const amrex::MultiFab& atm_u_star,
                 const amrex::MultiFab& atm_t_star,
                 const amrex::MultiFab& atm_q_star,
                 const amrex::MultiFab& xvel,
                 const amrex::MultiFab& yvel,
                 const amrex::MultiFab& z_phys_cc,
                 const amrex::MultiFab& T_atm_lowest,
                 const amrex::MultiFab& q_atm_lowest,
                 const amrex::MultiFab& atm_olen,
                 const amrex::Geometry& geom_atm,
                 amrex::Real time,
                 amrex::Real dt,
                 int nz_atm,
                 int lev);

private:
    UCMParams m_params;  ///< Copy of SLUCM parameters (immutable during run)
    int m_lev;           ///< AMR level this layer operates on (Phase 1.1: always 0)
    bool m_warn_radiation_placeholder_printed = false;  ///< One-time warning flag
};

#endif  // ERF_UCM_LAYER_H
