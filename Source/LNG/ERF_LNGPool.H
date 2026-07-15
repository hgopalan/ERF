/**
 * @file ERF_LNGPool.H
 * @brief LNG pool geometry and spreading dynamics
 * @details
 * Implements pool depth evolution and mask updates for LNG spreading model.
 * Includes functions for:
 * - Pool mask generation from depth threshold
 * - Spill source application (add liquid to pool region)
 * - Evaporation depletion (remove liquid due to vaporization)
 * - Mass and area tracking (diagnostics)
 * 
 * Pool depth evolution:
 *   dh/dt = (spill_rate / rho_LNG) / A_cell - (F_evap / rho_LNG)
 * 
 * where:
 *   h = pool depth [m]
 *   spill_rate = liquid mass rate into pool [kg/s]
 *   rho_LNG = liquid density [kg/m^3]
 *   A_cell = grid cell area [m^2]
 *   F_evap = evaporation flux [kg/m^2/s]
 * 
 * Model references:
 * - Webber, D.M., & Brighton, P.W.M. (1987). An integral model of spreading vaporizing pools.
 * - Didden, N. (1982). Gravity currents over porous substrates.
 * 
 * @note Phase 2: Simplified pool (no gravity current PDE, no radial spreading).
 *       Phase 5: Will add gravity-inertia current dynamics and radius evolution.
 * 
 * @see Source/LNG/ERF_LNGEvaporation.H — Evaporation kernel
 * @see Source/LNG/ERF_LNGLayer.cpp — Integration into advance() sequence
 */

#ifndef ERF_LNG_POOL_H
#define ERF_LNG_POOL_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Print.H>

/**
 * @brief Update pool mask based on depth threshold
 * @param[in,out] lng_pool_mask    Pool mask [0/1]
 * @param[in]     lng_pool_depth   Pool depth [m]
 * @param[in]     depth_threshold  Threshold for mask = 1 [m]
 * 
 * @details
 * Sets mask[i,j,k] = 1.0 if depth[i,j,k] > depth_threshold, else 0.0.
 * Typically depth_threshold = 1.0e-4 m (0.1 mm) to filter numerical noise.
 * 
 * @note Called from LNGLayer::advance() Step G.
 */
void update_pool_mask(amrex::MultiFab& lng_pool_mask,
                      const amrex::MultiFab& lng_pool_depth,
                      amrex::Real depth_threshold = 1.0e-4);

/**
 * @brief Apply spill source: add liquid to pool region
 * @param[in,out] lng_pool_depth     Pool depth [m]
 * @param[in]     geom_lng           LNG grid geometry
 * @param[in]     spill_rate_kg_s    Liquid spill rate [kg/s]
 * @param[in]     rho_LNG            Liquid density [kg/m^3]
 * @param[in]     pool_area_m2       Pool circular area [m^2]
 * @param[in]     cx, cy             Pool center [m]
 * @param[in]     dt                 Timestep [s]
 * 
 * @details
 * Adds liquid uniformly over circular pool region:
 *   dh = (spill_rate_kg_s * dt) / (rho_LNG * pool_area_m2 * N_active_cells)
 * 
 * Only cells within radius = sqrt(pool_area/pi) of (cx, cy) receive source.
 * 
 * @note Called from LNGLayer::advance() Step D.
 */
void apply_spill_source(amrex::MultiFab& lng_pool_depth,
                         const amrex::Geometry& geom_lng,
                         amrex::Real spill_rate_kg_s,
                         amrex::Real rho_LNG,
                         amrex::Real pool_area_m2,
                         amrex::Real cx, amrex::Real cy,
                         amrex::Real dt);

/**
 * @brief Deplete pool depth due to evaporation
 * @param[in,out] lng_pool_depth   Pool depth [m]
 * @param[in]     lng_evap_flux    Evaporation flux [kg/m^2/s]
 * @param[in]     rho_LNG          Liquid density [kg/m^3]
 * @param[in]     dt               Timestep [s]
 * @param[in]     lng_debug        If true, print summary statistics
 * 
 * @details
 * Updates depth by:
 *   h_new = max(0, h_old - (F_evap / rho_LNG) * dt)
 * 
 * Clamped to >= 0 to prevent negative depths (physical).
 * 
 * If lng_debug=true, prints after depletion:
 *   [LNG DEBUG] Phase 2: pool_depth_max=<val> m  pool_depth_min=<val> m  pool_mass=<val> kg  pool_area=<val> m^2
 * 
 * @note Called from LNGLayer::advance() Step F.
 */
void deplete_pool_from_evaporation(amrex::MultiFab& lng_pool_depth,
                                    const amrex::MultiFab& lng_evap_flux,
                                    amrex::Real rho_LNG,
                                    amrex::Real dt,
                                    bool lng_debug);

/**
 * @brief Compute total pool liquid mass [kg]
 * @param[in] lng_pool_depth  Pool depth [m]
 * @param[in] geom_lng        LNG grid geometry
 * @param[in] rho_LNG         Liquid density [kg/m^3]
 * @return Total pool mass [kg]
 * 
 * @details
 * Sum over all cells:
 *   M_pool = sum_i (h_i * rho_LNG * dA_i)
 * where dA_i is cell area [m^2].
 * 
 * Used for mass budget diagnostics and validation.
 * 
 * @note Called from LNGLayer::initialize() and advance() Step H.
 */
amrex::Real compute_pool_mass(const amrex::MultiFab& lng_pool_depth,
                               const amrex::Geometry& geom_lng,
                               amrex::Real rho_LNG);

/**
 * @brief Compute total pool area [m^2]
 * @param[in] lng_pool_mask   Pool mask [0/1]
 * @param[in] geom_lng        LNG grid geometry
 * @return Total pool area [m^2]
 * 
 * @details
 * Sum over all cells:
 *   A_pool = sum_i (mask_i * dA_i)
 * where mask_i = 1 if pool, 0 otherwise.
 * 
 * Used for diagnostics and spreading model (Phase 5+).
 * 
 * @note Called from LNGLayer::initialize() and advance() Step H.
 */
amrex::Real compute_pool_area(const amrex::MultiFab& lng_pool_mask,
                               const amrex::Geometry& geom_lng);

#endif /* ERF_LNG_POOL_H */
