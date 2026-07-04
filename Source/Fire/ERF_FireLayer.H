/**
 * \file ERF_FireLayer.H
 *
 * \brief Main fire simulation layer class.
 *
 * The FireLayer class orchestrates all fire simulation computations at each
 * atmospheric timestep. It owns all fire MultiFabs, calls wind extraction,
 * WAF, terrain corrections, and Rothermel ROS computation in sequence.
 *
 * References:
 * - Ported from concepts in hgopalan/wildfire_levelset
 */

#ifndef ERF_FIRE_LAYER_H
#define ERF_FIRE_LAYER_H

#include <memory>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

#include "ERF_FireParams.H"
#include "ERF_FireGrid.H"
#include "ERF_SurfaceLayer.H"

// Forward declarations
class ERF;

/**
 * \class FireLayer
 *
 * Main fire simulation module owned by ERF as m_fire_layer.
 * Holds all fire-related data structures and coordinates the simulation.
 *
 * MultiFabs (all on fire grid, C*nx × C*ny × 1):
 * - fire_phi: Level-set function (<0 burned, >0 unburned, ≈0 front)
 * - fire_wind_ref: 10-m wind from MOST extraction (u,v)
 * - fire_wind_eff: After WAF and terrain corrections (u,v)
 * - fire_slopes: Terrain slopes (dz/dx, dz/dy) — static after init
 * - fire_curvature: Terrain curvature — static after init
 * - fire_ros: Rate of spread [m/s]
 * - fire_fuel_load: Fuel load [kg/m²]
 * - fire_fuel_mc: Moisture content (3 comp: M_1hr, M_10hr, M_100hr)
 * - fire_heat_flux: Sensible heat flux [W/m²] (from Phase 6)
 * - fire_spread_vec: Spread vector for level-set (Phase 3 — FARSITE)
 */
class FireLayer
{
public:
    FireLayer() = default;
    ~FireLayer() = default;

    // Disable copy, allow move
    FireLayer(const FireLayer&) = delete;
    FireLayer& operator=(const FireLayer&) = delete;
    FireLayer(FireLayer&&) = default;
    FireLayer& operator=(FireLayer&&) = default;

    /**
     * \brief Initialize fire module.
     *
     * Called once at the end of ERF::InitData_post() after SurfaceLayer is ready.
     * Performs all prerequisite checks, allocates MultiFabs, initializes fire grid,
     * and sets ignition condition.
     *
     * \param erf            Reference to ERF instance (for grid access)
     * \param fire_params    Fire parameters from ParmParse
     *
     * Steps:
     * 1. Call verify_fire_prerequisites()
     * 2. Create fire grid via create_fire_grid()
     * 3. Allocate all fire MultiFabs
     * 4. Compute terrain slopes and curvature (static)
     * 5. Initialize fuel load and moisture fields
     * 6. Set ignition in fire_phi (circle around ignition_x, ignition_y, ignition_r)
     */
    void initialize(const ERF& erf, const FireParams& fire_params);

    /**
     * \brief Advance fire state by one atmospheric timestep.
     *
     * Called each timestep after SurfaceLayer::update_fluxes().
     * Runs the complete wind extraction + WAF + terrain + Rothermel pipeline.
     *
     * \param dt             Atmospheric timestep [s]
     * \param surface_layer  Reference to SurfaceLayer with updated winds/fluxes
     *
     * Steps:
     * 1. Extract 10-m wind from MOST via fill_fire_wind_from_most()
     * 2. Apply WAF to wind magnitude (if use_waf)
     * 3. Apply FARSITE terrain corrections (if use_terrain_wind)
     * 4. Compute ROS field via compute_ros_field()
     * 5. (Phase 3: Advect level-set via FARSITE algorithm)
     * 6. (Phase 6: Compute heat flux from ROS and fuel)
     */
    void advance(amrex::Real dt, const SurfaceLayer& surface_layer);

    /**
     * \brief Get heat flux MultiFab for coupling to atmosphere.
     *
     * \return Reference to fire_heat_flux MultiFab [W/m²]
     */
    const amrex::MultiFab& get_heat_flux() const { return *fire_heat_flux; }

    /**
     * \brief Get rate of spread MultiFab for diagnostics/visualization.
     *
     * \return Reference to fire_ros MultiFab [m/s]
     */
    const amrex::MultiFab& get_ros() const { return *fire_ros; }

    /**
     * \brief Get level-set function for visualization.
     *
     * \return Reference to fire_phi MultiFab
     */
    const amrex::MultiFab& get_phi() const { return *fire_phi; }

private:
    // Fire grid structure
    std::unique_ptr<FireGrid> m_fire_grid;

    // Fire MultiFabs (all C*nx × C*ny × 1)
    std::unique_ptr<amrex::MultiFab> fire_phi;           ///< Level-set
    std::unique_ptr<amrex::MultiFab> fire_wind_ref;      ///< 10-m wind (u,v)
    std::unique_ptr<amrex::MultiFab> fire_wind_eff;      ///< Effective wind
    std::unique_ptr<amrex::MultiFab> fire_slopes;        ///< Terrain slopes
    std::unique_ptr<amrex::MultiFab> fire_curvature;     ///< Terrain curvature
    std::unique_ptr<amrex::MultiFab> fire_ros;           ///< Rate of spread
    std::unique_ptr<amrex::MultiFab> fire_fuel_load;     ///< Fuel load
    std::unique_ptr<amrex::MultiFab> fire_fuel_mc;       ///< Moisture content (3 comp)
    std::unique_ptr<amrex::MultiFab> fire_heat_flux;     ///< Heat flux
    std::unique_ptr<amrex::MultiFab> fire_spread_vec;    ///< Spread vector

    // Parameters
    FireParams                       m_fire_params;
};

#endif // ERF_FIRE_LAYER_H
