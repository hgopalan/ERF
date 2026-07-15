/**
 * @file ERF_LNGLayer.H
 * @brief LNG container class with MultiFab management
 * @details
 * The LNGLayer class is the primary container for all LNG simulation state.
 * It owns a set of MultiFabs representing physical fields (pool depth, evaporation rate, etc.)
 * and implements the core interface methods: initialize(), advance(), and write_output().
 * 
 * In Phase 1, all physics methods are stubs. The class primarily validates configuration
 * and allocates data structures.
 * 
 * @note Phase 1 design goals:
 *   - Fully compilable with zero physics (advance() returns immediately)
 *   - Robust debug output (per-step one-liners or verbose MultiFab stats)
 *   - Clean API for phases 2+ to fill in
 *   - Minimal coupling to ERF main class (only initialization, advance, and output)
 * 
 * @see Source/Dust/ERF_DustLayer.H — Dust module container pattern (reference)
 * @see Source/LNG/LNG_DEVELOPMENT.md — Phase 1 design spec
 */

#ifndef ERF_LNG_LAYER_H
#define ERF_LNG_LAYER_H

#include "ERF_LNGParams.H"
#include "ERF_LNGGrid.H"
#include <AMReX_MultiFab.H>
#include <memory>

// Forward declarations
class ERF;

/**
 * @class LNGLayer
 * @brief LNG state container and main interface to ERF
 * @details
 * Manages all LNG-related MultiFabs and orchestrates physics updates.
 * Designed as a plug-and-play module that can be enabled/disabled via erf.lng.enable.
 */
class LNGLayer {
public:
    /**
     * @brief Default constructor (defers allocation to initialize())
     */
    LNGLayer() = default;
    
    /**
     * @brief Destructor
     */
    ~LNGLayer() = default;
    
    /**
     * @brief Initialize LNG grid and MultiFabs
     * @param[in] erf       ERF main object (used to extract ATM grid)
     * @param[in] params    LNG parameters (from ParmParse erf.lng.*)
     * @details
     * Phase 1 actions:
     * 1. Call check_lng_prerequisites() for validation
     * 2. Call m_lg.build() to construct 2D grid from ATM level-0
     * 3. Allocate all MultiFabs with 1 ghost cell
     * 4. Initialize pool_mask and pool_depth based on pool_area_m2 and pool_depth_init_m
     *    - Pool is a circle: radius = sqrt(pool_area / π)
     *    - Circle centered at domain midpoint
     *    - pool_mask = 1.0 inside, 0.0 outside
     * 5. Fill all other MultiFabs with 0.0
     * 6. If lng_debug = true, print per-step debug header
     * 7. Call write_lng_stats_header() to create CSV diagnostics file
     */
    void initialize(const ERF& erf, const LNGParams& params);
    
    /**
     * @brief Advance LNG state by one timestep
     * @param[in] dt        Timestep [s]
     * @param[in] params    LNG parameters
     * @param[in] xvel_mf   Atmospheric u-velocity [m/s] (optional, Phase 4+)
     * @param[in] yvel_mf   Atmospheric v-velocity [m/s] (optional, Phase 4+)
     * @param[in] zvel_mf   Atmospheric w-velocity [m/s] (optional, Phase 4+)
     * @param[in] z_phys_cc_mf  Atmospheric physical height [m] (optional, Phase 4+)
     * @param[in] geom_atm  Atmospheric geometry (optional, Phase 3+)
     * @param[in] nz        Number of ATM z-levels (optional, Phase 4+)
     * @details
     * Phase 1 stub: simply increments m_step and m_time, then returns.
     * If lng_debug = true, prints:
     *   [LNG DEBUG] Step <N>  time=<T>  advance() stub — no physics in Phase 1
     * If lng_debug = true, also checks for NaN in all MultiFabs and aborts if found.
     * If verbose >= 3, prints per-field min/max statistics.
     */
    void advance(amrex::Real dt, const LNGParams& params,
                 const amrex::MultiFab* xvel_mf = nullptr,
                 const amrex::MultiFab* yvel_mf = nullptr,
                 const amrex::MultiFab* zvel_mf = nullptr,
                 const amrex::MultiFab* z_phys_cc_mf = nullptr,
                 const amrex::Geometry* geom_atm = nullptr,
                 int nz = 0);
    
    /**
     * @brief Apply LNG tendencies to atmosphere (stub)
     * @param[in,out] cc_source     Atmosphere cell-centered source term
     * @param[in] z_phys_cc         Atmosphere physical height [m]
     * @param[in] geom_atm          Atmosphere geometry
     * @details
     * Phase 1 stub: does nothing (no LNG→ATM coupling yet).
     * Phase 3: will compute latent heat and mass injection to atmosphere.
     */
    void apply_to_cc_source(amrex::MultiFab& cc_source,
                            const amrex::MultiFab& z_phys_cc,
                            const amrex::Geometry& geom_atm);
    
    /**
     * @brief Extract atmospheric return fields for LNG feedback (stub)
     * @param[in] S_new_cons        Atmosphere conservative variables
     * @param[in] geom_atm          Atmosphere geometry
     * @details
     * Phase 1 stub: does nothing.
     * Phase 3: will extract temperature, water vapor, pressure from atmosphere.
     * Phase 4: will extract wind field at reference height.
     */
    void extract_atm_return_fields(const amrex::MultiFab& S_new_cons,
                                   const amrex::Geometry& geom_atm);
    
    /**
     * @brief Write output (plotfiles, diagnostics)
     * @param[in] nstep             Current timestep number
     * @param[in] cur_time          Current simulation time [s]
     * @param[in] is_final          True if this is the final output
     * @details
     * Phase 1 actions:
     * 1. If lng_plot_int > 0 and nstep % lng_plot_int == 0:
     *    Print: [LNG] Plotfile write stub at step <nstep>
     * 2. If lng_debug = true:
     *    Append CSV row to lng_diag_file with current statistics (all zeros in Phase 1)
     *    Print: [LNG DEBUG] write_output step=<N>  time=<T>  total_pool_mass=0.000 kg
     */
    void write_output(int nstep, double cur_time, bool is_final = false);
    
    /**
     * @name MultiFab Accessors
     * @details Const accessors for LNG fields (used by coupling routines)
     * @{
     */
    
    const amrex::MultiFab* get_pool_depth()       const { return m_lng_pool_depth.get(); }
    const amrex::MultiFab* get_pool_mask()        const { return m_lng_pool_mask.get(); }
    const amrex::MultiFab* get_evap_flux()        const { return m_lng_evap_flux.get(); }
    const amrex::MultiFab* get_latent_flux()      const { return m_lng_latent_flux.get(); }
    const amrex::MultiFab* get_vapor_conc()       const { return m_lng_vapor_conc.get(); }
    const amrex::MultiFab* get_flux_atm()         const { return m_lng_flux_atm.get(); }
    const amrex::MultiFab* get_wind_ref()         const { return m_lng_wind_ref.get(); }
    const amrex::MultiFab* get_ustar()            const { return m_lng_ustar.get(); }
    const amrex::MultiFab* get_tsfc()             const { return m_lng_tsfc.get(); }
    const amrex::MultiFab* get_pblh()             const { return m_lng_pblh.get(); }
    const amrex::MultiFab* get_conc_sfc()         const { return m_lng_conc_sfc.get(); }
    const amrex::MultiFab* get_lfl_mask()         const { return m_lng_lfl_mask.get(); }
    const amrex::MultiFab* get_ufl_mask()         const { return m_lng_ufl_mask.get(); }
    
    const LNGGrid&         get_lng_grid()         const { return m_lg; }
    const amrex::Geometry& get_lng_geom()         const { return m_lg.geom; }
    const LNGParams&       get_params()           const { return m_params; }
    
    int get_lng_scalar_comp()                      const { return m_lng_scalar_comp; }
    int get_step()                                 const { return m_step; }
    amrex::Real get_time()                         const { return m_time; }
    
    /** @} */

private:
    // Grid and parameters
    LNGGrid     m_lg;              ///< LNG 2D grid
    LNGParams   m_params;          ///< LNG parameters (cached from ParmParse)
    
    // Time bookkeeping
    int         m_step = 0;        ///< Number of advance() calls completed
    amrex::Real m_time = 0.0;      ///< Cumulative simulation time [s]
    
    // Scalar component index (for coupling to atmosphere)
    int         m_lng_scalar_comp = 0;  ///< Index of LNG scalar in ATM (Phase 3+)
    
    // LNG state MultiFabs (all allocated in initialize())
    std::unique_ptr<amrex::MultiFab>  m_lng_pool_depth;      ///< Pool depth [m]
    std::unique_ptr<amrex::MultiFab>  m_lng_pool_mask;       ///< Pool mask [0/1]
    std::unique_ptr<amrex::MultiFab>  m_lng_evap_flux;       ///< Evaporation flux [kg/m^2/s]
    std::unique_ptr<amrex::MultiFab>  m_lng_latent_flux;     ///< Latent heat flux [W/m^2]
    std::unique_ptr<amrex::MultiFab>  m_lng_vapor_conc;      ///< Vapor concentration [kg/m^3]
    std::unique_ptr<amrex::MultiFab>  m_lng_flux_atm;        ///< Flux on ATM grid [kg/m^2/s]
    std::unique_ptr<amrex::MultiFab>  m_lng_wind_ref;        ///< Wind at zref [m/s] (u,v)
    std::unique_ptr<amrex::MultiFab>  m_lng_ustar;           ///< Friction velocity [m/s]
    std::unique_ptr<amrex::MultiFab>  m_lng_tsfc;            ///< Surface temperature [K]
    std::unique_ptr<amrex::MultiFab>  m_lng_pblh;            ///< PBL height [m]
    std::unique_ptr<amrex::MultiFab>  m_lng_conc_sfc;        ///< Conc. at surface [kg/m^3]
    std::unique_ptr<amrex::MultiFab>  m_lng_lfl_mask;        ///< LFL exceedance [0/1]
    std::unique_ptr<amrex::MultiFab>  m_lng_ufl_mask;        ///< UFL exceedance [0/1]
};

#endif /* ERF_LNG_LAYER_H */
