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
#include "ERF_LNGSpillSchedule.H"
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
     *    - Pool is a circle: radius = sqrt(pool_area / pi)
     *    - Circle centered at domain midpoint
     *    - pool_mask = 1.0 inside, 0.0 outside
     * 5. Fill all other MultiFabs with 0.0
     * 6. If lng_debug = true, print per-step debug header
     * 7. Call write_lng_stats_header() to create CSV diagnostics file
     */
    void initialize(const ERF& erf, const LNGParams& params);

    /**
     * @brief Advance LNG state by one timestep
     * @param[in] dt              Timestep [s]
     * @param[in] params          LNG parameters
     * @param[in] surface_layer   SurfaceLayer pointer for extracting u*, T_sfc, PBLH (Phase 4, optional)
     * @param[in] xvel_mf         Atmospheric u-velocity [m/s] (optional, Phase 4+)
     * @param[in] yvel_mf         Atmospheric v-velocity [m/s] (optional, Phase 4+)
     * @param[in] zvel_mf         Atmospheric w-velocity [m/s] (optional, Phase 4+)
     * @param[in] z_phys_cc_mf    Atmospheric physical height [m] (optional, Phase 4+)
     * @param[in] S_cons          Atmospheric conserved state (for conc_sfc extraction, Phase 4+)
     * @param[in] geom_atm        Atmospheric geometry (optional, Phase 4+)
     * @param[in] nz              Number of ATM z-levels (optional, Phase 4+)
     * @details
     * Phase 2: Evaporation with placeholder atmospheric fields.
     * Phase 4: Live field extraction from atmosphere or fallback to placeholders.
     * If lng_debug = true, prints per-field debug diagnostics.
     * If verbose >= 3, prints per-field min/max statistics.
     */
    void advance(amrex::Real dt, const LNGParams& params,
                 class SurfaceLayer* surface_layer  = nullptr,
                 const amrex::MultiFab* xvel_mf     = nullptr,
                 const amrex::MultiFab* yvel_mf     = nullptr,
                 const amrex::MultiFab* zvel_mf     = nullptr,
                 const amrex::MultiFab* z_phys_cc_mf = nullptr,
                 const amrex::MultiFab* S_cons       = nullptr,
                 const amrex::Geometry* geom_atm     = nullptr,
                 int nz = 0);

    /**
     * @brief Apply LNG tendencies to atmosphere
     * @param[in,out] cc_source     Atmosphere cell-centered source term
     * @param[in] z_phys_cc         Atmosphere physical height [m]
     * @param[in] geom_atm          Atmosphere geometry
     * @details
     * Phase 3: computes latent heat and mass injection to atmosphere.
     */
    void apply_to_cc_source(amrex::MultiFab& cc_source,
                            const amrex::MultiFab& z_phys_cc,
                            const amrex::Geometry& geom_atm);

    /**
     * @brief Extract atmospheric return fields for LNG feedback
     * @param[in] S_new_cons        Atmosphere conservative variables
     * @param[in] geom_atm          Atmosphere geometry
     * @details
     * Phase 4: extracts concentration at surface from atmosphere.
     */
    void extract_atm_return_fields(const amrex::MultiFab& S_new_cons,
                                   const amrex::Geometry& geom_atm);

    /**
     * @brief Compute flammability diagnostics (Phase 5)
     * @param[in] dt        Timestep [s]
     * @param[in] cur_time  Current simulation time [s]
     * @param[in] nstep     Current timestep number
     * @details
     * Phase 5: Computes LFL and UFL exceedance masks from lng_conc_sfc field,
     * and calculates total flammable zone areas. Updates m_lfl_area and m_ufl_area.
     */
    void compute_flammability_diagnostics(amrex::Real dt, amrex::Real cur_time, int nstep);

    /**
     * @brief Write output (plotfiles, diagnostics)
     * @param[in] nstep             Current timestep number
     * @param[in] cur_time          Current simulation time [s]
     * @param[in] is_final          True if this is the final output
     * @details
     * 1. If lng_plot_int > 0 and nstep % lng_plot_int == 0: write plotfile stub.
     * 2. Always appends CSV row to lng_diag_file.
     * 3. If lng_debug = true: prints pool mass summary.
     * Guards duplicate calls via m_last_output_step.
     */
    void write_output(int nstep, double cur_time, bool is_final = false);

    /**
     * @name MultiFab Accessors
     * @details Const accessors for LNG fields (used by coupling routines)
     * @{
     */
    const amrex::MultiFab* get_pool_depth()   const { return m_lng_pool_depth.get(); }
    const amrex::MultiFab* get_pool_mask()    const { return m_lng_pool_mask.get(); }
    const amrex::MultiFab* get_evap_flux()    const { return m_lng_evap_flux.get(); }
    const amrex::MultiFab* get_latent_flux()  const { return m_lng_latent_flux.get(); }
    const amrex::MultiFab* get_vapor_conc()   const { return m_lng_vapor_conc.get(); }
    const amrex::MultiFab* get_flux_atm()     const { return m_lng_flux_atm.get(); }
    const amrex::MultiFab* get_wind_ref()     const { return m_lng_wind_ref.get(); }
    const amrex::MultiFab* get_ustar()        const { return m_lng_ustar.get(); }
    const amrex::MultiFab* get_tsfc()         const { return m_lng_tsfc.get(); }
    const amrex::MultiFab* get_pblh()         const { return m_lng_pblh.get(); }
    const amrex::MultiFab* get_conc_sfc()     const { return m_lng_conc_sfc.get(); }
    const amrex::MultiFab* get_lfl_mask()     const { return m_lng_lfl_mask.get(); }
    const amrex::MultiFab* get_ufl_mask()     const { return m_lng_ufl_mask.get(); }

    /// @name Phase 5: Gravity Current MultiFabs
    /// @{
    const amrex::MultiFab* get_gc_h()        const { return m_lng_gc_h.get(); }
    const amrex::MultiFab* get_gc_u()        const { return m_lng_gc_u.get(); }
    const amrex::MultiFab* get_gc_v()        const { return m_lng_gc_v.get(); }
    const amrex::MultiFab* get_gc_ri_flag()  const { return m_lng_gc_ri_flag.get(); }
    /// @}

    /// @name Phase 7: Regulatory Compliance MultiFabs
    /// @{
    const amrex::MultiFab* get_conc_1h_avg()   const { return m_lng_conc_1h_avg.get(); }
    const amrex::MultiFab* get_exceed_flag()   const { return m_lng_exceed_flag.get(); }
    /// @}

    amrex::Real get_lfl_area()               const { return m_lfl_area; }
    amrex::Real get_ufl_area()               const { return m_ufl_area; }

    const LNGGrid&         get_lng_grid()    const { return m_lg; }
    const amrex::Geometry& get_lng_geom()    const { return m_lg.geom; }
    const LNGParams&       get_params()      const { return m_params; }

    int         get_lng_scalar_comp()        const { return m_lng_scalar_comp; }
    int         get_step()                   const { return m_step; }
    amrex::Real get_time()                   const { return m_time; }

    /** @} */

private:
    // ── Grid and parameters ───────────────────────────────────────────────
    LNGGrid     m_lg;        ///< LNG 2D grid
    LNGParams   m_params;    ///< LNG parameters (cached from ParmParse)

    // ── Time bookkeeping ──────────────────────────────────────────────────
    int         m_step             = 0;    ///< Number of advance() calls completed
    amrex::Real m_time             = 0.0;  ///< Cumulative simulation time [s]
    int         m_last_output_step = -1;   ///< Guard duplicate write_output() calls

    // ── Phase 2 pool tracking ─────────────────────────────────────────────
    amrex::Real m_pool_cx = -1.0;  ///< Pool centre x [m]; -1 = domain centre
    amrex::Real m_pool_cy = -1.0;  ///< Pool centre y [m]; -1 = domain centre
    amrex::Real m_lg_z0   = 0.01;  ///< Aerodynamic roughness length for LNG surface [m]

    // ── Scalar component index ────────────────────────────────────────────
    int m_lng_scalar_comp = -1;  ///< Index of LNG scalar in ATM conserved state (Phase 3+)

    // ── Phase 4: ATM state pointers (non-owning, set each timestep) ───────
    class SurfaceLayer*    m_surface_layer_ptr = nullptr;
    const amrex::MultiFab* m_S_cons_ptr        = nullptr;
    const amrex::Geometry* m_geom_atm_ptr      = nullptr;

    // ── LNG state MultiFabs (allocated in initialize()) ───────────────────
    std::unique_ptr<amrex::MultiFab> m_lng_pool_depth;   ///< Pool depth [m]
    std::unique_ptr<amrex::MultiFab> m_lng_pool_mask;    ///< Pool mask [0/1]
    std::unique_ptr<amrex::MultiFab> m_lng_evap_flux;    ///< Evaporation flux [kg/m^2/s]
    std::unique_ptr<amrex::MultiFab> m_lng_latent_flux;  ///< Latent heat flux [W/m^2]
    std::unique_ptr<amrex::MultiFab> m_lng_vapor_conc;   ///< Vapor concentration [kg/m^3]
    std::unique_ptr<amrex::MultiFab> m_lng_flux_atm;     ///< Flux on ATM k=0 slab [kg/m^2/s]
    std::unique_ptr<amrex::MultiFab> m_lng_wind_ref;     ///< Wind at zref [m/s] — 2 comps (u,v)
    std::unique_ptr<amrex::MultiFab> m_lng_ustar;        ///< Friction velocity [m/s]
    std::unique_ptr<amrex::MultiFab> m_lng_tsfc;         ///< Surface temperature [K]
    std::unique_ptr<amrex::MultiFab> m_lng_pblh;         ///< PBL height [m]
    std::unique_ptr<amrex::MultiFab> m_lng_conc_sfc;     ///< LNG conc. at surface [kg/m^3]
    std::unique_ptr<amrex::MultiFab> m_lng_lfl_mask;     ///< LFL exceedance flag [0/1]
    std::unique_ptr<amrex::MultiFab> m_lng_ufl_mask;     ///< UFL exceedance flag [0/1]

    // ── Phase 5: gravity current state (2D shallow-water) ─────────────────
    std::unique_ptr<amrex::MultiFab> m_lng_gc_h;         ///< Cloud depth [m]
    std::unique_ptr<amrex::MultiFab> m_lng_gc_u;         ///< Depth-averaged u-velocity [m/s]
    std::unique_ptr<amrex::MultiFab> m_lng_gc_v;         ///< Depth-averaged v-velocity [m/s]
    std::unique_ptr<amrex::MultiFab> m_lng_gc_ri_flag;   ///< 0 = GC active, 1 = mixed regime

    // ── Phase 5: flammability diagnostics ─────────────────────────────────
    amrex::Real m_lfl_area = 0.0;  ///< Area within LFL zone [m^2]
    amrex::Real m_ufl_area = 0.0;  ///< Area within UFL zone [m^2]

    // ── Phase 7: regulatory diagnostics ──────────────────────────────────
    std::unique_ptr<amrex::MultiFab> m_lng_conc_1h_avg;  ///< 1-hr avg conc [kg/m^3]
    std::unique_ptr<amrex::MultiFab> m_lng_exceed_flag;   ///< NFPA exceedance [0/1]
    amrex::Real m_exclusion_radius_m = 0.0;  ///< NFPA 59A exclusion zone radius [m]

    // ── Phase 8: spill schedule ──────────────────────────────────────────
    LNGSpillSchedule m_spill_schedule;  ///< Loaded once in initialize(), used every advance()

    // ── Phase 4: ATM-resolution scratch MFs for MPI-safe wind extraction ──
    // These live on the ATM BoxArray / DistributionMapping so that
    // ParallelCopy from xvel_mf / yvel_mf / z_phys_cc_mf is always
    // rank-local and never causes cross-rank direct array access.
    // fill_lng_wind_from_interpolation reads these copies.
    std::unique_ptr<amrex::MultiFab> m_xvel_atm;   ///< x-face u-vel on ATM grid (ghost=1)
    std::unique_ptr<amrex::MultiFab> m_yvel_atm;   ///< y-face v-vel on ATM grid (ghost=1)
    std::unique_ptr<amrex::MultiFab> m_zphys_atm;  ///< cell-centred z_phys on ATM grid (ghost=1)
};

#endif /* ERF_LNG_LAYER_H */