#ifndef ERF_MOVING_BODIES_H
#define ERF_MOVING_BODIES_H

#include <memory>
#include <vector>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include "ERF_ActuatorPoints.H"
#include "ERF_BodyMotion.H"
#include "ERF_MovingBodiesInputs.H"
#include "ERF_PrescribedCtDisk.H"
#include "ERF_RunningStats.H"
#include "ERF_WakeLines.H"
#ifdef ERF_USE_OPENFAST
#include "ERF_FLLC.H"
#include "ERF_OpenFASTRotor.H"
#endif

#ifdef ERF_USE_OPENFAST
#include "ERF_OpenFASTDriver.H"
#endif

struct SolverChoice;

/**
 * Owner of the moving bodies (OpenFAST turbines in this version) and the single place ERF
 * talks to them: created once from erf.moving_bodies.*, advanced once per ERF step on the
 * anchor level. Each step the flow velocity is sampled at every body's velocity nodes (or the
 * prescribed uniform velocity is used) and handed to the bodies, which are then stepped; the
 * forces of the bodies (the OpenFAST turbines' loads as actuator-disk rings or as actuator
 * lines on the blade nodes, their tower loads and nacelle drag when asked for, the
 * prescribed-Ct disks) are spread onto face-centred momentum sources, which ERF adds to its momentum
 * right-hand side in every stage of the step through add_momentum_sources().
 */
class MovingBodies
{
public:
    /**
     * Read erf.moving_bodies.*, check the solver settings and initialise the bodies.
     * Returns nullptr when no bodies are given.
     * @param fixed_dt ERF's fixed step on every level (the anchor level's is OpenFAST's coupling step)
     */
    static std::unique_ptr<MovingBodies> create (const SolverChoice& sc, int max_level,
                                                 const amrex::Vector<double>& fixed_dt, bool restarting);

    /**
     * Write the bodies' state into ERF's checkpoint directory: <chkdir>/moving_bodies/state
     * (the step count) and each OpenFAST turbine's own checkpoint <chkdir>/moving_bodies/<name>.chkp.
     */
    void write_checkpoint (const std::string& chkdir) const;

    /**
     * Restore the state written by write_checkpoint() from <chkdir>; called once, on a restart,
     * before the first step. The momentum sources are not stored: they are rebuilt from the
     * restored OpenFAST loads at the first step, as in the run being continued.
     */
    void read_checkpoint (const std::string& chkdir);

    /**
     * Place the bodies on the terrain: the height of the terrain surface (the k = 0 node plane
     * of z_phys_nd, bilinear) under each body's base is added to its base_pos z, so that z is
     * the height above the ground there (on a uniform-dz mesh, z_phys_nd null, nothing is
     * added). Called once by ERF when the mesh exists: on a fresh start from InitData, before
     * the step-0 checkpoint or plotfile (it then initialises OpenFAST); on a restart before
     * read_checkpoint(), once the mesh is read. The heights are written to
     * <diagnostics_dir>/ground.csv.
     */
    void set_ground (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    bool ground_set () const { return m_ground_set; }

    /**
     * Advance the bodies by one ERF step of the anchor level; a no-op on every other level. The
     * momentum sources live on the anchor level's grids; coarser levels see the bodies through
     * ERF's average-down of the state, finer levels are not supported (the anchor is the finest
     * unless asked otherwise).
     * @param cons the level's conserved state at the start of the step (its density is audited at the hubs on the first step)
     * @param U,V,W the level's face-centred velocities at the start of the step, ghost cells filled
     * @param z_phys_nd nodal heights, or nullptr on a uniform-dz mesh
     * @param geom the level's geometry
     */
    void advance (int lev, double time, double dt,
                  const amrex::MultiFab& cons,
                  const amrex::MultiFab& U, const amrex::MultiFab& V, const amrex::MultiFab& W,
                  const amrex::MultiFab* z_phys_nd, const amrex::MultiFab* detJ_cc,
                  const amrex::Geometry& geom);

    /**
     * Add the bodies' momentum sources (N/m^3, held fixed over the step) to ERF's face-centred
     * momentum right-hand sides; a no-op unless lev is the anchor level and a body carries a
     * force.
     */
    void add_momentum_sources (int lev, amrex::MultiFab& xmom_src, amrex::MultiFab& ymom_src,
                               amrex::MultiFab& zmom_src) const;

    bool has_sources () const { return m_src_defined; }

    /** The level the bodies are sampled and forced on: the finest unless erf.moving_bodies.anchor_level says otherwise. */
    int anchor_level () const { return m_anchor; }
    const MovingBodiesInputs& inputs () const { return m_in; }

private:
    MovingBodies (MovingBodiesInputs in, int anchor, double dt, double t_max, bool restarting);

    /**
     * Sample (or prescribe) the flow at every body's velocity nodes and hand it to the bodies;
     * the actuator lines with the lifting-line correction on get their blade velocities corrected
     * first (time decides whether the correction has started).
     */
    void supply_velocities (double time, const amrex::MultiFab& U, const amrex::MultiFab& V, const amrex::MultiFab& W,
                            const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);

    MovingBodiesInputs m_in;
    int m_anchor{0};
    double m_dt{0.0};
    double m_t_max{0.0};
    int m_step{0};
    bool m_restarting{false};   //!< created for a restart: read_checkpoint() must follow
    bool m_ground_set{false};   //!< set_ground() done: the bases sit on the terrain
    std::vector<amrex::Real> m_ground;   //!< per body: the terrain height under its base (m)
    bool m_restored{false};
    bool m_logs_reopened{false};
    std::vector<std::unique_ptr<erf_actuator::BodyMotion>> m_motion;  //!< one per body
    erf_actuator::ActuatorPoints m_points;                             //!< all bodies' velocity nodes
    std::vector<std::unique_ptr<erf_actuator::PrescribedCtDisk>> m_disks;
    std::vector<std::string> m_turbine_mode;                           //!< per OpenFAST turbine: adm | alm | none
    std::vector<MovingBodyInputs> m_turbine_in;                        //!< per OpenFAST turbine: its inputs (nacelle, density)
    std::vector<int> m_turbine_points_t;                               //!< ring points per turbine (adm)
    amrex::Real m_epsilon_dx{2.0};                                     //!< spreading width, dx units
    bool m_axis_checked{false};
    bool m_src_defined{false};
    amrex::MultiFab m_src_x, m_src_y, m_src_z;                         //!< anchor-level momentum sources

    /** Spread the bodies' forces onto the momentum sources (defined on first use). */
    void spread_sources (const amrex::MultiFab& U, const amrex::MultiFab* z_phys_nd,
                         const amrex::MultiFab* detJ_cc, const amrex::Geometry& geom);

    /**
     * Append time, the sum of every body's load on the structure (the turbines' thrust, tower
     * and nacelle forces, the disks' thrust) and the turbines' total power to
     * <diagnostics_dir>/total_load.csv: the farm's totals, and what the integrated momentum
     * source must equal minus.
     */
    void write_total_load (double time, bool first);
    bool m_total_written{false};
    /** Append time and the integrated source vector to <diagnostics_dir>/momentum_source.csv. */
    void write_source_diagnostics (double time, const amrex::MultiFab* detJ_cc,
                                   const amrex::Geometry& geom, bool first);
    bool any_forcing () const;

    /** Build the wake lines from the rotors' current hub, axis and diameter; abort if a point leaves the domain or the anchor level's grids. */
    void build_wake_lines (const amrex::BoxArray& level_grids, const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    bool m_disk_coverage_checked{false};
    /** Sample the wake lines, accumulate the running averages and write both files. */
    void write_wake_diagnostics (double time, const amrex::MultiFab& U, const amrex::MultiFab& V,
                                 const amrex::MultiFab& W, const amrex::MultiFab* z_phys_nd,
                                 const amrex::Geometry& geom);
    std::vector<std::unique_ptr<erf_actuator::WakeLines>> m_wakes;   //!< one per rotor, built at the first write
    bool m_wakes_built{false};
    bool m_wake_written{false};
    std::string m_wake_state_dir;   //!< checkpoint directory whose running averages the lines take over

    /** Feed every body's statistics with the state after this step and rewrite the files. */
    void accumulate_statistics (double time);
    std::vector<erf_actuator::RunningStats> m_turbine_stats;   //!< one per OpenFAST turbine
    std::vector<erf_actuator::RunningStats> m_disk_stats;      //!< one per prescribed-Ct disk
#ifdef ERF_USE_OPENFAST
    std::unique_ptr<erf_openfast::OpenFASTDriver> m_driver;
    /**
     * The start-up audit of every OpenFAST turbine against ERF's set-up, in two parts at the
     * first step: the model files, the geometry and the overlaps before the flow is first
     * sampled (so a rotor outside the domain is explained rather than tripping the sampler),
     * then the density at the hubs and the facing once the flow at the nodes is known. Each
     * part prints its findings together and a fatal one aborts.
     */
    void audit_turbines_setup (const amrex::BoxArray& level_grids, const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    //! the terrain height under each turbine's hub (prob_lo z on a flat mesh), for the audit and the wake lines
    std::vector<amrex::Real> hub_ground (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom) const;
    //! where turbine i's node velocities are sampled: its nodes, or (sampling = upstream) the nodes shifted
    //! sample_diameters_upstream diameters ahead of the hub along the shaft axis
    std::vector<amrex::Real> sampling_positions (int i) const;
    void audit_turbines_flow (const amrex::MultiFab& cons, const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    bool m_audited_setup{false};
    bool m_audited_flow{false};
    /** Build the lifting-line correction of every turbine that asks for it, once the node layout and chords are known. */
    void build_fllc (const amrex::Geometry& geom);
    /** Correct the blade velocity nodes of turbine i in uvw (3 per velocity node) with its lifting-line correction. */
    void apply_fllc (int i, double time, std::vector<amrex::Real>& uvw);
    /** Append a row to <output_root>_fllc.csv for every corrected turbine. */
    void write_fllc_diagnostics (double time, bool first);

    /** sampling = disk_corrected: scale the hub and blade node velocities of turbine i from the disk
     *  velocity to the free stream recovered with the filtered-disk factor (Shapiro et al. 2019) and
     *  momentum theory on the previous step's thrust; the tower nodes keep the resolved flow. */
    void apply_disk_correction (int i, std::vector<amrex::Real>& uvw, const amrex::Geometry& geom);
    void write_correction_diagnostics (double time, bool first);
    std::vector<amrex::Real> m_corr_uinf;                     //!< per turbine: the corrected free stream of the previous step (checkpointed)
    std::vector<erf_actuator::DiskCorrection> m_corr_last;    //!< per turbine: the last correction, for the diagnostics
    bool m_corr_written{false};
    std::vector<std::vector<std::unique_ptr<erf_actuator::FLLC>>> m_fllc;   //!< per turbine, one per blade (empty: off)
    bool m_fllc_built{false};
    bool m_fllc_written{false};
    std::string m_fllc_state_dir;   //!< checkpoint directory whose correction state the blades take over
#endif
};

#endif
