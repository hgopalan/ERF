#ifndef ERF_MOVING_BODIES_H
#define ERF_MOVING_BODIES_H

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include "ERF_BodyExchange.H"
#include "ERF_BodyMotion.H"
#include "ERF_DiagnosticsLog.H"
#include "ERF_MovingBodiesInputs.H"
#include "ERF_PhaseTimers.H"
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
 *
 * Each OpenFAST turbine and each prescribed-Ct disk has one owner rank, chosen from the ranks whose
 * boxes of the anchor level overlap it where the balance allows (set_ground()). A turbine's owner alone
 * runs its model and holds its nodes and loads; a disk's owner alone samples the flow for it and holds
 * its thrust. Their points go only to the ranks whose boxes they touch (erf_actuator::BodyExchange), and
 * so do the wake lines behind each rotor, which the rotor's owner samples and averages.
 */
class MovingBodies
{
public:
    /**
     * Read erf.moving_bodies.*, check the solver settings and initialise the bodies.
     * Returns nullptr when no bodies are given.
     * @param fixed_dt ERF's fixed step on every level (the anchor level's is OpenFAST's coupling step)
     * @param stop_elapsed the seconds from the start to ERF's stop time, used when the run stops
     *        at stop_datetime (stop_time, or max_step times erf.fixed_dt, otherwise)
     */
    static std::unique_ptr<MovingBodies> create (const SolverChoice& sc, int max_level,
                                                 const amrex::Vector<double>& fixed_dt, bool restarting,
                                                 double stop_elapsed);

    /**
     * Write the bodies' state into ERF's checkpoint directory: <chkdir>/moving_bodies/state (the
     * step count, the time the bodies started, OpenFAST's stop time, whether OpenFAST's first
     * solution was taken, the anchor level and step, and one record of every body's placing and
     * forcing inputs) and each OpenFAST turbine's own checkpoint <chkdir>/moving_bodies/<name>.chkp.
     */
    void write_checkpoint (const std::string& chkdir) const;

    /**
     * Restore the state written by write_checkpoint() from <chkdir>; called once, on a restart,
     * before the first step, with ERF's time. A changed body or setting aborts naming the input; the
     * logs lose the rows written after the checkpoint. A checkpoint without bodies (a precursor)
     * starts the bodies at that time. The momentum sources are not stored: they are rebuilt from the
     * restored OpenFAST loads at the first step, as in the run being continued.
     */
    void read_checkpoint (const std::string& chkdir, double time);

    /**
     * Place the bodies on the terrain: the height of the terrain surface (the k = 0 node plane
     * of z_phys_nd, bilinear) under each body's base is added to its base_pos z, so that z is
     * the height above the ground there (on a uniform-dz mesh, z_phys_nd null, nothing is
     * added). Then choose each OpenFAST turbine's owner rank: a rank whose anchor-level boxes overlap its
     * footprint (the column under its base, widened by the model's rotor radius) while that keeps the
     * turbines evenly spread and apart from turbines of another OpenFAST time step, otherwise the nearest
     * rank that does (erf_actuator::choose_owners); and each prescribed-Ct disk's, by the same rule among
     * the disks alone (its footprint widened by its rotor_radius). Called once by ERF when the mesh exists: on a fresh start from
     * InitData, before the step-0 checkpoint or plotfile (it then initialises OpenFAST); on a restart
     * before read_checkpoint(), once the mesh is read. The heights are written to
     * <diagnostics_dir>/ground.csv. Aborts when the anchor level has no grids (finest_level below it).
     * @param grids,dmap the anchor level's (cell-centred) boxes and their distribution over the ranks
     */
    void set_ground (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom, const amrex::BoxArray& grids,
                     const amrex::DistributionMapping& dmap, double time, int finest_level);
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

    /**
     * Called by ERF at every coarse step: abort when the anchor level has no grids, or when a level
     * finer than the anchor covers part of a rotor or disk with the kernel's reach (the force lives on
     * the anchor level only, and ERF's average-down would overwrite it there).
     */
    void check_levels (int finest_level, const amrex::Vector<amrex::BoxArray>& grids,
                       const amrex::Vector<amrex::Geometry>& geoms) const;

    /** The level the bodies are sampled and forced on: amr.max_level unless erf.moving_bodies.anchor_level says otherwise. */
    int anchor_level () const { return m_anchor; }
    const MovingBodiesInputs& inputs () const { return m_in; }

private:
    MovingBodies (MovingBodiesInputs in, int anchor, double dt, double t_max, bool restarting);

    /** The bodies start at ERF time `time`: OpenFAST runs from there to the stop time, a whole number of steps. */
    void start_bodies (double time);
    /** Abort unless the span from time to the stop time is a whole number of erf.fixed_dt steps. */
    void check_whole_steps (double time) const;

    /** One step of advance() on the anchor level: sample, step the bodies, write the logs, spread the forces. */
    void advance_step (double time, double dt, const amrex::MultiFab& cons,
                       const amrex::MultiFab& U, const amrex::MultiFab& V, const amrex::MultiFab& W,
                       const amrex::MultiFab* z_phys_nd, const amrex::MultiFab* detJ_cc, const amrex::Geometry& geom);
    /**
     * Collective, every erf.moving_bodies.timing_int steps: print the seconds each phase took on the ranks
     * since the last summary (minimum, mean and maximum over the ranks) and the OpenFAST step of the fastest,
     * mean and slowest turbine, then start counting again.
     */
    void report_timing ();
    mutable erf_actuator::PhaseTimers m_timers;   // also timed by the const write_checkpoint(), check_levels(), add_momentum_sources()
    int m_timing_from{-1};                        //!< the first step of the next timing summary (-1: before the first step)

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
    double m_t_max{0.0};          //!< ERF's stop time (s)
    bool m_disk_density_checked{false};   //!< the disks' air_density checked against ERF's density in this run
    bool m_steps_end_first{false};        //!< max_step * erf.fixed_dt ends the run before the stop time (no step is cut)
    double m_input_fixed_dt{0.0}; //!< erf.fixed_dt as given (the anchor level's step is it over the sub-cycling ratios)
    double m_t0{0.0};             //!< ERF time at which the bodies started (0 on a fresh start; a precursor's time)
    double m_openfast_tmax{0.0};  //!< OpenFAST's stop time on its own clock, which starts at m_t0 (s)
    int m_step{0};
    bool m_restarting{false};   //!< created for a restart: read_checkpoint() must follow
    bool m_ground_set{false};   //!< set_ground() done: the bases sit on the terrain
    bool m_fitted{false};       //!< the anchor level has z_phys_nd (terrain-following or stretched): a cell's height is not k dz
    std::vector<amrex::Real> m_ground;   //!< per body: the terrain height under its base (m)
    bool m_restored{false};   //!< read_checkpoint() done
    bool m_resumed{false};    //!< the checkpoint held the bodies' state: their logs continue rather than start afresh
    std::vector<std::unique_ptr<erf_actuator::BodyMotion>> m_motion;  //!< one per body
    std::vector<std::unique_ptr<erf_actuator::PrescribedCtDisk>> m_disks;
    std::vector<std::string> m_turbine_mode;                           //!< per OpenFAST turbine: adm | alm | none
    std::vector<MovingBodyInputs> m_turbine_in;                        //!< per OpenFAST turbine: its inputs (nacelle, density)
    std::vector<int> m_turbine_points_t;                               //!< ring points per turbine (adm)
    std::vector<amrex::Real> m_disk_epsilon;                           //!< per prescribed-Ct disk: its spreading width (dx units)
    bool m_src_defined{false};
    amrex::MultiFab m_src_x, m_src_y, m_src_z;                         //!< anchor-level momentum sources

    /** Spread the bodies' forces onto the momentum sources (defined on first use). */
    void spread_sources (const amrex::MultiFab& U, const amrex::MultiFab* z_phys_nd,
                         const amrex::MultiFab* detJ_cc, const amrex::Geometry& geom);

    /**
     * Append time, the sum of every body's load on the structure (the turbines' thrust, tower
     * and nacelle forces, the disks' thrust) and the turbines' total power to
     * <diagnostics_dir>/total_load.csv: the farm's totals, and what the integrated momentum
     * source must equal minus. The loads join the step's packed gather (flush_logs()).
     */
    void write_total_load (double time, bool first);
    bool m_total_written{false};

    /**
     * The bodies' appended logs, each file opened once and kept open (flushed at every checkpoint
     * and at the end of the run): the bodies' time series in the per_turbine and per_rank formats on
     * their owners, the wake lines' rows on the rotors' owners, and in the farm format and for the
     * run's own files (total_load.csv, momentum_source.csv) on the I/O rank.
     */
    mutable erf_actuator::LogStreams m_logs;   // flushed by the const write_checkpoint()
    /**
     * Write the rows of one time-series log (erf, flow, correction, fllc, disk) for the bodies
     * listed (indices of m_exchange: the turbines, then the disks), in the format
     * erf.moving_bodies.diagnostics_format asks for. fill(b, row) gives body b's k values on its
     * owner, or returns false when the body has no row. Collective (every rank calls it alike); in
     * the farm format the rows join the step's packed gather (flush_logs()), otherwise each owner
     * writes its own at once.
     * @param truncate the file starts afresh when this call opens it (a fresh run), else it is appended to
     */
    void write_series (const std::string& log, const std::string& columns, double time, bool truncate,
                       const std::vector<int>& bodies, int k, const std::function<bool(int, amrex::Real*)>& fill);
    /** One part of the step's packed gather: this rank's contribution, and what the I/O rank does with the sum. */
    struct PendingLog {
        std::vector<double> packed;
        std::function<void(const double*)> write;
    };
    std::vector<PendingLog> m_pending_logs;
    /**
     * Collective: sum every part of the step's gather onto the I/O rank in one reduction, and let
     * it write them, in the order they were added. A no-op when nothing is pending.
     */
    void flush_logs ();
    /** The diagnostics file of log for body b in the per_turbine format: <output_root>_<log>.csv. */
    std::string body_log_file (int b, const std::string& log) const;
    /** The diagnostics file of log for rank r in the per_rank format: <diagnostics_dir>/rank<r>_<log>.csv. */
    std::string rank_log_file (int r, const std::string& log) const;
    /** The diagnostics file of log in the farm format: <diagnostics_dir>/farm_<log>.csv. */
    std::string farm_log_file (const std::string& log) const;
    /** Write the statistics of every body this step, in the format asked for (collective in the farm format). */
    void write_statistics ();
    /** Append time and the integrated source vector to <diagnostics_dir>/momentum_source.csv. */
    void write_source_diagnostics (double time, const amrex::MultiFab* detJ_cc,
                                   const amrex::Geometry& geom, bool first);
    bool any_forcing () const;

    /** Build the wake lines from the rotors' current hub, axis and diameter; abort if a point leaves the domain or the anchor level's grids. */
    void build_wake_lines (const amrex::BoxArray& level_grids, const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    /** Abort unless every body's points, with the kernel's reach, lie on the anchor level's grids (at the first step and after a regrid). */
    void check_coverage (const amrex::BoxArray& level_grids, const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    amrex::BoxArray m_audited_grids;   //!< the anchor grids the coverage was checked on: checked again when they change
    /** Sample each rotor's wake lines for its owner (m_wake_exchange), which accumulates the running average and writes both files. */
    void write_wake_diagnostics (double time, const amrex::MultiFab& U, const amrex::MultiFab& V,
                                 const amrex::MultiFab& W, const amrex::MultiFab* z_phys_nd,
                                 const amrex::Geometry& geom);
    std::vector<std::unique_ptr<erf_actuator::WakeLines>> m_wakes;   //!< one per rotor, built at the first write
    bool m_wakes_built{false};
    bool m_wake_written{false};
    std::string m_wake_state_dir;   //!< checkpoint directory whose running averages the lines take over

    /** Feed every body's statistics with the state after this step and rewrite the files (each body's on its owner). */
    void accumulate_statistics (double time);
    std::vector<erf_actuator::RunningStats> m_turbine_stats;   //!< one per OpenFAST turbine
    std::vector<erf_actuator::RunningStats> m_disk_stats;      //!< one per prescribed-Ct disk

    /** The OpenFAST turbines (0 without OpenFAST): the bodies 0 .. num_turbines() - 1 of m_exchange, the disks after them. */
    int num_turbines () const { return static_cast<int>(m_turbine_mode.size()); }
    /**
     * The owners and point-to-point exchanges of the bodies that sample the flow and force it: the
     * OpenFAST turbines, then the prescribed-Ct disks (disk k is body num_turbines() + k).
     */
    erf_actuator::BodyExchange m_exchange;
    std::vector<std::string> m_exchange_names;   //!< per body of m_exchange: its name, for the run log
    /** The owners and exchanges of the wake lines, one body per rotor in the order of m_wakes (each its rotor's owner). */
    erf_actuator::BodyExchange m_wake_exchange;
    std::vector<std::string> m_wake_exchange_names;
    /**
     * Choose every turbine's and every disk's owner rank from the anchor level's boxes, hand the owners to
     * the driver, the exchange and the writers of the logs, and print them.
     */
    void choose_owners (const amrex::Geometry& geom, const amrex::BoxArray& grids, const amrex::DistributionMapping& dmap);
    /** How far (m) body b of m_exchange reaches beyond its points when its envelope is built; on its owner. */
    amrex::Real exchange_margin (int b, const amrex::Geometry& geom) const;
    /** Hand the exchange the hub (and blade sweep) of every turbine this rank owns, for its throttled envelope checks. */
    void set_hubs ();
    /** Print, for each body of the exchange whose exchanges now reach other ranks than last time, the ranks it exchanges with. */
    static void report_participants (const erf_actuator::BodyExchange& ex, const std::vector<std::string>& names,
                                     std::vector<std::vector<int>>& reported);
    std::vector<std::vector<int>> m_reported_participants;        //!< per body of m_exchange: the ranks last printed
    std::vector<std::vector<int>> m_reported_wake_participants;   //!< per body of m_wake_exchange
    /**
     * Collective. A table of k values per prescribed-Ct disk, filled by each disk's owner (fill(k, row) for
     * the disks this rank owns) and summed over the ranks onto the I/O rank, each row exactly as its owner wrote it.
     */
    std::vector<amrex::Real> disk_table (int k, const std::function<void(int, amrex::Real*)>& fill) const;
#ifdef ERF_USE_OPENFAST
    std::unique_ptr<erf_openfast::OpenFASTDriver> m_driver;
    //! per distinct .fst path: what start-up takes from the model's files (erf_openfast::gather_model_facts)
    std::map<std::string, erf_openfast::ModelFacts> m_model_facts;
    //! the model files the owner's two start-up audits read, each once for both (emptied after the second)
    erf_openfast::ModelFiles m_audit_files;
    /**
     * Collective. A table of k values per turbine, filled by each turbine's owner (fill(i, row) for the
     * turbines this rank owns) and summed over the ranks, so that every rank (or only the I/O rank, with
     * io_only) holds all of it, each row exactly as its owner wrote it.
     */
    std::vector<amrex::Real> turbine_table (int k, bool io_only,
                                            const std::function<void(int, amrex::Real*)>& fill) const;
    /**
     * Collective. Print, on the I/O rank and in turbine order, a text each turbine's owner wrote
     * (text[i], empty on the other ranks), and return on every rank whether an owner flagged its
     * turbine (fatal[i]).
     */
    bool print_turbine_texts (const std::vector<std::string>& text, const std::vector<int>& fatal) const;
    /** How far (m) turbine i's envelope reaches beyond its points when it is built: its tip radius and two cells; its owner only. */
    amrex::Real envelope_margin (int i, const amrex::Geometry& geom) const;
    std::vector<std::vector<amrex::Real>> m_turbine_vel;   //!< per turbine, on its owner: the velocities last sampled at its points
    /**
     * The start-up audit of every OpenFAST turbine against ERF's set-up, in two parts at the
     * first step: the model files, the geometry and the overlaps before the flow is first
     * sampled (so a rotor outside the domain is explained rather than tripping the sampler),
     * then the density at the hubs and the facing once the flow at the nodes is known. Each
     * part prints its findings together and a fatal one aborts.
     */
    void audit_turbines_setup (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    //! where turbine i's node velocities are sampled: its nodes, or (sampling = upstream) the nodes shifted
    //! sample_diameters_upstream diameters ahead of the hub along the shaft axis; on its owner
    std::vector<amrex::Real> sampling_positions (int i) const;
    void audit_turbines_flow (const amrex::MultiFab& cons, const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    bool m_audited_setup{false};
    bool m_audited_flow{false};
    bool m_axis_checked{false};
    /**
     * Collective. The cell height that limits each actuator line's tip (erf_actuator::tip_cell_heights of
     * the region its tips sweep, turbine i's in m_tip_dz[i]), kept from step to step: searched for every
     * turbine after a regrid (and at the first step), and for a turbine whose region (hub, tip radius,
     * axis, blades' off-plane reach) has moved more than tip_cells_tolerance() from the one last searched,
     * all such turbines in one batched search. One integer reduction a step decides whether any has moved.
     * With erf.moving_bodies.verify_tip_cells every step also searches each cached region afresh, one
     * region at a time, and aborts unless the two agree.
     */
    void update_tip_cells (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    /** How far (m) a value of a turbine's tip region may move before its cell height is searched again: a tenth of the smallest nominal cell size. */
    static amrex::Real tip_cells_tolerance (const amrex::Geometry& geom);
    bool m_tip_cells_valid{false};         //!< m_tip_rows and m_tip_dz hold the anchor level's current grids (cleared by check_coverage)
    std::vector<amrex::Real> m_tip_rows;   //!< per turbine, on every rank: the nine values of the region last searched
    std::vector<amrex::Real> m_tip_dz;     //!< per turbine, on every rank: the limiting cell height found for it (m)
    /** Build the lifting-line correction of every turbine that asks for it, once the node layout and chords are known. */
    void build_fllc (const amrex::Geometry& geom);
    /** The points of turbine i grouped by the kernel they are spread with, and that kernel's width (m). */
    struct PointGroup { std::vector<amrex::Real> pos; amrex::Real eps{0.0}; };
    /**
     * Turbine i's force nodes by kernel on a level of x spacing dx: the hub and blade nodes with the rotor's
     * width (epsilon), the tower nodes with epsilon_tower and, with nacelle drag, the hub with epsilon_nacelle;
     * with mode = none, which forces nothing, every node with epsilon. The checks of what the kernels reach use them.
     * On turbine i's owner.
     */
    std::vector<PointGroup> turbine_point_groups (int i, amrex::Real dx) const;
    /** Correct the blade velocity nodes of turbine i in uvw (3 per velocity node) with its lifting-line correction. */
    void apply_fllc (int i, double time, std::vector<amrex::Real>& uvw);
    /** Append a row to <output_root>_fllc.csv for every corrected turbine. */
    void write_fllc_diagnostics (double time, bool first);

    /** sampling = disk_corrected: scale the hub and blade node velocities of turbine i from the disk
     *  velocity to the free stream recovered with the filtered-disk factor (Shapiro et al. 2019) and
     *  momentum theory on the previous step's thrust; the tower nodes keep the resolved flow. */
    void apply_disk_correction (int i, std::vector<amrex::Real>& uvw, const amrex::Geometry& geom);
    void write_correction_diagnostics (double time, bool first);
    //! every turbine's <output_root>_erf.csv and _flow.csv rows (or their rank or farm files); collective
    void write_turbine_logs (double time, bool solution0);
    std::vector<amrex::Real> m_corr_uinf;                     //!< per turbine: the corrected free stream of the previous step (checkpointed)
    std::vector<erf_actuator::DiskCorrection> m_corr_last;    //!< per turbine: the last correction, for the diagnostics
    bool m_corr_written{false};
    std::vector<bool> m_corr_clamp_warned;                    //!< per turbine: the Ct clamp was reported
    std::vector<std::vector<std::unique_ptr<erf_actuator::FLLC>>> m_fllc;   //!< per turbine, one per blade (empty: off)
    bool m_fllc_built{false};
    bool m_fllc_written{false};
    std::string m_fllc_state_dir;   //!< checkpoint directory whose correction state the blades take over
#endif
};

#endif
