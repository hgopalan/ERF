#ifndef ERF_CONDUCTORS_H
#define ERF_CONDUCTORS_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include "ERF_ConductorGeometry.H"
#include "ERF_ConductorInputs.H"
#include "ERF_ConductorSpan.H"
#include "ERF_RunningStats.H"
#include "ERF_Transformer.H"
#include "ERF_Tower.H"
#include "ERF_TowerDynamics.H"

/**
 * The conductor lines of a run (erf.conductors.*): MoorDyn lines between attachment points,
 * single spans or sections over suspension towers, stepped in lockstep with ERF on their anchor
 * level. ERF holds one of these and calls set_ground() once the mesh exists, then advance() every
 * step of the anchor level. Besides each line's own diagnostics, every pair of lines is watched
 * for how close their conductors come, against erf.conductors.flashover_distance, and every
 * transformer the lines end on for the load they put on it and how close any conductor comes to it.
 * The suspension towers of a line with a tower_type are lattice towers loaded by the wind (the
 * drag on their members, from the flow sampled at the members' nodes each step) and by the line
 * hanging from them; their foundations are checked for leg uplift and compression. Towers whose type
 * has a frequency bend under those loads (a one-mode tower model each): MoorDyn then takes their
 * cross-arms as coupled points that ERF moves, and hands back the lines' pull on them.
 */
class Conductors
{
public:
    using ConductorInputs_t = erf_conductors::ConductorInputs;

    /** nullptr unless erf.conductors.spans is given; aborts on inputs this version does not support. */
    static std::unique_ptr<Conductors> create (int max_level);

    /**
     * Place the attachments on the terrain (their z is the height above the surface at x, y),
     * write the MoorDyn inputs and create the spans. Called once, before the first step. On a
     * restart (restart_chkdir is the checkpoint ERF restarts from) the spans continue from the
     * state write_checkpoint() saved there, with their statistics, step count and logs; a
     * checkpoint without conductor state starts them afresh.
     */
    void set_ground (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom,
                     const std::string& restart_chkdir = "");
    /**
     * Take the terrain's height from a surface rather than from the bottom of the mesh: an
     * immersed terrain stands on a flat mesh. surface holds the height (m) at the nodes of geom's
     * cells, x and y, as ERF's terrain reader fills it (k = 0, any z index); call before set_ground().
     */
    void set_ground_surface (const amrex::FArrayBox& surface, const amrex::Geometry& geom);
    bool ground_set () const { return m_ground_set; }
    /** Whether set_ground() continued the spans from a checkpoint. */
    bool restored () const { return m_restored; }

    /**
     * Save the spans' state under <chkdir>/conductors: every span's MoorDyn state, the statistics,
     * the step count and the time. Nothing is written before set_ground().
     */
    void write_checkpoint (const std::string& chkdir) const;
    /**
     * After a restart with drag_on_flow, spread the restored lines' drag into the momentum sources
     * again, so that the sources (and a plotfile written at the restart) are those of the step the
     * checkpoint was written at. Nothing happens on other levels, without drag_on_flow or on a fresh start.
     */
    void restore_sources (int lev, const amrex::MultiFab& U, const amrex::MultiFab* z_phys_nd,
                          const amrex::MultiFab* detJ_cc, const amrex::Geometry& geom);

    /**
     * Advance the spans by one step of the anchor level (other levels return at once): hand each its
     * wind, step it, measure its clearance to the terrain, update the statistics and the outputs, and,
     * with drag_on_flow, spread the lines' drag on the air into the momentum sources of the next step.
     */
    void advance (int lev, double time, double dt,
                  const amrex::MultiFab& U, const amrex::MultiFab& V, const amrex::MultiFab& W,
                  const amrex::MultiFab* z_phys_nd, const amrex::MultiFab* detJ_cc, const amrex::Geometry& geom);

    /** Add the lines' drag on the air to the momentum sources of the anchor level (nothing without drag_on_flow). */
    void add_momentum_sources (int lev, amrex::MultiFab& xmom_src, amrex::MultiFab& ymom_src, amrex::MultiFab& zmom_src) const;
    /** Whether the lines put their drag into the flow (erf.conductors.drag_on_flow). */
    bool drag_on_flow () const { return m_in.drag_on_flow; }
    /**
     * The lines' momentum source on the air (N/m^3), averaged from the faces to the cell centres, into
     * component comp of dst (three components x, y, z); zero on levels other than the anchor level or
     * before the first step.
     */
    void cell_sources (int lev, amrex::MultiFab& dst, int comp) const;
    /** The integral of the momentum source over the anchor level (N), as last spread. */
    const std::array<amrex::Real,3>& source_integral () const { return m_source_integral; }

    /** The pairs of lines watched for clashes (indices into spans()), and their last closest approach. */
    const std::vector<std::pair<std::size_t,std::size_t>>& pairs () const { return m_pairs; }
    const std::vector<erf_conductors::Closest>& separations () const { return m_sep; }
    /** The transformers the lines end on, the lines' last load on each, and the closest any conductor came to each. */
    const std::vector<erf_conductors::Transformer>& transformers () const { return m_transformers; }
    const std::vector<erf_conductors::TransformerLoad>& transformer_loads () const { return m_tload; }
    const std::vector<erf_conductors::Closest>& transformer_clearances () const { return m_tclear; }
    /** The lattice towers of the lines with a tower_type, with the wind's last load on their members. */
    const std::vector<erf_towers::Tower>& towers () const { return m_towers; }
    /** The lines hanging from each tower, in the order of its attachments: (line, tower of that line from 0). */
    const std::vector<std::vector<std::pair<std::size_t,int>>>& tower_lines () const { return m_tower_lines; }
    /** Each tower's structural model, null for a tower that stands still. */
    const std::vector<std::unique_ptr<erf_towers::TowerModel>>& tower_models () const { return m_models; }
    /**
     * The coupling steps a line whose towers move takes over each period of its fastest tower, at
     * least: in each, its towers advance under the lines' pull from the end of the one before, then
     * MoorDyn moves their cross-arms from where they were to where they are now.
     */
    static constexpr int coupling_steps_per_period = 20;
    /**
     * Each coupling step is iterated until the lines' pull on the towers at its end changes by less
     * than coupling_tolerance of the largest pull, at most coupling_max_iterations times, starting
     * from the relaxation factor coupling_initial_relaxation (Aitken's method adapts it).
     */
    static constexpr double coupling_tolerance = 1.0e-4;
    static constexpr int coupling_max_iterations = 50;
    static constexpr double coupling_initial_relaxation = 0.5;
    /** The most iterations a coupling step took in the last step of the anchor level, and how many did not converge. */
    int coupling_iterations () const { return m_coupling_iterations; }
    int coupling_unconverged () const { return m_coupling_unconverged; }

    int anchor_level () const { return m_anchor; }
    const ConductorInputs_t& inputs () const { return m_in; }
    const std::vector<std::unique_ptr<erf_conductors::ConductorSpan>>& spans () const { return m_spans; }

private:
    Conductors (ConductorInputs_t in, int anchor);
    /**
     * The wind at a span's kinematics points where the line is now: ERF's velocity sampled at the line
     * nodes with the actuator core (each component from its own staggered grid, linear in the physical
     * height) and none at MoorDyn's fixed entries after them, or the prescribed velocity everywhere when
     * one is given. Aborts, naming the span and the node, when a node has left the domain or the anchor
     * level's grids do not cover the cells the sampler reads.
     */
    std::vector<amrex::Real> wind_at (const erf_conductors::ConductorSpan& span,
                                      const amrex::MultiFab& U, const amrex::MultiFab& V, const amrex::MultiFab& W,
                                      const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom) const;
    /** Abort, naming the span and the node, when a line node lies outside the domain (periodic directions wrap). */
    void check_nodes_in_domain (const erf_conductors::ConductorSpan& span, const std::vector<amrex::Real>& pos,
                                const amrex::Geometry& geom) const;
    /** The terrain height under each point (3 per point): the surface's when one was given, else the mesh's bottom. */
    void ground_heights (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom, const std::vector<amrex::Real>& pos,
                         std::vector<amrex::Real>& h) const;
    /** Read the terrain height under each node of every span (the k = 0 node plane, or the domain floor). */
    void update_ground_under_nodes (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    /** Spread the lines' drag on the air into the face momentum sources, and record their integral. */
    void spread_drag (const amrex::MultiFab& U, const amrex::MultiFab* z_phys_nd, const amrex::MultiFab* detJ_cc,
                      const amrex::Geometry& geom);
    void write_total_load (double time, bool first) const;

    /** Read the state write_checkpoint() saved under dir, create the spans from it and trim the logs to its time. */
    void restore (const std::string& dir);
    /** The running statistics of a line: one set per span, and one for its insulator strings. */
    void add_stats (const erf_conductors::SpanInputs& s);
    /** The pairs of lines and their statistics. */
    void add_pair_stats ();
    /** The closest approach of every pair of lines' conductors, where they are now. */
    void measure_separation ();
    /** Append a row of the pairs' closest approach to <diagnostics_dir>/separation.dat; header when first. */
    void write_separation (double time, bool first) const;
    /**
     * Stand the transformers on the terrain and dead-end the lines on them: an empty string, or the
     * message to abort with on a misplaced end (aborts at once on a transformer outside the domain).
     */
    std::string place_transformers (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    /** The transformers' statistics. */
    void add_transformer_stats ();
    /** The lines' load on every transformer and the closest any conductor comes to its box, where the lines are now. */
    void measure_transformers ();
    /** Append a row of the transformers' loads and clearances to <diagnostics_dir>/transformers.dat; header when first. */
    void write_transformers (double time, bool first) const;
    /** Stand a lattice tower at every suspension point of the lines with a tower_type. */
    void build_towers ();
    /** The towers' statistics. */
    void add_tower_stats ();
    /** The wind at every tower node (3 per node, the towers in order): the flow sampled there, or the prescribed velocity. */
    std::vector<amrex::Real> tower_wind (const amrex::MultiFab& U, const amrex::MultiFab& V, const amrex::MultiFab& W,
                                         const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom) const;
    /** The members' drag from the wind at their nodes (held for the step) and their motion, handed to each tower, with the lines' pull. */
    void load_towers (const std::vector<amrex::Real>& wind);
    /** The members' drag on tower t from the wind held for the step, at the nodes' current velocities. */
    void drag_on_tower (std::size_t t);
    /** Hand tower t its model's motion. */
    void move_tower (std::size_t t);
    /**
     * Advance the lines of a group whose towers move by dt from time, in coupling steps: each tower
     * advances under its members' drag and every line's pull, then MoorDyn moves each line's points
     * on the towers there. Lines that share towers form one group.
     */
    void step_moving_group (const std::vector<std::size_t>& lines, double time, double dt);
    /** Hand each tower the pull of its lines where they hang from it, as the lines are now. */
    void load_towers_with_lines ();
    /** The same for tower t. */
    void load_tower_with_lines (std::size_t t);
    /** Append a row of the towers' drag and base moment to <diagnostics_dir>/towers.dat; header when first. */
    void write_towers (double time, bool first) const;
    /** The logs a restart continues. */
    std::vector<std::string> log_files () const;

    ConductorInputs_t m_in;
    int m_anchor{0};
    bool m_ground_set{false};
    bool m_restored{false};
    int m_step{0};
    double m_time{0.0};                                  //!< the time the spans have reached
    double m_t0{0.0};                                    //!< ERF's time when the spans' MoorDyn clocks read zero
    std::vector<erf_conductors::SpanInputs> m_placed;   //!< the spans with their ends at absolute heights
    std::vector<amrex::Real> m_ground;                  //!< per end (2 per span): the terrain height
    std::vector<std::unique_ptr<erf_conductors::ConductorSpan>> m_spans;
    std::vector<std::vector<erf_actuator::RunningStats>> m_stats;   //!< per line: per span, then the strings', from stats_start
    std::vector<std::pair<std::size_t,std::size_t>> m_pairs;        //!< the pairs of lines watched for clashes
    std::vector<erf_actuator::RunningStats> m_pair_stats;           //!< per pair: the closest approach and the clash flag
    std::vector<erf_conductors::Closest> m_sep;                     //!< per pair: the last closest approach
    std::vector<erf_conductors::Transformer> m_transformers;        //!< standing on the terrain, with the ends on each
    std::vector<erf_conductors::TransformerLoad> m_tload;           //!< per transformer: the lines' last load
    std::vector<erf_conductors::Closest> m_tclear;                  //!< per transformer: the closest conductor, last
    std::vector<erf_actuator::RunningStats> m_tstats;               //!< per transformer: its load and clearance
    std::vector<erf_towers::Tower> m_towers;                        //!< the lines' lattice towers, line by line
    std::vector<std::vector<std::pair<std::size_t,int>>> m_tower_lines;   //!< per tower: (line, its tower index) per attachment
    std::vector<std::vector<std::pair<std::size_t,std::size_t>>> m_line_towers;   //!< per line: (tower, attachment) per tower of the line
    std::vector<std::vector<std::size_t>> m_groups;                 //!< lines that step together: those sharing towers
    int m_coupling_iterations{0};                                   //!< the most iterations of a coupling step, this step
    int m_coupling_unconverged{0};                                  //!< coupling steps that did not converge, this step
    bool m_coupling_warned{false};
    std::unique_ptr<amrex::FArrayBox> m_surface;                    //!< the terrain's height at the nodes, for an immersed terrain
    amrex::Geometry m_surface_geom;
    std::vector<std::unique_ptr<erf_towers::TowerModel>> m_models;  //!< per tower: its structural model, null when it stands still
    std::vector<std::vector<amrex::Real>> m_tower_wind;             //!< per tower: the wind at its nodes, held for the step
    std::unique_ptr<erf_towers::MemberAero> m_aero;                 //!< the members' aerodynamics
    std::vector<erf_actuator::RunningStats> m_tower_stats;          //!< per tower: its drag and base moment
    bool m_src_defined{false};
    amrex::MultiFab m_src_x, m_src_y, m_src_z;           //!< the lines' momentum sources on the anchor level
    std::array<amrex::Real,3> m_source_integral{{0.0, 0.0, 0.0}};
    std::array<amrex::Real,3> m_drag_total{{0.0, 0.0, 0.0}};   //!< the air's drag on all lines (N)
};

namespace erf_conductors {
/**
 * Drop the data rows of a whitespace-separated log whose time (the first column) is later than t,
 * keeping the header and every other line: a run restarted from a checkpoint at t continues the
 * log from there, without the rows the run it continues wrote after the checkpoint. With at_too
 * the row at t goes as well, for a log whose rows carry the time a step starts at (the restarted
 * run writes that row again). A missing file is left alone. Call on one rank.
 */
void trim_log_after (const std::string& fname, double t, bool at_too = false);
} // namespace erf_conductors

#endif
