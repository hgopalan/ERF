#ifndef ERF_CONDUCTORS_H
#define ERF_CONDUCTORS_H

/** Conductors: the run's conductor lines, transformers and lattice towers, stepped with ERF on one AMR level. */

#include <array>
#include <map>
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
#include "ERF_ConductorLine.H"
#include "ERF_Gusts.H"
#include "ERF_RunningStats.H"
#include "ERF_Transformer.H"
#include "ERF_Tower.H"
#include "ERF_FrameTower.H"
#include "ERF_TowerDynamics.H"

/**
 * The conductor lines of a run (erf.conductors.*), each one MoorDyn system between its attachment
 * points, single spans or sections over suspension towers, stepped in lockstep with ERF on their anchor
 * level. ERF holds one of these and calls set_ground() once the mesh exists, then advance() every
 * step of the anchor level. Besides each line's own diagnostics, every pair of lines is watched
 * for how close their conductors come, against erf.conductors.flashover_distance, and every
 * transformer the lines end on for the load they put on it and how close any conductor comes to it.
 * The suspension towers of a line with a tower_type are lattice towers loaded by the wind (the
 * drag on their members, from the flow sampled at the members' nodes each step) and by the line
 * hanging from them; their foundations are checked for leg uplift and compression. Towers whose type
 * has a frequency bend under those loads (a one-mode tower model each), and towers whose type gives a
 * frame model (a SubDyn file, or one generated from the type's dimensions) bend as that frame: MoorDyn
 * then takes their cross-arms as coupled points that ERF moves, and hands back the lines' pull on them.
 * A frame tower whose members' design data is known has every member checked against its strength
 * (ASCE 10-15) every step. With an erf.conductors.gust_type the k-equation RANS's k sets the gusts
 * (ERF_Gusts.H): every span's static gust factor, and with event or random a gust added to the wind
 * the lines and towers take.
 */
class Conductors
{
public:
    using ConductorInputs_t = erf_conductors::ConductorInputs;

    /** nullptr unless erf.conductors.lines is given; aborts on inputs the conductor module does not support (validate_solver). */
    static std::unique_ptr<Conductors> create (int max_level);

    /**
     * Tell the lines the turbulence closure of their anchor level, which ERF calls once after
     * create(): whether it is the k-equation RANS, and its constant Cmu0. With an
     * erf.conductors.gust_type this aborts unless it is the k-equation RANS (the gusts come from its
     * k) and sets sigma_u / sqrt(k) to erf.conductors.gust_sigma_factor, or to 2.5 Cmu0 when that is
     * not given (ERF_Gusts.H). advance() aborts with gusts on unless this was called.
     */
    void set_closure (bool keqn_rans, amrex::Real Cmu0);
    /** sigma_u / sqrt(k) of the gusts once set_closure() has run; 0 without a gust_type. */
    double gust_sigma_factor () const { return m_gust_sigma; }
    /** With gust_type = event, the travelling gust, its origin placed once set_ground() has run. */
    const erf_conductors::GustEvent& gust_event () const { return m_event; }

    /**
     * Place the attachment points on the terrain (z - geometry.prob_lo[2] is their height above the
     * surface at x, y), write the MoorDyn inputs and create the lines. It also checks that MoorDyn's
     * frame holds the domain (erf.conductors.surface_offset), stands the transformers on the terrain,
     * dead-ends lines on them and builds the lattice towers, writes <diagnostics_dir>/ground.dat, and
     * aborts on a misplaced attachment point after writing it. ERF calls it from ERF::InitData;
     * advance() calls it on its first call when ERF has not. On a restart (restart_chkdir is the
     * checkpoint ERF restarts from) the lines continue from the state write_checkpoint() saved there,
     * with their statistics, step count and logs; a checkpoint without conductor state starts them
     * afresh.
     */
    void set_ground (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom,
                     const std::string& restart_chkdir = "");
    /**
     * Take the terrain's height from a surface rather than from the bottom of the mesh: an
     * immersed terrain stands on a flat mesh. surface holds the height (m) at the nodes of geom's
     * cells, x and y, as ERF's terrain reader fills it: one z plane, the box's lowest z index is read;
     * heights between nodes are bilinear. Call before set_ground().
     */
    void set_ground_surface (const amrex::FArrayBox& surface, const amrex::Geometry& geom);
    bool ground_set () const { return m_ground_set; }
    /** Whether set_ground() continued the lines from a checkpoint. */
    bool restored () const { return m_restored; }

    /**
     * Save the lines' state under <chkdir>/conductors: the step count, the time, the MoorDyn clock
     * offset and surface_offset (file state), every line's MoorDyn state (<name>.moordyn), every
     * running statistic, the towers' member loads (tower_loads) and the moving towers' model state
     * (tower_motion). Nothing is written before set_ground().
     */
    void write_checkpoint (const std::string& chkdir) const;
    /**
     * After a restart with drag_on_flow, spread the force the restored lines and the towers' members
     * exert on the air into the momentum sources again, so that the sources (and a plotfile written at the restart) are those of the step the
     * checkpoint was written at. Nothing happens on other levels, without drag_on_flow or on a fresh start.
     */
    void restore_sources (int lev, const amrex::MultiFab& U, const amrex::MultiFab* z_phys_nd,
                          const amrex::MultiFab* detJ_cc, const amrex::Geometry& geom);

    /**
     * Advance the lines by one step of the anchor level (other levels return at once): hand each its
     * wind, step it, measure its clearance to the terrain, update the statistics and the outputs, and,
     * with drag_on_flow, spread the force the lines and the towers' members exert on the air into the
     * momentum sources ERF adds over this same step (ERF advances the lines before the flow).
     * Units: time, dt (s). U, V, W are the face velocities of level lev at the start of the step.
     * cons is the conserved state of level lev at the start of the step, with a filled ghost cell:
     * with an erf.conductors.gust_type, k = (rho k) / rho is read from it at the line nodes (and at
     * the towers' drag nodes with gust_type event or random), and a null cons aborts; it is not read
     * otherwise. With gust_type event or random the lines and towers take the flow's wind plus the
     * gust (add_gusts).
     */
    void advance (int lev, double time, double dt,
                  const amrex::MultiFab& U, const amrex::MultiFab& V, const amrex::MultiFab& W,
                  const amrex::MultiFab* z_phys_nd, const amrex::MultiFab* detJ_cc, const amrex::Geometry& geom,
                  const amrex::MultiFab* cons = nullptr);

    /**
     * Add the force the lines and the towers' members exert on the air (minus the air's drag on them,
     * N/m^3) to the face momentum sources of the anchor level; nothing without drag_on_flow.
     */
    void add_momentum_sources (int lev, amrex::MultiFab& xmom_src, amrex::MultiFab& ymom_src, amrex::MultiFab& zmom_src) const;
    /** Whether the lines and towers put their drag into the flow (erf.conductors.drag_on_flow). */
    bool drag_on_flow () const { return m_in.drag_on_flow; }
    /**
     * The momentum source of the lines and the towers' members on the air (N/m^3), averaged from the faces to the cell centres, into
     * component comp of dst (three components x, y, z); zero on levels other than the anchor level or
     * before the first step.
     */
    void cell_sources (int lev, amrex::MultiFab& dst, int comp) const;
    /** The integral of the momentum source over the anchor level (N), as last spread. */
    const std::array<amrex::Real,3>& source_integral () const { return m_source_integral; }

    /** The pairs of lines watched for clashes (0-based indices into lines()), and their last closest approach. */
    const std::vector<std::pair<std::size_t,std::size_t>>& pairs () const { return m_pairs; }
    const std::vector<erf_conductors::Closest>& separations () const { return m_sep; }
    /** The transformers the lines end on, the lines' last load on each, and the closest any conductor came to each. */
    const std::vector<erf_conductors::Transformer>& transformers () const { return m_transformers; }
    const std::vector<erf_conductors::TransformerLoad>& transformer_loads () const { return m_tload; }
    const std::vector<erf_conductors::Closest>& transformer_clearances () const { return m_tclear; }
    /** The lattice towers of the lines with a tower_type, with the wind's last load on their members. */
    const std::vector<erf_towers::Tower>& towers () const { return m_towers; }
    /** Per tower, the wind (m/s, 3 per drag node, the order of its nodes()) its members took in the last step. */
    const std::vector<std::vector<amrex::Real>>& tower_winds () const { return m_tower_wind; }
    /** The lines hanging from each tower, in the order of its attachments: (line, tower of that line from 0). */
    const std::vector<std::vector<std::pair<std::size_t,int>>>& tower_lines () const { return m_tower_lines; }
    /** Each tower's structural model, null for a tower that stands still. */
    const std::vector<std::unique_ptr<erf_towers::TowerModel>>& tower_models () const { return m_models; }
    /**
     * The load on tower t's footings: from its model's support reactions when the model has them (a
     * frame), else Tower::foundation()'s estimate from the drag nodes' loads and inertial forces.
     */
    erf_towers::FoundationLoad tower_foundation (std::size_t t) const;
    /** The strength checks of tower t's members at present (ERF_MemberChecks.H); empty unless it is a frame tower with design data. */
    std::vector<erf_towers::MemberCheck> member_checks (std::size_t t) const;
    /**
     * The fewest coupling steps a group of lines whose towers move takes over one period of its
     * highest tower frequency; the count is also at least erf.conductors.substeps. In each coupling
     * step the towers advance under the mean of the lines' pull at the step's start and the iterated
     * pull at its end, then MoorDyn moves the cross-arms (its coupled points) from where they were to
     * where the towers took them. The step is repeated until that end pull converges (coupling_tolerance).
     */
    static constexpr int coupling_steps_per_period = 20;
    /**
     * Each coupling step is iterated until no component of the lines' pull on the towers at its end
     * changes by more than coupling_tolerance times the largest pull component at the step's start
     * (at least 1 N), at most coupling_max_iterations times (erf_conductors::coupling_converged). The
     * relaxation factor starts at coupling_initial_relaxation and Aitken's method adapts it within
     * [0.01, 1]. A pull that is not finite aborts the run.
     */
    static constexpr double coupling_tolerance = 1.0e-4;
    static constexpr int coupling_max_iterations = 50;
    /**
     * A group of lines and their towers whose coupling steps fail to converge this many times in a row, and
     * at least two ERF steps' worth of them, are diverging: the run stops.
     */
    static constexpr int coupling_max_unconverged_in_a_row = 20;
    static constexpr double coupling_initial_relaxation = 0.5;
    /** The most iterations a coupling step took in the last step of the anchor level, and how many did not converge. */
    int coupling_iterations () const { return m_coupling_iterations; }
    int coupling_unconverged () const { return m_coupling_unconverged; }

    int anchor_level () const { return m_anchor; }
    /** How far around each line or tower node the anchor level's grids must reach (m): the sampler's cells, and the drag's kernel. */
    amrex::Real coverage_reach (const amrex::Geometry& geom) const;
    /**
     * Abort unless the anchor level exists (finest_level, the run's finest level now): the lines are
     * placed on its terrain and stepped with it, so without it they would stand on flat ground and
     * never move.
     */
    void require_anchor_level (int finest_level) const;
    const ConductorInputs_t& inputs () const { return m_in; }
    const std::vector<std::unique_ptr<erf_conductors::ConductorLine>>& lines () const { return m_lines; }

private:
    Conductors (ConductorInputs_t in, int anchor);
    /**
     * The wind (m/s) at a line's kinematics points at the line's current position: ERF's velocity
     * sampled at the line nodes with the actuator core (each component from its own staggered grid,
     * linear in the physical height) and zero at MoorDyn's point entries after them, or the prescribed
     * velocity everywhere when one is given. Aborts, naming the line and the node, when a node has
     * left the domain or the anchor level's grids do not cover the cells the sampler reads. With nodes
     * given (and the flow sampled), it receives the line nodes' positions the wind was sampled at.
     */
    std::vector<amrex::Real> wind_at (const erf_conductors::ConductorLine& span,
                                      const amrex::MultiFab& U, const amrex::MultiFab& V, const amrex::MultiFab& W,
                                      const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom,
                                      std::vector<amrex::Real>* nodes = nullptr) const;
    /**
     * Abort, naming the line and the node (0-based over the line), when a line node lies outside the domain;
     * periodic directions are not checked.
     */
    void check_nodes_in_domain (const erf_conductors::ConductorLine& span, const std::vector<amrex::Real>& pos,
                                const amrex::Geometry& geom) const;
    /**
     * The terrain height (m) under each point: pos holds 3 coordinates per point (m); h receives one
     * height per point, from the immersed surface when one was given, else from z_phys_nd's bottom
     * node plane, or prob_lo z without terrain.
     */
    void ground_heights (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom, const std::vector<amrex::Real>& pos,
                         std::vector<amrex::Real>& h) const;
    /**
     * Read the terrain height under each node of every line (the immersed surface when
     * set_ground_surface() gave one, else z_phys_nd's k = 0 node plane, or prob_lo z without
     * terrain); aborts when a node has left the domain.
     */
    void update_ground_under_nodes (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    /**
     * Spread the force the lines and the towers' members exert on the air (minus the air's drag on
     * them) into the face momentum sources, and record their integral. Aborts on a force or position
     * that is not finite.
     */
    void spread_drag (const amrex::MultiFab& U, const amrex::MultiFab* z_phys_nd, const amrex::MultiFab* detJ_cc,
                      const amrex::Geometry& geom);
    /**
     * Append a row to <diagnostics_dir>/total_load.dat: the air's drag on the lines and the towers'
     * members (N), the force put into the air (N; zero without drag_on_flow) and the spread source's
     * integral (N). first truncates the file; the header is written after truncating, or when
     * appending to a missing or empty file.
     */
    void write_total_load (double time, bool first) const;

    /** Read the state write_checkpoint() saved under dir, create the lines from it and trim the logs to its time. */
    void restore (const std::string& dir);
    /** The running statistics of a line: one set per span, and one for its insulator strings. */
    void add_stats (const erf_conductors::LineInputs& s);
    /**
     * k = (rho k) / rho (m^2/s^2) from cons at the points pos (3 coordinates per point, m), one pass over
     * rho and one over rho k; a negative value from extrapolation below the first cell centre counts as 0,
     * and a point where the density is not positive gets NaN, for the caller to name.
     */
    std::vector<double> sample_k (const std::vector<amrex::Real>& pos, const amrex::MultiFab& cons,
                                  const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom) const;
    /**
     * With a gust_type, add one sample of every line's gust statistics, stamped t: per span, the
     * root-mean-square over its nodes of the horizontal wind speed and of its component normal to the span's
     * chord (from uvw[i], the flow's wind at line i's nodes), and the mean of k over them (k: per node of every
     * line in order, from sample_k).
     */
    void sample_gusts (double t, const std::vector<std::vector<amrex::Real>>& uvw, const std::vector<double>& k);
    /**
     * Place gust_type = event's gust (its origin, unless given, the domain's centre) and number gust_type =
     * random's processes: one per span (a line that shares another's towers or names it in gust_with takes
     * the processes of that line's spans) and one per tower. set_ground() calls it once the towers stand.
     */
    void setup_gusts (const amrex::Geometry& geom);
    /**
     * With gust_type event or random, add the gust to the flow's wind of the step that starts at time: to
     * uvw[i] (the wind at line i's kinematics points, its nodes at pos[i]; k, per node of every line in order)
     * and to tuvw (the wind at every tower's drag nodes in order, at tpos; tk, k there). The gust acts along
     * the mean horizontal wind of the span, string or tower, none where that is zero; sigma_u = c sqrt(k) is
     * taken from the mean k over each span's nodes and over each tower's drag nodes. event: g sigma_u times
     * the shape at the node's phase at the step's middle; random: sigma_u sqrt(B) z, z stepped by dt with
     * T = L_u / U from the root-mean-square horizontal wind over the span's (the owner span's) or tower's
     * nodes. A string takes the mean amplitude (event) or the mean gust (random) of the two spans beside it.
     * The gust at each span's middle node and at each tower's highest drag node goes to m_gust_series.
     */
    void add_gusts (double time, double dt, const std::vector<std::vector<amrex::Real>>& pos,
                    std::vector<std::vector<amrex::Real>>& uvw, const std::vector<double>& k,
                    const std::vector<amrex::Real>& tpos, std::vector<amrex::Real>& tuvw, const std::vector<double>& tk);
    /**
     * Append the gusts of the step that starts at time to <diagnostics_dir>/gust_series.dat (I/O rank):
     * the gust (m/s) at each span's middle node, then at each tower's highest drag node. first truncates the
     * file; the header is written after truncating, or when appending to a missing or empty file.
     */
    void write_gust_series (double time, bool first) const;
    /** The unit normal (x, y) to the horizontal projection of line i's span k's chord. */
    std::array<double,2> span_normal (std::size_t i, int k) const;
    /**
     * Write <diagnostics_dir>/gusts.csv from the gust statistics' means (I/O rank): one row per span;
     * advance() calls it once they hold a sample.
     */
    void write_gusts () const;
    /** The mean height of line i's span k above the ground at its two conductor points, the attachment height (m). */
    double span_height (std::size_t i, int k) const;
    /** The pairs of lines and their statistics. */
    void add_pair_stats ();
    /** The closest approach of every pair of lines' conductors, at their current positions. */
    void measure_separation ();
    /**
     * Append a row of the pairs' closest approach to <diagnostics_dir>/separation.dat. first truncates
     * the file; the header is written after truncating, or when appending to a missing or empty file.
     */
    void write_separation (double time, bool first) const;
    /**
     * Stand the transformers on the terrain and dead-end the lines on them: an empty string, or the
     * message to abort with on a misplaced end (aborts at once on a transformer outside the domain).
     */
    std::string place_transformers (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    /** The transformers' statistics. */
    void add_transformer_stats ();
    /** The lines' load on every transformer and the closest any conductor comes to its box, at the lines' current positions. */
    void measure_transformers ();
    /**
     * Append a row of the transformers' loads and clearances to <diagnostics_dir>/transformers.dat.
     * first truncates the file; the header is written after truncating, or when appending to a missing
     * or empty file.
     */
    void write_transformers (double time, bool first) const;
    /** Stand a lattice tower at every suspension point of the lines with a tower_type. */
    void build_towers ();
    /**
     * Write <diagnostics_dir>/asce74.csv when erf.conductors.asce74_wind is given: per span its wire height (the mean
     * of the conductor's heights above the ground at its two ends, less a third of its still-air vertical sag with
     * asce74_wire_height = effective), chord, unstretched length, and ASCE 74's quasi-static wind load, swing, sag,
     * blowout and tension under that gust (ERF_ASCE74.H), the swing about an inclined chord with asce74_inclined_spans.
     */
    void write_asce74 () const;
    /** The towers' statistics, and their members' where they are checked. */
    void add_tower_stats ();
    /** Tower t's cross-arm displacement (m, ERF frame): its centre's at the cross-arm's height from its tower model, else zero. */
    std::array<amrex::Real,3> arm_displacement (std::size_t t) const;
    /** A frame model built once and shared by the towers it fits, with its members' design data (null: none). */
    struct TowerFrame
    {
        std::shared_ptr<const erf_towers::Frame> frame;
        std::shared_ptr<const std::vector<erf_towers::MemberDesign>> designs;
        std::string source;     //!< where it came from, for messages
        std::vector<std::size_t> link_nodes;   //!< the nodes the towers' loads may be tied to (empty: every node)
        double first_frequency{0.0};           //!< the frame's first natural frequency (Hz), found once for its towers
    };
    /**
     * The frame model of tower tw: read from its type's frame_file (with member_file's design data), or
     * generated from its type's dimensions and its cross-arm's height (frame_panels), and then written to
     * <diagnostics_dir>/frame_<tower>.dat with its design data in frame_<tower>_members.dat; the members at
     * the type's steel_temperature. Aborts, naming the key, when it cannot be built.
     */
    TowerFrame make_frame (const erf_towers::Tower& tw) const;
    /** Tower t's model when it is a frame, else null. */
    const erf_towers::FrameTower* frame_tower (std::size_t t) const;
    /**
     * Write <diagnostics_dir>/tower_<name>_members.csv for every tower whose members are checked: per member its role,
     * joints, length, slenderness, the design stresses and strengths at its temperature, and the slenderness and
     * width-to-thickness flags.
     */
    void write_member_tables () const;
    /**
     * Append a row of every frame tower's node displacements (from where it stands under its weight at its first
     * temperature, x y z per node, frame axes, m) and, where checked, its members' utilisation to
     * <diagnostics_dir>/tower_<name>_frame.dat.
     * first truncates the file; the header (the nodes' positions, then the member ids) is written after truncating,
     * or when appending to a missing or empty file.
     */
    void write_tower_frames (double time, bool first) const;
    /**
     * The wind (m/s) at every tower node (3 per node, the towers in order): the flow sampled there, or
     * the prescribed velocity. Aborts on a sampled value that is not finite. With nodes given, it
     * receives the nodes' current positions (3 per node, m).
     */
    std::vector<amrex::Real> tower_wind (const amrex::MultiFab& U, const amrex::MultiFab& V, const amrex::MultiFab& W,
                                         const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom,
                                         std::vector<amrex::Real>* nodes = nullptr) const;
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
    /** Hand each tower the pull of its lines where they hang from it, at the lines' current state. */
    void load_towers_with_lines ();
    /** Hand tower t the pull of every line hanging from it, where each hangs (displaced by a moving tower), at the lines' current state. */
    void load_tower_with_lines (std::size_t t);
    /**
     * Append a row of the towers' drag and base moment to <diagnostics_dir>/towers.dat. first
     * truncates the file; the header is written after truncating, or when appending to a missing or
     * empty file.
     */
    void write_towers (double time, bool first) const;
    /**
     * The logs that restore() trims to the checkpoint time with trim_log_after (towers.dat, whose
     * rows carry a step's start time, is trimmed separately with at_too).
     */
    std::vector<std::string> log_files () const;

    ConductorInputs_t m_in;
    int m_anchor{0};
    bool m_ground_set{false};
    bool m_restored{false};
    int m_step{0};
    double m_time{0.0};                                  //!< the time the lines have reached (s)
    double m_t0{0.0};                                    //!< ERF's time when the lines' MoorDyn clocks read zero (s)
    std::vector<erf_conductors::LineInputs> m_placed;   //!< the lines with every attachment point (ends and towers) at its absolute height
    //! per attachment point of every line, in line order (num_spans()+1 per line: end_a, the towers, end_b): the
    //! terrain height under it (m); a share_towers line's tower entries hold the ground under the shared tower's base
    std::vector<amrex::Real> m_ground;
    std::vector<std::unique_ptr<erf_conductors::ConductorLine>> m_lines;
    std::vector<std::vector<erf_actuator::RunningStats>> m_stats;   //!< per line: per span, then the strings', from stats_start
    //! per line with gust_type = factor: per span, the root-mean-square wind and normal wind and the mean k over its nodes
    std::vector<erf_actuator::RunningStats> m_gust_stats;
    bool m_closure_set{false};                                     //!< set_closure() has run
    double m_gust_sigma{0.0};                                      //!< sigma_u / sqrt(k) of the gusts
    erf_conductors::GustEvent m_event;                             //!< gust_type = event: the travelling gust
    //! gust_type = random: per line, per span, its process in m_gust_z (a line may take another's: gust_owner())
    std::vector<std::vector<std::size_t>> m_span_process;
    std::vector<std::pair<std::size_t,int>> m_process_span;        //!< per span process: the (line, span) it belongs to
    std::vector<double> m_gust_z;          //!< the random processes (unit variance): the spans', then one per tower
    std::vector<std::string> m_process_names;   //!< per process: <line>_span<k> of the span it belongs to, or the tower's name
    bool m_gust_z_set{false};              //!< m_gust_z holds the last step's values (else the next step draws them)
    std::vector<double> m_gust_series;     //!< this step's gust at each span's middle node, then at each tower's top (m/s)
    std::vector<std::pair<std::size_t,std::size_t>> m_pairs;        //!< the pairs of lines watched for clashes
    std::vector<erf_actuator::RunningStats> m_pair_stats;           //!< per pair: the closest approach and the clash flag
    std::vector<erf_conductors::Closest> m_sep;                     //!< per pair: the last closest approach
    std::vector<erf_conductors::Transformer> m_transformers;        //!< standing on the terrain, with the ends on each
    std::vector<erf_conductors::TransformerLoad> m_tload;           //!< per transformer: the lines' last load
    std::vector<erf_conductors::Closest> m_tclear;                  //!< per transformer: the closest conductor, last
    std::vector<erf_actuator::RunningStats> m_tstats;               //!< per transformer: its load and clearance
    std::vector<erf_towers::Tower> m_towers;                        //!< the lines' lattice towers, line by line
    std::vector<std::vector<std::pair<std::size_t,int>>> m_tower_lines;   //!< per tower: (line, its tower index) per attachment
    /**
     * per line: (index into m_towers, attachment index on that tower), both 0-based, per tower of the line in
     * order
     */
    std::vector<std::vector<std::pair<std::size_t,std::size_t>>> m_line_towers;
    std::vector<std::vector<std::size_t>> m_groups;                 //!< lines that step together: those sharing towers
    int m_coupling_iterations{0};                                   //!< the most iterations of a coupling step, this step
    int m_coupling_unconverged{0};                                  //!< coupling steps that did not converge, this step
    int m_coupling_logged_iterations{0};                            //!< the most iterations since coupling.dat's last row
    int m_coupling_logged_unconverged{0};                           //!< unconverged coupling steps since coupling.dat's last row
    std::map<std::size_t,int> m_coupling_unconverged_in_a_row;      //!< per group (by its first line): unconverged coupling steps in a row
    bool m_coupling_warned{false};
    std::unique_ptr<amrex::FArrayBox> m_surface;                    //!< the terrain's height at the nodes, for an immersed terrain
    amrex::Geometry m_surface_geom;
    std::vector<std::unique_ptr<erf_towers::TowerModel>> m_models;  //!< per tower: its structural model, null when it stands still
    /** The frame models: per tower type with a frame_file, per tower type and cross-arm height (mm) with frame_panels. */
    std::map<std::string, TowerFrame> m_frames;
    std::vector<std::vector<amrex::Real>> m_tower_wind;             //!< per tower: the wind at its nodes (m/s), held for the step
    std::unique_ptr<erf_towers::MemberAero> m_aero;                 //!< the members' aerodynamics
    std::vector<erf_actuator::RunningStats> m_tower_stats;          //!< per tower: its drag and base moment
    /** per tower: its members' axial force and utilisation, null when they are not checked */
    std::vector<std::unique_ptr<erf_actuator::RunningStats>> m_member_stats;
    bool m_src_defined{false};
    /** the momentum sources of the lines and the towers' members on the anchor level (N/m^3) */
    amrex::MultiFab m_src_x, m_src_y, m_src_z;
    std::array<amrex::Real,3> m_source_integral{{0.0, 0.0, 0.0}};   //!< the integral of the sources over the anchor level (N)
    /** the air's drag on all lines and on the towers' members (N); total_load.dat drag_x..z */
    std::array<amrex::Real,3> m_drag_total{{0.0, 0.0, 0.0}};
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

/**
 * The convergence test of one iteration of a coupling step between lines and their moving towers.
 * F0 holds the lines' pulls on the towers at the coupling step's start, F the iterate, Fn the pulls
 * MoorDyn returned from it (N, 3 per attachment). converged is set when no |Fn - F| exceeds
 * tolerance times max(1 N, the largest |F0|). Returns an empty string, or the message when any of
 * F0, F, Fn is not finite (converged is then false).
 */
std::string coupling_converged (const std::vector<double>& F0, const std::vector<double>& F, const std::vector<double>& Fn,
                                double tolerance, bool& converged);

/**
 * Count a coupling step that converged or did not in in_a_row (the unconverged steps in a row,
 * reset by a converged one); true when the count reaches limit, the coupling diverging.
 */
inline bool coupling_diverged (bool converged, int& in_a_row, int limit)
{
    in_a_row = converged ? 0 : in_a_row + 1;
    return in_a_row >= limit;
}

/**
 * Whether a conductor checkpoint fits the run restarted from it: an empty string, else the message.
 * saved_tower_motion: the checkpoint holds tower_motion; towers_move: some tower type of the run
 * moves (a frequency or a frame); saved_offset: the checkpoint's surface_offset (m), NaN when the checkpoint has
 * no surface_offset record (it is then not compared); surface_offset: the run's (m).
 */
std::string restart_mismatch (bool saved_tower_motion, bool towers_move, double saved_offset, double surface_offset);

/**
 * Whether a tower's forms in a conductor checkpoint fit the run restarted from it: an empty string, else the
 * message naming the keys of the tower's type, whose name is type_name. The forms are (diagonal_wind_factor, arm_outside_shaft,
 * angle_principal_axes on a generated frame), 1 or 0 each; recorded: the checkpoint records the forms (an older
 * checkpoint does not, and was written with every form 0). They decide the drag nodes, the drag and the frame
 * the checkpoint's sway, statistics and node forces belong to, so a restart cannot change them.
 */
std::string tower_forms_mismatch (const std::string& type_name, bool recorded, const std::array<int,3>& saved,
                                  const std::array<int,3>& now);
} // namespace erf_conductors

#endif
