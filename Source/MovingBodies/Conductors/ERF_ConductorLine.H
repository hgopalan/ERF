#ifndef ERF_CONDUCTOR_LINE_H
#define ERF_CONDUCTOR_LINE_H

/** ConductorLine: one conductor line (erf.conductors.<name>) as one MoorDyn-C system, with its outputs in ERF's frame. */

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_REAL.H>

#include "ERF_ConductorInputs.H"
#include "ERF_MoorDynSystem.H"

namespace erf_conductors {

/**
 * One conductor line of erf.conductors.lines, a single span or a section of spans over
 * suspension towers, as one MoorDyn system stepped in lockstep with ERF. The line writes its
 * MoorDyn input from the ERF inputs, creates and initialises the MoorDyn system, takes the wind
 * at its nodes each step through MoorDyn's external kinematics, and reports positions and
 * tensions in ERF's frame. MoorDyn's frame is ERF's frame shifted in z: z_md = z_erf -
 * surface_offset, so MoorDyn's origin and free surface sit at ERF z = surface_offset. Every rank
 * runs its own identical copy: MoorDyn is serial and a line is small, and identical inputs give
 * identical states.
 *
 * The line's nodes are numbered from 0 as MoorDyn lists them: span 0 first, then the other spans
 * in order, then the insulator strings, one per tower. The per-span quantities take the span's
 * index k from 0 (0 for a single span); the per-string quantities take the string's index j from
 * 0, the string at tower j+1.
 */
class ConductorLine
{
public:
    /**
     * Create the line: writes <input_file>, creates the MoorDyn system and solves its initial
     * shape. The attachment points are at their absolute heights (the terrain already added).
     * With a saved state (a file written by save()) the initial-shape solve is skipped and the
     * line continues from that state instead. When the line's towers move, their cross-arms are
     * MoorDyn's coupled points, displaced from where they stand unloaded by tower_displacement (3 per
     * tower, ERF's frame; empty: not displaced). The I/O rank writes <input_file>; every rank then
     * reads it.
     * Units: gravity (m/s^2); tower_displacement (m).
     */
    ConductorLine (const LineInputs& s, const ConductorInputs& in, amrex::Real gravity, const std::string& input_file,
                   const std::string& saved_state = "", const std::vector<amrex::Real>& tower_displacement = {});

    /** The line's name in erf.conductors.lines. */
    const std::string& name () const { return m_in.name; }
    /** The line's inputs, the attachment points at their absolute heights. */
    const LineInputs& inputs () const { return m_in; }
    /** The MoorDyn input file the line was created from. */
    const std::string& input_file () const { return m_file; }
    /** The number of spans: one more than the towers. */
    int num_spans () const { return m_in.num_spans(); }
    /** The insulator strings, one per tower when the line hangs from strings, else none. */
    int num_insulators () const { return m_in.has_insulators() ? static_cast<int>(m_in.towers.size()) : 0; }
    /** All nodes of the line, numbered from 0: the spans' in span order, then the strings'. */
    unsigned num_nodes () const { return m_first.back(); }
    /**
     * The first node (0-based over the line) of span k (k from 0) and the number of its nodes.
     * span_first_node(num_spans()) is the first insulator-string node (num_nodes() without strings).
     */
    unsigned span_first_node (int k) const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(k >= 0 && k <= num_spans(), "ConductorLine::span_first_node: no such span");
        return m_first[static_cast<std::size_t>(k)];
    }
    unsigned span_num_nodes (int k) const
    {
        check_span(k);
        return m_first[static_cast<std::size_t>(k) + 1] - m_first[static_cast<std::size_t>(k)];
    }
    /** The first node (0-based over the line) of insulator string j (j from 0, the string at tower j+1) and the number of its nodes. */
    unsigned string_first_node (int j) const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(j >= 0 && j < num_insulators(), "ConductorLine::string_first_node: no such string");
        return m_first[static_cast<std::size_t>(num_spans() + j)];
    }
    unsigned string_num_nodes (int j) const
    {
        const unsigned first = string_first_node(j);
        return m_first[static_cast<std::size_t>(num_spans() + j) + 1] - first;
    }
    /** The number of points MoorDyn takes the wind at (at least num_nodes(): the line nodes come first). */
    unsigned num_kinematics_points () const { return m_nkin; }

    /**
     * The points MoorDyn wants the wind at, in ERF's frame (3*n, m, n = num_kinematics_points()):
     * every node first, then MoorDyn's point entries (the attachment points and the strings' free
     * lower ends) and one entry at MoorDyn's origin.
     */
    std::vector<amrex::Real> kinematics_points () const;
    /**
     * Hand MoorDyn the wind (3*n velocities, m/s, ERF's frame, n = num_kinematics_points(); zero
     * acceleration), valid at time t (s). Aborts, naming the line and the point, on a value that is
     * not finite.
     */
    void set_wind (const std::vector<amrex::Real>& uvw, double t);
    /**
     * The wind (m/s) last handed to MoorDyn at kinematics point `point` (0-based; the line nodes come
     * first); zero before the first set_wind() call or for point >= num_kinematics_points().
     */
    std::array<amrex::Real,3> wind_at_point (unsigned point) const;
    /**
     * Advance the line by dt from time (both s), in erf.conductors.substeps MoorDyn calls, the towers
     * held where they are. Aborts, naming the line, when MoorDyn returns a node position or coupled
     * force that is not finite.
     */
    void step (double time, double dt);
    /** Whether the towers' cross-arms are MoorDyn's coupled points, which the caller moves (the towers move). */
    bool towers_move () const { return m_coupled; }
    /**
     * Advance the line by dt from time (both s) in one MoorDyn call, moving the towers' cross-arms:
     * each starts displaced by displacement (m, 3 per tower, ERF's frame, from where it stands
     * unloaded) and moves at velocity (m/s, 3 per tower) over the step. Only for a line whose towers
     * move. Aborts, naming the line, when MoorDyn returns a value that is not finite.
     */
    void step_coupled (double time, double dt, const std::vector<amrex::Real>& displacement,
                       const std::vector<amrex::Real>& velocity);
    /**
     * The force (N) MoorDyn's last step left on tower j+1's coupled point (j from 0): this line's pull
     * on the cross-arm there (its two spans', or its insulator string's). Force components are the
     * same in ERF's and MoorDyn's frames.
     */
    std::array<amrex::Real,3> coupled_force (int j) const;
    /** The line's whole state in memory (MoorDyn's and the coupled points'), so that a coupled step can be redone. */
    struct State {
        std::vector<std::uint64_t> moordyn;   //!< MoorDyn's serialized state
        std::vector<double> x;                //!< the coupled points (m, MoorDyn's frame)
        std::vector<double> f;                //!< the forces on them (N)
        double clock{0.0};                    //!< MoorDyn's clock (s)
    };
    State state () const { return State{m_sys->serialize(), m_x, m_f, m_clock}; }
    void set_state (const State& s)
    {
        m_sys->deserialize(s.moordyn);
        m_x = s.x;
        m_f = s.f;
        m_clock = s.clock;
    }
    /** Abort, naming the line, unless MoorDyn's clock after its last step plus the clock offset reads time. */
    void check_clock (double time) const;
    /**
     * ERF's time when MoorDyn's clock read zero: zero for lines created at the start of a run,
     * the restart time for lines first created on a restart. step() requires MoorDyn's clock plus
     * this to be ERF's.
     */
    void set_clock_offset (double t0) { m_t0 = t0; }

    /** Every node in ERF's frame (3*num_nodes(), m). */
    std::vector<amrex::Real> node_positions () const;
    /** Node `node` (0-based over the line) in ERF's frame (m). */
    std::array<amrex::Real,3> node_position (unsigned node) const;
    /** The nodes of the spans in order, end_a to end_b, in ERF's frame (3 per node, m): the conductor's path. */
    std::vector<amrex::Real> conductor_path () const;
    /** The tension magnitude at the first and last node of span k (N). */
    amrex::Real tension_a (int k = 0) const;
    amrex::Real tension_b (int k = 0) const;
    /** The largest tension magnitude along span k (N), as MoorDyn reports it. */
    amrex::Real max_tension (int k = 0) const;
    /**
     * The pull of the line on its dead end at end_a (end 0) or end_b (end 1) (N): the net force
     * MoorDyn finds on the line's end node, which the fixed support takes, the end segment's tension
     * with the node's share of weight and drag.
     * Frame: ERF (z up); the force ON the support BY the line.
     */
    std::array<amrex::Real,3> end_force (int end) const;
    /**
     * The pull of the line on tower j+1 (j from 0), where its attachment point is (N): the net force
     * MoorDyn finds on the top node of the insulator string hanging there, or, clamped at the
     * tower, on the end nodes of the two spans meeting there. In still air it carries the weight
     * of the conductor between the mid-spans either side (and of the string); in a crosswind also
     * the wind on that length.
     */
    std::array<amrex::Real,3> tower_force (int j) const;
    /**
     * The sag (m) of span k's node first + (n-1)/2 (the middle node when segments is even), measured
     * along the downward normal of the chord in the chord's vertical plane. The chord joins span k's
     * attachment points where they stand unloaded (the inputs with the terrain added; a moving
     * tower's displacement is not included). mid_offset (m) is the node's offset normal to that
     * plane, positive to the left of end_a -> end_b looking down. With insulator strings both include
     * how far the strings have let the conductor drop and swing.
     */
    amrex::Real mid_sag (int k = 0) const;
    amrex::Real mid_offset (int k = 0) const;
    /** The swing angle (rad) of span k's middle node about its chord: atan2(mid_offset, mid_sag), from the chord's downward normal. */
    amrex::Real swing_angle (int k = 0) const;
    /** The angle of insulator string j (at tower j+1) from the vertical (rad). */
    amrex::Real insulator_swing (int j) const;
    /**
     * The angle of insulator string j across the line, from the vertical (rad): positive towards the
     * horizontal normal of the line that points to the left of the direction from end_a to end_b
     * looking down, the side a positive mid_offset is on.
     */
    amrex::Real insulator_swing_across (int j) const;
    /** The tension at the top of insulator string j (N). */
    amrex::Real insulator_tension (int j) const;
    /**
     * The conductor's weight insulator string j carries (N): the line's downward pull on its tower
     * less the string's own weight in the fluid. Near zero or negative, the spans either side rise
     * away from the tower and pull it up (uplift): the string carries nothing and flips over the
     * cross-arm, where a strain tower or a taller one is needed.
     */
    amrex::Real string_load (int j) const;
    /**
     * Whether string j (at tower j+1) carries less than uplift_fraction times weight_per_length (N/m,
     * net of buoyancy) times half the horizontal lengths of the spans either side of its tower.
     */
    bool string_in_uplift (int j, amrex::Real weight_per_length) const;
    /** The fraction (-) of the half spans' weight below which a string counts as in uplift. */
    static constexpr amrex::Real uplift_fraction = 0.1;

    /** The terrain height (m) under each node (num_nodes() values), set by the caller after each step. */
    void set_ground_under_nodes (const std::vector<amrex::Real>& h);
    /** A node's height above the terrain under it (m); the ground is the plane z = 0 until set_ground_under_nodes(). */
    amrex::Real clearance (unsigned node) const;
    /** The smallest clearance along span k (m) and the node (0-based over the line) where it occurs. */
    amrex::Real min_clearance (unsigned& node, int k = 0) const;
    /** The air's drag on a node (N), from MoorDyn's last step. */
    std::array<amrex::Real,3> node_drag (unsigned node) const;
    /** The air's drag on the whole line, strings included (N). */
    std::array<amrex::Real,3> total_drag () const;
    /** The air's drag on span k (N). */
    std::array<amrex::Real,3> span_drag (int k) const;
    /** The tension magnitude at a node (N). */
    amrex::Real node_tension (unsigned node) const;

    /**
     * Append a diagnostics row per span (whitespace separated, like ERF's data logs) to
     * LineInputs::span_root(k).dat, and one for the strings to <output_root>_insulators.dat. first
     * truncates the files (a fresh run); a header is written after truncating, or when appending to
     * a missing or empty file.
     */
    void write_diagnostics (double time, bool first) const;
    /**
     * Append every node (position, clearance, tension, wind, drag) at this time to
     * <output_root>_nodes.dat. first truncates the file; the header is written after truncating, or
     * when appending to a missing or empty file.
     */
    void write_nodes (double time, bool first) const;

    /** Write the MoorDyn state of the line (positions, velocities, the integrator's) to a file; call on one rank. */
    void save (const std::string& path) const;

    const erf_moordyn::MoorDynSystem& system () const { return *m_sys; }

private:
    /** The MoorDyn line (numbered from 1) and the node on it (from 0) of line node `node` (0-based over this line). */
    void locate (unsigned node, unsigned& line, unsigned& local) const;
    /**
     * Split node `node`'s offset (m) from span k's first attachment point into along (on the chord),
     * down (on the chord's downward normal in its vertical plane) and side (normal to that plane,
     * positive to the left of end_a -> end_b looking down).
     */
    void chord_frame_offsets (unsigned node, int k, amrex::Real& along, amrex::Real& down, amrex::Real& side) const;
    /** The horizontal unit normal of the line at tower j+1, to the left of the direction from end_a to end_b. */
    std::array<amrex::Real,3> across_direction (int j) const;
    /** Abort unless k is a span of the line (0 .. num_spans()-1). */
    void check_span (int k) const;
    /** Abort unless j is an insulator string of the line (0 .. num_insulators()-1). */
    void check_string (int j) const;
    /** Abort unless node is a node of the line (0 .. num_nodes()-1). */
    void check_node (unsigned node) const;
    /** Abort, naming the line, when MoorDyn's last step left a node position or a coupled force that is not finite. */
    void check_outputs_finite (double time) const;

    LineInputs m_in;
    amrex::Real m_offset{0.0};   //!< the surface offset (m)
    amrex::Real m_rho{1.225};    //!< the air's density (kg/m^3)
    amrex::Real m_g{9.81};       //!< gravity (m/s^2)
    int m_substeps{1};           //!< MoorDyn calls per step()
    double m_t0{0.0};            //!< ERF's time at MoorDyn's time zero (s)
    double m_clock{0.0};         //!< MoorDyn's clock after its last step (s)
    std::string m_file;
    std::unique_ptr<erf_moordyn::MoorDynSystem> m_sys;
    std::vector<unsigned> m_first;     //!< the first node of each MoorDyn line, and the total after the last
    unsigned m_nkin{0};                //!< the number of kinematics points
    bool m_coupled{false};             //!< the towers' cross-arms are MoorDyn's coupled points
    std::vector<double> m_x, m_f;      //!< the coupled points where the last step left them (m, MoorDyn's frame) and their forces (N)
    std::vector<amrex::Real> m_wind;   //!< 3*m_nkin, the wind last handed to MoorDyn (m/s)
    std::vector<amrex::Real> m_ground; //!< the terrain height under each node (m)
};

} // namespace erf_conductors

#endif
