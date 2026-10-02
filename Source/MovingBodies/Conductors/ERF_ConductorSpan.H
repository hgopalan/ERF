#ifndef ERF_CONDUCTOR_SPAN_H
#define ERF_CONDUCTOR_SPAN_H

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

#include "ERF_ConductorInputs.H"
#include "ERF_MoorDynSystem.H"

namespace erf_conductors {

/**
 * One conductor line of erf.conductors.spans, a single span or a section of spans over
 * suspension towers, as one MoorDyn system stepped in lockstep with ERF. The line writes its
 * MoorDyn input from the ERF inputs, creates and initialises the MoorDyn system, takes the wind
 * at its nodes each step through MoorDyn's external kinematics, and reports positions and
 * tensions in ERF's frame (MoorDyn's frame is ERF's lowered by the surface offset). Every rank
 * runs its own identical copy: MoorDyn is serial and a line is small, and identical inputs give
 * identical states.
 *
 * The nodes are numbered as MoorDyn lists them: span 0 first, then the other spans in order, then
 * the insulator strings, one per tower. The per-span quantities take the span's index, 0 for a
 * single span.
 */
class ConductorSpan
{
public:
    /**
     * Create the line: writes <input_file>, creates the MoorDyn system and solves its initial
     * shape. The attachment points are at their absolute heights (the terrain already added).
     * With a saved state (a file written by save()) the initial-shape solve is skipped and the
     * line continues from that state instead.
     */
    ConductorSpan (const SpanInputs& s, const ConductorInputs& in, amrex::Real gravity, const std::string& input_file,
                   const std::string& saved_state = "");

    const std::string& name () const { return m_in.name; }
    const SpanInputs& inputs () const { return m_in; }
    const std::string& input_file () const { return m_file; }
    int num_spans () const { return m_in.num_spans(); }
    /** The insulator strings, one per tower when the line hangs from strings, else none. */
    int num_insulators () const { return m_in.has_insulators() ? static_cast<int>(m_in.towers.size()) : 0; }
    /** All nodes of the line: the spans', then the strings'. */
    unsigned num_nodes () const { return m_first.back(); }
    /** The first node of span k and the number of its nodes. */
    unsigned span_first_node (int k) const { return m_first[static_cast<std::size_t>(k)]; }
    unsigned span_num_nodes (int k) const { return m_first[static_cast<std::size_t>(k) + 1] - m_first[static_cast<std::size_t>(k)]; }
    unsigned num_kinematics_points () const { return m_nkin; }

    /**
     * The points MoorDyn wants the wind at, in ERF's frame (3*n): every node first, then MoorDyn's
     * points (the attachments and the strings' lower ends) and one entry at MoorDyn's origin.
     */
    std::vector<amrex::Real> kinematics_points () const;
    /** Hand MoorDyn the wind (3*n velocities, ERF's frame, no acceleration) valid at time t. */
    void set_wind (const std::vector<amrex::Real>& uvw, double t);
    /** The wind last handed to MoorDyn at a kinematics point (the nodes come first); zero before the first. */
    std::array<amrex::Real,3> wind_at_point (unsigned point) const;
    /** Advance the line by dt from time, in the configured number of MoorDyn calls. */
    void step (double time, double dt);
    /**
     * ERF's time when MoorDyn's clock read zero: zero for lines created at the start of a run,
     * the restart time for lines first created on a restart. step() requires MoorDyn's clock plus
     * this to be ERF's.
     */
    void set_clock_offset (double t0) { m_t0 = t0; }

    /** Every node in ERF's frame (3*num_nodes()). */
    std::vector<amrex::Real> node_positions () const;
    std::array<amrex::Real,3> node_position (unsigned node) const;
    /** The nodes of the spans in order, end_a to end_b, in ERF's frame (3 per node): the conductor's path. */
    std::vector<amrex::Real> conductor_path () const;
    /** The tension magnitude at the first and last node of span k (N). */
    amrex::Real tension_a (int k = 0) const;
    amrex::Real tension_b (int k = 0) const;
    amrex::Real max_tension (int k = 0) const;
    /**
     * The sag of span k's middle node below the chord between the span's attachment points (m),
     * and its lateral offset from the vertical plane of that chord (m): with insulator strings both
     * include how far the strings have let the conductor drop and swing.
     */
    amrex::Real mid_sag (int k = 0) const;
    amrex::Real mid_offset (int k = 0) const;
    /** The swing angle of span k's middle node about its chord, from the vertical (rad, positive with the offset). */
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

    /** The terrain height under each node (num_nodes() values), set by the caller after each step. */
    void set_ground_under_nodes (const std::vector<amrex::Real>& h);
    /** A node's height above the terrain under it (m); the ground is the plane z = 0 until set_ground_under_nodes(). */
    amrex::Real clearance (unsigned node) const;
    /** The smallest clearance along span k (m) and the node where it occurs. */
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
     * SpanInputs::span_root(k).dat, and one for the strings to <output_root>_insulators.dat; headers when first.
     */
    void write_diagnostics (double time, bool first) const;
    /** Append every node (position, clearance, tension, wind, drag) at this time to <output_root>_nodes.dat; header when first. */
    void write_nodes (double time, bool first) const;

    /** Write the MoorDyn state of the line (positions, velocities, the integrator's) to a file; call on one rank. */
    void save (const std::string& path) const;

    const erf_moordyn::MoorDynSystem& system () const { return *m_sys; }

private:
    /** The MoorDyn line (from 1) and the node on it of a node of this line. */
    void locate (unsigned node, unsigned& line, unsigned& local) const;
    void chord_frame_offsets (unsigned node, int k, amrex::Real& along, amrex::Real& down, amrex::Real& side) const;
    /** The horizontal unit normal of the line at tower j+1, to the left of the direction from end_a to end_b. */
    std::array<amrex::Real,3> across_direction (int j) const;

    SpanInputs m_in;
    amrex::Real m_offset{0.0};   //!< the surface offset (m)
    int m_substeps{1};
    double m_t0{0.0};            //!< ERF's time at MoorDyn's time zero
    std::string m_file;
    std::unique_ptr<erf_moordyn::MoorDynSystem> m_sys;
    std::vector<unsigned> m_first;     //!< the first node of each MoorDyn line, and the total after the last
    unsigned m_nkin{0};
    std::vector<amrex::Real> m_wind;   //!< 3*m_nkin, the wind last handed to MoorDyn
    std::vector<amrex::Real> m_ground; //!< the terrain height under each node
};

} // namespace erf_conductors

#endif
