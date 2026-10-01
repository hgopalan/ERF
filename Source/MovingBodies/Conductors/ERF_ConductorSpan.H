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
 * One conductor span: a MoorDyn line between two fixed attachment points, stepped in lockstep
 * with ERF. The span writes its MoorDyn input from the ERF inputs, creates and initialises the
 * MoorDyn system, takes the wind at its line nodes each step through MoorDyn's external
 * kinematics, and reports the node positions and tensions in ERF's frame (MoorDyn's frame is
 * ERF's lowered by the surface offset). Every rank runs its own identical copy of the line:
 * MoorDyn is serial and a span is small, and identical inputs give identical states.
 */
class ConductorSpan
{
public:
    /**
     * Create the span: writes <input_file>, creates the MoorDyn system and solves its initial
     * shape. ends are the attachment points at their absolute heights (the terrain already added).
     */
    ConductorSpan (const SpanInputs& s, const ConductorInputs& in, amrex::Real gravity, const std::string& input_file);

    const std::string& name () const { return m_in.name; }
    const SpanInputs& inputs () const { return m_in; }
    const std::string& input_file () const { return m_file; }
    unsigned num_nodes () const { return m_nodes; }
    unsigned num_kinematics_points () const { return m_nkin; }

    /**
     * The points MoorDyn wants the wind at, in ERF's frame (3*n): the line nodes first, then MoorDyn's
     * fixed entries (the attachment points and one entry at MoorDyn's origin).
     */
    std::vector<amrex::Real> kinematics_points () const;
    /** Hand MoorDyn the wind (3*n velocities, ERF's frame, no acceleration) valid at time t. */
    void set_wind (const std::vector<amrex::Real>& uvw, double t);
    /** The wind last handed to MoorDyn at a kinematics point (the line nodes come first); zero before the first. */
    std::array<amrex::Real,3> wind_at_point (unsigned point) const;
    /** Advance the line by dt from time, in the configured number of MoorDyn calls. */
    void step (double time, double dt);

    /** The line nodes in ERF's frame (3*n). */
    std::vector<amrex::Real> node_positions () const;
    std::array<amrex::Real,3> node_position (unsigned node) const;
    /** The tension magnitude at the first and last node (N). */
    amrex::Real tension_a () const;
    amrex::Real tension_b () const;
    amrex::Real max_tension () const;
    /** The sag of the middle node below the chord (m) and its lateral offset from the vertical plane of the chord (m). */
    amrex::Real mid_sag () const;
    amrex::Real mid_offset () const;
    /** The swing angle of the middle node about the chord, from the vertical (rad, positive with the offset). */
    amrex::Real swing_angle () const;

    /** Append a diagnostics row (whitespace separated, like ERF's data logs) to <output_root>.dat; header when first. */
    void write_diagnostics (double time, bool first) const;

    const erf_moordyn::MoorDynSystem& system () const { return *m_sys; }

private:
    std::array<amrex::Real,3> chord_frame_offsets (unsigned node, amrex::Real& along, amrex::Real& down, amrex::Real& side) const;

    SpanInputs m_in;
    amrex::Real m_offset{0.0};   //!< the surface offset (m)
    int m_substeps{1};
    std::string m_file;
    std::unique_ptr<erf_moordyn::MoorDynSystem> m_sys;
    unsigned m_nodes{0};
    unsigned m_nkin{0};
    std::vector<amrex::Real> m_wind;   //!< 3*m_nkin, the wind last handed to MoorDyn
};

} // namespace erf_conductors

#endif
