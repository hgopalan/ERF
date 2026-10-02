#ifndef ERF_CONDUCTORS_H
#define ERF_CONDUCTORS_H

#include <memory>
#include <string>
#include <vector>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include "ERF_ConductorInputs.H"
#include "ERF_ConductorSpan.H"
#include "ERF_RunningStats.H"

/**
 * The conductor spans of a run (erf.conductors.*): MoorDyn lines between fixed attachment
 * points, stepped in lockstep with ERF on their anchor level. ERF holds one of these and calls
 * set_ground() once the mesh exists, then advance() every step of the anchor level.
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
    /** Read the terrain height under each node of every span (the k = 0 node plane, or the domain floor). */
    void update_ground_under_nodes (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    /** Spread the lines' drag on the air into the face momentum sources, and record their integral. */
    void spread_drag (const amrex::MultiFab& U, const amrex::MultiFab* z_phys_nd, const amrex::MultiFab* detJ_cc,
                      const amrex::Geometry& geom);
    void write_total_load (double time, bool first) const;

    /** Read the state write_checkpoint() saved under dir, create the spans from it and trim the logs to its time. */
    void restore (const std::string& dir);

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
    std::vector<erf_actuator::RunningStats> m_stats;     //!< per span, from stats_start
    bool m_src_defined{false};
    amrex::MultiFab m_src_x, m_src_y, m_src_z;           //!< the lines' momentum sources on the anchor level
    std::array<amrex::Real,3> m_source_integral{{0.0, 0.0, 0.0}};
    std::array<amrex::Real,3> m_drag_total{{0.0, 0.0, 0.0}};   //!< the air's drag on all lines (N)
};

namespace erf_conductors {
/**
 * Drop the data rows of a whitespace-separated log whose time (the first column) is later than t,
 * keeping the header and every other line: a run restarted from a checkpoint at t continues the
 * log from there, without the rows the run it continues wrote after the checkpoint. A missing
 * file is left alone. Call on one rank.
 */
void trim_log_after (const std::string& fname, double t);
} // namespace erf_conductors

#endif
