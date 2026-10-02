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
     * write the MoorDyn inputs and create the spans. Called once, before the first step.
     */
    void set_ground (const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);
    bool ground_set () const { return m_ground_set; }

    /** Advance the spans by one step of the anchor level (other levels return at once). */
    void advance (int lev, double time, double dt,
                  const amrex::MultiFab& U, const amrex::MultiFab& V, const amrex::MultiFab& W,
                  const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);

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

    ConductorInputs_t m_in;
    int m_anchor{0};
    bool m_ground_set{false};
    int m_step{0};
    std::vector<erf_conductors::SpanInputs> m_placed;   //!< the spans with their ends at absolute heights
    std::vector<amrex::Real> m_ground;                  //!< per end (2 per span): the terrain height
    std::vector<std::unique_ptr<erf_conductors::ConductorSpan>> m_spans;
};

#endif
