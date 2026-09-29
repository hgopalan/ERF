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
#ifdef ERF_USE_OPENFAST
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
 * forces of the bodies (the OpenFAST turbines' loads as actuator-disk rings, the prescribed-Ct
 * disks) are spread onto face-centred momentum sources, which ERF adds to its momentum
 * right-hand side in every stage of the step through add_momentum_sources().
 */
class MovingBodies
{
public:
    /**
     * Read erf.moving_bodies.*, check the solver settings and initialise the bodies.
     * Returns nullptr when no bodies are given.
     */
    static std::unique_ptr<MovingBodies> create (const SolverChoice& sc, int max_level);

    /**
     * Advance the bodies by one ERF step; a no-op unless lev is the anchor level.
     * @param U,V,W the level's face-centred velocities at the start of the step, ghost cells filled
     * @param z_phys_nd nodal heights, or nullptr on a uniform-dz mesh
     * @param geom the level's geometry
     */
    void advance (int lev, double time, double dt,
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

    int anchor_level () const { return 0; }
    const MovingBodiesInputs& inputs () const { return m_in; }

private:
    MovingBodies (MovingBodiesInputs in, double dt, double t_max);

    /** Sample (or prescribe) the flow at every body's velocity nodes and hand it to the bodies. */
    void supply_velocities (const amrex::MultiFab& U, const amrex::MultiFab& V, const amrex::MultiFab& W,
                            const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);

    MovingBodiesInputs m_in;
    double m_dt{0.0};
    int m_step{0};
    std::vector<std::unique_ptr<erf_actuator::BodyMotion>> m_motion;  //!< one per body
    erf_actuator::ActuatorPoints m_points;                             //!< all bodies' velocity nodes
    std::vector<std::unique_ptr<erf_actuator::PrescribedCtDisk>> m_disks;
    std::vector<int> m_turbine_points_t;                               //!< ring points per turbine (0: mode none)
    amrex::Real m_epsilon_dx{2.0};                                     //!< spreading width, dx units
    bool m_axis_checked{false};
    bool m_src_defined{false};
    amrex::MultiFab m_src_x, m_src_y, m_src_z;                         //!< anchor-level momentum sources

    /** Spread the bodies' forces onto the momentum sources (defined on first use). */
    void spread_sources (const amrex::MultiFab& U, const amrex::MultiFab* z_phys_nd,
                         const amrex::MultiFab* detJ_cc, const amrex::Geometry& geom);

    /** Append time and the integrated source vector to <diagnostics_dir>/momentum_source.csv. */
    void write_source_diagnostics (double time, const amrex::MultiFab* detJ_cc,
                                   const amrex::Geometry& geom, bool first);
    bool any_forcing () const;
#ifdef ERF_USE_OPENFAST
    std::unique_ptr<erf_openfast::OpenFASTDriver> m_driver;
#endif
};

#endif
