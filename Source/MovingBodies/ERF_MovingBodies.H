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

#ifdef ERF_USE_OPENFAST
#include "ERF_OpenFASTDriver.H"
#endif

struct SolverChoice;

/**
 * Owner of the moving bodies (OpenFAST turbines in this version) and the single place ERF
 * talks to them: created once from erf.moving_bodies.*, advanced once per ERF step on the
 * anchor level. Each step the flow velocity is sampled at every body's velocity nodes (or the
 * prescribed uniform velocity is used) and handed to the bodies, which are then stepped. No
 * forcing is added to the flow yet.
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
                  const amrex::MultiFab* z_phys_nd, const amrex::Geometry& geom);

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
#ifdef ERF_USE_OPENFAST
    std::unique_ptr<erf_openfast::OpenFASTDriver> m_driver;
#endif
};

#endif
