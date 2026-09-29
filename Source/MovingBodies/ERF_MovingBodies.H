#ifndef ERF_MOVING_BODIES_H
#define ERF_MOVING_BODIES_H

#include <memory>

#include <AMReX_REAL.H>

#include "ERF_MovingBodiesInputs.H"

#ifdef ERF_USE_OPENFAST
#include "ERF_OpenFASTDriver.H"
#endif

struct SolverChoice;

/**
 * Owner of the moving bodies (OpenFAST turbines in this version) and the single place ERF
 * talks to them: created once from erf.moving_bodies.*, advanced once per ERF step on the
 * anchor level. This version drives OpenFAST with a uniform prescribed velocity and adds no
 * forcing to the flow.
 */
class MovingBodies
{
public:
    /**
     * Read erf.moving_bodies.*, check the solver settings and initialise the bodies.
     * Returns nullptr when no bodies are given.
     */
    static std::unique_ptr<MovingBodies> create (const SolverChoice& sc, int max_level);

    /** Advance the bodies by one ERF step; a no-op unless lev is the anchor level. */
    void advance (int lev, double time, double dt);

    int anchor_level () const { return 0; }
    const MovingBodiesInputs& inputs () const { return m_in; }

private:
    MovingBodies (MovingBodiesInputs in, double dt, double t_max);

    MovingBodiesInputs m_in;
    double m_dt{0.0};
    int m_step{0};
#ifdef ERF_USE_OPENFAST
    std::unique_ptr<erf_openfast::OpenFASTDriver> m_driver;
#endif
};

#endif
