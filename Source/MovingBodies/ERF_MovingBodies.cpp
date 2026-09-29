#include "ERF_MovingBodies.H"

#include <cmath>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "ERF_ActuatorSampling.H"
#include "ERF_DataStruct.H"

using namespace amrex;

std::unique_ptr<MovingBodies>
MovingBodies::create (const SolverChoice& sc, int max_level)
{
    MovingBodiesInputs in = MovingBodiesInputs::read();
    if (!in.active) { return nullptr; }

    bool all_anelastic = true;
    for (int lev = 0; lev <= max_level && lev < static_cast<int>(sc.anelastic.size()); ++lev) {
        all_anelastic = all_anelastic && (sc.anelastic[lev] != 0);
    }

    // the step and stop time live in ERF's own inputs
    ParmParse pp("erf");
    Real fixed_dt = -1.0;
    pp.query("fixed_dt", fixed_dt);
    Real stop_time = -1.0;
    {
        ParmParse pp_root;
        pp_root.query("stop_time", stop_time);
        int max_step = -1;
        pp_root.query("max_step", max_step);
        if (stop_time <= 0.0 && max_step > 0 && fixed_dt > 0.0) {
            stop_time = max_step * fixed_dt;
        }
    }

    bool fpe_traps = false;
    {
        ParmParse pp_amrex("amrex");
        int trap = 0;
        for (const char* key : {"fpe_trap_invalid", "fpe_trap_zero", "fpe_trap_overflow"}) {
            trap = 0;
            pp_amrex.query(key, trap);
            fpe_traps = fpe_traps || (trap != 0);
        }
    }

    const std::string err = MovingBodiesInputs::validate_solver(all_anelastic, fixed_dt > 0.0, max_level, fpe_traps);
    if (!err.empty()) { Abort(err); }
    if (stop_time <= 0.0) {
        Abort("erf.moving_bodies: set stop_time or max_step so the OpenFAST stop time is known");
    }

    for (const MovingBodyInputs& b : in.bodies) {
        if (b.type == "openfast_turbine") {
#ifndef ERF_USE_OPENFAST
            Abort("erf.moving_bodies." + b.name + " is an openfast_turbine but ERF was built without ERF_ENABLE_OPENFAST");
#endif
        }
    }

    return std::unique_ptr<MovingBodies>(new MovingBodies(std::move(in), fixed_dt, stop_time));
}

MovingBodies::MovingBodies (MovingBodiesInputs in, double dt, double t_max)
    : m_in(std::move(in)), m_dt(dt)
{
    Print() << "Moving bodies: " << m_in.bodies.size() << " body(ies), ERF dt " << m_dt
            << ", OpenFAST stop time " << t_max << ", velocities "
            << (m_in.has_prescribed_velocity ? "prescribed" : "sampled from the flow") << "\n";
    // every body stands still in this version; its base is where the motion puts it at t = 0
    std::vector<MovingBodyInputs> placed = m_in.bodies;
    for (MovingBodyInputs& b : placed) {
        m_motion.push_back(std::make_unique<erf_actuator::FixedMotion>(b.base_pos));
        b.base_pos = m_motion.back()->position(0.0);
    }
#ifdef ERF_USE_OPENFAST
    // the models are set up now, so their inputs are checked at start-up; the first OpenFAST
    // solution waits for the first step, when the flow exists to be sampled
    m_driver = std::make_unique<erf_openfast::OpenFASTDriver>(placed);
    m_driver->init(m_dt, t_max);
    for (const auto& t : m_driver->turbines()) {
        Print() << "  " << t.name << ": OpenFAST dt " << t.dt_fast << ", " << t.num_substeps
                << " substeps per ERF step, " << t.num_blades << " blades, "
                << t.num_vel_nodes << " velocity nodes, " << t.num_force_nodes << " force nodes\n";
    }
#else
    amrex::ignore_unused(t_max);
#endif
}

void
MovingBodies::advance (int lev, double time, double dt,
                       const MultiFab& U, const MultiFab& V, const MultiFab& W,
                       const MultiFab* z_phys_nd, const Geometry& geom)
{
    if (lev != anchor_level()) { return; }
    // the OpenFAST substep count was fixed at init from erf.fixed_dt
    if (std::abs(dt - m_dt) > 1.0e-10 * m_dt) {
        Abort("erf.moving_bodies: the time step changed from " + std::to_string(m_dt) + " to " +
              std::to_string(dt) + "; OpenFAST needs the fixed step it was initialised with");
    }
    ++m_step;
#ifdef ERF_USE_OPENFAST
    if (!m_driver->solved0()) {
        // first step: OpenFAST's first solution sees the initial flow at its nodes
        supply_velocities(U, V, W, z_phys_nd, geom);
        m_driver->solution0();
        m_driver->write_diagnostics(time);
    }
    supply_velocities(U, V, W, z_phys_nd, geom);
    m_driver->step();
    if (m_step % m_in.diagnostics_int == 0) {
        m_driver->write_diagnostics(time + dt);
    }
#else
    amrex::ignore_unused(time, U, V, W, z_phys_nd, geom);
#endif
}

void
MovingBodies::supply_velocities (const MultiFab& U, const MultiFab& V, const MultiFab& W,
                                 const MultiFab* z_phys_nd, const Geometry& geom)
{
#ifdef ERF_USE_OPENFAST
    if (m_in.has_prescribed_velocity) {
        m_driver->set_uniform_velocity(m_in.prescribed_velocity);
        amrex::ignore_unused(U, V, W, z_phys_nd, geom);
        return;
    }
    // the flow at every velocity node, in the node order OpenFAST uses
    m_points.clear();
    for (const auto& t : m_driver->turbines()) { m_points.add_body(t.vel_pos); }
    erf_actuator::sample_velocity(U, V, W, z_phys_nd, geom, m_points.positions(), m_points.velocities());
    for (int i = 0; i < static_cast<int>(m_driver->turbines().size()); ++i) {
        m_driver->set_node_velocities(i, m_points.body_velocities(i));
    }
#else
    amrex::ignore_unused(U, V, W, z_phys_nd, geom);
#endif
}
