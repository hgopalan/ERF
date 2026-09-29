#include "ERF_MovingBodies.H"

#include <cmath>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include "ERF_ActuatorSampling.H"
#include "ERF_ActuatorSpreading.H"
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
    // one spreading width for the run: the bodies share the momentum sources
    for (const MovingBodyInputs& b : in.bodies) {
        if (b.epsilon != in.bodies[0].epsilon) {
            Abort("erf.moving_bodies: every body must use the same epsilon in this version (" + b.name +
                  " differs from " + in.bodies[0].name + ")");
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
    std::vector<MovingBodyInputs> placed;
    std::vector<MovingBodyInputs> turbines;
    for (const MovingBodyInputs& b0 : m_in.bodies) {
        MovingBodyInputs b = b0;
        m_motion.push_back(std::make_unique<erf_actuator::FixedMotion>(b.base_pos));
        b.base_pos = m_motion.back()->position(0.0);
        placed.push_back(b);
        if (b.type == "ct_disk") {
            m_disks.push_back(std::make_unique<erf_actuator::PrescribedCtDisk>(b));
            Print() << "  " << b.name << ": uniform-Ct disk, R " << b.rotor_radius << " m, hub "
                    << b.hub_height << " m, Ct " << b.ct << ", " << m_disks.back()->num_points()
                    << " points, spreading " << b.epsilon << " dx\n";
        } else {
            turbines.push_back(b);
        }
    }
    m_epsilon_dx = m_in.bodies.empty() ? 2.0 : m_in.bodies[0].epsilon;
#ifdef ERF_USE_OPENFAST
    // the models are set up now, so their inputs are checked at start-up; the first OpenFAST
    // solution waits for the first step, when the flow exists to be sampled
    m_driver = std::make_unique<erf_openfast::OpenFASTDriver>(turbines);
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
                       const MultiFab* z_phys_nd, const MultiFab* detJ_cc, const Geometry& geom)
{
    if (lev != anchor_level()) { return; }
    // the OpenFAST substep count was fixed at init from erf.fixed_dt
    if (std::abs(dt - m_dt) > 1.0e-10 * m_dt) {
        Abort("erf.moving_bodies: the time step changed from " + std::to_string(m_dt) + " to " +
              std::to_string(dt) + "; OpenFAST needs the fixed step it was initialised with");
    }
    ++m_step;
    const bool first = (m_step == 1);
#ifdef ERF_USE_OPENFAST
    if (first && !m_driver->solved0()) {
        // first step: OpenFAST's first solution sees the initial flow at its nodes
        supply_velocities(U, V, W, z_phys_nd, geom);
        m_driver->solution0();
        m_driver->write_diagnostics(time);
    }
#endif
    supply_velocities(U, V, W, z_phys_nd, geom);
#ifdef ERF_USE_OPENFAST
    m_driver->step();
    if (m_step % m_in.diagnostics_int == 0) {
        m_driver->write_diagnostics(time + dt);
    }
#endif
    // the disks' forces come from the velocities just sampled, so the source is fixed over the
    // step ERF is about to take
    if (!m_disks.empty()) {
        spread_sources(U, z_phys_nd, detJ_cc, geom);
        if (first) { for (auto& d : m_disks) { d->open_diagnostics(); } }
        if (first || m_step % m_in.diagnostics_int == 0) {
            const Real fx = erf_actuator::integrate_source(0, m_src_x, detJ_cc, geom);
            const Real fy = erf_actuator::integrate_source(1, m_src_y, detJ_cc, geom);
            const Real fz = erf_actuator::integrate_source(2, m_src_z, detJ_cc, geom);
            for (auto& d : m_disks) {
                const auto n = d->normal();
                // the integrated source over all bodies, projected on this disk's normal and
                // signed as a thrust: equals the disk's thrust when it is the only body
                const Real spread_thrust = -(n[0]*fx + n[1]*fy + n[2]*fz);
                d->write_diagnostics(time, spread_thrust);
            }
        }
    }
}

void
MovingBodies::spread_sources (const MultiFab& U, const MultiFab* z_phys_nd,
                              const MultiFab* detJ_cc, const Geometry& geom)
{
    const BoxArray ba = amrex::convert(U.boxArray(), IntVect(0,0,0));
    if (!m_src_defined || m_src_x.boxArray() != amrex::convert(ba, IntVect(1,0,0))) {
        const DistributionMapping& dm = U.DistributionMap();
        m_src_x.define(amrex::convert(ba, IntVect(1,0,0)), dm, 1, 0);
        m_src_y.define(amrex::convert(ba, IntVect(0,1,0)), dm, 1, 0);
        m_src_z.define(amrex::convert(ba, IntVect(0,0,1)), dm, 1, 0);
        m_src_defined = true;
    }
    std::vector<Real> pos, force;
    for (const auto& d : m_disks) {
        const auto& p = d->disk_points();
        const auto& f = d->forces();
        pos.insert(pos.end(), p.begin(), p.end());
        force.insert(force.end(), f.begin(), f.end());
    }
    const Real eps = m_epsilon_dx * geom.CellSize(0);
    erf_actuator::spread_forces(pos, force, eps, z_phys_nd, detJ_cc, geom, m_src_x, m_src_y, m_src_z);
}

void
MovingBodies::add_momentum_sources (int lev, MultiFab& xmom_src, MultiFab& ymom_src, MultiFab& zmom_src) const
{
    if (lev != anchor_level() || !m_src_defined) { return; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(xmom_src.boxArray() == m_src_x.boxArray(),
                                     "MovingBodies: the momentum sources were built on a different grid than ERF's");
    MultiFab::Add(xmom_src, m_src_x, 0, 0, 1, 0);
    MultiFab::Add(ymom_src, m_src_y, 0, 0, 1, 0);
    MultiFab::Add(zmom_src, m_src_z, 0, 0, 1, 0);
}

void
MovingBodies::supply_velocities (const MultiFab& U, const MultiFab& V, const MultiFab& W,
                                 const MultiFab* z_phys_nd, const Geometry& geom)
{
    // one flattened list: the OpenFAST turbines' velocity nodes, then each disk's upstream
    // sampling points and disk points
    m_points.clear();
    int nturb = 0;
#ifdef ERF_USE_OPENFAST
    for (const auto& t : m_driver->turbines()) { m_points.add_body(t.vel_pos); ++nturb; }
#endif
    for (const auto& d : m_disks) {
        m_points.add_body(d->sample_points());
        m_points.add_body(d->disk_points());
    }
    if (m_in.has_prescribed_velocity) {
        auto& vel = m_points.velocities();
        for (std::size_t p = 0; p < vel.size() / 3; ++p) {
            for (int c = 0; c < 3; ++c) { vel[3*p+c] = m_in.prescribed_velocity[c]; }
        }
        amrex::ignore_unused(U, V, W, z_phys_nd, geom);
    } else if (m_points.num_points() > 0) {
        erf_actuator::sample_velocity(U, V, W, z_phys_nd, geom, m_points.positions(), m_points.velocities());
    }
#ifdef ERF_USE_OPENFAST
    for (int i = 0; i < nturb; ++i) {
        m_driver->set_node_velocities(i, m_points.body_velocities(i));
    }
#endif
    for (std::size_t k = 0; k < m_disks.size(); ++k) {
        const int b = nturb + 2 * static_cast<int>(k);
        m_disks[k]->update(m_points.body_velocities(b), m_points.body_velocities(b + 1));
    }
}
