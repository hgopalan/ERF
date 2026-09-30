#include "ERF_MovingBodies.H"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include "ERF_ActuatorSampling.H"
#include "ERF_DiagnosticsLog.H"
#include "ERF_ActuatorSpreading.H"
#include "ERF_DataStruct.H"

using namespace amrex;

std::unique_ptr<MovingBodies>
MovingBodies::create (const SolverChoice& sc, int max_level, bool restarting)
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

    return std::unique_ptr<MovingBodies>(new MovingBodies(std::move(in), fixed_dt, stop_time, restarting));
}

MovingBodies::MovingBodies (MovingBodiesInputs in, double dt, double t_max, bool restarting)
    : m_in(std::move(in)), m_dt(dt), m_t_max(t_max), m_restarting(restarting)
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
            m_turbine_points_t.push_back((b.mode == "none") ? 0 : b.num_points_t);
        }
    }
    m_epsilon_dx = m_in.bodies.empty() ? 2.0 : m_in.bodies[0].epsilon;
    for (const auto& d : m_disks) {
        m_disk_stats.emplace_back(d->name(), m_in.diagnostics_dir + "/" + d->name(),
                                  std::vector<std::string>{"u_inf", "u_disk", "thrust", "power"});
    }
    for (const MovingBodyInputs& b : turbines) {
        m_turbine_stats.emplace_back(b.name, b.output_root,
                                     std::vector<std::string>{"thrust_shaft", "thrust_x", "torque", "power", "rotor_speed",
                                                              "hub_u", "hub_v", "hub_w", "blade_mean_u"});
    }
#ifdef ERF_USE_OPENFAST
    // the models are set up now, so their inputs are checked at start-up; the first OpenFAST
    // solution waits for the first step, when the flow exists to be sampled
    m_driver = std::make_unique<erf_openfast::OpenFASTDriver>(turbines);
    if (!m_restarting) {
        m_driver->init(m_dt, t_max);
        for (const auto& t : m_driver->turbines()) {
            Print() << "  " << t.name << ": OpenFAST dt " << t.dt_fast << ", " << t.num_substeps
                    << " substeps per ERF step, " << t.num_blades << " blades, "
                    << t.num_vel_nodes << " velocity nodes, " << t.num_force_nodes << " force nodes\n";
        }
    }
    // on a restart the turbines are restored by read_checkpoint(), from ERF's checkpoint read
#else
    amrex::ignore_unused(t_max);
#endif
}

void
MovingBodies::write_checkpoint (const std::string& chkdir) const
{
    const std::string dir = chkdir + "/moving_bodies";
    if (ParallelDescriptor::IOProcessor()) {
        UtilCreateDirectory(dir, 0755);
        std::ofstream out(dir + "/state", std::ios::trunc);
        if (!out) { Abort("cannot write the moving-bodies checkpoint state '" + dir + "/state'"); }
        out << "step = " << m_step << "\n";
#ifdef ERF_USE_OPENFAST
        for (const auto& t : m_driver->turbines()) {
            out << "time_index " << t.name << " = " << t.time_index << "\n";
        }
#endif
    }
    ParallelDescriptor::Barrier();
    for (const auto& w : m_wakes) { w->write_state(dir); }
    for (const auto& st : m_turbine_stats) { st.write_state(dir); }
    for (const auto& st : m_disk_stats) { st.write_state(dir); }
#ifdef ERF_USE_OPENFAST
    m_driver->create_checkpoint(dir + "/");
#endif
}

void
MovingBodies::read_checkpoint (const std::string& chkdir)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_restarting && !m_restored,
                                     "MovingBodies::read_checkpoint: only once, and only on a restart");
    const std::string dir = chkdir + "/moving_bodies";
    if (!FileExists(dir + "/state")) {
        // a checkpoint written without bodies (a precursor, say): the bodies start afresh here
        Print() << "Moving bodies: the checkpoint " << chkdir << " holds no moving-bodies state; the bodies start now\n";
#ifdef ERF_USE_OPENFAST
        m_driver->init(m_dt, m_t_max);
#endif
        m_restored = true;
        return;
    }
    m_wake_state_dir = dir;
    for (auto& st : m_turbine_stats) { st.read_state(dir); }
    for (auto& st : m_disk_stats) { st.read_state(dir); }
    // the state file: the step count, and the OpenFAST time index each turbine must report
    std::map<std::string,int> time_index;
    {
        Vector<char> chars;
        ParallelDescriptor::ReadAndBcastFile(dir + "/state", chars);
        std::istringstream in(std::string(chars.dataPtr(), chars.size()));
        std::string line;
        bool have_step = false;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string key, name, eq;
            if (!(ls >> key)) { continue; }
            if (key == "step") {
                if (!(ls >> eq >> m_step) || eq != "=") { Abort("malformed step line in '" + dir + "/state'"); }
                have_step = true;
            } else if (key == "time_index") {
                int n = 0;
                if (!(ls >> name >> eq >> n) || eq != "=") { Abort("malformed time_index line in '" + dir + "/state'"); }
                time_index[name] = n;
            }
        }
        if (!have_step) { Abort("no step count in the moving-bodies checkpoint '" + dir + "/state'"); }
    }
#ifdef ERF_USE_OPENFAST
    m_driver->restart(dir + "/", m_dt);
    for (const auto& t : m_driver->turbines()) {
        const auto it = time_index.find(t.name);
        if (it == time_index.end()) {
            Abort("the moving-bodies checkpoint '" + dir + "' has no turbine " + t.name +
                  "; the erf.moving_bodies block must match the run being restarted");
        }
        if (it->second != t.time_index) {
            Abort("erf.moving_bodies." + t.name + ": the OpenFAST checkpoint is at time index " +
                  std::to_string(t.time_index) + " but ERF's checkpoint expects " + std::to_string(it->second));
        }
        Print() << "  " << t.name << ": restored from " << dir << "/" << t.name << ".chkp at OpenFAST time index "
                << t.time_index << ", " << t.num_substeps << " substeps per ERF step\n";
    }
#endif
    Print() << "Moving bodies: restarted after step " << m_step << "\n";
    m_restored = true;
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
    if (m_restarting && !m_restored) {
        Abort("erf.moving_bodies: restarting, but the checkpoint holds no moving-bodies state (was it written by a run with bodies?)");
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
    if (first || m_step % m_in.diagnostics_int == 0) {
        m_driver->write_diagnostics(time + dt);
    }
#endif
    // the bodies' forces come from the velocities just sampled (the disks) or from the
    // OpenFAST step just taken (the turbines), so the source is fixed over the step ERF is
    // about to take
    if (any_forcing()) {
        spread_sources(U, z_phys_nd, detJ_cc, geom);
        if (first) { for (auto& d : m_disks) { d->open_diagnostics(true); } }
        if (m_restored && !m_logs_reopened) {
            for (auto& d : m_disks) { d->open_diagnostics(false); }
            m_logs_reopened = true;
        }
        if (first || m_step % m_in.diagnostics_int == 0) {
            write_source_diagnostics(time, detJ_cc, geom, first);
        }
    }
    if (time + dt >= m_in.avg_start) { accumulate_statistics(time + dt); }
    if (m_in.wake.active() && !m_in.has_prescribed_velocity && (first || m_step % m_in.wake.interval == 0)) {
        write_wake_diagnostics(time, U, V, W, z_phys_nd, geom);
    }
}

void
MovingBodies::accumulate_statistics (double time)
{
#ifdef ERF_USE_OPENFAST
    const auto& turbs = m_driver->turbines();
    for (std::size_t i = 0; i < turbs.size(); ++i) {
        const auto& t = turbs[i];
        const std::array<Real,3> f = m_driver->thrust(t);
        std::array<Real,3> hub, blade;
        m_driver->node_velocity_means(t, hub, blade);
        const Real q = m_driver->torque(t);
        m_turbine_stats[i].accumulate(time, {f[0]*t.hub_axis[0] + f[1]*t.hub_axis[1] + f[2]*t.hub_axis[2], f[0], q,
                                             q * t.rotor_speed, t.rotor_speed, hub[0], hub[1], hub[2], blade[0]});
        if (m_step % m_in.diagnostics_int == 0) { m_turbine_stats[i].write(); }
    }
#endif
    for (std::size_t k = 0; k < m_disks.size(); ++k) {
        const auto& d = m_disks[k];
        m_disk_stats[k].accumulate(time, {d->free_stream_speed(), d->disk_speed(), d->thrust(), d->power()});
        if (m_step % m_in.diagnostics_int == 0) { m_disk_stats[k].write(); }
    }
}

void
MovingBodies::build_wake_lines (const Geometry& geom)
{
    const auto& w = m_in.wake;
    auto add = [&](const std::string& name, const std::string& root, const std::array<Real,3>& hub,
                   const std::array<Real,3>& axis, Real diameter) {
        m_wakes.push_back(std::make_unique<erf_actuator::WakeLines>(name, root, hub, axis, diameter,
                                                                    w.lines_xD, w.half_width, w.num_points,
                                                                    static_cast<Real>(geom.ProbLo(2))));
    };
#ifdef ERF_USE_OPENFAST
    for (const auto& t : m_driver->turbines()) {
        // the rotor diameter from the outermost blade force node
        Real r2max = 0.0;
        const int nfb = t.num_blades * t.num_force_pts_blade;
        for (int nd = 1; nd <= nfb && nd < t.num_force_nodes; ++nd) {
            Real r2 = 0.0;
            for (int d = 0; d < 3; ++d) { const Real dd = t.force_pos[3*nd+d] - t.hub_pos[d]; r2 += dd * dd; }
            r2max = std::max(r2max, r2);
        }
        if (!(r2max > 0.0)) { Abort("erf.moving_bodies.wake: " + t.name + " has no blade force nodes to size the wake lines from"); }
        add(t.name, t.output_root, t.hub_pos, t.hub_axis, Real(2.0) * std::sqrt(r2max));
    }
#endif
    for (const auto& d : m_disks) {
        add(d->name(), m_in.diagnostics_dir + "/" + d->name(), d->center(), d->normal(), Real(2.0) * d->radius());
    }
    // every sampling point must lie in the domain (periodic directions wrap)
    const auto plo = geom.ProbLoArray();
    const auto phi = geom.ProbHiArray();
    for (const auto& wl : m_wakes) {
        const auto& pos = wl->positions();
        for (std::size_t p = 0; p < pos.size() / 3; ++p) {
            for (int d = 0; d < 3; ++d) {
                if (geom.isPeriodic(d)) { continue; }
                if (pos[3*p+d] < plo[d] || pos[3*p+d] > phi[d]) {
                    Abort("erf.moving_bodies.wake: a sampling line of " + wl->name() + " leaves the domain at (" +
                          std::to_string(pos[3*p]) + ", " + std::to_string(pos[3*p+1]) + ", " + std::to_string(pos[3*p+2]) +
                          ") m; shorten lines_xD or half_width");
                }
            }
        }
        if (!m_wake_state_dir.empty() && wl->read_state(m_wake_state_dir)) {
            Print() << "  " << wl->name() << ": wake running average restored (" << wl->num_samples() << " samples)\n";
        }
        Print() << "  " << wl->name() << ": wake lines at " << w.lines_xD.size() << " downstream distances, "
                << wl->num_points() << " points, diameter " << wl->diameter() << " m\n";
    }
    m_wakes_built = true;
}

void
MovingBodies::write_wake_diagnostics (double time, const MultiFab& U, const MultiFab& V, const MultiFab& W,
                                      const MultiFab* z_phys_nd, const Geometry& geom)
{
    if (!m_wakes_built) { build_wake_lines(geom); }
    std::vector<Real> vel;
    for (auto& wl : m_wakes) {
        erf_actuator::sample_velocity(U, V, W, z_phys_nd, geom, wl->positions(), vel);
        if (time >= m_in.avg_start) { wl->accumulate(vel); }
        wl->write_instantaneous(time, vel, !m_restored && !m_wake_written);
        wl->write_average();
    }
    m_wake_written = true;
}

bool
MovingBodies::any_forcing () const
{
    if (!m_disks.empty()) { return true; }
    for (int n : m_turbine_points_t) { if (n > 0) { return true; } }
    return false;
}

void
MovingBodies::write_source_diagnostics (double time, const MultiFab* detJ_cc, const Geometry& geom, bool first)
{
    const Real fx = erf_actuator::integrate_source(0, m_src_x, detJ_cc, geom);
    const Real fy = erf_actuator::integrate_source(1, m_src_y, detJ_cc, geom);
    const Real fz = erf_actuator::integrate_source(2, m_src_z, detJ_cc, geom);
    for (auto& d : m_disks) {
        const auto n = d->normal();
        // the integrated source over all bodies, projected on this disk's normal and signed
        // as a thrust: equals the disk's thrust when it is the only body
        d->write_diagnostics(time, -(n[0]*fx + n[1]*fy + n[2]*fz));
    }
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream out;
        if (erf_actuator::open_log(out, m_in.diagnostics_dir + "/momentum_source.csv", first)) {
            out << "time,fx,fy,fz\n";
        }
        out << std::setprecision(10) << time << "," << fx << "," << fy << "," << fz << "\n";
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
#ifdef ERF_USE_OPENFAST
    // each turbine's loads as actuator-disk rings (mode = adm; none puts no force in the flow)
    {
        const auto& turbs = m_driver->turbines();
        for (std::size_t i = 0; i < turbs.size(); ++i) {
            if (m_turbine_points_t[i] == 0) { continue; }
            if (!m_axis_checked) {
                const Real s = erf_actuator::max_out_of_plane_sine(turbs[i]);
                const double deg = std::asin(std::min(s, Real(1.0))) * 180.0 / 3.14159265358979323846;
                if (s > Real(0.35)) {   // ~20 degrees: well beyond precone plus shaft tilt
                    Abort("erf.moving_bodies." + turbs[i].name + ": the blade nodes lie up to " +
                          std::to_string(deg) +
                          " degrees out of the plane normal to the hub axis; the hub orientation convention does not match this OpenFAST");
                }
                const auto& n = turbs[i].hub_axis;
                Print() << "erf.moving_bodies." << turbs[i].name << ": hub axis (" << n[0] << ", " << n[1] << ", " << n[2]
                        << "), blade force nodes within " << deg << " degrees of the rotor plane, "
                        << m_turbine_points_t[i] << " points per ring\n";
            }
            erf_actuator::adm_rings(turbs[i], m_turbine_points_t[i], pos, force);
        }
        m_axis_checked = true;
    }
#endif
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
