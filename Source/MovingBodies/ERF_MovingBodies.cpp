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
#ifdef ERF_USE_OPENFAST
#include "ERF_OpenFASTAudit.H"
#endif
#include "ERF_DiagnosticsLog.H"
#include "ERF_ActuatorSpreading.H"
#include "ERF_DataStruct.H"
#include "ERF_Constants.H"
#include "ERF_IndexDefines.H"

using namespace amrex;

std::unique_ptr<MovingBodies>
MovingBodies::create (const SolverChoice& sc, int max_level, const Vector<double>& fixed_dt_levels, bool restarting)
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

    const int anchor = MovingBodiesInputs::resolve_anchor_level(in.anchor_level, max_level);
    const std::string err = MovingBodiesInputs::validate_solver(all_anelastic, fixed_dt > 0.0, max_level, anchor, fpe_traps);
    if (!err.empty()) { Abort(err); }
    // the anchor level's step: erf.fixed_dt divided by the sub-cycling ratios down to that level
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(anchor < static_cast<int>(fixed_dt_levels.size()) && fixed_dt_levels[anchor] > 0.0,
                                     "erf.moving_bodies: no fixed step is known for the anchor level");
    fixed_dt = static_cast<Real>(fixed_dt_levels[anchor]);
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

    return std::unique_ptr<MovingBodies>(new MovingBodies(std::move(in), anchor, fixed_dt, stop_time, restarting));
}

MovingBodies::MovingBodies (MovingBodiesInputs in, int anchor, double dt, double t_max, bool restarting)
    : m_in(std::move(in)), m_anchor(anchor), m_dt(dt), m_t_max(t_max), m_restarting(restarting)
{
    Print() << "Moving bodies: " << m_in.bodies.size() << " body(ies) on level " << m_anchor << ", ERF dt there " << m_dt
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
            m_turbine_mode.push_back(b.mode);
            m_turbine_points_t.push_back(b.num_points_t);
            m_turbine_in.push_back(b);
            if (b.mode != "none" && b.num_force_points_tower > 0) {
                Print() << "  " << b.name << ": tower loads forced on " << b.num_force_points_tower << " points\n";
            }
            if (b.mode != "none" && b.nacelle_cd > 0.0) {
                Print() << "  " << b.name << ": nacelle drag point, cd " << b.nacelle_cd << ", area " << b.nacelle_area
                        << " m^2, density " << b.air_density << "\n";
            }
            if (b.fllc) {
                Print() << "  " << b.name << ": filtered lifting-line correction, optimal kernel " << b.fllc_eps_chord
                        << " chords, relaxation " << b.fllc_relax << ", from t = " << b.fllc_start_time << " s\n";
            }
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
                    << t.num_vel_nodes << " velocity nodes, " << t.num_force_nodes << " force nodes, on rank "
                    << t.owner_rank << "\n";
        }
    }
    for (std::size_t i = 0; i < turbines.size(); ++i) {
        // the tower's wake now reaches the blades through the flow: AeroDyn's own tower-shadow
        // correction would count it twice
        if (turbines[i].mode != "none" && turbines[i].num_force_points_tower > 0) {
            const std::string w = erf_openfast::check_tower_shadow_off(turbines[i].fst_file);
            if (!w.empty()) { Print() << "WARNING: erf.moving_bodies." << turbines[i].name << ": " << w << "\n"; }
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
    if (ParallelDescriptor::IOProcessor()) {
        const auto& turbs = m_driver->turbines();
        // every turbine's node loads and velocities at the checkpoint: what depends on the
        // loads of the step before a restart (the lifting-line correction) continues from them
        for (const auto& t : turbs) {
            std::ofstream out(dir + "/" + t.name + "_nodes.dat", std::ios::trunc);
            if (!out) { Abort("cannot write the node-load checkpoint for " + t.name + " in '" + dir + "'"); }
            out << std::setprecision(17) << "nodes " << t.name << " " << t.num_force_nodes << " " << t.num_vel_nodes << "\n";
            for (Real v : t.force) { out << v << "\n"; }
            for (Real v : t.node_vel) { out << v << "\n"; }
        }
        for (std::size_t i = 0; i < m_fllc.size(); ++i) {
            if (m_fllc[i].empty()) { continue; }
            std::ofstream out(dir + "/" + turbs[i].name + "_fllc.dat", std::ios::trunc);
            if (!out) { Abort("cannot write the lifting-line correction checkpoint for " + turbs[i].name + " in '" + dir + "'"); }
            for (const auto& f : m_fllc[i]) { f->write_state(out); }
        }
    }
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
#ifdef ERF_USE_OPENFAST
    m_fllc_state_dir = dir;
#endif
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
    // the node loads and velocities ERF checkpointed (a checkpoint from before they were written has none)
    {
        const auto& turbs = m_driver->turbines();
        for (std::size_t i = 0; i < turbs.size(); ++i) {
            const std::string fname = dir + "/" + turbs[i].name + "_nodes.dat";
            if (!FileExists(fname)) { continue; }
            Vector<char> chars;
            ParallelDescriptor::ReadAndBcastFile(fname, chars);
            std::istringstream in(std::string(chars.dataPtr(), chars.size()));
            std::string key, name;
            int nf = 0, nv = 0;
            if (!(in >> key >> name >> nf >> nv) || key != "nodes" || name != turbs[i].name ||
                nf != turbs[i].num_force_nodes || nv != turbs[i].num_vel_nodes) {
                Abort("the node-load checkpoint '" + fname + "' does not match turbine " + turbs[i].name);
            }
            std::vector<Real> force(3 * nf), node_vel(3 * nv);
            for (Real& v : force) { if (!(in >> v)) { Abort("the node-load checkpoint '" + fname + "' is truncated"); } }
            for (Real& v : node_vel) { if (!(in >> v)) { Abort("the node-load checkpoint '" + fname + "' is truncated"); } }
            m_driver->set_restored_loads(static_cast<int>(i), force, node_vel);
        }
    }
#endif
    Print() << "Moving bodies: restarted after step " << m_step << "\n";
    m_restored = true;
}

void
MovingBodies::advance (int lev, double time, double dt,
                       const MultiFab& cons,
                       const MultiFab& U, const MultiFab& V, const MultiFab& W,
                       const MultiFab* z_phys_nd, const MultiFab* detJ_cc, const Geometry& geom)
{
    if (lev != anchor_level()) { return; }
    // the OpenFAST substep count was fixed at init from erf.fixed_dt
    if (std::abs(dt - m_dt) > 1.0e-10 * m_dt) {
        Abort("erf.moving_bodies: the time step on level " + std::to_string(m_anchor) + " changed from " + std::to_string(m_dt) + " to " +
              std::to_string(dt) + "; OpenFAST needs the fixed step it was initialised with");
    }
    if (m_restarting && !m_restored) {
        Abort("erf.moving_bodies: restarting, but the checkpoint holds no moving-bodies state (was it written by a run with bodies?)");
    }
    ++m_step;
    const bool first = (m_step == 1);
#ifdef ERF_USE_OPENFAST
    if (!m_audited_setup) { audit_turbines_setup(U.boxArray(), geom); }
    if (first && !m_driver->solved0()) {
        // first step: OpenFAST's first solution sees the initial flow at its nodes
        supply_velocities(time, U, V, W, z_phys_nd, geom);
        m_driver->solution0();
        m_driver->write_diagnostics(time);
        write_total_load(time, true);
    }
#endif
    supply_velocities(time, U, V, W, z_phys_nd, geom);
#ifdef ERF_USE_OPENFAST
    if (!m_audited_flow) { audit_turbines_flow(cons, z_phys_nd, geom); }
    m_driver->step();
    if (first || m_step % m_in.diagnostics_int == 0) {
        m_driver->write_diagnostics(time + dt);
        write_fllc_diagnostics(time, first);
        write_total_load(time + dt, false);
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
#ifndef ERF_USE_OPENFAST
    amrex::ignore_unused(cons);
#endif
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
MovingBodies::build_wake_lines (const BoxArray& level_grids, const Geometry& geom)
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
        std::string outside;
        if (!erf_actuator::points_covered_by(level_grids, geom, pos, geom.CellSize(0), outside)) {
            Abort("erf.moving_bodies.wake: the sampling point " + outside + " of " + wl->name() + " is not covered by the grids of level " +
                  std::to_string(m_anchor) + "; shorten lines_xD or half_width, or enlarge the refinement region behind the rotor");
        }
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
    if (!m_wakes_built) { build_wake_lines(U.boxArray(), geom); }
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
    for (const std::string& m : m_turbine_mode) { if (m != "none") { return true; } }
    return false;
}

void
MovingBodies::write_total_load (double time, bool first)
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    std::array<Real,3> load{{0.0, 0.0, 0.0}};
    Real power = 0.0;
#ifdef ERF_USE_OPENFAST
    {
        const auto& turbs = m_driver->turbines();
        for (std::size_t i = 0; i < turbs.size(); ++i) {
            if (m_turbine_mode[i] == "none") { continue; }   // no force in the flow
            const auto& t = turbs[i];
            const auto f = m_driver->thrust(t);
            const auto ft = m_driver->tower_force(t);
            for (int d = 0; d < 3; ++d) { load[d] += f[d] + ft[d] + t.nacelle_force[d]; }
            power += m_driver->torque(t) * t.rotor_speed;
        }
    }
#endif
    for (const auto& d : m_disks) {
        const auto n = d->normal();
        for (int c = 0; c < 3; ++c) { load[c] += d->thrust() * n[c]; }
    }
    std::ofstream out;
    const bool truncate = first && !m_restored && !m_total_written;
    if (erf_actuator::open_log(out, m_in.diagnostics_dir + "/total_load.csv", truncate)) {
        out << "time,load_x,load_y,load_z,power\n";
    }
    out << std::setprecision(10) << time << "," << load[0] << "," << load[1] << "," << load[2] << "," << power << "\n";
    m_total_written = true;
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
    // each turbine's loads as actuator-disk rings (mode = adm) or as an actuator line on its
    // blade nodes (mode = alm); none puts no force in the flow
    {
        const auto& turbs = m_driver->turbines();
        const Real dx_min = std::min(geom.CellSize(0), std::min(geom.CellSize(1), geom.CellSize(2)));
        for (std::size_t i = 0; i < turbs.size(); ++i) {
            if (m_turbine_mode[i] == "none") { continue; }
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
                        << "), blade force nodes within " << deg << " degrees of the rotor plane, ";
                if (m_turbine_mode[i] == "adm") {
                    Print() << m_turbine_points_t[i] << " points per ring\n";
                } else {
                    Print() << "actuator line of " << turbs[i].num_blades * turbs[i].num_force_pts_blade
                            << " blade points, tip radius " << erf_actuator::tip_radius(turbs[i]) << " m\n";
                }
            }
            if (m_turbine_mode[i] == "adm") {
                erf_actuator::adm_rings(turbs[i], m_turbine_points_t[i], pos, force);
            } else {
                // the line's force must not jump over cells between two steps: the tip, the
                // fastest point, may sweep at most alm_max_tip_cells cells per ERF step
                const Real cells = erf_actuator::tip_cells_per_step(turbs[i], m_dt, dx_min);
                if (cells > m_in.alm_max_tip_cells) {
                    const Real r_tip = erf_actuator::tip_radius(turbs[i]);
                    const Real dt_max = m_in.alm_max_tip_cells * dx_min / std::max(std::abs(turbs[i].rotor_speed) * r_tip, Real(1.0e-30));
                    Abort("erf.moving_bodies." + turbs[i].name + ": the blade tip sweeps " + std::to_string(cells) +
                          " cells per step (rotor speed " + std::to_string(turbs[i].rotor_speed) + " rad/s, tip radius " +
                          std::to_string(r_tip) + " m, dt " + std::to_string(m_dt) + " s, smallest cell " + std::to_string(dx_min) +
                          " m), more than erf.moving_bodies.alm_max_tip_cells = " + std::to_string(m_in.alm_max_tip_cells) +
                          "; use erf.fixed_dt <= " + std::to_string(dt_max) + " or finer cells for the actuator line");
                }
                erf_actuator::alm_points(turbs[i], pos, force);
            }
            // the tower's loads (OpenFAST's tower force nodes, if the model has them and the
            // user asked for them) and the nacelle drag point at the hub, in either mode
            erf_actuator::tower_points(turbs[i], pos, force);
            if (m_turbine_in[i].nacelle_cd > 0.0) {
                for (int d = 0; d < 3; ++d) {
                    pos.push_back(turbs[i].hub_pos[d]);
                    force.push_back(-turbs[i].nacelle_force[d]);   // on the fluid
                }
            }
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
    if (!m_disk_coverage_checked) {
        m_disk_coverage_checked = true;
        for (const auto& d : m_disks) {
            std::string outside;
            if (!erf_actuator::points_covered_by(U.boxArray(), geom, d->disk_points(), Real(3.0) * eps + geom.CellSize(0), outside) ||
                !erf_actuator::points_covered_by(U.boxArray(), geom, d->sample_points(), geom.CellSize(0), outside)) {
                Abort("erf.moving_bodies." + d->name() + ": point " + outside + " (with the kernel reach) is not covered by the grids of level " +
                      std::to_string(m_anchor) + "; enlarge the refinement region around the disk");
            }
        }
    }
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

std::vector<Real>
MovingBodies::sampling_positions (int i) const
{
#ifdef ERF_USE_OPENFAST
    const erf_openfast::TurbineState& t = m_driver->turbines()[i];
    if (m_turbine_in[i].sampling == "upstream") {
        return erf_actuator::upstream_sampling_positions(t, m_turbine_in[i].sample_diameters_upstream);
    }
    return t.vel_pos;
#else
    amrex::ignore_unused(i);
    return {};
#endif
}

void
MovingBodies::supply_velocities (double time, const MultiFab& U, const MultiFab& V, const MultiFab& W,
                                 const MultiFab* z_phys_nd, const Geometry& geom)
{
    amrex::ignore_unused(time);
    // one flattened list: the OpenFAST turbines' velocity nodes, then each disk's upstream
    // sampling points and disk points
    m_points.clear();
    int nturb = 0;
#ifdef ERF_USE_OPENFAST
    for (int i = 0; i < static_cast<int>(m_driver->turbines().size()); ++i) { m_points.add_body(sampling_positions(i)); ++nturb; }
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
    if (!m_fllc_built) { build_fllc(geom); }
    for (int i = 0; i < nturb; ++i) {
        if (m_fllc[i].empty()) {
            m_driver->set_node_velocities(i, m_points.body_velocities(i));
        } else {
            std::vector<Real> uvw = m_points.body_velocities(i);
            apply_fllc(i, time, uvw);
            m_driver->set_node_velocities(i, uvw);
        }
        // the nacelle drag from the hub node's velocity (the first velocity node), corrected
        // for the drag point's own kernel; recorded on the structure for the diagnostics
        const MovingBodyInputs& b = m_turbine_in[i];
        std::array<Real,3> f_nac{{0.0, 0.0, 0.0}};
        if (m_turbine_mode[i] != "none" && b.nacelle_cd > 0.0) {
            const std::vector<Real>& uvw = m_points.body_velocities(i);
            if (uvw.size() >= 3) {
                const std::array<Real,3> u_hub{{uvw[0], uvw[1], uvw[2]}};
                const Real eps = m_epsilon_dx * geom.CellSize(0);
                const auto u_free = erf_actuator::nacelle_corrected_velocity(u_hub, b.nacelle_cd, b.nacelle_area, eps);
                const auto f_fluid = erf_actuator::nacelle_drag_force(u_free, b.air_density, b.nacelle_cd, b.nacelle_area);
                for (int d = 0; d < 3; ++d) { f_nac[d] = -f_fluid[d]; }
            }
        }
        m_driver->set_nacelle_force(i, f_nac);
    }
#endif
    for (std::size_t k = 0; k < m_disks.size(); ++k) {
        const int b = nturb + 2 * static_cast<int>(k);
        m_disks[k]->update(m_points.body_velocities(b), m_points.body_velocities(b + 1));
    }
}

#ifdef ERF_USE_OPENFAST
void
MovingBodies::audit_turbines_setup (const BoxArray& level_grids, const Geometry& geom)
{
    m_audited_setup = true;
    const auto& turbs = m_driver->turbines();
    if (turbs.empty()) { return; }
    // every node of every turbine, with the kernel's reach, must lie on the anchor level's grids:
    // on a refined level those are not the whole domain, and the sampler and spreader read only
    // the level's own cells
    {
        const Real reach = Real(3.0) * m_epsilon_dx * geom.CellSize(0) + geom.CellSize(0);
        for (int i = 0; i < static_cast<int>(turbs.size()); ++i) {
            const auto& t = turbs[i];
            std::string outside;
            if (!erf_actuator::points_covered_by(level_grids, geom, t.force_pos, reach, outside) ||
                !erf_actuator::points_covered_by(level_grids, geom, sampling_positions(i), geom.CellSize(0), outside)) {
                Abort("erf.moving_bodies." + t.name + ": node " + outside + " (with the kernel reach " + std::to_string(reach) +
                      " m) is not covered by the grids of level " + std::to_string(m_anchor) +
                      "; enlarge the refinement region around the rotor, or set erf.moving_bodies.anchor_level to a level that covers it");
            }
        }
    }
    std::array<Real,3> plo, phi, dx;
    std::array<int,3> per;
    for (int d = 0; d < 3; ++d) {
        plo[d] = static_cast<Real>(geom.ProbLo(d)); phi[d] = static_cast<Real>(geom.ProbHi(d));
        dx[d] = static_cast<Real>(geom.CellSize(d)); per[d] = geom.isPeriodic(d) ? 1 : 0;
    }
    const Real eps = m_epsilon_dx * geom.CellSize(0);
    bool any_fatal = false;
    for (std::size_t i = 0; i < turbs.size(); ++i) {
        const MovingBodyInputs& b = m_turbine_in[i];
        std::vector<erf_openfast::AuditFinding> f = erf_openfast::audit_model(b, CONST_GRAV);
        const auto fg = erf_openfast::audit_geometry(turbs[i], b, eps, plo, phi, dx, per);
        f.insert(f.end(), fg.begin(), fg.end());
        erf_openfast::report_audit(turbs[i].name + " (model and geometry)", f, any_fatal);
    }
    const auto overlap = erf_openfast::audit_overlap(turbs);
    if (!overlap.empty()) { erf_openfast::report_audit("the farm", overlap, any_fatal); }
    if (any_fatal) {
        Abort("erf.moving_bodies: the OpenFAST input audit found errors (listed above); fix them or set mode = none where no loads are wanted");
    }
}

void
MovingBodies::audit_turbines_flow (const MultiFab& cons, const MultiFab* z_phys_nd, const Geometry& geom)
{
    m_audited_flow = true;
    const auto& turbs = m_driver->turbines();
    if (turbs.empty()) { return; }
    // ERF's density at every hub
    std::vector<Real> hubs, rho_hub;
    for (const auto& t : turbs) { for (int d = 0; d < 3; ++d) { hubs.push_back(t.hub_pos[d]); } }
    erf_actuator::sample_cell_scalar(cons, Rho_comp, z_phys_nd, geom, hubs, rho_hub);
    bool any_fatal = false;
    for (std::size_t i = 0; i < turbs.size(); ++i) {
        const MovingBodyInputs& b = m_turbine_in[i];
        std::vector<erf_openfast::AuditFinding> f;
        Real rho_model = 0.0;
        if (erf_openfast::openfast_air_density(b.fst_file, rho_model)) {
            f = erf_openfast::audit_density(turbs[i].name, rho_model, rho_hub[i], m_in.density_tolerance);
            if (f.empty()) {
                f.push_back({false, "ERF's density at the hub is " + std::to_string(rho_hub[i]) + " kg/m^3, the model's " +
                                    std::to_string(rho_model) + " (within " + std::to_string(100.0 * m_in.density_tolerance) + " %)"});
            }
        } else {
            f.push_back({false, "ERF's density at the hub is " + std::to_string(rho_hub[i]) + " kg/m^3; the model gives none to compare with"});
        }
        const std::vector<Real>& uvw = m_points.body_velocities(static_cast<int>(i));
        if (uvw.size() >= 3) {
            const auto ff = erf_openfast::audit_facing(turbs[i], {{uvw[0], uvw[1], uvw[2]}});
            f.insert(f.end(), ff.begin(), ff.end());
        }
        erf_openfast::report_audit(turbs[i].name + " (density and wind)", f, any_fatal);
    }
    if (any_fatal) {
        Abort("erf.moving_bodies: the OpenFAST input audit found errors (listed above); fix them or set mode = none where no loads are wanted");
    }
}

// span coordinate of a node: its distance from the hub
namespace {
Real span_of (const std::vector<Real>& pos, int nd, const std::array<Real,3>& hub)
{
    Real r2 = 0.0;
    for (int d = 0; d < 3; ++d) { const Real dd = pos[3*nd+d] - hub[d]; r2 += dd * dd; }
    return std::sqrt(r2);
}
} // namespace

void
MovingBodies::build_fllc (const Geometry& geom)
{
    const auto& turbs = m_driver->turbines();
    m_fllc.clear();
    m_fllc.resize(turbs.size());
    const Real eps = m_epsilon_dx * geom.CellSize(0);
    for (std::size_t i = 0; i < turbs.size(); ++i) {
        const MovingBodyInputs& b = m_turbine_in[i];
        if (!b.fllc) { continue; }
        const auto& t = turbs[i];
        const int nfb = t.num_force_pts_blade;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nfb >= 2 && t.num_force_nodes >= 1 + t.num_blades * nfb,
                                         "erf.moving_bodies." + t.name + ".fllc needs at least two force points per blade");
        for (int bl = 0; bl < t.num_blades; ++bl) {
            std::vector<Real> r(nfb), chord(nfb);
            for (int k = 0; k < nfb; ++k) {
                const int nd = 1 + bl * nfb + k;
                r[k] = span_of(t.force_pos, nd, t.hub_pos);
                chord[k] = t.chord[nd];
                if (!(chord[k] > 0.0)) {
                    Abort("erf.moving_bodies." + t.name + ".fllc: OpenFAST reports no chord at blade " + std::to_string(bl) +
                          " force node " + std::to_string(k) + "; the correction needs the chord (forceNodesChord)");
                }
            }
            m_fllc[i].push_back(std::make_unique<erf_actuator::FLLC>(t.name + "_blade" + std::to_string(bl), r, chord, eps,
                                                                      b.fllc_eps_chord, b.fllc_eps_dr, b.fllc_relax));
        }
        if (!m_fllc_state_dir.empty()) {
            const std::string fname = m_fllc_state_dir + "/" + t.name + "_fllc.dat";
            if (FileExists(fname)) {
                Vector<char> chars;
                ParallelDescriptor::ReadAndBcastFile(fname, chars);
                std::istringstream in(std::string(chars.dataPtr(), chars.size()));
                for (auto& f : m_fllc[i]) {
                    if (!f->read_state(in)) { Abort("the lifting-line correction checkpoint '" + fname + "' lacks " + f->name()); }
                }
                Print() << "  " << t.name << ": lifting-line correction restored from " << fname << "\n";
            }
        }
        Print() << "erf.moving_bodies." << t.name << ": lifting-line correction on " << t.num_blades << " blades, kernel "
                << eps << " m, optimal kernel " << b.fllc_eps_chord * t.chord[1 + nfb - 1] << " m at the tip, "
                << m_fllc[i].front()->num_fine_points() << " fine points per blade\n";
    }
    m_fllc_built = true;
}

void
MovingBodies::apply_fllc (int i, double time, std::vector<Real>& uvw)
{
    const auto& t = m_driver->turbines()[i];
    const MovingBodyInputs& b = m_turbine_in[i];
    const int nfb = t.num_force_pts_blade;
    const int nbe = t.num_blade_elem;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(uvw.size()) >= 3 * (1 + t.num_blades * nbe),
                                     "apply_fllc: fewer velocities than blade velocity nodes for " + t.name);
    for (int bl = 0; bl < t.num_blades; ++bl) {
        erf_actuator::FLLC& f = *m_fllc[i][bl];
        // the sampled velocities at this blade's velocity nodes, and their span coordinates
        std::vector<Real> rv(nbe), uv(3 * nbe);
        for (int m = 0; m < nbe; ++m) {
            const int nd = 1 + bl * nbe + m;
            rv[m] = span_of(t.vel_pos, nd, t.hub_pos);
            for (int d = 0; d < 3; ++d) { uv[3*m+d] = uvw[3*nd+d]; }
        }
        if (time >= b.fllc_start_time) {
            // relative velocity and force on the fluid per unit density at the force nodes
            const std::vector<Real>& rf = f.span();
            std::vector<Real> uf, vel_rel(3 * nfb), force(3 * nfb);
            erf_actuator::interpolate_along(rv, uv, rf, uf);
            for (int k = 0; k < nfb; ++k) {
                const int nd = 1 + bl * nfb + k;
                for (int d = 0; d < 3; ++d) {
                    vel_rel[3*k+d] = uf[3*k+d] - t.force_vel[3*nd+d];
                    force[3*k+d] = -t.force[3*nd+d] / b.air_density;
                }
            }
            f.update(force, vel_rel);
        }
        // the relaxed correction, back onto the velocity nodes
        std::vector<Real> du_v;
        erf_actuator::interpolate_along(f.span(), f.correction(), rv, du_v);
        for (int m = 0; m < nbe; ++m) {
            const int nd = 1 + bl * nbe + m;
            for (int d = 0; d < 3; ++d) { uvw[3*nd+d] += du_v[3*m+d]; }
        }
    }
}

void
MovingBodies::write_fllc_diagnostics (double time, bool first)
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    const auto& turbs = m_driver->turbines();
    for (std::size_t i = 0; i < m_fllc.size(); ++i) {
        if (m_fllc[i].empty()) { continue; }
        Real du_max = 0.0, sum2 = 0.0;
        for (const auto& f : m_fllc[i]) {
            du_max = std::max(du_max, f->max_correction());
            sum2 += f->rms_correction() * f->rms_correction();
        }
        std::ofstream out;
        const bool truncate = first && !m_restored && !m_fllc_written;
        if (erf_actuator::open_log(out, turbs[i].output_root + "_fllc.csv", truncate)) {
            out << "time,du_max,du_rms\n";
        }
        out << std::setprecision(10) << time << "," << du_max << "," << std::sqrt(sum2 / static_cast<Real>(m_fllc[i].size())) << "\n";
    }
    m_fllc_written = true;
}
#endif
