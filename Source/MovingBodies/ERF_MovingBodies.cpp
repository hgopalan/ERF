#include "ERF_MovingBodies.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
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

namespace {

// The inputs that place a body and decide how it forces the flow, as key/value pairs for the
// checkpoint: a restart that changes one of them would continue another run's state
std::vector<std::pair<std::string,std::string>>
body_record (const MovingBodyInputs& b)
{
    auto num = [] (double v) { std::ostringstream o; o << std::setprecision(17) << v; return o.str(); };
    std::vector<std::pair<std::string,std::string>> r{
        {"type", b.type}, {"base_x", num(b.base_pos[0])}, {"base_y", num(b.base_pos[1])}, {"base_z", num(b.base_pos[2])},
        {"epsilon", num(b.epsilon)}, {"air_density", num(b.air_density)}};
    // the keys the body's options use: one it ignores may change freely
    if (b.type != "openfast_turbine" || b.mode == "adm") { r.push_back({"num_points_t", std::to_string(b.num_points_t)}); }
    if (b.type == "openfast_turbine") {
        r.insert(r.end(), {{"fst_file", b.fst_file}, {"mode", b.mode}, {"sampling", b.sampling},
                           {"num_force_points_blade", std::to_string(b.num_force_points_blade)},
                           {"num_force_points_tower", std::to_string(b.num_force_points_tower)},
                           {"fllc", b.fllc ? "1" : "0"}});
        if (b.mode != "none") { r.insert(r.end(), {{"nacelle_cd", num(b.nacelle_cd)}, {"nacelle_area", num(b.nacelle_area)}}); }
        if (b.sampling == "upstream") { r.push_back({"sample_diameters_upstream", num(b.sample_diameters_upstream)}); }
        if (b.sampling == "disk_corrected") {
            r.insert(r.end(), {{"correction_relax", num(b.correction_relax)}, {"correction_time", num(b.correction_time)}});
        }
        if (b.fllc) {
            r.insert(r.end(), {{"fllc_relax", num(b.fllc_relax)}, {"fllc_start_time", num(b.fllc_start_time)},
                               {"fllc_eps_chord", num(b.fllc_eps_chord)}, {"fllc_eps_dr", num(b.fllc_eps_dr)}});
        }
    } else {
        r.insert(r.end(), {{"rotor_radius", num(b.rotor_radius)}, {"hub_height", num(b.hub_height)}, {"ct", num(b.ct)},
                           {"yaw", num(b.yaw_deg)}, {"num_points_r", std::to_string(b.num_points_r)},
                           {"sample_diameters_upstream", num(b.sample_diameters_upstream)}});
    }
    return r;
}

// two record values agree: equal strings, or numbers equal to single precision (a checkpoint
// written by a double build may be continued by a single one)
bool same_record_value (const std::string& a, const std::string& b)
{
    if (a == b) { return true; }
    char* ea = nullptr;
    char* eb = nullptr;
    const double x = std::strtod(a.c_str(), &ea);
    const double y = std::strtod(b.c_str(), &eb);
    if (ea == a.c_str() || eb == b.c_str() || *ea != '\0' || *eb != '\0') { return false; }
    return std::abs(x - y) <= 1.0e-6 * std::max(1.0, std::max(std::abs(x), std::abs(y)));
}

// the input key a record entry comes from
std::string record_key (const std::string& k)
{
    return (k == "base_x" || k == "base_y" || k == "base_z") ? std::string("base_pos") : k;
}

} // namespace

std::unique_ptr<MovingBodies>
MovingBodies::create (const SolverChoice& sc, int max_level, const Vector<double>& fixed_dt_levels, bool restarting,
                      double stop_elapsed)
{
    MovingBodiesInputs in = MovingBodiesInputs::read();
    if (!in.active) { return nullptr; }

    bool all_anelastic = true;
    for (int lev = 0; lev <= max_level && lev < static_cast<int>(sc.anelastic.size()); ++lev) {
        all_anelastic = all_anelastic && (sc.anelastic[lev] != 0);
    }

    // the step and stop time live in ERF's own inputs; kept in double, as ERF keeps them, so a
    // single-precision build hands OpenFAST and the step check the step ERF actually takes
    ParmParse pp("erf");
    double fixed_dt = -1.0;
    pp.query("fixed_dt", fixed_dt);
    double stop_time = -1.0;
    bool steps_end_first = false;   // max_step ends the run before the stop time
    {
        ParmParse pp_root;
        int max_step = -1;
        pp_root.query("max_step", max_step);
        if (pp_root.contains("stop_datetime")) {
            // ERF then stops at that date and ignores stop_time: take the seconds it runs for
            stop_time = stop_elapsed;
            // without start_datetime (or a start date from the initial-condition file) the start is the
            // epoch, decades before the stop date
            if (stop_time > 1.0e8) {
                Abort("erf.moving_bodies: stop_datetime lies " + std::to_string(stop_time) + " s after the start (more than "
                      "three years); set start_datetime as well");
            }
        } else {
            pp_root.query("stop_time", stop_time);
            if (stop_time <= 0.0 && max_step > 0 && fixed_dt > 0.0) {
                stop_time = max_step * fixed_dt;
            }
        }
        // ERF stops at whichever comes first; at max_step no step is cut short
        steps_end_first = (max_step > 0 && fixed_dt > 0.0 && stop_time > 0.0 &&
                           max_step * fixed_dt < stop_time - 1.0e-6 * fixed_dt);
    }
    const double input_fixed_dt = fixed_dt;

    // OpenFAST is not floating-point-exception clean; runs with prescribed-Ct disks only may trap
    bool any_turbine = false;
    for (const MovingBodyInputs& b : in.bodies) { any_turbine = any_turbine || (b.type == "openfast_turbine"); }
    bool fpe_traps = false;
    if (any_turbine) {
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
    fixed_dt = fixed_dt_levels[anchor];
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

    auto bodies = std::unique_ptr<MovingBodies>(new MovingBodies(std::move(in), anchor, fixed_dt, stop_time, restarting));
    bodies->m_input_fixed_dt = input_fixed_dt;
    bodies->m_steps_end_first = steps_end_first;
    return bodies;
}

MovingBodies::MovingBodies (MovingBodiesInputs in, int anchor, double dt, double t_max, bool restarting)
    : m_in(std::move(in)), m_anchor(anchor), m_dt(dt), m_t_max(t_max), m_restarting(restarting)
{
    Print() << "Moving bodies: " << m_in.bodies.size() << " body(ies) on level " << m_anchor << ", ERF dt there " << m_dt
            << ", run stop time " << t_max << " s, velocities "
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
        m_disk_stats.emplace_back(d->name(), d->output_root(),
                                  std::vector<std::string>{"u_inf", "u_disk", "thrust", "power"});
    }
    for (const MovingBodyInputs& b : turbines) {
        m_turbine_stats.emplace_back(b.name, b.output_root,
                                     std::vector<std::string>{"thrust_shaft", "thrust_x", "torque", "power", "rotor_speed",
                                                              "hub_u", "hub_v", "hub_w", "blade_mean_u"});
    }
#ifdef ERF_USE_OPENFAST
    m_corr_uinf.assign(turbines.size(), 0.0);
    m_corr_clamp_warned.assign(turbines.size(), false);
    m_corr_last.assign(turbines.size(), erf_actuator::DiskCorrection{});
    // the models are set up now, so their inputs are checked at start-up. OpenFAST itself is
    // initialised by set_ground(), once the bases sit on the terrain, or restored by
    // read_checkpoint(); its first solution waits for the first step, when the flow exists to be
    // sampled
    m_driver = std::make_unique<erf_openfast::OpenFASTDriver>(turbines);
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
        out << std::setprecision(17);
        out << "step = " << m_step << "\n";
        // when the bodies started (ERF time), OpenFAST's stop time on its clock that starts there, and
        // the run's anchor level and its step: a restart must continue the same coupling
        out << "t0 = " << m_t0 << "\n" << "openfast_tmax = " << m_openfast_tmax << "\n"
            << "anchor = " << m_anchor << "\n" << "dt = " << m_dt << "\n";
#ifdef ERF_USE_OPENFAST
        out << "solved0 = " << (m_driver->solved0() ? 1 : 0) << "\n";
#endif
        for (const auto& b : m_in.bodies) {
            out << "body " << b.name;
            for (const auto& kv : body_record(b)) { out << " " << kv.first << "=" << kv.second; }
            out << "\n";
        }
#ifdef ERF_USE_OPENFAST
        const auto& turbs = m_driver->turbines();
        for (std::size_t i = 0; i < turbs.size(); ++i) {
            out << "time_index " << turbs[i].name << " = " << turbs[i].time_index << "\n";
            if (m_turbine_in[i].sampling == "disk_corrected") {
                out << "corr_uinf " << turbs[i].name << " = "
                    << std::setprecision(std::numeric_limits<Real>::max_digits10) << m_corr_uinf[i] << "\n";
            }
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
MovingBodies::set_ground (const MultiFab* z_phys_nd, const Geometry& geom, double time, int finest_level)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_ground_set, "MovingBodies::set_ground: called twice");
    if (m_anchor > finest_level) {
        Abort("erf.moving_bodies: the bodies live on level " + std::to_string(m_anchor) + " (erf.moving_bodies.anchor_level, "
              "or amr.max_level when it is not given), but the run starts with levels 0 to " + std::to_string(finest_level) +
              " only; refine the region around the bodies from the start, or set erf.moving_bodies.anchor_level to an existing level");
    }
    m_ground_set = true;
    m_fitted = (z_phys_nd != nullptr);
    std::vector<Real> pos;
    for (const auto& m : m_motion) { const auto b = m->position(0.0); pos.insert(pos.end(), {b[0], b[1], b[2]}); }
    erf_actuator::terrain_heights(z_phys_nd, geom, pos, m_ground);
    const Real floor_z = static_cast<Real>(geom.ProbLo(2));
    std::size_t idisk = 0, iturb = 0;
    for (std::size_t i = 0; i < m_in.bodies.size(); ++i) {
        const Real dz = m_ground[i] - floor_z;   // the terrain above the domain floor: zero on a flat mesh
        m_motion[i]->shift_base({{0.0, 0.0, dz}});
        if (m_in.bodies[i].type == "ct_disk") {
            m_disks[idisk++]->shift_z(dz);
        } else {
            m_turbine_in[iturb].base_pos[2] += dz;
#ifdef ERF_USE_OPENFAST
            m_driver->shift_base_z(static_cast<int>(iturb), dz);
#endif
            ++iturb;
        }
        if (dz != Real(0.0)) {
            Print() << "erf.moving_bodies." << m_in.bodies[i].name << ": base raised by " << dz
                    << " m onto the terrain surface at z = " << m_ground[i] << " m\n";
        }
    }
    // a fresh start: the bodies start now (a restart learns when they started from read_checkpoint())
    if (!m_restarting) { start_bodies(time); }
#ifdef ERF_USE_OPENFAST
    // a fresh start: OpenFAST sees the bases on the terrain; a restart is restored by read_checkpoint()
    if (!m_restarting && !m_driver->initialized()) {
        m_driver->init(m_dt, m_openfast_tmax);
        for (const auto& t : m_driver->turbines()) {
            Print() << "  " << t.name << ": OpenFAST dt " << t.dt_fast << ", " << t.num_substeps
                    << " substeps per ERF step, " << t.num_blades << " blades, "
                    << t.num_vel_nodes << " velocity nodes, " << t.num_force_nodes << " force nodes, on rank "
                    << t.owner_rank << "\n";
        }
    }
#endif
    if (ParallelDescriptor::IOProcessor()) {
        UtilCreateDirectory(m_in.diagnostics_dir, 0755);
        std::ofstream out(m_in.diagnostics_dir + "/ground.csv", std::ios::trunc);
        out << "body,x,y,ground,base_z\n" << std::setprecision(10);
        for (std::size_t i = 0; i < m_in.bodies.size(); ++i) {
            const auto b = m_motion[i]->position(0.0);
            out << m_in.bodies[i].name << "," << b[0] << "," << b[1] << "," << m_ground[i] << "," << b[2] << "\n";
        }
    }
}

void
MovingBodies::start_bodies (double time)
{
    m_t0 = time;
    m_openfast_tmax = m_t_max - m_t0;
    if (!(m_openfast_tmax > 0.0)) {
        Abort("erf.moving_bodies: the bodies start at t = " + std::to_string(m_t0) + " s, at or after the stop time " +
              std::to_string(m_t_max) + " s");
    }
    check_whole_steps(m_t0);
}

void
MovingBodies::check_whole_steps (double time) const
{
    // ERF cuts the last level-0 step to reach the stop time, and the finer levels' steps with it, while
    // the bodies step with a fixed step: the span must be a whole number of level-0 steps. ERF's clock
    // drifts by far less than 1e-4 of a step over a run, so that is the tolerance, in steps
    if (m_steps_end_first) { return; }   // the run ends at max_step, on a whole step
    const double n = (m_t_max - time) / m_input_fixed_dt;
    if (std::abs(n - std::round(n)) > 1.0e-4) {
        Abort("erf.moving_bodies: the run from t = " + std::to_string(time) + " s to the stop time " + std::to_string(m_t_max) +
              " s is " + std::to_string(n) + " steps of erf.fixed_dt = " + std::to_string(m_input_fixed_dt) + " s, not a whole "
              "number; ERF would cut the last step, and the bodies step with a fixed step, so choose stop_time (or max_step) "
              "to end on one");
    }
}

void
MovingBodies::read_checkpoint (const std::string& chkdir, double time)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_restarting && !m_restored,
                                     "MovingBodies::read_checkpoint: only once, and only on a restart");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_ground_set,
                                     "MovingBodies::read_checkpoint: set_ground() must come first (the mesh must be read)");
    const std::string dir = chkdir + "/moving_bodies";
    if (!erf_actuator::file_exists_everywhere(dir + "/state")) {
        // a checkpoint written without bodies (a precursor, say): the bodies start afresh here
        Print() << "Moving bodies: the checkpoint " << chkdir << " holds no moving-bodies state; the bodies start now, at t = "
                << time << " s\n";
        start_bodies(time);
#ifdef ERF_USE_OPENFAST
        // OpenFAST's clock starts at 0 here: it runs for the rest of the run only
        m_driver->init(m_dt, m_openfast_tmax);
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
    std::map<std::string,Real> corr_uinf;
    std::map<std::string,std::map<std::string,std::string>> bodies;   // body records of the run being continued
    double t0 = -1.0, openfast_tmax = -1.0, dt_chk = -1.0;
    int anchor_chk = -1, solved0 = 1;
    bool have_t0 = false;
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
            } else if (key == "corr_uinf") {
                Real u = 0.0;
                if (!(ls >> name >> eq >> u) || eq != "=") { Abort("malformed corr_uinf line in '" + dir + "/state'"); }
                corr_uinf[name] = u;
            } else if (key == "t0") {
                if (!(ls >> eq >> t0) || eq != "=") { Abort("malformed t0 line in '" + dir + "/state'"); }
                have_t0 = true;
            } else if (key == "openfast_tmax") {
                if (!(ls >> eq >> openfast_tmax) || eq != "=") { Abort("malformed openfast_tmax line in '" + dir + "/state'"); }
            } else if (key == "anchor") {
                if (!(ls >> eq >> anchor_chk) || eq != "=") { Abort("malformed anchor line in '" + dir + "/state'"); }
            } else if (key == "dt") {
                if (!(ls >> eq >> dt_chk) || eq != "=") { Abort("malformed dt line in '" + dir + "/state'"); }
            } else if (key == "solved0") {
                if (!(ls >> eq >> solved0) || eq != "=") { Abort("malformed solved0 line in '" + dir + "/state'"); }
            } else if (key == "body") {
                if (!(ls >> name)) { Abort("malformed body line in '" + dir + "/state'"); }
                std::string kv;
                auto& rec = bodies[name];
                while (ls >> kv) {
                    const auto e = kv.find('=');
                    if (e == std::string::npos) { Abort("malformed body line in '" + dir + "/state'"); }
                    rec[kv.substr(0, e)] = kv.substr(e + 1);
                }
            }
        }
        if (!have_step) { Abort("no step count in the moving-bodies checkpoint '" + dir + "/state'"); }
    }
    // the run being continued must be this one: same anchor level and step, same bodies with the same
    // placing and forcing inputs (checkpoints from before these were recorded are taken on trust)
    if (anchor_chk >= 0 && anchor_chk != m_anchor) {
        Abort("erf.moving_bodies: the checkpoint's bodies live on level " + std::to_string(anchor_chk) + ", this run's on level " +
              std::to_string(m_anchor) + " (erf.moving_bodies.anchor_level / amr.max_level); restart with the same anchor level");
    }
    if (dt_chk > 0.0 && std::abs(dt_chk - m_dt) > 1.0e-9 * m_dt) {
        Abort("erf.moving_bodies: the checkpoint's bodies stepped " + std::to_string(dt_chk) + " s on their level, this run " +
              std::to_string(m_dt) + " s; OpenFAST continues with the step it was started with, so keep erf.fixed_dt");
    }
    if (!bodies.empty()) {
        for (const auto& b : m_in.bodies) {
            const auto it = bodies.find(b.name);
            if (it == bodies.end()) {
                Abort("erf.moving_bodies." + b.name + " is not in the checkpoint '" + dir + "'; a restart continues the same bodies");
            }
            for (const auto& kv : body_record(b)) {
                const auto f = it->second.find(kv.first);
                if (f != it->second.end() && !same_record_value(f->second, kv.second)) {
                    Abort("erf.moving_bodies." + b.name + "." + record_key(kv.first) + " changed on restart (" + f->second + " in the "
                          "checkpoint, " + kv.second + " now); a restart continues the same bodies with the same inputs");
                }
            }
        }
        for (const auto& kv : bodies) {
            bool found = false;
            for (const auto& b : m_in.bodies) { found = found || (b.name == kv.first); }
            if (!found) {
                Abort("erf.moving_bodies: the checkpoint '" + dir + "' holds body " + kv.first + ", which erf.moving_bodies.bodies "
                      "no longer names; a restart continues the same bodies");
            }
        }
    } else {
        Print() << "Moving bodies: the checkpoint '" << dir << "' records no body inputs (written before they were); "
                << "taking the inputs as unchanged\n";
    }
    // when the bodies started, and how long OpenFAST runs (older checkpoints: from the step count)
    m_t0 = have_t0 ? t0 : time - static_cast<double>(m_step) * m_dt;
    m_openfast_tmax = (openfast_tmax > 0.0) ? openfast_tmax : m_t_max - m_t0;
    bool have_turbines = false;
#ifdef ERF_USE_OPENFAST
    have_turbines = !m_driver->turbines().empty();
#endif
    // without turbines nothing keeps the old stop time: the run may stop later
    if (!have_turbines) { m_openfast_tmax = std::max(m_openfast_tmax, m_t_max - m_t0); }
    check_whole_steps(time);
    if (m_t_max - m_t0 > m_openfast_tmax * (1.0 + 1.0e-9) + 1.0e-3 * m_dt) {
        Abort("erf.moving_bodies: this run stops at t = " + std::to_string(m_t_max) + " s, but the turbines of the run being continued "
              "were started for OpenFAST to stop at t = " + std::to_string(m_t0 + m_openfast_tmax) + " s, and OpenFAST keeps that "
              "stop time across its checkpoints; stop at or before it, or start the turbines again");
    }
    // the logs lose what the run before wrote after this checkpoint: rows stamped at the end of their
    // step keep the checkpoint time (unless OpenFAST's first solution, the row at t0, is still to come),
    // rows stamped at its start drop it. Rows lie whole steps apart and are printed to 10 digits, so
    // the tolerance is a quarter step
    {
        const double tol = 0.25 * m_dt;
        const bool keep_end_rows = (solved0 != 0);
        std::vector<std::string> end_stamped{m_in.diagnostics_dir + "/total_load.csv"};
        std::vector<std::string> start_stamped{m_in.diagnostics_dir + "/momentum_source.csv"};
        for (const auto& b : m_in.bodies) {
            if (b.type == "openfast_turbine") {
                end_stamped.push_back(b.output_root + "_erf.csv");
                end_stamped.push_back(b.output_root + "_flow.csv");
                start_stamped.push_back(b.output_root + "_correction.csv");
                start_stamped.push_back(b.output_root + "_fllc.csv");
            } else {
                start_stamped.push_back(b.output_root + "_disk.csv");
            }
            start_stamped.push_back(b.output_root + "_wake.csv");
        }
        // the I/O rank trims every body's files (the output directory is shared); a turbine's owner rank
        // appends to its own files only after this barrier
        for (const auto& f : end_stamped) { erf_actuator::trim_log_after(f, time, keep_end_rows, tol); }
        for (const auto& f : start_stamped) { erf_actuator::trim_log_after(f, time, false, tol); }
        ParallelDescriptor::Barrier();
    }
    m_resumed = true;
#ifdef ERF_USE_OPENFAST
    m_driver->restart(dir + "/", m_dt, solved0 != 0);
    for (std::size_t i = 0; i < m_driver->turbines().size(); ++i) {
        if (m_turbine_in[i].sampling != "disk_corrected") { continue; }
        const auto it = corr_uinf.find(m_driver->turbines()[i].name);
        if (it == corr_uinf.end()) {
            Abort("the moving-bodies checkpoint '" + dir + "/state' holds no corr_uinf line for turbine " +
                  m_driver->turbines()[i].name + " (sampling = disk_corrected)");
        }
        m_corr_uinf[i] = it->second;
    }
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
            if (!erf_actuator::file_exists_everywhere(fname)) { continue; }
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
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_ground_set && (!m_restarting || m_restored),
                                     "MovingBodies::advance: ERF calls set_ground() (and read_checkpoint() on a restart) first");
    // The OpenFAST substep count was fixed at init from the anchor level's step. ERF's clock is a
    // running sum of its steps and the last step of a run is cut to reach stop_time, so that step
    // differs from the fixed one by roundoff (1e-6 relative after 3600 s of 0.0125 s steps); the
    // whole-number-of-steps check at the start rules out a genuinely short last step.
    if (std::abs(dt - m_dt) > 1.0e-4 * m_dt) {
        Abort("erf.moving_bodies: the time step on level " + std::to_string(m_anchor) + " changed from " + std::to_string(m_dt) + " to " +
              std::to_string(dt) + " s; OpenFAST needs the fixed step it was initialised with");
    }
    // ERF and the bodies must agree on the time: a step the bodies missed (their level absent for a
    // while) would leave OpenFAST behind ERF's clock
    {
        const double expect = m_t0 + static_cast<double>(m_step) * m_dt;
        if (std::abs(time - expect) > 1.0e-3 * m_dt + 1.0e-9 * std::abs(time)) {
            Abort("erf.moving_bodies: ERF is at t = " + std::to_string(time) + " s but the bodies, started at t = " +
                  std::to_string(m_t0) + " s and advanced " + std::to_string(m_step) + " steps of " + std::to_string(m_dt) +
                  " s, are at t = " + std::to_string(expect) + " s (did the anchor level " + std::to_string(m_anchor) +
                  " miss steps?)");
        }
    }
#ifdef ERF_USE_OPENFAST
    // OpenFAST stops stepping, without an error, past the stop time it was initialised with
    if (!m_driver->turbines().empty() && time + dt - m_t0 > m_openfast_tmax * (1.0 + 1.0e-9) + 1.0e-3 * m_dt) {
        Abort("erf.moving_bodies: this step ends at t = " + std::to_string(time + dt) + " s, past the OpenFAST stop time (" +
              std::to_string(m_t0 + m_openfast_tmax) + " s) the turbines were initialised with; a run continued past "
              "the stop time of the run that started the turbines must start the turbines again");
    }
#endif
    ++m_step;
    const bool first = (m_step == 1);
    // the bodies' points on the anchor level's grids: at the first step, and again after a regrid
    if (first || m_audited_grids.empty() || U.boxArray() != m_audited_grids) {
        check_coverage(U.boxArray(), z_phys_nd, geom);
    }
    // a disk's thrust is 1/2 air_density Ct U^2 A: the density must be ERF's at the disk, as for the turbines
    // (checked at the first step of every run, a restarted one included)
    if (!m_disk_density_checked && !m_disks.empty()) {
        m_disk_density_checked = true;
        std::vector<Real> centres, rho;
        for (const auto& d : m_disks) { const auto c = d->center(); centres.insert(centres.end(), {c[0], c[1], c[2]}); }
        erf_actuator::sample_cell_scalar(cons, Rho_comp, z_phys_nd, geom, centres, rho);
        for (std::size_t k = 0; k < m_disks.size(); ++k) {
            Real rho_in = 0.0;
            for (const auto& b : m_in.bodies) { if (b.name == m_disks[k]->name()) { rho_in = b.air_density; } }
            const Real rel = std::abs(rho_in - rho[k]) / rho[k];
            if (rel > m_in.density_tolerance) {
                Abort("erf.moving_bodies." + m_disks[k]->name() + ".air_density = " + std::to_string(rho_in) + " kg/m^3, but ERF's "
                      "density at the disk centre is " + std::to_string(rho[k]) + " (" + std::to_string(100.0 * rel) + " % off, more "
                      "than erf.moving_bodies.density_tolerance); the thrust put into the flow would not be the one asked for");
            }
            Print() << "  " << m_disks[k]->name() << ": ERF's density at the disk centre is " << rho[k] << " kg/m^3, air_density "
                    << rho_in << "\n";
        }
    }
    // after a restart, the wake lines at once: a checkpoint written before their first sample would
    // otherwise lose the running averages they took over
    if (m_resumed && !m_wakes_built && m_in.wake.active() && !m_in.has_prescribed_velocity) {
        build_wake_lines(U.boxArray(), z_phys_nd, geom);
    }
#ifdef ERF_USE_OPENFAST
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_driver->initialized(), "MovingBodies::advance: OpenFAST is not initialised (set_ground() initialises it on a fresh start)");
    if (!m_audited_setup) { audit_turbines_setup(z_phys_nd, geom); }
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
        write_correction_diagnostics(time, first);
        write_total_load(time + dt, false);
    }
#endif
    // the bodies' forces come from the velocities just sampled (the disks) or from the
    // OpenFAST step just taken (the turbines), so the source is fixed over the step ERF is
    // about to take
    if (any_forcing()) {
        spread_sources(U, z_phys_nd, detJ_cc, geom);
        if (first) { for (auto& d : m_disks) { d->open_diagnostics(true); } }
        if (m_resumed && !m_logs_reopened) {
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
MovingBodies::build_wake_lines (const BoxArray& level_grids, const MultiFab* z_phys_nd, const Geometry& geom)
{
    const auto& w = m_in.wake;
    // each vertical line stops at the terrain under its own centre (downstream, the ground may rise or fall)
    auto add = [&](const std::string& name, const std::string& root, const std::array<Real,3>& hub,
                   const std::array<Real,3>& axis, Real diameter) {
        Real ah[2] = {axis[0], axis[1]};
        const Real la = std::sqrt(ah[0] * ah[0] + ah[1] * ah[1]);
        if (la > Real(1.0e-6)) { ah[0] /= la; ah[1] /= la; }
        std::vector<Real> centres;
        for (const Real xD : w.lines_xD) {
            centres.insert(centres.end(), {hub[0] + xD * diameter * ah[0], hub[1] + xD * diameter * ah[1], hub[2]});
        }
        for (std::size_t c = 0; c < centres.size() / 3; ++c) {
            for (int d = 0; d < 2; ++d) {
                if (!geom.isPeriodic(d) && (centres[3*c+d] < geom.ProbLo(d) || centres[3*c+d] > geom.ProbHi(d))) {
                    Abort("erf.moving_bodies.wake: the line " + std::to_string(w.lines_xD[c]) + " D behind " + name +
                          " lies outside the domain; shorten lines_xD");
                }
            }
        }
        std::vector<Real> ground;
        erf_actuator::terrain_heights(z_phys_nd, geom, centres, ground);
        m_wakes.push_back(std::make_unique<erf_actuator::WakeLines>(name, root, hub, axis, diameter,
                                                                    w.lines_xD, w.half_width, w.num_points, ground));
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
        add(d->name(), d->output_root(), d->center(), d->normal(), Real(2.0) * d->radius());
    }
    // every sampling point must lie in the domain (periodic directions wrap)
    const auto plo = geom.ProbLoArray();
    const auto phi = geom.ProbHiArray();
    for (const auto& wl : m_wakes) {
        const auto& pos = wl->positions();
        std::string outside;
        // with z_phys_nd a point's cell is not its height over dz: the sampler finds it in the column itself
        if (!erf_actuator::points_covered_by(level_grids, geom, pos, geom.CellSize(0), outside,
                                             (z_phys_nd == nullptr) ? erf_actuator::CoverZ::Reach : erf_actuator::CoverZ::Footprint)) {
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
    if (!m_wakes_built) { build_wake_lines(U.boxArray(), z_phys_nd, geom); }
    std::vector<Real> vel;
    for (auto& wl : m_wakes) {
        erf_actuator::sample_velocity(U, V, W, z_phys_nd, geom, wl->positions(), vel);
        if (time >= m_in.avg_start) { wl->accumulate(vel); }
        wl->write_instantaneous(time, vel, !m_resumed && !m_wake_written);
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
    const bool truncate = first && !m_resumed && !m_total_written;
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
                          "; use erf.fixed_dt <= " + std::to_string(dt_max * m_input_fixed_dt / m_dt) +
                          " (a step of " + std::to_string(dt_max) + " s on level " + std::to_string(m_anchor) +
                          ") or finer cells for the actuator line");
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

#ifdef ERF_USE_OPENFAST
std::vector<Real>
MovingBodies::sampling_positions (int i) const
{
    const erf_openfast::TurbineState& t = m_driver->turbines()[i];
    if (m_turbine_in[i].sampling == "upstream") {
        return erf_actuator::upstream_sampling_positions(t, m_turbine_in[i].sample_diameters_upstream);
    }
    return t.vel_pos;
}
#endif

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
        {
            std::vector<Real> uvw = m_points.body_velocities(i);
            if (m_turbine_in[i].sampling == "disk_corrected") { apply_disk_correction(i, uvw, geom); }
            if (!m_fllc[i].empty()) { apply_fllc(i, time, uvw); }
            m_driver->set_node_velocities(i, uvw);
        }
        // the nacelle drag from the velocity at the hub (the first velocity node), corrected for the
        // drag point's own kernel; recorded on the structure for the diagnostics. With sampling = upstream
        // the nodes were sampled ahead of the rotor, so the hub is sampled where the drag point is.
        const MovingBodyInputs& b = m_turbine_in[i];
        std::array<Real,3> f_nac{{0.0, 0.0, 0.0}};
        if (m_turbine_mode[i] != "none" && b.nacelle_cd > 0.0) {
            std::vector<Real> uvw = m_points.body_velocities(i);
            if (b.sampling == "upstream") {
                const auto& t = m_driver->turbines()[i];
                const std::vector<Real> hub{t.vel_pos[0], t.vel_pos[1], t.vel_pos[2]};
                if (m_in.has_prescribed_velocity) {
                    uvw = {m_in.prescribed_velocity[0], m_in.prescribed_velocity[1], m_in.prescribed_velocity[2]};
                } else {
                    erf_actuator::sample_velocity(U, V, W, z_phys_nd, geom, hub, uvw);
                }
            }
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

void
MovingBodies::check_levels (int finest_level, const Vector<BoxArray>& grids, const Vector<Geometry>& geoms) const
{
    if (m_anchor > finest_level) {
        Abort("erf.moving_bodies: the bodies live on level " + std::to_string(m_anchor) + ", but the run has levels 0 to " +
              std::to_string(finest_level) + " only now (did a regrid remove it?); the bodies would not be stepped");
    }
    if (finest_level == m_anchor || !m_ground_set) { return; }
    // what the bodies force, with the kernel's reach, in metres (centre x y z, half-size): each rotor's or
    // disk's extent about its hub or centre, and every force point of a turbine (blades, tower, hub)
    const Geometry& ga = geoms[m_anchor];
    const Real reach = Real(3.0) * m_epsilon_dx * ga.CellSize(0) + ga.CellSize(0);
    std::vector<std::pair<std::string, std::array<Real,4>>> extents;
#ifdef ERF_USE_OPENFAST
    for (const auto& t : m_driver->turbines()) {
        extents.push_back({t.name, {t.hub_pos[0], t.hub_pos[1], t.hub_pos[2], erf_actuator::tip_radius(t) + reach}});
        for (std::size_t p = 0; p + 2 < t.force_pos.size(); p += 3) {
            extents.push_back({t.name, {t.force_pos[p], t.force_pos[p+1], t.force_pos[p+2], reach}});
        }
    }
#endif
    for (const auto& d : m_disks) {
        const auto c = d->center();
        extents.push_back({d->name(), {c[0], c[1], c[2], d->radius() + reach}});
    }
    for (int lev = m_anchor + 1; lev <= finest_level; ++lev) {
        const Geometry& g = geoms[lev];
        const Box& dom = g.Domain();
        for (const auto& e : extents) {
            IntVect lo, hi;
            for (int d = 0; d < 3; ++d) {
                lo[d] = static_cast<int>(std::floor((e.second[d] - e.second[3] - g.ProbLo(d)) * g.InvCellSize(d)));
                hi[d] = static_cast<int>(std::floor((e.second[d] + e.second[3] - g.ProbLo(d)) * g.InvCellSize(d)));
            }
            // a terrain-following or stretched mesh: the whole column (a cell's height is not k dz)
            if (m_fitted) { lo[2] = dom.smallEnd(2); hi[2] = dom.bigEnd(2); }
            // the kernel wraps across a periodic boundary: test the images of the part outside the domain too
            std::vector<Box> boxes{Box(lo, hi)};
            for (int d = 0; d < 2; ++d) {
                if (!g.isPeriodic(d)) { continue; }
                const int n = dom.length(d);
                const std::size_t nb = boxes.size();
                for (std::size_t q = 0; q < nb; ++q) {
                    if (boxes[q].smallEnd(d) < dom.smallEnd(d)) { boxes.push_back(Box(boxes[q]).shift(d, n)); }
                    if (boxes[q].bigEnd(d) > dom.bigEnd(d)) { boxes.push_back(Box(boxes[q]).shift(d, -n)); }
                }
            }
            for (Box b : boxes) {
                b &= dom;
                if (b.ok() && grids[lev].intersects(b)) {
                    Abort("erf.moving_bodies." + e.first + ": level " + std::to_string(lev) + ", finer than the bodies' level " +
                          std::to_string(m_anchor) + ", covers part of it with the kernel's reach; the force lives on level " +
                          std::to_string(m_anchor) + " only, and ERF's average-down would overwrite it there. Put the bodies on the "
                          "finest level over them (erf.moving_bodies.anchor_level) or keep finer levels away from them");
                }
            }
        }
    }
}

void
MovingBodies::check_coverage (const BoxArray& level_grids, const MultiFab* z_phys_nd, const Geometry& geom)
{
    m_audited_grids = level_grids;
    // on a fitted or stretched mesh a point's cell is not its height over dz: the kernel must be on the
    // grids over every height its reach may span (mesh_z_bounds), a sampled point anywhere in its column
    erf_actuator::ZBounds zb;
    if (z_phys_nd != nullptr) { zb = erf_actuator::mesh_z_bounds(*z_phys_nd, geom); }
    const bool fitted = (z_phys_nd != nullptr);
    // a level with a box that does not reach the ground has no height bounds: the whole column, as
    // the spreading takes it
    const erf_actuator::ZBounds* zbp = (fitted && zb.all_ground) ? &zb : nullptr;
    const Real eps = m_epsilon_dx * geom.CellSize(0);
    const Real reach = Real(3.0) * eps + geom.CellSize(0);
#ifdef ERF_USE_OPENFAST
    // every node of every turbine, with the kernel's reach, must lie on the anchor level's grids, and the
    // whole swept disk with it (the rings of a disk, the blades of a line as they turn and the rotor as
    // it yaws, while the grids stay as they are): on a refined level those are not the whole domain, and
    // the sampler and spreader read only the level's own cells
    const auto& turbs = m_driver->turbines();
    for (int i = 0; i < static_cast<int>(turbs.size()); ++i) {
        const auto& t = turbs[i];
        // the swept disk: rings at half and the full tip radius in the plane normal to the shaft
        std::vector<Real> disk;
        {
            const auto& a = t.hub_axis;
            // a unit vector normal to the axis, then the third by the cross product
            std::array<Real,3> e1 = (std::abs(a[2]) < Real(0.9)) ? std::array<Real,3>{{-a[1], a[0], Real(0.0)}}
                                                                 : std::array<Real,3>{{Real(0.0), -a[2], a[1]}};
            const Real l1 = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
            for (auto& v : e1) { v /= l1; }
            const std::array<Real,3> e2{{a[1]*e1[2] - a[2]*e1[1], a[2]*e1[0] - a[0]*e1[2], a[0]*e1[1] - a[1]*e1[0]}};
            const Real R = erf_actuator::tip_radius(t);
            const int n = 72;
            for (const Real f : {Real(0.5), Real(1.0)}) {
                for (int k = 0; k < n; ++k) {
                    const Real th = Real(2.0 * 3.14159265358979323846) * Real(k) / Real(n);
                    for (int d = 0; d < 3; ++d) {
                        disk.push_back(t.hub_pos[d] + f * R * (std::cos(th) * e1[d] + std::sin(th) * e2[d]));
                    }
                }
            }
        }
        std::string outside;
        if (!erf_actuator::points_covered_by(level_grids, geom, t.force_pos, reach, outside,
                                             fitted ? erf_actuator::CoverZ::Column : erf_actuator::CoverZ::Reach, zbp) ||
            !erf_actuator::points_covered_by(level_grids, geom, disk, reach, outside,
                                             fitted ? erf_actuator::CoverZ::Column : erf_actuator::CoverZ::Reach, zbp) ||
            !erf_actuator::points_covered_by(level_grids, geom, sampling_positions(i), geom.CellSize(0), outside,
                                             fitted ? erf_actuator::CoverZ::Footprint : erf_actuator::CoverZ::Reach)) {
            Abort("erf.moving_bodies." + t.name + ": node " + outside + " (with the kernel reach " + std::to_string(reach) +
                  " m) is not covered by the grids of level " + std::to_string(m_anchor) +
                  "; enlarge the refinement region around the rotor (the force lives on the anchor level only, so that "
                  "level must hold the whole kernel)");
        }
    }
#endif
    for (const auto& d : m_disks) {
        std::string outside;
        if (!erf_actuator::points_covered_by(level_grids, geom, d->disk_points(), reach, outside,
                                             fitted ? erf_actuator::CoverZ::Column : erf_actuator::CoverZ::Reach, zbp) ||
            !erf_actuator::points_covered_by(level_grids, geom, d->sample_points(), geom.CellSize(0), outside,
                                             fitted ? erf_actuator::CoverZ::Footprint : erf_actuator::CoverZ::Reach)) {
            Abort("erf.moving_bodies." + d->name() + ": point " + outside + " (with the kernel reach) is not covered by the grids of level " +
                  std::to_string(m_anchor) + "; enlarge the refinement region around the disk");
        }
    }
    for (const auto& wl : m_wakes) {
        std::string outside;
        if (!erf_actuator::points_covered_by(level_grids, geom, wl->positions(), geom.CellSize(0), outside,
                                             fitted ? erf_actuator::CoverZ::Footprint : erf_actuator::CoverZ::Reach)) {
            Abort("erf.moving_bodies.wake: the sampling point " + outside + " of " + wl->name() + " is not covered by the grids of level " +
                  std::to_string(m_anchor) + " after a regrid; enlarge the refinement region behind the rotor");
        }
    }
}

#ifdef ERF_USE_OPENFAST
void
MovingBodies::apply_disk_correction (int i, std::vector<Real>& uvw, const Geometry& geom)
{
    const erf_openfast::TurbineState& t = m_driver->turbines()[i];
    const MovingBodyInputs& b = m_turbine_in[i];
    const Real u_disk = erf_actuator::disk_axial_velocity(t, uvw);
    // the rotor thrust of the previous step along the shaft (force on the structure)
    const auto f = m_driver->thrust(t);
    Real thrust = 0.0;
    for (int d = 0; d < 3; ++d) { thrust += f[d] * t.hub_axis[d]; }
    const Real eps = m_epsilon_dx * geom.CellSize(0);
    erf_actuator::DiskCorrection c =
        erf_actuator::filtered_disk_correction(u_disk, thrust, m_corr_uinf[i], b.air_density, erf_actuator::tip_radius(t), eps);
    // The update U = M u_disk / (1 - a), with Ct from the previous step's thrust, is unstable on its own
    // above Ct ~ 0.75 + 0.11 Delta/R (its gain passes -1, RANS or LES alike), and the flow's answer to a
    // change of thrust comes only after a delay, which can make even a gain-cancelled update ring. It is
    // relaxed towards the previous free stream, by default with the factor that cancels its linear error
    // in one step, but no faster than the time scale correction_time (by default the rotor radius over
    // the free stream); the fixed point, and so the converged answer, is the same.
    if (c.ct > Real(0.0) && m_corr_uinf[i] > Real(0.1)) {
        const Real tau = (b.correction_time < Real(0.0)) ? erf_actuator::tip_radius(t) / m_corr_uinf[i] : b.correction_time;
        c.relax = (b.correction_relax > Real(0.0)) ? b.correction_relax
                                                   : erf_actuator::disk_correction_relax(c.gain, static_cast<Real>(m_dt), tau);
        c.u_inf = (Real(1.0) - c.relax) * m_corr_uinf[i] + c.relax * c.u_inf;
        c.factor = c.u_inf / u_disk;
    }
    if (c.clamped && !m_corr_clamp_warned[i]) {
        m_corr_clamp_warned[i] = true;
        Print() << "Warning: erf.moving_bodies." << t.name << ": the thrust coefficient on the corrected free stream reached the "
                << "0.96 clamp (the rotor's own Ct, at fixed speed below rated, waked or at high tip-speed ratio, or an update "
                << "that has not settled); the free stream recovered there is biased low\n";
    }
    // the hub and the blade velocity nodes see the free stream; the tower nodes keep the resolved flow
    const int nb = t.num_blades * t.num_blade_elem;
    for (int nd = 0; nd <= nb && 3*nd+2 < static_cast<int>(uvw.size()); ++nd) {
        for (int d = 0; d < 3; ++d) { uvw[3*nd+d] *= c.factor; }
    }
    m_corr_uinf[i] = c.u_inf;
    m_corr_last[i] = c;
}

void
MovingBodies::write_correction_diagnostics (double time, bool first)
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    const auto& turbs = m_driver->turbines();
    bool any = false;
    for (std::size_t i = 0; i < turbs.size(); ++i) {
        if (m_turbine_in[i].sampling != "disk_corrected") { continue; }
        any = true;
        const erf_actuator::DiskCorrection& c = m_corr_last[i];
        std::ofstream out;
        const bool truncate = first && !m_resumed && !m_corr_written;
        if (erf_actuator::open_log(out, turbs[i].output_root + "_correction.csv", truncate)) {
            out << "time,u_disk,ct,a,ct_prime,M,u_inf,factor,gain,relax\n";
        }
        out << std::setprecision(10) << time << "," << c.u_disk << "," << c.ct << "," << c.a << "," << c.ct_prime << ","
            << c.M << "," << c.u_inf << "," << c.factor << "," << c.gain << "," << c.relax << "\n";
    }
    if (any) { m_corr_written = true; }
}

std::vector<Real>
MovingBodies::hub_ground (const MultiFab* z_phys_nd, const Geometry& geom) const
{
    std::vector<Real> pos, ground;
    for (const auto& t : m_driver->turbines()) { pos.insert(pos.end(), {t.hub_pos[0], t.hub_pos[1], t.hub_pos[2]}); }
    erf_actuator::terrain_heights(z_phys_nd, geom, pos, ground);
    return ground;
}

void
MovingBodies::audit_turbines_setup (const MultiFab* z_phys_nd, const Geometry& geom)
{
    m_audited_setup = true;
    const auto& turbs = m_driver->turbines();
    if (turbs.empty()) { return; }
    std::array<Real,3> plo, phi, dx;
    std::array<int,3> per;
    for (int d = 0; d < 3; ++d) {
        plo[d] = static_cast<Real>(geom.ProbLo(d)); phi[d] = static_cast<Real>(geom.ProbHi(d));
        dx[d] = static_cast<Real>(geom.CellSize(d)); per[d] = geom.isPeriodic(d) ? 1 : 0;
    }
    const Real eps = m_epsilon_dx * geom.CellSize(0);
    const std::vector<Real> ground = hub_ground(z_phys_nd, geom);
    bool any_fatal = false;
    for (std::size_t i = 0; i < turbs.size(); ++i) {
        const MovingBodyInputs& b = m_turbine_in[i];
        if (b.sampling == "disk_corrected") {
            // the filtered-disk factor was derived for filter widths up to about 1.25 rotor radii
            const Real ratio = erf_actuator::filter_width_from_eps(eps) / erf_actuator::tip_radius(turbs[i]);
            Print() << "erf.moving_bodies." << turbs[i].name << ": disk_corrected sampling, filter width sqrt(6) eps = "
                    << erf_actuator::filter_width_from_eps(eps) << " m, " << ratio << " rotor radii\n";
            if (ratio > Real(1.25) && ParallelDescriptor::IOProcessor()) {
                Warning("erf.moving_bodies." + turbs[i].name + ": the filter width is " + std::to_string(ratio) +
                        " rotor radii; the filtered-disk correction was derived for widths up to about 1.25 radii");
            }
        }
        std::vector<erf_openfast::AuditFinding> f = erf_openfast::audit_model(b, CONST_GRAV);
        const auto fg = erf_openfast::audit_geometry(turbs[i], b, eps, plo, phi, dx, per, ground[i]);
        f.insert(f.end(), fg.begin(), fg.end());
        erf_openfast::report_audit(turbs[i].name + " (model and geometry)", f, any_fatal);
    }
    const auto overlap = erf_openfast::audit_overlap(turbs);
    if (!overlap.empty()) { erf_openfast::report_audit("the farm", overlap, any_fatal); }
    if (any_fatal) {
        Abort("erf.moving_bodies: the OpenFAST input audit found errors (listed above); fix them (mode = none clears only "
              "the CompAero finding: a turbine whose loads are not wanted in the flow)");
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
        Abort("erf.moving_bodies: the OpenFAST input audit found errors (listed above); fix them (mode = none clears only "
              "the CompAero finding: a turbine whose loads are not wanted in the flow)");
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
            if (erf_actuator::file_exists_everywhere(fname)) {
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
        const bool truncate = first && !m_resumed && !m_fllc_written;
        if (erf_actuator::open_log(out, turbs[i].output_root + "_fllc.csv", truncate)) {
            out << "time,du_max,du_rms\n";
        }
        out << std::setprecision(10) << time << "," << du_max << "," << std::sqrt(sum2 / static_cast<Real>(m_fllc[i].size())) << "\n";
    }
    m_fllc_written = true;
}
#endif
