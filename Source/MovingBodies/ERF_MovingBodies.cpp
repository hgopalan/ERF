#include "ERF_MovingBodies.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
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

// the input key a record entry comes from
std::string record_key (const std::string& k)
{
    return (k == "base_x" || k == "base_y" || k == "base_z") ? std::string("base_pos") : k;
}

// Collective: a table of k values for each of n bodies, filled by each body's owner (owns(i)) and summed
// over the ranks (onto the I/O rank only with io_only); one row from one rank each, zeros from the others,
// so the sum is every row exactly
std::vector<Real> gather_rows (int n, int k, bool io_only, const std::function<bool(int)>& owns,
                               const std::function<void(int, Real*)>& fill)
{
    std::vector<Real> table(static_cast<std::size_t>(n) * k, Real(0.0));
    for (int i = 0; i < n; ++i) {
        if (owns(i)) { fill(i, &table[static_cast<std::size_t>(i) * k]); }
    }
    if (!table.empty()) {
        if (io_only) {
            ParallelDescriptor::ReduceRealSum(table.data(), static_cast<int>(table.size()), ParallelDescriptor::IOProcessorNumber());
        } else {
            ParallelDescriptor::ReduceRealSum(table.data(), static_cast<int>(table.size()));
        }
    }
    return table;
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
        std::string stop_datetime;
        const bool has_stop_datetime = pp_root.query("stop_datetime", stop_datetime);
        if (has_stop_datetime && !stop_datetime.empty()) {
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
    auto bodies = std::unique_ptr<MovingBodies>(new MovingBodies(std::move(in), anchor, fixed_dt, stop_time, restarting));
    bodies->m_input_fixed_dt = input_fixed_dt;
    bodies->m_steps_end_first = steps_end_first;
    return bodies;
}

MovingBodies::MovingBodies (MovingBodiesInputs in, int anchor, double dt, double t_max, bool restarting)
    : m_in(std::move(in)), m_anchor(anchor), m_dt(dt), m_t_max(t_max), m_restarting(restarting)
{
    BL_PROFILE("MovingBodies::MovingBodies()");
    m_timers.enable(m_in.timing_int > 0);
    erf_actuator::ScopedPhase timed(m_timers, erf_actuator::PhaseTimers::Startup);
    if (m_in.max_open_logs > 0) {
        m_logs = erf_actuator::LogStreams(static_cast<std::size_t>(m_in.max_open_logs));
        Print() << "Moving bodies: at most " << m_logs.max_open() << " diagnostics file(s) open per rank\n";
        const double limit = erf_actuator::open_file_limit();
        if (limit > 0.0 && static_cast<double>(m_in.max_open_logs) >= limit) {
            Print() << "WARNING: erf.moving_bodies.max_open_logs = " << m_in.max_open_logs << " is not below this process's limit of "
                    << limit << " open files; a rank that opens that many logs stops with 'cannot open'\n";
        }
    }
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
            m_disk_epsilon.push_back(b.epsilon);
            Print() << "  " << b.name << ": uniform-Ct disk, R " << b.rotor_radius << " m, hub "
                    << b.hub_height << " m, Ct " << b.ct << ", " << m_disks.back()->num_points()
                    << " points, spreading " << b.epsilon << " dx\n";
        } else {
            turbines.push_back(b);
            m_turbine_mode.push_back(b.mode);
            m_turbine_points_t.push_back(b.num_points_t);
            m_turbine_in.push_back(b);
            const KernelWidths w = b.kernel_widths(1.0);
            if (b.mode != "none") {
                Print() << "  " << b.name << ": rotor points spread " << w.rotor << " dx\n";
            }
            if (b.mode != "none" && b.num_force_points_tower > 0) {
                Print() << "  " << b.name << ": tower loads forced on " << b.num_force_points_tower << " points, spread "
                        << w.tower << " dx\n";
            }
            if (b.mode != "none" && b.nacelle_cd > 0.0) {
                Print() << "  " << b.name << ": nacelle drag point, cd " << b.nacelle_cd << ", area " << b.nacelle_area
                        << " m^2, density " << b.air_density << ", spread " << w.nacelle << " dx\n";
            }
            if (b.fllc) {
                Print() << "  " << b.name << ": filtered lifting-line correction, optimal kernel " << b.fllc_eps_chord
                        << " chords, relaxation " << b.fllc_relax << ", from t = " << b.fllc_start_time << " s\n";
            }
        }
    }
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
    // what start-up takes from each distinct model file (a farm shares a few, or gives every turbine its
    // own): each read on one rank, file p on rank p mod nprocs, and handed to every rank in one gather
    {
        std::vector<std::string> paths;   // in the order the turbines first name them
        std::set<std::string> seen;
        for (const MovingBodyInputs& b : turbines) {
            if (seen.insert(b.fst_file).second) { paths.push_back(b.fst_file); }
        }
        const std::vector<erf_openfast::ModelFacts> facts = erf_openfast::gather_model_facts(paths);
        for (std::size_t p = 0; p < paths.size(); ++p) { m_model_facts.emplace(paths[p], facts[p]); }
    }
    m_driver = std::make_unique<erf_openfast::OpenFASTDriver>(turbines, &m_model_facts);
    for (std::size_t i = 0; i < turbines.size(); ++i) {
        // the tower's wake now reaches the blades through the flow: AeroDyn's own tower-shadow
        // correction would count it twice
        if (turbines[i].mode != "none" && turbines[i].num_force_points_tower > 0) {
            const std::string& w = m_model_facts.at(turbines[i].fst_file).tower_shadow;
            if (!w.empty()) { Print() << "WARNING: erf.moving_bodies." << turbines[i].name << ": " << w << "\n"; }
        }
    }
    // on a restart the turbines are restored by read_checkpoint(), from ERF's checkpoint read
#else
    amrex::ignore_unused(t_max);
#endif
}

void
MovingBodies::write_checkpoint (const std::string& chkdir)
{
    BL_PROFILE("MovingBodies::write_checkpoint()");
    erf_actuator::ScopedPhase timed(m_timers, erf_actuator::PhaseTimers::Checkpoint);
    const std::string dir = chkdir + "/moving_bodies";
    // the statistics and wake averages written only every stats_int steps are brought up to date
    if (m_in.stats_int > 0) {
        rewrite_summaries();
        flush_logs();
    }
    // every row written so far reaches its file: a restart from this checkpoint trims the logs to it
    m_logs.flush();
#ifdef ERF_USE_OPENFAST
    // every turbine's OpenFAST time index and corrected free stream, which its owner alone knows, at the I/O rank
    const int nturb = static_cast<int>(m_driver->turbines().size());
    std::vector<Long> time_index(nturb, 0);
    for (int i = 0; i < nturb; ++i) { if (m_driver->is_owner(i)) { time_index[i] = m_driver->turbines()[i].time_index; } }
    if (nturb > 0) { ParallelDescriptor::ReduceLongSum(time_index.data(), nturb, ParallelDescriptor::IOProcessorNumber()); }
    const std::vector<Real> corr_uinf = turbine_table(1, true, [this] (int i, Real* row) { row[0] = m_corr_uinf[i]; });
#endif
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
        // the logs' layout: a restart trims the per_rank files of every rank this run wrote with
        out << "diagnostics_format = " << m_in.diagnostics_format << "\n"
            << "diagnostics_nranks = " << ParallelDescriptor::NProcs() << "\n";
#ifdef ERF_USE_OPENFAST
        out << "solved0 = " << (m_driver->solved0() ? 1 : 0) << "\n";
#endif
        for (const auto& b : m_in.bodies) {
            out << "body " << b.name;
            for (const auto& kv : MovingBodiesInputs::restart_record(b)) { out << " " << kv.first << "=" << kv.second; }
            out << "\n";
        }
#ifdef ERF_USE_OPENFAST
        const auto& turbs = m_driver->turbines();
        for (std::size_t i = 0; i < turbs.size(); ++i) {
            out << "time_index " << turbs[i].name << " = " << time_index[i] << "\n";
            if (m_turbine_in[i].sampling == "disk_corrected") {
                out << "corr_uinf " << turbs[i].name << " = "
                    << std::setprecision(std::numeric_limits<Real>::max_digits10) << corr_uinf[i] << "\n";
            }
        }
#endif
    }
    ParallelDescriptor::Barrier();
    for (const auto& w : m_wakes) { w->write_state(dir); }
    for (const auto& st : m_turbine_stats) { st.write_state(dir); }
    for (const auto& st : m_disk_stats) { st.write_state(dir); }
#ifdef ERF_USE_OPENFAST
    {
        const auto& turbs = m_driver->turbines();
        // every turbine's node loads and velocities at the checkpoint, written by its owner: what
        // depends on the loads of the step before a restart (the lifting-line correction) continues from them
        for (int i = 0; i < nturb; ++i) {
            if (!m_driver->is_owner(i)) { continue; }
            const auto& t = turbs[i];
            std::ofstream out(dir + "/" + t.name + "_nodes.dat", std::ios::trunc);
            if (!out) { Abort("cannot write the node-load checkpoint for " + t.name + " in '" + dir + "'"); }
            out << std::setprecision(17) << "nodes " << t.name << " " << t.num_force_nodes << " " << t.num_vel_nodes << "\n";
            for (Real v : t.force) { out << v << "\n"; }
            for (Real v : t.node_vel) { out << v << "\n"; }
        }
        for (std::size_t i = 0; i < m_fllc.size(); ++i) {
            if (m_fllc[i].empty()) { continue; }   // off, or not owned here
            std::ofstream out(dir + "/" + turbs[i].name + "_fllc.dat", std::ios::trunc);
            if (!out) { Abort("cannot write the lifting-line correction checkpoint for " + turbs[i].name + " in '" + dir + "'"); }
            for (const auto& f : m_fllc[i]) { f->write_state(out); }
        }
    }
    m_driver->create_checkpoint(dir + "/");
#endif
}

void
MovingBodies::set_ground (const MultiFab* z_phys_nd, const Geometry& geom, const BoxArray& grids, const DistributionMapping& dmap,
                          double time, int finest_level)
{
    BL_PROFILE("MovingBodies::set_ground()");
    erf_actuator::ScopedPhase timed(m_timers, erf_actuator::PhaseTimers::Startup);
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
    // each turbine's and disk's owner: a rank whose boxes it stands on where that keeps the balance (a restart
    // chooses again, from its own boxes)
    choose_owners(geom, grids, dmap);
#ifdef ERF_USE_OPENFAST
    // a fresh start: OpenFAST sees the bases on the terrain; a restart is restored by read_checkpoint()
    if (!m_restarting && !m_driver->initialized()) {
        m_driver->init(m_dt, m_openfast_tmax);
        for (int i = 0; i < static_cast<int>(m_driver->turbines().size()); ++i) {
            if (!m_driver->is_owner(i)) { continue; }
            const auto& t = m_driver->turbines()[i];
            AllPrint() << "  " << t.name << ": OpenFAST dt " << t.dt_fast << ", " << t.num_substeps
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
MovingBodies::choose_owners (const Geometry& geom, const BoxArray& grids, const DistributionMapping& dmap)
{
    BL_PROFILE("MovingBodies::choose_owners()");
    const BoxArray ba = amrex::convert(grids, IntVect::TheZeroVector());
    const Real dxy = static_cast<Real>(std::max(geom.CellSize(0), geom.CellSize(1)));
    // a body's footprint: the column of cells under its base, widened by its radius and two cells
    auto footprint = [&] (const std::array<Real,3>& b, Real radius) {
        const Real w = radius + Real(2.0) * dxy;
        return erf_actuator::cells_within({{b[0] - w, b[1] - w, b[2]}}, {{b[0] + w, b[1] + w, b[2]}}, geom, 0, true);
    };
    std::vector<int> owners;
    m_exchange_names.clear();
#ifdef ERF_USE_OPENFAST
    const auto& turbs = m_driver->turbines();
    const int n = static_cast<int>(turbs.size());
    if (n > 0) {
        // the rotor radius (zero when unknown: the base column alone) and the time step each model's files
        // give, read at start-up (gather_model_facts: each distinct .fst once, on one rank)
        std::vector<Real> radius(n), dt(n);
        for (int i = 0; i < n; ++i) {
            const auto f = m_model_facts.find(turbs[i].fst_file);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f != m_model_facts.end(), "MovingBodies::choose_owners: no model facts for " + turbs[i].fst_file);
            radius[i] = std::max(f->second.rotor_radius, Real(0.0));
            dt[i] = f->second.time_step;
        }
        Print() << "Moving bodies: " << n << " turbine(s) share " << m_model_facts.size() << " distinct OpenFAST model file(s)\n";
        std::vector<Box> footprints;
        for (int i = 0; i < n; ++i) { footprints.push_back(footprint(turbs[i].base_pos, radius[i])); }
        owners = erf_actuator::choose_owners(footprints, dt, ba, dmap, geom, ParallelDescriptor::NProcs());
        m_turbine_vel.assign(n, std::vector<Real>());
        for (int i = 0; i < n; ++i) {
            // the owner alone accumulates the statistics, so it writes them
            m_turbine_stats[i].set_writer(owners[i]);
            m_exchange_names.push_back(turbs[i].name);
            std::ostringstream step;
            if (dt[i] > Real(0.0)) { step << dt[i] << " s"; } else { step << "unknown"; }
            Print() << "  " << turbs[i].name << ": owned by rank " << owners[i] << " (footprint radius " << radius[i] << " m, OpenFAST dt "
                    << step.str() << ")\n";
        }
    }
    m_driver->set_owners(owners);
#endif
    const int ndisk = static_cast<int>(m_disks.size());
    if (ndisk > 0) {
        // the disks by the same rule among themselves: they run no model, so they share a rank with a turbine
        // without holding up its OpenFAST steps, and one time step (unknown) keeps them all in one class
        std::vector<Box> footprints;
        for (const auto& d : m_disks) { footprints.push_back(footprint(d->center(), d->radius())); }
        const std::vector<int> disk_owners = erf_actuator::choose_owners(footprints, std::vector<Real>(ndisk, Real(-1.0)), ba, dmap, geom,
                                                                         ParallelDescriptor::NProcs());
        for (int k = 0; k < ndisk; ++k) {
            // the owner alone samples the flow for the disk and knows its thrust, so it writes its log and statistics
            m_disks[k]->set_writer(disk_owners[k]);
            m_disk_stats[k].set_writer(disk_owners[k]);
            m_exchange_names.push_back(m_disks[k]->name());
            Print() << "  " << m_disks[k]->name() << ": owned by rank " << disk_owners[k] << " (footprint radius " << m_disks[k]->radius()
                    << " m)\n";
        }
        owners.insert(owners.end(), disk_owners.begin(), disk_owners.end());
    }
    m_exchange = erf_actuator::BodyExchange(owners);
    m_exchange.set_check_interval(m_in.envelope_check_interval);
    m_reported_participants.clear();
}

void
MovingBodies::report_participants (const erf_actuator::BodyExchange& ex, const std::vector<std::string>& names,
                                   std::vector<std::vector<int>>& reported)
{
    reported.resize(ex.num_bodies());
    for (int b = 0; b < ex.num_bodies(); ++b) {
        const std::vector<int>& part = ex.participants(b);
        if (part.empty() || part == reported[b]) { continue; }
        reported[b] = part;
        std::ostringstream line;
        line << "  " << names[b] << ": owner rank " << ex.owner(b) << " exchanges its points with "
             << part.size() << " of " << ParallelDescriptor::NProcs() << " ranks (";
        for (std::size_t k = 0; k < part.size(); ++k) { line << (k > 0 ? " " : "") << part[k]; }
        line << ")\n";
        Print() << line.str();
    }
}

Real
MovingBodies::exchange_margin (int b, const Geometry& geom) const
{
#ifdef ERF_USE_OPENFAST
    if (b < num_turbines()) { return envelope_margin(b, geom); }
#else
    amrex::ignore_unused(b);
#endif
    // a disk stands still: two cells beyond its points
    return Real(2.0) * static_cast<Real>(std::max(geom.CellSize(0), std::max(geom.CellSize(1), geom.CellSize(2))));
}

void
MovingBodies::set_hubs ()
{
#ifdef ERF_USE_OPENFAST
    // a turbine's hub, about which its blades turn: what the throttled envelope checks follow
    for (int i = 0; i < num_turbines(); ++i) {
        if (!m_driver->is_owner(i)) { continue; }
        const erf_openfast::TurbineState& t = m_driver->turbines()[i];
        m_exchange.set_hub(i, t.hub_pos, erf_actuator::tip_radius(t));
    }
#endif
    // a disk stands still: the centre of its points
}

std::vector<Real>
MovingBodies::disk_table (int k, const std::function<void(int, Real*)>& fill) const
{
    const int nturb = num_turbines();
    return gather_rows(static_cast<int>(m_disks.size()), k, true, [this, nturb] (int i) { return m_exchange.owns(nturb + i); }, fill);
}

#ifdef ERF_USE_OPENFAST
std::vector<Real>
MovingBodies::turbine_table (int k, bool io_only, const std::function<void(int, Real*)>& fill) const
{
    return gather_rows(static_cast<int>(m_driver->turbines().size()), k, io_only,
                       [this] (int i) { return m_driver->is_owner(i); }, fill);
}

Real
MovingBodies::tip_cells_tolerance (const Geometry& geom)
{
    return Real(0.1) * static_cast<Real>(std::min({geom.CellSize(0), geom.CellSize(1), geom.CellSize(2)}));
}

void
MovingBodies::update_tip_cells (const MultiFab* z_phys_nd, const Geometry& geom)
{
    BL_PROFILE("MovingBodies::update_tip_cells()");
    const auto& turbs = m_driver->turbines();
    const int nturb = static_cast<int>(turbs.size());
    // the region turbine i's tips sweep, as one row of erf_actuator::tip_cell_heights; on its owner
    auto region = [this, &turbs] (int i, Real* row) {
        if (m_turbine_mode[i] != "alm") {
            for (int d = 0; d < 3; ++d) { row[d] = Real(1.0); row[d+3] = Real(0.0); }   // empty: no line
            row[6] = Real(0.0); row[7] = Real(0.0); row[8] = Real(0.0);
            return;
        }
        erf_actuator::tip_sweep_region(turbs[i], row);
    };
    // a region has moved when a value differs from the one last searched by more than the tolerance
    // (a non-finite value, a diverged hub, always has); a radius that becomes or stops being positive
    // changes whether the region counts at all
    const Real tol = tip_cells_tolerance(geom);
    auto moved = [tol] (const Real* now, const Real* was) {
        for (int d = 0; d < 9; ++d) {
            if (!std::isfinite(now[d]) || !std::isfinite(was[d]) ||
                !(now[d] == was[d] || std::abs(now[d] - was[d]) <= tol)) { return true; }
        }
        return (now[7] > Real(0.0)) != (was[7] > Real(0.0));
    };
    if (static_cast<int>(m_tip_rows.size()) != 9 * nturb || static_cast<int>(m_tip_dz.size()) != nturb) {
        m_tip_cells_valid = false;
    }
    // MovingFittedMesh updates z_phys_nd in place every step, even when its boxes do not change.
    if (SolverChoice::terrain_type == TerrainType::MovingFittedMesh) { m_tip_cells_valid = false; }
    // each owner compares its own turbines, without communication; one reduction tells every rank
    int stale = m_tip_cells_valid ? 0 : 1;
    if (stale == 0) {
        std::array<Real,9> row{};
        for (int i = 0; i < nturb && stale == 0; ++i) {
            if (!m_driver->is_owner(i)) { continue; }
            region(i, row.data());
            if (moved(row.data(), &m_tip_rows[static_cast<std::size_t>(9 * i)])) { stale = 1; }
        }
    }
    ParallelDescriptor::ReduceIntMax(stale);
    if (stale != 0) {
        // every rank now holds every region and so picks the same turbines to search again
        const std::vector<Real> rows = turbine_table(9, false, region);
        std::vector<int> redo;
        std::vector<Real> sub;
        for (int i = 0; i < nturb; ++i) {
            const Real* now = &rows[static_cast<std::size_t>(9 * i)];
            if (m_tip_cells_valid && !moved(now, &m_tip_rows[static_cast<std::size_t>(9 * i)])) { continue; }
            redo.push_back(i);
            sub.insert(sub.end(), now, now + 9);
        }
        if (!m_tip_cells_valid) {
            m_tip_rows.assign(static_cast<std::size_t>(9 * nturb), Real(0.0));
            m_tip_dz.assign(nturb, static_cast<Real>(geom.CellSize(2)));
        }
        std::vector<Real> dz;
        erf_actuator::tip_cell_heights_batched(z_phys_nd, geom, sub, dz);
        for (std::size_t k = 0; k < redo.size(); ++k) {
            const int i = redo[k];
            std::copy(sub.begin() + static_cast<std::ptrdiff_t>(9 * k), sub.begin() + static_cast<std::ptrdiff_t>(9 * k + 9),
                      m_tip_rows.begin() + static_cast<std::ptrdiff_t>(9 * i));
            m_tip_dz[i] = dz[k];
        }
        m_tip_cells_valid = true;
    }
    if (m_in.verify_tip_cells) {
        // the cached heights against the per-region search of the same regions on the current mesh
        std::vector<Real> fresh;
        erf_actuator::tip_cell_heights(z_phys_nd, geom, m_tip_rows, fresh);
        const Real eps = std::numeric_limits<Real>::epsilon();
        for (int i = 0; i < nturb; ++i) {
            const Real a = m_tip_dz[i], b = fresh[i];
            if (!(a == b || std::abs(a - b) <= Real(8.0) * eps * std::max(std::abs(a), std::abs(b)))) {
                std::ostringstream msg;
                msg << std::setprecision(17) << "erf.moving_bodies.verify_tip_cells: the cached limiting cell height of " << m_turbine_in[i].name
                    << " is " << a << " m but a fresh search of its region finds " << b << " m (step " << m_step << ")";
                Abort(msg.str());
            }
        }
    }
}

bool
MovingBodies::print_turbine_texts (const std::vector<std::string>& text, const std::vector<int>& fatal) const
{
    const int n = static_cast<int>(m_driver->turbines().size());
    const int io = ParallelDescriptor::IOProcessorNumber();
    // two tags of the sequence every rank draws from alike
    const int tag_len = ParallelDescriptor::SeqNum();
    const int tag_text = ParallelDescriptor::SeqNum();
    std::vector<std::string> all(text);
    // each owner sends its turbines' texts to the I/O rank, one message per owner (a start-up audit, once)
    std::set<int> owners;
    for (int i = 0; i < n; ++i) { owners.insert(m_driver->turbines()[i].owner_rank); }
    for (int r : owners) {
        if (r == io) { continue; }
        if (ParallelDescriptor::MyProc() == r) {
            std::string msg;
            for (int i = 0; i < n; ++i) {
                if (m_driver->is_owner(i)) { msg += std::to_string(text[i].size()) + " " + text[i]; }
            }
            Vector<char> chars(msg.begin(), msg.end());
            int len = static_cast<int>(chars.size());
            ParallelDescriptor::Send(&len, 1, io, tag_len);
            if (len > 0) { ParallelDescriptor::Send(chars.data(), static_cast<std::size_t>(len), io, tag_text); }
        } else if (ParallelDescriptor::IOProcessor()) {
            int len = 0;
            ParallelDescriptor::Recv(&len, 1, r, tag_len);
            Vector<char> chars(len);
            if (len > 0) { ParallelDescriptor::Recv(chars.data(), static_cast<std::size_t>(len), r, tag_text); }
            std::istringstream in(std::string(chars.begin(), chars.end()));
            for (int i = 0; i < n; ++i) {
                if (m_driver->turbines()[i].owner_rank != r) { continue; }
                std::size_t m = 0;
                in >> m;
                in.get();   // the space after the length
                std::string t(m, '\0');
                in.read(&t[0], static_cast<std::streamsize>(m));
                all[i] = t;
            }
        }
    }
    for (int i = 0; i < n; ++i) { Print() << all[i]; }
    int any = 0;
    for (int i = 0; i < n; ++i) { if (m_driver->is_owner(i) && fatal[i] != 0) { any = 1; } }
    ParallelDescriptor::ReduceIntMax(any);
    return any != 0;
}

Real
MovingBodies::envelope_margin (int i, const Geometry& geom) const
{
    const Real dx = static_cast<Real>(std::max(geom.CellSize(0), std::max(geom.CellSize(1), geom.CellSize(2))));
    return erf_actuator::tip_radius(m_driver->turbines()[i]) + Real(2.0) * dx;
}
#endif

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
    BL_PROFILE("MovingBodies::read_checkpoint()");
    erf_actuator::ScopedPhase timed(m_timers, erf_actuator::PhaseTimers::Startup);
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
    // each body's statistics, read by its owner alone (which alone keeps them); a malformed file stops every
    // rank with its owner's message, the first in body order as when every rank read every file
    {
        const int nturb = num_turbines();
        const int nbody = nturb + static_cast<int>(m_disks.size());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_exchange.num_bodies() == nbody, "MovingBodies::read_checkpoint: the owners are chosen first");
        std::string failed;
        int key = -1;
        bool restored = false;
        for (int b = 0; b < nbody && failed.empty(); ++b) {
            if (!m_exchange.owns(b)) { continue; }
            auto& st = (b < nturb) ? m_turbine_stats[b] : m_disk_stats[b - nturb];
            std::string err;
            st.read_state(dir, &err);
            if (!err.empty()) { failed = err; key = b; }
            if (st.num_samples() > 0) { restored = true; }
        }
        const std::string first = erf_actuator::first_failure(failed, key);
        if (!first.empty()) { Abort(first); }
        // rewritten every stats_int steps: the files the restored statistics stand for (a run that went on
        // past the checkpoint left later ones), even when no sample follows before the end; decided alike
        // on every rank, as the farm format's rewrite is collective
        if (m_in.stats_int > 0) {
            ParallelDescriptor::ReduceBoolOr(restored);
            m_stats_unwritten = restored;
        }
    }
    // the state file: the step count, and the OpenFAST time index each turbine must report
    std::map<std::string,int> time_index;
    std::map<std::string,Real> corr_uinf;
    std::map<std::string,std::map<std::string,std::string>> bodies;   // body records of the run being continued
    double t0 = -1.0, openfast_tmax = -1.0, dt_chk = -1.0;
    int anchor_chk = -1, solved0 = 1, nranks_chk = 0;
    std::string format_chk;
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
            } else if (key == "diagnostics_format") {
                if (!(ls >> eq >> format_chk) || eq != "=") { Abort("malformed diagnostics_format line in '" + dir + "/state'"); }
            } else if (key == "diagnostics_nranks") {
                if (!(ls >> eq >> nranks_chk) || eq != "=") { Abort("malformed diagnostics_nranks line in '" + dir + "/state'"); }
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
            std::string key, chk_value, now_value;
            if (MovingBodiesInputs::changed_record_entry(it->second, MovingBodiesInputs::restart_record(b), key, chk_value, now_value)) {
                Abort("erf.moving_bodies." + b.name + "." + record_key(key) + " changed on restart (" + chk_value + " in the "
                      "checkpoint, " + now_value + " now); a restart continues the same bodies with the same inputs");
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
        // each file is trimmed by the rank that appends to it in this run: the I/O rank the run's own files
        // and the farm files, each rank its per_rank files (and the I/O rank those of ranks this run does not
        // have), each body's owner the body's files and its wake lines'. Every file is cut from its end, so a
        // rank reads only the rows past the checkpoint
        std::vector<std::string> end_stamped, start_stamped;
        const bool io = ParallelDescriptor::IOProcessor();
        if (io) {
            end_stamped.push_back(m_in.diagnostics_dir + "/total_load.csv");
            start_stamped.push_back(m_in.diagnostics_dir + "/momentum_source.csv");
        }
        // the bodies' time series where this run writes them (a run continued in another format starts
        // new files and leaves the old ones as they are)
        if (!format_chk.empty() && format_chk != m_in.diagnostics_format) {
            Print() << "Moving bodies: the run being continued wrote its logs as " << format_chk << ", this one as "
                    << m_in.diagnostics_format << " (erf.moving_bodies.diagnostics_format); the old files are not trimmed\n";
        }
        const int nturb = num_turbines();
        const int nbody = nturb + static_cast<int>(m_disks.size());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_exchange.num_bodies() == nbody, "MovingBodies::read_checkpoint: the owners are chosen before the logs are trimmed");
        if (m_in.diagnostics_format == "per_turbine") {
            for (int b = 0; b < nbody; ++b) {
                if (!m_exchange.owns(b)) { continue; }
                if (b < nturb) {
                    end_stamped.push_back(body_log_file(b, "erf"));
                    end_stamped.push_back(body_log_file(b, "flow"));
                    start_stamped.push_back(body_log_file(b, "correction"));
                    start_stamped.push_back(body_log_file(b, "fllc"));
                } else {
                    start_stamped.push_back(body_log_file(b, "disk"));
                }
            }
        } else {
            // farm: one file per log; per_rank: the files of every rank of this run and of the run being continued
            const int me = ParallelDescriptor::MyProc();
            const int np = ParallelDescriptor::NProcs();
            const int nr = std::max(np, nranks_chk);
            auto add = [&] (std::vector<std::string>& to, const std::string& log) {
                if (m_in.diagnostics_format == "farm") {
                    if (io) { to.push_back(farm_log_file(log)); }
                } else {
                    for (int r = 0; r < nr; ++r) {
                        if (r == me || (r >= np && io)) { to.push_back(rank_log_file(r, log)); }
                    }
                }
            };
            for (const char* log : {"erf", "flow"}) { add(end_stamped, log); }
            for (const char* log : {"correction", "fllc", "disk"}) { add(start_stamped, log); }
        }
        for (int b = 0; b < nbody; ++b) {
            if (m_exchange.owns(b)) { start_stamped.push_back(body_log_file(b, "wake")); }
        }
        BL_PROFILE("MovingBodies::read_checkpoint: trim the logs");
        for (const auto& f : end_stamped) { erf_actuator::trim_log_after(f, time, keep_end_rows, tol); }
        for (const auto& f : start_stamped) { erf_actuator::trim_log_after(f, time, false, tol); }
        // every file is cut before any rank goes on (a shared file system shows the cut files alike)
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
        // the owner restored the turbine and knows its time index
        if (t.owner_rank != ParallelDescriptor::MyProc()) { continue; }
        if (it->second != t.time_index) {
            Abort("erf.moving_bodies." + t.name + ": the OpenFAST checkpoint is at time index " +
                  std::to_string(t.time_index) + " but ERF's checkpoint expects " + std::to_string(it->second));
        }
        AllPrint() << "  " << t.name << ": restored from " << dir << "/" << t.name << ".chkp at OpenFAST time index "
                   << t.time_index << ", " << t.num_substeps << " substeps per ERF step, on rank " << t.owner_rank << "\n";
    }
    // the node loads and velocities ERF checkpointed, read by each turbine's owner (a checkpoint from
    // before they were written has none)
    {
        const auto& turbs = m_driver->turbines();
        for (std::size_t i = 0; i < turbs.size(); ++i) {
            if (!m_driver->is_owner(static_cast<int>(i))) { continue; }
            const std::string fname = dir + "/" + turbs[i].name + "_nodes.dat";
            std::ifstream in(fname);
            if (!in) { continue; }
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
    BL_PROFILE("MovingBodies::advance()");
    if (m_timing_from < 0) { m_timing_from = m_step + 1; }
    {
        erf_actuator::ScopedPhase timed(m_timers, erf_actuator::PhaseTimers::Advance);
        advance_step(time, dt, cons, U, V, W, z_phys_nd, detJ_cc, geom);
    }
    // every rank counts the same steps, so every rank takes part in the summary's reductions
    if (m_in.timing_int > 0 && m_step % m_in.timing_int == 0) { report_timing(); }
}

void
MovingBodies::report_timing ()
{
    BL_PROFILE("MovingBodies::report_timing()");
    using P = erf_actuator::PhaseTimers;
    const int io = ParallelDescriptor::IOProcessorNumber();
    std::array<double, P::NumPhases> mn = m_timers.seconds(), sum = m_timers.seconds(), mx = m_timers.seconds();
    ParallelDescriptor::ReduceRealMin(mn.data(), P::NumPhases, io);
    ParallelDescriptor::ReduceRealSum(sum.data(), P::NumPhases, io);
    ParallelDescriptor::ReduceRealMax(mx.data(), P::NumPhases, io);
    std::string text;
    if (ParallelDescriptor::IOProcessor()) {
        text = P::summary(m_timing_from, m_step, ParallelDescriptor::NProcs(), mn.data(), sum.data(), mx.data());
    }
#ifdef ERF_USE_OPENFAST
    if (num_turbines() > 0) {
        // each turbine's seconds from its owner, onto the I/O rank
        const std::vector<Real> sec = turbine_table(1, true, [this] (int i, Real* row) {
            row[0] = static_cast<Real>(m_driver->turbines()[i].step_seconds);
        });
        if (ParallelDescriptor::IOProcessor()) {
            std::vector<std::string> names;
            for (const auto& t : m_driver->turbines()) { names.push_back(t.name); }
            text += P::turbine_summary(names, std::vector<double>(sec.begin(), sec.end()));
        }
        m_driver->reset_step_seconds();
    }
#endif
    Print() << text;
    m_timers.reset();
    m_timing_from = m_step + 1;
}

void
MovingBodies::advance_step (double time, double dt, const MultiFab& cons,
                            const MultiFab& U, const MultiFab& V, const MultiFab& W,
                            const MultiFab* z_phys_nd, const MultiFab* detJ_cc, const Geometry& geom)
{
    using P = erf_actuator::PhaseTimers;
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
    {
        // the start-up and regrid checks: the coverage, the disks' density, the wake lines of a restart, the audits
        erf_actuator::ScopedPhase timed(m_timers, P::Checks);
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
#endif
    }
#ifdef ERF_USE_OPENFAST
    if (first && !m_driver->solved0()) {
        // first step: OpenFAST's first solution sees the initial flow at its nodes
        { erf_actuator::ScopedPhase timed(m_timers, P::Sample); supply_velocities(time, U, V, W, z_phys_nd, geom); }
        { erf_actuator::ScopedPhase timed(m_timers, P::OpenFAST); m_driver->solution0(); }
        erf_actuator::ScopedPhase timed(m_timers, P::Diagnostics);
        write_turbine_logs(time, true);
        write_total_load(time, true);
    }
#endif
    { erf_actuator::ScopedPhase timed(m_timers, P::Sample); supply_velocities(time, U, V, W, z_phys_nd, geom); }
#ifdef ERF_USE_OPENFAST
    if (!m_audited_flow) { erf_actuator::ScopedPhase timed(m_timers, P::Checks); audit_turbines_flow(cons, z_phys_nd, geom); }
    { erf_actuator::ScopedPhase timed(m_timers, P::OpenFAST); m_driver->step(); }
    if (first || m_step % m_in.diagnostics_int == 0) {
        erf_actuator::ScopedPhase timed(m_timers, P::Diagnostics);
        write_turbine_logs(time + dt, false);
        write_fllc_diagnostics(time, first);
        write_correction_diagnostics(time, first);
        write_total_load(time + dt, false);
    }
#endif
    // the bodies' forces come from the velocities just sampled (the disks) or from the
    // OpenFAST step just taken (the turbines), so the source is fixed over the step ERF is
    // about to take
    if (any_forcing()) {
        { erf_actuator::ScopedPhase timed(m_timers, P::Spread); spread_sources(U, z_phys_nd, detJ_cc, geom); }
        if (first || m_step % m_in.diagnostics_int == 0) {
            erf_actuator::ScopedPhase timed(m_timers, P::Diagnostics);
            write_source_diagnostics(time, detJ_cc, geom, first);
        }
    }
#ifndef ERF_USE_OPENFAST
    amrex::ignore_unused(cons);
#endif
    if (time + dt >= m_in.avg_start) {
        erf_actuator::ScopedPhase timed(m_timers, P::Statistics);
        accumulate_statistics(time + dt);
    }
    if (m_in.wake.active() && !m_in.has_prescribed_velocity && (first || m_step % m_in.wake.interval == 0)) {
        erf_actuator::ScopedPhase timed(m_timers, P::Wake);
        write_wake_diagnostics(time, U, V, W, z_phys_nd, geom);
    }
    if (m_in.stats_int > 0 && m_step % m_in.stats_int == 0) {
        erf_actuator::ScopedPhase timed(m_timers, P::Statistics);
        rewrite_summaries();
    }
    // the step's rows that go through the I/O rank (the farm totals, and every log in the farm format), in one reduction
    erf_actuator::ScopedPhase timed(m_timers, P::Diagnostics);
    flush_logs();
}

void
MovingBodies::accumulate_statistics (double time)
{
    BL_PROFILE("MovingBodies::accumulate_statistics()");
#ifdef ERF_USE_OPENFAST
    const auto& turbs = m_driver->turbines();
    for (std::size_t i = 0; i < turbs.size(); ++i) {
        // the owner alone holds the turbine's loads, so it alone keeps (and writes) its statistics
        if (!m_driver->is_owner(static_cast<int>(i))) { continue; }
        const auto& t = turbs[i];
        const std::array<Real,3> f = m_driver->thrust(t);
        std::array<Real,3> hub, blade;
        m_driver->node_velocity_means(t, hub, blade);
        const Real q = m_driver->torque(t);
        m_turbine_stats[i].accumulate(time, {f[0]*t.hub_axis[0] + f[1]*t.hub_axis[1] + f[2]*t.hub_axis[2], f[0], q,
                                             q * t.rotor_speed, t.rotor_speed, hub[0], hub[1], hub[2], blade[0]});
    }
#endif
    for (std::size_t k = 0; k < m_disks.size(); ++k) {
        // the owner alone knows the disk's speeds and thrust
        if (!m_exchange.owns(num_turbines() + static_cast<int>(k))) { continue; }
        const auto& d = m_disks[k];
        m_disk_stats[k].accumulate(time, {d->free_stream_speed(), d->disk_speed(), d->thrust(), d->power()});
    }
    m_stats_unwritten = true;
    if (m_in.stats_int == 0 && m_step % m_in.diagnostics_int == 0) { write_statistics(); }
}

void
MovingBodies::rewrite_summaries ()
{
    BL_PROFILE("MovingBodies::rewrite_summaries()");
    if (m_stats_unwritten) { write_statistics(); }
    if (m_wake_avg_unwritten) {
        for (int b = 0; b < static_cast<int>(m_wakes.size()); ++b) {
            if (m_wake_exchange.owns(b)) { m_wakes[b]->write_average(); }
        }
        m_wake_avg_unwritten = false;
    }
}

void
MovingBodies::finalize ()
{
    BL_PROFILE("MovingBodies::finalize()");
    if (m_in.stats_int > 0) {
        rewrite_summaries();
        flush_logs();
    }
    m_logs.flush();
}

void
MovingBodies::write_statistics ()
{
    BL_PROFILE("MovingBodies::write_statistics()");
    m_stats_unwritten = false;
    // body b of m_exchange: its statistics (a turbine's, then a disk's)
    const int nturb = num_turbines();
    const int nbody = nturb + static_cast<int>(m_disks.size());
    auto stats = [this, nturb] (int b) -> const erf_actuator::RunningStats& {
        return (b < nturb) ? m_turbine_stats[b] : m_disk_stats[b - nturb];
    };
    const std::string& format = m_in.diagnostics_format;
    if (format == "per_turbine") {
        // each owner rewrites its bodies' files
        for (int b = 0; b < nbody; ++b) { stats(b).write(); }
        return;
    }
    const std::string header = std::string("body,") + erf_actuator::RunningStats::columns();
    if (format == "per_rank") {
        // each rank rewrites its own file with the statistics of the bodies it owns
        bool any = false;
        for (int b = 0; b < nbody && !any; ++b) { any = m_exchange.owns(b); }
        if (!any) { return; }
        std::ofstream out;
        erf_actuator::open_log(out, rank_log_file(ParallelDescriptor::MyProc(), "stats"), true);
        out << header << "\n";
        for (int b = 0; b < nbody; ++b) {
            if (m_exchange.owns(b)) { stats(b).write_rows(out, m_exchange_names[b] + ","); }
        }
        return;
    }
    // farm: per body, a flag, the count, the first and last times and four values per quantity, from its
    // owner; the I/O rank rewrites the one file
    std::vector<std::size_t> offset(nbody + 1, 0);
    for (int b = 0; b < nbody; ++b) { offset[b+1] = offset[b] + 4 + 4 * stats(b).quantities().size(); }
    PendingLog part;
    part.packed.assign(offset[nbody], 0.0);
    for (int b = 0; b < nbody; ++b) {
        if (!m_exchange.owns(b)) { continue; }
        const auto& st = stats(b);
        double* v = &part.packed[offset[b]];
        v[0] = 1.0;
        v[1] = st.num_samples();
        v[2] = st.t_first();
        v[3] = st.t_last();
        for (std::size_t q = 0; q < st.quantities().size(); ++q) {
            v[4+4*q] = st.mean(q); v[5+4*q] = st.rms(q); v[6+4*q] = st.min(q); v[7+4*q] = st.max(q);
        }
    }
    part.write = [this, nbody, offset, stats, header] (const double* all) {
        std::ofstream out;
        erf_actuator::open_log(out, farm_log_file("stats"), true);
        out << header << "\n";
        for (int b = 0; b < nbody; ++b) {
            const double* v = all + offset[b];
            if (v[0] == 0.0) { continue; }
            const auto& names = stats(b).quantities();
            std::vector<Real> q(4 * names.size());
            for (std::size_t k = 0; k < q.size(); ++k) { q[k] = static_cast<Real>(v[4+k]); }
            erf_actuator::RunningStats::write_rows(out, m_exchange_names[b] + ",", names, static_cast<int>(v[1]), v[2], v[3], q.data());
        }
    };
    m_pending_logs.push_back(std::move(part));
}

void
MovingBodies::build_wake_lines (const BoxArray& level_grids, const MultiFab* z_phys_nd, const Geometry& geom)
{
    BL_PROFILE("MovingBodies::build_wake_lines()");
    const auto& w = m_in.wake;
    // each vertical line stops at the terrain under its own centre (downstream, the ground may rise or fall):
    // the lines' centres of every rotor first, then the terrain under all of them in one collective
    struct Rotor { std::string name, root; std::array<Real,3> hub, axis; Real diameter; };
    std::vector<Rotor> rotors;
    std::vector<Real> centres;
    auto add = [&](const std::string& name, const std::string& root, const std::array<Real,3>& hub,
                   const std::array<Real,3>& axis, Real diameter) {
        const std::size_t first = centres.size();
        Real ah[2] = {axis[0], axis[1]};
        const Real la = std::sqrt(ah[0] * ah[0] + ah[1] * ah[1]);
        if (la > Real(1.0e-6)) { ah[0] /= la; ah[1] /= la; }
        for (const Real xD : w.lines_xD) {
            centres.insert(centres.end(), {hub[0] + xD * diameter * ah[0], hub[1] + xD * diameter * ah[1], hub[2]});
        }
        for (std::size_t c = 0; c < w.lines_xD.size(); ++c) {
            for (int d = 0; d < 2; ++d) {
                const Real x = centres[first + 3*c + d];
                if (!geom.isPeriodic(d) && (x < geom.ProbLo(d) || x > geom.ProbHi(d))) {
                    Abort("erf.moving_bodies.wake: the line " + std::to_string(w.lines_xD[c]) + " D behind " + name +
                          " lies outside the domain; shorten lines_xD");
                }
            }
        }
        rotors.push_back({name, root, hub, axis, diameter});
    };
#ifdef ERF_USE_OPENFAST
    {
        // every rotor's hub, axis and diameter, from its owner (every rank keeps the lines' geometry)
        const std::vector<Real> rotor = turbine_table(7, false, [this] (int i, Real* row) {
            const auto& t = m_driver->turbines()[i];
            // the rotor diameter from the outermost blade force node
            Real r2max = 0.0;
            const int nfb = t.num_blades * t.num_force_pts_blade;
            for (int nd = 1; nd <= nfb && nd < t.num_force_nodes; ++nd) {
                Real r2 = 0.0;
                for (int d = 0; d < 3; ++d) { const Real dd = t.force_pos[3*nd+d] - t.hub_pos[d]; r2 += dd * dd; }
                r2max = std::max(r2max, r2);
            }
            if (!(r2max > 0.0)) { Abort("erf.moving_bodies.wake: " + t.name + " has no blade force nodes to size the wake lines from"); }
            for (int d = 0; d < 3; ++d) { row[d] = t.hub_pos[d]; row[3+d] = t.hub_axis[d]; }
            row[6] = Real(2.0) * std::sqrt(r2max);
        });
        const auto& turbs = m_driver->turbines();
        for (std::size_t i = 0; i < turbs.size(); ++i) {
            const Real* row = &rotor[7*i];
            add(turbs[i].name, turbs[i].output_root, {{row[0], row[1], row[2]}}, {{row[3], row[4], row[5]}}, row[6]);
        }
    }
#endif
    for (const auto& d : m_disks) {
        add(d->name(), d->output_root(), d->center(), d->normal(), Real(2.0) * d->radius());
    }
    {
        std::vector<Real> ground;
        erf_actuator::terrain_heights(z_phys_nd, geom, centres, ground);
        const std::size_t nl = w.lines_xD.size();
        for (std::size_t r = 0; r < rotors.size(); ++r) {
            const std::vector<Real> g(ground.begin() + static_cast<std::ptrdiff_t>(r * nl), ground.begin() + static_cast<std::ptrdiff_t>((r + 1) * nl));
            m_wakes.push_back(std::make_unique<erf_actuator::WakeLines>(rotors[r].name, rotors[r].root, rotors[r].hub, rotors[r].axis,
                                                                        rotors[r].diameter, w.lines_xD, w.half_width, w.num_points, g));
        }
    }
    // every sampling point must lie on the level's grids, checked by the rotor's owner (the lines behind
    // rotor b are body b of m_exchange), and in the domain (periodic directions wrap)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(m_wakes.size()) == m_exchange.num_bodies(),
                                     "MovingBodies::build_wake_lines: one set of wake lines per turbine and disk");
    {
        const erf_actuator::CoverageGrids cover(level_grids, geom);
        std::string failed;
        int key = 0;
        for (int b = 0; b < static_cast<int>(m_wakes.size()) && failed.empty(); ++b) {
            if (!m_exchange.owns(b)) { continue; }
            const auto& wl = m_wakes[b];
            std::string outside;
            // with z_phys_nd a point's cell is not its height over dz: the sampler finds it in the column itself
            if (!erf_actuator::points_covered_by(cover, geom, wl->positions(), geom.CellSize(0), outside,
                                                 (z_phys_nd == nullptr) ? erf_actuator::CoverZ::Reach : erf_actuator::CoverZ::Footprint)) {
                failed = "erf.moving_bodies.wake: the sampling point " + outside + " of " + wl->name() + " is not covered by the grids of level " +
                         std::to_string(m_anchor) + "; shorten lines_xD or half_width, or enlarge the refinement region behind the rotor";
                key = b;
            }
        }
        const std::string first = erf_actuator::first_failure(failed, key);
        if (!first.empty()) { Abort(first); }
    }
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

        Print() << "  " << wl->name() << ": wake lines at " << w.lines_xD.size() << " downstream distances, "
                << wl->num_points() << " points, diameter " << wl->diameter() << " m\n";
    }
    // after a restart, each rotor's running average, read by its owner alone (which alone samples the lines);
    // a malformed file stops every rank with its owner's message, the first in rotor order
    if (!m_wake_state_dir.empty()) {
        std::string failed;
        int key = -1;
        for (int b = 0; b < static_cast<int>(m_wakes.size()) && failed.empty(); ++b) {
            if (!m_exchange.owns(b)) { continue; }
            const auto& wl = m_wakes[b];
            std::string err;
            if (wl->read_state(m_wake_state_dir, &err)) {
                AllPrint() << "  " << wl->name() << ": wake running average restored (" << wl->num_samples() << " samples) on rank "
                           << ParallelDescriptor::MyProc() << "\n";
                // as the statistics on a restart (read_checkpoint); the wake averages' rewrite is the owner's alone
                if (m_in.stats_int > 0) { m_wake_avg_unwritten = true; }
            }
            if (!err.empty()) { failed = err; key = b; }
        }
        const std::string first = erf_actuator::first_failure(failed, key);
        if (!first.empty()) { Abort(first); }
    }
    // the lines behind a rotor go with it (the turbines' then the disks', as in m_exchange): its owner samples
    // them, keeps their running average and writes their files and checkpoint state
    std::vector<int> owners;
    m_wake_exchange_names.clear();
    for (int b = 0; b < m_exchange.num_bodies(); ++b) {
        owners.push_back(m_exchange.owner(b));
        m_wakes[b]->set_writer(m_exchange.owner(b));
        m_wake_exchange_names.push_back(m_wakes[b]->name() + " wake lines");
    }
    m_wake_exchange = erf_actuator::BodyExchange(owners);
    m_reported_wake_participants.clear();
    m_wakes_built = true;
}

void
MovingBodies::write_wake_diagnostics (double time, const MultiFab& U, const MultiFab& V, const MultiFab& W,
                                      const MultiFab* z_phys_nd, const Geometry& geom)
{
    BL_PROFILE("MovingBodies::write_wake_diagnostics()");
    if (!m_wakes_built) { build_wake_lines(U.boxArray(), z_phys_nd, geom); }
    // each rotor's owner sends its lines' points to the ranks whose boxes hold them; the lines stand
    // still, so the envelope reaches two cells beyond them
    const int nw = static_cast<int>(m_wakes.size());
    const Real margin_dx = Real(2.0) * static_cast<Real>(std::max(geom.CellSize(0), std::max(geom.CellSize(1), geom.CellSize(2))));
    std::vector<std::vector<Real>> pts(nw), vel;
    std::vector<Real> margin(nw, Real(0.0));
    for (int b = 0; b < nw; ++b) {
        if (!m_wake_exchange.owns(b)) { continue; }
        pts[b] = m_wakes[b]->positions();
        margin[b] = margin_dx;
    }
    m_wake_exchange.sample_velocity(U, V, W, z_phys_nd, geom, pts, margin, vel);
    report_participants(m_wake_exchange, m_wake_exchange_names, m_reported_wake_participants);
    for (int b = 0; b < nw; ++b) {
        if (!m_wake_exchange.owns(b)) { continue; }
        auto& wl = m_wakes[b];
        if (time >= m_in.avg_start) { wl->accumulate(vel[b]); }
        wl->write_instantaneous(m_logs, time, vel[b], !m_resumed && !m_wake_written);
        if (m_in.stats_int == 0) { wl->write_average(); }
    }
    m_wake_written = true;
    m_wake_avg_unwritten = (m_in.stats_int > 0);
}

bool
MovingBodies::any_forcing () const
{
    if (!m_disks.empty()) { return true; }
    for (const std::string& m : m_turbine_mode) { if (m != "none") { return true; } }
    return false;
}

std::string
MovingBodies::body_log_file (int b, const std::string& log) const
{
    const int nturb = num_turbines();
    const std::string& root = (b < nturb) ? m_turbine_in[b].output_root : m_disks[b - nturb]->output_root();
    return root + "_" + log + ".csv";
}

std::string
MovingBodies::rank_log_file (int r, const std::string& log) const
{
    return m_in.diagnostics_dir + "/" + amrex::Concatenate("rank", r, 5) + "_" + log + ".csv";
}

std::string
MovingBodies::farm_log_file (const std::string& log) const
{
    return m_in.diagnostics_dir + "/farm_" + log + ".csv";
}

void
MovingBodies::write_series (const std::string& log, const std::string& columns, double time, bool truncate,
                            const std::vector<int>& bodies, int k, const std::function<bool(int, Real*)>& fill)
{
    const std::string& format = m_in.diagnostics_format;
    std::vector<Real> row(k);
    if (format == "per_turbine" || format == "per_rank") {
        // each owner writes its bodies' rows at once: into each body's own file, or into its rank's
        const bool per_rank = (format == "per_rank");
        for (const int b : bodies) {
            if (!m_exchange.owns(b) || !fill(b, row.data())) { continue; }
            if (per_rank) {
                std::ofstream& out = m_logs.get(rank_log_file(ParallelDescriptor::MyProc(), log), truncate, "time,body," + columns);
                erf_actuator::write_log_row(out, time, &m_exchange_names[b], row.data(), k);
            } else {
                erf_actuator::write_log_row(m_logs.get(body_log_file(b, log), truncate, "time," + columns), time, nullptr, row.data(), k);
            }
        }
        return;
    }
    // farm: per body a flag and its row, from its owner, into the step's gather; the I/O rank writes them in body order
    const int n = static_cast<int>(bodies.size());
    PendingLog part;
    part.packed.assign(static_cast<std::size_t>(n) * (k + 1), 0.0);
    for (int j = 0; j < n; ++j) {
        if (!m_exchange.owns(bodies[j]) || !fill(bodies[j], row.data())) { continue; }
        double* v = &part.packed[static_cast<std::size_t>(j) * (k + 1)];
        v[0] = 1.0;
        for (int c = 0; c < k; ++c) { v[1+c] = static_cast<double>(row[c]); }
    }
    part.write = [this, log, columns, time, truncate, bodies, k] (const double* all) {
        std::ofstream& out = m_logs.get(farm_log_file(log), truncate, "time,body," + columns);
        std::vector<Real> r(k);
        for (std::size_t j = 0; j < bodies.size(); ++j) {
            const double* v = all + j * (k + 1);
            if (v[0] == 0.0) { continue; }
            for (int c = 0; c < k; ++c) { r[c] = static_cast<Real>(v[1+c]); }
            erf_actuator::write_log_row(out, time, &m_exchange_names[bodies[j]], r.data(), k);
        }
    };
    m_pending_logs.push_back(std::move(part));
}

void
MovingBodies::flush_logs ()
{
    BL_PROFILE("MovingBodies::flush_logs()");
    // the same parts, of the same sizes, on every rank: they were added under conditions every rank shares
    if (m_pending_logs.empty()) { return; }
    std::size_t total = 0;
    for (const auto& p : m_pending_logs) { total += p.packed.size(); }
    std::vector<double> all;
    all.reserve(total);
    for (const auto& p : m_pending_logs) { all.insert(all.end(), p.packed.begin(), p.packed.end()); }
    // each value comes from one rank, zeros from the others, so the sum is every value exactly
    if (!all.empty()) {
        ParallelDescriptor::ReduceRealSum(all.data(), static_cast<int>(all.size()), ParallelDescriptor::IOProcessorNumber());
    }
    if (ParallelDescriptor::IOProcessor()) {
        std::size_t at = 0;
        for (const auto& p : m_pending_logs) {
            p.write(all.data() + at);
            at += p.packed.size();
        }
    }
    m_pending_logs.clear();
}

void
MovingBodies::write_total_load (double time, bool first)
{
    BL_PROFILE("MovingBodies::write_total_load()");
    const int nturb = num_turbines();
    const int ndisk = static_cast<int>(m_disks.size());
    // every turbine's load, torque and rotor speed (the torque and the rotor speed apart, so that the power
    // is formed as the turbine's own log forms it) and every disk's thrust, from their owners, into the
    // step's gather; the I/O rank sums them in body order
    PendingLog part;
    part.packed.assign(5 * static_cast<std::size_t>(nturb) + ndisk, 0.0);
#ifdef ERF_USE_OPENFAST
    for (int i = 0; i < nturb; ++i) {
        if (!m_driver->is_owner(i)) { continue; }
        const auto& t = m_driver->turbines()[i];
        const auto f = m_driver->thrust(t);
        const auto ft = m_driver->tower_force(t);
        double* row = &part.packed[5 * static_cast<std::size_t>(i)];
        for (int d = 0; d < 3; ++d) { row[d] = static_cast<Real>(f[d] + ft[d] + t.nacelle_force[d]); }
        row[3] = m_driver->torque(t);
        row[4] = t.rotor_speed;
    }
#endif
    for (int k = 0; k < ndisk; ++k) {
        if (m_exchange.owns(nturb + k)) { part.packed[5 * static_cast<std::size_t>(nturb) + k] = m_disks[k]->thrust(); }
    }
    const bool truncate = first && !m_resumed && !m_total_written;
    m_total_written = true;
    part.write = [this, nturb, ndisk, time, truncate] (const double* all) {
        std::array<Real,3> load{{0.0, 0.0, 0.0}};
        Real power = 0.0;
        for (int i = 0; i < nturb; ++i) {
            if (m_turbine_mode[i] == "none") { continue; }   // no force in the flow
            const double* row = all + 5 * static_cast<std::size_t>(i);
            for (int d = 0; d < 3; ++d) { load[d] += static_cast<Real>(row[d]); }
            power += static_cast<Real>(row[3]) * static_cast<Real>(row[4]);
        }
        for (int k = 0; k < ndisk; ++k) {
            const auto n = m_disks[k]->normal();
            const Real thrust = static_cast<Real>(all[5 * static_cast<std::size_t>(nturb) + k]);
            for (int c = 0; c < 3; ++c) { load[c] += thrust * n[c]; }
        }
        const Real row[4] = {load[0], load[1], load[2], power};
        erf_actuator::write_log_row(m_logs.get(m_in.diagnostics_dir + "/total_load.csv", truncate, "time,load_x,load_y,load_z,power"),
                                    time, nullptr, row, 4);
    };
    m_pending_logs.push_back(std::move(part));
}

void
MovingBodies::write_source_diagnostics (double time, const MultiFab* detJ_cc, const Geometry& geom, bool first)
{
    BL_PROFILE("MovingBodies::write_source_diagnostics()");
    const Real fx = erf_actuator::integrate_source(0, m_src_x, detJ_cc, geom);
    const Real fy = erf_actuator::integrate_source(1, m_src_y, detJ_cc, geom);
    const Real fz = erf_actuator::integrate_source(2, m_src_z, detJ_cc, geom);
    // each disk's row from its owner, with the integrated source over all bodies projected on its normal and
    // signed as a thrust: equals the disk's thrust when it is the only body
    const int nturb = num_turbines();
    std::vector<int> bodies;
    for (int k = 0; k < static_cast<int>(m_disks.size()); ++k) { bodies.push_back(nturb + k); }
    if (!bodies.empty()) {
        write_series("disk", erf_actuator::PrescribedCtDisk::log_columns(), time, first, bodies, erf_actuator::PrescribedCtDisk::log_size,
                     [this, nturb, fx, fy, fz] (int b, Real* row) {
                         const auto& d = m_disks[b - nturb];
                         const auto n = d->normal();
                         d->log_row(-(n[0]*fx + n[1]*fy + n[2]*fz), row);
                         return true;
                     });
    }
    if (ParallelDescriptor::IOProcessor()) {
        const Real row[3] = {fx, fy, fz};
        erf_actuator::write_log_row(m_logs.get(m_in.diagnostics_dir + "/momentum_source.csv", first, "time,fx,fy,fz"), time, nullptr, row, 3);
    }
}

void
MovingBodies::spread_sources (const MultiFab& U, const MultiFab* z_phys_nd,
                              const MultiFab* detJ_cc, const Geometry& geom)
{
    BL_PROFILE("MovingBodies::spread_sources()");
    const BoxArray ba = amrex::convert(U.boxArray(), IntVect(0,0,0));
    if (!m_src_defined || m_src_x.boxArray() != amrex::convert(ba, IntVect(1,0,0)) || m_src_x.DistributionMap() != U.DistributionMap()) {
        const DistributionMapping& dm = U.DistributionMap();
        m_src_x.define(amrex::convert(ba, IntVect(1,0,0)), dm, 1, 0);
        m_src_y.define(amrex::convert(ba, IntVect(0,1,0)), dm, 1, 0);
        m_src_z.define(amrex::convert(ba, IntVect(0,0,1)), dm, 1, 0);
        m_src_defined = true;
    }
    {
        BL_PROFILE("MovingBodies::spread_sources: zero the source");
        m_src_x.setVal(0.0);
        m_src_y.setVal(0.0);
        m_src_z.setVal(0.0);
    }
    const Real dx0 = static_cast<Real>(geom.CellSize(0));
    // every body's points, forces and widths on its owner: the turbines', then the disks'
    const int nbody = m_exchange.num_bodies();
    std::vector<std::vector<Real>> pos(nbody), force(nbody), widths(nbody);
    std::vector<Real> margin(nbody, Real(0.0));
#ifdef ERF_USE_OPENFAST
    // each turbine's loads as actuator-disk rings (mode = adm) or as an actuator line on its
    // blade nodes (mode = alm), built by its owner; none puts no force in the flow
    const int nturb = static_cast<int>(m_driver->turbines().size());
    bool any_turbine_force = false;
    for (const std::string& m : m_turbine_mode) { any_turbine_force = any_turbine_force || (m != "none"); }
    if (any_turbine_force) {
        const auto& turbs = m_driver->turbines();
        // the cells an actuator line's tip sweeps are physical cells: on a terrain-following or
        // stretched mesh the height that limits the tip is the smallest over the cells of the disk
        // of dz over the fraction of the tip speed that can cross the cell there (vertically, less
        // towards the top and bottom of the disk; sideways across a sloping face), from z_phys_nd,
        // not the nominal dz (a collective: every rank takes part, each rotor's region from its owner;
        // kept from step to step and searched again only after a regrid or when the region moves)
        std::vector<Real> dz_rotor(nturb, static_cast<Real>(geom.CellSize(2)));
        bool any_alm = false;
        for (const std::string& m : m_turbine_mode) { any_alm = any_alm || (m == "alm"); }
        if (any_alm && z_phys_nd != nullptr) {
            update_tip_cells(z_phys_nd, geom);
            dz_rotor = m_tip_dz;
        }
        for (int i = 0; i < nturb; ++i) {
            if (m_turbine_mode[i] == "none" || !m_driver->is_owner(i)) { continue; }
            if (!m_axis_checked) {
                const Real s = erf_actuator::max_out_of_plane_sine(turbs[i]);
                const double deg = std::asin(std::min(s, Real(1.0))) * 180.0 / 3.14159265358979323846;
                if (s > Real(0.35)) {   // ~20 degrees: well beyond precone plus shaft tilt
                    Abort("erf.moving_bodies." + turbs[i].name + ": the blade nodes lie up to " +
                          std::to_string(deg) +
                          " degrees out of the plane normal to the hub axis; the hub orientation convention does not match this OpenFAST");
                }
                const auto& n = turbs[i].hub_axis;
                std::ostringstream line;
                line << "erf.moving_bodies." << turbs[i].name << ": hub axis (" << n[0] << ", " << n[1] << ", " << n[2]
                     << "), blade force nodes within " << deg << " degrees of the rotor plane, ";
                if (m_turbine_mode[i] == "adm") {
                    line << m_turbine_points_t[i] << " points per ring\n";
                } else {
                    line << "actuator line of " << turbs[i].num_blades * turbs[i].num_force_pts_blade
                         << " blade points, tip radius " << erf_actuator::tip_radius(turbs[i]) << " m\n";
                }
                AllPrint() << line.str();
            }
            if (m_turbine_mode[i] == "alm") {
                // the line's force must not jump over cells between two steps: the tip, the
                // fastest point, may sweep at most alm_max_tip_cells cells per ERF step
                const Real dx_min = std::min({static_cast<Real>(geom.CellSize(0)), static_cast<Real>(geom.CellSize(1)), dz_rotor[i]});
                const Real cells = erf_actuator::tip_cells_per_step(turbs[i], m_dt, dx_min);
                if (cells > m_in.alm_max_tip_cells) {
                    const Real r_tip = erf_actuator::tip_radius(turbs[i]);
                    const Real dt_max = m_in.alm_max_tip_cells * dx_min / std::max(std::abs(turbs[i].rotor_speed) * r_tip, Real(1.0e-30));
                    Abort("erf.moving_bodies." + turbs[i].name + ": the blade tip sweeps " + std::to_string(cells) +
                          " cells per step (rotor speed " + std::to_string(turbs[i].rotor_speed) + " rad/s, tip radius " +
                          std::to_string(r_tip) + " m, dt " + std::to_string(m_dt) + " s, limiting cell height " + std::to_string(dx_min) +
                          " m), more than erf.moving_bodies.alm_max_tip_cells = " + std::to_string(m_in.alm_max_tip_cells) +
                          "; use erf.fixed_dt <= " + std::to_string(dt_max * m_input_fixed_dt / m_dt) +
                          " (a step of " + std::to_string(dt_max) + " s on level " + std::to_string(m_anchor) +
                          ") for the actuator line");
                }
            }
            // the rotor's points, the tower's loads (OpenFAST's tower force nodes, if the model has
            // them and the user asked for them) and the nacelle drag point at the hub, in either
            // mode, each with its own kernel
            erf_actuator::turbine_points(turbs[i], m_turbine_mode[i], m_turbine_points_t[i], m_turbine_in[i].nacelle_cd > 0.0,
                                         m_turbine_in[i].kernel_widths(dx0), pos[i], force[i], widths[i]);
            margin[i] = exchange_margin(i, geom);
        }
        m_axis_checked = true;
    }
#endif
    // the prescribed-Ct disks after the turbines, each on its owner with its own width
    for (std::size_t k = 0; k < m_disks.size(); ++k) {
        const int b = num_turbines() + static_cast<int>(k);
        if (!m_exchange.owns(b)) { continue; }
        pos[b] = m_disks[k]->disk_points();
        force[b] = m_disks[k]->forces();
        widths[b].assign(pos[b].size() / 3, m_disk_epsilon[k] * dx0);
        margin[b] = exchange_margin(b, geom);
    }
    set_hubs();
    // each point onto the faces of the ranks its kernel reaches, normalised over all of them
    m_exchange.spread_forces(pos, force, widths, margin, z_phys_nd, detJ_cc, geom, m_src_x, m_src_y, m_src_z);
    report_participants(m_exchange, m_exchange_names, m_reported_participants);
}

void
MovingBodies::add_momentum_sources (int lev, MultiFab& xmom_src, MultiFab& ymom_src, MultiFab& zmom_src) const
{
    BL_PROFILE("MovingBodies::add_momentum_sources()");
    if (lev != anchor_level() || !m_src_defined) { return; }
    erf_actuator::ScopedPhase timed(m_timers, erf_actuator::PhaseTimers::AddSources);
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

std::vector<MovingBodies::PointGroup>
MovingBodies::turbine_point_groups (int i, Real dx) const
{
    const erf_openfast::TurbineState& t = m_driver->turbines()[i];
    const MovingBodyInputs& b = m_turbine_in[i];
    const KernelWidths w = b.kernel_widths(dx);
    if (m_turbine_mode[i] == "none") { return {PointGroup{t.force_pos, w.rotor}}; }
    // OpenFAST's force nodes: the hub, the blades' nodes, then the tower's
    const std::size_t n_rotor = std::min(t.force_pos.size(), static_cast<std::size_t>(3 * (1 + t.num_blades * t.num_force_pts_blade)));
    std::vector<PointGroup> g;
    g.push_back({std::vector<Real>(t.force_pos.begin(), t.force_pos.begin() + n_rotor), w.rotor});
    if (t.force_pos.size() > n_rotor) {
        g.push_back({std::vector<Real>(t.force_pos.begin() + n_rotor, t.force_pos.end()), w.tower});
    }
    if (b.nacelle_cd > 0.0) { g.push_back({{t.hub_pos[0], t.hub_pos[1], t.hub_pos[2]}, w.nacelle}); }
    return g;
}
#endif

void
MovingBodies::supply_velocities (double time, const MultiFab& U, const MultiFab& V, const MultiFab& W,
                                 const MultiFab* z_phys_nd, const Geometry& geom)
{
    BL_PROFILE("MovingBodies::supply_velocities()");
    amrex::ignore_unused(time);
    // every body's points on its owner, sampled on the ranks whose boxes hold them: a turbine's velocity
    // nodes (where its sampling mode puts them), a disk's upstream sampling points followed by its disk points
    const int nturb = num_turbines();
    const int nbody = m_exchange.num_bodies();
    std::vector<std::vector<Real>> pts(nbody), vel(nbody);
    std::vector<Real> margin(nbody, Real(0.0));
#ifdef ERF_USE_OPENFAST
    std::vector<int> n_nodes(nturb, 0);
    for (int i = 0; i < nturb; ++i) {
        if (!m_driver->is_owner(i)) { continue; }
        pts[i] = sampling_positions(i);
        n_nodes[i] = static_cast<int>(pts[i].size() / 3);
        // with sampling = upstream the nodes are sampled ahead of the rotor, so the nacelle drag needs
        // the hub itself (the first velocity node), where its drag point is
        const MovingBodyInputs& b = m_turbine_in[i];
        if (m_turbine_mode[i] != "none" && b.nacelle_cd > 0.0 && b.sampling == "upstream") {
            const auto& t = m_driver->turbines()[i];
            pts[i].insert(pts[i].end(), {t.vel_pos[0], t.vel_pos[1], t.vel_pos[2]});
        }
        margin[i] = exchange_margin(i, geom);
    }
#endif
    for (std::size_t k = 0; k < m_disks.size(); ++k) {
        const int b = nturb + static_cast<int>(k);
        if (!m_exchange.owns(b)) { continue; }
        const auto& ds = m_disks[k]->sample_points();
        const auto& dd = m_disks[k]->disk_points();
        pts[b] = ds;
        pts[b].insert(pts[b].end(), dd.begin(), dd.end());
        margin[b] = exchange_margin(b, geom);
    }
    if (m_in.has_prescribed_velocity) {
        for (int b = 0; b < nbody; ++b) {
            if (!m_exchange.owns(b)) { continue; }
            vel[b].resize(pts[b].size());
            for (std::size_t p = 0; p < vel[b].size() / 3; ++p) {
                for (int c = 0; c < 3; ++c) { vel[b][3*p+c] = m_in.prescribed_velocity[c]; }
            }
        }
        amrex::ignore_unused(U, V, W, z_phys_nd);
    } else if (nbody > 0) {
        set_hubs();
        m_exchange.sample_velocity(U, V, W, z_phys_nd, geom, pts, margin, vel);
        report_participants(m_exchange, m_exchange_names, m_reported_participants);
    }
#ifdef ERF_USE_OPENFAST
    if (nturb > 0) {
        if (!m_fllc_built) { build_fllc(geom); }
        for (int i = 0; i < nturb; ++i) {
            if (!m_driver->is_owner(i)) { continue; }
            m_turbine_vel[i].assign(vel[i].begin(), vel[i].begin() + 3 * n_nodes[i]);
            {
                std::vector<Real> uvw = m_turbine_vel[i];
                if (m_turbine_in[i].sampling == "disk_corrected") { apply_disk_correction(i, uvw, geom); }
                if (!m_fllc[i].empty()) { apply_fllc(i, time, uvw); }
                m_driver->set_node_velocities(i, uvw);
            }
            // the nacelle drag from the velocity at the hub (the first velocity node, or the point added for
            // it), corrected for the drag point's own kernel; recorded on the structure for the diagnostics
            const MovingBodyInputs& b = m_turbine_in[i];
            std::array<Real,3> f_nac{{0.0, 0.0, 0.0}};
            if (m_turbine_mode[i] != "none" && b.nacelle_cd > 0.0) {
                const std::size_t h = (b.sampling == "upstream") ? 3 * static_cast<std::size_t>(n_nodes[i]) : 0;
                if (vel[i].size() >= h + 3) {
                    const std::array<Real,3> u_hub{{vel[i][h], vel[i][h+1], vel[i][h+2]}};
                    const Real eps = b.kernel_widths(static_cast<Real>(geom.CellSize(0))).nacelle;
                    const auto u_free = erf_actuator::nacelle_corrected_velocity(u_hub, b.nacelle_cd, b.nacelle_area, eps);
                    const auto f_fluid = erf_actuator::nacelle_drag_force(u_free, b.air_density, b.nacelle_cd, b.nacelle_area);
                    for (int d = 0; d < 3; ++d) { f_nac[d] = -f_fluid[d]; }
                }
            }
            m_driver->set_nacelle_force(i, f_nac);
        }
    }
#endif
    // each disk's owner: the free stream and the thrust from the velocities at its sampling and disk points
    for (std::size_t k = 0; k < m_disks.size(); ++k) {
        const int b = nturb + static_cast<int>(k);
        if (!m_exchange.owns(b)) { continue; }
        const std::size_t ns = m_disks[k]->sample_points().size();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(vel[b].size() == ns + m_disks[k]->disk_points().size(),
                                         "MovingBodies::supply_velocities: one velocity per disk point");
        m_disks[k]->update(std::vector<Real>(vel[b].begin(), vel[b].begin() + static_cast<std::ptrdiff_t>(ns)),
                           std::vector<Real>(vel[b].begin() + static_cast<std::ptrdiff_t>(ns), vel[b].end()));
    }
}

void
MovingBodies::check_levels (int finest_level, const Vector<BoxArray>& grids, const Vector<Geometry>& geoms) const
{
    BL_PROFILE("MovingBodies::check_levels()");
    erf_actuator::ScopedPhase timed(m_timers, erf_actuator::PhaseTimers::CheckLevels);
    if (m_anchor > finest_level) {
        Abort("erf.moving_bodies: the bodies live on level " + std::to_string(m_anchor) + ", but the run has levels 0 to " +
              std::to_string(finest_level) + " only now (did a regrid remove it?); the bodies would not be stepped");
    }
    if (finest_level == m_anchor || !m_ground_set) { return; }
    // what the bodies force, with each kernel's reach, in metres (centre x y z, half-size): each rotor's or
    // disk's extent about its hub or centre with the rotor's kernel, and every force point of a turbine
    // (blades, tower, hub) with the kernel it is spread with
    const Geometry& ga = geoms[m_anchor];
    const Real dxa = static_cast<Real>(ga.CellSize(0));
    std::vector<std::pair<std::string, std::array<Real,4>>> extents;
#ifdef ERF_USE_OPENFAST
    for (int i = 0; i < static_cast<int>(m_driver->turbines().size()); ++i) {
        // a turbine's points are known to its owner alone, which checks them
        if (!m_driver->is_owner(i)) { continue; }
        const auto& t = m_driver->turbines()[i];
        // the widths in cells, so each reach is (3 eps) dx + dx as for one shared width
        const Real reach = Real(3.0) * m_turbine_in[i].kernel_widths(1.0).rotor * dxa + dxa;
        extents.push_back({t.name, {t.hub_pos[0], t.hub_pos[1], t.hub_pos[2], erf_actuator::tip_radius(t) + reach}});
        for (const auto& g : turbine_point_groups(i, 1.0)) {
            for (std::size_t p = 0; p + 2 < g.pos.size(); p += 3) {
                extents.push_back({t.name, {g.pos[p], g.pos[p+1], g.pos[p+2], Real(3.0) * g.eps * dxa + dxa}});
            }
        }
    }
#endif
    for (std::size_t k = 0; k < m_disks.size(); ++k) {
        const auto c = m_disks[k]->center();
        extents.push_back({m_disks[k]->name(), {c[0], c[1], c[2], m_disks[k]->radius() + Real(3.0) * m_disk_epsilon[k] * dxa + dxa}});
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
    BL_PROFILE("MovingBodies::check_coverage()");
    m_audited_grids = level_grids;
#ifdef ERF_USE_OPENFAST
    // the cells under the rotors may have changed with the grids: the tip cell heights are searched again
    m_tip_cells_valid = false;
#endif
    // on a fitted or stretched mesh a point's cell is not its height over dz: the kernel must be on the
    // grids over every height its reach may span (mesh_z_bounds), a sampled point anywhere in its column
    erf_actuator::ZBounds zb;
    if (z_phys_nd != nullptr) { zb = erf_actuator::mesh_z_bounds(*z_phys_nd, geom); }
    const bool fitted = (z_phys_nd != nullptr);
    // a level with a box that does not reach the ground has no global height bounds: the whole column
    // (the spreading bounds each rank's own boxes by their table instead)
    const erf_actuator::ZBounds* zbp = (fitted && zb.all_ground) ? &zb : nullptr;
    const Real dx0 = static_cast<Real>(geom.CellSize(0));
    // the level's boxes as the coverage tests take them, built once for every body
    const erf_actuator::CoverageGrids cover(level_grids, geom);
    // every body is checked by its owner (a turbine's points are known to it alone; every rank knows a disk's
    // and a wake line's), each rank stopping at its first failure; one reduction tells every rank
    std::string failed;
    int key = 0;   // the failing body: turbine i, disk num_turbines() + k, the lines behind body b after all the bodies
#ifdef ERF_USE_OPENFAST
    // every node of every turbine, with the kernel's reach, must lie on the anchor level's grids, and the
    // whole swept disk with it (the rings of a disk, the blades of a line as they turn and the rotor as
    // it yaws, while the grids stay as they are): on a refined level those are not the whole domain, and
    // the sampler and spreader read only the level's own cells
    const auto& turbs = m_driver->turbines();
    for (int i = 0; i < static_cast<int>(turbs.size()) && failed.empty(); ++i) {
        if (!m_driver->is_owner(i)) { continue; }
        const auto& t = turbs[i];
        const Real reach = Real(3.0) * m_turbine_in[i].kernel_widths(dx0).rotor + dx0;
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
        // every force node with the reach of the kernel it is spread with, then the swept disk with the rotor's
        std::string outside;
        Real missed = reach;
        bool covered = true;
        for (const auto& g : turbine_point_groups(i, dx0)) {
            if (!covered) { break; }
            missed = Real(3.0) * g.eps + dx0;
            covered = erf_actuator::points_covered_by(cover, geom, g.pos, missed, outside,
                                                      fitted ? erf_actuator::CoverZ::Column : erf_actuator::CoverZ::Reach, zbp);
        }
        if (covered) {
            missed = reach;
            covered = erf_actuator::points_covered_by(cover, geom, disk, reach, outside,
                                                      fitted ? erf_actuator::CoverZ::Column : erf_actuator::CoverZ::Reach, zbp);
        }
        if (covered) {
            missed = static_cast<Real>(geom.CellSize(0));
            covered = erf_actuator::points_covered_by(cover, geom, sampling_positions(i), geom.CellSize(0), outside,
                                                      fitted ? erf_actuator::CoverZ::Footprint : erf_actuator::CoverZ::Reach);
        }
        if (!covered) {
            key = i;
            failed = "erf.moving_bodies." + t.name + ": node " + outside + " (with the kernel reach " + std::to_string(missed) +
                     " m) is not covered by the grids of level " + std::to_string(m_anchor) +
                     "; enlarge the refinement region around the rotor (the force lives on the anchor level only, so that "
                     "level must hold the whole kernel)";
        }
    }
#endif
    for (std::size_t k = 0; k < m_disks.size() && failed.empty(); ++k) {
        if (!m_exchange.owns(num_turbines() + static_cast<int>(k))) { continue; }
        const auto& d = m_disks[k];
        const Real reach = Real(3.0) * (m_disk_epsilon[k] * dx0) + dx0;
        std::string outside;
        if (!erf_actuator::points_covered_by(cover, geom, d->disk_points(), reach, outside,
                                             fitted ? erf_actuator::CoverZ::Column : erf_actuator::CoverZ::Reach, zbp) ||
            !erf_actuator::points_covered_by(cover, geom, d->sample_points(), geom.CellSize(0), outside,
                                             fitted ? erf_actuator::CoverZ::Footprint : erf_actuator::CoverZ::Reach)) {
            key = num_turbines() + static_cast<int>(k);
            failed = "erf.moving_bodies." + d->name() + ": point " + outside + " (with the kernel reach) is not covered by the grids of level " +
                     std::to_string(m_anchor) + "; enlarge the refinement region around the disk";
        }
    }
    // the lines behind rotor b are body b of m_exchange
    for (int b = 0; b < static_cast<int>(m_wakes.size()) && failed.empty(); ++b) {
        if (!m_exchange.owns(b)) { continue; }
        const auto& wl = m_wakes[b];
        std::string outside;
        if (!erf_actuator::points_covered_by(cover, geom, wl->positions(), geom.CellSize(0), outside,
                                             fitted ? erf_actuator::CoverZ::Footprint : erf_actuator::CoverZ::Reach)) {
            key = m_exchange.num_bodies() + b;
            failed = "erf.moving_bodies.wake: the sampling point " + outside + " of " + wl->name() + " is not covered by the grids of level " +
                     std::to_string(m_anchor) + " after a regrid; enlarge the refinement region behind the rotor";
        }
    }
    {
        const std::string first = erf_actuator::first_failure(failed, key);
        if (!first.empty()) { Abort(first); }
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
    const Real eps = b.kernel_widths(static_cast<Real>(geom.CellSize(0))).rotor;
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
        AllPrint() << "Warning: erf.moving_bodies." << t.name << ": the thrust coefficient on the corrected free stream reached the "
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
MovingBodies::write_turbine_logs (double time, bool solution0)
{
    BL_PROFILE("MovingBodies::write_turbine_logs()");
    // every turbine's ERF-side and flow logs from its owner; the solution-0 row starts the files
    std::vector<int> all(num_turbines());
    for (int i = 0; i < num_turbines(); ++i) { all[i] = i; }
    write_series("erf", erf_openfast::OpenFASTDriver::erf_log_columns(), time, solution0, all,
                 erf_openfast::OpenFASTDriver::erf_log_size, [this] (int i, Real* row) { m_driver->erf_log_row(i, row); return true; });
    write_series("flow", erf_openfast::OpenFASTDriver::flow_log_columns(), time, solution0, all,
                 erf_openfast::OpenFASTDriver::flow_log_size, [this] (int i, Real* row) { m_driver->flow_log_row(i, row); return true; });
}

void
MovingBodies::write_correction_diagnostics (double time, bool first)
{
    BL_PROFILE("MovingBodies::write_correction_diagnostics()");
    std::vector<int> bodies;
    for (int i = 0; i < num_turbines(); ++i) {
        if (m_turbine_in[i].sampling == "disk_corrected") { bodies.push_back(i); }
    }
    if (bodies.empty()) { return; }
    // the owner made the correction and writes its row
    write_series("correction", "u_disk,ct,a,ct_prime,M,u_inf,factor,gain,relax", time, first && !m_resumed && !m_corr_written,
                 bodies, 9, [this] (int i, Real* row) {
                     const erf_actuator::DiskCorrection& c = m_corr_last[i];
                     const Real v[9] = {c.u_disk, c.ct, c.a, c.ct_prime, c.M, c.u_inf, c.factor, c.gain, c.relax};
                     std::copy(v, v + 9, row);
                     return true;
                 });
    m_corr_written = true;
}

void
MovingBodies::audit_turbines_setup (const MultiFab* z_phys_nd, const Geometry& geom)
{
    BL_PROFILE("MovingBodies::audit_turbines_setup()");
    m_audited_setup = true;
    const auto& turbs = m_driver->turbines();
    const int nturb = static_cast<int>(turbs.size());
    if (nturb == 0) { return; }
    std::array<Real,3> plo, phi, dx;
    std::array<int,3> per;
    for (int d = 0; d < 3; ++d) {
        plo[d] = static_cast<Real>(geom.ProbLo(d)); phi[d] = static_cast<Real>(geom.ProbHi(d));
        dx[d] = static_cast<Real>(geom.CellSize(d)); per[d] = geom.isPeriodic(d) ? 1 : 0;
    }
    // every rotor's hub, axis and tip radius from its owner, in one gather: the terrain under the hubs (a
    // collective) for the geometry audits, the swept discs of every pair for the overlap audit
    const std::vector<Real> disc = turbine_table(7, false, [this] (int i, Real* row) {
        const auto& t = m_driver->turbines()[i];
        for (int d = 0; d < 3; ++d) { row[d] = t.hub_pos[d]; row[3+d] = t.hub_axis[d]; }
        row[6] = erf_actuator::tip_radius(t);
    });
    std::vector<Real> ground;
    {
        std::vector<Real> pos(3 * static_cast<std::size_t>(nturb));
        for (int i = 0; i < nturb; ++i) {
            for (int d = 0; d < 3; ++d) { pos[3*i+d] = disc[7 * static_cast<std::size_t>(i) + d]; }
        }
        erf_actuator::terrain_heights(z_phys_nd, geom, pos, ground);
    }
    // each owner audits its turbines, each model file read once for both audits (m_audit_files); the I/O rank
    // prints the reports in turbine order
    erf_openfast::ModelFiles& files = m_audit_files;
    std::vector<std::string> text(nturb);
    std::vector<int> fatal(nturb, 0);
    for (int i = 0; i < nturb; ++i) {
        if (!m_driver->is_owner(i)) { continue; }
        const MovingBodyInputs& b = m_turbine_in[i];
        const KernelWidths w = b.kernel_widths(static_cast<Real>(geom.CellSize(0)));
        const Real eps = w.rotor;
        if (b.sampling == "disk_corrected") {
            // the filtered-disk factor was derived for filter widths up to about 1.25 rotor radii
            const Real ratio = erf_actuator::filter_width_from_eps(eps) / erf_actuator::tip_radius(turbs[i]);
            std::ostringstream line;
            line << "erf.moving_bodies." << turbs[i].name << ": disk_corrected sampling, filter width sqrt(6) eps = "
                 << erf_actuator::filter_width_from_eps(eps) << " m, " << ratio << " rotor radii\n";
            text[i] += line.str();
            if (ratio > Real(1.25)) {
                Warning("erf.moving_bodies." + turbs[i].name + ": the filter width is " + std::to_string(ratio) +
                        " rotor radii; the filtered-disk correction was derived for widths up to about 1.25 radii");
            }
        }
        std::vector<erf_openfast::AuditFinding> f = erf_openfast::audit_model(files, b, CONST_GRAV);
        const auto fg = erf_openfast::audit_geometry(turbs[i], b, w, plo, phi, dx, per, ground[i]);
        f.insert(f.end(), fg.begin(), fg.end());
        bool bad = false;
        text[i] += erf_openfast::format_audit(turbs[i].name + " (model and geometry)", f, bad);
        fatal[i] = bad ? 1 : 0;
    }
    bool any_fatal = print_turbine_texts(text, fatal);
    // the swept discs of every pair, audited on the I/O rank
    int overlap_fatal = 0;
    if (ParallelDescriptor::IOProcessor()) {
        std::vector<erf_openfast::RotorDisc> discs;
        for (int i = 0; i < nturb; ++i) {
            const Real* r = &disc[7 * static_cast<std::size_t>(i)];
            discs.push_back({turbs[i].name, {{r[0], r[1], r[2]}}, {{r[3], r[4], r[5]}}, r[6]});
        }
        const auto overlap = erf_openfast::audit_overlap(discs);
        bool bad = false;
        if (!overlap.empty()) { erf_openfast::report_audit("the farm", overlap, bad); }
        overlap_fatal = bad ? 1 : 0;
    }
    ParallelDescriptor::ReduceIntMax(overlap_fatal);
    any_fatal = any_fatal || (overlap_fatal != 0);
    if (any_fatal) {
        Abort("erf.moving_bodies: the OpenFAST input audit found errors (listed above); fix them (mode = none clears only "
              "the CompAero finding: a turbine whose loads are not wanted in the flow)");
    }
}

void
MovingBodies::audit_turbines_flow (const MultiFab& cons, const MultiFab* z_phys_nd, const Geometry& geom)
{
    BL_PROFILE("MovingBodies::audit_turbines_flow()");
    m_audited_flow = true;
    const auto& turbs = m_driver->turbines();
    const int nturb = static_cast<int>(turbs.size());
    if (nturb == 0) { return; }
    // ERF's density at every hub (each hub from its owner, sampled by every rank alike: once, at start-up)
    const std::vector<Real> hubs = turbine_table(3, false, [this] (int i, Real* row) {
        for (int d = 0; d < 3; ++d) { row[d] = m_driver->turbines()[i].hub_pos[d]; }
    });
    std::vector<Real> rho_hub;
    erf_actuator::sample_cell_scalar(cons, Rho_comp, z_phys_nd, geom, hubs, rho_hub);
    erf_openfast::ModelFiles& files = m_audit_files;   // the files the setup audit read, kept for this one
    std::vector<std::string> text(nturb);
    std::vector<int> fatal(nturb, 0);
    for (int i = 0; i < nturb; ++i) {
        if (!m_driver->is_owner(i)) { continue; }
        const MovingBodyInputs& b = m_turbine_in[i];
        std::vector<erf_openfast::AuditFinding> f;
        Real rho_model = 0.0;
        if (erf_openfast::openfast_air_density(files, b.fst_file, rho_model)) {
            f = erf_openfast::audit_density(turbs[i].name, rho_model, rho_hub[i], m_in.density_tolerance);
            if (f.empty()) {
                f.push_back({false, "ERF's density at the hub is " + std::to_string(rho_hub[i]) + " kg/m^3, the model's " +
                                    std::to_string(rho_model) + " (within " + std::to_string(100.0 * m_in.density_tolerance) + " %)"});
            }
        } else {
            f.push_back({false, "ERF's density at the hub is " + std::to_string(rho_hub[i]) + " kg/m^3; the model gives none to compare with"});
        }
        // the velocity sampled at the hub (the first point) before any correction
        const std::vector<Real>& uvw = m_turbine_vel[i];
        if (uvw.size() >= 3) {
            const auto ff = erf_openfast::audit_facing(turbs[i], {{uvw[0], uvw[1], uvw[2]}});
            f.insert(f.end(), ff.begin(), ff.end());
        }
        bool bad = false;
        text[i] = erf_openfast::format_audit(turbs[i].name + " (density and wind)", f, bad);
        fatal[i] = bad ? 1 : 0;
    }
    m_audit_files = erf_openfast::ModelFiles();   // the start-up audits are done: let the lines go
    if (print_turbine_texts(text, fatal)) {
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
    BL_PROFILE("MovingBodies::build_fllc()");
    const auto& turbs = m_driver->turbines();
    m_fllc.clear();
    m_fllc.resize(turbs.size());
    for (std::size_t i = 0; i < turbs.size(); ++i) {
        const MovingBodyInputs& b = m_turbine_in[i];
        // the owner corrects the velocities it hands OpenFAST; the other ranks keep no correction
        if (!b.fllc || !m_driver->is_owner(static_cast<int>(i))) { continue; }
        // the blades' own kernel: the correction is for the width the line is spread with
        const Real eps = b.kernel_widths(static_cast<Real>(geom.CellSize(0))).rotor;
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
            // read by the owner alone (the file is the turbine's own)
            const std::string fname = m_fllc_state_dir + "/" + t.name + "_fllc.dat";
            std::ifstream in(fname);
            if (in) {
                for (auto& f : m_fllc[i]) {
                    if (!f->read_state(in)) { Abort("the lifting-line correction checkpoint '" + fname + "' lacks " + f->name()); }
                }
                AllPrint() << "  " << t.name << ": lifting-line correction restored from " << fname << "\n";
            }
        }
        AllPrint() << "erf.moving_bodies." << t.name << ": lifting-line correction on " << t.num_blades << " blades, kernel "
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
    BL_PROFILE("MovingBodies::write_fllc_diagnostics()");
    std::vector<int> bodies;
    for (int i = 0; i < num_turbines(); ++i) {
        if (m_turbine_in[i].fllc) { bodies.push_back(i); }
    }
    if (!bodies.empty()) {
        // built on the owner alone, which writes the row
        write_series("fllc", "du_max,du_rms", time, first && !m_resumed && !m_fllc_written, bodies, 2, [this] (int i, Real* row) {
            if (m_fllc[i].empty()) { return false; }
            Real du_max = 0.0, sum2 = 0.0;
            for (const auto& f : m_fllc[i]) {
                du_max = std::max(du_max, f->max_correction());
                sum2 += f->rms_correction() * f->rms_correction();
            }
            row[0] = du_max;
            row[1] = std::sqrt(sum2 / static_cast<Real>(m_fllc[i].size()));
            return true;
        });
    }
    m_fllc_written = true;
}
#endif
