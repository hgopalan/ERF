#include "ERF_OpenFASTDriver.H"

#include "ERF_DiagnosticsLog.H"
#include "ERF_OpenFASTAudit.H"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <iomanip>
#include <limits>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

using namespace amrex;

namespace erf_openfast {

int
substep_count (double dt_cfd, double dt_fast, std::string& err)
{
    err.clear();
    if (!(dt_fast > 0.0)) {
        err = "OpenFAST reported a non-positive time step";
        return 0;
    }
    if (!(dt_cfd > 0.0)) {
        err = "the ERF time step must be positive";
        return 0;
    }
    const double ratio = dt_cfd / dt_fast;
    const int n = static_cast<int>(std::lround(ratio));
    // the ERF step is a whole multiple of the OpenFAST step up to roundoff in the ratio
    if (n < 1 || std::abs(ratio - n) > 1.0e-8 * ratio) {
        err = "the ERF step on the bodies' level (erf.fixed_dt over the sub-cycling ratios) = " + std::to_string(dt_cfd) +
              " s is not a positive whole multiple of the OpenFAST dt = " +
              std::to_string(dt_fast) + " (ratio " + std::to_string(ratio) + ")";
        return 0;
    }
    return n;
}


std::string
check_induction (const std::string& fst_file, bool want_off)
{
    ModelFiles files;
    return check_induction(files, fst_file, want_off);
}

std::string
check_induction (ModelFiles& files, const std::string& fst_file, bool want_off)
{
    const std::string path = openfast_module_path(files, fst_file, "AeroFile");
    if (path.empty()) { return {}; }
    const std::string comp = openfast_value(files, fst_file, "CompAero");
    if (comp == "0") { return {}; }
    std::string wake = openfast_value(files, path, "Wake_Mod");
    if (wake.empty()) { wake = openfast_value(files, path, "WakeMod"); }
    if (wake.empty()) {
        return "the AeroDyn file '" + path + "' named by '" + fst_file + "' has no Wake_Mod (or WakeMod) line";
    }
    if (want_off && wake != "0") {
        return "the AeroDyn file '" + path + "' sets Wake_Mod = " + wake +
               "; set it to 0: the velocities ERF samples at the rotor already contain its induction "
               "once its loads act on the flow (or sample the free stream with sampling = upstream)";
    }
    if (!want_off && wake == "0") {
        return "the AeroDyn file '" + path + "' sets Wake_Mod = 0 but sampling = upstream or disk_corrected feeds OpenFAST the free "
               "stream, so its own induction model is needed: set Wake_Mod = 1 (BEMT)";
    }
    return {};
}

std::string
check_induction_off (const std::string& fst_file) { return check_induction(fst_file, true); }

std::string
check_tower_shadow_off (const std::string& fst_file)
{
    ModelFiles files;
    return check_tower_shadow_off(files, fst_file);
}

std::string
check_tower_shadow_off (ModelFiles& files, const std::string& fst_file)
{
    const std::string path = openfast_module_path(files, fst_file, "AeroFile");
    if (path.empty()) { return {}; }
    const std::string comp = openfast_value(files, fst_file, "CompAero");
    if (comp == "0") { return {}; }
    const std::string shadow = openfast_value(files, path, "TwrShadow");
    if (shadow.empty() || shadow == "0") { return {}; }
    return "the AeroDyn file '" + path + "' sets TwrShadow = " + shadow +
           "; with the tower's loads in the flow its wake reaches the blades through the sampled "
           "velocities, so AeroDyn's tower-shadow correction counts it twice: set TwrShadow = 0";
}

namespace {

// every rank's text, on every rank, in rank order (one exchange of the lengths, one of the characters)
std::vector<std::string>
all_gather_texts (const std::string& mine)
{
    const int np = ParallelDescriptor::NProcs();
    std::vector<std::string> all(np);
#ifdef AMREX_USE_MPI
    if (np > 1) {
        MPI_Comm comm = ParallelDescriptor::Communicator();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mine.size() < static_cast<std::size_t>(std::numeric_limits<int>::max() / np),
                                         "gather_model_facts: the model texts are too long to gather");
        int len = static_cast<int>(mine.size());
        std::vector<int> lens(np, 0), offsets(np, 0);
        MPI_Allgather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, comm);
        for (int r = 1; r < np; ++r) { offsets[r] = offsets[r-1] + lens[r-1]; }
        std::vector<char> chars(static_cast<std::size_t>(offsets[np-1] + lens[np-1]) + 1);
        MPI_Allgatherv(mine.data(), len, MPI_CHAR, chars.data(), lens.data(), offsets.data(), MPI_CHAR, comm);
        for (int r = 0; r < np; ++r) { all[r].assign(chars.data() + offsets[r], static_cast<std::size_t>(lens[r])); }
        return all;
    }
#endif
    all[0] = mine;
    return all;
}

// one text as its length, a space and its characters, so that any text (spaces, new lines) reads back whole
void put_text (std::ostringstream& out, const std::string& t) { out << t.size() << ' ' << t; }

std::string get_text (std::istringstream& in)
{
    std::size_t n = 0;
    in >> n;
    in.get();   // the space after the length
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(in.good() && n < (std::size_t(1) << 24), "gather_model_facts: a malformed text length");
    std::string t(n, '\0');
    in.read(&t[0], static_cast<std::streamsize>(n));
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<std::size_t>(in.gcount()) == n, "gather_model_facts: a truncated text");
    return t;
}

} // namespace

std::vector<ModelFacts>
gather_model_facts (const std::vector<std::string>& paths)
{
    BL_PROFILE("erf_openfast::gather_model_facts()");
    const int npath = static_cast<int>(paths.size());
    std::vector<ModelFacts> facts(npath);
    if (npath == 0) { return facts; }
    const int np = ParallelDescriptor::NProcs();
    const int me = ParallelDescriptor::MyProc();
    // this rank's models, each of its files read once, their texts packed as: index, then three texts
    ModelFiles files;
    std::vector<Real> numbers(2 * static_cast<std::size_t>(npath), Real(0.0));
    std::ostringstream mine;
    for (int p = me; p < npath; p += np) {
        ModelFacts& f = facts[p];
        f.induction_off = check_induction(files, paths[p], true);
        f.induction_on = check_induction(files, paths[p], false);
        f.tower_shadow = check_tower_shadow_off(files, paths[p]);
        numbers[2*p] = model_rotor_radius(files, paths[p]);
        numbers[2*p+1] = model_time_step(files, paths[p]);
        mine << p << ' ';
        put_text(mine, f.induction_off);
        put_text(mine, f.induction_on);
        put_text(mine, f.tower_shadow);
        mine << ' ';
    }
    // each number comes from its model's rank alone, zeros from the others: the sum is that number exactly
    ParallelDescriptor::ReduceRealSum(numbers.data(), 2 * npath);
    const std::vector<std::string> all = all_gather_texts(mine.str());
    for (int r = 0; r < np; ++r) {
        if (r == me) { continue; }
        std::istringstream in(all[r]);
        int p = -1;
        while (in >> p) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(p >= 0 && p < npath && p % np == r, "gather_model_facts: a model from the wrong rank");
            in.get();
            facts[p].induction_off = get_text(in);
            facts[p].induction_on = get_text(in);
            facts[p].tower_shadow = get_text(in);
        }
    }
    for (int p = 0; p < npath; ++p) {
        facts[p].rotor_radius = numbers[2*p];
        facts[p].time_step = numbers[2*p+1];
    }
    return facts;
}

OpenFASTDriver::OpenFASTDriver (const std::vector<MovingBodyInputs>& bodies, const std::map<std::string, ModelFacts>* facts)
{
    BL_PROFILE("OpenFASTDriver::OpenFASTDriver()");
    // a farm shares a few models: each distinct model file (and check) is read once
    ModelFiles files;
    std::map<std::pair<std::string,bool>, std::string> induction;
    for (const MovingBodyInputs& b : bodies) {
        TurbineState t;
        t.name = b.name;
        t.fst_file = b.fst_file;
        t.output_root = b.output_root;
        t.base_pos = b.base_pos;
        t.num_force_pts_blade = b.num_force_points_blade;
        t.num_force_pts_tower = b.num_force_points_tower;
        if (b.mode != "none") {
            const auto key = std::make_pair(b.fst_file, b.sampling == "disk");
            auto it = induction.find(key);
            if (it == induction.end()) {
                std::string err;
                if (facts != nullptr) {
                    const auto f = facts->find(key.first);
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f != facts->end(), "OpenFASTDriver: no model facts for " + key.first);
                    err = key.second ? f->second.induction_off : f->second.induction_on;
                } else {
                    err = check_induction(files, key.first, key.second);
                }
                it = induction.emplace(key, err).first;
            }
            const std::string& err = it->second;
            if (!err.empty()) { Abort("erf.moving_bodies." + b.name + ": " + err); }
        }
        m_turb.push_back(t);
    }
}

void
OpenFASTDriver::set_owners (const std::vector<int>& owners)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_initialized, "OpenFASTDriver::set_owners: the turbines are already initialised");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(owners.size() == m_turb.size(), "OpenFASTDriver::set_owners: one owner rank per turbine");
    m_num_local = 0;
    for (std::size_t i = 0; i < m_turb.size(); ++i) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(owners[i] >= 0 && owners[i] < ParallelDescriptor::NProcs(),
                                         "OpenFASTDriver::set_owners: an owner is not a rank of this run");
        m_turb[i].owner_rank = owners[i];
        // OpenFAST numbers the instances of a process from 0, in turbine order
        m_turb[i].tid_local = (owners[i] == ParallelDescriptor::MyProc()) ? m_num_local++ : -1;
    }
    m_owners_set = true;
}

OpenFASTDriver::~OpenFASTDriver ()
{
    if (m_initialized && m_num_local > 0) {
        // FAST_End closes each turbine as OpenFAST's own driver does: it writes the binary output
        // (OutFileFmt 2 or 3), closes the text output and ends the modules (a ServoDyn controller library)
        for (const auto& t : m_turb) {
            if (t.owner_rank != ParallelDescriptor::MyProc()) { continue; }
            int tid = t.tid_local;
            bool stop = false;
            FAST_End(&tid, &stop);
        }
        int err_stat = ErrID_None;
        char err_msg[INTERFACE_STRING_LENGTH];
        FAST_DeallocateTurbines(&err_stat, err_msg);
        // destructors do not abort; a failure here only leaks the OpenFAST buffers
    }
}

bool
OpenFASTDriver::is_owner (int i) const
{
    return m_turb[i].owner_rank == ParallelDescriptor::MyProc();
}

void
OpenFASTDriver::fast_check (int err_stat, const char* err_msg, const std::string& where) const
{
    if (err_stat >= ErrID_Fatal) {
        Abort("OpenFAST failed in " + where + ": " + std::string(err_msg));
    } else if (err_stat >= ErrID_Warn) {
        // printed by the rank that made the call (it owns the turbine), not only the I/O rank
        AllPrint() << "OpenFAST warning in " << where << ": " << err_msg << "\n";
    }
}

void
OpenFASTDriver::allocate ()
{
    if (m_num_local > 0) {
        int err_stat = ErrID_None;
        char err_msg[INTERFACE_STRING_LENGTH];
        int n = m_num_local;
        FAST_AllocateTurbines(&n, &err_stat, err_msg);
        fast_check(err_stat, err_msg, "FAST_AllocateTurbines");
    }
}

// after FAST_ExtInfw_Init or FAST_ExtInfw_Restart on the owner rank: the node counts, the
// substep count and the first pull of the positions
void
OpenFASTDriver::finish_setup (TurbineState& t, double dt_cfd)
{
    if (t.to_cfd.fx_Len == 1 + t.num_blades * t.num_force_pts_blade) {
        // no tower in the model: OpenFAST allocated no tower force nodes
        t.num_force_pts_tower = 0;
    }
    t.num_vel_nodes = t.to_cfd.pxVel_Len;
    t.num_force_nodes = t.to_cfd.fx_Len;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        t.num_force_nodes == 1 + t.num_blades * t.num_force_pts_blade + t.num_force_pts_tower,
        "OpenFAST force-node count does not match hub + blades + tower for " + t.name);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        t.from_cfd.u_Len == t.num_vel_nodes && t.from_cfd.v_Len == t.num_vel_nodes && t.from_cfd.w_Len == t.num_vel_nodes &&
        t.to_cfd.pyVel_Len == t.num_vel_nodes && t.to_cfd.pzVel_Len == t.num_vel_nodes &&
        t.to_cfd.pxForce_Len == t.num_force_nodes && t.to_cfd.pyForce_Len == t.num_force_nodes &&
        t.to_cfd.pzForce_Len == t.num_force_nodes && t.to_cfd.fy_Len == t.num_force_nodes && t.to_cfd.fz_Len == t.num_force_nodes &&
        t.to_cfd.xdotForce_Len == t.num_force_nodes && t.to_cfd.ydotForce_Len == t.num_force_nodes &&
        t.to_cfd.zdotForce_Len == t.num_force_nodes && t.to_cfd.forceNodesChord_Len == t.num_force_nodes,
        "OpenFAST ExtInfw array lengths are inconsistent for " + t.name);

    std::string err;
    t.num_substeps = substep_count(dt_cfd, t.dt_fast, err);
    if (t.num_substeps == 0) { Abort("erf.moving_bodies." + t.name + ": " + err); }
    pull_from_fast(t);
}

void
OpenFASTDriver::init (double dt_cfd, double t_max)
{
    BL_PROFILE("OpenFASTDriver::init()");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_initialized, "OpenFASTDriver::init called twice");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_owners_set, "OpenFASTDriver::init: set_owners() first");
    int err_stat = ErrID_None;
    char err_msg[INTERFACE_STRING_LENGTH];
    allocate();

    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        TurbineState& t = m_turb[i];
        if (is_owner(i)) {
            int turb_id = i;
            char out_root[INTERFACE_STRING_LENGTH];
            float pos[3] = {static_cast<float>(t.base_pos[0]),
                            static_cast<float>(t.base_pos[1]),
                            static_cast<float>(t.base_pos[2])};
            int abort_lev = ErrID_Fatal;
            int inflow_type = 0;
            int node_cluster_type = 0;   // uniform force-node spacing
            double dt_driver = dt_cfd;
            double tmax = t_max;
            FAST_ExtInfw_Init(&t.tid_local, &tmax, t.fst_file.c_str(), &turb_id, out_root,
                              &t.num_force_pts_blade, &t.num_force_pts_tower, pos, &abort_lev,
                              &dt_driver, &t.dt_fast, &inflow_type, &t.num_blades, &t.num_blade_elem,
                              &t.num_tower_elem, &node_cluster_type, &t.to_cfd, &t.from_cfd,
                              &err_stat, err_msg);
            fast_check(err_stat, err_msg, "FAST_ExtInfw_Init for " + t.name);
            if (inflow_type != 2) {
                Abort("OpenFAST model '" + t.fst_file + "' for " + t.name +
                      " does not take external inflow; set CompInflow = 2 in the .fst file");
            }
            // still air until the caller supplies the flow at the nodes
            for (int nd = 0; nd < t.to_cfd.pxVel_Len; ++nd) {
                t.from_cfd.u[nd] = 0.0f;
                t.from_cfd.v[nd] = 0.0f;
                t.from_cfd.w[nd] = 0.0f;
            }
            t.time_index = 0;
            finish_setup(t, dt_cfd);
            // the owner keeps the node velocities; still air until the flow is supplied
            t.node_vel.assign(3 * static_cast<std::size_t>(t.num_vel_nodes), 0.0);
        }
    }
    check_lockstep();
    m_initialized = true;
}

void
OpenFASTDriver::restart (const std::string& prefix, double dt_cfd, bool solved0)
{
    BL_PROFILE("OpenFASTDriver::restart()");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_initialized, "OpenFASTDriver::restart after init or restart");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_owners_set, "OpenFASTDriver::restart: set_owners() first");
    int err_stat = ErrID_None;
    char err_msg[INTERFACE_STRING_LENGTH];
    allocate();

    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        TurbineState& t = m_turb[i];
        if (is_owner(i)) {
            const std::string root = prefix + t.name;
            int abort_lev = ErrID_Fatal;
            int n_t_global = 0;
            FAST_ExtInfw_Restart(&t.tid_local, root.c_str(), &abort_lev, &t.dt_fast,
                                 &t.num_blades, &t.num_blade_elem, &t.num_tower_elem, &n_t_global,
                                 &t.to_cfd, &t.from_cfd, &err_stat, err_msg);
            fast_check(err_stat, err_msg, "FAST_ExtInfw_Restart for " + t.name + " from " + root + ".chkp");
            t.time_index = n_t_global;
            finish_setup(t, dt_cfd);
            // the velocities OpenFAST restored at its nodes, so the diagnostics continue from them
            t.node_vel.assign(3 * static_cast<std::size_t>(t.num_vel_nodes), 0.0);
            for (int nd = 0; nd < t.num_vel_nodes; ++nd) {
                t.node_vel[3*nd]   = t.from_cfd.u[nd];
                t.node_vel[3*nd+1] = t.from_cfd.v[nd];
                t.node_vel[3*nd+2] = t.from_cfd.w[nd];
            }
        }
    }
    check_lockstep();
    m_initialized = true;
    // the logs continue; a restart in a clean directory starts them with a header
    m_log_truncate = false;
    // a checkpoint written before OpenFAST's first solution (the step-0 checkpoint) still needs it
    m_solved0 = solved0;
}

void
OpenFASTDriver::create_checkpoint (const std::string& prefix) const
{
    BL_PROFILE("OpenFASTDriver::create_checkpoint()");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized, "OpenFASTDriver::create_checkpoint needs init() or restart() first");
    int err_stat = ErrID_None;
    char err_msg[INTERFACE_STRING_LENGTH];
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        const TurbineState& t = m_turb[i];
        if (!is_owner(i)) { continue; }
        const std::string root = prefix + t.name;
        int tid = t.tid_local;
        FAST_CreateCheckpoint(&tid, root.c_str(), &err_stat, err_msg);
        fast_check(err_stat, err_msg, "FAST_CreateCheckpoint for " + t.name + " to " + root + ".chkp");
    }
}

void
OpenFASTDriver::solution0 ()
{
    BL_PROFILE("OpenFASTDriver::solution0()");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized && !m_solved0, "OpenFASTDriver::solution0 needs init() first and runs once");
    int err_stat = ErrID_None;
    char err_msg[INTERFACE_STRING_LENGTH];
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        TurbineState& t = m_turb[i];
        if (is_owner(i)) {
            FAST_CFD_Solution0(&t.tid_local, &err_stat, err_msg);
            fast_check(err_stat, err_msg, "FAST_CFD_Solution0 for " + t.name);
            pull_from_fast(t);
        }
    }
    m_solved0 = true;
    // the logs start afresh at the first solution
    m_log_truncate = true;
    m_logs_started = false;
}

void
OpenFASTDriver::set_uniform_velocity (const std::array<Real,3>& vel)
{
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        if (!is_owner(i)) { continue; }
        std::vector<Real> uvw(3 * static_cast<std::size_t>(m_turb[i].num_vel_nodes));
        for (int nd = 0; nd < m_turb[i].num_vel_nodes; ++nd) {
            for (int d = 0; d < 3; ++d) { uvw[3*nd+d] = vel[d]; }
        }
        set_node_velocities(i, uvw);
    }
}

void
OpenFASTDriver::set_node_velocities (int i, const std::vector<Real>& uvw)
{
    TurbineState& t = m_turb[i];
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_owner(i), "OpenFASTDriver::set_node_velocities: only the owner of " + t.name + " holds its nodes");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(uvw.size()) == 3 * t.num_vel_nodes,
                                     "OpenFASTDriver::set_node_velocities: one u,v,w triple per velocity node of " + t.name);
    t.node_vel = uvw;
    for (int nd = 0; nd < t.num_vel_nodes; ++nd) {
        t.from_cfd.u[nd] = static_cast<float>(uvw[3*nd]);
        t.from_cfd.v[nd] = static_cast<float>(uvw[3*nd+1]);
        t.from_cfd.w[nd] = static_cast<float>(uvw[3*nd+2]);
    }
}

void
OpenFASTDriver::step ()
{
    BL_PROFILE("OpenFASTDriver::step()");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_solved0, "OpenFASTDriver::step before solution0");
    int err_stat = ErrID_None;
    char err_msg[INTERFACE_STRING_LENGTH];
    // OpenFAST's C library keeps ONE step counter (n_t_global) for all the turbines of a process and
    // advances it only after the call for its last turbine, so every OpenFAST step has to be applied to
    // the local turbines in turn (tid_local order): substeps outermost, turbines inner. Stepping one
    // turbine through all its substeps first leaves its clock frozen and OpenFAST aborts with
    // "t(1) must not equal t(2)" in ED_Input_ExtrapInterp. check_lockstep() made the substep counts equal.
    const int nsub = local_substeps();
    for (int s = 0; s < nsub; ++s) {
        for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
            TurbineState& t = m_turb[i];
            if (!is_owner(i)) { continue; }
            // FAST_CFD_Step leaves the status unset for all but the last turbine past OpenFAST's stop
            // time, and reports that case for the last one as information only: reset it before each
            // call and treat "Simulation completed" as the end it is (the loads would stay frozen)
            err_stat = ErrID_None;
            err_msg[0] = '\0';
            // a nanosecond clock: a call can take less than MPI_Wtime's tick (a microsecond on some MPIs)
            const auto t_start = std::chrono::steady_clock::now();
            FAST_CFD_Step(&t.tid_local, &err_stat, err_msg);
            t.step_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            if (err_stat == ErrID_Info && std::string(err_msg).find("completed") != std::string::npos) {
                Abort("OpenFAST reports \"" + std::string(err_msg) + "\" for " + t.name + " at its time index " +
                      std::to_string(t.time_index) + ": it has reached the stop time it was started with and no longer steps");
            }
            fast_check(err_stat, err_msg, "FAST_CFD_Step for " + t.name);
            ++t.time_index;
        }
    }
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        if (is_owner(i)) { pull_from_fast(m_turb[i]); }
    }
}

void
OpenFASTDriver::reset_step_seconds ()
{
    for (auto& t : m_turb) { t.step_seconds = 0.0; }
}

// The turbines owned by this rank share OpenFAST's step counter, so they must run the same OpenFAST
// time step: the substep count per ERF step has to be the same for all of them.
int
OpenFASTDriver::local_substeps () const
{
    int nsub = -1;
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        if (is_owner(i)) { nsub = m_turb[i].num_substeps; break; }
    }
    return nsub;
}

void
OpenFASTDriver::check_lockstep () const
{
    const int nsub = local_substeps();
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        if (is_owner(i) && m_turb[i].num_substeps != nsub) {
            Abort("OpenFAST turbines " + m_turb[i].name + " and the first one owned by this rank use different "
                  "time steps (" + std::to_string(m_turb[i].num_substeps) + " vs " + std::to_string(nsub) +
                  " substeps per ERF step): OpenFAST steps all the turbines of a process together, so their "
                  "DT must be equal; set the same DT in the .fst files, or run on at least as many ranks as there are "
                  "distinct DTs (the owners keep turbines of different DTs apart whenever the ranks allow)");
        }
    }
}

// Copy the float ExtInfw buffers into the Real node arrays. OpenFAST reports the node
// positions in the turbine's own frame (ExternalInflow.f90 sets pxVel to the AeroDyn mesh
// Position + TranslationDisp and never applies the TurbinePosition given at init), so the base
// position is added here to put them in ERF's frame; the hub position from FAST_HubPosition is
// in the same frame and is shifted alike. OpenFAST reports the forces the fluid exerts on the
// structure; the reaction on the fluid is their negative, which the force spreading applies. The
// stored force values are OpenFAST's, unchanged.
void
OpenFASTDriver::pull_from_fast (TurbineState& t)
{
    const int nv = t.num_vel_nodes;
    const int nf = t.num_force_nodes;
    t.vel_pos.resize(3 * nv);
    t.force_pos.resize(3 * nf);
    t.force.resize(3 * nf);
    t.force_vel.resize(3 * nf);
    t.chord.resize(nf);
    for (int n = 0; n < nv; ++n) {
        t.vel_pos[3*n+0] = t.base_pos[0] + t.to_cfd.pxVel[n];
        t.vel_pos[3*n+1] = t.base_pos[1] + t.to_cfd.pyVel[n];
        t.vel_pos[3*n+2] = t.base_pos[2] + t.to_cfd.pzVel[n];
    }
    for (int n = 0; n < nf; ++n) {
        t.force_pos[3*n+0] = t.base_pos[0] + t.to_cfd.pxForce[n];
        t.force_pos[3*n+1] = t.base_pos[1] + t.to_cfd.pyForce[n];
        t.force_pos[3*n+2] = t.base_pos[2] + t.to_cfd.pzForce[n];
        t.force[3*n+0] = t.to_cfd.fx[n];
        t.force[3*n+1] = t.to_cfd.fy[n];
        t.force[3*n+2] = t.to_cfd.fz[n];
        t.force_vel[3*n+0] = t.to_cfd.xdotForce[n];
        t.force_vel[3*n+1] = t.to_cfd.ydotForce[n];
        t.force_vel[3*n+2] = t.to_cfd.zdotForce[n];
        t.chord[n] = t.to_cfd.forceNodesChord[n];
    }
    // OpenFAST runs with the floating-point traps off: a diverged turbine would otherwise put
    // NaN into the momentum source silently
    for (int n = 0; n < 3 * nf; ++n) {
        if (!std::isfinite(t.force[n]) || !std::isfinite(t.force_pos[n])) {
            Abort("OpenFAST returned a non-finite force or position for " + t.name + " at force node " + std::to_string(n / 3) +
                  " (time index " + std::to_string(t.time_index) + "): the turbine model has diverged");
        }
    }
    int err_stat = ErrID_None;
    char err_msg[INTERFACE_STRING_LENGTH];
    float hub[3] = {0.0f, 0.0f, 0.0f};
    float rot[3] = {0.0f, 0.0f, 0.0f};
    double dcm[9] = {0.0};
    FAST_HubPosition(&t.tid_local, hub, rot, dcm, &err_stat, err_msg);
    fast_check(err_stat, err_msg, "FAST_HubPosition for " + t.name);
    for (int d = 0; d < 3; ++d) { t.hub_pos[d] = t.base_pos[d] + hub[d]; }
    // OpenFAST orientation matrices map global to local: their rows are the local axes in
    // global coordinates. The 9 doubles are the Fortran column-major flattening, so the hub
    // frame's x axis (the shaft) is elements 0, 3, 6.
    {
        Real n[3] = {static_cast<Real>(dcm[0]), static_cast<Real>(dcm[3]), static_cast<Real>(dcm[6])};
        const Real len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        if (len > Real(0.0)) { for (int d = 0; d < 3; ++d) { t.hub_axis[d] = n[d] / len; } }
    }
    // the rotation rate about the shaft axis, signed like the torque about it, so that their
    // product is the power whichever way the model turns
    t.rotor_speed = rot[0] * t.hub_axis[0] + rot[1] * t.hub_axis[1] + rot[2] * t.hub_axis[2];
    // the first velocity node is the hub: the two positions must agree, else the frame
    // conventions above no longer hold for this OpenFAST build
    if (nv > 0) {
        Real dist = 0.0;
        for (int d = 0; d < 3; ++d) { dist += (t.vel_pos[d] - t.hub_pos[d]) * (t.vel_pos[d] - t.hub_pos[d]); }
        dist = std::sqrt(dist);
        if (dist > 1.0) {
            Abort("OpenFAST hub position (" + std::to_string(t.hub_pos[0]) + ", " + std::to_string(t.hub_pos[1]) +
                  ", " + std::to_string(t.hub_pos[2]) + ") and hub velocity node (" + std::to_string(t.vel_pos[0]) +
                  ", " + std::to_string(t.vel_pos[1]) + ", " + std::to_string(t.vel_pos[2]) + ") of " + t.name +
                  " differ by " + std::to_string(dist) + " m; the node positions are not in the expected turbine frame");
        }
    }
}

std::array<Real,3>
OpenFASTDriver::thrust (const TurbineState& t) const
{
    // the rotor: the hub node and the blade nodes (the tower nodes follow the blades)
    std::array<Real,3> f{{0.0, 0.0, 0.0}};
    const int nfb = t.num_blades * t.num_force_pts_blade;
    for (int n = 0; n <= nfb && n < t.num_force_nodes; ++n) {
        for (int d = 0; d < 3; ++d) { f[d] += t.force[3*n+d]; }
    }
    return f;
}

// hub node first, then the blade nodes; the tower nodes (if any) come after the blades
void
OpenFASTDriver::node_velocity_means (const TurbineState& t, std::array<Real,3>& hub, std::array<Real,3>& blade) const
{
    hub = {{0.0, 0.0, 0.0}};
    blade = {{0.0, 0.0, 0.0}};
    const int nbn = t.num_blades * t.num_blade_elem;
    if (static_cast<int>(t.node_vel.size()) >= 3 * (1 + nbn)) {
        for (int d = 0; d < 3; ++d) { hub[d] = t.node_vel[d]; }
        for (int nd = 1; nd <= nbn; ++nd) {
            for (int d = 0; d < 3; ++d) { blade[d] += t.node_vel[3*nd+d]; }
        }
        if (nbn > 0) { for (int d = 0; d < 3; ++d) { blade[d] /= nbn; } }
    }
}

// the tower nodes follow the blades; zero without tower force nodes
std::array<Real,3>
OpenFASTDriver::tower_force (const TurbineState& t) const
{
    std::array<Real,3> f{{0.0, 0.0, 0.0}};
    const int first = 1 + t.num_blades * t.num_force_pts_blade;
    const int last = std::min(t.num_force_nodes, first + t.num_force_pts_tower);
    for (int n = first; n < last; ++n) {
        for (int d = 0; d < 3; ++d) { f[d] += t.force[3*n+d]; }
    }
    return f;
}

void
OpenFASTDriver::set_restored_loads (int i, const std::vector<Real>& force, const std::vector<Real>& node_vel)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(i >= 0 && i < static_cast<int>(m_turb.size()), "set_restored_loads: no such turbine");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_owner(i), "set_restored_loads: only the owner of a turbine holds its nodes");
    TurbineState& t = m_turb[i];
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(force.size() == 3 * static_cast<std::size_t>(t.num_force_nodes) &&
                                     node_vel.size() == 3 * static_cast<std::size_t>(t.num_vel_nodes),
                                     "set_restored_loads: the checkpointed node arrays do not match the node counts of " + t.name);
    t.force = force;
    set_node_velocities(i, node_vel);
}

void
OpenFASTDriver::set_nacelle_force (int i, const std::array<Real,3>& f)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(i >= 0 && i < static_cast<int>(m_turb.size()), "set_nacelle_force: no such turbine");
    m_turb[i].nacelle_force = f;
}

// torque of the rotor's node forces about the hub axis through the hub
Real
OpenFASTDriver::torque (const TurbineState& t) const
{
    Real q = 0.0;
    const int nfb = t.num_blades * t.num_force_pts_blade;
    for (int n = 0; n <= nfb && n < t.num_force_nodes; ++n) {
        const Real rx = t.force_pos[3*n]   - t.hub_pos[0];
        const Real ry = t.force_pos[3*n+1] - t.hub_pos[1];
        const Real rz = t.force_pos[3*n+2] - t.hub_pos[2];
        const Real fx = t.force[3*n], fy = t.force[3*n+1], fz = t.force[3*n+2];
        q += t.hub_axis[0] * (ry * fz - rz * fy) + t.hub_axis[1] * (rz * fx - rx * fz) + t.hub_axis[2] * (rx * fy - ry * fx);
    }
    return q;
}

const std::string&
OpenFASTDriver::erf_log_columns ()
{
    static const std::string c = "rotor_speed,thrust_x,thrust_y,thrust_z,torque,power,axis_x,axis_y,axis_z,"
                                 "tower_x,tower_y,tower_z,nacelle_x,nacelle_y,nacelle_z,load_x,load_y,load_z";
    return c;
}

const std::string&
OpenFASTDriver::flow_log_columns ()
{
    static const std::string c = "hub_u,hub_v,hub_w,blade_mean_u,blade_mean_v,blade_mean_w";
    return c;
}

void
OpenFASTDriver::erf_log_row (int i, Real* row) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_owner(i), "OpenFASTDriver::erf_log_row: only the owner of a turbine holds its loads");
    const TurbineState& t = m_turb[i];
    const std::array<Real,3> f = thrust(t);
    const std::array<Real,3> ft = tower_force(t);
    const Real q = torque(t);
    const Real r[erf_log_size] = {t.rotor_speed, f[0], f[1], f[2], q, q * t.rotor_speed,
                                  t.hub_axis[0], t.hub_axis[1], t.hub_axis[2], ft[0], ft[1], ft[2],
                                  t.nacelle_force[0], t.nacelle_force[1], t.nacelle_force[2],
                                  f[0] + ft[0] + t.nacelle_force[0], f[1] + ft[1] + t.nacelle_force[1],
                                  f[2] + ft[2] + t.nacelle_force[2]};
    for (int k = 0; k < erf_log_size; ++k) { row[k] = r[k]; }
}

void
OpenFASTDriver::flow_log_row (int i, Real* row) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(is_owner(i), "OpenFASTDriver::flow_log_row: only the owner of a turbine holds its velocities");
    std::array<Real,3> hub, blade;
    node_velocity_means(m_turb[i], hub, blade);
    for (int d = 0; d < 3; ++d) { row[d] = hub[d]; row[3+d] = blade[d]; }
}

void
OpenFASTDriver::write_diagnostics (double time)
{
    // the first row after solution0() starts the files afresh; after restart() they are appended to
    const bool truncate = m_log_truncate && !m_logs_started;
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        if (!is_owner(i)) { continue; }
        const TurbineState& t = m_turb[i];
        Real erf_row[erf_log_size], flow_row[flow_log_size];
        erf_log_row(i, erf_row);
        flow_log_row(i, flow_row);
        std::ofstream out;
        if (erf_actuator::open_log(out, t.output_root + "_erf.csv", truncate)) { out << "time," << erf_log_columns() << "\n"; }
        erf_actuator::write_log_row(out, time, nullptr, erf_row, erf_log_size);
        std::ofstream flow;
        if (erf_actuator::open_log(flow, t.output_root + "_flow.csv", truncate)) { flow << "time," << flow_log_columns() << "\n"; }
        erf_actuator::write_log_row(flow, time, nullptr, flow_row, flow_log_size);
    }
    m_logs_started = true;
}

} // namespace erf_openfast
