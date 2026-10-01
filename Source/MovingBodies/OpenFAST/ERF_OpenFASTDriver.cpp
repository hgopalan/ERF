#include "ERF_OpenFASTDriver.H"

#include "ERF_DiagnosticsLog.H"
#include "ERF_OpenFASTAudit.H"

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <AMReX.H>
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
        err = "erf.fixed_dt = " + std::to_string(dt_cfd) + " is not a positive whole multiple of the OpenFAST dt = " +
              std::to_string(dt_fast) + " (ratio " + std::to_string(ratio) + ")";
        return 0;
    }
    return n;
}


std::string
check_induction (const std::string& fst_file, bool want_off)
{
    const std::string path = openfast_module_path(fst_file, "AeroFile");
    if (path.empty()) { return {}; }
    const std::string comp = openfast_value(fst_file, "CompAero");
    if (comp == "0") { return {}; }
    std::string wake = openfast_value(path, "Wake_Mod");
    if (wake.empty()) { wake = openfast_value(path, "WakeMod"); }
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
    const std::string path = openfast_module_path(fst_file, "AeroFile");
    if (path.empty()) { return {}; }
    const std::string comp = openfast_value(fst_file, "CompAero");
    if (comp == "0") { return {}; }
    const std::string shadow = openfast_value(path, "TwrShadow");
    if (shadow.empty() || shadow == "0") { return {}; }
    return "the AeroDyn file '" + path + "' sets TwrShadow = " + shadow +
           "; with the tower's loads in the flow its wake reaches the blades through the sampled "
           "velocities, so AeroDyn's tower-shadow correction counts it twice: set TwrShadow = 0";
}

int
owner_rank_for (int turbine_index, int nprocs)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(turbine_index >= 0 && nprocs >= 1, "owner_rank_for: a non-negative turbine index and at least one rank");
    return turbine_index % nprocs;
}

OpenFASTDriver::OpenFASTDriver (const std::vector<MovingBodyInputs>& bodies)
{
    const int nprocs = ParallelDescriptor::NProcs();
    int i = 0;
    for (const MovingBodyInputs& b : bodies) {
        TurbineState t;
        t.name = b.name;
        t.fst_file = b.fst_file;
        t.output_root = b.output_root;
        t.base_pos = b.base_pos;
        t.num_force_pts_blade = b.num_force_points_blade;
        t.num_force_pts_tower = b.num_force_points_tower;
        if (b.mode != "none") {
            const std::string err = check_induction(b.fst_file, b.sampling == "disk");
            if (!err.empty()) { Abort("erf.moving_bodies." + b.name + ": " + err); }
        }
        t.owner_rank = owner_rank_for(i, nprocs);
        if (t.owner_rank == ParallelDescriptor::MyProc()) {
            t.tid_local = m_num_local++;
        }
        m_turb.push_back(t);
        ++i;
    }
}

OpenFASTDriver::~OpenFASTDriver ()
{
    if (m_initialized && m_num_local > 0) {
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
        Print() << "OpenFAST warning in " << where << ": " << err_msg << "\n";
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
        t.from_cfd.u_Len == t.num_vel_nodes && t.to_cfd.pxForce_Len == t.num_force_nodes,
        "OpenFAST ExtInfw array lengths are inconsistent for " + t.name);

    std::string err;
    t.num_substeps = substep_count(dt_cfd, t.dt_fast, err);
    if (t.num_substeps == 0) { Abort("erf.moving_bodies." + t.name + ": " + err); }
    pull_from_fast(t);
}

void
OpenFASTDriver::init (double dt_cfd, double t_max)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_initialized, "OpenFASTDriver::init called twice");
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
        }
        broadcast_state(t);
        // every rank keeps the node velocities; still air until the flow is supplied
        t.node_vel.assign(3 * static_cast<std::size_t>(t.num_vel_nodes), 0.0);
    }
    check_lockstep();
    m_initialized = true;
}

void
OpenFASTDriver::restart (const std::string& prefix, double dt_cfd)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_initialized, "OpenFASTDriver::restart after init or restart");
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
        }
        broadcast_state(t);
        // the velocities OpenFAST restored at its nodes, so the diagnostics continue from them
        t.node_vel.assign(3 * static_cast<std::size_t>(t.num_vel_nodes), 0.0);
        if (is_owner(i)) {
            for (int nd = 0; nd < t.num_vel_nodes; ++nd) {
                t.node_vel[3*nd]   = t.from_cfd.u[nd];
                t.node_vel[3*nd+1] = t.from_cfd.v[nd];
                t.node_vel[3*nd+2] = t.from_cfd.w[nd];
            }
            // the logs continue; a restart in a clean directory starts them with a header
            open_diagnostics(t, false);
        }
        ParallelDescriptor::Bcast(t.node_vel.data(), static_cast<int>(t.node_vel.size()), t.owner_rank);
    }
    check_lockstep();
    m_initialized = true;
    m_solved0 = true;
}

void
OpenFASTDriver::create_checkpoint (const std::string& prefix) const
{
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
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_initialized && !m_solved0, "OpenFASTDriver::solution0 needs init() first and runs once");
    int err_stat = ErrID_None;
    char err_msg[INTERFACE_STRING_LENGTH];
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        TurbineState& t = m_turb[i];
        if (is_owner(i)) {
            FAST_CFD_Solution0(&t.tid_local, &err_stat, err_msg);
            fast_check(err_stat, err_msg, "FAST_CFD_Solution0 for " + t.name);
            pull_from_fast(t);
            open_diagnostics(t, true);
        }
        broadcast_state(t);
    }
    m_solved0 = true;
}

void
OpenFASTDriver::set_uniform_velocity (const std::array<Real,3>& vel)
{
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
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
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(uvw.size()) == 3 * t.num_vel_nodes,
                                     "OpenFASTDriver::set_node_velocities: one u,v,w triple per velocity node of " + t.name);
    t.node_vel = uvw;
    if (!is_owner(i)) { return; }
    for (int nd = 0; nd < t.num_vel_nodes; ++nd) {
        t.from_cfd.u[nd] = static_cast<float>(uvw[3*nd]);
        t.from_cfd.v[nd] = static_cast<float>(uvw[3*nd+1]);
        t.from_cfd.w[nd] = static_cast<float>(uvw[3*nd+2]);
    }
}

void
OpenFASTDriver::step ()
{
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
            FAST_CFD_Step(&t.tid_local, &err_stat, err_msg);
            fast_check(err_stat, err_msg, "FAST_CFD_Step for " + t.name);
            ++t.time_index;
        }
    }
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        TurbineState& t = m_turb[i];
        if (is_owner(i)) { pull_from_fast(t); }
        broadcast_state(t);
    }
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
                  "DT must be equal; set the same DT in the .fst files or give each turbine its own rank");
        }
    }
}

// Copy the float ExtInfw buffers into the Real node arrays. OpenFAST reports the node
// positions in the turbine's own frame (ExternalInflow.f90 sets pxVel to the AeroDyn mesh
// Position + TranslationDisp and never applies the TurbinePosition given at init), so the base
// position is added here to put them in ERF's frame; the hub position from FAST_HubPosition is
// in the same frame and is shifted alike. OpenFAST reports the forces the fluid exerts on the
// structure; the reaction on the fluid is their negative, applied later by the force spreading,
// once it exists. The stored force values are OpenFAST's, unchanged.
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
    int err_stat = ErrID_None;
    char err_msg[INTERFACE_STRING_LENGTH];
    float hub[3] = {0.0f, 0.0f, 0.0f};
    float rot[3] = {0.0f, 0.0f, 0.0f};
    double dcm[9] = {0.0};
    FAST_HubPosition(&t.tid_local, hub, rot, dcm, &err_stat, err_msg);
    fast_check(err_stat, err_msg, "FAST_HubPosition for " + t.name);
    for (int d = 0; d < 3; ++d) { t.hub_pos[d] = t.base_pos[d] + hub[d]; }
    t.rotor_speed = std::sqrt(rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2]);
    // OpenFAST orientation matrices map global to local: their rows are the local axes in
    // global coordinates. The 9 doubles are the Fortran column-major flattening, so the hub
    // frame's x axis (the shaft) is elements 0, 3, 6.
    {
        Real n[3] = {static_cast<Real>(dcm[0]), static_cast<Real>(dcm[3]), static_cast<Real>(dcm[6])};
        const Real len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
        if (len > Real(0.0)) { for (int d = 0; d < 3; ++d) { t.hub_axis[d] = n[d] / len; } }
    }
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

void
OpenFASTDriver::broadcast_state (TurbineState& t)
{
    if (ParallelDescriptor::NProcs() == 1) { return; }
    const int root = t.owner_rank;
    int sizes[8] = {t.num_blades, t.num_blade_elem, t.num_tower_elem, t.num_vel_nodes,
                    t.num_force_nodes, t.num_substeps, t.time_index, t.num_force_pts_tower};
    ParallelDescriptor::Bcast(sizes, 8, root);
    t.num_blades = sizes[0]; t.num_blade_elem = sizes[1]; t.num_tower_elem = sizes[2];
    t.num_vel_nodes = sizes[3]; t.num_force_nodes = sizes[4]; t.num_substeps = sizes[5];
    t.time_index = sizes[6]; t.num_force_pts_tower = sizes[7];
    ParallelDescriptor::Bcast(&t.dt_fast, 1, root);
    t.vel_pos.resize(3 * t.num_vel_nodes);
    t.force_pos.resize(3 * t.num_force_nodes);
    t.force.resize(3 * t.num_force_nodes);
    t.force_vel.resize(3 * t.num_force_nodes);
    t.chord.resize(t.num_force_nodes);
    if (!t.vel_pos.empty())   { ParallelDescriptor::Bcast(t.vel_pos.data(),   static_cast<int>(t.vel_pos.size()),   root); }
    if (!t.force_pos.empty()) { ParallelDescriptor::Bcast(t.force_pos.data(), static_cast<int>(t.force_pos.size()), root); }
    if (!t.force.empty())     { ParallelDescriptor::Bcast(t.force.data(),     static_cast<int>(t.force.size()),     root); }
    if (!t.force_vel.empty()) { ParallelDescriptor::Bcast(t.force_vel.data(), static_cast<int>(t.force_vel.size()), root); }
    if (!t.chord.empty())     { ParallelDescriptor::Bcast(t.chord.data(),     static_cast<int>(t.chord.size()),     root); }
    ParallelDescriptor::Bcast(t.hub_pos.data(), 3, root);
    ParallelDescriptor::Bcast(t.hub_axis.data(), 3, root);
    ParallelDescriptor::Bcast(&t.rotor_speed, 1, root);
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

void
OpenFASTDriver::open_diagnostics (const TurbineState& t, bool truncate) const
{
    std::ofstream out;
    if (erf_actuator::open_log(out, t.output_root + "_erf.csv", truncate)) {
        out << "time,rotor_speed,thrust_x,thrust_y,thrust_z,torque,power,axis_x,axis_y,axis_z,"
               "tower_x,tower_y,tower_z,nacelle_x,nacelle_y,nacelle_z,load_x,load_y,load_z\n";
    }
    std::ofstream flow;
    if (erf_actuator::open_log(flow, t.output_root + "_flow.csv", truncate)) {
        flow << "time,hub_u,hub_v,hub_w,blade_mean_u,blade_mean_v,blade_mean_w\n";
    }
}

void
OpenFASTDriver::write_diagnostics (double time)
{
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        if (!is_owner(i)) { continue; }
        const TurbineState& t = m_turb[i];
        const std::array<Real,3> f = thrust(t);
        const Real q = torque(t);
        std::array<Real,3> hub, blade;
        node_velocity_means(t, hub, blade);
        std::ofstream out(t.output_root + "_erf.csv", std::ios::app);
        const std::array<Real,3> ft = tower_force(t);
        out << std::setprecision(10) << time << "," << t.rotor_speed << ","
            << f[0] << "," << f[1] << "," << f[2] << "," << q << "," << q * t.rotor_speed << ","
            << t.hub_axis[0] << "," << t.hub_axis[1] << "," << t.hub_axis[2] << ","
            << ft[0] << "," << ft[1] << "," << ft[2] << ","
            << t.nacelle_force[0] << "," << t.nacelle_force[1] << "," << t.nacelle_force[2] << ","
            << f[0] + ft[0] + t.nacelle_force[0] << "," << f[1] + ft[1] + t.nacelle_force[1] << ","
            << f[2] + ft[2] + t.nacelle_force[2] << "\n";
        std::ofstream flow(t.output_root + "_flow.csv", std::ios::app);
        flow << std::setprecision(10) << time << ","
             << hub[0] << "," << hub[1] << "," << hub[2] << ","
             << blade[0] << "," << blade[1] << "," << blade[2] << "\n";
    }
}

} // namespace erf_openfast
