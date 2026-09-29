#include "ERF_OpenFASTDriver.H"

#include <cmath>
#include <cstring>
#include <fstream>
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
        t.owner_rank = i % nprocs;
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
OpenFASTDriver::init (double dt_cfd, double t_max)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!m_initialized, "OpenFASTDriver::init called twice");
    int err_stat = ErrID_None;
    char err_msg[INTERFACE_STRING_LENGTH];

    if (m_num_local > 0) {
        int n = m_num_local;
        FAST_AllocateTurbines(&n, &err_stat, err_msg);
        fast_check(err_stat, err_msg, "FAST_AllocateTurbines");
    }

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

            // still air until the caller supplies the flow at the nodes
            for (int nd = 0; nd < t.num_vel_nodes; ++nd) {
                t.from_cfd.u[nd] = 0.0f;
                t.from_cfd.v[nd] = 0.0f;
                t.from_cfd.w[nd] = 0.0f;
            }
            t.time_index = 0;
            pull_from_fast(t);
        }
        broadcast_state(t);
        // every rank keeps the node velocities; still air until the flow is supplied
        t.node_vel.assign(3 * static_cast<std::size_t>(t.num_vel_nodes), 0.0);
    }
    m_initialized = true;
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
            open_diagnostics(t);
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
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        TurbineState& t = m_turb[i];
        if (is_owner(i)) {
            for (int s = 0; s < t.num_substeps; ++s) {
                FAST_CFD_Step(&t.tid_local, &err_stat, err_msg);
                fast_check(err_stat, err_msg, "FAST_CFD_Step for " + t.name);
                ++t.time_index;
            }
            pull_from_fast(t);
        }
        broadcast_state(t);
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
    if (!t.vel_pos.empty())   { ParallelDescriptor::Bcast(t.vel_pos.data(),   static_cast<int>(t.vel_pos.size()),   root); }
    if (!t.force_pos.empty()) { ParallelDescriptor::Bcast(t.force_pos.data(), static_cast<int>(t.force_pos.size()), root); }
    if (!t.force.empty())     { ParallelDescriptor::Bcast(t.force.data(),     static_cast<int>(t.force.size()),     root); }
    ParallelDescriptor::Bcast(t.hub_pos.data(), 3, root);
    ParallelDescriptor::Bcast(&t.rotor_speed, 1, root);
}

std::array<Real,3>
OpenFASTDriver::thrust (const TurbineState& t) const
{
    std::array<Real,3> f{{0.0, 0.0, 0.0}};
    const int nfb = t.num_blades * t.num_force_pts_blade;
    for (int n = 1; n <= nfb && n < t.num_force_nodes; ++n) {
        for (int d = 0; d < 3; ++d) { f[d] += t.force[3*n+d]; }
    }
    return f;
}

// torque about the hub axis, taken as the x axis of the turbine frame in this version
Real
OpenFASTDriver::torque (const TurbineState& t) const
{
    Real q = 0.0;
    const int nfb = t.num_blades * t.num_force_pts_blade;
    for (int n = 1; n <= nfb && n < t.num_force_nodes; ++n) {
        const Real ry = t.force_pos[3*n+1] - t.hub_pos[1];
        const Real rz = t.force_pos[3*n+2] - t.hub_pos[2];
        q += ry * t.force[3*n+2] - rz * t.force[3*n+1];
    }
    return q;
}

void
OpenFASTDriver::open_diagnostics (const TurbineState& t) const
{
    const std::string fname = t.output_root + "_erf.csv";
    const auto slash = fname.rfind('/');
    if (slash != std::string::npos) {
        UtilCreateDirectory(fname.substr(0, slash), 0755);
    }
    std::ofstream out(fname, std::ios::trunc);
    if (!out) { Abort("cannot open moving-bodies diagnostics file '" + fname + "'"); }
    out << "time,rotor_speed,thrust_x,thrust_y,thrust_z,torque,power\n";
    const std::string fflow = t.output_root + "_flow.csv";
    std::ofstream flow(fflow, std::ios::trunc);
    if (!flow) { Abort("cannot open moving-bodies diagnostics file '" + fflow + "'"); }
    flow << "time,hub_u,hub_v,hub_w,blade_mean_u,blade_mean_v,blade_mean_w\n";
}

void
OpenFASTDriver::write_diagnostics (double time)
{
    for (int i = 0; i < static_cast<int>(m_turb.size()); ++i) {
        if (!is_owner(i)) { continue; }
        const TurbineState& t = m_turb[i];
        const std::array<Real,3> f = thrust(t);
        const Real q = torque(t);
        // hub node first, then the blade nodes; the tower nodes (if any) come after the blades
        std::array<Real,3> hub{{0.0, 0.0, 0.0}};
        std::array<Real,3> blade{{0.0, 0.0, 0.0}};
        const int nbn = t.num_blades * t.num_blade_elem;
        if (static_cast<int>(t.node_vel.size()) >= 3 * (1 + nbn)) {
            for (int d = 0; d < 3; ++d) { hub[d] = t.node_vel[d]; }
            for (int nd = 1; nd <= nbn; ++nd) {
                for (int d = 0; d < 3; ++d) { blade[d] += t.node_vel[3*nd+d]; }
            }
            if (nbn > 0) { for (int d = 0; d < 3; ++d) { blade[d] /= nbn; } }
        }
        std::ofstream out(t.output_root + "_erf.csv", std::ios::app);
        out << std::setprecision(10) << time << "," << t.rotor_speed << ","
            << f[0] << "," << f[1] << "," << f[2] << "," << q << "," << q * t.rotor_speed << "\n";
        std::ofstream flow(t.output_root + "_flow.csv", std::ios::app);
        flow << std::setprecision(10) << time << ","
             << hub[0] << "," << hub[1] << "," << hub[2] << ","
             << blade[0] << "," << blade[1] << "," << blade[2] << "\n";
    }
}

} // namespace erf_openfast
