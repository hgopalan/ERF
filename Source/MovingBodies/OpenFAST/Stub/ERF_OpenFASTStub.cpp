// Stand-in for libopenfastlib, implementing the subset of the OpenFAST 5 C API (unchanged since 4.0) declared in
// Stub/FAST_Library.h. It lets the ERF coupling be built and tested where OpenFAST is not
// installed (CI). The turbine is a rigid three-axis rotor: straight blades turning at a fixed
// speed about the +x axis, forces from a uniform-Ct disk model spread over the blade nodes, and
// a tangential force chosen so that the aerodynamic torque matches a prescribed Cp. Nothing
// here is a turbine model; it only has the geometry, ordering and data flow of the real API.
//
// The "input file" is a small key = value text file:
//   dt                  OpenFAST time step (s)
//   num_blades          (default 3)
//   num_blade_nodes     structural nodes per blade, returned as NumBlElem (default 10)
//   num_tower_nodes     structural tower nodes, returned as NumTwrElem (default 0)
//   rotor_radius        (m)
//   hub_height          (m, above the turbine base position)
//   rotor_speed_rpm     fixed rotor speed
//   ct, cp              thrust and power coefficients of the disk model
//   air_density         (kg/m^3, default 1.225)
//
// Node ordering follows ExtInfw: node 0 is the hub, then the blades in turn, root to tip, then
// the tower from base to top. Positions are in the turbine's own frame (the base at the origin,
// as OpenFAST reports them; the driver adds the turbine position), and positions and forces are
// float, like the real interface.

#include "FAST_Library.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double pi = 3.14159265358979323846;

struct StubTurbine {
    std::string fst_file;
    std::string out_root;
    double base_pos[3] = {0.0, 0.0, 0.0};
    double dt = 0.0;
    int num_blades = 3;
    int num_blade_nodes = 10;   // velocity nodes per blade (NumBlElem)
    int num_tower_nodes = 0;    // velocity nodes on the tower (NumTwrElem)
    int num_force_pts_blade = 0;
    int num_force_pts_tower = 0;
    double rotor_radius = 1.0;
    double hub_height = 1.0;
    double rotor_speed = 0.0;   // rad/s
    double ct = 0.0;
    double cp = 0.0;
    double air_density = 1.225;
    double azimuth = 0.0;       // rad, blade 0
    int time_index = 0;
    ExtInfw_InputType_t* to_cfd = nullptr;
    ExtInfw_OutputType_t* from_cfd = nullptr;

    int num_vel_nodes () const { return 1 + num_blades * num_blade_nodes + num_tower_nodes; }
    int num_force_nodes () const { return 1 + num_blades * num_force_pts_blade + num_force_pts_tower; }
};

std::vector<std::unique_ptr<StubTurbine>> g_turbines;

void set_error (int* err_stat, char* err_msg, int level, const std::string& msg)
{
    *err_stat = level;
    std::strncpy(err_msg, msg.c_str(), INTERFACE_STRING_LENGTH - 1);
    err_msg[INTERFACE_STRING_LENGTH - 1] = '\0';
}

void set_ok (int* err_stat, char* err_msg)
{
    *err_stat = ErrID_None;
    err_msg[0] = '\0';
}

StubTurbine* turbine_at (int* iTurb, int* err_stat, char* err_msg)
{
    const int i = *iTurb;
    if (i < 0 || i >= static_cast<int>(g_turbines.size()) || !g_turbines[i]) {
        set_error(err_stat, err_msg, ErrID_Fatal,
                  "OpenFAST stub: turbine index " + std::to_string(i) + " was not allocated");
        return nullptr;
    }
    return g_turbines[i].get();
}

bool read_input_file (StubTurbine& t, const std::string& fname, std::string& err)
{
    std::ifstream in(fname);
    if (!in) {
        err = "OpenFAST stub: cannot open input file '" + fname + "'";
        return false;
    }
    bool have_dt = false;
    std::string line;
    while (std::getline(in, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) { line.erase(hash); }
        std::istringstream ss(line);
        std::string key, eq;
        double value = 0.0;
        if (!(ss >> key >> eq >> value) || eq != "=") { continue; }
        if      (key == "dt")               { t.dt = value; have_dt = true; }
        else if (key == "num_blades")       { t.num_blades = static_cast<int>(value); }
        else if (key == "num_blade_nodes")  { t.num_blade_nodes = static_cast<int>(value); }
        else if (key == "num_tower_nodes")  { t.num_tower_nodes = static_cast<int>(value); }
        else if (key == "rotor_radius")     { t.rotor_radius = value; }
        else if (key == "hub_height")       { t.hub_height = value; }
        else if (key == "rotor_speed_rpm")  { t.rotor_speed = value * 2.0 * pi / 60.0; }
        else if (key == "ct")               { t.ct = value; }
        else if (key == "cp")               { t.cp = value; }
        else if (key == "air_density")      { t.air_density = value; }
        else {
            err = "OpenFAST stub: unknown key '" + key + "' in '" + fname + "'";
            return false;
        }
    }
    if (!have_dt || t.dt <= 0.0) {
        err = "OpenFAST stub: '" + fname + "' must set dt > 0";
        return false;
    }
    if (t.num_blades < 1 || t.num_blade_nodes < 1 || t.num_tower_nodes < 0 ||
        t.rotor_radius <= 0.0 || t.hub_height <= 0.0) {
        err = "OpenFAST stub: '" + fname + "' has a non-positive geometry or node count";
        return false;
    }
    return true;
}

template <typename T>
T* alloc_array (int n, int& len)
{
    len = n;
    T* p = static_cast<T*>(std::calloc(static_cast<std::size_t>(std::max(n, 1)), sizeof(T)));
    return p;
}

void allocate_interface (StubTurbine& t)
{
    const int nv = t.num_vel_nodes();
    const int nf = t.num_force_nodes();
    ExtInfw_InputType_t& in = *t.to_cfd;
    ExtInfw_OutputType_t& out = *t.from_cfd;
    std::memset(&in, 0, sizeof(in));
    std::memset(&out, 0, sizeof(out));
    in.pxVel = alloc_array<float>(nv, in.pxVel_Len);
    in.pyVel = alloc_array<float>(nv, in.pyVel_Len);
    in.pzVel = alloc_array<float>(nv, in.pzVel_Len);
    in.pxForce = alloc_array<float>(nf, in.pxForce_Len);
    in.pyForce = alloc_array<float>(nf, in.pyForce_Len);
    in.pzForce = alloc_array<float>(nf, in.pzForce_Len);
    in.xdotForce = alloc_array<float>(nf, in.xdotForce_Len);
    in.ydotForce = alloc_array<float>(nf, in.ydotForce_Len);
    in.zdotForce = alloc_array<float>(nf, in.zdotForce_Len);
    in.pOrientation = alloc_array<float>(9 * nf, in.pOrientation_Len);
    in.fx = alloc_array<float>(nf, in.fx_Len);
    in.fy = alloc_array<float>(nf, in.fy_Len);
    in.fz = alloc_array<float>(nf, in.fz_Len);
    in.momentx = alloc_array<float>(nf, in.momentx_Len);
    in.momenty = alloc_array<float>(nf, in.momenty_Len);
    in.momentz = alloc_array<float>(nf, in.momentz_Len);
    in.forceNodesChord = alloc_array<float>(nf, in.forceNodesChord_Len);
    out.u = alloc_array<float>(nv, out.u_Len);
    out.v = alloc_array<float>(nv, out.v_Len);
    out.w = alloc_array<float>(nv, out.w_Len);
    out.WriteOutput = alloc_array<float>(0, out.WriteOutput_Len);
    for (int n = 0; n < nf; ++n) {
        in.pOrientation[9 * n + 0] = 1.0f;
        in.pOrientation[9 * n + 4] = 1.0f;
        in.pOrientation[9 * n + 8] = 1.0f;
        in.forceNodesChord[n] = static_cast<float>(0.05 * t.rotor_radius);
    }
}

void free_interface (StubTurbine& t)
{
    if (t.to_cfd) {
        ExtInfw_InputType_t& in = *t.to_cfd;
        for (float* p : {in.pxVel, in.pyVel, in.pzVel, in.pxForce, in.pyForce, in.pzForce,
                         in.xdotForce, in.ydotForce, in.zdotForce, in.pOrientation,
                         in.fx, in.fy, in.fz, in.momentx, in.momenty, in.momentz,
                         in.forceNodesChord}) {
            std::free(p);
        }
        std::memset(&in, 0, sizeof(in));
    }
    if (t.from_cfd) {
        ExtInfw_OutputType_t& out = *t.from_cfd;
        for (float* p : {out.u, out.v, out.w, out.WriteOutput}) { std::free(p); }
        std::memset(&out, 0, sizeof(out));
    }
}

// Positions of the velocity nodes (structural mesh) and force nodes (actuator mesh) at the
// current azimuth. Blade b points along angle azimuth + 2 pi b / num_blades in the y-z plane.
void update_positions (StubTurbine& t)
{
    ExtInfw_InputType_t& in = *t.to_cfd;
    const double hub[3] = {0.0, 0.0, t.hub_height};   // turbine frame: base at the origin
    auto place = [&](float* px, float* py, float* pz, int nodes_per_blade, int tower_nodes) {
        int n = 0;
        px[n] = static_cast<float>(hub[0]); py[n] = static_cast<float>(hub[1]); pz[n] = static_cast<float>(hub[2]);
        ++n;
        for (int b = 0; b < t.num_blades; ++b) {
            const double ang = t.azimuth + 2.0 * pi * b / t.num_blades;
            for (int i = 0; i < nodes_per_blade; ++i, ++n) {
                const double r = (i + 0.5) / nodes_per_blade * t.rotor_radius;
                px[n] = static_cast<float>(hub[0]);
                py[n] = static_cast<float>(hub[1] + r * std::cos(ang));
                pz[n] = static_cast<float>(hub[2] + r * std::sin(ang));
            }
        }
        for (int i = 0; i < tower_nodes; ++i, ++n) {
            const double z = (i + 0.5) / tower_nodes * t.hub_height;
            px[n] = 0.0f;
            py[n] = 0.0f;
            pz[n] = static_cast<float>(z);
        }
    };
    place(in.pxVel, in.pyVel, in.pzVel, t.num_blade_nodes, t.num_tower_nodes);
    place(in.pxForce, in.pyForce, in.pzForce, t.num_force_pts_blade, t.num_force_pts_tower);
    // node velocities from the rigid rotation about +x through the hub
    const int nf = t.num_force_nodes();
    for (int n = 0; n < nf; ++n) {
        const double ry = in.pyForce[n] - hub[1];
        const double rz = in.pzForce[n] - hub[2];
        const bool on_rotor = (n >= 1) && (n < 1 + t.num_blades * t.num_force_pts_blade);
        in.xdotForce[n] = 0.0f;
        in.ydotForce[n] = on_rotor ? static_cast<float>(-t.rotor_speed * rz) : 0.0f;
        in.zdotForce[n] = on_rotor ? static_cast<float>( t.rotor_speed * ry) : 0.0f;
    }
}

// Uniform-Ct disk: thrust T = 1/2 rho Ct U^2 A (U the mean axial velocity the CFD supplied at
// the blade nodes), spread evenly over the blade force nodes; a tangential force proportional
// to radius whose torque is Q = 1/2 rho Cp U^3 A / omega. Like OpenFAST, the forces are those
// the fluid exerts ON THE STRUCTURE: the thrust points along the inflow (+x for +x wind), the
// torque along the rotation. Tower and hub carry no force.
void update_forces (StubTurbine& t)
{
    ExtInfw_InputType_t& in = *t.to_cfd;
    ExtInfw_OutputType_t& out = *t.from_cfd;
    const int nbn = t.num_blades * t.num_blade_nodes;
    double u_mean = 0.0;
    for (int n = 1; n <= nbn; ++n) { u_mean += out.u[n]; }
    u_mean /= nbn;
    const double area = pi * t.rotor_radius * t.rotor_radius;
    const double thrust = 0.5 * t.air_density * t.ct * u_mean * u_mean * area;
    const double power  = 0.5 * t.air_density * t.cp * u_mean * u_mean * u_mean * area;
    const double torque = (t.rotor_speed > 0.0) ? power / t.rotor_speed : 0.0;

    const int nfb = t.num_blades * t.num_force_pts_blade;
    const int nf = t.num_force_nodes();
    for (int n = 0; n < nf; ++n) { in.fx[n] = in.fy[n] = in.fz[n] = 0.0f; in.momentx[n] = in.momenty[n] = in.momentz[n] = 0.0f; }
    if (nfb == 0) { return; }
    const double hub[3] = {0.0, 0.0, t.hub_height};
    double sum_r2 = 0.0;
    for (int n = 1; n <= nfb; ++n) {
        const double ry = in.pyForce[n] - hub[1];
        const double rz = in.pzForce[n] - hub[2];
        sum_r2 += ry * ry + rz * rz;
    }
    const double sign = (u_mean >= 0.0) ? 1.0 : -1.0;   // force on the structure: along the inflow
    for (int n = 1; n <= nfb; ++n) {
        const double ry = in.pyForce[n] - hub[1];
        const double rz = in.pzForce[n] - hub[2];
        const double r2 = ry * ry + rz * rz;
        // tangential unit vector for rotation about +x: (0, -rz, ry)/r; torque = sum r * ft = k sum r^2
        const double k = (sum_r2 > 0.0) ? torque / sum_r2 : 0.0;
        in.fx[n] = static_cast<float>(sign * thrust / nfb);
        in.fy[n] = static_cast<float>(-k * rz);
        in.fz[n] = static_cast<float>( k * ry);
        (void) r2;
    }
}

bool write_checkpoint (const StubTurbine& t, const std::string& root, std::string& err)
{
    std::ofstream out(root + ".chkp");
    if (!out) { err = "OpenFAST stub: cannot write checkpoint '" + root + ".chkp'"; return false; }
    out.precision(17);
    out << "fst_file = " << t.fst_file << "\n"
        << "out_root = " << t.out_root << "\n"
        << "base_pos = " << t.base_pos[0] << " " << t.base_pos[1] << " " << t.base_pos[2] << "\n"
        << "num_force_pts_blade = " << t.num_force_pts_blade << "\n"
        << "num_force_pts_tower = " << t.num_force_pts_tower << "\n"
        << "time_index = " << t.time_index << "\n"
        << "azimuth = " << t.azimuth << "\n";
    return static_cast<bool>(out);
}

} // namespace

extern "C" {

void FAST_AllocateTurbines (int* iTurb, int* ErrStat, char* ErrMsg)
{
    if (*iTurb < 0) {
        set_error(ErrStat, ErrMsg, ErrID_Fatal, "OpenFAST stub: negative turbine count");
        return;
    }
    g_turbines.clear();
    g_turbines.resize(static_cast<std::size_t>(*iTurb));
    for (auto& t : g_turbines) { t = std::make_unique<StubTurbine>(); }
    set_ok(ErrStat, ErrMsg);
}

void FAST_DeallocateTurbines (int* ErrStat, char* ErrMsg)
{
    for (auto& t : g_turbines) { if (t) { free_interface(*t); } }
    g_turbines.clear();
    set_ok(ErrStat, ErrMsg);
}

void FAST_ExtInfw_Init (int* iTurb, double* /*TMax*/, const char* InputFileName, int* /*TurbIDforName*/, char* OutFileRoot,
                        int* NumActForcePtsBlade, int* NumActForcePtsTower, float* TurbinePosition, int* AbortErrLev,
                        double* /*dtDriver*/, double* dt, int* InflowType, int* NumBl, int* NumBlElem, int* NumTwrElem, int* /*NodeClusterType*/,
                        ExtInfw_InputType_t* ExtInfw_Input, ExtInfw_OutputType_t* ExtInfw_Output,
                        int* ErrStat, char* ErrMsg)
{
    StubTurbine* t = turbine_at(iTurb, ErrStat, ErrMsg);
    if (!t) { return; }
    std::string err;
    t->fst_file = InputFileName;
    if (!read_input_file(*t, t->fst_file, err)) { set_error(ErrStat, ErrMsg, ErrID_Fatal, err); return; }
    if (*NumActForcePtsBlade < 1 || *NumActForcePtsTower < 0) {
        set_error(ErrStat, ErrMsg, ErrID_Fatal, "OpenFAST stub: NumActForcePtsBlade must be >= 1 and NumActForcePtsTower >= 0");
        return;
    }
    t->num_force_pts_blade = *NumActForcePtsBlade;
    t->num_force_pts_tower = (t->num_tower_nodes > 0) ? *NumActForcePtsTower : 0;
    for (int d = 0; d < 3; ++d) { t->base_pos[d] = TurbinePosition[d]; }
    // the real library derives the output root from the input file name
    t->out_root = t->fst_file.substr(0, t->fst_file.rfind('.')) + "_stub";
    std::strncpy(OutFileRoot, t->out_root.c_str(), INTERFACE_STRING_LENGTH - 1);
    OutFileRoot[INTERFACE_STRING_LENGTH - 1] = '\0';
    t->to_cfd = ExtInfw_Input;
    t->from_cfd = ExtInfw_Output;
    allocate_interface(*t);
    t->azimuth = 0.0;
    t->time_index = 0;
    update_positions(*t);

    *AbortErrLev = ErrID_Fatal;
    *dt = t->dt;
    *InflowType = 2;
    *NumBl = t->num_blades;
    *NumBlElem = t->num_blade_nodes;
    *NumTwrElem = t->num_tower_nodes;
    set_ok(ErrStat, ErrMsg);
}

void FAST_ExtInfw_Restart (int* iTurb, const char* CheckpointRootName, int* AbortErrLev, double* dt,
                           int* NumBl, int* NumBlElem, int* NumTwrElem, int* n_t_global,
                           ExtInfw_InputType_t* ExtInfw_Input, ExtInfw_OutputType_t* ExtInfw_Output,
                           int* ErrStat, char* ErrMsg)
{
    StubTurbine* t = turbine_at(iTurb, ErrStat, ErrMsg);
    if (!t) { return; }
    const std::string fname = std::string(CheckpointRootName) + ".chkp";
    std::ifstream in(fname);
    if (!in) { set_error(ErrStat, ErrMsg, ErrID_Fatal, "OpenFAST stub: cannot open checkpoint '" + fname + "'"); return; }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string key, eq;
        if (!(ss >> key >> eq) || eq != "=") { continue; }
        if      (key == "fst_file") { ss >> t->fst_file; }
        else if (key == "out_root") { ss >> t->out_root; }
        else if (key == "base_pos") { ss >> t->base_pos[0] >> t->base_pos[1] >> t->base_pos[2]; }
        else if (key == "num_force_pts_blade") { ss >> t->num_force_pts_blade; }
        else if (key == "num_force_pts_tower") { ss >> t->num_force_pts_tower; }
        else if (key == "time_index") { ss >> t->time_index; }
        else if (key == "azimuth") { ss >> t->azimuth; }
    }
    std::string err;
    if (!read_input_file(*t, t->fst_file, err)) { set_error(ErrStat, ErrMsg, ErrID_Fatal, err); return; }
    t->to_cfd = ExtInfw_Input;
    t->from_cfd = ExtInfw_Output;
    allocate_interface(*t);
    update_positions(*t);
    *AbortErrLev = ErrID_Fatal;
    *dt = t->dt;
    *NumBl = t->num_blades;
    *NumBlElem = t->num_blade_nodes;
    *NumTwrElem = t->num_tower_nodes;
    *n_t_global = t->time_index;
    set_ok(ErrStat, ErrMsg);
}

void FAST_CFD_Solution0 (int* iTurb, int* ErrStat, char* ErrMsg)
{
    StubTurbine* t = turbine_at(iTurb, ErrStat, ErrMsg);
    if (!t) { return; }
    update_positions(*t);
    update_forces(*t);
    set_ok(ErrStat, ErrMsg);
}

void FAST_CFD_Step (int* iTurb, int* ErrStat, char* ErrMsg)
{
    StubTurbine* t = turbine_at(iTurb, ErrStat, ErrMsg);
    if (!t) { return; }
    t->azimuth = std::fmod(t->azimuth + t->rotor_speed * t->dt, 2.0 * pi);
    ++t->time_index;
    update_positions(*t);
    update_forces(*t);
    set_ok(ErrStat, ErrMsg);
}

void FAST_HubPosition (int* iTurb, float* absolute_position, float* rotation_veocity, double* orientation_dcm, int* ErrStat, char* ErrMsg)
{
    StubTurbine* t = turbine_at(iTurb, ErrStat, ErrMsg);
    if (!t) { return; }
    // the same frame as the node positions: the turbine's own, base at the origin
    absolute_position[0] = 0.0f;
    absolute_position[1] = 0.0f;
    absolute_position[2] = static_cast<float>(t->hub_height);
    rotation_veocity[0] = static_cast<float>(t->rotor_speed);
    rotation_veocity[1] = 0.0f;
    rotation_veocity[2] = 0.0f;
    for (int i = 0; i < 9; ++i) { orientation_dcm[i] = (i % 4 == 0) ? 1.0 : 0.0; }
    set_ok(ErrStat, ErrMsg);
}

void FAST_CreateCheckpoint (int* iTurb, const char* CheckpointRootName, int* ErrStat, char* ErrMsg)
{
    StubTurbine* t = turbine_at(iTurb, ErrStat, ErrMsg);
    if (!t) { return; }
    std::string err;
    if (!write_checkpoint(*t, CheckpointRootName, err)) { set_error(ErrStat, ErrMsg, ErrID_Fatal, err); return; }
    set_ok(ErrStat, ErrMsg);
}

} // extern "C"
