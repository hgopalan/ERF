#include "ERF_OpenFASTRotor.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include <AMReX.H>
#include <AMReX_BLassert.H>

using namespace amrex;

namespace erf_actuator {

namespace {

constexpr Real pi = 3.14159265358979323846;

// an orthonormal pair (e1, e2) spanning the plane normal to n
void plane_basis (const std::array<Real,3>& n, std::array<Real,3>& e1, std::array<Real,3>& e2)
{
    // pick the global axis least aligned with n to start from
    const std::array<Real,3> a = (std::abs(n[2]) < Real(0.9)) ? std::array<Real,3>{{0.0, 0.0, 1.0}}
                                                              : std::array<Real,3>{{0.0, 1.0, 0.0}};
    // e1 = a x n, normalised; e2 = n x e1
    e1 = {{a[1]*n[2] - a[2]*n[1], a[2]*n[0] - a[0]*n[2], a[0]*n[1] - a[1]*n[0]}};
    const Real l = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
    for (int d = 0; d < 3; ++d) { e1[d] /= l; }
    e2 = {{n[1]*e1[2] - n[2]*e1[1], n[2]*e1[0] - n[0]*e1[2], n[0]*e1[1] - n[1]*e1[0]}};
}

} // namespace

void
alm_points (const erf_openfast::TurbineState& t, std::vector<Real>& pos, std::vector<Real>& force)
{
    // the hub node and the blade nodes, in OpenFAST's order; the tower nodes that may follow
    // them are left out
    const int n = std::min(t.num_force_nodes, 1 + t.num_blades * t.num_force_pts_blade);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(t.force_pos.size()) >= 3 * n &&
                                     static_cast<int>(t.force.size()) >= 3 * n,
                                     "alm_points: the turbine's node arrays are shorter than its node count");
    for (int nd = 0; nd < n; ++nd) {
        for (int d = 0; d < 3; ++d) {
            pos.push_back(t.force_pos[3*nd+d]);
            force.push_back(-t.force[3*nd+d]);   // on the fluid
        }
    }
}

void
tower_points (const erf_openfast::TurbineState& t, std::vector<Real>& pos, std::vector<Real>& force)
{
    const int first = 1 + t.num_blades * t.num_force_pts_blade;
    const int last = std::min(t.num_force_nodes, first + t.num_force_pts_tower);
    if (last <= first) { return; }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(static_cast<int>(t.force_pos.size()) >= 3 * last &&
                                     static_cast<int>(t.force.size()) >= 3 * last,
                                     "tower_points: the turbine's node arrays are shorter than its node count");
    for (int nd = first; nd < last; ++nd) {
        for (int d = 0; d < 3; ++d) {
            pos.push_back(t.force_pos[3*nd+d]);
            force.push_back(-t.force[3*nd+d]);   // on the fluid
        }
    }
}

std::array<Real,3>
nacelle_drag_force (const std::array<Real,3>& u, Real rho, Real cd, Real area)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(rho > Real(0.0) && cd >= Real(0.0) && area >= Real(0.0),
                                     "nacelle_drag_force: rho must be positive, cd and area non-negative");
    const Real speed = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    const Real c = -Real(0.5) * rho * cd * area * speed;
    return {{c * u[0], c * u[1], c * u[2]}};
}

std::array<Real,3>
nacelle_corrected_velocity (const std::array<Real,3>& u, Real cd, Real area, Real epsilon)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(epsilon > Real(0.0), "nacelle_corrected_velocity: epsilon must be positive");
    const Real fac = Real(1.0) - cd * area / (Real(4.0) * pi * epsilon * epsilon);
    if (!(fac > Real(0.0))) {
        amrex::Abort("nacelle drag: cd * area = " + std::to_string(cd * area) + " m^2 exceeds 4 pi epsilon^2 = " +
                     std::to_string(4.0 * pi * epsilon * epsilon) + " m^2; the spreading kernel is too narrow for the nacelle");
    }
    return {{u[0] / fac, u[1] / fac, u[2] / fac}};
}

std::vector<Real>
upstream_sampling_positions (const erf_openfast::TurbineState& t, Real diameters)
{
    const Real shift = diameters * Real(2.0) * tip_radius(t);
    std::vector<Real> pos(t.vel_pos);
    for (std::size_t n = 0; n + 2 < pos.size(); n += 3) {
        for (int d = 0; d < 3; ++d) { pos[n+d] -= shift * t.hub_axis[d]; }
    }
    return pos;
}

Real
tip_radius (const erf_openfast::TurbineState& t)
{
    const std::array<Real,3>& n = t.hub_axis;
    Real r2max = 0.0;
    const int nfb = t.num_blades * t.num_force_pts_blade;
    for (int nd = 1; nd <= nfb && nd < t.num_force_nodes; ++nd) {
        Real rv[3], rn = 0.0;
        for (int d = 0; d < 3; ++d) { rv[d] = t.force_pos[3*nd+d] - t.hub_pos[d]; rn += rv[d] * n[d]; }
        Real r2 = 0.0;
        for (int d = 0; d < 3; ++d) { const Real c = rv[d] - rn * n[d]; r2 += c * c; }   // in the rotor plane
        r2max = std::max(r2max, r2);
    }
    return std::sqrt(r2max);
}

Real
tip_cells_per_step (const erf_openfast::TurbineState& t, Real dt, Real dx)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dx > Real(0.0), "tip_cells_per_step: dx must be positive");
    return std::abs(t.rotor_speed) * tip_radius(t) * dt / dx;
}

void
adm_rings (const erf_openfast::TurbineState& t, int num_points_t,
           std::vector<Real>& pos, std::vector<Real>& force)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(num_points_t >= 1, "adm_rings: num_points_t must be >= 1");
    const std::array<Real,3>& n = t.hub_axis;
    std::array<Real,3> e1, e2;
    plane_basis(n, e1, e2);

    // the hub node: one point, its force on the fluid
    if (t.num_force_nodes > 0) {
        for (int d = 0; d < 3; ++d) { pos.push_back(t.force_pos[d]); force.push_back(-t.force[d]); }
    }
    const int nfb = t.num_blades * t.num_force_pts_blade;
    for (int nd = 1; nd <= nfb && nd < t.num_force_nodes; ++nd) {
        // the node's radius vector in the rotor plane and its force in (axial, radial, tangential)
        std::array<Real,3> rv{{0.0, 0.0, 0.0}}, f{{0.0, 0.0, 0.0}};
        for (int d = 0; d < 3; ++d) {
            rv[d] = t.force_pos[3*nd+d] - t.hub_pos[d];
            f[d] = -t.force[3*nd+d];   // on the fluid
        }
        const Real rn = rv[0]*n[0] + rv[1]*n[1] + rv[2]*n[2];
        for (int d = 0; d < 3; ++d) { rv[d] -= rn * n[d]; }     // project into the rotor plane
        const Real r = std::sqrt(rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2]);
        const Real f_ax = f[0]*n[0] + f[1]*n[1] + f[2]*n[2];
        Real f_rad = 0.0, f_tan = 0.0;
        std::array<Real,3> rhat{{0.0, 0.0, 0.0}}, that{{0.0, 0.0, 0.0}};
        if (r > Real(0.0)) {
            for (int d = 0; d < 3; ++d) { rhat[d] = rv[d] / r; }
            that = {{n[1]*rhat[2] - n[2]*rhat[1], n[2]*rhat[0] - n[0]*rhat[2], n[0]*rhat[1] - n[1]*rhat[0]}};
            f_rad = f[0]*rhat[0] + f[1]*rhat[1] + f[2]*rhat[2];
            f_tan = f[0]*that[0] + f[1]*that[1] + f[2]*that[2];
        }
        // the ring: the same radius, the force per point rotated with the azimuth
        for (int j = 0; j < num_points_t; ++j) {
            const Real th = 2.0 * pi * (j + 0.5) / num_points_t;
            const Real c = std::cos(th), s = std::sin(th);
            std::array<Real,3> rj;
            for (int d = 0; d < 3; ++d) { rj[d] = c * e1[d] + s * e2[d]; }   // radial unit vector at th
            const std::array<Real,3> tj{{n[1]*rj[2] - n[2]*rj[1], n[2]*rj[0] - n[0]*rj[2], n[0]*rj[1] - n[1]*rj[0]}};
            for (int d = 0; d < 3; ++d) {
                pos.push_back(t.hub_pos[d] + rn * n[d] + r * rj[d]);
                force.push_back((f_ax * n[d] + f_rad * rj[d] + f_tan * tj[d]) / num_points_t);
            }
        }
    }
}

Real
max_out_of_plane_sine (const erf_openfast::TurbineState& t)
{
    Real worst = 0.0;
    const int nfb = t.num_blades * t.num_force_pts_blade;
    for (int nd = 1; nd <= nfb && nd < t.num_force_nodes; ++nd) {
        Real rv[3], len2 = 0.0, rn = 0.0;
        for (int d = 0; d < 3; ++d) {
            rv[d] = t.force_pos[3*nd+d] - t.hub_pos[d];
            len2 += rv[d] * rv[d];
            rn += rv[d] * t.hub_axis[d];
        }
        if (len2 > Real(0.0)) { worst = std::max(worst, std::abs(rn) / std::sqrt(len2)); }
    }
    return worst;
}

} // namespace erf_actuator
