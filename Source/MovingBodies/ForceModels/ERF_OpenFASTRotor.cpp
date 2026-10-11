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
    Real ah[2] = {t.hub_axis[0], t.hub_axis[1]};
    const Real la = std::sqrt(ah[0] * ah[0] + ah[1] * ah[1]);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(la > Real(1.0e-6), "upstream_sampling_positions: the shaft axis must not be vertical");
    ah[0] /= la; ah[1] /= la;
    std::vector<Real> pos(t.vel_pos);
    for (std::size_t n = 0; n + 2 < pos.size(); n += 3) {
        pos[n]   -= shift * ah[0];
        pos[n+1] -= shift * ah[1];
    }
    return pos;
}

Real
filter_width_from_eps (Real eps) { return std::sqrt(Real(6.0)) * eps; }

Real
filtered_disk_factor (Real ct_prime, Real delta_over_R)
{
    const Real inv_sqrt_3pi = Real(1.0) / std::sqrt(Real(3.0) * pi);
    return Real(1.0) / (Real(1.0) + Real(0.25) * ct_prime * delta_over_R * inv_sqrt_3pi);
}

DiskCorrection
filtered_disk_correction (Real u_disk, Real thrust, Real u_inf_prev, Real rho, Real tip_radius, Real eps)
{
    DiskCorrection c;
    c.u_disk = u_disk;
    if (!(u_disk > Real(0.1)) || !(thrust > Real(0.0)) || !(u_inf_prev > Real(0.1)) || !(tip_radius > Real(0.0))) {
        c.u_inf = u_disk; c.factor = Real(1.0); return c;
    }
    const Real area = pi * tip_radius * tip_radius;
    c.ct = std::min(Real(0.96), std::max(Real(0.0), thrust / (Real(0.5) * rho * area * u_inf_prev * u_inf_prev)));
    c.a = Real(0.5) * (Real(1.0) - std::sqrt(Real(1.0) - c.ct));
    c.ct_prime = Real(4.0) * c.a / (Real(1.0) - c.a);
    c.M = filtered_disk_factor(c.ct_prime, filter_width_from_eps(eps) / tip_radius);
    c.clamped = (thrust / (Real(0.5) * rho * area * u_inf_prev * u_inf_prev) >= Real(0.96));
    c.u_inf = c.M * u_disk / (Real(1.0) - c.a);
    c.factor = c.u_inf / u_disk;
    c.gain = disk_correction_gain(c.a, filter_width_from_eps(eps) / tip_radius);
    return c;
}

Real
disk_correction_gain (Real a, Real delta_over_R)
{
    const Real delta = delta_over_R / std::sqrt(Real(3.0) * pi);
    // a <= 0.4 under the Ct clamp, so 1 - 2a >= 0.2 and the denominator stays positive
    return -Real(2.0) * a * (Real(1.0) - a) * (Real(1.0) - delta) /
           ((Real(1.0) - Real(2.0) * a) * (Real(1.0) - a + delta * a));
}

Real
disk_correction_relax (Real gain)
{
    return std::min(Real(1.0), std::max(Real(0.2), Real(1.0) / (Real(1.0) - std::min(gain, Real(0.0)))));
}

Real
disk_correction_relax (Real gain, Real dt, Real tau)
{
    // with omega <= dt / (tau (1 - G)) an error decays as exp(-t / tau) however large the gain
    const Real w = disk_correction_relax(gain);
    return (tau > Real(0.0)) ? std::min(w, dt / (tau * (Real(1.0) - std::min(gain, Real(0.0))))) : w;
}

Real
disk_axial_velocity (const erf_openfast::TurbineState& t, const std::vector<Real>& uvw)
{
    const auto& n = t.hub_axis;
    const int ne = t.num_blade_elem;                  // velocity nodes per blade, root to tip
    const int nb = t.num_blades * ne;                 // velocity nodes 1 .. nb are the blades
    const int nmax = static_cast<int>(std::min(uvw.size(), t.vel_pos.size()) / 3);
    if (nb + 1 > nmax || ne < 1) { return Real(0.0); }
    // the distance of each blade node from the hub axis
    std::vector<Real> r(nb + 1, Real(0.0));
    for (int nd = 1; nd <= nb; ++nd) {
        Real rv[3], rn = 0.0;
        for (int d = 0; d < 3; ++d) { rv[d] = t.vel_pos[3*nd+d] - t.hub_pos[d]; rn += rv[d] * n[d]; }
        Real r2 = 0.0;
        for (int d = 0; d < 3; ++d) { const Real p = rv[d] - rn * n[d]; r2 += p * p; }
        r[nd] = std::sqrt(r2);
    }
    Real wsum = 0.0, usum = 0.0;
    for (int bl = 0; bl < t.num_blades; ++bl) {
        for (int k = 0; k < ne; ++k) {
            const int nd = 1 + bl * ne + k;
            Real dr = 0.0;
            if (ne == 1) { dr = Real(1.0); }
            else if (k == 0) { dr = std::abs(r[nd+1] - r[nd]); }
            else if (k == ne - 1) { dr = std::abs(r[nd] - r[nd-1]); }
            else { dr = Real(0.5) * std::abs(r[nd+1] - r[nd-1]); }
            const Real w = r[nd] * dr;
            Real ua = 0.0;
            for (int d = 0; d < 3; ++d) { ua += uvw[3*nd+d] * n[d]; }
            wsum += w; usum += w * ua;
        }
    }
    return (wsum > Real(0.0)) ? usum / wsum : Real(0.0);
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
