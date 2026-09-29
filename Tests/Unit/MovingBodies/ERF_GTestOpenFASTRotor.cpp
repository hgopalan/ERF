// The actuator-disk representation of an OpenFAST rotor: each blade force node becomes a ring
// of points at its radius about the hub axis; the rotor's total force and its torque about
// the axis must be preserved, the ring points must lie in the node's rotor plane at its
// radius, and the forces must be those on the fluid.

#include <array>
#include <cmath>
#include <vector>

#include <AMReX_REAL.H>

#include <gtest/gtest.h>

#include "ERF_OpenFASTRotor.H"

namespace {

using amrex::Real;
using erf_openfast::TurbineState;

constexpr double pi = 3.14159265358979323846;

Real tol () { return (sizeof(Real) == 8) ? Real(1.0e-9) : Real(2.0e-4); }

std::array<Real,3> cross (const std::array<Real,3>& a, const std::array<Real,3>& b)
{
    return {{a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]}};
}

// A rigid three-blade rotor with `nodes` force nodes per blade in the plane normal to `axis`
// through the hub; each node carries an axial force fa (on the structure, along the axis) and
// a tangential force ft * r / R. Blade b, node i: radius (i + 0.5) / nodes * R, azimuth
// 2 pi b / 3 + phase.
TurbineState make_rotor (const std::array<Real,3>& hub, std::array<Real,3> axis, int nodes,
                         Real R, Real fa, Real ft, Real phase = 0.3)
{
    const Real la = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    for (auto& a : axis) { a /= la; }
    // in-plane basis: e1 = z x axis (horizontal), e2 = axis x e1
    std::array<Real,3> e1 = cross({{0.0, 0.0, 1.0}}, axis);
    const Real l1 = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
    for (auto& e : e1) { e /= l1; }
    const std::array<Real,3> e2 = cross(axis, e1);

    TurbineState t;
    t.name = "T1";
    t.num_blades = 3;
    t.num_force_pts_blade = nodes;
    t.num_force_pts_tower = 0;
    t.num_force_nodes = 1 + 3 * nodes;
    t.hub_pos = hub;
    t.hub_axis = axis;
    t.force_pos.assign(3 * t.num_force_nodes, 0.0);
    t.force.assign(3 * t.num_force_nodes, 0.0);
    for (int d = 0; d < 3; ++d) { t.force_pos[d] = hub[d]; }
    for (int b = 0; b < 3; ++b) {
        for (int i = 0; i < nodes; ++i) {
            const int nd = 1 + b * nodes + i;
            const Real r = (i + Real(0.5)) / nodes * R;
            const Real th = 2.0 * pi * b / 3.0 + phase;
            std::array<Real,3> rhat, that;
            for (int d = 0; d < 3; ++d) { rhat[d] = std::cos(th) * e1[d] + std::sin(th) * e2[d]; }
            that = cross(axis, rhat);
            for (int d = 0; d < 3; ++d) {
                t.force_pos[3*nd+d] = hub[d] + r * rhat[d];
                t.force[3*nd+d] = fa * axis[d] + ft * (r / R) * that[d];
            }
        }
    }
    return t;
}

// total force and torque about `axis` through `hub` of a point list
void totals (const std::vector<Real>& pos, const std::vector<Real>& f, const std::array<Real,3>& hub,
             const std::array<Real,3>& axis, std::array<Real,3>& force, Real& torque)
{
    force = {{0.0, 0.0, 0.0}};
    torque = 0.0;
    for (std::size_t p = 0; p < pos.size() / 3; ++p) {
        std::array<Real,3> r, fp;
        for (int d = 0; d < 3; ++d) { r[d] = pos[3*p+d] - hub[d]; fp[d] = f[3*p+d]; force[d] += fp[d]; }
        const auto m = cross(r, fp);
        torque += m[0]*axis[0] + m[1]*axis[1] + m[2]*axis[2];
    }
}

} // namespace

TEST(OpenFASTRotor, RingsCarryTheNegativeOfTheRotorForceAndTorque)
{
    const std::array<Real,3> hub{{500.0, 500.0, 150.0}};
    const std::array<Real,3> axis{{1.0, 0.0, 0.0}};
    const TurbineState t = make_rotor(hub, axis, 4, 120.0, 2.0e4, 3.0e3);
    std::vector<Real> pos, f;
    erf_actuator::adm_rings(t, 16, pos, f);
    ASSERT_EQ(pos.size(), 3u * (1 + 3 * 4 * 16));
    ASSERT_EQ(f.size(), pos.size());

    std::array<Real,3> got, want;
    Real q_got, q_want;
    totals(pos, f, hub, axis, got, q_got);
    totals(t.force_pos, t.force, hub, axis, want, q_want);
    EXPECT_NEAR(want[0], 12 * 2.0e4, 1.0e-6 * 12 * 2.0e4);   // the rotor's thrust on the structure
    EXPECT_GT(q_want, 0.0);
    for (int d = 0; d < 3; ++d) { EXPECT_NEAR(got[d], -want[d], tol() * 12 * 2.0e4) << "component " << d; }
    EXPECT_NEAR(q_got, -q_want, tol() * std::abs(q_want));
}

TEST(OpenFASTRotor, RingPointsLieInTheRotorPlaneAtTheNodeRadius)
{
    const std::array<Real,3> hub{{500.0, 500.0, 150.0}};
    std::array<Real,3> axis{{static_cast<Real>(std::cos(0.4)), static_cast<Real>(std::sin(0.4)), Real(0.05)}};   // yawed and tilted
    const Real la = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    for (auto& a : axis) { a /= la; }
    const int nodes = 3, npt = 12;
    const Real R = 120.0;
    const TurbineState t = make_rotor(hub, axis, nodes, R, 1.0e4, 2.0e3);
    std::vector<Real> pos, f;
    erf_actuator::adm_rings(t, npt, pos, f);
    // point 0 is the hub; then blade b node i gives ring (b * nodes + i) of npt points
    for (int d = 0; d < 3; ++d) { EXPECT_EQ(pos[d], hub[d]); }
    for (int ring = 0; ring < 3 * nodes; ++ring) {
        const Real r_node = ((ring % nodes) + Real(0.5)) / nodes * R;
        for (int j = 0; j < npt; ++j) {
            const std::size_t p = 1 + ring * npt + j;
            std::array<Real,3> rv;
            for (int d = 0; d < 3; ++d) { rv[d] = pos[3*p+d] - hub[d]; }
            const Real along = rv[0]*axis[0] + rv[1]*axis[1] + rv[2]*axis[2];
            const Real r = std::sqrt(rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2]);
            EXPECT_NEAR(along, 0.0, tol() * R) << "ring " << ring << " point " << j;
            EXPECT_NEAR(r, r_node, tol() * R) << "ring " << ring << " point " << j;
            // the ring's share of the axial force, on the fluid: -fa / npt along the axis
            const Real f_ax = f[3*p]*axis[0] + f[3*p+1]*axis[1] + f[3*p+2]*axis[2];
            EXPECT_NEAR(f_ax, -1.0e4 / npt, tol() * 1.0e4) << "ring " << ring << " point " << j;
        }
    }
    // total force and torque preserved on the tilted axis too
    std::array<Real,3> got, want;
    Real q_got, q_want;
    totals(pos, f, hub, axis, got, q_got);
    totals(t.force_pos, t.force, hub, axis, want, q_want);
    for (int d = 0; d < 3; ++d) { EXPECT_NEAR(got[d], -want[d], tol() * 9 * 1.0e4); }
    EXPECT_NEAR(q_got, -q_want, tol() * std::abs(q_want));
}

TEST(OpenFASTRotor, OnePointPerRingKeepsTheAxialForceAndTorque)
{
    // num_points_t = 1 is the degenerate disk: one point per node at the node's radius with
    // the node's force rotated to the single ring azimuth. The axial force and the torque are
    // still exact; the in-plane components no longer cancel between the blades (they do for
    // any ring of two or more points), which is why one point per ring is no actuator disk.
    const std::array<Real,3> hub{{0.0, 0.0, 100.0}};
    const std::array<Real,3> axis{{1.0, 0.0, 0.0}};
    const TurbineState t = make_rotor(hub, axis, 2, 50.0, 5.0e3, 1.0e3);
    std::vector<Real> pos, f;
    erf_actuator::adm_rings(t, 1, pos, f);
    ASSERT_EQ(pos.size(), 3u * 7);
    std::array<Real,3> got, want;
    Real q_got, q_want;
    totals(pos, f, hub, axis, got, q_got);
    totals(t.force_pos, t.force, hub, axis, want, q_want);
    EXPECT_NEAR(got[0], -want[0], tol() * 6 * 5.0e3);
    EXPECT_NEAR(q_got, -q_want, tol() * std::abs(q_want));
    // two points per ring already cancel the in-plane components again
    pos.clear(); f.clear();
    erf_actuator::adm_rings(t, 2, pos, f);
    totals(pos, f, hub, axis, got, q_got);
    for (int d = 0; d < 3; ++d) { EXPECT_NEAR(got[d], -want[d], tol() * 6 * 5.0e3) << "component " << d; }
    EXPECT_NEAR(q_got, -q_want, tol() * std::abs(q_want));
}

TEST(OpenFASTRotor, OutOfPlaneSineFlagsAWrongAxis)
{
    const std::array<Real,3> hub{{500.0, 500.0, 150.0}};
    const std::array<Real,3> axis{{1.0, 0.0, 0.0}};
    TurbineState t = make_rotor(hub, axis, 4, 120.0, 1.0, 0.0);
    EXPECT_NEAR(erf_actuator::max_out_of_plane_sine(t), 0.0, tol());
    // a small precone-like tilt of the nodes: 4 degrees
    for (int nd = 1; nd < t.num_force_nodes; ++nd) {
        const Real r = std::sqrt(std::pow(t.force_pos[3*nd+1] - hub[1], 2) + std::pow(t.force_pos[3*nd+2] - hub[2], 2));
        t.force_pos[3*nd] = hub[0] + r * std::tan(4.0 * pi / 180.0);
    }
    EXPECT_NEAR(erf_actuator::max_out_of_plane_sine(t), std::sin(4.0 * pi / 180.0), 1.0e-6 + tol());
    // the axis taken along the blades instead of the shaft
    t.hub_axis = {{0.0, 0.0, 1.0}};
    EXPECT_GT(erf_actuator::max_out_of_plane_sine(t), 0.9);
}
