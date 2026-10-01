// The actuator-disk representation of an OpenFAST rotor: each blade force node becomes a ring
// of points at its radius about the hub axis; the rotor's total force and its torque about
// the axis must be preserved, the ring points must lie in the node's rotor plane at its
// radius, and the forces must be those on the fluid. The actuator-line representation: the
// points are the force nodes themselves with the forces on the fluid, and the tip travel per
// step that limits the time step follows from the rotor speed and the tip radius. The tower
// points are the tower force nodes after the blades, and the nacelle drag point follows the
// drag law with the kernel's self-induction correction.

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

TEST(OpenFASTRotor, LinePointsAreTheForceNodesWithTheForceOnTheFluid)
{
    const std::array<Real,3> hub{{500.0, 500.0, 150.0}};
    std::array<Real,3> axis{{static_cast<Real>(std::cos(0.3)), static_cast<Real>(std::sin(0.3)), Real(-0.1)}};   // yawed and tilted
    const Real la = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    for (auto& a : axis) { a /= la; }
    const int nodes = 5;
    TurbineState t = make_rotor(hub, axis, nodes, 120.0, 2.0e4, 3.0e3, 1.1);
    // two tower nodes after the blades: they must be left out
    t.num_force_pts_tower = 2;
    t.num_force_nodes += 2;
    for (int k = 0; k < 2; ++k) {
        t.force_pos.insert(t.force_pos.end(), {hub[0], hub[1], Real(20.0 + 60.0 * k)});
        t.force.insert(t.force.end(), {Real(1.0e3), Real(0.0), Real(0.0)});
    }
    std::vector<Real> pos, f;
    erf_actuator::alm_points(t, pos, f);
    ASSERT_EQ(pos.size(), 3u * (1 + 3 * nodes));
    ASSERT_EQ(f.size(), pos.size());
    for (std::size_t k = 0; k < pos.size(); ++k) {
        EXPECT_EQ(pos[k], t.force_pos[k]) << "position entry " << k;
        EXPECT_EQ(f[k], -t.force[k]) << "force entry " << k;
    }
    // the line carries minus the rotor's force and torque exactly, tower excluded
    std::array<Real,3> got, want;
    Real q_got, q_want;
    totals(pos, f, hub, axis, got, q_got);
    std::vector<Real> rotor_pos(t.force_pos.begin(), t.force_pos.begin() + 3 * (1 + 3 * nodes));
    std::vector<Real> rotor_f(t.force.begin(), t.force.begin() + 3 * (1 + 3 * nodes));
    totals(rotor_pos, rotor_f, hub, axis, want, q_want);
    EXPECT_GT(q_want, 0.0);
    for (int d = 0; d < 3; ++d) { EXPECT_NEAR(got[d], -want[d], tol() * 15 * 2.0e4) << "component " << d; }
    EXPECT_NEAR(q_got, -q_want, tol() * std::abs(q_want));
    // appending to non-empty lists keeps what was there
    std::vector<Real> pos2(3, Real(7.0)), f2(3, Real(8.0));
    erf_actuator::alm_points(t, pos2, f2);
    EXPECT_EQ(pos2.size(), 3u + pos.size());
    EXPECT_EQ(pos2[0], Real(7.0));
    EXPECT_EQ(pos2[3], pos[0]);
}

TEST(OpenFASTRotor, TipTravelPerStepFromRotorSpeedAndTipRadius)
{
    const std::array<Real,3> hub{{500.0, 500.0, 150.0}};
    std::array<Real,3> axis{{static_cast<Real>(std::cos(0.5)), static_cast<Real>(std::sin(0.5)), Real(0.1)}};
    const Real la = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    for (auto& a : axis) { a /= la; }
    const int nodes = 4;
    const Real R = 120.0;
    TurbineState t = make_rotor(hub, axis, nodes, R, 1.0e4, 1.0e3);
    // make_rotor puts the outermost node at (nodes - 0.5) / nodes * R, in the rotor plane;
    // shift every blade node along the axis: the tip radius is measured in the plane
    for (int nd = 1; nd < t.num_force_nodes; ++nd) {
        for (int d = 0; d < 3; ++d) { t.force_pos[3*nd+d] += Real(3.0) * axis[d]; }
    }
    const Real r_tip = (nodes - Real(0.5)) / nodes * R;
    EXPECT_NEAR(erf_actuator::tip_radius(t), r_tip, tol() * R);

    t.rotor_speed = 0.79;   // rad/s: the IEA 15 MW at rated, tip speed ~ 90 m/s
    const Real dt = 0.5, dx = 50.0;
    const Real want = Real(0.79) * r_tip * dt / dx;
    EXPECT_NEAR(erf_actuator::tip_cells_per_step(t, dt, dx), want, tol() * want);
    EXPECT_GT(want, Real(0.7));
    EXPECT_LT(want, Real(1.0));   // the regression deck's margin: half a metre per step under one cell
    // a step twice as long, or cells half as wide, doubles the travel
    EXPECT_NEAR(erf_actuator::tip_cells_per_step(t, 2 * dt, dx), 2 * want, tol() * want);
    EXPECT_NEAR(erf_actuator::tip_cells_per_step(t, dt, dx / 2), 2 * want, tol() * want);
    // a parked rotor sweeps nothing; a rotor turning the other way the same amount
    t.rotor_speed = 0.0;
    EXPECT_EQ(erf_actuator::tip_cells_per_step(t, dt, dx), Real(0.0));
    t.rotor_speed = -0.79;
    EXPECT_NEAR(erf_actuator::tip_cells_per_step(t, dt, dx), want, tol() * want);
    // no blade nodes: zero radius
    TurbineState bare;
    bare.num_blades = 3; bare.num_force_pts_blade = 0; bare.num_force_nodes = 1;
    bare.force_pos.assign(3, 0.0); bare.force.assign(3, 0.0);
    bare.rotor_speed = 1.0;
    EXPECT_EQ(erf_actuator::tip_radius(bare), Real(0.0));
}

TEST(OpenFASTRotor, TowerPointsAreTheTowerNodesAfterTheBlades)
{
    const std::array<Real,3> hub{{500.0, 500.0, 150.0}};
    const std::array<Real,3> axis{{1.0, 0.0, 0.0}};
    const int nodes = 4, ntow = 3;
    TurbineState t = make_rotor(hub, axis, nodes, 120.0, 2.0e4, 3.0e3);
    // no tower nodes yet: nothing appended, zero tower force
    std::vector<Real> pos, f;
    erf_actuator::tower_points(t, pos, f);
    EXPECT_TRUE(pos.empty());
    EXPECT_TRUE(f.empty());
    // three tower nodes base to top carrying drag along +x (on the structure) and a little in y
    t.num_force_pts_tower = ntow;
    t.num_force_nodes += ntow;
    for (int k = 0; k < ntow; ++k) {
        t.force_pos.insert(t.force_pos.end(), {hub[0], hub[1], Real(25.0 + 50.0 * k)});
        t.force.insert(t.force.end(), {Real(1.0e3 * (k + 1)), Real(10.0), Real(0.0)});
    }
    pos.assign(3, Real(-1.0)); f.assign(3, Real(-2.0));
    erf_actuator::tower_points(t, pos, f);
    ASSERT_EQ(pos.size(), 3u * (1 + ntow));
    ASSERT_EQ(f.size(), pos.size());
    EXPECT_EQ(pos[0], Real(-1.0));   // what was there is kept
    for (int k = 0; k < ntow; ++k) {
        const int nd = 1 + 3 * nodes + k;
        for (int d = 0; d < 3; ++d) {
            EXPECT_EQ(pos[3 * (1 + k) + d], t.force_pos[3*nd+d]) << "tower node " << k << " component " << d;
            EXPECT_EQ(f[3 * (1 + k) + d], -t.force[3*nd+d]) << "tower node " << k << " component " << d;
        }
    }
    // the blade points are unaffected by the tower nodes
    std::vector<Real> pb, fb;
    erf_actuator::alm_points(t, pb, fb);
    EXPECT_EQ(pb.size(), 3u * (1 + 3 * nodes));
    // a node count shorter than the tower claims: only the nodes that exist are taken
    t.num_force_nodes -= 1;
    pos.clear(); f.clear();
    erf_actuator::tower_points(t, pos, f);
    EXPECT_EQ(pos.size(), 3u * (ntow - 1));
}

TEST(OpenFASTRotor, NacelleDragFollowsTheDragLawWithTheKernelCorrection)
{
    const Real rho = 1.2, cd = 1.0, area = 50.0;
    // head-on: -1/2 rho cd A U^2 along the flow
    auto f = erf_actuator::nacelle_drag_force({{10.0, 0.0, 0.0}}, rho, cd, area);
    EXPECT_NEAR(f[0], -0.5 * rho * cd * area * 100.0, tol() * 3000.0);
    EXPECT_EQ(f[1], Real(0.0));
    EXPECT_EQ(f[2], Real(0.0));
    // oblique: along -u with magnitude 1/2 rho cd A |u|^2
    const std::array<Real,3> u{{6.0, -8.0, 2.0}};
    f = erf_actuator::nacelle_drag_force(u, rho, cd, area);
    const Real speed2 = 36.0 + 64.0 + 4.0;
    const Real mag = std::sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    EXPECT_NEAR(mag, 0.5 * rho * cd * area * speed2, tol() * 3200.0);
    for (int d = 0; d < 3; ++d) { EXPECT_NEAR(f[d] / mag, -u[d] / std::sqrt(speed2), tol()); }
    // at rest, or without drag: nothing
    f = erf_actuator::nacelle_drag_force({{0.0, 0.0, 0.0}}, rho, cd, area);
    EXPECT_EQ(f[0], Real(0.0));
    f = erf_actuator::nacelle_drag_force(u, rho, 0.0, area);
    EXPECT_EQ(f[1], Real(0.0));
    // the correction: u / (1 - cd A / (4 pi eps^2)); with eps = sqrt(2 cd A / pi) (Kynema's
    // choice) the factor is 1 - 1/8, so the corrected velocity is 8/7 of the sampled one
    const Real eps = std::sqrt(2.0 * cd * area / pi);
    const auto uc = erf_actuator::nacelle_corrected_velocity(u, cd, area, eps);
    for (int d = 0; d < 3; ++d) { EXPECT_NEAR(uc[d], u[d] * 8.0 / 7.0, tol() * 10.0) << "component " << d; }
    // a wide kernel changes almost nothing; a zero drag area nothing at all
    const auto uw = erf_actuator::nacelle_corrected_velocity(u, cd, area, 1.0e4);
    EXPECT_NEAR(uw[0], u[0], 1.0e-6);
    const auto u0 = erf_actuator::nacelle_corrected_velocity(u, 0.0, area, 1.0);
    EXPECT_EQ(u0[0], u[0]);
}

// sampling = upstream: every velocity node moves by sample_diameters_upstream * 2 R against the shaft
// axis, so the hub node leads the rotor by that distance and the blade nodes keep their radii
TEST(OpenFASTRotor, UpstreamSamplingPositionsShiftEveryNodeAlongTheAxis)
{
    const std::array<Real,3> hub{{600.0, 600.0, 150.0}};
    std::array<Real,3> axis{{0.8, 0.6, 0.0}};   // yawed rotor: the shift follows the shaft, not x
    TurbineState t = make_rotor(hub, axis, 6, 120.0, 1.0e4, 1.0e3);
    t.vel_pos = t.force_pos;   // the fixture places the force nodes; the velocity nodes sit on them here
    t.num_vel_nodes = t.num_force_nodes;
    const Real R = erf_actuator::tip_radius(t);
    ASSERT_GT(R, 0.0);   // the fixture's outermost node sits inside the nominal radius; the shift uses the node radius
    const std::vector<Real> up = erf_actuator::upstream_sampling_positions(t, 1.5);
    ASSERT_EQ(up.size(), t.vel_pos.size());
    const Real shift = 1.5 * 2.0 * R;
    for (std::size_t n = 0; n + 2 < up.size(); n += 3) {
        for (int d = 0; d < 3; ++d) { EXPECT_NEAR(up[n+d] - t.vel_pos[n+d], -shift * t.hub_axis[d], 1.0e-9); }
        // the radius about the shifted hub is unchanged
        Real r0 = 0.0, r1 = 0.0;
        for (int d = 0; d < 3; ++d) {
            const Real a = t.vel_pos[n+d] - hub[d], b = up[n+d] - (hub[d] - shift * t.hub_axis[d]);
            r0 += a * a; r1 += b * b;
        }
        EXPECT_NEAR(std::sqrt(r0), std::sqrt(r1), 1.0e-9);
    }
    // the hub node (node 0) leads by the shift along the axis
    Real lead = 0.0;
    for (int d = 0; d < 3; ++d) { lead += (hub[d] - up[d]) * t.hub_axis[d]; }
    EXPECT_NEAR(lead, shift, 1.0e-9);
    // zero diameters: the nodes themselves
    EXPECT_EQ(erf_actuator::upstream_sampling_positions(t, 0.0), t.vel_pos);
}


TEST(OpenFASTRotor, FilteredDiskFactorMatchesShapiroEq25)
{
    // Ct' = 1, Delta/R = 1: M = 1 / (1 + 0.25 / sqrt(3 pi))
    EXPECT_NEAR(erf_actuator::filtered_disk_factor(1.0, 1.0), 1.0 / (1.0 + 0.25 / std::sqrt(3.0 * pi)), 1.0e-12);
    EXPECT_NEAR(erf_actuator::filtered_disk_factor(1.0, 1.0), 0.924698, 1.0e-6);
    // no thrust or a vanishing filter width: no correction
    EXPECT_DOUBLE_EQ(erf_actuator::filtered_disk_factor(0.0, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(erf_actuator::filtered_disk_factor(2.0, 0.0), 1.0);
    // a wider kernel corrects more
    EXPECT_LT(erf_actuator::filtered_disk_factor(1.5, 1.2), erf_actuator::filtered_disk_factor(1.5, 0.6));
    // ERF's kernel exp(-r^2/eps^2) is the paper's filter with Delta = sqrt(6) eps
    EXPECT_NEAR(erf_actuator::filter_width_from_eps(40.0), std::sqrt(6.0) * 40.0, 1.0e-12);
}

TEST(OpenFASTRotor, FilteredDiskCorrectionRecoversTheFreeStreamAtTheFixedPoint)
{
    // a disk of radius R in a free stream U with thrust coefficient Ct, smeared by ERF's kernel of
    // width eps: the paper's filtered disk velocity is U (1 - a) / M, and the correction applied to
    // it, with the thrust and the free stream of the previous step, must hand back U exactly
    const Real U = 10.0, ct = 0.8, rho = 1.2, Rr = 110.0, eps = 40.0;
    const Real a = 0.5 * (1.0 - std::sqrt(1.0 - ct));
    const Real ct_prime = 4.0 * a / (1.0 - a);
    const Real M = erf_actuator::filtered_disk_factor(ct_prime, erf_actuator::filter_width_from_eps(eps) / Rr);
    const Real u_disk = U * (1.0 - a) / M;
    const Real thrust = 0.5 * rho * pi * Rr * Rr * U * U * ct;
    const auto c = erf_actuator::filtered_disk_correction(u_disk, thrust, U, rho, Rr, eps);
    EXPECT_NEAR(c.ct, ct, 1.0e-12);
    EXPECT_NEAR(c.a, a, 1.0e-12);
    EXPECT_NEAR(c.ct_prime, ct_prime, 1.0e-12);
    EXPECT_NEAR(c.M, M, 1.0e-12);
    EXPECT_NEAR(c.u_inf, U, 1.0e-9);
    EXPECT_NEAR(c.factor, U / u_disk, 1.0e-12);
    EXPECT_GT(c.factor, 1.0);
    // a smaller previous free stream raises Ct and the correction; the fixed point is stable from below
    const auto c_low = erf_actuator::filtered_disk_correction(u_disk, thrust, 0.9 * U, rho, Rr, eps);
    EXPECT_GT(c_low.ct, ct);
    EXPECT_GT(c_low.u_inf, U);
    // no previous loads (the first step, or a stopped rotor): the disk velocity is the free stream
    const auto c0 = erf_actuator::filtered_disk_correction(u_disk, 0.0, U, rho, Rr, eps);
    EXPECT_DOUBLE_EQ(c0.u_inf, u_disk);
    EXPECT_DOUBLE_EQ(c0.factor, 1.0);
    EXPECT_DOUBLE_EQ(erf_actuator::filtered_disk_correction(u_disk, thrust, 0.0, rho, Rr, eps).factor, 1.0);
    // an absurd thrust is clamped to Ct = 0.96 and stays finite
    const auto c_big = erf_actuator::filtered_disk_correction(u_disk, 1.0e3 * thrust, U, rho, Rr, eps);
    EXPECT_NEAR(c_big.ct, 0.96, 1.0e-12);
    EXPECT_TRUE(std::isfinite(c_big.u_inf));
}

TEST(OpenFASTRotor, DiskAxialVelocityWeightsTheBladeNodesByRadius)
{
    const std::array<Real,3> hub{{600.0, 600.0, 150.0}};
    std::array<Real,3> axis{{0.8, 0.6, 0.0}};
    const int nodes = 5;
    const Real Rn = 100.0;
    TurbineState t = make_rotor(hub, axis, nodes, Rn, 1.0e4, 1.0e3);
    t.vel_pos = t.force_pos;
    t.num_vel_nodes = t.num_force_nodes;
    t.num_blade_elem = nodes;
    const std::array<Real,3> perp{{-0.6, 0.8, 0.0}};   // in the rotor plane: no axial part
    std::vector<Real> uvw(t.vel_pos.size(), 0.0);
    // hub node: a large axial velocity that must carry no weight
    for (int d = 0; d < 3; ++d) { uvw[d] = 100.0 * t.hub_axis[d]; }
    // blade nodes: axial velocity 7 r / R plus an in-plane component
    for (int nd = 1; nd <= 3 * nodes; ++nd) {
        Real r2 = 0.0, rn = 0.0;
        for (int d = 0; d < 3; ++d) { const Real v = t.vel_pos[3*nd+d] - hub[d]; rn += v * t.hub_axis[d]; }
        for (int d = 0; d < 3; ++d) { const Real p = t.vel_pos[3*nd+d] - hub[d] - rn * t.hub_axis[d]; r2 += p * p; }
        const Real r = std::sqrt(r2);
        for (int d = 0; d < 3; ++d) { uvw[3*nd+d] = 7.0 * r / Rn * t.hub_axis[d] + 3.0 * perp[d]; }
    }
    // radii (i + 1/2) / 5 R: the radius-weighted mean of 7 r / R is 7 sum r^2 / (R sum r) = 4.62
    EXPECT_NEAR(erf_actuator::disk_axial_velocity(t, uvw), 4.62, 1.0e-9);
    // a uniform axial flow is returned unchanged
    for (int nd = 0; nd < t.num_vel_nodes; ++nd) { for (int d = 0; d < 3; ++d) { uvw[3*nd+d] = 8.5 * t.hub_axis[d]; } }
    EXPECT_NEAR(erf_actuator::disk_axial_velocity(t, uvw), 8.5, 1.0e-9);
    // no blade nodes: zero
    t.num_blade_elem = 0;
    EXPECT_DOUBLE_EQ(erf_actuator::disk_axial_velocity(t, uvw), 0.0);
}
