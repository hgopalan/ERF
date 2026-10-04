// Contract of the lattice tower (shaft and cross-arm drag, foundation reactions) and of its type's inputs.
//
// - MemberDrag.OnlyTheFlowNormalToTheMemberAndRelativeToItLoadsIt: the drag on a drag node is
//   1/2 rho Cd w L |U_n| U_n, only the flow normal to the member and relative to it counting.
// - MemberDrag.TheLatticeForceCoefficientIsTheSquareTowerCurve: the lattice force coefficient is
//   the square-tower curve of ASCE 7 (American Society of Civil Engineers, minimum design loads).
// - TowerType.EveryValueOutsideItsRangeIsRefusedByName: validate() names the key of every value
//   outside its range, NaN and infinity included.
// - Tower.TheBodyTapersUpToTheCrossArmWhichRunsAcrossTheLine: the shaft's drag nodes stand up the
//   tapering shaft, the cross-arm's across the line at the conductor's height.
// - Tower.InAUniformWindTheDragAndBaseMomentAreTheHandValues: the shaft's drag exactly (its width
//   is linear in height), the base moment to the midpoint rule's known error.
// - Tower.InALogLawWindTheDragIsTheFineIntegral: within the segments' quadrature error.
// - Tower.TheLegsShareTheLoadAndResistTheOverturningMoment: the four legs share the downward load
//   equally and the overturning moment linearly, in equilibrium, whichever way the tower faces;
//   the corner leg takes sqrt(2) times more from a diagonal pull.
// - Tower.EachLegLoadIsFlaggedOverItsAllowable.
// - Tower.EachLinePullsWhereItHangsAndTheFootingsTakeThemAll: several lines on one tower.
// - Tower.ANonFiniteLinePullIsRefusedNamingTheTower and
//   Tower.AFlatOrBadlyAimedTowerIsRefusedNamingIt: the preconditions abort with the tower's name.

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ERF_GTestThrowOnAbort.H"
#include "ERF_MemberDrag.H"
#include "ERF_Tower.H"
#include "ERF_TowerInputs.H"

using amrex::Real;
using erf_towers::MemberDrag;
using erf_towers::MemberNode;
using erf_towers::Tower;
using erf_towers::TowerType;
using P3 = std::array<Real,3>;

namespace {

constexpr Real tol = (std::is_same<Real, float>::value) ? Real(1.0e-5) : Real(1.0e-12);
constexpr Real rho = 1.2;

TowerType lattice ()
{
    TowerType t;
    t.name = "lattice";
    t.base_width = 6.0; t.top_width = 1.5; t.solidity = 0.2; t.arm_length = 12.0; t.arm_depth = 1.2;
    return t;
}

// the drag on every node of a tower in a wind given as a function of the node's height above its base
template <typename F>
void blow (Tower& tw, F wind)
{
    std::vector<Real> u, v;
    for (const auto& n : tw.nodes()) {
        const P3 w = wind(n.pos[2] - tw.base()[2]);
        u.insert(u.end(), {w[0], w[1], w[2]});
        v.insert(v.end(), {0.0, 0.0, 0.0});
    }
    std::vector<Real> f;
    MemberDrag(rho).loads(tw.nodes(), u, v, f);
    tw.set_loads(f);
}

} // namespace

TEST(MemberDrag, OnlyTheFlowNormalToTheMemberAndRelativeToItLoadsIt)
{
    MemberNode n;
    n.axis = {{0.0, 0.0, 1.0}}; n.length = 2.0; n.drag_width = 0.5; n.drag_coefficient = 1.5;
    const P3 still{{0.0, 0.0, 0.0}};
    // normal: 1/2 rho Cd w L U^2 along the wind
    auto f = erf_towers::member_drag(n, {{10.0, 0.0, 0.0}}, still, rho);
    EXPECT_NEAR(f[0], 0.5 * 1.2 * 1.5 * 0.5 * 2.0 * 100.0, 1.0e3 * tol);
    EXPECT_EQ(f[1], 0.0); EXPECT_EQ(f[2], 0.0);
    // along the axis: nothing
    f = erf_towers::member_drag(n, {{0.0, 0.0, 10.0}}, still, rho);
    EXPECT_EQ(f[0], 0.0); EXPECT_EQ(f[1], 0.0); EXPECT_EQ(f[2], 0.0);
    // oblique, 30 degrees off the axis: the normal half of the speed, in the normal direction
    f = erf_towers::member_drag(n, {{Real(5.0), Real(0.0), Real(5.0 * std::sqrt(Real(3.0)))}}, still, rho);
    EXPECT_NEAR(f[0], 0.5 * 1.2 * 1.5 * 0.5 * 2.0 * 25.0, 1.0e3 * tol);
    EXPECT_NEAR(f[2], 0.0, 1.0e3 * tol);
    // a node moving with the wind feels none of it; one moving against it, the sum
    f = erf_towers::member_drag(n, {{10.0, 0.0, 0.0}}, {{10.0, 0.0, 0.0}}, rho);
    EXPECT_EQ(f[0], 0.0);
    f = erf_towers::member_drag(n, {{10.0, 0.0, 0.0}}, {{-10.0, 0.0, 0.0}}, rho);
    EXPECT_NEAR(f[0], 4.0 * 0.5 * 1.2 * 1.5 * 0.5 * 2.0 * 100.0, 1.0e4 * tol);
}

TEST(MemberDrag, TheLatticeForceCoefficientIsTheSquareTowerCurve)
{
    EXPECT_NEAR(erf_towers::lattice_force_coefficient(0.0), 4.0, tol);
    EXPECT_NEAR(erf_towers::lattice_force_coefficient(0.2), 4.0 * 0.04 - 5.9 * 0.2 + 4.0, 10 * tol);   // 2.98
    EXPECT_NEAR(erf_towers::lattice_force_coefficient(0.5), 2.05, 10 * tol);
    TowerType t = lattice();
    EXPECT_NEAR(t.force_coefficient(), 2.98, 10 * tol);
    t.drag_coefficient = 3.4;
    EXPECT_EQ(t.force_coefficient(), Real(3.4)) << "an explicit coefficient wins";
}

TEST(TowerType, EveryValueOutsideItsRangeIsRefusedByName)
{
    EXPECT_TRUE(lattice().validate().empty());
    auto bad = [](auto mutate, const std::string& key) {
        TowerType t = lattice();
        mutate(t);
        const std::string err = t.validate();
        EXPECT_NE(err.find("erf.conductors.lattice." + key), std::string::npos) << key << ": " << err;
    };
    bad([](TowerType& t) { t.base_width = 0.0; }, "base_width");
    bad([](TowerType& t) { t.top_width = 7.0; }, "top_width");
    bad([](TowerType& t) { t.top_width = 0.0; }, "top_width");
    bad([](TowerType& t) { t.solidity = 0.0; }, "solidity");
    bad([](TowerType& t) { t.solidity = 1.0; }, "solidity");
    bad([](TowerType& t) { t.arm_length = 0.0; }, "arm_length");
    bad([](TowerType& t) { t.arm_depth = -1.0; }, "arm_depth");
    bad([](TowerType& t) { t.peak = -1.0; }, "peak");
    bad([](TowerType& t) { t.drag_coefficient = -1.0; }, "drag_coefficient");
    bad([](TowerType& t) { t.segments = 0; }, "segments");
    bad([](TowerType& t) { t.weight = -1.0; }, "weight");
    bad([](TowerType& t) { t.leg_spacing = -1.0; }, "leg_spacing");
    bad([](TowerType& t) { t.allowable_uplift = -1.0; }, "allowable_uplift");
    bad([](TowerType& t) { t.allowable_compression = -1.0; }, "allowable_compression");
    bad([](TowerType& t) { t.frequency = -1.0; }, "frequency");
    bad([](TowerType& t) { t.frequency = 2.0; t.weight = 0.0; }, "frequency");
    bad([](TowerType& t) { t.damping_ratio = -0.1; }, "damping_ratio");
    bad([](TowerType& t) { t.damping_ratio = 1.0; }, "damping_ratio");
    bad([](TowerType& t) { t.foundation_rotational_stiffness = -1.0; }, "foundation_rotational_stiffness");
    bad([](TowerType& t) { t.foundation_lateral_stiffness = -1.0; }, "foundation_lateral_stiffness");
    // NaN passes a test written as x < 0 and infinity one written as x > 0: every key refuses both
    const Real nan = std::numeric_limits<Real>::quiet_NaN(), inf = std::numeric_limits<Real>::infinity();
    for (const Real v : {nan, inf}) {
        bad([v](TowerType& t) { t.base_width = v; }, "base_width");
        bad([v](TowerType& t) { t.top_width = v; }, "top_width");
        bad([v](TowerType& t) { t.solidity = v; }, "solidity");
        bad([v](TowerType& t) { t.arm_length = v; }, "arm_length");
        bad([v](TowerType& t) { t.arm_depth = v; }, "arm_depth");
        bad([v](TowerType& t) { t.peak = v; }, "peak");
        bad([v](TowerType& t) { t.drag_coefficient = v; }, "drag_coefficient");
        bad([v](TowerType& t) { t.weight = v; }, "weight");
        bad([v](TowerType& t) { t.leg_spacing = v; }, "leg_spacing");
        bad([v](TowerType& t) { t.allowable_uplift = v; }, "allowable_uplift");
        bad([v](TowerType& t) { t.allowable_compression = v; }, "allowable_compression");
        bad([v](TowerType& t) { t.frequency = v; }, "frequency");
        bad([v](TowerType& t) { t.damping_ratio = v; }, "damping_ratio");
        bad([v](TowerType& t) { t.foundation_rotational_stiffness = v; }, "foundation_rotational_stiffness");
        bad([v](TowerType& t) { t.foundation_lateral_stiffness = v; }, "foundation_lateral_stiffness");
    }
    TowerType moving = lattice();
    moving.frequency = 2.0;
    moving.weight = 9.0e4;
    EXPECT_TRUE(moving.validate().empty()) << moving.validate();
    EXPECT_TRUE(moving.moves());
    EXPECT_FALSE(lattice().moves()) << "a tower stands still unless it has a frequency";
    TowerType d = lattice();
    d.arm_depth = 0.0;
    EXPECT_EQ(d.arm_face(), d.top_width) << "the arm's face defaults to the top width";
}

TEST(Tower, TheBodyTapersUpToTheCrossArmWhichRunsAcrossTheLine)
{
    TowerType t = lattice();
    t.peak = 4.5;
    const Tower tw("L1_t1", t, {{100.0, 200.0, 50.0}}, 30.0, {{0.0, 1.0, 0.0}});
    // ten body segments of 3 m, two peak segments of 2.25 m, four arm segments of 3 m
    ASSERT_EQ(tw.num_body_nodes(), 12);
    ASSERT_EQ(tw.nodes().size(), 16u);
    const auto& n0 = tw.nodes()[0];
    EXPECT_NEAR(n0.pos[2], 51.5, tol * 100);
    EXPECT_NEAR(n0.length, 3.0, tol * 10);
    EXPECT_NEAR(n0.drag_width, 0.2 * (6.0 - 4.5 * 1.5 / 30.0), tol * 10) << "solidity x width at 1.5 m";
    EXPECT_NEAR(tw.nodes()[11].pos[2], 50.0 + 30.0 + 3.375, tol * 100);
    EXPECT_NEAR(tw.nodes()[11].drag_width, 0.2 * 1.5, tol * 10) << "the peak keeps the top width";
    Real arm = 0.0;
    for (std::size_t i = 12; i < 16; ++i) {
        const auto& n = tw.nodes()[i];
        EXPECT_NEAR(n.pos[2], 80.0, tol * 100);
        EXPECT_NEAR(n.pos[0], 100.0, tol * 100);
        EXPECT_EQ(n.axis, (P3{{0.0, 1.0, 0.0}}));
        EXPECT_NEAR(n.drag_width, 0.2 * 1.2, tol * 10);
        arm += n.pos[1] - 200.0;
    }
    EXPECT_NEAR(arm, 0.0, tol * 1000) << "the arm is centred on the body";
    EXPECT_NEAR(tw.nodes()[12].pos[1], 200.0 - 4.5, tol * 1000);
}

TEST(Tower, InAUniformWindTheDragAndBaseMomentAreTheHandValues)
{
    const TowerType t = lattice();
    const Real H = 30.0, U = 20.0, q = 0.5 * rho * U * U, cf = t.force_coefficient(), phi = t.solidity;
    Tower tw("T", t, {{0.0, 0.0, 10.0}}, H, {{0.0, 1.0, 0.0}});
    // wind along x, across the arm (along y): body and arm both loaded
    blow(tw, [&](Real) { return P3{{U, 0.0, 0.0}}; });
    const Real body = q * cf * phi * 0.5 * (t.base_width + t.top_width) * H;    // exact: the width is linear
    const Real armF = q * cf * phi * t.arm_depth * t.arm_length;
    const auto F = tw.total_force();
    RecordProperty("body_drag_N", std::to_string(body));
    RecordProperty("arm_drag_N", std::to_string(armF));
    EXPECT_NEAR(F[0], body + armF, 1.0e-6 * (body + armF) + tol);
    EXPECT_NEAR(F[1], 0.0, tol); EXPECT_NEAR(F[2], 0.0, tol);
    // the base moment: the body's q cf phi int w(z) z dz, with w linear, and the arm's force times
    // its height. The midpoint sum over n segments is exact on z and short by H^3 / (12 n^2) on z^2.
    const Real n = t.segments;
    const Real body_m = q * cf * phi * (t.base_width * H * H / 2.0 +
                                        (t.top_width - t.base_width) / H * (H * H * H / 3.0 - H * H * H / (12.0 * n * n)));
    const auto M = tw.base_moment();
    EXPECT_NEAR(M[1], body_m + armF * H, 1.0e-6 * (body_m + armF * H) + tol);
    EXPECT_NEAR(M[0], 0.0, 1.0e-6 * M[1] + tol);
    // wind along the arm: only the body carries it
    blow(tw, [&](Real) { return P3{{0.0, U, 0.0}}; });
    EXPECT_NEAR(tw.total_force()[1], body, 1.0e-6 * body + tol);
    EXPECT_NEAR(tw.total_force()[0], 0.0, tol);
}

TEST(Tower, InALogLawWindTheDragIsTheFineIntegral)
{
    const TowerType t = lattice();
    const Real H = 30.0, us = 1.26, z0 = 0.1, kappa = 0.4;
    auto u = [&](Real z) { return us / kappa * std::log((z + z0) / z0); };
    Tower tw("T", t, {{0.0, 0.0, 0.0}}, H, {{0.0, 1.0, 0.0}});
    blow(tw, [&](Real z) { return P3{{u(z), 0.0, 0.0}}; });
    // the body's drag and base moment by a 20000-point midpoint sum
    const Real k = 0.5 * rho * t.force_coefficient() * t.solidity;
    Real drag = 0.0, mom = 0.0;
    const int N = 20000;
    for (int i = 0; i < N; ++i) {
        const Real z = (i + 0.5) * H / N;
        const Real w = t.base_width + (t.top_width - t.base_width) * z / H;
        drag += k * w * u(z) * u(z) * H / N;
        mom += k * w * u(z) * u(z) * z * H / N;
    }
    const Real arm = k * t.arm_depth * t.arm_length * u(H) * u(H);
    // ten segments resolve the log law's curvature near the ground to about 1 %
    EXPECT_NEAR(tw.total_force()[0] / (drag + arm), 1.0, 0.01);
    EXPECT_NEAR(tw.base_moment()[1] / (mom + arm * H), 1.0, 0.01);
}

namespace {
// the legs' reactions balance the tower's loads: their sum is the downward load and their moments
// about the base cancel the loads' overturning moment
void expect_equilibrium (const Tower& tw, const erf_towers::FoundationLoad& L)
{
    const Real a = 0.5 * tw.type().legs();
    const std::array<Real,2> across{{tw.across()[0], tw.across()[1]}}, along{{tw.across()[1], -tw.across()[0]}};
    Real sum = 0.0, mx = 0.0, my = 0.0;
    int i = 0;
    for (const Real su : {Real(1.0), Real(-1.0)}) {
        for (const Real sv : {Real(1.0), Real(-1.0)}) {
            const Real x = a * (su * along[0] + sv * across[0]), y = a * (su * along[1] + sv * across[1]);
            const Real R = L.legs[static_cast<std::size_t>(i++)];
            sum += R; mx += y * R; my -= x * R;
        }
    }
    const Real scale = std::abs(L.vertical) + L.overturning / a + 1.0;
    EXPECT_NEAR(sum, L.vertical, 1.0e2 * tol * scale);
    EXPECT_NEAR(mx, -L.moment[0], 1.0e2 * tol * scale * a);
    EXPECT_NEAR(my, -L.moment[1], 1.0e2 * tol * scale * a);
}
} // namespace

TEST(Tower, TheLegsShareTheLoadAndResistTheOverturningMoment)
{
    TowerType t = lattice();
    t.weight = 9.0e4;
    const Real H = 30.0, a = 3.0;   // legs 6 m apart, the base width
    Tower tw("T", t, {{50.0, 60.0, 10.0}}, H, {{0.0, 1.0, 0.0}});
    // the line pulls 10 kN along x and 8 kN down at the cross-arm, no wind
    const Real f = 1.0e4, down = 8.0e3;
    tw.set_line_load({{f, Real(0.0), Real(-down)}}, {{Real(50.0), Real(60.0), Real(10.0 + H)}});
    auto L = tw.foundation();
    const Real P = t.weight + down;
    EXPECT_NEAR(L.vertical, P, 1.0e2 * tol * P);
    EXPECT_NEAR(L.shear, f, 1.0e2 * tol * f);
    EXPECT_NEAR(L.overturning, H * f, 1.0e2 * tol * H * f);
    // face-on: two legs at P/4 + H f / (4 a), two at P/4 - H f / (4 a)
    EXPECT_NEAR(L.max_compression, P / 4 + H * f / (4 * a), 1.0e2 * tol * P);
    EXPECT_NEAR(L.max_uplift, std::max(Real(0.0), -(P / 4 - H * f / (4 * a))), 1.0e2 * tol * P);
    EXPECT_GT(L.max_uplift, 0.0) << "this pull lifts the windward legs: H f / 4a = 25 kN > P/4 = 24.5 kN";
    expect_equilibrium(tw, L);
    // the same pull along the diagonal: the corner leg takes sqrt(2) times the face-on moment share
    tw.set_line_load({{Real(f / std::sqrt(Real(2.0))), Real(f / std::sqrt(Real(2.0))), Real(-down)}}, {{Real(50.0), Real(60.0), Real(10.0 + H)}});
    L = tw.foundation();
    EXPECT_NEAR(L.max_compression - P / 4, std::sqrt(Real(2.0)) * H * f / (4 * a), 1.0e3 * tol * P);
    expect_equilibrium(tw, L);
    // a tower facing another way carries a pull across its line the same way
    Tower turned("T", t, {{50.0, 60.0, 10.0}}, H, {{1.0, 0.0, 0.0}});
    turned.set_line_load({{Real(0.0), f, Real(-down)}}, {{Real(50.0), Real(60.0), Real(10.0 + H)}});
    const auto Lt = turned.foundation();
    EXPECT_NEAR(Lt.max_compression, P / 4 + H * f / (4 * a), 1.0e2 * tol * P);
    expect_equilibrium(turned, Lt);
    // the wind's drag adds to the line's pull
    blow(tw, [&](Real) { return P3{{20.0, 0.0, 0.0}}; });
    tw.set_line_load({{f, Real(0.0), Real(-down)}}, {{Real(50.0), Real(60.0), Real(10.0 + H)}});
    L = tw.foundation();
    EXPECT_NEAR(L.force[0], f + tw.total_force()[0], 1.0e2 * tol * f);
    EXPECT_NEAR(L.moment[1], H * f + tw.base_moment()[1], 1.0e2 * tol * H * f);
    expect_equilibrium(tw, L);
}

TEST(Tower, EachLegLoadIsFlaggedOverItsAllowable)
{
    TowerType t = lattice();
    t.weight = 9.0e4;
    auto flagged = [&](Real uplift, Real compression) {
        TowerType u = t;
        u.allowable_uplift = uplift;
        u.allowable_compression = compression;
        Tower tw("T", u, {{0.0, 0.0, 0.0}}, 30.0, {{0.0, 1.0, 0.0}});
        tw.set_line_load({{1.0e4, 0.0, -8.0e3}}, {{0.0, 0.0, 30.0}});   // 49.5 kN compression, 0.5 kN uplift
        return tw.foundation().over_allowable;
    };
    EXPECT_FALSE(flagged(0.0, 0.0)) << "no allowable: not checked";
    EXPECT_TRUE(flagged(400.0, 0.0));
    EXPECT_FALSE(flagged(600.0, 0.0));
    EXPECT_TRUE(flagged(0.0, 4.9e4));
    EXPECT_FALSE(flagged(0.0, 5.0e4));
    EXPECT_TRUE(flagged(600.0, 4.9e4)) << "the compression alone is over";
    // legs default to the base width; a type gives its own spacing
    TowerType w = t;
    w.leg_spacing = 8.0;
    EXPECT_EQ(t.legs(), t.base_width);
    EXPECT_EQ(w.legs(), Real(8.0));
}

TEST(Tower, EachLinePullsWhereItHangsAndTheFootingsTakeThemAll)
{
    TowerType t = lattice();
    t.weight = 9.0e4;
    t.peak = 8.0;
    Tower tw("t", t, P3{{500.0, 500.0, 20.0}}, Real(30.0), P3{{0.0, 1.0, 0.0}});
    // three phases across the cross-arm and a shield wire on the peak
    const std::vector<P3> at{{{500.0, 500.0, 50.0}}, {{500.0, 505.5, 50.0}}, {{500.0, 494.5, 50.0}}, {{500.0, 500.0, 57.0}}};
    for (const auto& p : at) { tw.add_attachment(p); }
    EXPECT_EQ(tw.attachments().size(), 4u);
    const std::vector<P3> F{{{0.0, 1000.0, -5000.0}}, {{0.0, 1200.0, -5000.0}}, {{0.0, 800.0, -5000.0}}, {{0.0, 300.0, -1300.0}}};
    tw.set_line_loads(F, at);
    const P3 sum = tw.line_force();
    EXPECT_NEAR(sum[1], 3300.0, tol * 1.0e4);
    EXPECT_NEAR(sum[2], -16300.0, tol * 1.0e5);
    // each pull's own lever arm: M_x = sum (y_i - y_b) F_z - (z_i - z_b) F_y
    double Mx = 0.0;
    for (std::size_t i = 0; i < at.size(); ++i) { Mx += (at[i][1] - 500.0) * F[i][2] - (at[i][2] - 20.0) * F[i][1]; }
    const auto L = tw.foundation();
    EXPECT_NEAR(L.moment[0], Mx, 10.0 * tol * std::abs(Mx));
    EXPECT_NEAR(L.vertical, 9.0e4 + 16300.0, 1.0e-6 * L.vertical);
    expect_equilibrium(tw, L);
}

TEST(Tower, ANonFiniteLinePullIsRefusedNamingTheTower)
{
    TowerType t = lattice();
    t.weight = 9.0e4;
    Tower tw("L1_t3", t, P3{{500.0, 500.0, 20.0}}, Real(30.0), P3{{0.0, 1.0, 0.0}});
    tw.add_attachment(P3{{500.0, 500.0, 50.0}});
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    // a NaN pull, as a diverged MoorDyn line would give: refused, and the pulls already set are kept
    tw.set_line_loads({P3{{0.0, 1000.0, -5000.0}}}, {P3{{500.0, 500.0, 50.0}}});
    std::string msg = erf_gtest::abort_message([&] { tw.set_line_loads({P3{{nan, 0.0, 0.0}}}, {P3{{500.0, 500.0, 50.0}}}); });
    EXPECT_NE(msg.find("L1_t3"), std::string::npos) << msg;
    EXPECT_NE(msg.find("erf.conductors.moordyn_cfl"), std::string::npos) << msg;
    EXPECT_EQ(tw.line_forces()[0][1], Real(1000.0));
    // two pulls for one attachment
    msg = erf_gtest::abort_message([&] {
        tw.set_line_loads({P3{{0.0, 0.0, 0.0}}, P3{{0.0, 0.0, 0.0}}}, {P3{{500.0, 500.0, 50.0}}, P3{{500.0, 500.0, 50.0}}});
    });
    EXPECT_NE(msg.find("2 line pulls for 1 attachments"), std::string::npos) << msg;
}

TEST(Tower, AFlatOrBadlyAimedTowerIsRefusedNamingIt)
{
    const TowerType t = lattice();
    std::string msg = erf_gtest::abort_message([&] { Tower tw("T9", t, P3{{0.0, 0.0, 0.0}}, Real(0.0), P3{{0.0, 1.0, 0.0}}); });
    EXPECT_NE(msg.find("T9"), std::string::npos) << msg;
    EXPECT_NE(msg.find("cross-arm height"), std::string::npos) << msg;
    msg = erf_gtest::abort_message([&] { Tower tw("T9", t, P3{{0.0, 0.0, 0.0}}, Real(30.0), P3{{0.0, 0.6, 0.8}}); });
    EXPECT_NE(msg.find("horizontal unit vector"), std::string::npos) << msg;
    TowerType bad = t;
    bad.solidity = 0.0;
    msg = erf_gtest::abort_message([&] { Tower tw("T9", bad, P3{{0.0, 0.0, 0.0}}, Real(30.0), P3{{0.0, 1.0, 0.0}}); });
    EXPECT_NE(msg.find("erf.conductors.lattice.solidity"), std::string::npos) << msg;
    EXPECT_TRUE(erf_gtest::abort_message([&] { Tower tw("T9", t, P3{{0.0, 0.0, 0.0}}, Real(30.0), P3{{0.0, 1.0, 0.0}}); }).empty());
}
