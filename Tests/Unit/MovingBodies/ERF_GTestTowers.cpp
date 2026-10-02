// Contract of the tower aerodynamics: the drag on a member node is 1/2 rho Cd w L |U_n| U_n with
// only the flow normal to the member and relative to it counting; the lattice force coefficient
// is ASCE 7's square-tower curve; a tower type refuses every value outside its range by key; a
// tower stands its body's nodes up the tapering body and its cross-arm across the line at the
// conductor's height; and in a uniform wind the drag and base moment are the hand values, the
// body's drag exactly (its width is linear in height), in a log-law wind within the segments'
// quadrature error of a fine integral.

#include <array>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

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

constexpr Real tol = std::is_same<Real, float>::value ? Real(1.0e-5) : Real(1.0e-12);
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
