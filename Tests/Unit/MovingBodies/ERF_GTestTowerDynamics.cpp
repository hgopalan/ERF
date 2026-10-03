// Contract of the one-mode tower: on a rigid foundation it sways at its type's frequency, its mass
// the type's weight, its shape the cantilever's (z/H)^2 with the cross-arm at 1; a load held over a
// step is integrated exactly, so the result does not depend on how the time is cut and any step is
// stable; free, it rings down at the damped frequency with the logarithmic decrement of its damping
// ratio; the foundation's tilt and slide add their compliances to the bending's in series, so a
// load at the cross-arm deflects it by F (1/K_b + H^2/k_r + 1/k_l); a load spread up the body
// drives it by the shape, the hand value of the midpoint sum of p (z/H)^2; the foundation takes the
// loads less the nodes' inertia, so swaying freely its base shear is omega^2 q sum m phi; swaying
// in a steady wind, the drag on the members relative to their own velocity damps it by
// sum phi^2 rho Cf w L U / (2 M omega) more; its state restores the same motion; and a line on
// the peak moves, and pulls, by the shape at its height.

#include <array>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ERF_MemberDrag.H"
#include "ERF_Tower.H"
#include "ERF_TowerDynamics.H"
#include "ERF_TowerInputs.H"

using amrex::Real;
using erf_towers::MemberDrag;
using erf_towers::OneModeTower;
using erf_towers::Tower;
using erf_towers::TowerType;
using P3 = std::array<Real,3>;

namespace {

constexpr double pi = 3.14159265358979323846;
constexpr double g = 9.81;
// the model integrates in double whatever Real is; what reaches it as Real carries Real's roundoff
constexpr double tol = std::is_same<Real, float>::value ? 1.0e-5 : 1.0e-12;

TowerType swaying (Real f = 2.0, Real zeta = 0.02)
{
    TowerType t;
    t.name = "lattice";
    t.base_width = 6.0; t.top_width = 1.5; t.solidity = 0.2; t.arm_length = 12.0; t.arm_depth = 1.2;
    t.weight = 9.0e4; t.frequency = f; t.damping_ratio = zeta;
    return t;
}

Tower standing (const TowerType& t)
{
    return Tower("t", t, P3{{500.0, 500.0, 20.0}}, Real(30.0), P3{{0.0, 1.0, 0.0}});
}

std::vector<Real> no_load (const Tower& tw) { return std::vector<Real>(3 * tw.nodes().size(), Real(0.0)); }

} // namespace

TEST(OneModeTower, OnARigidFoundationItSwaysAtItsTypesFrequencyInTheCantileverShape)
{
    const Tower tw = standing(swaying());
    const OneModeTower m(tw, Real(g));
    EXPECT_NEAR(m.frequency() / 2.0, 1.0, tol);
    EXPECT_NEAR(m.stiffness() / m.bending_stiffness(), 1.0, 1.0e-14) << "nothing but the bending on a rigid foundation";
    double mass = 0.0, M = 0.0;
    for (std::size_t i = 0; i < tw.nodes().size(); ++i) {
        const double z = (tw.nodes()[i].pos[2] - tw.base()[2]) / 30.0;
        EXPECT_NEAR(m.mode_shape(i), z * z, 1.0e-14) << i;
        mass += m.node_mass(i);
        M += m.node_mass(i) * z * z * z * z;
    }
    EXPECT_NEAR(mass, 9.0e4 / g, tol * mass) << "the type's weight over gravity";
    EXPECT_NEAR(m.generalized_mass(), M, 1.0e-9);
    EXPECT_NEAR(m.stiffness(), M * std::pow(2.0 * pi * 2.0, 2), 1.0e-6);
    // the cross-arm's nodes are at its height: they move as the cross-arm, phi = 1
    for (std::size_t i = static_cast<std::size_t>(tw.num_body_nodes()); i < tw.nodes().size(); ++i) {
        EXPECT_NEAR(m.mode_shape(i), 1.0, 1.0e-14) << i;
    }
}

TEST(OneModeTower, AHeldLoadIsIntegratedExactlyHoweverTheTimeIsCut)
{
    const Tower tw = standing(swaying());
    const P3 F{{Real(1000.0), Real(-500.0), Real(0.0)}};
    OneModeTower one(tw, Real(g)), many(tw, Real(g));
    one.step(Real(0.75), no_load(tw), F);
    for (int s = 0; s < 6; ++s) { many.step(Real(0.125), no_load(tw), F); }
    // from rest, q(t) = F/K (1 - e^(-zeta w t) (cos w_d t + zeta w / w_d sin w_d t))
    const double K = one.stiffness(), w = 2.0 * pi * 2.0, z = 0.02, wd = w * std::sqrt(1.0 - z * z), t = 0.75;
    const double shape = 1.0 - std::exp(-z * w * t) * (std::cos(wd * t) + z * w / wd * std::sin(wd * t));
    for (int d = 0; d < 2; ++d) {
        const double exact = F[d] / K * shape;
        EXPECT_NEAR(one.q()[d], exact, tol * std::abs(F[d] / K)) << d;
        EXPECT_NEAR(many.q()[d], exact, tol * std::abs(F[d] / K)) << d;
        EXPECT_NEAR(many.v()[d], one.v()[d], tol * std::abs(F[d] / K) * w) << d;
    }
    // a step of a thousand periods is stable: it lands on the static deflection (e^(-zeta w t) = e^(-126))
    OneModeTower big(tw, Real(g));
    big.step(Real(500.0), no_load(tw), F);
    EXPECT_NEAR(big.q()[0], F[0] / K, 1.0e-6 * F[0] / K);
    EXPECT_NEAR(big.v()[0], 0.0, 1.0e-6 * F[0] / K * w);
}

TEST(OneModeTower, FreeItRingsDownAtTheDampedFrequencyWithItsDecrement)
{
    const double f = 1.5, z = 0.05;
    const Tower tw = standing(swaying(Real(f), Real(z)));
    OneModeTower m(tw, Real(g));
    ASSERT_TRUE(m.set_state({0.1, 0.0, 0.0, 0.0, 0.0, 0.0}));   // displaced 0.1 m in x, at rest
    const double dt = 1.0 / (f * 64.0);
    std::vector<double> peaks, crossings;
    double prev = m.q()[0], prev2 = prev, t = 0.0;
    for (int s = 0; s < 64 * 6; ++s) {
        m.step(Real(dt), no_load(tw), P3{{0.0, 0.0, 0.0}});
        t += dt;
        const double q = m.q()[0];
        if (prev > 0.0 && q <= 0.0) { crossings.push_back(t - dt + dt * prev / (prev - q)); }
        if (s > 1 && prev > prev2 && prev > q) { peaks.push_back(prev); }
        prev2 = prev; prev = q;
    }
    ASSERT_GE(crossings.size(), 5u);
    ASSERT_GE(peaks.size(), 4u);
    const double Td = (crossings.back() - crossings.front()) / (crossings.size() - 1);
    EXPECT_NEAR(Td * f * std::sqrt(1.0 - z * z), 1.0, 1.0e-4) << "the damped period";
    const double decrement = std::log(peaks.front() / peaks.back()) / (peaks.size() - 1);
    EXPECT_NEAR(decrement / (2.0 * pi * z / std::sqrt(1.0 - z * z)), 1.0, 1.0e-3) << "the logarithmic decrement";
}

TEST(OneModeTower, TheFoundationsTiltAndSlideAddTheirCompliancesInSeries)
{
    TowerType t = swaying();
    t.foundation_rotational_stiffness = 2.0e8;
    t.foundation_lateral_stiffness = 5.0e6;
    const Tower tw = standing(t);
    const OneModeTower rigid(standing(swaying()), Real(g));
    OneModeTower soft(tw, Real(g));
    const double Kb = rigid.stiffness(), H = 30.0;
    const double c = 1.0 / Kb + H * H / 2.0e8 + 1.0 / 5.0e6;
    EXPECT_NEAR(soft.bending_stiffness() / Kb, 1.0, 1.0e-12) << "the bending is the rigid foundation's";
    EXPECT_NEAR(soft.stiffness() * c, 1.0, 1.0e-12);
    // a load at the cross-arm, held: the tower comes to rest at F c
    const P3 F{{Real(2000.0), Real(0.0), Real(0.0)}};
    soft.step(Real(1000.0), no_load(tw), F);
    EXPECT_NEAR(soft.q()[0] / (2000.0 * c), 1.0, tol);
    // the shape: the slide moves the base, the tilt adds a straight line, the bending the parabola
    const double ab = (1.0 / Kb) / c, ar = (H * H / 2.0e8) / c, al = (1.0 / 5.0e6) / c;
    for (std::size_t i = 0; i < tw.nodes().size(); ++i) {
        const double z = (tw.nodes()[i].pos[2] - tw.base()[2]) / H;
        EXPECT_NEAR(soft.mode_shape(i), ab * z * z + ar * z + al, 1.0e-12) << i;
    }
    // softer, it sways slower: omega^2 = K / M with the shape's own generalized mass
    EXPECT_LT(soft.frequency(), rigid.frequency());
    EXPECT_NEAR(soft.frequency(), std::sqrt(soft.stiffness() / soft.generalized_mass()) / (2.0 * pi), tol);
    // a stiff enough foundation is a rigid one
    TowerType stiff = swaying();
    stiff.foundation_rotational_stiffness = 1.0e20;
    stiff.foundation_lateral_stiffness = 1.0e20;
    EXPECT_NEAR(OneModeTower(standing(stiff), Real(g)).frequency() / rigid.frequency(), 1.0, 1.0e-6);
}

TEST(OneModeTower, ALoadSpreadUpTheBodyDrivesItByTheShape)
{
    const Tower tw = standing(swaying());
    OneModeTower m(tw, Real(g));
    // p = 100 N/m in y on the body below the cross-arm, nothing on the arm
    const int n = tw.type().segments;
    const double p = 100.0, H = 30.0, dz = H / n;
    std::vector<Real> f = no_load(tw);
    for (int i = 0; i < n; ++i) { f[3 * static_cast<std::size_t>(i) + 1] = static_cast<Real>(p * dz); }
    m.step(Real(1000.0), f, P3{{0.0, 0.0, 0.0}});
    // the midpoint sum of p (z/H)^2 dz over n segments is p H (1/3 - 1/(12 n^2))
    const double Q = p * H * (1.0 / 3.0 - 1.0 / (12.0 * n * n));
    EXPECT_NEAR(m.q()[1] * m.stiffness() / Q, 1.0, tol);
    EXPECT_NEAR(m.q()[0], 0.0, 1.0e-15);
}

TEST(OneModeTower, TheFoundationTakesTheLoadsLessTheInertia)
{
    Tower tw = standing(swaying());
    OneModeTower m(tw, Real(g));
    auto hand_over = [&] () {
        std::vector<Real> x(3 * tw.nodes().size()), v(x.size()), a(x.size());
        for (std::size_t i = 0; i < tw.nodes().size(); ++i) {
            const auto xi = m.displacement(i), vi = m.velocity(i), ai = m.inertial_force(i);
            for (int d = 0; d < 3; ++d) { x[3*i+d] = xi[d]; v[3*i+d] = vi[d]; a[3*i+d] = ai[d]; }
        }
        tw.set_motion(x, v, a);
    };
    // swaying freely at its extreme, q = 0.05 m, at rest: every node accelerates back at omega^2 phi q
    ASSERT_TRUE(m.set_state({0.05, 0.0, 0.0, 0.0, 0.0, 0.0}));
    hand_over();
    double sm = 0.0;
    for (std::size_t i = 0; i < tw.nodes().size(); ++i) { sm += m.node_mass(i) * m.mode_shape(i); }
    const double w = 2.0 * pi * 2.0;
    const auto L = tw.foundation();
    EXPECT_NEAR(L.force[0] / (w * w * 0.05 * sm), 1.0, tol) << "the base shear of the sway";
    EXPECT_NEAR(L.force[1], 0.0, 1.0e-9);
    EXPECT_NEAR(L.vertical, 9.0e4, 1.0e-9 * 9.0e4) << "the sway is horizontal";
    EXPECT_NEAR(tw.arm_displacement()[0], 0.05, tol * 0.05);
    EXPECT_NEAR(tw.current_nodes().back().pos[0], tw.nodes().back().pos[0] + Real(0.05), 1.0e-4);
    // at rest under a held load the inertia is gone: the foundation takes the load, as a rigid tower's
    const P3 F{{Real(3000.0), Real(0.0), Real(-5000.0)}};
    m.step(Real(1000.0), no_load(tw), F);
    hand_over();
    tw.set_line_load(F, P3{{500.0, 500.0, 50.0}});
    const auto R = tw.foundation();
    EXPECT_NEAR(R.force[0] / 3000.0, 1.0, 1.0e-6);
    EXPECT_NEAR(R.overturning / (3000.0 * 30.0), 1.0, 1.0e-6);
}

TEST(OneModeTower, TheWindDampsItsSwayByTheDragRelativeToTheMembers)
{
    const double f = 2.0, z = 0.005, U = 20.0, rho = 1.2;
    Tower tw = standing(swaying(Real(f), Real(z)));
    OneModeTower m(tw, Real(g));
    const MemberDrag aero{Real(rho)};
    // linearised, the drag 1/2 rho Cf w L (U - v)^2 on a node moving at v with the wind damps by
    // rho Cf w L U: the mode by sum phi^2 rho Cf w L U, a damping ratio of that over 2 M omega
    double C = 0.0;
    for (std::size_t i = 0; i < tw.nodes().size(); ++i) {
        const auto& n = tw.nodes()[i];
        C += m.mode_shape(i) * m.mode_shape(i) * rho * n.drag_coefficient * n.drag_width * n.length * U;
    }
    const double w = 2.0 * pi * f, za = C / (2.0 * m.generalized_mass() * w);
    std::vector<Real> wind;
    for (std::size_t i = 0; i < tw.nodes().size(); ++i) { wind.insert(wind.end(), {Real(U), Real(0.0), Real(0.0)}); }
    // settle on the mean drag, then sway 2 cm about it
    std::vector<Real> load;
    aero.loads(tw.nodes(), wind, std::vector<Real>(wind.size(), Real(0.0)), load);
    m.step(Real(1000.0), load, P3{{0.0, 0.0, 0.0}});
    const double mean = m.q()[0];
    ASSERT_TRUE(m.set_state({mean + 0.02, 0.0, 0.0, 0.0, 0.0, 0.0}));
    const double dt = 1.0 / (f * 200.0);
    std::vector<double> peaks;
    double prev = 0.02, prev2 = 0.02;
    for (int s = 0; s < 200 * 8; ++s) {
        std::vector<Real> vel(wind.size());
        for (std::size_t i = 0; i < tw.nodes().size(); ++i) { const auto v = m.velocity(i); for (int d = 0; d < 3; ++d) { vel[3*i+d] = v[d]; } }
        aero.loads(tw.nodes(), wind, vel, load);
        m.step(Real(dt), load, P3{{0.0, 0.0, 0.0}});
        const double q = m.q()[0] - mean;
        if (s > 1 && prev > prev2 && prev > q) { peaks.push_back(prev); }
        prev2 = prev; prev = q;
    }
    ASSERT_GE(peaks.size(), 6u);
    const double decrement = std::log(peaks.front() / peaks.back()) / (peaks.size() - 1);
    RecordProperty("aerodynamic_damping_ratio", std::to_string(za));
    RecordProperty("measured_damping_ratio", std::to_string(decrement / (2.0 * pi)));
    EXPECT_GT(za, 0.5 * z) << "the test needs the wind's damping to matter";
    EXPECT_NEAR(decrement / (2.0 * pi) / (z + za), 1.0, 0.02);
}

TEST(OneModeTower, ItsStateRestoresTheSameMotion)
{
    const Tower tw = standing(swaying());
    const P3 F{{Real(1000.0), Real(700.0), Real(0.0)}};
    OneModeTower a(tw, Real(g)), b(tw, Real(g));
    for (int s = 0; s < 5; ++s) { a.step(Real(0.05), no_load(tw), F); }
    ASSERT_TRUE(b.set_state(a.state()));
    EXPECT_FALSE(b.set_state({1.0, 2.0}));
    for (int s = 0; s < 5; ++s) { a.step(Real(0.05), no_load(tw), F); b.step(Real(0.05), no_load(tw), F); }
    for (int d = 0; d < 2; ++d) { EXPECT_EQ(a.q()[d], b.q()[d]); EXPECT_EQ(a.v()[d], b.v()[d]); }
    for (std::size_t i = 0; i < tw.nodes().size(); ++i) { EXPECT_EQ(a.inertial_force(i)[0], b.inertial_force(i)[0]); }
}

TEST(OneModeTower, ALineOnThePeakMovesAndPullsByTheShapeThere)
{
    TowerType t = swaying();
    t.peak = 8.0;
    Tower tw = standing(t);
    // a phase at the cross-arm (phi = 1) and a shield wire 7 m above it on the peak
    tw.add_attachment(P3{{500.0, 500.0, 50.0}});
    tw.add_attachment(P3{{500.0, 500.0, 57.0}});
    OneModeTower m(tw, Real(g));
    ASSERT_EQ(m.num_attachments(), 2u);
    const double z = 37.0 / 30.0;
    EXPECT_NEAR(m.attachment_shape(0), 1.0, 1.0e-14);
    EXPECT_NEAR(m.attachment_shape(1), z * z, 1.0e-12) << "the bending shape on the peak";
    // held loads at both: the generalized force weights each by its shape
    const std::vector<P3> F{{{Real(1000.0), 0.0, 0.0}}, {{Real(400.0), 0.0, 0.0}}};
    m.step(Real(1000.0), no_load(tw), F);
    EXPECT_NEAR(m.q()[0] * m.stiffness() / (1000.0 + z * z * 400.0), 1.0, tol);
    EXPECT_NEAR(m.attachment_displacement(1)[0] / m.attachment_displacement(0)[0], z * z, tol);
    // a tower with no attachment takes its line at the cross-arm
    OneModeTower bare(standing(t), Real(g));
    EXPECT_EQ(bare.num_attachments(), 1u);
    EXPECT_NEAR(bare.attachment_shape(0), 1.0, 1.0e-14);
}
