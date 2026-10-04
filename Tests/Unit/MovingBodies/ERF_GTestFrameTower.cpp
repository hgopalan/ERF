// A lattice tower that bends as its frame model (FrameTower), and the rigid links that tie the
// tower's drag nodes and attachments to the frame. The frame is case T of
// Tests/test_files/FrameSubDynTower: the Conductors_FrameTowers deck's tower in tower-local axes.
//
// RigidLink.KeepsTheForceAndItsMomentAndDoesNoWork: a force shared among a link's nodes keeps its
//   sum and its moment, the link's motion is the load sharing's adjoint (F . u_p = sum F_i . u_i), and
//   a point on a node is that node alone.
// FrameTower.RingsAtSubDynsFirstFrequency: the frame's first natural frequency is SubDyn's.
// FrameTower.SettlesAtTheFramesStaticResponse: a damped tower turned 30 degrees, under steady drag
//   and a line's pull, comes to rest where the frame's static solution of the linked loads puts its
//   cross-arm and drag nodes.
// FrameTower.ItsFootingsCarryTheLoadsAndTheWeight: at rest, the footings' force and moment are the
//   applied loads' (ERF axes, about the base centre), the legs carry the weight and the vertical
//   loads, and the legs on the downwind side are the more compressed.
// FrameTower.BeforeItsFirstStepTheFootingsTakeTheStaticLoads: at rest, the footings take the tower's
//   present drag and line pull exactly, as Tower::foundation() reckons a tower at rest.
// FrameTower.TheStateRestoresTheSameMotion: a state saved and restored continues bit for bit.
// FrameTower.RefusesAFrameThatDoesNotFitItsTowerOrItsType: a frame whose cross-arm is far from the
//   lines' attachment aborts naming the frame file; a type giving a frame and a frequency, a weight or
//   a foundation stiffness is refused.
// FrameTower.ItsMembersAreCheckedUnderTheFramesForces: on a generated frame with design data, the
//   members' checks before the first step and once settled under steady loads are those of the
//   frame's static solution under the linked loads and the weight; without design data, none.
// FrameTower.HeatingKeepsItsPlaceThenSettlesOnTheSofterFrame: heated to 600 C while settled, the
//   cross-arm stays put, the first frequency drops by sqrt(k_E), and the tower settles where the hot
//   frame's static solution puts it, its checks those of the hot steel; bad temperatures are refused.
// FrameTower.ItsStateCarriesTheTemperatures: a cold tower restored from a heated one's state heats up
//   and continues bit for bit; a state without temperatures keeps the present ones.
// TowerType.RefusesFrameKeysThatDoNotFit: frame_panels with frame_file, frame keys without their frame,
//   angles without two values in order, an unknown bracing, a bad yield strength or temperature.

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ERF_FrameTower.H"
#include "ERF_LatticeFrame.H"
#include "ERF_GTestThrowOnAbort.H"

using amrex::Real;
using namespace erf_towers;
using P3 = std::array<Real,3>;

namespace {

constexpr double g = 9.80665;
// loads reach the tower as Real; the frame works in double
constexpr double tol = (std::is_same<Real, float>::value) ? 2.0e-5 : 1.0e-9;

std::string frame_file () { return std::string(ERF_FRAME_TEST_FILES) + "/caseT/towerT.dat"; }

std::shared_ptr<const Frame> frame_t ()
{
    FrameInputs in;
    const std::string rerr = read_subdyn(frame_file(), in);
    EXPECT_TRUE(rerr.empty()) << rerr;
    std::string err;
    std::shared_ptr<const Frame> f = Frame::create(in, err);
    EXPECT_TRUE(f) << err;
    return f;
}

TowerType framed (double zeta = 0.02)
{
    TowerType t;
    t.name = "lattice";
    t.base_width = 6.0; t.top_width = 1.5; t.solidity = 0.2; t.arm_length = 12.0; t.arm_depth = 1.2;
    t.damping_ratio = static_cast<Real>(zeta);
    t.frame_file = frame_file();
    return t;
}

/** A tower of case T's geometry at (100, 200, 50), its cross-arm turned 30 degrees from y, one line on the cross-arm's centre. */
Tower turned (const TowerType& t, Real arm_height = 30.0)
{
    const double c = std::cos(0.5235987755982988), s = std::sin(0.5235987755982988);
    Tower tw("t1", t, P3{{100.0, 200.0, 50.0}}, arm_height, P3{{static_cast<Real>(-s), static_cast<Real>(c), 0.0}});
    tw.add_attachment(P3{{100.0, 200.0, static_cast<Real>(50.0 + arm_height)}});
    return tw;
}

std::vector<double> subdyn_frequencies ()
{
    std::ifstream f(std::string(ERF_FRAME_TEST_FILES) + "/caseT/towerT_frequencies.txt");
    std::vector<double> v;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') { continue; }
        std::istringstream is(line);
        double x = 0.0;
        while (is >> x) { v.push_back(x); }
    }
    return v;
}

/** Steady drag on every drag node (N, ERF axes) and a line's pull at the attachment. */
std::vector<Real> drag (const Tower& tw)
{
    std::vector<Real> f(3 * tw.nodes().size());
    for (std::size_t i = 0; i < tw.nodes().size(); ++i) { f[3 * i] = 150.0; f[3 * i + 1] = -60.0; f[3 * i + 2] = 5.0; }
    return f;
}
const std::array<Real,3> pull{{Real(1.2e4), Real(-3.0e3), Real(-2.0e4)}};

} // namespace

TEST(RigidLink, KeepsTheForceAndItsMomentAndDoesNoWork)
{
    const auto f = frame_t();
    ASSERT_TRUE(f);
    const std::array<double,3> p{{0.3, -0.2, 11.0}}, F{{2.0e3, -7.0e2, 4.0e2}};
    const RigidLink link(*f, p);
    ASSERT_EQ(link.nodes().size(), 4u);
    std::vector<double> loads(f->num_dofs(), 0.0);
    link.add_load(F, loads);
    std::array<double,3> sum{{0, 0, 0}}, mom{{0, 0, 0}};
    for (std::size_t n = 0; n < f->num_nodes(); ++n) {
        const auto& x = f->node_position(n);
        const double* q = &loads[6 * n];
        for (int d = 0; d < 3; ++d) { sum[static_cast<std::size_t>(d)] += q[d]; }
        mom[0] += x[1] * q[2] - x[2] * q[1];
        mom[1] += x[2] * q[0] - x[0] * q[2];
        mom[2] += x[0] * q[1] - x[1] * q[0];
    }
    const std::array<double,3> pm{{p[1] * F[2] - p[2] * F[1], p[2] * F[0] - p[0] * F[2], p[0] * F[1] - p[1] * F[0]}};
    for (std::size_t d = 0; d < 3; ++d) {
        EXPECT_NEAR(sum[d], F[d], 1e-9 * 2.0e3);
        EXPECT_NEAR(mom[d], pm[d], 1e-9 * 2.0e4);
    }
    // the motion is the adjoint of the load sharing
    std::vector<double> u(f->num_dofs());
    for (std::size_t i = 0; i < u.size(); ++i) { u[i] = std::sin(0.37 * static_cast<double>(i) + 0.1); }
    const auto up = link.motion(u);
    double work_point = 0.0, work_nodes = 0.0;
    for (std::size_t d = 0; d < 3; ++d) { work_point += F[d] * up[d]; }
    for (std::size_t i = 0; i < u.size(); ++i) { work_nodes += loads[i] * u[i]; }
    EXPECT_NEAR(work_point, work_nodes, 1e-9 * std::abs(work_nodes) + 1e-9);
    // a point on a node is that node
    const RigidLink on(*f, f->node_position(f->node_of_joint(21)));
    ASSERT_EQ(on.nodes().size(), 1u);
    EXPECT_EQ(on.nodes()[0], f->node_of_joint(21));
    const auto uo = on.motion(u);
    for (std::size_t d = 0; d < 3; ++d) { EXPECT_DOUBLE_EQ(uo[d], u[6 * f->node_of_joint(21) + d]); }
}

TEST(FrameTower, RingsAtSubDynsFirstFrequency)
{
    const Tower tw = turned(framed());
    const FrameTower ft(tw, frame_t(), g);
    const std::vector<double> theirs = subdyn_frequencies();
    ASSERT_FALSE(theirs.empty());
    EXPECT_NEAR(static_cast<double>(ft.frequency()), theirs[0], 1e-6 * theirs[0]);
    EXPECT_NEAR(ft.mass(), 6800.110, 0.01);   // SubDyn's total mass of case T (kg)
}

TEST(FrameTower, SettlesAtTheFramesStaticResponse)
{
    const Tower tw = turned(framed(0.3));
    const auto frame = frame_t();
    FrameTower ft(tw, frame, g);
    const std::vector<Real> fd = drag(tw);
    const double h = 1.0 / (20.0 * static_cast<double>(ft.frequency()));
    for (int n = 0; n < 1500; ++n) { ft.step(static_cast<Real>(h), fd, std::vector<std::array<Real,3>>{pull}); }
    // the frame's static solution of the same linked loads, in tower-local axes
    const double c = std::cos(0.5235987755982988), s = std::sin(0.5235987755982988);
    const std::array<double,3> along{{c, s, 0.0}}, across{{-s, c, 0.0}};
    auto local = [&] (double x, double y, double z) { return std::array<double,3>{{x * along[0] + y * along[1], x * across[0] + y * across[1], z}}; };
    std::vector<double> loads(frame->num_dofs(), 0.0);
    for (std::size_t i = 0; i < tw.nodes().size(); ++i) {
        const auto& p = tw.nodes()[i].pos;
        RigidLink(*frame, local(p[0] - 100.0, p[1] - 200.0, p[2] - 50.0)).add_load(local(fd[3 * i], fd[3 * i + 1], fd[3 * i + 2]), loads);
    }
    const RigidLink arm(*frame, {{0.0, 0.0, 30.0}});
    arm.add_load(local(pull[0], pull[1], pull[2]), loads);
    const FrameSolution st = frame->solve(loads, 0.0);
    const auto ul = arm.motion(st.displacement);
    const std::array<double,3> expected{{ul[0] * along[0] + ul[1] * across[0], ul[0] * along[1] + ul[1] * across[1], ul[2]}};
    const auto got = ft.attachment_displacement(0);
    const double scale = std::sqrt(expected[0] * expected[0] + expected[1] * expected[1] + expected[2] * expected[2]);
    ASSERT_GT(scale, 1.0e-4) << "the cross-arm must have moved: the check is not vacuous";
    for (std::size_t d = 0; d < 3; ++d) { EXPECT_NEAR(static_cast<double>(got[d]), expected[d], 1e-6 * scale + tol * scale) << d; }
    // and at rest: the velocity is a small part of the sway's amplitude times its angular frequency
    const double vscale = scale * 2.0 * 3.14159265358979323846 * static_cast<double>(ft.frequency());
    for (std::size_t d = 0; d < 3; ++d) { EXPECT_NEAR(static_cast<double>(ft.attachment_velocity(0)[d]), 0.0, 1e-4 * vscale); }
}

TEST(FrameTower, ItsFootingsCarryTheLoadsAndTheWeight)
{
    const Tower tw = turned(framed(0.3));
    FrameTower ft(tw, frame_t(), g);
    const std::vector<Real> fd = drag(tw);
    const double h = 1.0 / (20.0 * static_cast<double>(ft.frequency()));
    for (int n = 0; n < 1500; ++n) { ft.step(static_cast<Real>(h), fd, std::vector<std::array<Real,3>>{pull}); }
    FoundationLoad L;
    ASSERT_TRUE(ft.foundation(tw, L));
    // the applied loads and their moment about the base centre
    std::array<double,3> F{{0, 0, 0}}, M{{0, 0, 0}};
    auto add = [&] (const std::array<double,3>& r, const std::array<double,3>& f) {
        for (std::size_t d = 0; d < 3; ++d) { F[d] += f[d]; }
        M[0] += r[1] * f[2] - r[2] * f[1];
        M[1] += r[2] * f[0] - r[0] * f[2];
        M[2] += r[0] * f[1] - r[1] * f[0];
    };
    for (std::size_t i = 0; i < tw.nodes().size(); ++i) {
        const auto& p = tw.nodes()[i].pos;
        add({{p[0] - 100.0, p[1] - 200.0, p[2] - 50.0}}, {{fd[3 * i], fd[3 * i + 1], fd[3 * i + 2]}});
    }
    add({{0.0, 0.0, 30.0}}, {{pull[0], pull[1], pull[2]}});
    for (std::size_t d = 0; d < 3; ++d) {
        EXPECT_NEAR(static_cast<double>(L.force[d]), F[d], 1e-5 * 2.0e4) << d;
        EXPECT_NEAR(static_cast<double>(L.moment[d]), M[d], 1e-5 * 6.0e5) << d;
    }
    const double weight = ft.mass() * g;
    EXPECT_NEAR(static_cast<double>(L.vertical), weight - F[2], 1e-5 * weight);
    // the load leans along +along (cos 30 * 150 - sin 30 * 60 > 0 per node, and the pull): legs 0 and 1 are downwind
    EXPECT_GT(std::min(L.legs[0], L.legs[1]), std::max(L.legs[2], L.legs[3]));
}

TEST(FrameTower, BeforeItsFirstStepTheFootingsTakeTheStaticLoads)
{
    Tower tw = turned(framed());
    const FrameTower ft(tw, frame_t(), g);
    const std::vector<Real> fd = drag(tw);
    tw.set_loads(fd);
    tw.set_line_loads({pull}, {tw.attachments()[0]});
    FoundationLoad L;
    ASSERT_TRUE(ft.foundation(tw, L));
    std::array<double,3> F{{static_cast<double>(pull[0]), static_cast<double>(pull[1]), static_cast<double>(pull[2])}};
    std::array<double,3> M{{-30.0 * static_cast<double>(pull[1]), 30.0 * static_cast<double>(pull[0]), 0.0}};
    for (std::size_t i = 0; i < tw.nodes().size(); ++i) {
        const auto& p = tw.nodes()[i].pos;
        const std::array<double,3> r{{p[0] - 100.0, p[1] - 200.0, p[2] - 50.0}}, f{{fd[3 * i], fd[3 * i + 1], fd[3 * i + 2]}};
        for (std::size_t d = 0; d < 3; ++d) { F[d] += f[d]; }
        M[0] += r[1] * f[2] - r[2] * f[1];
        M[1] += r[2] * f[0] - r[0] * f[2];
        M[2] += r[0] * f[1] - r[1] * f[0];
    }
    for (std::size_t d = 0; d < 3; ++d) {
        EXPECT_NEAR(static_cast<double>(L.force[d]), F[d], tol * 2.0e4 + 1e-9 * 2.0e4) << d;
        EXPECT_NEAR(static_cast<double>(L.moment[d]), M[d], tol * 6.0e5 + 1e-9 * 6.0e5) << d;
    }
    EXPECT_NEAR(static_cast<double>(L.vertical), ft.mass() * g - F[2], 1e-6 * ft.mass() * g);
}

TEST(FrameTower, TheStateRestoresTheSameMotion)
{
    const Tower tw = turned(framed());
    const auto frame = frame_t();
    FrameTower a(tw, frame, g), b(tw, frame, g);
    const std::vector<Real> fd = drag(tw);
    for (int n = 0; n < 10; ++n) { a.step(Real(0.01), fd, std::vector<std::array<Real,3>>{pull}); }
    ASSERT_TRUE(b.set_state(a.state()));
    for (int n = 0; n < 10; ++n) {
        a.step(Real(0.01), fd, std::vector<std::array<Real,3>>{pull});
        b.step(Real(0.01), fd, std::vector<std::array<Real,3>>{pull});
    }
    EXPECT_EQ(a.state(), b.state());
    std::vector<double> bad = a.state();
    bad.pop_back();
    EXPECT_FALSE(b.set_state(bad));
}

TEST(FrameTower, RefusesAFrameThatDoesNotFitItsTowerOrItsType)
{
    // the line hangs 60 m up, the frame's cross-arm is at 30 m
    const Tower tall = turned(framed(), Real(60.0));
    const std::string msg = erf_gtest::abort_message([&] { FrameTower ft(tall, frame_t(), g); });
    EXPECT_NE(msg.find("frame_file"), std::string::npos) << msg;
    EXPECT_NE(msg.find("from the nearest node of"), std::string::npos) << msg;
    TowerType t = framed();
    EXPECT_TRUE(t.validate().empty()) << t.validate();
    t.frequency = 2.0;
    EXPECT_NE(t.validate().find("frequency is not given with"), std::string::npos) << t.validate();
    t = framed();
    t.weight = 6.0e4;
    EXPECT_NE(t.validate().find("weight is not given with"), std::string::npos) << t.validate();
    t = framed();
    t.foundation_rotational_stiffness = 1.0e9;
    EXPECT_NE(t.validate().find("foundation_rotational_stiffness is not given with"), std::string::npos) << t.validate();
}

namespace {

/** The generated case G (a 30 m tower with a 3 m peak) and the type that stands it. */
TowerType generated_type (double zeta = 0.02)
{
    TowerType t;
    t.name = "generated";
    t.base_width = 6.0; t.top_width = 1.5; t.solidity = 0.2; t.arm_length = 12.0; t.arm_depth = 1.5; t.peak = 3.0;
    t.damping_ratio = static_cast<Real>(zeta);
    t.frame_panels = 6;
    t.leg_angle = {Real(0.15), Real(0.012)};
    t.brace_angle = {Real(0.09), Real(0.007)};
    return t;
}

struct Generated
{
    std::shared_ptr<const Frame> frame;
    std::shared_ptr<const std::vector<MemberDesign>> designs;
    std::vector<std::size_t> links;     //!< the nodes loads may be tied to: all but the diagonals' crossings
};

Generated generated (double theta = 20.0)
{
    LatticeSpec s;
    s.base_width = 6.0; s.top_width = 1.5; s.arm_height = 30.0; s.arm_length = 12.0; s.arm_depth = 1.5; s.peak = 3.0;
    s.panels = 6; s.leg_b = 0.15; s.leg_t = 0.012; s.brace_b = 0.09; s.brace_t = 0.007;
    FrameInputs in;
    std::vector<MemberDesign> d;
    std::vector<int> load_joints;
    const std::string gerr = lattice_frame(s, in, d, &load_joints);
    EXPECT_TRUE(gerr.empty()) << gerr;
    if (theta != 20.0) { in.temperature.assign(in.members.size(), theta); }
    std::string err;
    Generated g;
    g.frame = Frame::create(in, err);
    EXPECT_TRUE(g.frame) << err;
    g.designs = std::make_shared<const std::vector<MemberDesign>>(d);
    for (const int id : load_joints) { g.links.push_back(g.frame->node_of_joint(id)); }
    return g;
}

/**
 * The frame loads of the tower's drag fd and the line's pull at the cross-arm's centre, as FrameTower links them
 * to the nodes links (tower-local axes).
 */
std::vector<double> linked_loads (const Frame& frame, const Tower& tw, const std::vector<Real>& fd, const std::vector<std::size_t>& links)
{
    const double c = std::cos(0.5235987755982988), s = std::sin(0.5235987755982988);
    const std::array<double,3> along{{c, s, 0.0}}, across{{-s, c, 0.0}};
    auto local = [&] (double x, double y, double z) {
        return std::array<double,3>{{x * along[0] + y * along[1], x * across[0] + y * across[1], z}};
    };
    std::vector<double> loads(frame.num_dofs(), 0.0);
    for (std::size_t i = 0; i < tw.nodes().size(); ++i) {
        const auto& p = tw.nodes()[i].pos;
        RigidLink(frame, local(p[0] - 100.0, p[1] - 200.0, p[2] - 50.0), 4, &links)
            .add_load(local(fd[3 * i], fd[3 * i + 1], fd[3 * i + 2]), loads);
    }
    RigidLink(frame, {{0.0, 0.0, 30.0}}, 4, &links).add_load(local(pull[0], pull[1], pull[2]), loads);
    return loads;
}

} // namespace

TEST(FrameTower, ItsMembersAreCheckedUnderTheFramesForces)
{
    Tower tw = turned(generated_type(0.3));
    const Generated gen = generated();
    FrameTower ft(tw, gen.frame, g, gen.designs, "case G", gen.links);
    ASSERT_TRUE(ft.has_member_checks());
    const std::vector<Real> fd = drag(tw);
    // before the first step: the static forces under the tower's present loads and the weight
    tw.set_loads(fd);
    tw.set_line_loads({pull}, {tw.attachments()[0]});
    const std::vector<double> loads = linked_loads(*gen.frame, tw, fd, gen.links);
    const FrameSolution st = gen.frame->solve(loads, g);
    const auto expected = check_members(*gen.frame, *gen.designs, st.element_force, {});
    auto at_rest = ft.member_checks(tw);
    ASSERT_EQ(at_rest.size(), expected.size());
    double umax = 0.0;
    for (std::size_t m = 0; m < expected.size(); ++m) {
        EXPECT_NEAR(at_rest[m].utilisation, expected[m].utilisation, tol * (expected[m].utilisation + 1e-3)) << "member " << m + 1;
        umax = std::max(umax, expected[m].utilisation);
    }
    ASSERT_GT(umax, 0.01) << "the loads must work the members: the check is not vacuous";
    // settled under the same loads, the dynamic forces about the weight's equilibrium add up to the same
    const double h = 1.0 / (20.0 * static_cast<double>(ft.frequency()));
    for (int n = 0; n < 1500; ++n) { ft.step(static_cast<Real>(h), fd, std::vector<std::array<Real,3>>{pull}); }
    const auto settled = ft.member_checks(tw);
    for (std::size_t m = 0; m < expected.size(); ++m) {
        EXPECT_NEAR(settled[m].utilisation, expected[m].utilisation, 1e-5 * umax + tol * umax) << "member " << m + 1;
    }
    // a frame without design data checks nothing
    const FrameTower plain(tw, gen.frame, g);
    EXPECT_FALSE(plain.has_member_checks());
    EXPECT_TRUE(plain.member_checks(tw).empty());
}

TEST(FrameTower, HeatingKeepsItsPlaceThenSettlesOnTheSofterFrame)
{
    const Tower tw = turned(generated_type(0.3));
    const Generated cold = generated(), hot = generated(600.0);
    FrameTower ft(tw, cold.frame, g, cold.designs, "case G", cold.links);
    const std::vector<Real> fd = drag(tw);
    double h = 1.0 / (20.0 * static_cast<double>(ft.frequency()));
    for (int n = 0; n < 1500; ++n) { ft.step(static_cast<Real>(h), fd, std::vector<std::array<Real,3>>{pull}); }
    const auto before = ft.attachment_displacement(0);
    const double f_cold = static_cast<double>(ft.frequency());
    const std::string err = ft.set_temperature(std::vector<double>(cold.frame->inputs().members.size(), 600.0));
    ASSERT_TRUE(err.empty()) << err;
    // the same place at the switch
    const auto after = ft.attachment_displacement(0);
    for (std::size_t d = 0; d < 3; ++d) { EXPECT_NEAR(static_cast<double>(after[d]), static_cast<double>(before[d]), tol * 1.0e-2) << d; }
    // every member softened by k_E = 0.31: the frequency by its square root
    EXPECT_NEAR(static_cast<double>(ft.frequency()), f_cold * std::sqrt(0.31), 1e-6 * f_cold);
    // the Rayleigh damping is the cold frame's, lighter on some of the hot frame's modes: settle twice as long
    h = 1.0 / (20.0 * static_cast<double>(ft.frequency()));
    for (int n = 0; n < 3000; ++n) { ft.step(static_cast<Real>(h), fd, std::vector<std::array<Real,3>>{pull}); }
    // settled: the hot frame's static response to the loads and its weight, measured from the cold frame's sag
    const std::vector<double> loads = linked_loads(*cold.frame, tw, fd, cold.links);
    const FrameSolution s_hot = hot.frame->solve(loads, g), sag_cold = cold.frame->solve({}, g);
    const RigidLink arm(*hot.frame, {{0.0, 0.0, 30.0}}, 4, &cold.links);
    const auto u_hot = arm.motion(s_hot.displacement), u_sag = arm.motion(sag_cold.displacement);
    const double c = std::cos(0.5235987755982988), s = std::sin(0.5235987755982988);
    const std::array<double,3> ul{{u_hot[0] - u_sag[0], u_hot[1] - u_sag[1], u_hot[2] - u_sag[2]}};
    const std::array<double,3> expected{{ul[0] * c - ul[1] * s, ul[0] * s + ul[1] * c, ul[2]}};
    const auto got = ft.attachment_displacement(0);
    const double scale = std::sqrt(expected[0] * expected[0] + expected[1] * expected[1] + expected[2] * expected[2]);
    for (std::size_t d = 0; d < 3; ++d) { EXPECT_NEAR(static_cast<double>(got[d]), expected[d], 1e-5 * scale + tol * scale) << d; }
    // and the members' checks see the hot steel: the same forces, 0.47 of the tension strength
    const auto checks = ft.member_checks(tw);
    const auto expected_checks = check_members(*hot.frame, *hot.designs, s_hot.element_force, hot.frame->inputs().temperature);
    double umax = 0.0;
    for (const auto& c : expected_checks) { umax = std::max(umax, c.utilisation); }
    for (std::size_t m = 0; m < checks.size(); ++m) {
        EXPECT_NEAR(checks[m].utilisation, expected_checks[m].utilisation, 1e-5 * umax + tol * umax) << m + 1;
    }
    // a temperature for each member is needed, below 1200 C; nothing changes otherwise
    EXPECT_NE(ft.set_temperature({600.0}).find("temperatures for"), std::string::npos);
    EXPECT_NE(ft.set_temperature(std::vector<double>(checks.size(), 1250.0)).find("below 1200"), std::string::npos);
    EXPECT_EQ(ft.temperature().front(), 600.0);
}

TEST(FrameTower, ItsStateCarriesTheTemperatures)
{
    const Tower tw = turned(generated_type());
    const Generated gen = generated();
    FrameTower a(tw, gen.frame, g, gen.designs, "case G", gen.links), b(tw, gen.frame, g, gen.designs, "case G", gen.links);
    const std::vector<Real> fd = drag(tw);
    for (int n = 0; n < 10; ++n) { a.step(Real(0.01), fd, std::vector<std::array<Real,3>>{pull}); }
    std::vector<double> theta(gen.frame->inputs().members.size(), 20.0);
    for (std::size_t m = 0; m < theta.size(); ++m) { theta[m] = 20.0 + 5.0 * static_cast<double>(m % 100); }
    ASSERT_TRUE(a.set_temperature(theta).empty());
    for (int n = 0; n < 10; ++n) { a.step(Real(0.01), fd, std::vector<std::array<Real,3>>{pull}); }
    // a cold tower restored from the hot one's state heats up and continues bit for bit
    ASSERT_TRUE(b.set_state(a.state()));
    EXPECT_EQ(b.temperature(), theta);
    for (int n = 0; n < 10; ++n) {
        a.step(Real(0.01), fd, std::vector<std::array<Real,3>>{pull});
        b.step(Real(0.01), fd, std::vector<std::array<Real,3>>{pull});
    }
    EXPECT_EQ(a.state(), b.state());
    EXPECT_EQ(a.attachment_displacement(0), b.attachment_displacement(0));
    // a state without the temperatures keeps the present ones
    std::vector<double> short_state = a.state();
    short_state.resize(short_state.size() - theta.size());
    ASSERT_TRUE(b.set_state(short_state));
    EXPECT_EQ(b.temperature(), theta);
    // a temperature that is not finite is refused
    std::vector<double> bad = a.state();
    bad.back() = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(b.set_state(bad));
}

TEST(TowerType, RefusesFrameKeysThatDoNotFit)
{
    auto refused = [] (const TowerType& t, const std::string& part) {
        const std::string e = t.validate();
        EXPECT_NE(e.find(part), std::string::npos) << "expected '" << part << "' in: " << e;
    };
    EXPECT_TRUE(generated_type().validate().empty()) << generated_type().validate();
    { TowerType t = generated_type(); t.frame_file = "x.dat"; refused(t, "both give the tower's frame"); }
    { TowerType t = generated_type(); t.weight = 1.0e4; refused(t, "weight is not given with erf.conductors.generated.frame_panels"); }
    { TowerType t = generated_type(); t.leg_angle = {Real(0.1)}; refused(t, "leg_angle needs two values"); }
    { TowerType t = generated_type(); t.brace_angle = {Real(0.01), Real(0.02)}; refused(t, "brace_angle needs two values"); }
    { TowerType t = generated_type(); t.frame_panels = 0; refused(t, "leg_angle needs erf.conductors.generated.frame_panels"); }
    { TowerType t = generated_type(); t.frame_panels = 201; refused(t, "frame_panels must be in [0, 200]"); }
    { TowerType t = generated_type(); t.bracing = "k"; refused(t, "bracing must be crossed or single"); }
    { TowerType t = generated_type(); t.yield_strength = Real(-1.0); refused(t, "yield_strength must be finite"); }
    { TowerType t = generated_type(); t.steel_temperature = Real(1200.0); refused(t, "below 1200"); }
    { TowerType t = generated_type(); t.steel_temperature = std::numeric_limits<Real>::quiet_NaN(); refused(t, "below 1200"); }
    TowerType bare;
    bare.name = "bare";
    bare.base_width = 6.0; bare.top_width = 1.5; bare.solidity = 0.2; bare.arm_length = 12.0;
    ASSERT_TRUE(bare.validate().empty());
    { TowerType t = bare; t.member_file = "m.dat"; refused(t, "member_file needs erf.conductors.bare.frame_file"); }
    { TowerType t = bare; t.bracing = "single"; refused(t, "bracing needs erf.conductors.bare.frame_panels"); }
    { TowerType t = bare; t.yield_strength = Real(3.0e8); refused(t, "yield_strength needs erf.conductors.bare.frame_panels"); }
    { TowerType t = bare; t.steel_temperature = Real(500.0); refused(t, "steel_temperature needs a frame model"); }
    EXPECT_TRUE(generated_type().moves());
    EXPECT_TRUE(generated_type().has_frame());
    EXPECT_FALSE(bare.moves());
    EXPECT_EQ(generated_type().yield(), Real(3.45e8));
}
