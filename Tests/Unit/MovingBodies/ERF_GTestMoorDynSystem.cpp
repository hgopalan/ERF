// Contract of erf_moordyn::MoorDynSystem over the MoorDyn-C v2 C API (the bundled stub or the real
// library, whichever the build links). "Line" here is a MoorDyn line; the input is one span.
//
// - VersionAndErrorNamesAreKnown.
// - MissingInputFileIsReportedNotFatal; AnInputWithoutWaveKinIsRefusedAtCreation (both libraries);
//   AMalformedStubInputIsReportedNotFatal (stub only).
// - FixedSpanHangsBetweenItsPointsWithTheCatenarySag: no coupled degree of freedom, the end nodes on
//   the end points, the catenary sag at mid-span, the end tensions directed into the line.
// - ExternalKinematicsPointsStartWithTheLineNodes: then the points.
// - CrosswindBlowsTheSpanOutTowardsTheStaticAngle: atan(q / w), and the end tension rises.
// - SavedStateContinuesIdenticallyInAFreshSystem.
// - AStateKeptInMemoryRedoesAStepExactly: so that a coupled step can be iterated.
// - InitRefusesWrongSizesAndNonFiniteValues: as error strings.
// - BadStepsKinematicsAndStatesAreRefusedNamingTheInput: the aborts name the input file.

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ERF_GTestThrowOnAbort.H"
#include "ERF_MoorDynSystem.H"

namespace {

using erf_moordyn::MoorDynSystem;

constexpr double pi = 3.14159265358979323846;   // MSVC has no M_PI

// a 300 m span of 795 kcmil (thousand circular mils) ACSR (aluminium conductor, steel reinforced)
// "Drake", 1.5 m slack, both ends fixed 100 m below MoorDyn's surface so that the fluid loads act
// (MoorDyn applies them below z = 0 only)
struct Span {
    double chord = 300.0, slack = 1.5, z = -100.0;
    double diam = 0.0281, mass = 1.628, EA = 3.0e7, Cd = 1.0;
    double rho = 1.2, g = 9.81;
    int nseg = 20;
    std::string edit_from, edit_to;   // a text replacement applied to the written input, for malformed inputs
    double length () const { return chord + slack; }
    double weight () const { return (mass - rho * 0.25 * pi * diam * diam) * g; }    // N/m, in the fluid
    double drag (double U) const { return 0.5 * rho * Cd * diam * U * U; }             // N/m, normal wind U
    double sag () const { return std::sqrt(3.0 * chord * slack / 8.0); }               // parabola with the line's length
    double horizontal_tension () const { return weight() * chord * chord / (8.0 * sag()); }
};

std::string write_input (const std::filesystem::path& dir, const Span& s)
{
    std::filesystem::create_directories(dir);
    const auto fname = dir / "span.txt";
    std::ostringstream out;
    out << "MoorDyn-C input for a single fixed-fixed conductor span (unit test)\n"
        << "----------------------- LINE TYPES ------------------------------------------\n"
        << "TypeName   Diam     Mass/m     EA         BA/-zeta    EI         Cd     Ca     CdAx    CaAx\n"
        << "(name)     (m)      (kg/m)     (N)        (N-s/-)     (N-m^2)    (-)    (-)    (-)     (-)\n"
        << "drake      " << s.diam << "   " << s.mass << "   " << s.EA << "   -0.5   0   " << s.Cd << "   1.0   0.0   0.0\n"
        << "---------------------- POINT PROPERTIES --------------------------------\n"
        << "ID    Type      X       Y       Z       Mass   Volume  CdA    Ca\n"
        << "(#)   (-)       (m)     (m)     (m)     (kg)   (m^3)   (m^2)  (-)\n"
        << "1     Fixed     0.0     0.0     " << s.z << "   0   0   0   0\n"
        << "2     Fixed     " << s.chord << "   0.0     " << s.z << "   0   0   0   0\n"
        << "---------------------- LINES ----------------------------------------\n"
        << "ID   LineType   AttachA  AttachB  UnstrLen  NumSegs  LineOutputs\n"
        << "(#)   (name)     (#)      (#)       (m)       (-)     (-)\n"
        << "1     drake      1        2         " << s.length() << "   " << s.nseg << "   -\n"
        << "---------------------- OPTIONS -----------------------------------------\n"
        << "0             writeLog      Write a log file\n"
        << "0.001         dtM           time step to use in the line integration (s)\n"
        << s.g << "          g             gravity (m/s^2)\n"
        << s.rho << "           WtrDnsty      fluid density (kg/m^3)\n"
        << "1000          WtrDpth       depth of the flat bottom (m)\n"
        << "1             WaveKin       the fluid kinematics are provided through the API (-)\n"
        << "0             ICgenDynamic  stationary initial-condition solver\n"
        << "1             disableOutput\n"
        << "1             disableOutTime\n"
        << "------------------------- need this line --------------------------------------\n";
    std::string text = out.str();
    if (!s.edit_from.empty()) {
        const auto at = text.find(s.edit_from);
        EXPECT_NE(at, std::string::npos) << "the input has no '" << s.edit_from << "'";
        if (at != std::string::npos) { text.replace(at, s.edit_from.size(), s.edit_to); }
    }
    std::ofstream(fname, std::ios::trunc) << text;
    return fname.string();
}

std::unique_ptr<MoorDynSystem> make_span (const std::string& tag, const Span& s, bool compute_ic = true)
{
    const auto dir = std::filesystem::temp_directory_path() / ("erf_gtest_moordyn_" + tag);
    const std::string fname = write_input(dir, s);
    std::string err;
    auto sys = MoorDynSystem::create(fname, "", MOORDYN_ERR_LEVEL, err);
    EXPECT_TRUE(sys != nullptr) << err;
    if (sys) {
        err = sys->init({}, {}, compute_ic);
        EXPECT_TRUE(err.empty()) << err;
    }
    return sys;
}

// the kinematics points (all line nodes, then the points) under a uniform wind
void set_uniform_wind (MoorDynSystem& sys, const std::array<double,3>& U, double t)
{
    const unsigned n = sys.num_kinematics_points();
    std::vector<double> u(3 * n), ud(3 * n, 0.0);
    for (unsigned i = 0; i < n; ++i) { for (int d = 0; d < 3; ++d) { u[3*i+d] = U[static_cast<std::size_t>(d)]; } }
    sys.set_kinematics(u, ud, t);
}

std::array<double,3> mid_node (const MoorDynSystem& sys)
{
    const unsigned nn = sys.line_num_nodes(1);
    return sys.line_node_position(1, (nn - 1) / 2);
}

double norm (const std::array<double,3>& v) { return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); }

} // namespace

TEST(MoorDynSystem, VersionAndErrorNamesAreKnown)
{
    EXPECT_FALSE(erf_moordyn::library_version().empty());
    EXPECT_EQ(erf_moordyn::is_stub(), erf_moordyn::library_version() == "stub");
    EXPECT_EQ(erf_moordyn::error_name(MOORDYN_SUCCESS), "MOORDYN_SUCCESS");
    EXPECT_EQ(erf_moordyn::error_name(MOORDYN_INVALID_INPUT_FILE), "MOORDYN_INVALID_INPUT_FILE");
    EXPECT_NE(erf_moordyn::error_name(-1234).find("1234"), std::string::npos);
}

TEST(MoorDynSystem, MissingInputFileIsReportedNotFatal)
{
    std::string err;
    auto sys = MoorDynSystem::create("/no/such/dir/no_such_span.txt", "", MOORDYN_ERR_LEVEL, err);
    EXPECT_EQ(sys, nullptr);
    EXPECT_NE(err.find("no_such_span.txt"), std::string::npos) << err;
}

TEST(MoorDynSystem, FixedSpanHangsBetweenItsPointsWithTheCatenarySag)
{
    const Span s;
    auto sys = make_span("span", s);
    ASSERT_TRUE(sys);
    EXPECT_EQ(sys->num_coupled_dof(), 0u);
    ASSERT_EQ(sys->num_lines(), 1u);
    ASSERT_EQ(sys->num_points(), 2u);
    EXPECT_EQ(sys->point_type(1), 1);
    EXPECT_EQ(sys->point_type(2), 1);
    const unsigned nn = sys->line_num_nodes(1);
    ASSERT_EQ(nn, static_cast<unsigned>(s.nseg + 1));
    EXPECT_NEAR(sys->line_unstretched_length(1), s.length(), 1.0e-9);

    const auto a = sys->line_node_position(1, 0);
    const auto b = sys->line_node_position(1, nn - 1);
    EXPECT_NEAR(a[0], 0.0, 1.0e-6); EXPECT_NEAR(a[1], 0.0, 1.0e-6); EXPECT_NEAR(a[2], s.z, 1.0e-6);
    EXPECT_NEAR(b[0], s.chord, 1.0e-6); EXPECT_NEAR(b[1], 0.0, 1.0e-6); EXPECT_NEAR(b[2], s.z, 1.0e-6);

    // the mid-span node sags below the chord by the catenary sag of the slack, in the vertical plane
    const auto m = mid_node(*sys);
    EXPECT_NEAR(m[0], 0.5 * s.chord, 0.05 * s.chord);
    EXPECT_NEAR(m[1], 0.0, 1.0e-3);
    const double drop = s.z - m[2];
    EXPECT_NEAR(drop, s.sag(), 0.25 * s.sag()) << "sag " << drop << " m, parabola " << s.sag() << " m";

    // the end tension is the catenary tension: the horizontal component plus the weight of half the span
    const double H = s.horizontal_tension();
    const double T_end = std::sqrt(H * H + std::pow(0.5 * s.weight() * s.chord, 2));
    EXPECT_NEAR(sys->line_end_tension(1), T_end, 0.25 * T_end);
    EXPECT_GE(sys->line_max_tension(1), 0.99 * sys->line_end_tension(1));
    // the end-node tension vectors are the pull on the end points, directed into the line: the same
    // horizontal tension H at both ends, and the weight of half the span downwards at the first node
    // (the line leaves it downwards) and upwards at the last
    const auto ta = sys->line_node_tension(1, 0);
    const auto tb = sys->line_node_tension(1, nn - 1);
    const double V = 0.5 * s.weight() * s.chord;
    EXPECT_NEAR(ta[0], H, 0.15 * H);
    EXPECT_NEAR(tb[0], H, 0.15 * H);
    EXPECT_NEAR(ta[2], -V, 0.25 * V);
    EXPECT_NEAR(tb[2], V, 0.25 * V);
    // MoorDyn computes the net force only for free and coupled points: a fixed point reports none
    for (unsigned p = 1; p <= 2; ++p) {
        const auto fp = sys->point_force(p);
        EXPECT_NEAR(norm(fp), 0.0, 1.0e-9) << "fixed point " << p;
    }
}

TEST(MoorDynSystem, ExternalKinematicsPointsStartWithTheLineNodes)
{
    const Span s;
    auto sys = make_span("kin", s);
    ASSERT_TRUE(sys);
    std::string err;
    const unsigned n = sys->external_kinematics_init(err);
    ASSERT_GT(n, 0u) << err;
    const unsigned nn = sys->line_num_nodes(1);
    EXPECT_GE(n, nn);
    EXPECT_EQ(sys->num_kinematics_points(), n);
    const auto r = sys->kinematics_points();
    ASSERT_EQ(r.size(), 3 * static_cast<std::size_t>(n));
    for (unsigned i = 0; i < nn; ++i) {
        const auto p = sys->line_node_position(1, i);
        for (int d = 0; d < 3; ++d) { EXPECT_NEAR(r[3*i+d], p[static_cast<std::size_t>(d)], 1.0e-9) << "node " << i << " dir " << d; }
    }
    // after the nodes MoorDyn lists the two end points; MoorDyn-C 2.7.1 (and the stub) then add one
    // more entry, the ground body at MoorDyn's origin, although the input has no bodies: a caller
    // that samples a flow at every entry would sample far outside its domain
    ASSERT_GE(n, nn + 2) << "the line nodes and the two points";
    for (unsigned p = 1; p <= 2; ++p) {
        const auto a = sys->point_position(p);
        for (int d = 0; d < 3; ++d) { EXPECT_NEAR(r[3*(nn+p-1)+d], a[static_cast<std::size_t>(d)], 1.0e-9) << "point " << p << " dir " << d; }
    }
    if (erf_moordyn::is_stub() || erf_moordyn::library_version() == "2.7.1") {
        ASSERT_EQ(n, nn + 3) << "the line nodes, the two points and the entry at the origin";
        for (int d = 0; d < 3; ++d) { EXPECT_NEAR(r[3*(nn+2)+d], 0.0, 1.0e-9) << "entry at the origin, dir " << d; }
    }
}

TEST(MoorDynSystem, CrosswindBlowsTheSpanOutTowardsTheStaticAngle)
{
    const Span s;
    auto sys = make_span("wind", s);
    ASSERT_TRUE(sys);
    std::string err;
    ASSERT_GT(sys->external_kinematics_init(err), 0u) << err;

    const double U = 20.0;
    const double phi_static = std::atan2(s.drag(U), s.weight());   // about 23 degrees
    const double dt = 0.05;
    double t = 0.0;
    std::vector<double> f;
    double phi_sum = 0.0, ten_sum = 0.0;
    int nsum = 0;
    // let the swing settle (the real line oscillates about the static angle, the stub relaxes to it),
    // then average over several swing periods
    while (t < 60.0 - 0.5 * dt) {
        set_uniform_wind(*sys, {{0.0, U, 0.0}}, t + 0.5 * dt);
        sys->step({}, {}, f, t, dt);
        if (t > 20.0) {
            const auto m = mid_node(*sys);
            phi_sum += std::atan2(m[1], s.z - m[2]);
            ten_sum += sys->line_end_tension(1);
            ++nsum;
        }
    }
    ASSERT_GT(nsum, 0);
    const double phi_mean = phi_sum / nsum;
    const double ten_mean = ten_sum / nsum;
    EXPECT_NEAR(phi_mean, phi_static, 0.35 * phi_static) << "mean swing " << phi_mean * 180.0 / pi
        << " deg, quasi-static " << phi_static * 180.0 / pi << " deg";
    const auto m = mid_node(*sys);
    EXPECT_GT(m[1], 0.0) << "the span must blow out with the wind (+y)";
    EXPECT_GT(s.z - m[2], 0.5 * s.sag()) << "the span must still hang below the chord";

    // the end tension rises with the effective weight sqrt(w^2 + q^2)
    const double H0 = s.horizontal_tension();
    const double T0 = std::sqrt(H0 * H0 + std::pow(0.5 * s.weight() * s.chord, 2));
    EXPECT_GT(ten_mean, 0.8 * T0);
    EXPECT_LT(ten_mean, 1.6 * T0);
}

TEST(MoorDynSystem, SavedStateContinuesIdenticallyInAFreshSystem)
{
    const Span s;
    auto a = make_span("restart_a", s);
    ASSERT_TRUE(a);
    std::string err;
    ASSERT_GT(a->external_kinematics_init(err), 0u) << err;
    const double U = 15.0, dt = 0.05;
    double t = 0.0;
    std::vector<double> f;
    while (t < 10.0 - 0.5 * dt) {
        set_uniform_wind(*a, {{0.0, U, 0.0}}, t + 0.5 * dt);
        a->step({}, {}, f, t, dt);
    }
    const auto dir = std::filesystem::temp_directory_path() / "erf_gtest_moordyn_restart_a";
    const std::string state = (dir / "state.dat").string();
    a->save(state);

    // the fresh system is created from the same input, initialised without an initial-condition solve
    // and given the saved state; both then take the same five seconds of wind
    auto b = make_span("restart_b", s, false);
    ASSERT_TRUE(b);
    ASSERT_GT(b->external_kinematics_init(err), 0u) << err;
    b->load(state);
    // before any further step the restored line is where the saved one is and feels the same drag:
    // ERF samples the wind at these nodes, and spreads this drag, before the first step after a restart
    {
        const unsigned nn = a->line_num_nodes(1);
        ASSERT_EQ(b->line_num_nodes(1), nn);
        for (unsigned i = 0; i < nn; ++i) {
            const auto pa = a->line_node_position(1, i);
            const auto pb = b->line_node_position(1, i);
            const auto da = a->line_node_drag(1, i);
            const auto db = b->line_node_drag(1, i);
            for (std::size_t d = 0; d < 3; ++d) {
                EXPECT_NEAR(pb[d], pa[d], 1.0e-9 * std::max(1.0, std::fabs(pa[d]))) << "node " << i << " position " << d;
                EXPECT_NEAR(db[d], da[d], 1.0e-9 * std::max(1.0, std::fabs(da[d]))) << "node " << i << " drag " << d;
            }
        }
        const auto m = mid_node(*b);
        EXPECT_GT(m[1], 1.0) << "the restored span must still be blown out along the wind (+y)";
    }
    double tb = t;
    while (t < 15.0 - 0.5 * dt) {
        set_uniform_wind(*a, {{0.0, U, 0.0}}, t + 0.5 * dt);
        set_uniform_wind(*b, {{0.0, U, 0.0}}, tb + 0.5 * dt);
        a->step({}, {}, f, t, dt);
        b->step({}, {}, f, tb, dt);
    }
    EXPECT_NEAR(t, tb, 1.0e-12);
    const unsigned nn = a->line_num_nodes(1);
    ASSERT_EQ(b->line_num_nodes(1), nn);
    double max_diff = 0.0;
    for (unsigned i = 0; i < nn; ++i) {
        const auto pa = a->line_node_position(1, i);
        const auto pb = b->line_node_position(1, i);
        for (int d = 0; d < 3; ++d) { max_diff = std::max(max_diff, std::fabs(pa[static_cast<std::size_t>(d)] - pb[static_cast<std::size_t>(d)])); }
    }
    EXPECT_LT(max_diff, 1.0e-6) << "restored run differs by " << max_diff << " m";
    EXPECT_NEAR(a->line_end_tension(1), b->line_end_tension(1), 1.0e-6 * std::max(1.0, a->line_end_tension(1)));
    EXPECT_GT(norm(mid_node(*a)), 0.0);
}

TEST(MoorDynSystem, AStateKeptInMemoryRedoesAStepExactly)
{
    const Span s;
    auto a = make_span("memory", s);
    ASSERT_TRUE(a);
    std::string err;
    ASSERT_GT(a->external_kinematics_init(err), 0u) << err;
    const double dt = 0.05;
    double t = 0.0;
    std::vector<double> f;
    while (t < 2.0 - 0.5 * dt) {
        set_uniform_wind(*a, {{0.0, 15.0, 0.0}}, t + 0.5 * dt);
        a->step({}, {}, f, t, dt);
    }
    const auto kept = a->serialize();
    const double t0 = t;
    ASSERT_FALSE(kept.empty());
    std::vector<std::array<double,3>> before;
    for (unsigned i = 0; i < a->line_num_nodes(1); ++i) { before.push_back(a->line_node_position(1, i)); }
    // a step, then back to the kept state and the same step again
    // (positions and velocities: a node's velocity carries where it was before the step)
    std::vector<std::array<double,3>> first, first_v;
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) { a->deserialize(kept); t = t0; }
        a->step({}, {}, f, t, dt);
        EXPECT_NEAR(t, t0 + dt, 1.0e-12) << "the clock comes back with the state";
        const unsigned nn = a->line_num_nodes(1);
        for (unsigned i = 0; i < nn; ++i) {
            const auto p = a->line_node_position(1, i);
            const auto v = a->line_node_velocity(1, i);
            if (pass == 0) { first.push_back(p); first_v.push_back(v); continue; }
            for (std::size_t d = 0; d < 3; ++d) {
                EXPECT_EQ(p[d], first[i][d]) << "node " << i << " dir " << d;
                EXPECT_EQ(v[d], first_v[i][d]) << "node " << i << " velocity " << d;
            }
        }
    }
    // restoring brings back the state before the step, not the last result: the nodes stand where
    // they stood when the state was kept, which the step had moved them from
    a->deserialize(kept);
    const unsigned mid = a->line_num_nodes(1) / 2;
    for (unsigned i = 0; i < a->line_num_nodes(1); ++i) {
        const auto p = a->line_node_position(1, i);
        for (std::size_t d = 0; d < 3; ++d) { EXPECT_EQ(p[d], before[i][d]) << "node " << i << " dir " << d; }
    }
    EXPECT_NE(before[mid][1], first[mid][1]) << "the step must move the mid-span node for this check to mean anything";
}

TEST(MoorDynSystem, AnInputWithoutWaveKinIsRefusedAtCreation)
{
    // MoorDyn-C reports kinematics points whatever WaveKin is and then ignores the wind: the
    // wrapper reads the option from the file, so this holds for the real library and the stub
    for (const std::string& row : {std::string("0             WaveKin"), std::string("0             writeLog2")}) {
        Span s;
        s.edit_from = "1             WaveKin";
        s.edit_to = row;
        const auto dir = std::filesystem::temp_directory_path() / "erf_gtest_moordyn_wavekin";
        const std::string fname = write_input(dir, s);
        std::string err;
        auto sys = MoorDynSystem::create(fname, "", MOORDYN_ERR_LEVEL, err);
        EXPECT_EQ(sys, nullptr) << row;
        EXPECT_NE(err.find("WaveKin"), std::string::npos) << err;
        EXPECT_NE(err.find(fname), std::string::npos) << err;
    }
    EXPECT_EQ(erf_moordyn::check_wave_kinematics_option("/no/such/dir/span.txt").find("cannot read"), 0u);
}

TEST(MoorDynSystem, AMalformedStubInputIsReportedNotFatal)
{
    if (!erf_moordyn::is_stub()) { GTEST_SKIP() << "the stub's parser; MoorDyn-C reads numbers its own way"; }
    // a malformed number, and a negative segment count, make creation fail instead of ending the run
    for (const auto& edit : {std::array<std::string,2>{{"3e+07", "x3e7"}}, std::array<std::string,2>{{"   20   -", "   -20   -"}}}) {
        Span s;
        s.edit_from = edit[0];
        s.edit_to = edit[1];
        const std::string fname = write_input(std::filesystem::temp_directory_path() / "erf_gtest_moordyn_malformed", s);
        std::string err;
        auto sys = MoorDynSystem::create(fname, "", MOORDYN_ERR_LEVEL, err);
        EXPECT_EQ(sys, nullptr) << edit[1];
        EXPECT_NE(err.find("malformed"), std::string::npos) << err;
    }
}

TEST(MoorDynSystem, InitRefusesWrongSizesAndNonFiniteValues)
{
    // the far end coupled: three coupled degrees of freedom
    Span s;
    s.edit_from = "2     Fixed     ";
    s.edit_to = "2     Coupled   ";
    const std::string fname = write_input(std::filesystem::temp_directory_path() / "erf_gtest_moordyn_init", s);
    std::string err;
    auto sys = MoorDynSystem::create(fname, "", MOORDYN_ERR_LEVEL, err);
    ASSERT_TRUE(sys) << err;
    ASSERT_EQ(sys->num_coupled_dof(), 3u);
    err = sys->init({1.0, 2.0}, {0.0, 0.0});
    EXPECT_NE(err.find("coupled degrees of freedom"), std::string::npos) << err;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    err = sys->init({s.chord, 0.0, nan}, {0.0, 0.0, 0.0});
    EXPECT_NE(err.find("not finite"), std::string::npos) << err;
    err = sys->init({s.chord, 0.0, s.z}, {0.0, 0.0, 0.0});
    EXPECT_TRUE(err.empty()) << err;
}

TEST(MoorDynSystem, BadStepsKinematicsAndStatesAreRefusedNamingTheInput)
{
    const Span s;
    auto sys = make_span("refused", s);
    ASSERT_TRUE(sys);
    std::string err;
    ASSERT_GT(sys->external_kinematics_init(err), 0u) << err;
    std::vector<double> f;
    double t = 0.0;
    // a step that is not positive: MoorDyn would return the forces without stepping
    std::string msg = erf_gtest::abort_message([&] { sys->step({}, {}, f, t, 0.0); });
    EXPECT_NE(msg.find("step must be finite and positive"), std::string::npos) << msg;
    EXPECT_NE(msg.find(sys->input_file()), std::string::npos) << msg;
    // a non-finite wind
    const unsigned n = sys->num_kinematics_points();
    std::vector<double> u(3 * n, 0.0), ud(3 * n, 0.0);
    u[4] = std::numeric_limits<double>::quiet_NaN();
    msg = erf_gtest::abort_message([&] { sys->set_kinematics(u, ud, 0.0); });
    EXPECT_NE(msg.find("kinematics point 1"), std::string::npos) << msg;
    // an internal step that is not positive
    msg = erf_gtest::abort_message([&] { sys->set_dt(0.0); });
    EXPECT_NE(msg.find("internal step must be finite and positive"), std::string::npos) << msg;
    // a state of another size than serialize() returns
    auto kept = sys->serialize();
    ASSERT_GT(kept.size(), 1u);
    kept.pop_back();
    msg = erf_gtest::abort_message([&] { sys->deserialize(kept); });
    EXPECT_NE(msg.find("expected from serialize()"), std::string::npos) << msg;
    msg = erf_gtest::abort_message([&] { sys->deserialize({}); });
    EXPECT_NE(msg.find("0 words given"), std::string::npos) << msg;
    // a saved state that is not there (the stub's error code; MoorDyn-C's handling of a missing file is its own)
    if (erf_moordyn::is_stub()) {
        msg = erf_gtest::abort_message([&] { sys->load("/no/such/dir/state.dat"); });
        EXPECT_NE(msg.find("MoorDyn_Load failed"), std::string::npos) << msg;
    }
    // the system still steps after every refusal
    set_uniform_wind(*sys, {{0.0, 10.0, 0.0}}, 0.025);
    EXPECT_TRUE(erf_gtest::abort_message([&] { sys->step({}, {}, f, t, 0.05); }).empty());
}
