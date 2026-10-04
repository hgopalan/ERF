// Unit tests of erf_conductors::ConductorLine, one conductor line as one MoorDyn system.
//
// HangsAtTheCatenarySagInStillAirInERFsFrame: a single-span line hangs at the catenary sag with the
//     catenary end tensions, its nodes in ERF's frame, its kinematics points starting with the nodes.
// APrescribedCrosswindBlowsItOutInLockstepWithERFsClock: a crosswind handed over step by step blows
//     the span out towards the quasi-static angle atan(q / w) and raises the tension.
// FourSubstepsAdvanceMoorDynsClockByOneStep: four MoorDyn calls per step end on ERF's clock.
// ANonFiniteWindIsRefusedNamingTheLineAndThePoint: set_wind aborts on NaN, naming the line and the point.
// DiagnosticsRowsCarryTheHeaderOnce: the diagnostics file has one row per call and one header.
// ALineCreatedFromASavedStateContinuesIt: a line created from a saved state is where the saved one
//     is and continues as it does.
// ASectionHangsFromItsInsulatorStrings: strings hang plumb in still air and let the conductor swing
//     further across the wind than clamps at the towers do.
// TheDeadEndsOfALevelSpanArePulledTogetherAndDown: the dead-end pulls balance along the chord and
//     carry the line's weight.
// ATowerOfALevelSectionCarriesOneSpansWeight: a tower carries one span's weight (and its string's),
//     the spans either side balancing along the line.
// MovingTowersCrossArmsAreCoupledPointsThatTakeTheLinesPull: a coupled step moves the cross-arms at
//     the velocity it is given, and MoorDyn's force on them is the pull the tower carries.
// HeldStillACoupledPointIsAFixedOne: a coupled point held still behaves as a fixed one.
// AStringInADipIsFlaggedInUplift: a string the spans either side pull up is flagged in uplift.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ERF_ConductorInputs.H"
#include "ERF_ConductorLine.H"
#include "ERF_GTestThrowOnAbort.H"
#include "ERF_MoorDynSystem.H"
#include "ERF_TowerInputs.H"

using erf_conductors::ConductorInputs;
using erf_conductors::ConductorLine;
using erf_conductors::LineInputs;

namespace {

constexpr double pi = 3.14159265358979323846;   // MSVC has no M_PI
// positions of a few hundred metres carry a Real's spacing there: 3e-5 m at 500 m in single precision
constexpr double ptol = (std::is_same<amrex::Real, float>::value) ? 1.0e-4 : 1.0e-6;
// the same computation repeated (a restored state, a coupled point held still): roundoff only
constexpr double ttol = (std::is_same<amrex::Real, float>::value) ? 1.0e-4 : 1.0e-9;
constexpr double rtol = (std::is_same<amrex::Real, float>::value) ? 1.0e-5 : 1.0e-9;

LineInputs drake_span (const std::string& name)
{
    LineInputs s;
    s.name = name;
    s.end_a = {{100.0, 500.0, 30.0}};
    s.end_b = {{400.0, 500.0, 30.0}};
    s.lengths = {301.5}; s.diameter = 0.0281; s.mass_per_length = 1.628; s.axial_stiffness = 3.0e7;
    s.output_root = (std::filesystem::temp_directory_path() / "erf_gtest_conductor_span" / name).string();
    return s;
}

ConductorInputs settings ()
{
    ConductorInputs in;
    in.air_density = 1.2;
    in.diagnostics_dir = (std::filesystem::temp_directory_path() / "erf_gtest_conductor_span").string();
    return in;
}

std::unique_ptr<ConductorLine> make (const std::string& name, const ConductorInputs& in)
{
    const std::string file = in.diagnostics_dir + "/" + name + ".moordyn.txt";
    return std::make_unique<ConductorLine>(drake_span(name), in, 9.81, file);
}

double weight (const LineInputs& s, double rho) { return (s.mass_per_length - rho * 0.25 * pi * s.diameter * s.diameter) * 9.81; }
double drag (const LineInputs& s, double rho, double U) { return 0.5 * rho * s.drag_coefficient * s.diameter * U * U; }

std::vector<amrex::Real> uniform (unsigned n, double u, double v, double w)
{
    std::vector<amrex::Real> uvw(3 * n);
    for (unsigned i = 0; i < n; ++i) { uvw[3*i] = u; uvw[3*i+1] = v; uvw[3*i+2] = w; }
    return uvw;
}

} // namespace

TEST(ConductorLine, HangsAtTheCatenarySagInStillAirInERFsFrame)
{
    const ConductorInputs in = settings();
    auto span = make("still", in);
    const LineInputs& s = span->inputs();
    EXPECT_EQ(span->num_nodes(), 21u);
    EXPECT_GE(span->num_kinematics_points(), 21u);
    const auto a = span->node_position(0);
    const auto b = span->node_position(20);
    EXPECT_NEAR(a[0], 100.0, ptol); EXPECT_NEAR(a[1], 500.0, ptol); EXPECT_NEAR(a[2], 30.0, ptol);
    EXPECT_NEAR(b[0], 400.0, ptol); EXPECT_NEAR(b[1], 500.0, ptol); EXPECT_NEAR(b[2], 30.0, ptol);
    EXPECT_NEAR(span->mid_sag(), s.catenary_sag(), 0.25 * s.catenary_sag());
    EXPECT_NEAR(span->mid_offset(), 0.0, 1.0e-3);
    EXPECT_NEAR(span->swing_angle(), 0.0, 1.0e-4);
    const auto m = span->node_position(10);
    EXPECT_NEAR(m[2], 30.0 - span->mid_sag(), ptol);
    // the catenary tension at both ends
    const double w = weight(s, in.air_density);
    const double H = w * 300.0 * 300.0 / (8.0 * s.catenary_sag());
    const double T = std::sqrt(H * H + std::pow(0.5 * w * 300.0, 2));
    EXPECT_NEAR(span->tension_a(), T, 0.25 * T);
    EXPECT_NEAR(span->tension_b(), T, 0.25 * T);
    EXPECT_GE(span->max_tension(), 0.99 * span->tension_b());
    // the kinematics points start with the line nodes, in ERF's frame
    const auto k = span->kinematics_points();
    const auto n = span->node_positions();
    ASSERT_GE(k.size(), n.size());
    for (std::size_t i = 0; i < n.size(); ++i) { EXPECT_NEAR(k[i], n[i], ttol) << "component " << i; }
    // the MoorDyn input is where the line says
    EXPECT_TRUE(std::filesystem::exists(span->input_file()));
}

TEST(ConductorLine, APrescribedCrosswindBlowsItOutInLockstepWithERFsClock)
{
    const ConductorInputs in = settings();
    auto span = make("wind", in);
    const LineInputs& s = span->inputs();
    const double U = 20.0;
    const double phi_static = std::atan2(drag(s, in.air_density, U), weight(s, in.air_density));
    const double dt = 0.05;
    double t = 0.0;
    double phi_sum = 0.0, ten_sum = 0.0;
    int n = 0;
    while (t < 60.0 - 0.5 * dt) {
        span->set_wind(uniform(span->num_kinematics_points(), 0.0, U, 0.0), t + 0.5 * dt);
        span->step(t, dt);
        t += dt;
        if (t > 20.0) { phi_sum += span->swing_angle(); ten_sum += span->tension_b(); ++n; }
    }
    ASSERT_GT(n, 0);
    EXPECT_NEAR(phi_sum / n, phi_static, 0.35 * phi_static) << "mean swing " << phi_sum / n * 180.0 / pi
        << " deg, quasi-static " << phi_static * 180.0 / pi << " deg";
    EXPECT_GT(span->mid_offset(), 0.0) << "the span must blow out with the wind (+y)";
    EXPECT_GT(span->mid_sag(), 0.5 * s.catenary_sag());
    const double w = weight(s, in.air_density);
    const double H = w * 300.0 * 300.0 / (8.0 * s.catenary_sag());
    const double T0 = std::sqrt(H * H + std::pow(0.5 * w * 300.0, 2));
    EXPECT_GT(ten_sum / n, 0.8 * T0);
    EXPECT_LT(ten_sum / n, 1.6 * T0);
}

TEST(ConductorLine, FourSubstepsAdvanceMoorDynsClockByOneStep)
{
    ConductorInputs in = settings();
    in.substeps = 4;
    auto span = make("substeps", in);
    span->set_wind(uniform(span->num_kinematics_points(), 0.0, 10.0, 0.0), 0.025);
    span->step(0.0, 0.05);   // four MoorDyn calls of 0.0125 s
    EXPECT_NEAR(span->state().clock, 0.05, 1.0e-12) << "MoorDyn's clock after four substeps";
    EXPECT_GT(span->mid_offset(), -1.0e-9);
}

TEST(ConductorLine, ANonFiniteWindIsRefusedNamingTheLineAndThePoint)
{
    const ConductorInputs in = settings();
    auto line = make("nan_wind", in);
    std::vector<amrex::Real> uvw = uniform(line->num_kinematics_points(), 0.0, 10.0, 0.0);
    uvw[3 * 3 + 1] = std::numeric_limits<amrex::Real>::quiet_NaN();
    const std::string msg = erf_gtest::abort_message([&] { line->set_wind(uvw, 0.025); });
    EXPECT_NE(msg.find("erf.conductors.nan_wind"), std::string::npos) << msg;
    EXPECT_NE(msg.find("kinematics point 3 is not finite"), std::string::npos) << msg;
    // a finite wind is taken
    EXPECT_TRUE(erf_gtest::abort_message([&] { line->set_wind(uniform(line->num_kinematics_points(), 0.0, 10.0, 0.0), 0.025); }).empty());
}

TEST(ConductorLine, DiagnosticsRowsCarryTheHeaderOnce)
{
    const ConductorInputs in = settings();
    auto span = make("diag", in);
    const std::string fname = span->inputs().output_root + ".dat";
    std::filesystem::remove(fname);
    span->write_diagnostics(0.0, true);
    span->set_wind(uniform(span->num_kinematics_points(), 0.0, 10.0, 0.0), 0.025);
    span->step(0.0, 0.05);
    span->write_diagnostics(0.05, false);
    std::ifstream f(fname);
    ASSERT_TRUE(f.good());
    std::string line;
    int rows = 0, headers = 0;
    while (std::getline(f, line)) {
        if (line.rfind("time ", 0) == 0) { ++headers; } else if (!line.empty()) { ++rows; }
    }
    EXPECT_EQ(headers, 1);
    EXPECT_EQ(rows, 2);
}

TEST(ConductorLine, ALineCreatedFromASavedStateContinuesIt)
{
    const ConductorInputs in = settings();
    auto a = make("saved_a", in);
    const unsigned nk = a->num_kinematics_points();
    const double dt = 0.1;
    double t = 0.0;
    for (int n = 0; n < 40; ++n) {
        a->set_wind(uniform(nk, 0.0, 15.0, 0.0), t + 0.5 * dt);
        a->step(t, dt);
        t += dt;
    }
    ASSERT_GT(a->mid_offset(), 1.0) << "the saved line must be blown out, so a restart from rest would show";
    const std::string state = in.diagnostics_dir + "/saved_a.state";
    a->save(state);

    const std::string file = in.diagnostics_dir + "/saved_b.moordyn.txt";
    auto b = std::make_unique<ConductorLine>(drake_span("saved_b"), in, 9.81, file, state);
    // before any step: where the saved line is, with its drag and tensions
    ASSERT_EQ(b->num_nodes(), a->num_nodes());
    for (unsigned n = 0; n < a->num_nodes(); ++n) {
        const auto pa = a->node_position(n);
        const auto pb = b->node_position(n);
        const auto da = a->node_drag(n);
        const auto db = b->node_drag(n);
        for (int d = 0; d < 3; ++d) {
            EXPECT_NEAR(pb[d], pa[d], rtol * std::max(amrex::Real(1.0), std::abs(pa[d]))) << "node " << n << " position " << d;
            EXPECT_NEAR(db[d], da[d], rtol * std::max(amrex::Real(1.0), std::abs(da[d]))) << "node " << n << " drag " << d;
        }
    }
    EXPECT_NEAR(b->tension_a(), a->tension_a(), rtol * a->tension_a());
    // the same wind from here on: the two stay together, on the same clock
    for (int n = 0; n < 20; ++n) {
        const double gust = 15.0 + 3.0 * std::sin(0.7 * t);
        a->set_wind(uniform(nk, 0.0, gust, 0.0), t + 0.5 * dt);
        b->set_wind(uniform(nk, 0.0, gust, 0.0), t + 0.5 * dt);
        a->step(t, dt);
        b->step(t, dt);
        t += dt;
    }
    for (unsigned n = 0; n < a->num_nodes(); ++n) {
        const auto pa = a->node_position(n);
        const auto pb = b->node_position(n);
        for (int d = 0; d < 3; ++d) {
            EXPECT_NEAR(pb[d], pa[d], rtol * std::max(amrex::Real(1.0), std::abs(pa[d]))) << "node " << n << " position " << d;
        }
    }
    EXPECT_NEAR(b->tension_a(), a->tension_a(), rtol * a->tension_a());
}

namespace {
LineInputs section (const std::string& name, double insulator)
{
    LineInputs s = drake_span(name);
    s.end_a = {{100.0, 500.0, 30.0}};
    s.end_b = {{1000.0, 500.0, 30.0}};
    s.towers = {{{400.0, 500.0, 30.0}}, {{700.0, 500.0, 30.0}}};
    s.lengths = {301.5, 301.5, 301.5};
    s.insulator_length = insulator;
    s.insulator_mass = 60.0;
    return s;
}
} // namespace

TEST(ConductorLine, ASectionHangsFromItsInsulatorStrings)
{
    const ConductorInputs in = settings();
    const std::string file = in.diagnostics_dir + "/strings.moordyn.txt";
    ConductorLine line(section("strings", 2.5), in, 9.81, file);
    EXPECT_EQ(line.num_spans(), 3);
    EXPECT_EQ(line.num_insulators(), 2);
    EXPECT_EQ(line.num_nodes(), 3u * 21u + 2u * 3u);
    EXPECT_EQ(line.span_first_node(2), 42u);
    EXPECT_GE(line.num_kinematics_points(), line.num_nodes());
    EXPECT_EQ(line.conductor_path().size(), 3u * 3u * 21u);
    // still air: the strings hang plumb and the conductor ends hang 2.5 m under the towers
    for (int j = 0; j < 2; ++j) {
        EXPECT_LT(line.insulator_swing(j), 0.01) << "string " << j;
        EXPECT_GT(line.insulator_tension(j), 1000.0) << "string " << j << " carries the conductor";
    }
    const auto a2 = line.node_position(line.span_first_node(1));
    EXPECT_NEAR(a2[0], 400.0, 0.05);
    EXPECT_NEAR(a2[2], 27.5, 0.05);
    // the sag is measured from the towers, so it includes the string
    EXPECT_NEAR(line.mid_sag(1), line.inputs().catenary_sag(1) + 2.5, 0.25 * line.inputs().catenary_sag(1));
    EXPECT_NEAR(line.mid_offset(1), 0.0, 1.0e-3);

    // a steady crosswind along +y: the strings swing across the line, with the wind, and let the
    // middle span blow out further than the same section clamped at its towers
    ConductorLine clamped(section("clamped", 0.0), in, 9.81, in.diagnostics_dir + "/clamped.moordyn.txt");
    EXPECT_EQ(clamped.num_insulators(), 0);
    const double dt = 0.1;
    double t = 0.0;
    for (int n = 0; n < 100; ++n) {
        line.set_wind(uniform(line.num_kinematics_points(), 0.0, 20.0, 0.0), t + 0.5 * dt);
        clamped.set_wind(uniform(clamped.num_kinematics_points(), 0.0, 20.0, 0.0), t + 0.5 * dt);
        line.step(t, dt);
        clamped.step(t, dt);
        t += dt;
    }
    for (int j = 0; j < 2; ++j) {
        EXPECT_GT(line.insulator_swing_across(j), 0.05) << "string " << j << " swings with the wind";
        EXPECT_NEAR(line.insulator_swing_across(j), line.insulator_swing(j), 0.02) << "string " << j << " swings across the line";
    }
    EXPECT_GT(line.mid_offset(1), clamped.mid_offset(1) + 0.3) << "the strings add their swing to the span's";
    // each span's quantities are its own: the middle node of span k, against the chord between its towers
    for (int k = 0; k < 3; ++k) {
        const auto m = line.node_position(line.span_first_node(k) + 10);
        EXPECT_NEAR(line.mid_offset(k), m[1] - 500.0, ptol) << "span " << k;
        EXPECT_NEAR(line.mid_sag(k), 30.0 - m[2], ptol) << "span " << k;
    }
    EXPECT_GT(std::abs(line.mid_offset(1) - line.mid_offset(0)), 1.0e-3) << "the middle span hangs from strings at both ends";
    EXPECT_GT(clamped.mid_offset(1), 1.0);
}

TEST(ConductorLine, TheDeadEndsOfALevelSpanArePulledTogetherAndDown)
{
    const ConductorInputs in = settings();
    auto span = make("dead_ends", in);
    const LineInputs& s = span->inputs();
    const auto a = span->end_force(0);
    const auto b = span->end_force(1);
    // end_a is pulled towards end_b (+x) and down, end_b the other way along x, the same down
    EXPECT_GT(a[0], 0.0);
    EXPECT_LT(b[0], 0.0);
    EXPECT_NEAR(a[0], -b[0], 1.0e-3 * a[0]);
    EXPECT_LT(a[2], 0.0);
    EXPECT_NEAR(a[2], b[2], 1.0e-3 * std::abs(a[2]));
    EXPECT_NEAR(a[1], 0.0, 1.0e-6 * a[0]);
    // between them the line's weight (the stub's parabola within its sag error)
    const double W = weight(s, in.air_density) * s.lengths[0];
    EXPECT_NEAR(-(a[2] + b[2]), W, 0.05 * W);
    // the pull is the end tension
    EXPECT_NEAR(std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]), span->tension_a(), 0.02 * span->tension_a());
}

TEST(ConductorLine, ATowerOfALevelSectionCarriesOneSpansWeight)
{
    const ConductorInputs in = settings();
    // the stub's parabola is a few per cent from the catenary; the real MoorDyn's lumped masses much closer
    const double tol = erf_moordyn::is_stub() ? 0.1 : 0.01;
    const double w = weight(drake_span("w"), in.air_density);
    // clamped: every tower of a level section carries one span's weight
    {
        ConductorLine line(section("tower_clamped", 0.0), in, 9.81, in.diagnostics_dir + "/tower_clamped.moordyn.txt");
        for (int j = 0; j < 2; ++j) {
            const auto F = line.tower_force(j);
            EXPECT_NEAR(-F[2] / (w * 301.5), 1.0, tol) << "tower " << j + 1 << ": half of each span either side";
            EXPECT_NEAR(F[0], 0.0, tol * w * 301.5) << "tower " << j + 1 << ": the spans balance along the line";
            EXPECT_NEAR(F[1], 0.0, 1.0e-6 * w * 301.5) << "tower " << j + 1;
        }
    }
    // on strings: the middle tower of four spans, whose spans either side both hang from strings
    // (a span from a dead end drops to the string's bottom, and the lower end takes less weight)
    LineInputs s = section("tower_strings", 2.5);
    s.end_b = {{1300.0, 500.0, 30.0}};
    s.towers.push_back({{1000.0, 500.0, 30.0}});
    s.lengths.push_back(301.5);
    ConductorLine line(s, in, 9.81, in.diagnostics_dir + "/tower_strings.moordyn.txt");
    const double Wi = (s.insulator_mass - in.air_density * 0.25 * pi * s.insulator_diameter * s.insulator_diameter * 2.5) * 9.81;
    const auto F = line.tower_force(1);
    EXPECT_NEAR(-F[2] / (w * 301.5 + Wi), 1.0, tol) << "a span's weight and the string's";
    EXPECT_NEAR(F[0], 0.0, tol * w * 301.5);
    // and the force on the tower is the string's pull: down the string
    EXPECT_LT(F[2], 0.0);
}

namespace {
// the section's towers as a type that bends at 2 Hz: their cross-arms are coupled points
ConductorInputs moving_settings ()
{
    ConductorInputs in = settings();
    erf_towers::TowerType t;
    t.name = "lat";
    t.base_width = 6.0; t.top_width = 1.5; t.solidity = 0.2; t.arm_length = 12.0;
    t.weight = 9.0e4; t.frequency = 2.0;
    in.tower_types = {t};
    return in;
}
std::string slurp (const std::string& fname)
{
    std::ifstream f(fname);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
} // namespace

TEST(ConductorLine, MovingTowersCrossArmsAreCoupledPointsThatTakeTheLinesPull)
{
    const ConductorInputs in = moving_settings();
    for (const double ins : {2.5, 0.0}) {
        const std::string name = ins > 0.0 ? "coupled_strings" : "coupled_clamped";
        LineInputs s = section(name, ins);
        s.tower_type = "lat";
        const std::string file = in.diagnostics_dir + "/" + name + ".moordyn.txt";
        ConductorLine line(s, in, 9.81, file);
        ASSERT_TRUE(line.towers_move()) << name;
        const std::string text = slurp(file);
        EXPECT_NE(text.find("2     Coupled   400   500   -9970"), std::string::npos) << text;
        EXPECT_NE(text.find("3     Coupled   700   500   -9970"), std::string::npos) << text;
        EXPECT_NE(text.find("1     Fixed     100   500   -9970"), std::string::npos) << "the dead ends stay fixed";
        // tower 1's cross-arm moves at 0.2 m/s across the line for 0.1 s; tower 2's stays
        line.set_wind(uniform(line.num_kinematics_points(), 0.0, 0.0, 0.0), 0.05);
        const std::vector<amrex::Real> start(6, 0.0), velocity{0.0, 0.2, 0.0, 0.0, 0.0, 0.0};
        line.step_coupled(0.0, 0.1, start, velocity);
        // where the line meets each cross-arm: the string's top, or the next span's first node
        auto at_tower = [&] (int j) {
            const unsigned string_top = line.span_first_node(3) + static_cast<unsigned>((LineInputs::insulator_segments + 1) * j);
            return line.node_position(ins > 0.0 ? string_top : line.span_first_node(j + 1));
        };
        // positions 500 m from the origin carry a Real's spacing there: 3e-5 m in single precision
        const auto p1 = at_tower(0), p2 = at_tower(1);
        EXPECT_NEAR(p1[0], 400.0, ttol) << name;
        EXPECT_NEAR(p1[1], 500.02, ttol) << name << ": moved 0.2 m/s for 0.1 s";
        EXPECT_NEAR(p1[2], 30.0, ttol) << name;
        EXPECT_NEAR(p2[1], 500.0, ttol) << name;
        // the force MoorDyn hands back on each coupled point is the pull the tower is reported to carry
        for (int j = 0; j < 2; ++j) {
            const auto f = line.coupled_force(j), F = line.tower_force(j);
            const double mag = std::sqrt(F[0] * F[0] + F[1] * F[1] + F[2] * F[2]);
            EXPECT_GT(mag, 1000.0) << name << " tower " << j + 1;
            for (int d = 0; d < 3; ++d) { EXPECT_NEAR(f[d], F[d], 1.0e-6 * mag) << name << " tower " << j + 1 << " dir " << d; }
        }
        // a step that holds the towers keeps the cross-arm where the coupled step left it
        line.step(0.1, 0.1);
        EXPECT_NEAR(at_tower(0)[1], 500.02, ttol) << name;
    }
}

TEST(ConductorLine, HeldStillACoupledPointIsAFixedOne)
{
    const ConductorInputs in = moving_settings();
    LineInputs s = section("held_coupled", 2.5);
    s.tower_type = "lat";
    ConductorLine coupled(s, in, 9.81, in.diagnostics_dir + "/held_coupled.moordyn.txt");
    ConductorLine fixed(section("held_fixed", 2.5), in, 9.81, in.diagnostics_dir + "/held_fixed.moordyn.txt");
    ASSERT_TRUE(coupled.towers_move());
    ASSERT_FALSE(fixed.towers_move());
    const double dt = 0.1;
    for (int n = 0; n < 30; ++n) {
        coupled.set_wind(uniform(coupled.num_kinematics_points(), 0.0, 20.0, 0.0), n * dt + 0.5 * dt);
        fixed.set_wind(uniform(fixed.num_kinematics_points(), 0.0, 20.0, 0.0), n * dt + 0.5 * dt);
        coupled.step(n * dt, dt);
        fixed.step(n * dt, dt);
    }
    for (unsigned i = 0; i < fixed.num_nodes(); ++i) {
        const auto a = coupled.node_position(i), b = fixed.node_position(i);
        for (int d = 0; d < 3; ++d) { EXPECT_NEAR(a[d], b[d], ttol) << "node " << i << " dir " << d; }
    }
    for (int j = 0; j < 2; ++j) {
        const auto a = coupled.tower_force(j), b = fixed.tower_force(j);
        for (int d = 0; d < 3; ++d) { EXPECT_NEAR(a[d], b[d], 1.0e-6 * std::abs(b[2])) << "tower " << j + 1 << " dir " << d; }
    }
}

TEST(ConductorLine, AStringInADipIsFlaggedInUplift)
{
    if (erf_moordyn::is_stub()) {
        GTEST_SKIP() << "needs the real MoorDyn-C: the stub hangs every string under its spans' weight";
    }
    const ConductorInputs in = settings();
    const double w = weight(drake_span("w"), in.air_density);
    // level: each string carries a span's weight
    {
        ConductorLine line(section("level_strings", 2.5), in, 9.81, in.diagnostics_dir + "/level_strings.moordyn.txt");
        for (int j = 0; j < 2; ++j) {
            EXPECT_NEAR(line.string_load(j) / (w * 301.5), 1.0, 0.05) << "string " << j + 1;
            EXPECT_FALSE(line.string_in_uplift(j, static_cast<amrex::Real>(w))) << "string " << j + 1;
        }
    }
    // the dead ends 80 m above the towers, strung to 20 kN: each span rises 80 m over 300 m away from
    // its tower, pulling up 20 kN x 80 / 300 = 5.3 kN a side against 2.4 kN of weight
    LineInputs s = section("dip_strings", 2.5);
    s.end_a[2] = 110.0;
    s.end_b[2] = 110.0;
    s.lengths.clear();
    s.stringing_tension = 20000.0;
    s.lengths_from_stringing_tension(static_cast<amrex::Real>(w));
    ConductorLine line(s, in, 9.81, in.diagnostics_dir + "/dip_strings.moordyn.txt");
    for (int j = 0; j < 2; ++j) {
        EXPECT_TRUE(line.string_in_uplift(j, static_cast<amrex::Real>(w)))
            << "string " << j + 1 << " carries " << line.string_load(j) << " N";
    }
}
