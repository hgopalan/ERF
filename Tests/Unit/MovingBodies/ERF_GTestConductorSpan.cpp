// Contract of erf_conductors::ConductorSpan: a span hangs at the catenary sag in still air
// with its nodes reported in ERF's frame and the catenary end tensions; its kinematics points
// start with the line nodes, in ERF's frame; a prescribed crosswind, handed over step by step
// in lockstep with ERF's clock, blows it out towards the quasi-static angle atan(q / w) and
// raises the tension; the diagnostics file carries one row per call with the header once; a
// span created from a saved state is where the saved one is and continues as it does; and a
// section hangs from its insulator strings, which hang plumb in still air and let the conductor
// swing further across the wind than clamps at the towers do.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ERF_ConductorInputs.H"
#include "ERF_ConductorSpan.H"

using erf_conductors::ConductorInputs;
using erf_conductors::ConductorSpan;
using erf_conductors::SpanInputs;

namespace {

constexpr double pi = 3.14159265358979323846;   // MSVC has no M_PI

SpanInputs drake_span (const std::string& name)
{
    SpanInputs s;
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

std::unique_ptr<ConductorSpan> make (const std::string& name, const ConductorInputs& in)
{
    const std::string file = in.diagnostics_dir + "/" + name + ".moordyn.txt";
    return std::make_unique<ConductorSpan>(drake_span(name), in, 9.81, file);
}

double weight (const SpanInputs& s, double rho) { return (s.mass_per_length - rho * 0.25 * pi * s.diameter * s.diameter) * 9.81; }
double drag (const SpanInputs& s, double rho, double U) { return 0.5 * rho * s.drag_coefficient * s.diameter * U * U; }

std::vector<amrex::Real> uniform (unsigned n, double u, double v, double w)
{
    std::vector<amrex::Real> uvw(3 * n);
    for (unsigned i = 0; i < n; ++i) { uvw[3*i] = u; uvw[3*i+1] = v; uvw[3*i+2] = w; }
    return uvw;
}

} // namespace

TEST(ConductorSpan, HangsAtTheCatenarySagInStillAirInERFsFrame)
{
    const ConductorInputs in = settings();
    auto span = make("still", in);
    const SpanInputs& s = span->inputs();
    EXPECT_EQ(span->num_nodes(), 21u);
    EXPECT_GE(span->num_kinematics_points(), 21u);
    const auto a = span->node_position(0);
    const auto b = span->node_position(20);
    EXPECT_NEAR(a[0], 100.0, 1.0e-6); EXPECT_NEAR(a[1], 500.0, 1.0e-6); EXPECT_NEAR(a[2], 30.0, 1.0e-6);
    EXPECT_NEAR(b[0], 400.0, 1.0e-6); EXPECT_NEAR(b[1], 500.0, 1.0e-6); EXPECT_NEAR(b[2], 30.0, 1.0e-6);
    EXPECT_NEAR(span->mid_sag(), s.catenary_sag(), 0.25 * s.catenary_sag());
    EXPECT_NEAR(span->mid_offset(), 0.0, 1.0e-3);
    EXPECT_NEAR(span->swing_angle(), 0.0, 1.0e-4);
    const auto m = span->node_position(10);
    EXPECT_NEAR(m[2], 30.0 - span->mid_sag(), 1.0e-6);
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
    for (std::size_t i = 0; i < n.size(); ++i) { EXPECT_NEAR(k[i], n[i], 1.0e-9) << "component " << i; }
    // the MoorDyn input is where the span says
    EXPECT_TRUE(std::filesystem::exists(span->input_file()));
}

TEST(ConductorSpan, APrescribedCrosswindBlowsItOutInLockstepWithERFsClock)
{
    const ConductorInputs in = settings();
    auto span = make("wind", in);
    const SpanInputs& s = span->inputs();
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

TEST(ConductorSpan, SubstepsReachTheSameClock)
{
    ConductorInputs in = settings();
    in.substeps = 4;
    auto span = make("substeps", in);
    span->set_wind(uniform(span->num_kinematics_points(), 0.0, 10.0, 0.0), 0.025);
    span->step(0.0, 0.05);   // four MoorDyn calls of 0.0125 s; the clock check inside would abort otherwise
    EXPECT_GT(span->mid_offset(), -1.0e-9);
}

TEST(ConductorSpan, DiagnosticsRowsCarryTheHeaderOnce)
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

TEST(ConductorSpan, ASpanCreatedFromASavedStateContinuesIt)
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
    ASSERT_GT(a->mid_offset(), 1.0) << "the saved span must be blown out, so a restart from rest would show";
    const std::string state = in.diagnostics_dir + "/saved_a.state";
    a->save(state);

    const std::string file = in.diagnostics_dir + "/saved_b.moordyn.txt";
    auto b = std::make_unique<ConductorSpan>(drake_span("saved_b"), in, 9.81, file, state);
    // before any step: where the saved span is, with its drag and tensions
    ASSERT_EQ(b->num_nodes(), a->num_nodes());
    for (unsigned n = 0; n < a->num_nodes(); ++n) {
        const auto pa = a->node_position(n);
        const auto pb = b->node_position(n);
        const auto da = a->node_drag(n);
        const auto db = b->node_drag(n);
        for (int d = 0; d < 3; ++d) {
            EXPECT_NEAR(pb[d], pa[d], 1.0e-9 * std::max(amrex::Real(1.0), std::abs(pa[d]))) << "node " << n << " position " << d;
            EXPECT_NEAR(db[d], da[d], 1.0e-9 * std::max(amrex::Real(1.0), std::abs(da[d]))) << "node " << n << " drag " << d;
        }
    }
    EXPECT_NEAR(b->tension_a(), a->tension_a(), 1.0e-9 * a->tension_a());
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
            EXPECT_NEAR(pb[d], pa[d], 1.0e-9 * std::max(amrex::Real(1.0), std::abs(pa[d]))) << "node " << n << " position " << d;
        }
    }
    EXPECT_NEAR(b->tension_a(), a->tension_a(), 1.0e-9 * a->tension_a());
}

namespace {
SpanInputs section (const std::string& name, double insulator)
{
    SpanInputs s = drake_span(name);
    s.end_a = {{100.0, 500.0, 30.0}};
    s.end_b = {{1000.0, 500.0, 30.0}};
    s.towers = {{{400.0, 500.0, 30.0}}, {{700.0, 500.0, 30.0}}};
    s.lengths = {301.5, 301.5, 301.5};
    s.insulator_length = insulator;
    s.insulator_mass = 60.0;
    return s;
}
} // namespace

TEST(ConductorSpan, ASectionHangsFromItsInsulatorStrings)
{
    const ConductorInputs in = settings();
    const std::string file = in.diagnostics_dir + "/strings.moordyn.txt";
    ConductorSpan line(section("strings", 2.5), in, 9.81, file);
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
    ConductorSpan clamped(section("clamped", 0.0), in, 9.81, in.diagnostics_dir + "/clamped.moordyn.txt");
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
        EXPECT_NEAR(line.mid_offset(k), m[1] - 500.0, 1.0e-6) << "span " << k;
        EXPECT_NEAR(line.mid_sag(k), 30.0 - m[2], 1.0e-6) << "span " << k;
    }
    EXPECT_GT(std::abs(line.mid_offset(1) - line.mid_offset(0)), 1.0e-3) << "the middle span hangs from strings at both ends";
    EXPECT_GT(clamped.mid_offset(1), 1.0);
}
