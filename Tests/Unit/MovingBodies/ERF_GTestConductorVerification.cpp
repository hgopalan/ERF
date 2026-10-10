// Verification of conductor lines against analytic results, run against the real MoorDyn-C (the
// bundled stub has no line dynamics, so these tests skip on it). The line is a single 300 m span of
// 795 kcmil (thousand circular mils) ACSR (aluminium conductor, steel reinforced) "Drake" with
// 1.5 m of slack in air, or a section of three such spans over two suspension towers. q = rho Cd D
// U^2 / 2 is the drag (N/m) and w the weight (N/m) per unit length; c the chord (m), H the horizontal
// tension (N), m the mass per unit length (kg/m); the wind span and the weight span of a tower are
// the lengths of conductor whose wind load and weight it carries.
//
// StillAirShapeIsTheElasticCatenary: the still-air sag and end tension of the elastic catenary.
// MeanSwingIsTheQuasiStaticBlowoutAngle: in a steady crosswind U normal to the span, the mean swing of
//     the middle node about the chord is atan(q / w).
// FreeSwingHasTheCableOutOfPlanePeriod: released after a short gust, the span swings with the cable's
//     first out-of-plane period T = 2 c / sqrt(H / m), which does not depend on the sag (Irvine,
//     Cable Structures, 1981).
// SuspensionStringsSwingToTheWindSpanOverWeightSpanAngle: the strings swing across the line to the
//     angle of the wind span's load over the weight span's.
// TheDeadEndsCarryTheWeightAndTheCatenarysHorizontalTension: the dead-end pulls of a level span carry
//     its weight between them, with the elastic catenary's horizontal tension.
// ASuspensionTowerTakesTheWindSpanAndTheWeightSpan: a tower takes q L plus the string's own drag
//     across the line, and the weight span's weight down.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../ERF_GTestTempDir.H"
#include "ERF_ConductorInputs.H"
#include "ERF_ConductorLine.H"
#include "ERF_MoorDynSystem.H"

using erf_conductors::ConductorInputs;
using erf_conductors::ConductorLine;
using erf_conductors::LineInputs;

namespace {

// a scratch root drawn once per test process (ERF_GTestTempDir.H): the fixed names below it are this
// process's alone, so ctest -j and the shuffled rerun never share them; removed when the process exits
const std::filesystem::path& gtest_scratch_root ()
{
    struct Root {
        std::filesystem::path p;
        ~Root () { std::error_code ec; std::filesystem::remove_all(p, ec); }
    };
    static const Root root{[] {
        const std::filesystem::path p = erf_gtest_temp_path("erf_gtest_conductorverification");
        std::filesystem::create_directories(p);
        return p;
    }()};
    return root.p;
}

constexpr double pi = 3.14159265358979323846;   // MSVC has no M_PI
constexpr double g = 9.81;

LineInputs drake (const std::string& name)
{
    LineInputs s;
    s.name = name;
    s.end_a = {{100.0, 500.0, 30.0}};
    s.end_b = {{400.0, 500.0, 30.0}};
    s.lengths = {301.5}; s.diameter = 0.0281; s.mass_per_length = 1.628; s.axial_stiffness = 3.0e7;
    s.output_root = (gtest_scratch_root() / "erf_gtest_conductor_verification" / name).string();
    return s;
}

std::unique_ptr<ConductorLine> make (const std::string& name, ConductorInputs& in)
{
    in.air_density = 1.2;
    in.diagnostics_dir = (gtest_scratch_root() / "erf_gtest_conductor_verification").string();
    return std::make_unique<ConductorLine>(drake(name), in, g, in.diagnostics_dir + "/" + name + ".moordyn.txt");
}

void blow (ConductorLine& span, double U, double t, double dt)
{
    std::vector<amrex::Real> uvw(3 * static_cast<std::size_t>(span.num_kinematics_points()), 0.0);
    for (std::size_t p = 0; p < uvw.size() / 3; ++p) { uvw[3*p+1] = U; }
    span.set_wind(uvw, t + 0.5 * dt);
    span.step(t, dt);
}

} // namespace

TEST(ConductorVerification, StillAirShapeIsTheElasticCatenary)
{
    if (erf_moordyn::is_stub()) { GTEST_SKIP() << "needs the real MoorDyn-C: the stub hangs a parabola"; }
    ConductorInputs in;
    auto span = make("catenary", in);
    const LineInputs& s = span->inputs();
    const double w = (s.mass_per_length - in.air_density * 0.25 * pi * s.diameter * s.diameter) * g;
    const auto cat = erf_conductors::elastic_catenary(s.chord(), s.lengths[0], w, s.axial_stiffness);
    // MoorDyn's lumped masses converge to the catenary from above: 0.11 % in sag with 20 segments
    EXPECT_NEAR(span->mid_sag() / cat.sag, 1.0, 0.005) << "sag " << span->mid_sag() << " m, catenary " << cat.sag << " m";
    EXPECT_NEAR(span->tension_a() / cat.end_tension, 1.0, 0.005) << "end tension " << span->tension_a() << " N, catenary "
        << cat.end_tension << " N";
}

TEST(ConductorVerification, MeanSwingIsTheQuasiStaticBlowoutAngle)
{
    if (erf_moordyn::is_stub()) { GTEST_SKIP() << "needs the real MoorDyn-C: the stub has no line dynamics"; }
    for (const double U : {10.0, 20.0, 30.0}) {
        ConductorInputs in;
        auto span = make("blowout" + std::to_string(static_cast<int>(U)), in);
        const LineInputs& s = span->inputs();
        const double w = (s.mass_per_length - in.air_density * 0.25 * pi * s.diameter * s.diameter) * g;
        const double q = 0.5 * in.air_density * s.drag_coefficient * s.diameter * U * U;
        const double phi_static = std::atan2(q, w);
        const double dt = 0.05;
        double t = 0.0, sum = 0.0;
        int n = 0;
        while (t < 60.0 - 0.5 * dt) {
            blow(*span, U, t, dt);
            t += dt;
            if (t > 20.0) { sum += span->swing_angle(); ++n; }
        }
        const double phi = sum / n;
        EXPECT_NEAR(phi / phi_static, 1.0, 0.01) << U << " m/s: mean swing " << phi * 180.0 / pi
            << " deg, atan(q/w) " << phi_static * 180.0 / pi << " deg";
    }
}

TEST(ConductorVerification, FreeSwingHasTheCableOutOfPlanePeriod)
{
    if (erf_moordyn::is_stub()) { GTEST_SKIP() << "needs the real MoorDyn-C: the stub has no line dynamics"; }
    ConductorInputs in;
    auto span = make("swing", in);
    const LineInputs& s = span->inputs();
    // the horizontal tension of the still-air span from MoorDyn's own solution (the chord runs along x)
    const double H = std::fabs(span->system().line_node_tension(1, 0)[0]);
    const double T_theory = 2.0 * s.chord() / std::sqrt(H / s.mass_per_length);
    const double dt = 0.05;
    double t = 0.0, prev = 0.0;
    std::vector<double> crossings;
    while (t < 40.0 - 0.5 * dt) {
        blow(*span, (t < 1.0) ? 10.0 : 0.0, t, dt);   // a 1 s gust, then still air
        t += dt;
        const double y = span->mid_offset();
        if (t > 1.0 && prev > 0.0 && y <= 0.0) { crossings.push_back(t - dt * y / (y - prev)); }
        prev = y;
    }
    ASSERT_GE(crossings.size(), 4u) << "the span must keep swinging after the gust";
    const double T = (crossings.back() - crossings.front()) / static_cast<double>(crossings.size() - 1);
    EXPECT_NEAR(T / T_theory, 1.0, 0.01) << "swing period " << T << " s, cable theory " << T_theory
        << " s (H " << H << " N)";
}

TEST(ConductorVerification, SuspensionStringsSwingToTheWindSpanOverWeightSpanAngle)
{
    if (erf_moordyn::is_stub()) { GTEST_SKIP() << "needs the real MoorDyn-C: the stub hangs its strings at the span's swing"; }
    // three equal spans over two suspension towers: each string carries half of each span next to
    // it, and the spans either side balance their pull along the line, so the string swings across
    // the line to tan(theta) = (q L + q_i L_i cos(theta) / 2) / (w L + W_i / 2), the moments about the
    // string's top of its wind span and weight span, its own weight, and its own drag q_i L_i cos^2(theta)
    // normal to it at half its length (the swung string sees the normal wind U cos(theta))
    ConductorInputs in;
    in.air_density = 1.2;
    in.diagnostics_dir = (gtest_scratch_root() / "erf_gtest_conductor_verification").string();
    LineInputs s = drake("section");
    s.end_b = {{1000.0, 500.0, 30.0}};
    s.towers = {{{400.0, 500.0, 30.0}}, {{700.0, 500.0, 30.0}}};
    s.lengths = {301.5, 301.5, 301.5};
    s.insulator_length = 2.5;
    s.insulator_mass = 60.0;
    ConductorLine line(s, in, g, in.diagnostics_dir + "/section.moordyn.txt");
    const double U = 20.0, dt = 0.05;
    double t = 0.0, sum = 0.0, along = 0.0;
    int n = 0;
    while (t < 60.0 - 0.5 * dt) {
        blow(line, U, t, dt);
        t += dt;
        if (t > 30.0) {
            sum += 0.5 * (line.insulator_swing_across(0) + line.insulator_swing_across(1));
            along = std::max(along, std::sqrt(std::max(0.0, std::pow(double(line.insulator_swing(0)), 2) -
                                                             std::pow(double(line.insulator_swing_across(0)), 2))));
            ++n;
        }
    }
    const double rho = in.air_density, L = 301.5, Li = s.insulator_length;
    const double q = 0.5 * rho * s.drag_coefficient * s.diameter * U * U;
    const double w = (s.mass_per_length - rho * 0.25 * pi * s.diameter * s.diameter) * g;
    const double qi = 0.5 * rho * LineInputs::insulator_drag_coefficient * s.insulator_diameter * U * U;
    const double Wi = (s.insulator_mass - rho * 0.25 * pi * s.insulator_diameter * s.insulator_diameter * Li) * g;
    double theta = std::atan((q * L + 0.5 * qi * Li) / (w * L + 0.5 * Wi));
    for (int it = 0; it < 50; ++it) { theta = std::atan((q * L + 0.5 * qi * Li * std::cos(theta)) / (w * L + 0.5 * Wi)); }
    const double mean = sum / n;
    RecordProperty("string_swing_deg", std::to_string(mean * 180.0 / pi));
    RecordProperty("wind_span_over_weight_span_deg", std::to_string(theta * 180.0 / pi));
    RecordProperty("largest_along_line_deg", std::to_string(along * 180.0 / pi));
    EXPECT_NEAR(mean / theta, 1.0, 0.002) << "mean string swing " << mean * 180.0 / pi << " deg, wind span / weight span "
                                         << theta * 180.0 / pi << " deg";
    EXPECT_LT(along, 0.01) << "the spans either side balance the pull along the line";
}

TEST(ConductorVerification, TheDeadEndsCarryTheWeightAndTheCatenarysHorizontalTension)
{
    if (erf_moordyn::is_stub()) { GTEST_SKIP() << "needs the real MoorDyn-C: the stub hangs a parabola"; }
    ConductorInputs in;
    auto span = make("dead_ends", in);
    const LineInputs& s = span->inputs();
    const double w = (s.mass_per_length - in.air_density * 0.25 * pi * s.diameter * s.diameter) * g;
    const auto cat = erf_conductors::elastic_catenary(s.chord(), s.lengths[0], w, s.axial_stiffness);
    const auto a = span->end_force(0);
    const auto b = span->end_force(1);
    // each support is pulled towards the other and down: H along the chord, half the weight each
    RecordProperty("end_a_horizontal_N", std::to_string(a[0]));
    RecordProperty("catenary_horizontal_N", std::to_string(cat.horizontal_tension));
    RecordProperty("ends_vertical_N", std::to_string(-(a[2] + b[2])));
    RecordProperty("line_weight_N", std::to_string(w * s.lengths[0]));
    EXPECT_NEAR(a[0] / cat.horizontal_tension, 1.0, 0.005);
    EXPECT_NEAR(-b[0] / cat.horizontal_tension, 1.0, 0.005);
    EXPECT_NEAR(-(a[2] + b[2]) / (w * s.lengths[0]), 1.0, 0.005) << "the two ends share the line's weight";
    EXPECT_NEAR(a[2], b[2], 1.0e-3 * w * s.lengths[0]) << "a level span loads its ends alike";
    EXPECT_NEAR(a[1], 0.0, 1.0e-6 * cat.horizontal_tension);
}

TEST(ConductorVerification, ASuspensionTowerTakesTheWindSpanAndTheWeightSpan)
{
    if (erf_moordyn::is_stub()) { GTEST_SKIP() << "needs the real MoorDyn-C: the stub hangs its strings at the span's swing"; }
    // four 300 m spans on strings: the middle tower's spans either side both hang from strings, so
    // it takes half of each, with nothing shifted onto a dead end
    ConductorInputs in;
    in.air_density = 1.2;
    in.diagnostics_dir = (gtest_scratch_root() / "erf_gtest_conductor_verification").string();
    LineInputs s = drake("tower_loads");
    s.end_b = {{1300.0, 500.0, 30.0}};
    s.towers = {{{400.0, 500.0, 30.0}}, {{700.0, 500.0, 30.0}}, {{1000.0, 500.0, 30.0}}};
    s.lengths = {301.5, 301.5, 301.5, 301.5};
    s.insulator_length = 2.5;
    s.insulator_mass = 60.0;
    ConductorLine line(s, in, g, in.diagnostics_dir + "/tower_loads.moordyn.txt");
    const double U = 20.0, dt = 0.05;
    double t = 0.0, swing = 0.0;
    std::array<double,3> mean{{0.0, 0.0, 0.0}};
    int n = 0;
    while (t < 60.0 - 0.5 * dt) {
        blow(line, U, t, dt);
        t += dt;
        if (t > 30.0) {
            const auto F = line.tower_force(1);
            for (int d = 0; d < 3; ++d) { mean[d] += F[d]; }
            swing += line.insulator_swing_across(1);
            ++n;
        }
    }
    for (auto& m : mean) { m /= n; }
    swing /= n;
    // across the line: the wind span (half of each span either side) and the string's drag; down:
    // the weight span and the string's weight, less the lift of the string's drag. A string swung
    // theta across the line sees the normal wind U cos(theta), along (cos theta, sin theta) in the
    // plane across the line, so its drag is q_i L_i cos^2(theta) (cos theta across, sin theta up).
    // (The conductor's sloping segments, swung out of the vertical, lift a few newtons more.)
    const double rho = in.air_density, L = 301.5, Li = s.insulator_length;
    const double q = 0.5 * rho * s.drag_coefficient * s.diameter * U * U;
    const double w = (s.mass_per_length - rho * 0.25 * pi * s.diameter * s.diameter) * g;
    const double Di = 0.5 * rho * LineInputs::insulator_drag_coefficient * s.insulator_diameter * U * U * Li *
                      std::cos(swing) * std::cos(swing);
    const double Wi = (s.insulator_mass - rho * 0.25 * pi * s.insulator_diameter * s.insulator_diameter * Li) * g;
    const double across = q * L + Di * std::cos(swing), down = w * L + Wi - Di * std::sin(swing);
    RecordProperty("tower_across_N", std::to_string(mean[1]));
    RecordProperty("wind_span_N", std::to_string(across));
    RecordProperty("tower_down_N", std::to_string(-mean[2]));
    RecordProperty("weight_span_N", std::to_string(down));
    EXPECT_NEAR(mean[1] / across, 1.0, 0.01);
    EXPECT_NEAR(-mean[2] / down, 1.0, 0.01);
    EXPECT_NEAR(mean[0], 0.0, 0.01 * down) << "the spans either side balance along the line";
}
