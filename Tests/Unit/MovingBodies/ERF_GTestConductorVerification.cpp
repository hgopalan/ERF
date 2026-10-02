// Verification of a conductor span against analytic results, run against the real MoorDyn-C (the
// bundled stub has no line dynamics, so these tests skip on it). A 300 m span of 795 kcmil ACSR with
// 1.5 m of slack in air: in a steady crosswind U normal to it, the mean swing angle of the middle
// node about the chord is the quasi-static blowout angle atan(q / w), q = rho Cd D U^2 / 2 the drag
// and w the weight per unit length; released after a short gust into still air, it swings at the
// first out-of-plane frequency of a cable, whose period T = 2 c / sqrt(H / m) does not depend on the
// sag (Irvine, Cable Structures, 1981), with c the chord, H the horizontal tension and m the mass
// per unit length; and in still air it hangs in the elastic catenary.

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ERF_ConductorInputs.H"
#include "ERF_ConductorSpan.H"
#include "ERF_MoorDynSystem.H"

using erf_conductors::ConductorInputs;
using erf_conductors::ConductorSpan;
using erf_conductors::SpanInputs;

namespace {

constexpr double pi = 3.14159265358979323846;   // MSVC has no M_PI
constexpr double g = 9.81;

SpanInputs drake (const std::string& name)
{
    SpanInputs s;
    s.name = name;
    s.end_a = {{100.0, 500.0, 30.0}};
    s.end_b = {{400.0, 500.0, 30.0}};
    s.length = 301.5; s.diameter = 0.0281; s.mass_per_length = 1.628; s.axial_stiffness = 3.0e7;
    s.output_root = (std::filesystem::temp_directory_path() / "erf_gtest_conductor_verification" / name).string();
    return s;
}

std::unique_ptr<ConductorSpan> make (const std::string& name, ConductorInputs& in)
{
    in.air_density = 1.2;
    in.diagnostics_dir = (std::filesystem::temp_directory_path() / "erf_gtest_conductor_verification").string();
    return std::make_unique<ConductorSpan>(drake(name), in, g, in.diagnostics_dir + "/" + name + ".moordyn.txt");
}

void blow (ConductorSpan& span, double U, double t, double dt)
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
    const SpanInputs& s = span->inputs();
    const double w = (s.mass_per_length - in.air_density * 0.25 * pi * s.diameter * s.diameter) * g;
    const auto cat = erf_conductors::elastic_catenary(s.chord(), s.length, w, s.axial_stiffness);
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
        const SpanInputs& s = span->inputs();
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
    const SpanInputs& s = span->inputs();
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
