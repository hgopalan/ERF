// The quasi-static wind load on a conductor span by ASCE Manual of Practice 74 (ERF_ASCE74.H).
//
// ASCE74.ExposureFactorIsOneAtTenMetresOverOpenCountry: k_z at 33 ft in exposure C is 1 (to the 2.01
//   rounding of the constant), lower over suburban terrain, and rises with height.
// ASCE74.WireLoadRatiosArePeyrots: k_z G_w for a Drake conductor at the five attachment heights and
//   spans of Peyrot (ASCE ETSS Conference 2009, Tables 1 and 2, "ASCE 2008"): exposure C 0.80 0.78 0.80
//   0.83 0.85, exposure B 0.64 0.62 0.65 0.68 0.70. The 100 m spans agree to the table's two digits;
//   the longer spans within 0.03, the paper not stating the height it took for a sagging span.
// ASCE74.GustResponseFactorLimits: G_w falls with the span towards 1/k_v^2 and reaches
//   (1 + 2.7 E)/k_v^2 as the span vanishes; rougher terrain gusts more.
// ASCE74.TheSpanSwingsAndStretchesUnderTheResultant: the load is Q k_z V^2 G_w C_f d; the swing is
//   atan(load/weight); the sag and the tension are the elastic catenary's under the resultant; the
//   blowout is the sag times sin(swing); no wind leaves the still-air catenary.

#include <cmath>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "ERF_ASCE74.H"
#include "ERF_ConductorInputs.H"

using namespace erf_conductors;

namespace {
constexpr double ft = 0.3048;
// the catenary is solved in amrex::Real
constexpr double rel = (std::is_same<amrex::Real, float>::value) ? 1.0e-5 : 1.0e-10;
}

TEST(ASCE74, ExposureFactorIsOneAtTenMetresOverOpenCountry)
{
    EXPECT_NEAR(exposure_factor(Exposure::C, 33.0 * ft), 1.0, 0.005);
    EXPECT_LT(exposure_factor(Exposure::B, 33.0 * ft), exposure_factor(Exposure::C, 33.0 * ft));
    EXPECT_LT(exposure_factor(Exposure::C, 20.0), exposure_factor(Exposure::C, 40.0));
    // at the gradient height the factor is 2.01 in either exposure
    EXPECT_NEAR(exposure_factor(Exposure::C, 900.0 * ft), 2.01, 1e-12);
    EXPECT_NEAR(exposure_factor(Exposure::B, 1200.0 * ft), 2.01, 1e-12);
    Exposure e = Exposure::C;
    EXPECT_TRUE(parse_exposure("b", e));
    EXPECT_EQ(e, Exposure::B);
    EXPECT_FALSE(parse_exposure("D", e));
}

TEST(ASCE74, WireLoadRatiosArePeyrots)
{
    const double heights[5] = {10.0, 20.0, 30.0, 40.0, 50.0}, spans[5] = {100.0, 350.0, 500.0, 620.0, 720.0};
    const double table_c[5] = {0.80, 0.78, 0.80, 0.83, 0.85}, table_b[5] = {0.64, 0.62, 0.65, 0.68, 0.70};
    for (int i = 0; i < 5; ++i) {
        const double tol = (i == 0) ? 0.005 : 0.03;
        const double c = exposure_factor(Exposure::C, heights[i]) * wire_gust_response_factor(Exposure::C, heights[i], spans[i]);
        const double b = exposure_factor(Exposure::B, heights[i]) * wire_gust_response_factor(Exposure::B, heights[i], spans[i]);
        EXPECT_NEAR(c, table_c[i], tol) << heights[i] << " m, " << spans[i] << " m, exposure C";
        EXPECT_NEAR(b, table_b[i], tol) << heights[i] << " m, " << spans[i] << " m, exposure B";
    }
    // the reference load of the paper: a 1 m Drake rod 10 m up in a 40 m/s gust, 0.613 x 40^2 x 0.0281 = 27.6 N/m
    const auto w = wire_wind_load(Exposure::C, 40.0, 10.0, 100.0, 0.0281, 1.0, 15.97, 101.0, 2.9e7, 1.226);
    EXPECT_NEAR(w.load / (0.613 * 40.0 * 40.0 * 0.0281), 0.80, 0.005);
}

TEST(ASCE74, GustResponseFactorLimits)
{
    const double kv2 = 1.43 * 1.43;
    for (const Exposure e : {Exposure::B, Exposure::C}) {
        const ExposureConstants c = exposure_constants(e);
        const double z = 25.0;
        const double E = 4.9 * std::sqrt(c.kappa) * std::pow(33.0 * ft / z, 1.0 / c.alpha);
        EXPECT_NEAR(wire_gust_response_factor(e, z, 1.0e-9), (1.0 + 2.7 * E) / kv2, 1e-9);
        EXPECT_NEAR(wire_gust_response_factor(e, z, 1.0e9), 1.0 / kv2, 1e-3);
        double last = 10.0;
        for (const double L : {50.0, 100.0, 200.0, 400.0, 800.0}) {
            const double g = wire_gust_response_factor(e, z, L);
            EXPECT_LT(g, last) << L;
            last = g;
        }
    }
    EXPECT_GT(wire_gust_response_factor(Exposure::B, 25.0, 300.0), wire_gust_response_factor(Exposure::C, 25.0, 300.0));
}

TEST(ASCE74, TheSpanSwingsAndStretchesUnderTheResultant)
{
    const double d = 0.0281, W = 15.97, chord = 300.0, length = 301.5, EA = 3.0e7, rho = 1.2, V = 35.0, z = 25.0;
    const auto w = wire_wind_load(Exposure::C, V, z, chord, d, 1.1, W, length, EA, rho);
    const double kz = exposure_factor(Exposure::C, z), gw = wire_gust_response_factor(Exposure::C, z, chord);
    EXPECT_NEAR(w.kz, kz, 1e-15);
    EXPECT_NEAR(w.gust_response, gw, 1e-15);
    EXPECT_NEAR(w.pressure, 0.5 * rho * kz * V * V, 1e-12);
    EXPECT_NEAR(w.load, 0.5 * rho * kz * V * V * gw * 1.1 * d, 1e-12);
    EXPECT_NEAR(w.swing, std::atan(w.load / W), 1e-15);
    const Catenary cat = elastic_catenary(amrex::Real(chord), amrex::Real(length), amrex::Real(std::hypot(w.load, W)), amrex::Real(EA));
    EXPECT_NEAR(w.sag, static_cast<double>(cat.sag), rel * w.sag);
    EXPECT_NEAR(w.tension, static_cast<double>(cat.end_tension), rel * w.tension);
    EXPECT_NEAR(w.blowout, w.sag * std::sin(w.swing), 1e-12);
    // under its weight and the wind the span pulls harder than in still air, and swings out
    const auto still = wire_wind_load(Exposure::C, 0.0, z, chord, d, 1.1, W, length, EA, rho);
    EXPECT_EQ(still.load, 0.0);
    EXPECT_EQ(still.swing, 0.0);
    EXPECT_EQ(still.blowout, 0.0);
    const Catenary air = elastic_catenary(amrex::Real(chord), amrex::Real(length), amrex::Real(W), amrex::Real(EA));
    EXPECT_NEAR(still.tension, static_cast<double>(air.end_tension), rel * still.tension);
    EXPECT_GT(w.tension, still.tension);
    EXPECT_GT(w.swing, 0.8);   // a 35 m/s gust blows a Drake span well out (about 50 degrees)
}
