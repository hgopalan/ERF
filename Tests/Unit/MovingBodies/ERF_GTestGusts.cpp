// Gusts on a conductor span from a RANS wind (ERF_Gusts.H).
//
// Gusts.TheGustTypeParses: none and factor in any case; anything else is refused.
// Gusts.TheDefaultSigmaFactorGivesTwoAndAHalfUStar: c = 2.5 Cmu0, so at the RANS equilibrium near the ground,
//   k = u*^2 / Cmu0^2, sigma_u = c sqrt(k) is 2.5 u*.
// Gusts.TheBackgroundFactorFallsWithTheSpan: B = 1 / (1 + 0.8 L / L_s) is 1 for a vanishing span, 1/2 when
//   0.8 L = L_s, and falls with L.
// Gusts.ASpanGustMatchesTheHandValues: every field of span_gust against the formulas written out, for a wind at
//   asin(10/12) to the span.
// Gusts.TheNormalFluctuationGoesFromStreamwiseToLateral: sigma_n is sigma_u for a wind normal to the span and
//   0.8 sigma_u (gust_sigma_v_ratio) for one along it, with intensity, factor and loads 0 when U_n = 0; the
//   linear form is marked valid while U_n >= sigma_n.
// Gusts.WithTheExposuresIntensityItIsASCE74sFactor: for a wind normal to the span with the turbulence intensity
//   of ASCE 74's exposure B or C, I = E/2 = 2.45 sqrt(kappa) (33 ft / z)^(1/alpha), g = 2.7 and that exposure's
//   L_s (the default, 220 ft, is exposure C's), the gust response factor is ASCE 74's G_w kv^2 = 1 + 2.7 E sqrt(B_w)
//   at every height and span.
// Gusts.CalmAirHasNoIntensityAndNoLoad: U = 0 leaves the intensity, the factor and the loads at 0, the point
//   gust at g sigma_u.
// Gusts.BadArgumentsAbort: each refused argument aborts with its message.

#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "ERF_ASCE74.H"
#include "ERF_Gusts.H"
#include "ERF_GTestThrowOnAbort.H"

using namespace erf_conductors;

TEST(Gusts, TheGustTypeParses)
{
    GustType g = GustType::Factor;
    EXPECT_TRUE(parse_gust_type("none", g));
    EXPECT_EQ(g, GustType::None);
    EXPECT_TRUE(parse_gust_type("Factor", g));
    EXPECT_EQ(g, GustType::Factor);
    EXPECT_TRUE(parse_gust_type("NONE", g));
    EXPECT_EQ(g, GustType::None);
    EXPECT_FALSE(parse_gust_type("event", g));
    EXPECT_FALSE(parse_gust_type("", g));
    EXPECT_EQ(g, GustType::None) << "a refused name leaves the type alone";
}

TEST(Gusts, TheDefaultSigmaFactorGivesTwoAndAHalfUStar)
{
    const double Cmu0 = 0.5562;   // ERF's default erf.Cmu0
    EXPECT_DOUBLE_EQ(default_gust_sigma_factor(Cmu0), 2.5 * Cmu0);
    for (const double ustar : {0.3, 0.6, 1.1}) {
        const double k = ustar * ustar / (Cmu0 * Cmu0);
        EXPECT_NEAR(default_gust_sigma_factor(Cmu0) * std::sqrt(k), 2.5 * ustar, 1e-12);
    }
}

TEST(Gusts, TheBackgroundFactorFallsWithTheSpan)
{
    EXPECT_NEAR(gust_background_factor(1.0e-9, 67.056), 1.0, 1e-9);
    EXPECT_DOUBLE_EQ(gust_background_factor(67.056 / 0.8, 67.056), 0.5);
    EXPECT_GT(gust_background_factor(100.0, 67.056), gust_background_factor(300.0, 67.056));
}

TEST(Gusts, ASpanGustMatchesTheHandValues)
{
    const double U = 12.0, Un = 10.0, k = 2.0, L = 200.0, d = 0.0281, cd = 1.0, rho = 1.225;
    const double c = 1.39, g = 2.7, Ls = 67.056;
    const SpanGust s = span_gust(U, Un, k, L, d, cd, rho, c, g, Ls);
    const double sigma = 1.39 * std::sqrt(2.0);
    const double sin_phi = 10.0 / 12.0;
    const double sigma_n = std::sqrt(sigma * sigma * sin_phi * sin_phi + 0.64 * sigma * sigma * (1.0 - sin_phi * sin_phi));
    const double I = sigma_n / 10.0;
    const double B = 1.0 / (1.0 + 0.8 * 200.0 / 67.056);
    const double G = 1.0 + 2.0 * 2.7 * I * std::sqrt(B);
    const double mean = 0.5 * 1.225 * 1.0 * 0.0281 * 10.0 * 10.0;
    EXPECT_NEAR(s.sigma, sigma, 1e-12);
    EXPECT_NEAR(s.normal_sigma, sigma_n, 1e-12);
    EXPECT_NEAR(s.intensity, I, 1e-12);
    EXPECT_NEAR(s.gust_response, G, 1e-12);
    EXPECT_NEAR(s.gust_wind, 12.0 + 2.7 * sigma, 1e-12);
    EXPECT_NEAR(s.mean_load, mean, 1e-12);
    EXPECT_NEAR(s.peak_load, G * mean, 1e-12);
    // the values themselves, so that a slip in the formulas above cannot cancel one in the code
    EXPECT_NEAR(s.normal_sigma, 1.854491, 1e-6);
    EXPECT_NEAR(s.intensity, 0.185449, 1e-6);
    EXPECT_NEAR(s.gust_response, 1.544215, 1e-6);
    EXPECT_NEAR(s.peak_load, 2.657788, 1e-6);
    EXPECT_TRUE(s.linear_valid) << "U_n = 10 m/s against sigma_n = 1.85 m/s";
}

TEST(Gusts, TheNormalFluctuationGoesFromStreamwiseToLateral)
{
    const double sigma = 1.39 * std::sqrt(1.5);
    const SpanGust normal = span_gust(10.0, 10.0, 1.5, 150.0, 0.03, 1.0, 1.225, 1.39, 2.7, 67.056);
    EXPECT_NEAR(normal.normal_sigma, sigma, 1e-12);
    const SpanGust along = span_gust(10.0, 0.0, 1.5, 150.0, 0.03, 1.0, 1.225, 1.39, 2.7, 67.056);
    EXPECT_NEAR(along.normal_sigma, gust_sigma_v_ratio * sigma, 1e-12);
    EXPECT_EQ(along.intensity, 0.0);
    EXPECT_EQ(along.gust_response, 0.0);
    EXPECT_EQ(along.mean_load, 0.0);
    EXPECT_EQ(along.peak_load, 0.0);
    EXPECT_NEAR(along.gust_wind, 10.0 + 2.7 * sigma, 1e-12);
    // in between, sigma_n lies between the two and the factor rises as the normal wind falls
    const SpanGust oblique = span_gust(10.0, 5.0, 1.5, 150.0, 0.03, 1.0, 1.225, 1.39, 2.7, 67.056);
    EXPECT_GT(oblique.normal_sigma, along.normal_sigma);
    EXPECT_LT(oblique.normal_sigma, normal.normal_sigma);
    EXPECT_GT(oblique.gust_response, normal.gust_response);
    // the linear form holds while U_n >= sigma_n: here sigma_n is about 1.6 m/s
    EXPECT_TRUE(normal.linear_valid);
    EXPECT_TRUE(oblique.linear_valid);
    EXPECT_FALSE(span_gust(10.0, 1.0, 1.5, 150.0, 0.03, 1.0, 1.225, 1.39, 2.7, 67.056).linear_valid);
    EXPECT_FALSE(along.linear_valid);
}

TEST(Gusts, WithTheExposuresIntensityItIsASCE74sFactor)
{
    const double ft = 0.3048, kv = 1.43, U = 10.0;
    for (const Exposure e : {Exposure::B, Exposure::C}) {
        const ExposureConstants ec = exposure_constants(e);
        for (const double z : {10.0, 25.0, 60.0}) {
            const double E = 4.9 * std::sqrt(ec.kappa) * std::pow(33.0 * ft / z, 1.0 / ec.alpha);
            const double I = 0.5 * E;
            const double k = (I * U) * (I * U);   // sigma_u = I U with c = 1
            for (const double L : {50.0, 200.0, 600.0}) {
                const SpanGust s = span_gust(U, U, k, L, 0.03, 1.0, 1.225, 1.0, 2.7, ec.Ls);
                EXPECT_NEAR(s.gust_response, wire_gust_response_factor(e, z, L) * kv * kv, 1e-12)
                    << "exposure " << (e == Exposure::B ? "B" : "C") << ", z " << z << ", span " << L;
            }
        }
    }
}

TEST(Gusts, CalmAirHasNoIntensityAndNoLoad)
{
    const SpanGust s = span_gust(0.0, 0.0, 0.5, 150.0, 0.03, 1.0, 1.225, 1.39, 2.7, 67.056);
    EXPECT_EQ(s.intensity, 0.0);
    EXPECT_EQ(s.gust_response, 0.0);
    EXPECT_EQ(s.mean_load, 0.0);
    EXPECT_EQ(s.peak_load, 0.0);
    EXPECT_NEAR(s.gust_wind, 2.7 * 1.39 * std::sqrt(0.5), 1e-12);
    EXPECT_NEAR(s.normal_sigma, gust_sigma_v_ratio * 1.39 * std::sqrt(0.5), 1e-12) << "calm air: no direction, the lateral part";
}

TEST(Gusts, BadArgumentsAbort)
{
    using erf_gtest::abort_message;
    auto gust = [] (double U, double Un, double k, double L, double Ls) {
        return [=] { span_gust(U, Un, k, L, 0.03, 1.0, 1.225, 1.39, 2.7, Ls); };
    };
    EXPECT_NE(abort_message(gust(-1.0, 0.0, 1.0, 100.0, 67.0)).find("the wind must be finite"), std::string::npos);
    EXPECT_NE(abort_message(gust(5.0, 6.0, 1.0, 100.0, 67.0)).find("the normal wind must lie between"), std::string::npos);
    EXPECT_NE(abort_message(gust(5.0, 4.0, -0.1, 100.0, 67.0)).find("k must be finite"), std::string::npos);
    EXPECT_NE(abort_message(gust(5.0, 4.0, 1.0, 0.0, 67.0)).find("the span must be positive"), std::string::npos);
    EXPECT_NE(abort_message(gust(5.0, 4.0, 1.0, 100.0, 0.0)).find("the length scale must be positive"), std::string::npos);
    EXPECT_NE(abort_message([] { span_gust(5.0, 4.0, 1.0, 100.0, 0.0, 1.0, 1.225, 1.39, 2.7, 67.0); })
                  .find("the diameter and air density"), std::string::npos);
    EXPECT_NE(abort_message([] { span_gust(5.0, 4.0, 1.0, 100.0, 0.03, 1.0, 1.225, 0.0, 2.7, 67.0); })
                  .find("the sigma and peak factors"), std::string::npos);
    EXPECT_NE(abort_message([] { default_gust_sigma_factor(0.0); }).find("Cmu0 must be positive"), std::string::npos);
    // a normal wind above the speed by roundoff only is accepted
    EXPECT_TRUE(abort_message(gust(5.0, 5.0 * (1.0 + 1.0e-12), 1.0, 100.0, 67.0)).empty());
}
