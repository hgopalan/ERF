// Gusts on the conductors from a RANS wind (ERF_Gusts.H).
//
// Gusts.TheGustTypeParses: none, factor, event and random in any case; anything else is refused.
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
// Gusts.TheTowersBackgroundFactorIsASCE74s: B_t = 1 / (1 + 0.375 h / L_s) is 1/2 when 0.375 h = L_s and falls with h.
// Gusts.TheIntegralLengthIsIECs: L_u = 8.1 Lambda_1, Lambda_1 = 0.7 z below 60 m and 42 m above, continuous at 60 m.
// Gusts.TheEventIsOneMinusCosineAndArrivesWithItsFront: the shape is 0 at its ends and outside, 1 in the middle and
//   symmetric; the front reaches a point along the gust's direction d / c after the origin, and a point beside it
//   at the same time.
// Gusts.TheNormalNumbersAreStandardAndIndependent: over 200000 steps of one stream the mean is 0, the variance 1
//   and the fourth moment 3; neighbouring steps, two streams and two seeds are uncorrelated; the same arguments
//   give the same number.
// Gusts.TheOrnsteinUhlenbeckStepIsExact: one step against the formula, dt = 0 and T infinite hold z, dt >> T forgets it.
// Gusts.TheRandomGustHasUnitVarianceAndExponentialCorrelation: a 200000-step series (dt = 1 s, T = 10 s) has
//   variance 1 and the autocorrelation exp(-tau / T) at tau = 5, 10 and 20 s.
// Gusts.BadArgumentsAbort: each refused argument aborts with its message.

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

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
    EXPECT_TRUE(parse_gust_type("Event", g));
    EXPECT_EQ(g, GustType::Event);
    EXPECT_TRUE(parse_gust_type("random", g));
    EXPECT_EQ(g, GustType::Random);
    EXPECT_TRUE(parse_gust_type("none", g));
    EXPECT_FALSE(parse_gust_type("turbsim", g));
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

TEST(Gusts, TheTowersBackgroundFactorIsASCE74s)
{
    EXPECT_DOUBLE_EQ(tower_background_factor(67.056 / 0.375, 67.056), 0.5);
    EXPECT_NEAR(tower_background_factor(40.0, 67.056), 0.817198011114, 1e-12);
    EXPECT_GT(tower_background_factor(20.0, 67.056), tower_background_factor(60.0, 67.056));
}

TEST(Gusts, TheIntegralLengthIsIECs)
{
    EXPECT_NEAR(gust_integral_length(30.0), 170.1, 1e-12);
    EXPECT_NEAR(gust_integral_length(59.9), 339.633, 1e-12);
    EXPECT_NEAR(gust_integral_length(60.0), 340.2, 1e-12);
    EXPECT_NEAR(gust_integral_length(150.0), 340.2, 1e-12);
    EXPECT_NEAR(gust_integral_length(60.0 - 1e-9), gust_integral_length(60.0), 1e-6) << "continuous at 60 m";
}

TEST(Gusts, TheEventIsOneMinusCosineAndArrivesWithItsFront)
{
    EXPECT_EQ(gust_event_shape(-0.1), 0.0);
    EXPECT_EQ(gust_event_shape(1.1), 0.0);
    EXPECT_NEAR(gust_event_shape(0.0), 0.0, 1e-15);
    EXPECT_NEAR(gust_event_shape(1.0), 0.0, 1e-15);
    EXPECT_NEAR(gust_event_shape(0.5), 1.0, 1e-15);
    EXPECT_NEAR(gust_event_shape(0.25), 0.5, 1e-15);
    EXPECT_NEAR(gust_event_shape(0.3), gust_event_shape(0.7), 1e-15);
    // a front crossing (500, 500) at t = 100 s, moving at 10 m/s towards 30 degrees, lasting 8 s at a point
    GustEvent e;
    e.time = 100.0; e.x0 = 500.0; e.y0 = 500.0; e.speed = 10.0; e.duration = 8.0;
    const double a = 30.0 * 3.14159265358979323846 / 180.0;
    e.ex = std::cos(a); e.ey = std::sin(a);
    EXPECT_NEAR(gust_event_phase(e, 500.0, 500.0, 100.0), 0.0, 1e-12) << "the front at the origin at t0";
    EXPECT_NEAR(gust_event_phase(e, 500.0, 500.0, 104.0), 0.5, 1e-12);
    // 50 m on along the direction: 5 s later
    EXPECT_NEAR(gust_event_phase(e, 500.0 + 50.0 * e.ex, 500.0 + 50.0 * e.ey, 105.0), 0.0, 1e-12);
    EXPECT_NEAR(gust_event_phase(e, 500.0 + 50.0 * e.ex, 500.0 + 50.0 * e.ey, 109.0), 0.5, 1e-12);
    // 80 m beside the origin, across the direction: with the origin
    EXPECT_NEAR(gust_event_phase(e, 500.0 - 80.0 * e.ey, 500.0 + 80.0 * e.ex, 104.0), 0.5, 1e-12);
    // 20 m before the origin: 2 s earlier
    EXPECT_NEAR(gust_event_phase(e, 500.0 - 20.0 * e.ex, 500.0 - 20.0 * e.ey, 98.0), 0.0, 1e-12);
}

TEST(Gusts, TheNormalNumbersAreStandardAndIndependent)
{
    // sampling errors over N = 200000: the mean 0.0022, the variance 0.0032, the fourth moment 0.022, a correlation 0.0022
    const int N = 200000;
    std::vector<double> x(N), y(N);
    for (int n = 0; n < N; ++n) {
        x[static_cast<std::size_t>(n)] = gust_normal(1, 0, static_cast<std::uint64_t>(n));
        y[static_cast<std::size_t>(n)] = gust_normal(1, 1, static_cast<std::uint64_t>(n));
    }
    auto mean = [&] (const std::vector<double>& v) { double s = 0.0; for (const double a : v) { s += a; } return s / N; };
    const double mx = mean(x), my = mean(y);
    double vx = 0.0, vy = 0.0, m4 = 0.0, lag = 0.0, cross = 0.0, seeds = 0.0;
    for (std::size_t n = 0; n < x.size(); ++n) {
        vx += (x[n] - mx) * (x[n] - mx);
        vy += (y[n] - my) * (y[n] - my);
        m4 += x[n] * x[n] * x[n] * x[n];
        cross += (x[n] - mx) * (y[n] - my);
        if (n + 1 < x.size()) { lag += (x[n] - mx) * (x[n+1] - mx); }
        seeds += x[n] * gust_normal(2, 0, n);
    }
    vx /= N; vy /= N; m4 /= N;
    EXPECT_NEAR(mx, 0.0, 0.01);
    EXPECT_NEAR(vx, 1.0, 0.015);
    EXPECT_NEAR(m4, 3.0, 0.1) << "Gaussian tails";
    EXPECT_NEAR(lag / (N * vx), 0.0, 0.01) << "neighbouring steps";
    EXPECT_NEAR(cross / (N * std::sqrt(vx * vy)), 0.0, 0.01) << "two streams";
    EXPECT_NEAR(seeds / N, 0.0, 0.01) << "two seeds";
    EXPECT_EQ(gust_normal(1, 0, 5), gust_normal(1, 0, 5));
    EXPECT_NE(gust_normal(1, 0, 5), gust_normal(2, 0, 5));
}

TEST(Gusts, TheOrnsteinUhlenbeckStepIsExact)
{
    EXPECT_NEAR(gust_ou_step(0.7, 0.5, 5.0, -1.3), 0.079901750840029, 1e-14);
    EXPECT_EQ(gust_ou_step(0.7, 0.0, 5.0, -1.3), 0.7) << "no time, no change";
    EXPECT_EQ(gust_ou_step(0.7, 0.5, std::numeric_limits<double>::infinity(), -1.3), 0.7) << "an infinite time scale holds z";
    EXPECT_NEAR(gust_ou_step(0.7, 500.0, 5.0, -1.3), -1.3, 1e-14) << "dt >> T forgets z";
}

TEST(Gusts, TheRandomGustHasUnitVarianceAndExponentialCorrelation)
{
    // about 10000 independent samples (N dt / 2T): sampling errors of about 0.015 in the variance and the correlations
    const int N = 200000;
    const double dt = 1.0, T = 10.0;
    std::vector<double> z(N);
    z[0] = gust_normal(7, 3, 0);
    for (int n = 1; n < N; ++n) {
        const auto i = static_cast<std::size_t>(n);
        z[i] = gust_ou_step(z[i-1], dt, T, gust_normal(7, 3, static_cast<std::uint64_t>(n)));
    }
    double m = 0.0;
    for (const double a : z) { m += a; }
    m /= N;
    double v = 0.0;
    for (const double a : z) { v += (a - m) * (a - m); }
    v /= N;
    EXPECT_NEAR(m, 0.0, 0.05);
    EXPECT_NEAR(v, 1.0, 0.05);
    for (const int lag : {5, 10, 20}) {
        double c = 0.0;
        const auto L = static_cast<std::size_t>(lag);
        for (std::size_t n = 0; n + L < z.size(); ++n) { c += (z[n] - m) * (z[n + L] - m); }
        EXPECT_NEAR(c / (N * v), std::exp(-lag * dt / T), 0.04) << "lag " << lag << " s";
    }
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
    EXPECT_NE(abort_message([] { tower_background_factor(0.0, 67.0); }).find("the height must be positive"), std::string::npos);
    EXPECT_NE(abort_message([] { tower_background_factor(30.0, 0.0); }).find("the length scale must be positive"), std::string::npos);
    EXPECT_NE(abort_message([] { gust_integral_length(0.0); }).find("the height must be positive"), std::string::npos);
    EXPECT_NE(abort_message([] { GustEvent e; e.speed = 0.0; gust_event_phase(e, 0.0, 0.0, 0.0); })
                  .find("the speed and duration must be positive"), std::string::npos);
    EXPECT_NE(abort_message([] { gust_ou_step(0.0, -1.0, 5.0, 0.0); }).find("dt must be finite"), std::string::npos);
    EXPECT_NE(abort_message([] { gust_ou_step(0.0, 1.0, 0.0, 0.0); }).find("the time scale must be positive"), std::string::npos);
    // a normal wind above the speed by roundoff only is accepted
    EXPECT_TRUE(abort_message(gust(5.0, 5.0 * (1.0 + 1.0e-12), 1.0, 100.0, 67.0)).empty());
}
