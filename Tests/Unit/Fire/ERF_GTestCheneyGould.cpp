#include <gtest/gtest.h>
#include <cfenv>
#include <cmath>
#include <AMReX_REAL.H>

#include "ERF_FireParams.H"   // ERF_CheneyGouldModel.H reads FireParams without including it
#include "ERF_CheneyGouldModel.H"
#include "ERF_MacArthurModel.H"

/**
 * @file ERF_GTestCheneyGould.cpp
 * @brief Cheney, Gould and Catchpole (1998), Int. J. Wildland Fire 8(1):1-13,
 *        as implemented for ros_model = cheney_gould: the two wind branches
 *        joined at 5 km/h, the moisture and curing coefficients, natural and
 *        grazed pasture, and the pre-2026-10 fit kept as grass_simple. Every
 *        expectation is the paper's equation written out here, in km/h and
 *        percent, so a transcription slip in the kernel is caught.
 */

using namespace amrex;

namespace {

constexpr double TOL = (sizeof(Real) == 8) ? 1.0e-10 : 1.0e-4;

/// Head rate [km/h] of the paper: eqs. 2-3 (natural), 4-5 (grazed), times phi_M and phi_C
double cg98_kmh (double U_kmh, double M_pct, double C_pct, bool grazed, bool cheney_curve)
{
    const double R = (U_kmh < 5.0)
        ? (grazed ? 0.054 + 0.209 * U_kmh : 0.054 + 0.269 * U_kmh)
        : (grazed ? 1.1 + 0.715 * std::pow(U_kmh - 5.0, 0.844) : 1.4 + 0.838 * std::pow(U_kmh - 5.0, 0.844));
    double phi_M;
    if (M_pct < 12.0)        { phi_M = std::exp(-0.108 * M_pct); }
    else if (U_kmh < 10.0)   { phi_M = std::max(0.684 - 0.0342 * M_pct, 0.0); }
    else                     { phi_M = std::max(0.547 - 0.0228 * M_pct, 0.0); }
    const double phi_C = cheney_curve
        ? 1.12 / (1.0 + 59.2 * std::exp(-0.124 * (C_pct - 50.0)))
        : 1.036 / (1.0 + 103.989 * std::exp(-0.0996 * (C_pct - 20.0)));
    return R * phi_M * phi_C;
}

CheneyGouldComputed cg_state (double M, double curing, int model, int pasture = cheney_gould::pasture_natural,
                              int curve = cheney_gould::curing_cruz2015)
{
    FireParams::CheneyGouldParams p;
    p.moisture = M;
    p.curing   = curing;
    return compute_cheney_gould_params(p, model, pasture, curve);
}

} // namespace

TEST(CheneyGould98, HeadRateAtTwentyKilometresPerHour)
{
    // natural pasture, 6 % moisture, fully cured, 20 km/h: 1.4 + 0.838 15^0.844
    // km/h times e^{-0.648}, about 5.04 km/h = 1.40 m/s (the paper's Fig. 7 range)
    const CheneyGouldComputed cg = cg_state(6.0, 1.0, cheney_gould::model_cg98);
    const double U = 20.0 / 3.6;
    const double r = cheney_gould_ros(U, cg);
    EXPECT_NEAR(r, cg98_kmh(20.0, 6.0, 100.0, false, false) / 3.6, 1.0e3 * TOL);
    EXPECT_NEAR(r, 1.4006, 0.002);
    // grazed pasture is slower at the same wind
    const CheneyGouldComputed gz = cg_state(6.0, 1.0, cheney_gould::model_cg98, cheney_gould::pasture_grazed);
    EXPECT_NEAR(cheney_gould_ros(U, gz), cg98_kmh(20.0, 6.0, 100.0, true, false) / 3.6, 1.0e3 * TOL);
    EXPECT_LT(cheney_gould_ros(U, gz), r);
}

TEST(CheneyGould98, TheTwoWindBranchesJoinAtFiveKilometresPerHour)
{
    const CheneyGouldComputed cg = cg_state(8.0, 1.0, cheney_gould::model_cg98);
    const double below = cheney_gould_ros((5.0 - 1.0e-6) / 3.6, cg);
    const double above = cheney_gould_ros((5.0 + 1.0e-6) / 3.6, cg);
    // the published fits meet to 0.001 km/h (1.399 and 1.400 at 5 km/h)
    EXPECT_NEAR(above / below, 1.4 / 1.399, 1.0e-4);
    // low wind: the linear branch, with the 0.054 km/h calm rate
    EXPECT_NEAR(cheney_gould_ros(1.0 / 3.6, cg), cg98_kmh(1.0, 8.0, 100.0, false, false) / 3.6, 1.0e3 * TOL);
    EXPECT_NEAR(cheney_gould_ros(0.0, cg), 0.054 * std::exp(-0.108 * 8.0) * cg.phi_C / 3.6, 1.0e3 * TOL);
    // a backing (negative) wind is treated as calm
    EXPECT_NEAR(cheney_gould_ros(-3.0, cg), cheney_gould_ros(0.0, cg), TOL);
}

TEST(CheneyGould98, MoistureCoefficientIsPiecewise)
{
    EXPECT_NEAR(cheney_gould_phi_M(6.0, 20.0), std::exp(-0.648), 1.0e3 * TOL);
    EXPECT_NEAR(cheney_gould_phi_M(12.0, 5.0),  0.684 - 0.0342 * 12.0, 1.0e3 * TOL) << "12-20 %, below 10 km/h";
    EXPECT_NEAR(cheney_gould_phi_M(12.0, 15.0), 0.547 - 0.0228 * 12.0, 1.0e3 * TOL) << "12-20 %, 10 km/h and above";
    EXPECT_NEAR(cheney_gould_phi_M(20.0, 15.0), 0.547 - 0.0228 * 20.0, 1.0e3 * TOL);
    EXPECT_NEAR(cheney_gould_phi_M(25.0, 15.0), 0.0, TOL) << "beyond the fit, no spread rather than a negative rate";
    // the model's rate at 20 % moisture and 20 km/h: the paper's Table 2 region
    const CheneyGouldComputed cg = cg_state(20.0, 1.0, cheney_gould::model_cg98);
    EXPECT_NEAR(cheney_gould_ros(20.0 / 3.6, cg), cg98_kmh(20.0, 20.0, 100.0, false, false) / 3.6, 1.0e3 * TOL);
}

TEST(CheneyGould98, CuringCurves)
{
    EXPECT_NEAR(cheney_gould_phi_C(100.0, cheney_gould::curing_cruz2015),
                1.036 / (1.0 + 103.989 * std::exp(-0.0996 * 80.0)), 1.0e3 * TOL);
    EXPECT_NEAR(cheney_gould_phi_C(50.0, cheney_gould::curing_cruz2015),
                1.036 / (1.0 + 103.989 * std::exp(-0.0996 * 30.0)), 1.0e3 * TOL);
    EXPECT_NEAR(cheney_gould_phi_C(100.0, cheney_gould::curing_cheney1998),
                1.12 / (1.0 + 59.2 * std::exp(-0.124 * 50.0)), 1.0e3 * TOL);
    EXPECT_NEAR(cheney_gould_phi_C(50.0, cheney_gould::curing_cheney1998), 1.12 / 60.2, 1.0e3 * TOL);
    // Cruz (2015) keeps a half-cured sward spreading, Cheney (1998) nearly stops it
    EXPECT_GT(cheney_gould_phi_C(50.0, cheney_gould::curing_cruz2015), 5.0 * cheney_gould_phi_C(50.0, cheney_gould::curing_cheney1998));
    // the state carries the curve's coefficient of the deck's curing
    const CheneyGouldComputed a = cg_state(8.0, 0.5, cheney_gould::model_cg98, cheney_gould::pasture_natural, cheney_gould::curing_cheney1998);
    EXPECT_NEAR(a.phi_C, 1.12 / 60.2, 1.0e3 * TOL);
    EXPECT_NEAR(cheney_gould_ros(3.0, a), cg98_kmh(10.8, 8.0, 50.0, false, true) / 3.6, 1.0e3 * TOL);
}

TEST(CheneyGould98, GrassSimpleIsADifferentModel)
{
    // the pre-2026-10 fit: backing 0.08 c e^{-0.01 M}, forward = backing (1 +
    // 0.15 U (c + 0.2)) 20 / (M + 1), in m/s on the midflame wind
    const CheneyGouldComputed gs = cg_state(6.0, 1.0, cheney_gould::model_grass_simple);
    const double backing = 0.08 * std::exp(-0.06);
    EXPECT_NEAR(gs.ros_backing, backing, 1.0e3 * TOL);
    EXPECT_NEAR(cheney_gould_ros(0.0, gs), backing * 20.0 / 7.0, 1.0e3 * TOL);
    EXPECT_NEAR(cheney_gould_ros(4.0, gs), backing * (1.0 + 0.15 * 4.0 * 1.2) * 20.0 / 7.0, 1.0e3 * TOL);
    EXPECT_NEAR(grass_simple_ros(4.0, backing, 6.0, 1.0), cheney_gould_ros(4.0, gs), TOL);
    // in calm it runs more than twenty times the 1998 model, at 20 km/h a fifth of it
    const CheneyGouldComputed cg = cg_state(6.0, 1.0, cheney_gould::model_cg98);
    EXPECT_GT(cheney_gould_ros(0.0, gs), 20.0 * cheney_gould_ros(0.0, cg));
    EXPECT_LT(cheney_gould_ros(20.0 / 3.6, gs), 0.35 * cheney_gould_ros(20.0 / 3.6, cg));
}

TEST(MacArthurCap, RateIsCappedAtSixMetresPerSecond)
{
    // R = 0.18 e^{0.8424 U}: 12.2 m/s at a 5 m/s wind without the cap
    EXPECT_NEAR(macarthur_ros(1.0), 0.18 * std::exp(0.8424), 1.0e3 * TOL) << "below the cap, unchanged";
    EXPECT_NEAR(macarthur_ros(5.0), 6.0, TOL);
    EXPECT_NEAR(macarthur_ros(5.0, 2.0), 2.0, TOL);
    EXPECT_NEAR(macarthur_ros(5.0, 0.0), 0.18 * std::exp(0.8424 * 5.0), 1.0e3 * TOL) << "0 removes the cap";
}

// The cap is reached at U = ln(6/0.18)/0.8424 = 4.16 m/s, so a stronger wind
// must not evaluate the exponential: e^{0.8424 U} overflowed at U = 1000 m/s
// in double (about 105 m/s in single), raising FE_OVERFLOW (an abort under the
// FPE traps) before min() took the cap (Copilot's review of hgopalan/ERF#501).
TEST(MacArthurCap, AWindBeyondTheCapDoesNotOverflow)
{
    volatile Real u = 1000.0_rt;
    std::feclearexcept(FE_OVERFLOW | FE_INVALID | FE_DIVBYZERO);
    volatile Real r = macarthur_ros(u);
    volatile Real r_neg_cap = macarthur_ros(u, -1.0_rt);   // uncapped: bounded, not infinite
    EXPECT_FALSE(std::fetestexcept(FE_OVERFLOW | FE_INVALID | FE_DIVBYZERO));
    EXPECT_EQ(r, 6.0_rt) << "the cap, exactly";
    EXPECT_TRUE(std::isfinite(r_neg_cap));
    // below the cap the rate is the formula's, unchanged
    for (Real w : {0.0_rt, 1.0_rt, 3.0_rt, 4.15_rt}) {
        const Real old_form = 0.18_rt + 0.18_rt * (std::exp(0.8424_rt * w) - 1.0_rt);
        EXPECT_NEAR(macarthur_ros(w), old_form, 10.0 * TOL * old_form) << "U = " << w;
    }
}
