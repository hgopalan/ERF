#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <cmath>
#include <type_traits>

#include "ERF_DustThreshold.H"
#include "ERF_DustEmission.H"

/**
 * @file ERF_GTestDustThreshold.cpp
 * @brief The threshold friction velocity, written to fail on the code before the
 *        October 2026 validation:
 *  - moisture and suppression RAISE the threshold (the code divided by both
 *    factors, so a wet or treated surface emitted more than a dry one);
 *  - the base threshold is Shao and Lu's for the saltating grains: non-monotone
 *    in the diameter with its minimum near 75 um, and 0.2-0.6 m/s for the
 *    default inputs (Bagnold at the 7 um bin gave 0.0385 m/s);
 *  - with that base, crust LOWERS the emission (in the saturated regime the
 *    Owen factor rose with the threshold, so crust raised it by 2.8 %);
 *  - the slope factor is signed along the wind: up 1.11, across 1.00, down 0.86
 *    on a 10 degree slope (it was 1.11 on every face).
 */

using namespace amrex;

constexpr double REAL_RTOL = std::is_same<amrex::Real, float>::value ? 1.0e-5 : 1.0e-9;

namespace {
constexpr Real RHO_P = 2650.0, RHO_A = 1.225;
constexpr Real A_N = DustThresholdConst::SHAO_LU_A_N;
constexpr Real GAMMA = DustThresholdConst::SHAO_LU_GAMMA;
}

TEST(DustThreshold, MoistureAndSuppressionRaiseTheThreshold)
{
    const Real base = 0.25;
    const Real dry  = compute_ustar_t_full(base, 0.0, 0.0, 0.5, 0.3, 0.0, 0.0);
    const Real wet  = compute_ustar_t_full(base, 0.0, 0.0, 0.5, 0.3, 1.0, 0.0);
    const Real supp = compute_ustar_t_full(base, 0.0, 0.0, 0.5, 0.3, 0.0, 1.0);
    EXPECT_NEAR(dry, base, REAL_RTOL * base);
    // f_moist = 1 + 4 w, f_supp = 1 + 6 s (the mutant returned base/5 and base/7)
    EXPECT_NEAR(wet,  base * 5.0, REAL_RTOL * base * 5.0);
    EXPECT_NEAR(supp, base * 7.0, REAL_RTOL * base * 7.0) << "a treated surface is harder to erode";
    EXPECT_GT(wet, dry);
    EXPECT_GT(supp, dry);
}

TEST(DustThreshold, ShaoLuBaseHasACohesionMinimumNearSeventyFiveMicrons)
{
    // quartz in standard air: 0.204 m/s at 75 um, 0.488 m/s at 7 um
    const Real u75 = compute_ustar_t_shao_lu(A_N, GAMMA, RHO_P, 75.0e-6, RHO_A);
    const Real u7  = compute_ustar_t_shao_lu(A_N, GAMMA, RHO_P,  7.0e-6, RHO_A);
    EXPECT_NEAR(u75, 0.2041, 2.0e-3);
    EXPECT_NEAR(u7,  0.4882, 4.0e-3);
    EXPECT_GT(u7, u75) << "the cohesion term raises the threshold of fine particles";
    // non-monotone: the minimum lies between 50 and 120 um (Bagnold is monotone)
    Real d_min = 0.0, u_min = 1.0e30;
    for (int n = 1; n <= 400; ++n) {
        const Real d = n * 1.0e-6;
        const Real u = compute_ustar_t_shao_lu(A_N, GAMMA, RHO_P, d, RHO_A);
        if (u < u_min) { u_min = u; d_min = d; }
    }
    EXPECT_GT(d_min, 50.0e-6);
    EXPECT_LT(d_min, 120.0e-6);
    // the default inputs give a base in the physical range
    EXPECT_GT(u75, Real(0.15));
    EXPECT_LT(u75, Real(0.6));
    // Bagnold at the old particle, the saturated value
    EXPECT_NEAR(compute_ustar_t_bagnold(0.1, RHO_P, 7.0e-6, RHO_A), 0.0385, 5.0e-4);
}

TEST(DustThreshold, CrustLowersTheEmissionAtTheDefaultThreshold)
{
    // u* = 0.5 m/s, the canonical ABL; base from the default model
    const Real base = compute_ustar_t_shao_lu(A_N, GAMMA, RHO_P, 75.0e-6, RHO_A);
    const Real ut_bare  = compute_ustar_t_full(base, 0.0, 0.0, 0.5, 0.3, 0.0, 0.0);
    const Real ut_crust = compute_ustar_t_full(base, 1.0, 0.0, 0.5, 0.3, 0.0, 0.0);
    const Real q_bare   = compute_saltation_flux(Real(0.5), ut_bare,  RHO_A);
    const Real q_crust  = compute_saltation_flux(Real(0.5), ut_crust, RHO_A);
    ASSERT_GT(q_bare, Real(0.0));
    EXPECT_LT(q_crust, q_bare) << "a crusted surface emits less (the 0.0385 m/s base gave +2.8 %)";
    EXPECT_LT(q_crust / q_bare, Real(0.9));
}

TEST(DustThreshold, SlopeFactorIsSignedAlongTheWind)
{
    // 10 degree slope rising in +x: tan(10 deg) = 0.1763
    const Real s = std::tan(10.0 * (4.0 * std::atan(1.0)) / 180.0);
    const Real up     = compute_slope_factor(s, 0.0,  1.0, 0.0);
    const Real across = compute_slope_factor(s, 0.0,  0.0, 1.0);
    const Real down   = compute_slope_factor(s, 0.0, -1.0, 0.0);
    const Real calm   = compute_slope_factor(s, 0.0,  0.0, 0.0);
    // Iversen & Rasmussen (1994): sqrt(cos 10 + sin 10 / tan 35) = 1.110,
    // sqrt(cos 10 - sin 10 / tan 35) = 0.858
    EXPECT_NEAR(up,     1.110, 2.0e-3);
    EXPECT_NEAR(across, 1.000, 1.0e-6);
    EXPECT_NEAR(down,   0.858, 2.0e-3) << "the lee face has a lower threshold (it was 1.11 on every face)";
    EXPECT_NEAR(calm,   up,    1.0e-6) << "no wind: isotropic in |grad z|";
    EXPECT_NEAR(compute_slope_factor(0.0, 0.0, 1.0, 0.0), 1.0, 1.0e-12);
}
