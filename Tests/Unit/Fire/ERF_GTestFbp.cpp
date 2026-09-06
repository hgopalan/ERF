#include <gtest/gtest.h>
#include <cmath>
#include <AMReX_REAL.H>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

#include "ERF_FbpModel.H"

/**
 * @file ERF_GTestFbp.cpp
 * @brief Invariants of the Canadian FBP rate of spread: the buildup effect,
 *        the curing factor, the mixedwood limits, the wind response and the
 *        slope-equivalent wind.
 */

static FbpComputed make(int type, double ffmc, double bui, double curing = 60.0, double pc = 50.0, double pdf = 50.0)
{
    FbpComputed s;
    s.type = type; s.fF = fbp_ffmc_function(ffmc); s.be = fbp_buildup_effect(type, bui);
    s.cf = (type == FBP_O1A || type == FBP_O1B) ? fbp_curing_factor(curing) : 1.0;
    s.pc = pc; s.pdf = pdf; s.use_slope = true;
    return s;
}

TEST(Fbp, BuildupEffectAndCuring)
{
    EXPECT_NEAR(fbp_buildup_effect(FBP_C2, 64.0), 1.0, TOL);      // BUI = BUI0
    EXPECT_LT(fbp_buildup_effect(FBP_C2, 30.0), 1.0);                // drier than BUI0: less
    EXPECT_GT(fbp_buildup_effect(FBP_C2, 120.0), 1.0);
    EXPECT_NEAR(fbp_buildup_effect(FBP_O1B, 10.0), 1.0, TOL);     // grass: none
    EXPECT_NEAR(fbp_curing_factor(100.0), 1.0, TOL);               // fully cured
    // The two published branches, each side of the 58.8 % join (they differ by
    // 4e-4 there, and in single precision 58.8 itself lands on the exponential side)
    EXPECT_NEAR(fbp_curing_factor(58.81), 0.176 + 0.02 * 0.01, TOL);
    EXPECT_NEAR(fbp_curing_factor(58.7), 0.005 * (std::exp(0.061 * 58.7) - 1.0), TOL);
    EXPECT_NEAR(fbp_curing_factor(0.0), 0.0, TOL);
}

TEST(Fbp, MixedwoodLimitsAndWind)
{
    const FbpComputed c2 = make(FBP_C2, 90, 60), d1 = make(FBP_D1, 90, 60);
    const FbpComputed m1_100 = make(FBP_M1, 90, 60, 60, 100.0), m1_0 = make(FBP_M1, 90, 60, 60, 0.0);
    // the buildup effects differ (q, BUI0), so compare the surface rates before BE
    const double isi = fbp_isi(c2.fF, 20.0);
    EXPECT_NEAR(fbp_rsi(m1_100, isi), fbp_rsi(c2, isi), TOL);
    EXPECT_NEAR(fbp_rsi(m1_0, isi), fbp_rsi(d1, isi), TOL);
    // wind speeds the fire; the FFMC function is the published value at FFMC 90 (about 20.6)
    EXPECT_LT(fbp_ros(c2, 0.0, 0.0), fbp_ros(c2, 20.0 / 3.6, 0.0));
    EXPECT_NEAR(c2.fF, 20.6, 0.2);
    // the published order of magnitude: C-2 at FFMC 90, BUI 60, 20 km/h is about 16 m/min
    EXPECT_NEAR(fbp_ros(c2, 20.0 / 3.6, 0.0) * 60.0, 16.2, 0.5);
}

TEST(Fbp, SlopeEquivalentWind)
{
    for (int type : {FBP_C2, FBP_D1, FBP_M1, FBP_S2, FBP_O1B}) {
        const FbpComputed s = make(type, 90, 60, 80.0, 60.0);
        // zero wind on a 30 % slope equals the flat zero-wind rate times SF
        const double slope = 0.30, sf = std::exp(3.533 * std::pow(0.30, 1.2));
        const double flat = fbp_ros(s, 0.0, 0.0), sloped = fbp_ros(s, 0.0, slope);
        // Exact for the basic types and grass; the mixedwood types weight the
        // C-2 and D-1 inverse curves (the system's own prescription), which
        // does not invert the weighted rate exactly.
        const double tol = (type == FBP_M1) ? 0.15 * sf : (TOL * 1e6);
        EXPECT_NEAR(sloped / flat, sf, tol) << "type " << type;
        EXPECT_NEAR(fbp_ros(s, 0.0, -0.3), flat, TOL);            // downslope: no effect
        EXPECT_GE(fbp_slope_equivalent_wind(s, 0.3), 0.0);
    }
    FbpComputed off = make(FBP_C2, 90, 60); off.use_slope = false;
    EXPECT_NEAR(fbp_ros(off, 0.0, 0.5), fbp_ros(off, 0.0, 0.0), TOL);
}

TEST(Fbp, Names)
{
    EXPECT_EQ(fbp_type_from_name("c2"), FBP_C2);
    EXPECT_EQ(fbp_type_from_name("O1b"), FBP_O1B);
    EXPECT_EQ(fbp_type_from_name("Z9"), -1);
}
