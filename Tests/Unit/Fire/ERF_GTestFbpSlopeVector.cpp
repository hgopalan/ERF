#include <gtest/gtest.h>
#include <cmath>
#include <AMReX_REAL.H>

#include "ERF_FbpModel.H"

/**
 * @file ERF_GTestFbpSlopeVector.cpp
 * @brief The isotropic FBP rate combines the wind and the slope-equivalent
 *        wind as the system's vector sum (Forestry Canada 1992 eqs. 45-47;
 *        Wotton et al. 2009 eqs. 44-48): the equivalent wind of the slope
 *        points upslope, so a slope across the wind raises the rate and turns
 *        the head, where the earlier form took only the slope component along
 *        the wind (none, across it). M-2 weights the D-1 share of its
 *        slope-equivalent wind by 0.2, as its rate does.
 */

namespace {

constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-10 : 1e-4;

FbpComputed make (int type, double ffmc, double bui, double pc = 50.0)
{
    FbpComputed s;
    s.type = type; s.fF = fbp_ffmc_function(ffmc); s.be = fbp_buildup_effect(type, bui);
    s.cf = 1.0; s.pc = pc; s.pdf = 50.0; s.use_slope = true;
    return s;
}

} // namespace

TEST(FbpSlopeVector, ASlopeAcrossTheWindRaisesTheRateAndTurnsTheHead)
{
    const FbpComputed st = make(FBP_C2, 90, 60);
    const double u = 5.0, slope = 0.3;
    amrex::Real hx = 0, hy = 0;
    const double flat = fbp_ros(st, u, 0.0);
    const double r = fbp_ros_vector(st, u, 0.0, 0.0, slope, hx, hy);
    EXPECT_GT(r, flat) << "the slope-equivalent wind adds in quadrature";
    // the system's vector: W = (3.6 u, WSE), the rate at |W| and the head along it
    const double wse = fbp_slope_equivalent_wind(st, slope);
    ASSERT_GT(wse, 0.0);
    const double W = std::sqrt(3.6 * u * 3.6 * u + wse * wse);
    EXPECT_NEAR(r, fbp_rsi(st, fbp_isi(st.fF, W)) * st.be / 60.0, 1.0e3 * TOL);
    EXPECT_NEAR(hx, 3.6 * u / W, 1.0e3 * TOL);
    EXPECT_NEAR(hy, wse / W, 1.0e3 * TOL);
}

TEST(FbpSlopeVector, AlongTheWindItIsTheScalarForm)
{
    const FbpComputed st = make(FBP_C2, 90, 60);
    amrex::Real hx = 0, hy = 0;
    EXPECT_NEAR(fbp_ros_vector(st, 5.0, 0.0, 0.3, 0.0, hx, hy), fbp_ros(st, 5.0, 0.3), 1.0e3 * TOL);
    EXPECT_NEAR(hx, 1.0, TOL); EXPECT_NEAR(hy, 0.0, TOL);
    EXPECT_NEAR(fbp_ros_vector(st, 0.0, 0.0, 0.3, 0.0, hx, hy), fbp_ros(st, 0.0, 0.3), 1.0e3 * TOL) << "slope alone";
    EXPECT_NEAR(hx, 1.0, TOL);
    EXPECT_NEAR(fbp_ros_vector(st, 0.0, 0.0, 0.0, 0.0, hx, hy), fbp_ros(st, 0.0, 0.0), 1.0e3 * TOL) << "calm and flat";
    EXPECT_NEAR(hx, 1.0, TOL); EXPECT_NEAR(hy, 0.0, TOL);
    // a wind blowing downslope is opposed by the upslope equivalent wind
    EXPECT_LT(fbp_ros_vector(st, 5.0, 0.0, -0.3, 0.0, hx, hy), fbp_ros(st, 5.0, 0.0));
    FbpComputed off = st; off.use_slope = false;
    EXPECT_NEAR(fbp_ros_vector(off, 5.0, 0.0, 0.0, 0.3, hx, hy), fbp_ros(off, 5.0, 0.0), 1.0e3 * TOL);
}

TEST(FbpSlopeVector, M2WeightsTheD1ShareByAFifth)
{
    const FbpComputed m1 = make(FBP_M1, 90, 60, 50.0), m2 = make(FBP_M2, 90, 60, 50.0);
    EXPECT_LT(fbp_slope_equivalent_wind(m2, 0.3), fbp_slope_equivalent_wind(m1, 0.3));
    // pure conifer: no D-1 share, the two mixedwood types agree
    const FbpComputed m1c = make(FBP_M1, 90, 60, 100.0), m2c = make(FBP_M2, 90, 60, 100.0);
    EXPECT_NEAR(fbp_slope_equivalent_wind(m2c, 0.3), fbp_slope_equivalent_wind(m1c, 0.3), 1.0e3 * TOL);
}
