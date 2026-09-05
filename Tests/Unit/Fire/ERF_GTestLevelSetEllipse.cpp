#include <gtest/gtest.h>
#include <cmath>

#include "ERF_LevelSetEllipse.H"

/**
 * @file ERF_GTestLevelSetEllipse.cpp
 * @brief The spread ellipse of the level-set path: rates at the head, flanks
 *        and back, the length-to-width and head-to-back ratios, the calm-wind
 *        circle and the cap on the ratio.
 */

TEST(LevelSetEllipse, HeadFlankBackRates)
{
    const amrex::Real R = 0.5;          // head rate [m/s]
    const amrex::Real U = 3.0;          // midflame wind [m/s], 6.71 mph
    const SpreadEllipse e = spread_ellipse(R, U, 0.0, 8.0);
    const amrex::Real LB = anderson_LW_ratio(U * 2.23694);
    const amrex::Real s  = std::sqrt(LB * LB - 1.0);
    const amrex::Real HB = (LB + s) / (LB - s);
    EXPECT_NEAR(e.LB, LB, 1e-12);
    EXPECT_NEAR(e.HB, HB, 1e-9);
    // head, flank and back from the support function
    EXPECT_NEAR(ellipse_normal_speed(e, 1.0, 0.0), R, 1e-12);
    EXPECT_NEAR(ellipse_normal_speed(e, 0.0, 1.0), e.a, 1e-12);
    EXPECT_NEAR(ellipse_normal_speed(e, -1.0, 0.0), R / HB, 1e-12);
    // length over width of the envelope after unit time is LB
    const amrex::Real length = ellipse_normal_speed(e, 1.0, 0.0) + ellipse_normal_speed(e, -1.0, 0.0);
    const amrex::Real width  = 2.0 * ellipse_normal_speed(e, 0.0, 1.0);
    EXPECT_NEAR(length / width, LB, 1e-9);
    // the normal speed never goes negative around the ellipse
    for (int n = 0; n <= 360; n += 5) {
        const amrex::Real th = n * M_PI / 180.0;
        EXPECT_GE(ellipse_normal_speed(e, std::cos(th), std::sin(th)), 0.0);
    }
}

TEST(LevelSetEllipse, CalmWindIsCircle)
{
    const SpreadEllipse e = spread_ellipse(0.5, 0.0, 0.0, 8.0);
    EXPECT_NEAR(e.LB, 1.0, 1e-12);
    EXPECT_NEAR(e.a, 0.5, 1e-12);
    EXPECT_NEAR(e.c, 0.0, 1e-12);
    for (int n = 0; n < 360; n += 30) {
        const amrex::Real th = n * M_PI / 180.0;
        EXPECT_NEAR(ellipse_normal_speed(e, std::cos(th), std::sin(th)), 0.5, 1e-12);
    }
}

TEST(LevelSetEllipse, FixedRatioAndCap)
{
    const SpreadEllipse f = spread_ellipse(1.0, 10.0, 3.0, 8.0);   // fixed LB = 3 regardless of the wind
    EXPECT_NEAR(f.LB, 3.0, 1e-12);
    const SpreadEllipse c = spread_ellipse(1.0, 30.0, 0.0, 4.0);   // Anderson would exceed 4 at 67 mph
    EXPECT_NEAR(c.LB, 4.0, 1e-12);
    const SpreadEllipse d = spread_ellipse(1.0, 30.0, 0.0, 8.0);   // Anderson's own cap
    EXPECT_NEAR(d.LB, 8.0, 1e-12);
}
