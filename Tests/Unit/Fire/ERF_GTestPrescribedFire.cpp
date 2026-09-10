#include <gtest/gtest.h>
#include <cmath>
#include <AMReX_REAL.H>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

#include "ERF_PrescribedFire.H"

/**
 * @file ERF_GTestPrescribedFire.cpp
 * @brief The prescribed rate of spread (constant, linear gradient, floor) and
 *        the prescribed heat-flux disc and its time window.
 */

TEST(PrescribedFire, ConstantRate)
{
    for (const amrex::Real x : {amrex::Real(-50.0), amrex::Real(0.0), amrex::Real(125.0)}) {
        EXPECT_NEAR(prescribed_ros_at(0.75, 0.0, 0.0, 10.0, 20.0, 0.0, x, 3.0 * x), 0.75, TOL);
    }
}

TEST(PrescribedFire, LinearGradientAndFloor)
{
    // R = 1 + 0.004 (x - 200) - 0.002 (y - 100)
    const amrex::Real R = prescribed_ros_at(1.0, 0.004, -0.002, 200.0, 100.0, 0.0, 300.0, 150.0);
    EXPECT_NEAR(R, 1.0 + 0.4 - 0.1, 1e3 * TOL);
    // far down the gradient the rate is floored, and never negative
    EXPECT_NEAR(prescribed_ros_at(1.0, 0.004, 0.0, 200.0, 0.0, 0.05, -300.0, 0.0), 0.05, TOL);
    EXPECT_NEAR(prescribed_ros_at(1.0, 0.004, 0.0, 200.0, 0.0, 0.0, -300.0, 0.0), 0.0, TOL);
    EXPECT_NEAR(prescribed_ros_at(1.0, 0.004, 0.0, 200.0, 0.0, -1.0, -300.0, 0.0), 0.0, TOL);
}

TEST(PrescribedFire, HeatFluxDiscAndWindow)
{
    const amrex::Real q = 5.0e4, cx = 800.0, cy = 700.0, r = 100.0;
    // inside the disc, on its edge, outside
    EXPECT_NEAR(prescribed_heat_flux_at(q, cx, cy, r, 0.0, -1.0, 850.0, 700.0, 10.0), q, TOL * q);
    EXPECT_NEAR(prescribed_heat_flux_at(q, cx, cy, r, 0.0, -1.0, 900.0, 700.0, 10.0), q, TOL * q);
    EXPECT_NEAR(prescribed_heat_flux_at(q, cx, cy, r, 0.0, -1.0, 880.0, 780.0, 10.0), 0.0, TOL);
    // before start, at and after end, and never switched off
    EXPECT_NEAR(prescribed_heat_flux_at(q, cx, cy, r, 60.0, 120.0, cx, cy, 59.0), 0.0, TOL);
    EXPECT_NEAR(prescribed_heat_flux_at(q, cx, cy, r, 60.0, 120.0, cx, cy, 60.0), q, TOL * q);
    EXPECT_NEAR(prescribed_heat_flux_at(q, cx, cy, r, 60.0, 120.0, cx, cy, 120.0), 0.0, TOL);
    EXPECT_NEAR(prescribed_heat_flux_at(q, cx, cy, r, 60.0, -1.0, cx, cy, 1.0e6), q, TOL * q);
    // off when the flux is not positive
    EXPECT_NEAR(prescribed_heat_flux_at(0.0, cx, cy, r, 0.0, -1.0, cx, cy, 10.0), 0.0, TOL);
}
