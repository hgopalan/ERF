#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "ERF_FuelMoistureStick.H"

/**
 * @file ERF_GTestFuelMoistureStick.cpp
 * @brief The stick moisture model: relaxation to the surface value, the
 *        lag-calibrated long-time response, the ordering of surface and core,
 *        and the rain surface condition.
 */

TEST(FuelMoistureStick, VolumeAverage)
{
    amrex::Real M[4] = {0.1, 0.1, 0.1, 0.1};
    EXPECT_NEAR(stick_volume_average(M, 4), 0.1, 1e-14);
    amrex::Real N[2] = {0.0, 1.0};                  // outer shell 3/4 of the area
    EXPECT_NEAR(stick_volume_average(N, 2), 0.75, 1e-14);
}

TEST(FuelMoistureStick, RelaxesToSurfaceWithTheClassLag)
{
    // A 10-h stick of radius 0.635 cm, surface held at 0.20 from a dry 0.05:
    // after the fast modes die the volume average decays with the 10 h lag.
    const int N = 8; const amrex::Real R = 0.635, tau = 10.0, D = stick_diffusivity(R, tau);
    amrex::Real M[N]; for (auto& m : M) m = 0.05;
    const amrex::Real dt = 0.05;                     // 3 min
    std::vector<amrex::Real> avg;
    amrex::Real surf_1h = 0.0, core_1h = 0.0;
    for (int n = 0; n < 2000; ++n) {                 // 100 h
        stick_advance(M, N, R, D, 0.20, dt);
        avg.push_back(stick_volume_average(M, N));
        if (n == 19) { surf_1h = M[N - 1]; core_1h = M[0]; }   // after one hour
    }
    // approach from below, surface ahead of the core while the profile is still developing
    EXPECT_GT(avg[10], 0.05); EXPECT_LT(avg[10], 0.20);
    EXPECT_GT(surf_1h, core_1h);
    EXPECT_NEAR(avg.back(), 0.20, 1e-4);            // 0.15 exp(-100/10) = 7e-6 is still left
    // e-folding of the remaining deficit between 20 h and 40 h (first mode only left)
    const amrex::Real d1 = 0.20 - avg[400 - 1], d2 = 0.20 - avg[800 - 1];
    const amrex::Real tau_fit = 20.0 / std::log(d1 / d2);
    EXPECT_NEAR(tau_fit, tau, 0.1 * tau);            // to 10 %: the discrete radial mode vs lambda_1
}

TEST(FuelMoistureStick, FastSurfaceSlowCore)
{
    // After one hour of a 100-h stick the surface shell has moved, the core has not.
    const int N = 8; const amrex::Real R = 2.5, D = stick_diffusivity(R, 100.0);
    amrex::Real M[N]; for (auto& m : M) m = 0.10;
    for (int n = 0; n < 20; ++n) { stick_advance(M, N, R, D, 0.30, 0.05); }
    EXPECT_GT(M[N - 1] - 0.10, 0.02);
    EXPECT_LT(M[0] - 0.10, 1e-4);
}

TEST(FuelMoistureStick, RainWetsTheSurface)
{
    const int N = 6; amrex::Real M[N]; for (auto& m : M) m = 0.08;
    // dry air (RH 30 %) but raining: the surface is held at the rain value, the average rises
    const amrex::Real avg_rain = stick_advance_class(M, N, 0.635, 10.0, 30.0, 20.0, 2.0, 0.35, 1.0, 0.5);
    EXPECT_GT(avg_rain, 0.08);
    amrex::Real M2[N]; for (auto& m : M2) m = 0.08;
    const amrex::Real avg_dry = stick_advance_class(M2, N, 0.635, 10.0, 30.0, 20.0, 0.0, 0.35, 1.0, 0.5);
    EXPECT_LT(avg_dry, avg_rain);
}
