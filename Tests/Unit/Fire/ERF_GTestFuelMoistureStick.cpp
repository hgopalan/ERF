#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <AMReX_REAL.H>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

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
    EXPECT_NEAR(stick_volume_average(M, 4), 0.1, TOL);
    amrex::Real N[2] = {0.0, 1.0};                  // outer shell 3/4 of the area
    EXPECT_NEAR(stick_volume_average(N, 2), 0.75, TOL);
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

TEST(FuelMoistureStick, SurfaceFollowsTheHysteresisCurves)
{
    // Dry air (RH 1 %: wetting curve E_w = 0.035 below drying curve E_d = 0.060), no rain.
    // A 1-h stick drying from 0.20 approaches E_d from above, so no shell ever drops
    // below it; one wetting from 0.02 approaches E_w from below and never passes it;
    // one between the curves stays where it is.
    const int N = 6; const amrex::Real R = 0.15, tau = 1.0, RH = 1.0, T = 20.0, dt = 0.01;
    const amrex::Real E_w = compute_emc_adsorption(RH), E_d = compute_emc_desorption(RH);
    amrex::Real dry[N], wet[N], mid[N];
    for (int i = 0; i < N; ++i) { dry[i] = 0.20; wet[i] = 0.02; mid[i] = 0.05; }
    amrex::Real dry_min = 1.0, wet_max = 0.0, avg_dry = 0.0, avg_wet = 0.0, avg_mid = 0.0;
    for (int n = 0; n < 2400; ++n) {                 // 24 h
        avg_dry = stick_advance_class(dry, N, R, tau, RH, T, 0.0, 0.35, 1.0, dt);
        avg_wet = stick_advance_class(wet, N, R, tau, RH, T, 0.0, 0.35, 1.0, dt);
        avg_mid = stick_advance_class(mid, N, R, tau, RH, T, 0.0, 0.35, 1.0, dt);
        for (int i = 0; i < N; ++i) {
            dry_min = std::min(dry_min, dry[i]);
            wet_max = std::max(wet_max, wet[i]);
        }
    }
    EXPECT_GE(dry_min, E_d - TOL);
    EXPECT_LE(wet_max, E_w + TOL);
    EXPECT_NEAR(avg_dry, E_d, 1e-4);
    EXPECT_NEAR(avg_wet, E_w, 1e-4);
    EXPECT_NEAR(avg_mid, 0.05, TOL);
}
