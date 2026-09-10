#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <AMReX_REAL.H>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

#include "ERF_FuelMoisture.H"
#include "ERF_FuelMoistureStick.H"

/**
 * @file ERF_GTestFuelMoistureEMC.cpp
 * @brief The equilibrium moisture curves chosen by erf.fire.emc_model: the
 *        legacy quartics stay the default and unchanged, and the Van Wagner
 *        and Pickett (1985) pair matches the published equations, orders
 *        drying above wetting, and sends the time-lag and stick updates to
 *        its own equilibrium. The long-time checks hold whichever hysteresis
 *        branch the kernel picks, since both end on or between the curves.
 */

namespace {
// Van Wagner (1987) eqs. 2a and 2b, written independently of the kernel:
// h in percent, T in C, result in percent of dry mass.
double vw_ed_percent(double h, double T)
{
    return 0.942 * std::pow(h, 0.679) + 11.0 * std::exp((h - 100.0) / 10.0)
         + 0.18 * (21.1 - T) * (1.0 - 1.0 / std::exp(0.115 * h));
}
double vw_ew_percent(double h, double T)
{
    return 0.618 * std::pow(h, 0.753) + 10.0 * std::exp((h - 100.0) / 10.0)
         + 0.18 * (21.1 - T) * (1.0 - 1.0 / std::exp(0.115 * h));
}

/// Integrate one time-lag class for `hours` [h] in one-minute steps.
amrex::Real relax(amrex::Real M, amrex::Real RH, amrex::Real T_C, amrex::Real tau,
                  amrex::Real hours, int emc_model)
{
    const amrex::Real dt = 1.0 / 60.0;
    const int n = static_cast<int>(hours / dt + 0.5);
    for (int k = 0; k < n; ++k) {
        M = advance_fuel_moisture_one_class(M, RH, T_C, 0.0, dt, tau, emc_model);
    }
    return M;
}
}

TEST(FuelMoistureEMC, LegacyCurvesUnchanged)
{
    // The default curves, pinned at a few humidities (values of the ported quartics).
    EXPECT_NEAR(compute_emc_adsorption(0.0),  0.0351, 1e-4);   // RH clamped to 1 %
    EXPECT_NEAR(compute_emc_desorption(0.0),  0.0600, 1e-4);
    EXPECT_NEAR(compute_emc_adsorption(40.0), 0.1659, 1e-4);
    EXPECT_NEAR(compute_emc_desorption(40.0), 0.1887, 1e-4);
    EXPECT_NEAR(compute_emc_adsorption(90.0), 0.35, TOL);      // capped
    EXPECT_NEAR(compute_emc_desorption(90.0), 0.35, TOL);
}

TEST(FuelMoistureEMC, DefaultArgumentsAreLegacy)
{
    for (amrex::Real RH : {0.0, 25.0, 55.0, 85.0}) {
        for (amrex::Real M : {0.02, 0.10, 0.30}) {
            EXPECT_EQ(compute_emc_with_hysteresis(RH, M),
                      compute_emc_with_hysteresis(RH, M, 35.0, FuelMoistureEMC::LEGACY));
            EXPECT_EQ(advance_fuel_moisture_one_class(M, RH, 30.0, 0.0, 0.01, 1.0),
                      advance_fuel_moisture_one_class(M, RH, 30.0, 0.0, 0.01, 1.0, FuelMoistureEMC::LEGACY));
        }
    }
    const int N = 6;
    amrex::Real A[N], B[N];
    for (int i = 0; i < N; ++i) { A[i] = B[i] = 0.12; }
    EXPECT_EQ(stick_advance_class(A, N, 0.635, 10.0, 40.0, 25.0, 0.0, 0.35, 1.0, 0.5),
              stick_advance_class(B, N, 0.635, 10.0, 40.0, 25.0, 0.0, 0.35, 1.0, 0.5, FuelMoistureEMC::LEGACY));
}

TEST(FuelMoistureEMC, VanWagnerMatchesPublishedEquations)
{
    const double tol = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-6;
    for (double T : {0.0, 20.0, 35.0}) {
        for (double h : {0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 95.0, 100.0}) {
            EXPECT_NEAR(compute_emc_drying_van_wagner(h, T),  std::max(0.0, vw_ed_percent(h, T)) / 100.0, tol) << "h " << h << " T " << T;
            EXPECT_NEAR(compute_emc_wetting_van_wagner(h, T), std::max(0.0, vw_ew_percent(h, T)) / 100.0, tol) << "h " << h << " T " << T;
        }
    }
    // Saturated air at 21.1 C, where the temperature term vanishes:
    // E_d = (0.942 * 100^0.679 + 11) / 100, E_w = (0.618 * 100^0.753 + 10) / 100.
    EXPECT_NEAR(compute_emc_drying_van_wagner(100.0, 21.1),  0.3248, 1e-4);
    EXPECT_NEAR(compute_emc_wetting_van_wagner(100.0, 21.1), 0.2982, 1e-4);
    // RH beyond [0, 100] is held at the ends.
    EXPECT_EQ(compute_emc_drying_van_wagner(120.0, 20.0), compute_emc_drying_van_wagner(100.0, 20.0));
    EXPECT_EQ(compute_emc_wetting_van_wagner(-5.0, 20.0), compute_emc_wetting_van_wagner(0.0, 20.0));
}

TEST(FuelMoistureEMC, VanWagnerShape)
{
    for (amrex::Real T : {0.0, 20.0, 35.0}) {
        amrex::Real prev_d = -1.0, prev_w = -1.0;
        for (int h = 0; h <= 100; ++h) {
            const amrex::Real Ed = compute_emc_drying_van_wagner(h, T);
            const amrex::Real Ew = compute_emc_wetting_van_wagner(h, T);
            EXPECT_GE(Ed, Ew) << "h " << h << " T " << T;          // drying curve above wetting
            EXPECT_GE(Ed, prev_d); EXPECT_GE(Ew, prev_w);         // rises with humidity
            EXPECT_LT(Ed, 0.40);                                   // below the moisture ceiling
            prev_d = Ed; prev_w = Ew;
        }
    }
    // Warmer air holds fuel drier.
    EXPECT_LT(compute_emc_drying_van_wagner(40.0, 35.0), compute_emc_drying_van_wagner(40.0, 5.0));
    // At 40 % RH and 20 C, inside 0.08-0.12 rather than the legacy 0.17-0.19.
    EXPECT_GT(compute_emc_wetting_van_wagner(40.0, 20.0), 0.08);
    EXPECT_LT(compute_emc_drying_van_wagner(40.0, 20.0), 0.12);
}

TEST(FuelMoistureEMC, TimeLagEndsOnTheSelectedCurves)
{
    // Dry air (the RH = 0 of a deck with no moisture model, clamped to 1 % by
    // the legacy curves), 1-h fuel from 0.20 for 20 h at 26.85 C.
    EXPECT_NEAR(relax(0.20, 0.0, 26.85, 1.0, 20.0, FuelMoistureEMC::LEGACY),     0.0600, 1e-3);
    EXPECT_NEAR(relax(0.20, 0.0, 26.85, 1.0, 20.0, FuelMoistureEMC::VAN_WAGNER), FuelMoistureConst::M_MIN, TOL);

    // Humid air, 90 % RH at 20 C, wetting from 0.05: van_wagner ends between
    // its wetting and drying curves, legacy at the 0.35 cap.
    const amrex::Real Ew = compute_emc_wetting_van_wagner(90.0, 20.0), Ed = compute_emc_drying_van_wagner(90.0, 20.0);
    const amrex::Real M_vw = relax(0.05, 90.0, 20.0, 1.0, 20.0, FuelMoistureEMC::VAN_WAGNER);
    EXPECT_GE(M_vw, Ew - 1e-3);
    EXPECT_LE(M_vw, Ed + 1e-3);
    EXPECT_NEAR(relax(0.05, 90.0, 20.0, 1.0, 20.0, FuelMoistureEMC::LEGACY), 0.35, 1e-3);

    // One step already differs between the two choices.
    EXPECT_NE(advance_fuel_moisture_one_class(0.10, 40.0, 25.0, 0.0, 0.05, 1.0, FuelMoistureEMC::LEGACY),
              advance_fuel_moisture_one_class(0.10, 40.0, 25.0, 0.0, 0.05, 1.0, FuelMoistureEMC::VAN_WAGNER));
}

TEST(FuelMoistureEMC, StickSurfaceFollowsTheSelectedCurves)
{
    // A 1-h stick in 90 % RH at 20 C for 20 h: the volume average ends on the
    // van_wagner curves, well below the legacy cap.
    const int N = 6;
    amrex::Real A[N], B[N];
    for (int i = 0; i < N; ++i) { A[i] = B[i] = 0.05; }
    amrex::Real avg_vw = 0.0, avg_legacy = 0.0;
    for (int k = 0; k < 400; ++k) {
        avg_vw     = stick_advance_class(A, N, 0.15, 1.0, 90.0, 20.0, 0.0, 0.35, 1.0, 0.05, FuelMoistureEMC::VAN_WAGNER);
        avg_legacy = stick_advance_class(B, N, 0.15, 1.0, 90.0, 20.0, 0.0, 0.35, 1.0, 0.05, FuelMoistureEMC::LEGACY);
    }
    EXPECT_GE(avg_vw, compute_emc_wetting_van_wagner(90.0, 20.0) - 1e-3);
    EXPECT_LE(avg_vw, compute_emc_drying_van_wagner(90.0, 20.0) + 1e-3);
    EXPECT_NEAR(avg_legacy, 0.35, 1e-3);
}
