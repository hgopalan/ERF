#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <AMReX_REAL.H>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

#include "ERF_FuelMoisture.H"

/**
 * @file ERF_GTestFuelMoisture.cpp
 * @brief The time-lag dead-fuel moisture model: which equilibrium curve the
 *        hysteresis picks, and the relaxation it gives in constant air.
 *
 * The desorption (drying) curve E_d lies above the adsorption (wetting) curve
 * E_w. A fuel wetter than E_d dries toward E_d, a fuel drier than E_w wets
 * toward E_w, and a fuel between the two does not change. In constant air the
 * forward Euler update is then M_n = E + (M0 - E) (1 - dt/tau_eff)^n, the
 * discrete form of E + (M0 - E) exp(-t/tau_eff).
 */

namespace {
    /// Air temperature of the Moisture_Relaxation deck (theta = 300 K) [C]
    constexpr amrex::Real T_DECK = 26.85;

    /// n forward Euler steps of one class in constant air, no rain
    amrex::Real relax (amrex::Real M, amrex::Real RH, amrex::Real T_C,
                       amrex::Real dt_h, amrex::Real tau_h, int n)
    {
        for (int i = 0; i < n; ++i) {
            M = advance_fuel_moisture_one_class(M, RH, T_C, 0.0, dt_h, tau_h);
        }
        return M;
    }
}

TEST(FuelMoisture, DesorptionCurveLiesAboveAdsorption)
{
    // Dry air (RH clamped to 1 %), and humidities below the crossing near 66 %
    EXPECT_NEAR(compute_emc_adsorption(0.0), 0.0351, 1e-4);
    EXPECT_NEAR(compute_emc_desorption(0.0), 0.0600, 1e-4);
    for (amrex::Real RH : {1.0, 10.0, 20.0, 40.0, 60.0}) {
        EXPECT_GT(compute_emc_desorption(RH), compute_emc_adsorption(RH)) << "RH " << RH;
    }
}

TEST(FuelMoisture, HysteresisPicksTheCurveTheFuelApproaches)
{
    const amrex::Real RH = 1.0;
    const amrex::Real E_w = compute_emc_adsorption(RH), E_d = compute_emc_desorption(RH);
    EXPECT_NEAR(compute_emc_with_hysteresis(RH, 0.20), E_d, TOL);   // wetter than E_d: dries toward it
    EXPECT_NEAR(compute_emc_with_hysteresis(RH, 0.02), E_w, TOL);   // drier than E_w: wets toward it
    EXPECT_NEAR(compute_emc_with_hysteresis(RH, 0.05), 0.05, TOL);  // between: already in equilibrium
    EXPECT_NEAR(compute_emc_with_hysteresis(RH, E_d), E_d, TOL);
    EXPECT_NEAR(compute_emc_with_hysteresis(RH, E_w), E_w, TOL);
}

TEST(FuelMoisture, BandNeverInverts)
{
    // The polynomials cross near 66 % RH and meet at the 0.35 cap above ~70 %:
    // the result always lies between the lower and the upper curve, or is M itself.
    for (int k = 0; k <= 200; ++k) {
        const amrex::Real RH = amrex::Real(0.5) * k;
        const amrex::Real lo = std::min(compute_emc_adsorption(RH), compute_emc_desorption(RH));
        const amrex::Real hi = std::max(compute_emc_adsorption(RH), compute_emc_desorption(RH));
        for (int j = 1; j <= 40; ++j) {
            const amrex::Real M = amrex::Real(0.01) * j;
            const amrex::Real expected = (M > hi) ? hi : ((M < lo) ? lo : M);
            EXPECT_NEAR(compute_emc_with_hysteresis(RH, M), expected, TOL) << "RH " << RH << " M " << M;
        }
    }
}

TEST(FuelMoisture, DryingFuelRelaxesToTheDesorptionCurve)
{
    // Moisture_Relaxation: the 1-hour class from 0.20 in still dry air, 5 s steps.
    // Drying toward E_d gives 0.1062 after an hour and 0.0753 after two; the
    // reversed choice read 0.0896 after an hour and pinned at E_d = 0.0600 by 6100 s.
    const amrex::Real RH = 1.0, M0 = 0.20, dt_h = amrex::Real(5.0 / 3600.0);
    const amrex::Real E_d = compute_emc_desorption(RH);
    const amrex::Real tau = compute_temp_correction(T_DECK);          // 1-hour class
    EXPECT_NEAR(tau, 0.9024, 1e-4);
    for (int n : {720, 1440}) {
        const amrex::Real t_h = n * dt_h;
        const amrex::Real M = relax(M0, RH, T_DECK, dt_h, 1.0, n);
        const amrex::Real discrete = E_d + (M0 - E_d) * std::pow(1.0 - dt_h / tau, n);
        const amrex::Real exact    = E_d + (M0 - E_d) * std::exp(-t_h / tau);
        EXPECT_NEAR(M, discrete, (sizeof(amrex::Real) == 8) ? 1e-10 : 1e-5) << "t " << t_h << " h";
        EXPECT_NEAR(M, exact, 1e-4) << "t " << t_h << " h";
        EXPECT_GT(M, E_d + 0.01);
    }
    EXPECT_NEAR(relax(M0, RH, T_DECK, dt_h, 1.0, 720), 0.1062, 5e-4);
}

TEST(FuelMoisture, WettingFuelRelaxesToTheAdsorptionCurve)
{
    // A 1-hour class at 0.05 in air of 40 % RH at 20 C (no temperature factor):
    // it wets toward E_w = 0.166, not toward E_d = 0.189.
    const amrex::Real RH = 40.0, M0 = 0.05, dt_h = amrex::Real(5.0 / 3600.0);
    const amrex::Real E_w = compute_emc_adsorption(RH);
    EXPECT_NEAR(compute_temp_correction(20.0), 1.0, TOL);
    EXPECT_NEAR(relax(M0, RH, 20.0, dt_h, 1.0, 720), E_w + (M0 - E_w) * std::exp(-1.0), 1e-4);
    EXPECT_LT(relax(M0, RH, 20.0, dt_h, 1.0, 720 * 20), E_w + TOL);   // never overshoots E_w
}

TEST(FuelMoisture, FuelBetweenTheCurvesHoldsUnlessItRains)
{
    const amrex::Real RH = 1.0, M0 = 0.05, dt_h = amrex::Real(5.0 / 3600.0);
    EXPECT_NEAR(relax(M0, RH, T_DECK, dt_h, 1.0, 1440), M0, TOL);
    const amrex::Real wet = advance_fuel_moisture_one_class(M0, RH, T_DECK, 2.0, dt_h, 1.0);
    EXPECT_NEAR(wet, M0 + dt_h * compute_precip_wetting_rate(2.0), TOL);
}
