#include <gtest/gtest.h>
#include <cmath>
#include <AMReX_REAL.H>

#include "ERF_FireHeatFlux.H"
#include "ERF_FireDiagnostics.H"

/**
 * @file ERF_GTestFireHeatBudget.cpp
 * @brief The heat a burning cell hands the atmosphere over a step is the heat
 *        of the fuel the step consumes: Q = h w (1 - e^{-dt/tau}) / dt, so the
 *        sum of Q dt over the steps is h (w_0 - w_N) exactly, where the
 *        instantaneous h w / tau at the start of the step overshot it by
 *        (dt/tau) / (1 - e^{-dt/tau}). Byram's intensity is h w_0 R on a cell
 *        the front has reached and still burns, not h (w_0 - w) R, which was
 *        zero on arrival and largest on burned-out cells; that form is kept as
 *        the heat-release diagnostic.
 */

using namespace amrex;

namespace {
constexpr double REL = (sizeof(Real) == 8) ? 1.0e-10 : 1.0e-4;
}

TEST(FireHeatBudget, StepMeanFluxIntegratesToTheFuelBurned)
{
    const Real h = 1.8e7_rt, tau = 60.0_rt, w0 = 1.2_rt;
    for (Real dt : {6.0_rt, 60.0_rt, 180.0_rt}) {
        Real w = w0, E = 0.0_rt;
        for (int n = 0; n < 20; ++n) {
            E += compute_heat_flux_cell(-1.0_rt, w, h, tau, dt) * dt;
            w  = deplete_fuel_load(-1.0_rt, w, tau, dt);
        }
        EXPECT_NEAR(E, h * (w0 - w), REL * h * w0) << "dt / tau = " << dt / tau;
    }
}

TEST(FireHeatBudget, TheInstantaneousFormOvershoots)
{
    const Real h = 1.8e7_rt, tau = 60.0_rt, w0 = 1.2_rt;
    const Real q_inst = compute_heat_flux_cell(-1.0_rt, w0, h, tau, 0.0_rt);
    const Real q_mean = compute_heat_flux_cell(-1.0_rt, w0, h, tau, tau);
    EXPECT_NEAR(q_inst, w0 * h / tau, REL * w0 * h / tau) << "dt <= 0: the instantaneous power";
    EXPECT_NEAR(q_inst / q_mean, 1.0 / (1.0 - std::exp(-1.0)), 1.0e3 * REL) << "1.58x at dt = tau";
    // the step mean tends to the instantaneous power as dt -> 0
    EXPECT_NEAR(compute_heat_flux_cell(-1.0_rt, w0, h, tau, 1.0e-3_rt), q_inst, 1.0e-4 * q_inst);
}

TEST(FireHeatBudget, UnburnedAndExhaustedCellsGiveNothing)
{
    EXPECT_EQ(compute_heat_flux_cell( 0.5_rt, 1.0_rt, 1.8e7_rt, 60.0_rt, 10.0_rt), 0.0_rt);
    EXPECT_EQ(compute_heat_flux_cell(-0.5_rt, 0.0_rt, 1.8e7_rt, 60.0_rt, 10.0_rt), 0.0_rt);
    EXPECT_EQ(compute_heat_flux_cell(-0.5_rt, 1.0_rt, 1.8e7_rt,  0.0_rt, 10.0_rt), 0.0_rt);
    EXPECT_EQ(deplete_fuel_load(0.5_rt, 1.0_rt, 60.0_rt, 10.0_rt), 1.0_rt);
}

TEST(ByramIntensity, AFreshCellCarriesTheFullLoad)
{
    const Real h = 18000.0_rt, w0 = 2.0_rt, R = 0.5_rt;
    EXPECT_NEAR(compute_fireline_intensity_kW_per_m(-1.0_rt, R, w0, w0, h), h * w0 * R, REL * h * w0 * R)
        << "nothing consumed yet: I_B = h w0 R";
    EXPECT_NEAR(compute_fireline_intensity_kW_per_m(-1.0_rt, R, w0, 0.5_rt * w0, h), h * w0 * R, REL * h * w0 * R)
        << "half consumed: still the load the front consumes";
    EXPECT_EQ(compute_fireline_intensity_kW_per_m(-1.0_rt, R, w0, 0.005_rt * w0, h), 0.0_rt) << "burned out";
    EXPECT_EQ(compute_fireline_intensity_kW_per_m( 1.0_rt, R, w0, w0, h), 0.0_rt) << "unburned";
    EXPECT_EQ(compute_fireline_intensity_kW_per_m(-1.0_rt, 0.0_rt, w0, w0, h), 0.0_rt) << "no spread";
    EXPECT_NEAR(compute_flame_length_m(h * w0 * R), 0.0775 * std::pow(h * w0 * R, 0.46), 1.0e3 * REL);
}

TEST(ByramIntensity, HeatReleaseIsTheConsumedLoad)
{
    const Real h = 18000.0_rt, w0 = 2.0_rt, R = 0.5_rt;
    EXPECT_EQ(compute_heat_release_kW_per_m(-1.0_rt, R, w0, w0, h), 0.0_rt) << "nothing consumed yet";
    EXPECT_NEAR(compute_heat_release_kW_per_m(-1.0_rt, R, w0, 0.5_rt * w0, h), 0.5 * h * w0 * R, REL * h * w0 * R);
    EXPECT_NEAR(compute_heat_release_kW_per_m(-1.0_rt, R, w0, 0.0_rt, h), h * w0 * R, REL * h * w0 * R) << "burned out";
    EXPECT_EQ(compute_heat_release_kW_per_m( 1.0_rt, R, w0, 0.0_rt, h), 0.0_rt);
}
