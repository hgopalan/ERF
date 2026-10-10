#include <gtest/gtest.h>
#include <cmath>
#include <AMReX_REAL.H>

#include "ERF_FuelModels.H"
#include "ERF_BehaveModel.H"

/**
 * @file ERF_GTestBehaveNetLoad.cpp
 * @brief The net fuel load of the BEHAVE multi-class state is Rothermel's
 *        (1972, eq. 53) surface-area-weighted load with Albini's (1976) size
 *        classes (Andrews 2018, eq. 61), not the plain sum of the classes.
 *
 * The weights are transcribed here from Anderson's (1982) Table 1 for FM10
 * and FM13 (loads in lb/ft2, SAVs 1-h from the table, 10-h 109 and 100-h 30
 * ft^-1, particle density 32 lb/ft3, total mineral content 0.0555): the
 * surface-area share of class j is A_j / sum A with A_j = sigma_j w_j / rho_p,
 * and the three Anderson dead classes fall in different size classes, so
 * w_n = (1 - S_T) sum_j f_j w_j. The sum, kept as erf.fire.behave.net_load =
 * sum, overstates it by 3.3x (FM10) and 5.2x (FM13), the zero-wind rate by
 * 3.0x and 4.3x since the heat sink changes with it; a single-class fuel
 * (FM1) is the same either way, heat sink included.
 */

using namespace amrex;

namespace {

constexpr double REL = (sizeof(Real) == 8) ? 1.0e-10 : 1.0e-4;
constexpr double S_T = 0.0555;

BehaveState state (const FuelModelParams& fp, int net_load)
{
    // 6/7/8 % dead, live herbaceous above the transfer window (no transfer)
    return compute_behave_state(fp, Real(0.06), Real(0.07), Real(0.08), Real(1.2), Real(1.2),
                                Real(0.3), Real(1.2), true, fire_wind_limit::rothermel, net_load);
}

struct Dead { double w1, w10, w100, s1; };

double weighted_net_load (const Dead& d)
{
    const double rho_p = 32.0, s10 = 109.0, s100 = 30.0;
    const double A1 = d.s1 * d.w1 / rho_p, A10 = s10 * d.w10 / rho_p, A100 = s100 * d.w100 / rho_p;
    const double A  = A1 + A10 + A100;
    return (1.0 - S_T) * (A1 / A * d.w1 + A10 / A * d.w10 + A100 / A * d.w100);
}

double summed_net_load (const Dead& d) { return (1.0 - S_T) * (d.w1 + d.w10 + d.w100); }

} // namespace

TEST(BehaveNetLoad, FM10IsTheSurfaceAreaWeightedLoad)
{
    const Dead fm10{0.138, 0.092, 0.230, 2000.0};
    const FuelModelParams fp = get_fuel_params(10, FUEL_SET_ANDERSON13);
    const BehaveState w = state(fp, behave_net_load::weighted);
    const BehaveState s = state(fp, behave_net_load::sum);

    EXPECT_NEAR(w.wn_dead, weighted_net_load(fm10), REL * weighted_net_load(fm10));
    EXPECT_NEAR(s.wn_dead, summed_net_load(fm10),   REL * summed_net_load(fm10)) << "the form before 2026-10";
    EXPECT_NEAR(s.wn_dead / w.wn_dead, 3.3, 0.05);
    // the live class is alone in its size class: the same load either way
    EXPECT_NEAR(w.wn_live, (1.0 - S_T) * 0.092, REL);
    EXPECT_NEAR(s.wn_live, (1.0 - S_T) * 0.092, REL);
    // the reaction intensity, and the zero-wind rate with it, follow the load
    EXPECT_GT(s.r_0, 2.0 * w.r_0) << "the summed load spreads FM10 more than twice as fast";
    EXPECT_GT(w.r_0, 0.0);
}

TEST(BehaveNetLoad, FM13OverstatementIsFiveFold)
{
    const Dead fm13{0.322, 1.058, 1.288, 1500.0};
    const FuelModelParams fp = get_fuel_params(13, FUEL_SET_ANDERSON13);
    const BehaveState w = state(fp, behave_net_load::weighted);
    const BehaveState s = state(fp, behave_net_load::sum);
    EXPECT_NEAR(w.wn_dead, weighted_net_load(fm13), REL * weighted_net_load(fm13));
    EXPECT_NEAR(s.wn_dead / w.wn_dead, 5.2, 0.05);
}

TEST(BehaveNetLoad, ASingleClassFuelIsTheSameEitherWay)
{
    const FuelModelParams fp = get_fuel_params(1, FUEL_SET_ANDERSON13);
    const BehaveState w = state(fp, behave_net_load::weighted);
    const BehaveState s = state(fp, behave_net_load::sum);
    EXPECT_NEAR(w.wn_dead, (1.0 - S_T) * 0.034, REL);
    EXPECT_NEAR(w.wn_dead,   s.wn_dead,   REL * s.wn_dead);
    EXPECT_NEAR(w.heat_sink, s.heat_sink, REL * s.heat_sink) << "one class: eps and Q_ig of that class";
    EXPECT_NEAR(w.r_0,       s.r_0,       REL * s.r_0);
}

TEST(BehaveNetLoad, HeatSinkWeightsEachClass)
{
    // FM10: rho_b sum_j f_j eps_j Q_ig(M_j) over the dead classes, the live
    // class with its own eps and moisture, categories weighted by area share
    const FuelModelParams fp = get_fuel_params(10, FUEL_SET_ANDERSON13);
    const BehaveState w = state(fp, behave_net_load::weighted);
    const double rho_p = 32.0, s1 = 2000.0, s10 = 109.0, s100 = 30.0, slh = 1500.0;
    const double w1 = 0.138, w10 = 0.092, w100 = 0.230, wlh = 0.092, depth = 1.0;
    const double A1 = s1 * w1 / rho_p, A10 = s10 * w10 / rho_p, A100 = s100 * w100 / rho_p, Alh = slh * wlh / rho_p;
    const double Ad = A1 + A10 + A100;
    auto eps = [] (double s) { return std::exp(-138.0 / s); };
    auto qig = [] (double m) { return 250.0 + 1116.0 * m; };
    const double sink_dead = (A1 / Ad) * eps(s1) * qig(0.06) + (A10 / Ad) * eps(s10) * qig(0.07) + (A100 / Ad) * eps(s100) * qig(0.08);
    const double sink_live = eps(slh) * qig(1.2);
    const double rho_b = (w1 + w10 + w100 + wlh) / depth;
    const double expect = rho_b * (Ad / (Ad + Alh) * sink_dead + Alh / (Ad + Alh) * sink_live);
    EXPECT_NEAR(w.heat_sink, expect, REL * expect);
}
