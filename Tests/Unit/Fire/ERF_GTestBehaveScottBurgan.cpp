#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <AMReX_REAL.H>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

#include "ERF_FuelModels.H"
#include "ERF_BehaveModel.H"

/**
 * @file ERF_GTestBehaveScottBurgan.cpp
 * @brief BEHAVE on the Scott-Burgan dynamic fuels: the herbaceous load
 *        transfers once, in BEHAVE's own window, and the dead herbaceous
 *        class takes the 1-h moisture.
 */

using amrex::Real;

namespace {

constexpr int GR2 = 102;
constexpr Real S_T = Real(0.0555);   // total mineral content in compute_behave_state

BehaveState state(const FuelModelParams& fp, Real mc_1hr, Real M_live)
{
    return compute_behave_state(fp, mc_1hr, Real(0.07), Real(0.08), M_live, M_live,
                                Real(0.30), Real(1.20));
}

void expect_close(Real a, Real b)
{
    EXPECT_NEAR(a, b, TOL * std::max(Real(1.0), std::abs(b)));
}

void expect_same_state(const BehaveState& a, const BehaveState& b)
{
    expect_close(a.wn_dead, b.wn_dead);           expect_close(a.wn_live, b.wn_live);
    expect_close(a.fuelheat_dead, b.fuelheat_dead); expect_close(a.fuelheat_live, b.fuelheat_live);
    expect_close(a.etam_dead, b.etam_dead);       expect_close(a.etam_live, b.etam_live);
    expect_close(a.etas_dead, b.etas_dead);       expect_close(a.etas_live, b.etas_live);
    expect_close(a.heat_sink, b.heat_sink);       expect_close(a.gamma, b.gamma);
    expect_close(a.xifr, b.xifr);                 expect_close(a.r_0, b.r_0);
    expect_close(a.C, b.C);                       expect_close(a.B, b.B);
    expect_close(a.beta_ratio_E, b.beta_ratio_E); expect_close(a.phi_s_const, b.phi_s_const);
    expect_close(a.U_max_ftmin, b.U_max_ftmin);   expect_close(a.wind_conv, b.wind_conv);
}

} // namespace

TEST(BehaveScottBurgan, GR2HerbaceousLoadTransfersOnce)
{
    const Real M = Real(0.90);
    const FuelModelParams table = get_scott_burgan_fuel_params(GR2);   // untransferred
    ASSERT_GT(table.w_lh, 0.0);
    ASSERT_EQ(table.w_d10 + table.w_d100 + table.w_lw, 0.0);

    // BEHAVE starts from the untransferred table ...
    const FuelModelParams fp = behave_fuel_params(GR2, FUEL_SET_SCOTT_BURGAN40);
    expect_close(fp.w_d1, table.w_d1);
    expect_close(fp.w_lh, table.w_lh);

    // ... so only its window moves load: (1.20 - 0.90) / 0.90 = 1/3 of w_lh is dead.
    const BehaveState bs = state(fp, Real(0.06), M);
    expect_close(bs.wn_dead, (table.w_d1 + table.w_lh / Real(3.0)) * (Real(1.0) - S_T));
    expect_close(bs.wn_live, (table.w_lh * Real(2.0) / Real(3.0)) * (Real(1.0) - S_T));

    // The cured table the single-class paths use is unchanged, and handing it
    // to BEHAVE would transfer twice: 1/3 + (2/3)(1/3) = 5/9 of w_lh dead.
    const FuelModelParams cured = get_fuel_params(GR2, FUEL_SET_SCOTT_BURGAN40, M);
    expect_close(cured.w_d1, table.w_d1 + table.w_lh / Real(3.0));
    const BehaveState twice = state(cured, Real(0.06), M);
    expect_close(twice.wn_dead, (table.w_d1 + table.w_lh * Real(5.0) / Real(9.0)) * (Real(1.0) - S_T));
}

TEST(BehaveScottBurgan, AndersonCodesUnaffected)
{
    for (int code : {2, 4, 5, 7, 10}) {
        const FuelModelParams a = get_anderson_fuel_params(code);
        expect_same_state(state(behave_fuel_params(code, FUEL_SET_SCOTT_BURGAN40), Real(0.06), Real(0.90)),
                          state(a, Real(0.06), Real(0.90)));
        expect_same_state(state(behave_fuel_params(code, FUEL_SET_ANDERSON13), Real(0.06), Real(0.90)),
                          state(a, Real(0.06), Real(0.90)));
    }
}

TEST(BehaveScottBurgan, DeadHerbaceousTakesOneHourMoisture)
{
    // Fully cured (M below the window), GR2's herbaceous load is dead. With the
    // 1-h SAV set to the herbaceous one, the dead herbaceous class is then the
    // same fuel as that load placed in the 1-h class, so the states agree only
    // if the class carries the 1-h moisture.
    FuelModelParams fp = behave_fuel_params(GR2, FUEL_SET_SCOTT_BURGAN40);
    fp.sigma_d1 = fp.sigma_lh;
    FuelModelParams merged = fp;
    merged.w_d1 += merged.w_lh;
    merged.w_lh  = 0.0;

    for (Real mc_1hr : {Real(0.04), Real(0.06), Real(0.10)}) {
        expect_same_state(state(fp, mc_1hr, Real(0.20)), state(merged, mc_1hr, Real(0.20)));
    }
    // and the dead damping follows the 1-h moisture
    EXPECT_GT(state(fp, Real(0.04), Real(0.20)).etam_dead,
              state(fp, Real(0.10), Real(0.20)).etam_dead);
}
