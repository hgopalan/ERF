#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <AMReX_REAL.H>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

#include "ERF_FuelModels.H"
#include "ERF_BehaveModel.H"

/**
 * @file ERF_GTestBehaveTransfer.cpp
 * @brief The BEHAVE live-to-dead herbaceous transfer window
 *        (erf.fire.behave.dynamic_transfer_lo/hi): the clamped linear
 *        fraction, that the window reaches compute_behave_state, and that
 *        fuels without a live herbaceous load ignore it.
 */

using amrex::Real;

namespace {

BehaveState state(const FuelModelParams& fp, Real M_herb, Real lo, Real hi)
{
    return compute_behave_state(fp, Real(0.06), Real(0.08), Real(0.10), M_herb, Real(0.90), lo, hi);
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

TEST(BehaveTransfer, FractionIsClampedLinearRamp)
{
    const Real lo = Real(0.30), hi = Real(1.20);
    EXPECT_NEAR(behave_herb_transfer_fraction(Real(0.10), lo, hi), 1.0, TOL);  // below: all
    EXPECT_NEAR(behave_herb_transfer_fraction(lo, lo, hi), 1.0, TOL);
    EXPECT_NEAR(behave_herb_transfer_fraction(Real(0.525), lo, hi), 0.75, TOL);
    EXPECT_NEAR(behave_herb_transfer_fraction(Real(0.75), lo, hi), 0.5, TOL);
    EXPECT_NEAR(behave_herb_transfer_fraction(hi, lo, hi), 0.0, TOL);
    EXPECT_NEAR(behave_herb_transfer_fraction(Real(2.00), lo, hi), 0.0, TOL);  // above: none

    // A narrower window
    EXPECT_NEAR(behave_herb_transfer_fraction(Real(0.40), Real(0.5), Real(1.0)), 1.0, TOL);
    EXPECT_NEAR(behave_herb_transfer_fraction(Real(0.60), Real(0.5), Real(1.0)), 0.8, TOL);
    EXPECT_NEAR(behave_herb_transfer_fraction(Real(1.10), Real(0.5), Real(1.0)), 0.0, TOL);

    // The default window is the line the model hard-coded before, 1.333 - 1.11 M,
    // to within its three-digit rounding: at most 1e-3, at M = 1.20.
    for (int k = 0; k < 90; ++k) {
        const Real M = Real(0.30) + Real(0.01) * Real(k);
        EXPECT_NEAR(behave_herb_transfer_fraction(M, lo, hi), 1.333 - 1.11 * M, 1.0e-3 + 1.0e-5);
    }
}

TEST(BehaveTransfer, WindowReachesState)
{
    const FuelModelParams fp = get_anderson_fuel_params(2);   // timber-grass: live herbaceous load
    ASSERT_GT(fp.w_lh, 0.0);
    const Real M = Real(0.75);

    // Windows giving the same fraction (one half) give the same state.
    const BehaveState half = state(fp, M, Real(0.30), Real(1.20));
    expect_same_state(state(fp, M, Real(0.50), Real(1.00)), half);
    expect_same_state(state(fp, M, Real(0.60), Real(0.90)), half);

    // Moving the window moves the load: all of it (M below lo), none (M above hi).
    const BehaveState all  = state(fp, M, Real(0.80), Real(1.20));
    const BehaveState none = state(fp, M, Real(0.10), Real(0.70));
    const Real moved = fp.w_lh * Real(1.0 - 0.0555);          // net of mineral content
    expect_close(all.wn_dead - none.wn_dead, moved);
    expect_close(half.wn_dead - none.wn_dead, Real(0.5) * moved);
    expect_close(all.wn_dead + all.wn_live, none.wn_dead + none.wn_live);
    EXPECT_GT(std::abs(all.r_0 - none.r_0), 0.01 * none.r_0);

    // With the whole load transferred and no live woody load, the live
    // herbaceous moisture no longer enters the state.
    ASSERT_EQ(fp.w_lw, 0.0);
    expect_same_state(state(fp, Real(0.10), Real(0.30), Real(1.20)),
                      state(fp, Real(0.25), Real(0.30), Real(1.20)));
}

TEST(BehaveTransfer, NoLiveHerbaceousLoadIgnoresWindow)
{
    const FuelModelParams fp = get_anderson_fuel_params(1);   // short grass: dead only
    ASSERT_EQ(fp.w_lh, 0.0);
    const BehaveState ref = state(fp, Real(0.90), Real(0.30), Real(1.20));
    expect_same_state(state(fp, Real(0.90), Real(0.80), Real(1.50)), ref);
    expect_same_state(state(fp, Real(0.90), Real(0.05), Real(0.10)), ref);
}
