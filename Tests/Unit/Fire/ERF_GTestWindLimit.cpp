#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <AMReX_REAL.H>

#include "ERF_Rothermel.H"
#include "ERF_FuelModels.H"
#include "ERF_BehaveModel.H"

/**
 * @file ERF_GTestWindLimit.cpp
 * @brief erf.fire.use_wind_limit and erf.fire.wind_limit: with the limit on
 *        (default) the Rothermel and BEHAVE kernels cap the midflame wind at
 *        Rothermel's (1972) eq. 87, U_max = 0.9 I_R (ft/min, I_R in
 *        BTU/ft2/min), the "wind limit" of Andrews (2018); wind_limit =
 *        fuel_class keeps the 300/500 ft/min rule this code used until
 *        October 2026; false removes the cap from the uniform coefficients,
 *        the per-fuel table and the BEHAVE state.
 *
 * The reaction intensity the limit is built from is transcribed here from
 * Rothermel (1972) independently of the kernel. Pre-fix numbers (the
 * fuel_class rule): short grass at 8 % capped at 300 ft/min = 1.52 m/s, so a
 * 6 m/s reference wind (2.17 m/s midflame) gave 0.216 m/s where Rothermel's
 * own limit (687 ft/min = 3.49 m/s) gives 0.428 m/s.
 */

using amrex::Real;

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;
/// Relative tolerance of a transcription against the kernel
static constexpr double REAL_RTOL = std::is_same<amrex::Real, float>::value ? 1.0e-5 : 1.0e-9;

static constexpr Real M_DEAD = 0.055;                ///< Coen et al. (2013) moisture
static constexpr Real M_LIVE = 1.0;
static constexpr Real TRANSFER_LO = 0.30;            ///< erf.fire.behave.dynamic_transfer_lo default
static constexpr Real TRANSFER_HI = 1.20;            ///< erf.fire.behave.dynamic_transfer_hi default
static constexpr Real FTMIN_PER_MS = 196.85;
static constexpr Real UNCAPPED = std::numeric_limits<Real>::max();

/// Relative closeness at the build precision.
static void expect_close(Real a, Real b)
{
    EXPECT_NEAR(a, b, TOL * std::max(Real(1.0), std::abs(b)));
}

/// Rothermel's R0 (1 + phi_w) with no cap, evaluated independently of the kernel.
static Real rothermel_uncapped(const RothermelComputed& rc, Real U)
{
    return rc.R0 * (Real(1.0) + rc.C * std::pow(U * FTMIN_PER_MS, rc.B) * rc.beta_ratio_E);
}

/// Rothermel (1972) reaction intensity [BTU/ft2/min] of a single-class fuel at
/// one dead moisture, transcribed from INT-115 (eqs. 27, 29, 30, 36-38, 52).
static double reaction_intensity_1972(const FuelModelParams& fp, double M)
{
    const double w0 = fp.w_d1 + fp.w_d10 + fp.w_d100 + fp.w_lh + fp.w_lw;
    const double wn = w0 * (1.0 - 0.0555);
    const double rho_b = w0 / fp.delta;
    const double beta = rho_b / fp.rho_p;
    const double sigma = fp.sigma_d1;
    const double beta_op = 3.348 * std::pow(sigma, -0.8189);
    const double gmax = std::pow(sigma, 1.5) / (495.0 + 0.0594 * std::pow(sigma, 1.5));
    const double A = 133.0 * std::pow(sigma, -0.7913);
    const double gamma = gmax * std::pow(beta / beta_op, A) * std::exp(A * (1.0 - beta / beta_op));
    const double rm = std::min(M / fp.Mx, 1.0);
    const double eta_M = std::max(0.0, 1.0 - 2.59 * rm + 5.11 * rm * rm - 3.52 * rm * rm * rm);
    const double eta_s = 0.174 * std::pow(0.010, -0.19);
    return gamma * wn * fp.heat_content * eta_M * eta_s;
}

TEST(WindLimit, RothermelLimitIsNinetyPercentOfReactionIntensity)
{
    // Rothermel (1972) eq. 87: U_max [ft/min] = 0.9 I_R; the finding's
    // number for short grass at 8 % (I_R about 765, U_max about 690 ft/min)
    const FuelModelParams fp = get_fuel_params(1, 0);
    const double I_R = reaction_intensity_1972(fp, 0.08);
    EXPECT_NEAR(I_R, 765.0, 15.0) << "FM1 reaction intensity at 8 %";
    EXPECT_NEAR(rothermel_wind_limit_ftmin(Real(I_R)), 0.9 * I_R, REAL_RTOL * I_R);
    const RothermelComputed rc = compute_rothermel_params(fp, 0.08, 0.08, 0.08);
    EXPECT_NEAR(rc.I_R, I_R, 1e-6 * I_R) << "the kernel's I_R against the transcription";
    EXPECT_NEAR(rc.U_max_ftmin, 0.9 * I_R, REAL_RTOL * I_R) << "default = Rothermel's limit";
    EXPECT_GT(rc.U_max_ftmin, Real(600.0));   // not the 300 ft/min of the fuel-class rule
    // the fuel-class rule, kept as erf.fire.wind_limit = fuel_class
    EXPECT_EQ(rothermel_wind_cap_ftmin(3500.0, Real(I_R), true, fire_wind_limit::fuel_class), Real(300.0));
    EXPECT_EQ(rothermel_wind_cap_ftmin(800.0, Real(I_R), true, fire_wind_limit::fuel_class), Real(500.0));
    EXPECT_EQ(rothermel_wind_cap_ftmin(3500.0, Real(I_R), false, fire_wind_limit::rothermel), UNCAPPED);
    EXPECT_EQ(rothermel_wind_cap_ftmin(800.0, Real(I_R), false, fire_wind_limit::fuel_class), UNCAPPED);
    // the limit rises with drier fuel
    const RothermelComputed rc_dry = compute_rothermel_params(fp, 0.04, 0.04, 0.04);
    EXPECT_GT(rc_dry.U_max_ftmin, rc.U_max_ftmin);
}

TEST(WindLimit, ShortGrassHeadRateAtSixMetresPerSecond)
{
    // FM1 at 8 %, reference wind 6 m/s under the Andrews WAF 0.362: midflame
    // 2.17 m/s = 428 ft/min, below Rothermel's 687 ft/min limit and above the
    // fuel-class rule's 300. Rothermel's limit leaves the rate uncapped
    // (0.43 m/s); the fuel-class rule gave 0.22 m/s (the pre-fix number).
    const FuelModelParams fp = get_fuel_params(1, 0);
    const RothermelComputed rc_roth = compute_rothermel_params(fp, 0.08, 0.08, 0.08, true, fire_wind_limit::rothermel);
    const RothermelComputed rc_fuel = compute_rothermel_params(fp, 0.08, 0.08, 0.08, true, fire_wind_limit::fuel_class);
    const RothermelComputed rc_off  = compute_rothermel_params(fp, 0.08, 0.08, 0.08, false);
    const Real U_mid = Real(6.0) * Real(0.362);
    const Real R_roth = rothermel_ros_cell(U_mid, 0.0, 0.0, 0.0, rc_roth);
    const Real R_fuel = rothermel_ros_cell(U_mid, 0.0, 0.0, 0.0, rc_fuel);
    const Real R_off  = rothermel_ros_cell(U_mid, 0.0, 0.0, 0.0, rc_off);
    expect_close(R_roth, R_off);                       // the limit does not bind here
    EXPECT_NEAR(R_roth, 0.428, 0.02);
    EXPECT_NEAR(R_fuel, 0.216, 0.02);                  // the pre-fix rate
    EXPECT_GT(R_roth / R_fuel, Real(1.8));
    // 2 and 3 m/s midflame differ under Rothermel's limit (both were capped at 1.52 m/s before)
    EXPECT_GT(rothermel_ros_cell(3.0, 0.0, 0.0, 0.0, rc_roth), rothermel_ros_cell(2.0, 0.0, 0.0, 0.0, rc_roth) * Real(1.1));
    expect_close(rothermel_ros_cell(3.0, 0.0, 0.0, 0.0, rc_fuel), rothermel_ros_cell(2.0, 0.0, 0.0, 0.0, rc_fuel));
    // above Rothermel's limit the capped rate saturates at U_max and the uncapped one keeps growing
    const Real U_lim = rc_roth.U_max_ftmin / FTMIN_PER_MS;
    expect_close(rothermel_ros_cell(U_lim * Real(2.0), 0.0, 0.0, 0.0, rc_roth), rothermel_uncapped(rc_roth, U_lim));
    EXPECT_GT(rothermel_ros_cell(U_lim * Real(2.0), 0.0, 0.0, 0.0, rc_off), rothermel_ros_cell(U_lim * Real(2.0), 0.0, 0.0, 0.0, rc_roth));
}

TEST(WindLimit, RothermelDefaultKeepsCap)
{
    const FuelModelParams fp = get_fuel_params(1, 0);
    const RothermelComputed rc_default = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD);
    const RothermelComputed rc_on  = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true);
    const RothermelComputed rc_off = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, false);

    EXPECT_NEAR(rc_default.U_max_ftmin, 0.9 * rc_on.I_R, REAL_RTOL * rc_on.I_R);
    EXPECT_NEAR(rc_on.U_max_ftmin, 0.9 * rc_on.I_R, REAL_RTOL * rc_on.I_R);
    EXPECT_EQ(rc_off.U_max_ftmin, UNCAPPED);
    // Nothing else depends on the flag
    expect_close(rc_off.R0, rc_on.R0);
    expect_close(rc_off.C, rc_on.C);
    expect_close(rc_off.B, rc_on.B);
    expect_close(rc_off.beta_ratio_E, rc_on.beta_ratio_E);
}

TEST(WindLimit, RothermelRateOfSpreadFuelClassRule)
{
    const FuelModelParams fp = get_fuel_params(1, 0);
    const RothermelComputed rc_on  = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true, fire_wind_limit::fuel_class);
    const RothermelComputed rc_off = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, false);
    const Real U_mews = Real(300.0) / FTMIN_PER_MS;   // 1.524 m/s

    // Below the cap the flag changes nothing
    const Real U_low = Real(1.0);
    expect_close(rothermel_ros_cell(U_low, 0.0, 0.0, 0.0, rc_off), rothermel_ros_cell(U_low, 0.0, 0.0, 0.0, rc_on));

    // Above it the capped rate saturates at the rule's cap and the uncapped one keeps growing
    for (Real U : {Real(1.81), Real(3.0), Real(10.0)}) {
        const Real R_on  = rothermel_ros_cell(U, 0.0, 0.0, 0.0, rc_on);
        const Real R_off = rothermel_ros_cell(U, 0.0, 0.0, 0.0, rc_off);
        expect_close(R_on, rothermel_uncapped(rc_on, U_mews));
        expect_close(R_off, rothermel_uncapped(rc_off, U));
        EXPECT_GT(R_off, R_on) << "U = " << U;
    }
    // Slope is untouched: an uncapped run on a slope adds the same phi_s
    const Real U = Real(3.0), s = Real(0.3);
    expect_close(rothermel_ros_cell(U, 0.0, s, 0.0, rc_off) - rothermel_ros_cell(U, 0.0, 0.0, 0.0, rc_off),
                 rc_off.R0 * rc_off.phi_s_const * s * s);
}

TEST(WindLimit, PerFuelTable)
{
    const auto on   = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD);
    const auto fcls = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD, 0, -1.0, true, nullptr, fire_wind_limit::fuel_class);
    const auto off  = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD, 0, -1.0, false);
    ASSERT_EQ(on.size(), off.size());
    ASSERT_EQ(on.size(), fcls.size());

    EXPECT_EQ(off[0].R0, Real(0.0));                  // non-burnable slot stays zero

    // The published sets: every slot is a real fuel, capped by its own I_R or by the rule, or not.
    for (int slot = 1; slot < FUEL_SLOT_CUSTOM_BASE; ++slot) {
        EXPECT_NEAR(on[slot].U_max_ftmin, 0.9 * on[slot].I_R, REAL_RTOL * std::max(Real(1.0), on[slot].I_R)) << "slot " << slot;
        EXPECT_TRUE(fcls[slot].U_max_ftmin == Real(300.0) || fcls[slot].U_max_ftmin == Real(500.0)) << "slot " << slot;
        EXPECT_EQ(off[slot].U_max_ftmin, UNCAPPED) << "slot " << slot;
        EXPECT_NEAR(off[slot].R0, on[slot].R0, TOL * std::max(Real(1.0), on[slot].R0)) << "slot " << slot;
    }

    // The deck-defined slots with no slot table handed in: non-burnable, so a
    // custom code that reached the table without a deck block cannot spread.
    for (int slot = FUEL_SLOT_CUSTOM_BASE; slot < FUEL_SLOT_COUNT; ++slot) {
        EXPECT_EQ(on[slot].R0, Real(0.0))  << "slot " << slot;
        EXPECT_EQ(off[slot].R0, Real(0.0)) << "slot " << slot;
    }
}

TEST(WindLimit, Behave)
{
    const FuelModelParams fp = get_fuel_params(1, 0);
    const BehaveState bs_default = compute_behave_state(fp, M_DEAD, M_DEAD, M_DEAD, M_LIVE, M_LIVE,
                                                        TRANSFER_LO, TRANSFER_HI);
    const BehaveState bs_fcls = compute_behave_state(fp, M_DEAD, M_DEAD, M_DEAD, M_LIVE, M_LIVE,
                                                     TRANSFER_LO, TRANSFER_HI, true, fire_wind_limit::fuel_class);
    const BehaveState bs_off = compute_behave_state(fp, M_DEAD, M_DEAD, M_DEAD, M_LIVE, M_LIVE,
                                                    TRANSFER_LO, TRANSFER_HI, false);

    // BEHAVE on FM1 is single-class Rothermel: the same limit as compute_rothermel_params
    const RothermelComputed rc = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD);
    EXPECT_NEAR(bs_default.U_max_ftmin, rc.U_max_ftmin, 1e-6 * rc.U_max_ftmin);
    EXPECT_EQ(bs_fcls.U_max_ftmin, Real(300.0));
    EXPECT_EQ(bs_off.U_max_ftmin, UNCAPPED);
    expect_close(bs_off.r_0, bs_default.r_0);

    const Real U_low = Real(1.0), U_high = Real(3.0), U_mews = Real(300.0) / FTMIN_PER_MS;
    expect_close(behave_ros_cell(U_low, 0.0, 0.0, 0.0, bs_off), behave_ros_cell(U_low, 0.0, 0.0, 0.0, bs_default));
    expect_close(behave_ros_cell(U_high, 0.0, 0.0, 0.0, bs_fcls), behave_ros_cell(U_mews, 0.0, 0.0, 0.0, bs_off));
    expect_close(behave_ros_cell(U_high, 0.0, 0.0, 0.0, bs_off),
                 bs_off.r_0 * (Real(1.0) + bs_off.C * std::pow(U_high * FTMIN_PER_MS, bs_off.B) * bs_off.beta_ratio_E));
    EXPECT_GT(behave_ros_cell(U_high, 0.0, 0.0, 0.0, bs_off), behave_ros_cell(U_high, 0.0, 0.0, 0.0, bs_fcls));
    // Rothermel's limit (3.5 m/s) is above 3 m/s, so the default is uncapped there
    expect_close(behave_ros_cell(U_high, 0.0, 0.0, 0.0, bs_default), behave_ros_cell(U_high, 0.0, 0.0, 0.0, bs_off));
}
