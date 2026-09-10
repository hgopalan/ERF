#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <AMReX_REAL.H>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

#include "ERF_Rothermel.H"
#include "ERF_FuelModels.H"
#include "ERF_BehaveModel.H"

/**
 * @file ERF_GTestWindLimit.cpp
 * @brief erf.fire.use_wind_limit: the maximum effective wind speed cap of the
 *        Rothermel and BEHAVE kernels is on by default, and false removes it
 *        from the uniform coefficients, the per-fuel table and the BEHAVE state.
 */

using amrex::Real;

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

TEST(WindLimit, CapValue)
{
    EXPECT_EQ(rothermel_wind_cap_ftmin(3500.0, true), Real(300.0));   // fine fuel
    EXPECT_EQ(rothermel_wind_cap_ftmin(800.0, true), Real(500.0));    // coarse fuel
    EXPECT_EQ(rothermel_wind_cap_ftmin(3500.0, false), UNCAPPED);
    EXPECT_EQ(rothermel_wind_cap_ftmin(800.0, false), UNCAPPED);
}

TEST(WindLimit, RothermelDefaultKeepsCap)
{
    const FuelModelParams fp = get_fuel_params(1, 0);
    const RothermelComputed rc_default = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD);
    const RothermelComputed rc_on  = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true);
    const RothermelComputed rc_off = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, false);

    EXPECT_EQ(rc_default.U_max_ftmin, Real(300.0));
    EXPECT_EQ(rc_on.U_max_ftmin, Real(300.0));
    EXPECT_EQ(rc_off.U_max_ftmin, UNCAPPED);
    // Nothing else depends on the flag
    expect_close(rc_off.R0, rc_on.R0);
    expect_close(rc_off.C, rc_on.C);
    expect_close(rc_off.B, rc_on.B);
    expect_close(rc_off.beta_ratio_E, rc_on.beta_ratio_E);
}

TEST(WindLimit, RothermelRateOfSpread)
{
    const FuelModelParams fp = get_fuel_params(1, 0);
    const RothermelComputed rc_on  = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true);
    const RothermelComputed rc_off = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, false);
    const Real U_mews = Real(300.0) / FTMIN_PER_MS;   // 1.524 m/s

    // Below the cap the flag changes nothing
    const Real U_low = Real(1.0);
    expect_close(rothermel_ros_cell(U_low, 0.0, 0.0, 0.0, rc_off), rothermel_ros_cell(U_low, 0.0, 0.0, 0.0, rc_on));

    // Above it the capped rate saturates at the MEWS and the uncapped one keeps growing
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
    const auto on  = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD);
    const auto off = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD, 0, -1.0, false);
    ASSERT_EQ(on.size(), off.size());

    EXPECT_EQ(off[0].R0, Real(0.0));                  // non-burnable slot stays zero
    for (std::size_t slot = 1; slot < off.size(); ++slot) {
        EXPECT_TRUE(on[slot].U_max_ftmin == Real(300.0) || on[slot].U_max_ftmin == Real(500.0)) << "slot " << slot;
        EXPECT_EQ(off[slot].U_max_ftmin, UNCAPPED) << "slot " << slot;
        EXPECT_NEAR(off[slot].R0, on[slot].R0, TOL * std::max(Real(1.0), on[slot].R0)) << "slot " << slot;
    }
}

TEST(WindLimit, Behave)
{
    const FuelModelParams fp = get_fuel_params(1, 0);
    const BehaveState bs_default = compute_behave_state(fp, M_DEAD, M_DEAD, M_DEAD, M_LIVE, M_LIVE,
                                                        TRANSFER_LO, TRANSFER_HI);
    const BehaveState bs_off = compute_behave_state(fp, M_DEAD, M_DEAD, M_DEAD, M_LIVE, M_LIVE,
                                                    TRANSFER_LO, TRANSFER_HI, false);

    EXPECT_EQ(bs_default.U_max_ftmin, Real(300.0));
    EXPECT_EQ(bs_off.U_max_ftmin, UNCAPPED);
    expect_close(bs_off.r_0, bs_default.r_0);

    const Real U_low = Real(1.0), U_high = Real(3.0), U_mews = Real(300.0) / FTMIN_PER_MS;
    expect_close(behave_ros_cell(U_low, 0.0, 0.0, 0.0, bs_off), behave_ros_cell(U_low, 0.0, 0.0, 0.0, bs_default));
    expect_close(behave_ros_cell(U_high, 0.0, 0.0, 0.0, bs_default), behave_ros_cell(U_mews, 0.0, 0.0, 0.0, bs_off));
    expect_close(behave_ros_cell(U_high, 0.0, 0.0, 0.0, bs_off),
                 bs_off.r_0 * (Real(1.0) + bs_off.C * std::pow(U_high * FTMIN_PER_MS, bs_off.B) * bs_off.beta_ratio_E));
    EXPECT_GT(behave_ros_cell(U_high, 0.0, 0.0, 0.0, bs_off), behave_ros_cell(U_high, 0.0, 0.0, 0.0, bs_default));
}
