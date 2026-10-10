#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <AMReX_REAL.H>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

#include "ERF_Rothermel.H"
#include "ERF_FuelModels.H"
#include "ERF_CustomFuel.H"

/**
 * @file ERF_GTestReactionVelocityFormula.cpp
 * @brief erf.fire.reaction_velocity_formula ("albini" default | "rothermel")
 *        and erf.fire.wrf_bmst_compat (bool, default false): two independent
 *        WRF-Fire-matching knobs threaded through compute_rothermel_params()
 *        and build_fuel_rothermel_table().
 */

using amrex::Real;

static constexpr Real M_DEAD = 0.055;  ///< Coen et al. (2013) moisture, matches ERF_GTestWindLimit.cpp

/// Closeness at the build precision, absolute for values below one.
static void expect_close(Real a, Real b)
{
    EXPECT_NEAR(a, b, TOL * std::max(Real(1.0), std::abs(b)));
}

/// Relative closeness at the build precision, for small values such as the
/// packing ratio (about 1e-3), where the absolute form above would accept a
/// 1% error in single precision.
static void expect_rel_close(Real a, Real b)
{
    EXPECT_NEAR(a, b, TOL * std::abs(b));
}

/// Independent re-evaluation of Eq. 38's A coefficient, both published forms.
static Real albini_A(Real sigma) { return Real(133.0) * std::pow(sigma, Real(-0.7913)); }
static Real rothermel_A(Real sigma) { return Real(1.0) / (Real(4.774) * std::pow(sigma, Real(0.1)) - Real(7.27)); }

// ---------------------------------------------------------------------------
// erf.fire.reaction_velocity_formula
// ---------------------------------------------------------------------------

TEST(ReactionVelocityFormula, DefaultMatchesExplicitAlbini)
{
    const FuelModelParams fp = get_fuel_params(1, 0);  // FM1, sigma_d1 = 3500
    const RothermelComputed rc_default = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD);
    const RothermelComputed rc_albini  = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true, false);

    expect_close(rc_default.R0, rc_albini.R0);
    expect_close(rc_default.I_R, rc_albini.I_R);
}

TEST(ReactionVelocityFormula, OnlyChangesReactionPath)
{
    // A only enters Gamma_prime -> I_R -> R0 (Eq. 1); it has no business
    // touching the wind-factor, slope-factor or packing-ratio fields.
    const FuelModelParams fp = get_fuel_params(1, 0);
    const RothermelComputed rc_albini    = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true, false);
    const RothermelComputed rc_rothermel = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true, true);

    expect_close(rc_albini.beta, rc_rothermel.beta);
    expect_close(rc_albini.C, rc_rothermel.C);
    expect_close(rc_albini.B, rc_rothermel.B);
    expect_close(rc_albini.phi_s_const, rc_rothermel.phi_s_const);
    EXPECT_EQ(rc_albini.U_max_ftmin, rc_rothermel.U_max_ftmin);
    expect_close(rc_albini.wind_conv, rc_rothermel.wind_conv);
    expect_close(rc_albini.ros_conv, rc_rothermel.ros_conv);

    // The reaction path itself must actually differ for this fuel model.
    EXPECT_NE(rc_albini.I_R, rc_rothermel.I_R);
    EXPECT_NE(rc_albini.R0, rc_rothermel.R0);
}

TEST(ReactionVelocityFormula, MagnitudeDifferenceMatchesPublishedFormulas)
{
    // FM1 (sigma_d1 = 3500 ft^-1): the two forms of A differ by about 36%
    // here, and that difference must reach I_R and R0 by exactly the ratio
    // predicted from A and beta_ratio alone -- everything else in Eq. 38
    // (Gamma_max, w_n, heat_content, eta_M, eta_s) is common to both and
    // cancels.
    const FuelModelParams fp = get_fuel_params(1, 0);
    const RothermelComputed rc_albini    = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true, false);
    const RothermelComputed rc_rothermel = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true, true);

    const Real sigma = amrex::max(fp.sigma_d1, Real(100.0));
    const Real A_a = albini_A(sigma);
    const Real A_r = rothermel_A(sigma);
    // Hand values at sigma = 3500: 133 * 3500^-0.7913 = 0.2087 (Albini 1976)
    // and 1 / (4.774 * 3500^0.1 - 7.27) = 0.2836 (Rothermel 1972, Eq. 39;
    // WRF-Fire module_fr_fire_phys.F uses the same constants).
    EXPECT_NEAR(A_a, Real(0.2087), 1.0e-4);
    EXPECT_NEAR(A_r, Real(0.2836), 1.0e-4);
    EXPECT_GT(A_r, A_a);

    const Real beta_op = Real(3.348) * std::pow(sigma, Real(-0.8189));  // Eq. 37
    const Real beta_ratio = rc_albini.beta / beta_op;                   // beta unaffected by the flag

    const Real predicted_ratio = std::pow(beta_ratio, A_r - A_a) * std::exp((A_r - A_a) * (Real(1.0) - beta_ratio));
    expect_close(rc_rothermel.I_R / rc_albini.I_R, predicted_ratio);
    expect_close(rc_rothermel.R0 / rc_albini.R0, predicted_ratio);
}

TEST(ReactionVelocityFormula, LowSigmaGuardPreventsWrfPole)
{
    // WRF-Fire's original form has a real pole at sigma ~= (7.27/4.774)^10
    // ~= 67.1 ft^-1; below it A goes negative/blows up. Anderson-13 fuel
    // models never get near this, but a coarse synthetic fuel (a 1000-hr
    // log-type SAV) could -- compute_rothermel_params's minimum-SAV guard
    // (sigma clamped to >= 100) must keep every call finite regardless.
    FuelModelParams fp = get_fuel_params(1, 0);
    fp.sigma_d1 = Real(50.0);  // below the pole; clamp must intervene

    const RothermelComputed rc = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true, true);

    EXPECT_TRUE(std::isfinite(rc.R0));
    EXPECT_TRUE(std::isfinite(rc.I_R));
    EXPECT_GT(rc.R0, Real(0.0));
    EXPECT_GT(rc.I_R, Real(0.0));

    // Confirm the guard actually clamped to 100 (not just that the result
    // happens to be finite): C/B are sigma-only, so they pin down which
    // sigma was actually used.
    const Real C_clamped = Real(7.47) * std::exp(Real(-0.133) * std::pow(Real(100.0), Real(0.55)));
    const Real B_clamped = Real(0.02526) * std::pow(Real(100.0), Real(0.54));
    expect_close(rc.C, C_clamped);
    expect_close(rc.B, B_clamped);
}

TEST(ReactionVelocityFormula, PerFuelTable)
{
    const auto albini    = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD, 0, -1.0, true, nullptr, false);
    const auto rothermel = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD, 0, -1.0, true, nullptr, true);
    ASSERT_EQ(albini.size(), rothermel.size());

    EXPECT_EQ(albini[0].R0, Real(0.0));  // non-burnable slot stays zero
    // The published sets: the I_R ratio of every slot is the closed-form
    // ratio of its own A values. Seven Scott-Burgan fuels have
    // sigma = 1800 ft^-1, next to where the two forms cross (about 1800.5),
    // and GS4 also sits near beta_op, so its ratio is 1 - 9e-7: a float
    // cannot tell the two apart there, and EXPECT_NE is kept for the slots
    // whose predicted ratio is far from 1 at the build precision.
    const auto fp_tbl = build_fuel_params_slot_table(0, Real(-1.0), CustomFuelTable{});
    int n_distinct = 0;
    for (int slot = 1; slot < FUEL_SLOT_CUSTOM_BASE; ++slot) {
        SCOPED_TRACE(::testing::Message() << "slot " << slot);
        expect_rel_close(albini[slot].beta, rothermel[slot].beta);
        const Real sigma = amrex::max(fp_tbl[slot].sigma_d1, Real(100.0));
        const Real dA = rothermel_A(sigma) - albini_A(sigma);
        const Real br = albini[slot].beta / (Real(3.348) * std::pow(sigma, Real(-0.8189)));
        const Real predicted = std::pow(br, dA) * std::exp(dA * (Real(1.0) - br));
        EXPECT_NEAR(rothermel[slot].I_R / albini[slot].I_R, predicted, Real(100.0) * TOL);
        if (std::abs(predicted - Real(1.0)) > Real(1000.0) * TOL) {
            EXPECT_NE(albini[slot].I_R, rothermel[slot].I_R);
            ++n_distinct;
        }
    }
    EXPECT_GT(n_distinct, 40);  // all but the fuels next to the crossing
    // The deck-defined slots with no slot table handed in stay non-burnable.
    for (int slot = FUEL_SLOT_CUSTOM_BASE; slot < FUEL_SLOT_COUNT; ++slot) {
        SCOPED_TRACE(::testing::Message() << "slot " << slot);
        EXPECT_EQ(albini[slot].R0, Real(0.0));
        EXPECT_EQ(rothermel[slot].R0, Real(0.0));
    }
}

// The slot table is how deck-defined (custom) fuels reach the coefficients:
// both options must act on a custom slot exactly as on the same fuel computed
// directly. Before the slot table existed the loop above covered every slot;
// this guards the custom path the merge with the slot table added.
TEST(ReactionVelocityFormula, CustomSlotGetsBothOptions)
{
    auto tbl = build_fuel_params_slot_table(0, Real(-1.0), CustomFuelTable{});
    const FuelModelParams fm1 = get_fuel_params(1, 0);
    tbl[FUEL_SLOT_CUSTOM_BASE] = fm1;   // a deck-defined fuel with FM1's properties

    for (bool a_rothermel : {false, true}) {
        for (bool bmst : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "rothermel A " << a_rothermel << ", bmst " << bmst);
            const auto table = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD, 0, -1.0, true,
                                                          tbl.data(), a_rothermel, bmst);
            const RothermelComputed direct = compute_rothermel_params(fm1, M_DEAD, M_DEAD, M_DEAD,
                                                                      true, a_rothermel, bmst);
            EXPECT_EQ(table[FUEL_SLOT_CUSTOM_BASE].R0,   direct.R0);
            EXPECT_EQ(table[FUEL_SLOT_CUSTOM_BASE].I_R,  direct.I_R);
            EXPECT_EQ(table[FUEL_SLOT_CUSTOM_BASE].beta, direct.beta);
        }
    }
    // and the options do change that slot
    const auto plain = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD, 0, -1.0, true,
                                                  tbl.data(), false, false);
    const auto both  = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD, 0, -1.0, true,
                                                  tbl.data(), true, true);
    EXPECT_NE(plain[FUEL_SLOT_CUSTOM_BASE].I_R, both[FUEL_SLOT_CUSTOM_BASE].I_R);
    EXPECT_NE(plain[FUEL_SLOT_CUSTOM_BASE].beta, both[FUEL_SLOT_CUSTOM_BASE].beta);
}

// ---------------------------------------------------------------------------
// erf.fire.wrf_bmst_compat
// ---------------------------------------------------------------------------

TEST(WrfBmstCompat, DefaultOffMatchesExplicitFalse)
{
    const FuelModelParams fp = get_fuel_params(1, 0);
    const RothermelComputed rc_default = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD);
    const RothermelComputed rc_off = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true, false, false);

    expect_close(rc_default.R0, rc_off.R0);
    expect_close(rc_default.beta, rc_off.beta);
}

TEST(WrfBmstCompat, ZeroMoistureIsExactNoOp)
{
    // bmst = M_f/(1+M_f) is identically zero at zero fuel moisture, so the
    // deflation must vanish and both settings must agree bit-for-bit.
    const FuelModelParams fp = get_fuel_params(1, 0);
    const RothermelComputed rc_off = compute_rothermel_params(fp, 0.0, 0.0, 0.0, true, false, false);
    const RothermelComputed rc_on  = compute_rothermel_params(fp, 0.0, 0.0, 0.0, true, false, true);

    EXPECT_EQ(rc_off.beta, rc_on.beta);
    EXPECT_EQ(rc_off.I_R, rc_on.I_R);
    EXPECT_EQ(rc_off.R0, rc_on.R0);
}

TEST(WrfBmstCompat, DeflatesPackingRatioByExactBmstFactor)
{
    // FM1 has only w_d1 nonzero, so the weighted dead moisture M_f collapses
    // to moisture_1hr exactly, and beta = rho_b/rho_p = (w_0/delta)/rho_p is
    // linear in w_0 -- so the deflation must show up as an exact
    // multiplicative factor on beta, independent of every downstream
    // nonlinearity (Gamma_prime, I_R, R0).
    const FuelModelParams fp = get_fuel_params(1, 0);
    const RothermelComputed rc_off = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true, false, false);
    const RothermelComputed rc_on  = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true, false, true);

    const Real bmst = M_DEAD / (Real(1.0) + M_DEAD);
    expect_rel_close(rc_on.beta, rc_off.beta * (Real(1.0) - bmst));

    // Less fuel available to burn -> weaker reaction intensity -> slower R0.
    EXPECT_LT(rc_on.I_R, rc_off.I_R);
    EXPECT_LT(rc_on.R0, rc_off.R0);
}

TEST(WrfBmstCompat, IndependentOfReactionVelocityFormula)
{
    // The two new knobs are documented as independent; wrf_bmst_compat's
    // exact beta-deflation factor must hold the same way regardless of
    // which A formula reaction_velocity_formula selects.
    const FuelModelParams fp = get_fuel_params(1, 0);
    const RothermelComputed rc_off = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true, true, false);
    const RothermelComputed rc_on  = compute_rothermel_params(fp, M_DEAD, M_DEAD, M_DEAD, true, true, true);

    const Real bmst = M_DEAD / (Real(1.0) + M_DEAD);
    expect_rel_close(rc_on.beta, rc_off.beta * (Real(1.0) - bmst));
}

TEST(WrfBmstCompat, PerFuelTable)
{
    // Every slot here is built from the same (uniform) moisture triple, so
    // M_f -- and therefore bmst = M_f/(1+M_f) -- is identical across the
    // whole table regardless of each fuel model's own w_d1/w_d10/w_d100
    // mix; beta = rho_b/rho_p is linear in w_0, so its (1-bmst) deflation is
    // exact and universal. R0 is *not*: it also carries rho_b in its own
    // denominator (Eq. 1), and for some fuel models that "less dense burns
    // faster" effect outweighs the weaker reaction intensity -- so R0 can
    // go either direction and must not be asserted monotonic here (see
    // DeflatesPackingRatioByExactBmstFactor for the FM1 case where it does
    // decrease).
    const auto off = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD, 0, -1.0, true, nullptr, false, false);
    const auto on  = build_fuel_rothermel_table(M_DEAD, M_DEAD, M_DEAD, 0, -1.0, true, nullptr, false, true);
    ASSERT_EQ(off.size(), on.size());

    const Real bmst = M_DEAD / (Real(1.0) + M_DEAD);
    EXPECT_EQ(off[0].R0, Real(0.0));  // non-burnable slot stays zero
    for (int slot = 1; slot < FUEL_SLOT_CUSTOM_BASE; ++slot) {
        SCOPED_TRACE(::testing::Message() << "slot " << slot);
        expect_rel_close(on[slot].beta, off[slot].beta * (Real(1.0) - bmst));
        EXPECT_NE(on[slot].R0, off[slot].R0);
    }
    // deck-defined slots with no slot table: non-burnable either way
    for (int slot = FUEL_SLOT_CUSTOM_BASE; slot < FUEL_SLOT_COUNT; ++slot) {
        SCOPED_TRACE(::testing::Message() << "slot " << slot);
        EXPECT_EQ(on[slot].R0, Real(0.0));
        EXPECT_EQ(off[slot].R0, Real(0.0));
    }
}
