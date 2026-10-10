#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>
#include <cmath>

#include "ERF_FuelModels.H"
#include "ERF_Rothermel.H"

/**
 * @file ERF_GTestRothermelCellMoisture.cpp
 * @brief With erf.fire.rothermel_cell_moisture (the default under
 *        moisture_dynamic) the Rothermel kernels read a coefficient field
 *        built once per step from each cell's own dead moistures; before,
 *        every cell took the coefficients of the domain-mean moisture. The
 *        cell's own fuel enters that field only under rothermel_per_fuel; a
 *        fuel map without it keeps the uniform fuel, as the one-set path does.
 */

using namespace amrex;

namespace {
constexpr double REL = (sizeof(Real) == 8) ? 1.0e-12 : 1.0e-5;

struct TwoCells
{
    BoxArray ba; DistributionMapping dm;
    TwoCells () { ba = BoxArray(Box(IntVect(0, 0, 0), IntVect(1, 0, 0))); dm = DistributionMapping(ba); }
};

Real cell (const MultiFab& mf, int i)
{
    Real v = -1.0;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) { v = mf.const_array(mfi)(i, 0, 0); }
    return v;
}

/// moisture field: cell 0 at 6/7/8 %, cell 1 at 20/21/22 %
void set_moistures (MultiFab& mc)
{
    mc.setVal(0.0_rt);
    for (MFIter mfi(mc); mfi.isValid(); ++mfi) {
        auto m = mc.array(mfi);
        m(0, 0, 0, 0) = 0.06_rt; m(0, 0, 0, 1) = 0.07_rt; m(0, 0, 0, 2) = 0.08_rt;
        m(1, 0, 0, 0) = 0.20_rt; m(1, 0, 0, 1) = 0.21_rt; m(1, 0, 0, 2) = 0.22_rt;
    }
}
}

TEST(RothermelCellMoisture, PackAndUnpackRoundTrip)
{
    TwoCells g;
    MultiFab rcc(g.ba, g.dm, ROTHERMEL_RC_NCOMP, 0);
    const RothermelComputed rc = compute_rothermel_params(get_fuel_params(10, FUEL_SET_ANDERSON13), 0.06_rt, 0.07_rt, 0.08_rt);
    for (MFIter mfi(rcc); mfi.isValid(); ++mfi) {
        auto a = rcc.array(mfi);
        pack_rothermel(rc, a, 1, 0);
        const RothermelComputed back = unpack_rothermel(rcc.const_array(mfi), 1, 0);
        EXPECT_EQ(back.R0, rc.R0); EXPECT_EQ(back.C, rc.C); EXPECT_EQ(back.B, rc.B);
        EXPECT_EQ(back.beta_ratio_E, rc.beta_ratio_E); EXPECT_EQ(back.beta, rc.beta);
        EXPECT_EQ(back.phi_s_const, rc.phi_s_const); EXPECT_EQ(back.U_max_ftmin, rc.U_max_ftmin);
        EXPECT_EQ(back.wind_conv, rc.wind_conv); EXPECT_EQ(back.ros_conv, rc.ros_conv); EXPECT_EQ(back.I_R, rc.I_R);
    }
}

TEST(RothermelCellMoisture, EachCellTakesItsOwnMoisture)
{
    TwoCells g;
    const FuelModelParams fp10 = get_fuel_params(10, FUEL_SET_ANDERSON13);
    MultiFab ros(g.ba, g.dm, 1, 0), wind(g.ba, g.dm, 2, 0), slopes(g.ba, g.dm, 2, 0), mc(g.ba, g.dm, 5, 0);
    MultiFab rcc(g.ba, g.dm, ROTHERMEL_RC_NCOMP, 0);
    wind.setVal(2.0_rt, 0, 1); wind.setVal(0.0_rt, 1, 1);
    slopes.setVal(0.0_rt);
    set_moistures(mc);
    const RothermelComputed rc_mean = compute_rothermel_params(fp10, 0.06_rt, 0.07_rt, 0.08_rt);

    // the mean coefficients: both cells alike
    compute_ros_field(ros, wind, slopes, rc_mean, nullptr, nullptr, 0, FUEL_SET_ANDERSON13, nullptr);
    const Real r_mean = rothermel_ros_cell(2.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, rc_mean);
    EXPECT_NEAR(cell(ros, 0), r_mean, REL * r_mean);
    EXPECT_NEAR(cell(ros, 1), r_mean, REL * r_mean);

    // per-cell: the wet cell carries its coefficients at 20/21/22 %
    build_cell_rothermel_coefficients(rcc, mc, nullptr, nullptr, 0, FUEL_SET_ANDERSON13, -1.0_rt, fp10,
                                      true, fire_wind_limit::rothermel);
    RothermelCellInputs cin;
    cin.rc_cell = &rcc;
    compute_ros_field(ros, wind, slopes, rc_mean, nullptr, nullptr, 0, FUEL_SET_ANDERSON13, &cin);
    const RothermelComputed rc_wet = rothermel_coefficients(fp10, 0.20_rt, 0.21_rt, 0.22_rt, true, fire_wind_limit::rothermel);
    const Real r_wet = rothermel_ros_cell(2.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, rc_wet);
    EXPECT_NEAR(cell(ros, 0), r_mean, REL * r_mean);
    EXPECT_NEAR(cell(ros, 1), r_wet,  REL * r_wet);
    EXPECT_LT(r_wet, 0.5 * r_mean) << "FM10 at 20 % spreads less than half as fast as at 6 %";
}

TEST(RothermelCellMoisture, TheFuelMapEntersOnlyUnderPerFuel)
{
    TwoCells g;
    MultiFab ros(g.ba, g.dm, 1, 0), wind(g.ba, g.dm, 2, 0), slopes(g.ba, g.dm, 2, 0), mc(g.ba, g.dm, 5, 0), fuel(g.ba, g.dm, 1, 0);
    MultiFab rcc(g.ba, g.dm, ROTHERMEL_RC_NCOMP, 0);
    wind.setVal(2.0_rt, 0, 1); wind.setVal(0.0_rt, 1, 1);
    slopes.setVal(0.0_rt);
    mc.setVal(0.06_rt, 0, 1); mc.setVal(0.07_rt, 1, 1); mc.setVal(0.08_rt, 2, 1); mc.setVal(0.0_rt, 3, 2);
    for (MFIter mfi(fuel); mfi.isValid(); ++mfi) {
        auto f = fuel.array(mfi);
        f(0, 0, 0) = 1.0_rt;    // short grass
        f(1, 0, 0) = 10.0_rt;   // timber litter and understory
    }
    const FuelModelParams fp1 = get_fuel_params(1, FUEL_SET_ANDERSON13);
    const RothermelComputed rc_default = compute_rothermel_params(fp1, 0.06_rt, 0.07_rt, 0.08_rt);
    RothermelCellInputs cin;
    cin.rc_cell = &rcc;

    // rothermel_per_fuel: the field takes each cell's code
    build_cell_rothermel_coefficients(rcc, mc, &fuel, nullptr, 0, FUEL_SET_ANDERSON13, -1.0_rt, fp1,
                                      true, fire_wind_limit::rothermel);
    compute_ros_field(ros, wind, slopes, rc_default, nullptr, nullptr, 0, FUEL_SET_ANDERSON13, &cin);
    for (int code : {1, 10}) {
        const RothermelComputed rc = rothermel_coefficients(get_fuel_params(code, FUEL_SET_ANDERSON13),
                                                            0.06_rt, 0.07_rt, 0.08_rt, true, fire_wind_limit::rothermel);
        const Real r = rothermel_ros_cell(2.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, rc);
        EXPECT_NEAR(cell(ros, code == 1 ? 0 : 1), r, REL * r) << "fuel " << code;
    }
    EXPECT_GT(cell(ros, 0), 1.5 * cell(ros, 1)) << "grass runs well ahead of timber litter";

    // without it the map is not read: the uniform fuel in both cells, as the
    // one-coefficient-set path spreads it (a map without rothermel_per_fuel
    // must not change the answer when the moisture field is switched on)
    build_cell_rothermel_coefficients(rcc, mc, nullptr, nullptr, 0, FUEL_SET_ANDERSON13, -1.0_rt, fp1,
                                      true, fire_wind_limit::rothermel);
    compute_ros_field(ros, wind, slopes, rc_default, nullptr, nullptr, 0, FUEL_SET_ANDERSON13, &cin);
    const Real r1 = rothermel_ros_cell(2.0_rt, 0.0_rt, 0.0_rt, 0.0_rt, rc_default);
    EXPECT_NEAR(cell(ros, 0), r1, REL * r1);
    EXPECT_NEAR(cell(ros, 1), r1, REL * r1) << "the timber cell keeps the uniform grass coefficients";
}

TEST(RothermelCellMoisture, ACodeOutsideTheTableKeepsTheUniformFuel)
{
    // the per-fuel table path keeps the uniform entry for a code it does not
    // know; the coefficient field does the same (code 77 is in no table). The
    // uniform fuel is model 10, not the model 1 the Anderson lookup falls
    // through to, so a field that looked the code up would give cell 0 the
    // model-1 coefficients (R0 differs by a factor of about 30) and fail here.
    TwoCells g;
    const FuelModelParams fp1  = get_fuel_params(1, FUEL_SET_ANDERSON13);
    const FuelModelParams fp10 = get_fuel_params(10, FUEL_SET_ANDERSON13);
    MultiFab mc(g.ba, g.dm, 5, 0), rcc(g.ba, g.dm, ROTHERMEL_RC_NCOMP, 0), fuel(g.ba, g.dm, 1, 0);
    mc.setVal(0.06_rt, 0, 1); mc.setVal(0.07_rt, 1, 1); mc.setVal(0.08_rt, 2, 1); mc.setVal(0.0_rt, 3, 2);
    for (MFIter mfi(fuel); mfi.isValid(); ++mfi) { auto f = fuel.array(mfi); f(0, 0, 0) = 77.0_rt; f(1, 0, 0) = 1.0_rt; }
    build_cell_rothermel_coefficients(rcc, mc, &fuel, nullptr, 0, FUEL_SET_ANDERSON13, -1.0_rt, fp10,
                                      true, fire_wind_limit::rothermel);
    const RothermelComputed rc1  = rothermel_coefficients(fp1,  0.06_rt, 0.07_rt, 0.08_rt, true, fire_wind_limit::rothermel);
    const RothermelComputed rc10 = rothermel_coefficients(fp10, 0.06_rt, 0.07_rt, 0.08_rt, true, fire_wind_limit::rothermel);
    ASSERT_GT(std::abs(rc10.R0 - rc1.R0), 0.01 * rc1.R0) << "the two fuels must differ for the check to mean anything (9.6 % in R0)";
    for (MFIter mfi(rcc); mfi.isValid(); ++mfi) {
        const RothermelComputed rc0 = unpack_rothermel(rcc.const_array(mfi), 0, 0);
        const RothermelComputed rc1c = unpack_rothermel(rcc.const_array(mfi), 1, 0);
        EXPECT_NEAR(rc0.R0,  rc10.R0, REL * rc10.R0) << "code 77 keeps the uniform model 10";
        EXPECT_NEAR(rc1c.R0, rc1.R0,  REL * rc1.R0)  << "code 1 is looked up";
    }
    // the Scott-Burgan set falls through to GR1 (101): the uniform fuel GR4 (104) must stay
    const FuelModelParams fp104 = get_fuel_params(104, FUEL_SET_SCOTT_BURGAN40);
    const FuelModelParams fp101 = get_fuel_params(101, FUEL_SET_SCOTT_BURGAN40);
    build_cell_rothermel_coefficients(rcc, mc, &fuel, nullptr, 0, FUEL_SET_SCOTT_BURGAN40, -1.0_rt, fp104,
                                      true, fire_wind_limit::rothermel);
    const RothermelComputed rc104 = rothermel_coefficients(fp104, 0.06_rt, 0.07_rt, 0.08_rt, true, fire_wind_limit::rothermel);
    const RothermelComputed rc101 = rothermel_coefficients(fp101, 0.06_rt, 0.07_rt, 0.08_rt, true, fire_wind_limit::rothermel);
    ASSERT_GT(std::abs(rc104.R0 - rc101.R0), 0.01 * rc101.R0);
    for (MFIter mfi(rcc); mfi.isValid(); ++mfi) {
        const RothermelComputed rc0 = unpack_rothermel(rcc.const_array(mfi), 0, 0);
        EXPECT_NEAR(rc0.R0, rc104.R0, REL * rc104.R0) << "code 77 keeps the uniform GR4 under Scott-Burgan";
    }
}

TEST(RothermelCellMoisture, MoisturesAreClampedAsTheMeanIs)
{
    TwoCells g;
    const FuelModelParams fp1 = get_fuel_params(1, FUEL_SET_ANDERSON13);
    MultiFab mc(g.ba, g.dm, 5, 0), rcc(g.ba, g.dm, ROTHERMEL_RC_NCOMP, 0);
    mc.setVal(0.0_rt);
    mc.setVal(0.001_rt, 0, 3);                                   // below the 1 % floor
    build_cell_rothermel_coefficients(rcc, mc, nullptr, nullptr, 0, FUEL_SET_ANDERSON13, -1.0_rt, fp1,
                                      true, fire_wind_limit::rothermel);
    const RothermelComputed rc_floor = rothermel_coefficients(fp1, 0.01_rt, 0.01_rt, 0.01_rt, true, fire_wind_limit::rothermel);
    for (MFIter mfi(rcc); mfi.isValid(); ++mfi) {
        const RothermelComputed rc = unpack_rothermel(rcc.const_array(mfi), 0, 0);
        EXPECT_NEAR(rc.R0, rc_floor.R0, REL * rc_floor.R0);
    }
}

/**
 * The per-cell coefficients (erf.fire.rothermel_cell_moisture, on by default
 * with moisture_dynamic) honour erf.fire.reaction_velocity_formula and
 * erf.fire.wrf_bmst_compat (#439) as the one-coefficient-set path does: the
 * wet cell's packed coefficients are those of rothermel_coefficients() with
 * both options at its own moistures, and differ from the build without them.
 */
TEST(RothermelCellMoisture, TheCoefficientOptionsReachEveryCell)
{
    TwoCells g;
    const FuelModelParams fp10 = get_fuel_params(10, FUEL_SET_ANDERSON13);
    MultiFab mc(g.ba, g.dm, 5, 0), rc_on(g.ba, g.dm, ROTHERMEL_RC_NCOMP, 0), rc_off(g.ba, g.dm, ROTHERMEL_RC_NCOMP, 0);
    set_moistures(mc);
    build_cell_rothermel_coefficients(rc_on, mc, nullptr, nullptr, 0, FUEL_SET_ANDERSON13, -1.0_rt, fp10,
                                      true, fire_wind_limit::rothermel, /*rothermel A*/ true, /*bmst*/ true);
    build_cell_rothermel_coefficients(rc_off, mc, nullptr, nullptr, 0, FUEL_SET_ANDERSON13, -1.0_rt, fp10,
                                      true, fire_wind_limit::rothermel);
    const RothermelComputed want = rothermel_coefficients(fp10, 0.20_rt, 0.21_rt, 0.22_rt, true,
                                                          fire_wind_limit::rothermel, true, true);
    for (MFIter mfi(rc_on); mfi.isValid(); ++mfi) {
        const RothermelComputed on  = unpack_rothermel(rc_on.const_array(mfi), 1, 0);
        const RothermelComputed off = unpack_rothermel(rc_off.const_array(mfi), 1, 0);
        EXPECT_EQ(on.I_R, want.I_R);
        EXPECT_EQ(on.R0, want.R0);
        EXPECT_EQ(on.beta, want.beta);
        EXPECT_NE(on.I_R, off.I_R) << "the options change the per-cell coefficients";
        EXPECT_NE(on.beta, off.beta) << "wrf_bmst_compat deflates the load, so the packing ratio";
    }
}
