#include <gtest/gtest.h>
#include <AMReX_REAL.H>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

#include "ERF_FuelMoisture.H"

/**
 * @file ERF_GTestFuelMoistureLive.cpp
 * @brief The live herbaceous and live woody updates of
 *        erf.fire.moisture_live_model: "fixed" holds the class in any air;
 *        "legacy" is the historical inline update bit for bit under both
 *        equilibrium curves, and that update drops a live moisture to the
 *        dead-fuel clamp at once and then to the live floor. None of the
 *        checks depends on which hysteresis branch the dead-fuel kernel picks.
 */

namespace {
const int EMC_MODELS[] = {FuelMoistureEMC::LEGACY, FuelMoistureEMC::VAN_WAGNER};
}

TEST(FuelMoistureLive, FixedHoldsInAnyAir)
{
    // A day of 5 s steps leaves the live class exactly where it started.
    const amrex::Real dt_h = amrex::Real(5.0 / 3600.0);
    for (int emc : EMC_MODELS) {
        for (amrex::Real M0 : {0.30, 0.60, 0.90, 1.50, 2.50}) {
            for (amrex::Real RH : {0.0, 30.0, 99.0}) {
                for (amrex::Real T_C : {0.0, 20.0, 40.0}) {
                    amrex::Real M = M0;
                    for (int n = 0; n < 17280; ++n) { M = advance_live_fuel_moisture(M, RH, T_C, dt_h, true, emc); }
                    EXPECT_EQ(M, M0) << "emc " << emc << " M0 " << M0 << " RH " << RH << " T " << T_C;
                }
            }
        }
    }
}

TEST(FuelMoistureLive, LegacyIsTheHistoricalUpdate)
{
    // The inline update the fire layer carried before the switch, with the
    // emc_model argument it gained in PR #378.
    for (int emc : EMC_MODELS) {
        for (amrex::Real RH : {0.0, 1.0, 30.0, 70.0, 99.0}) {
            for (amrex::Real T_C : {0.0, 26.85, 40.0}) {
                for (amrex::Real M : {0.20, 0.35, 0.60, 0.90, 2.00, 3.00}) {
                    for (amrex::Real dt_h : {amrex::Real(5.0 / 3600.0), amrex::Real(0.25)}) {
                        const amrex::Real historical = amrex::max(amrex::Real(0.30), amrex::min(
                            advance_fuel_moisture_one_class(M, RH, T_C, amrex::Real(0.0), dt_h,
                                                            FuelMoistureConst::TAU_100HR, emc),
                            amrex::Real(2.50)));
                        EXPECT_EQ(advance_live_fuel_moisture(M, RH, T_C, dt_h, false, emc), historical)
                            << "emc " << emc << " M " << M << " RH " << RH << " T " << T_C << " dt " << dt_h;
                    }
                }
            }
        }
    }
    // The curves default to legacy, as in advance_fuel_moisture_one_class.
    EXPECT_EQ(advance_live_fuel_moisture(0.90, 30.0, 20.0, 0.01, false),
              advance_live_fuel_moisture(0.90, 30.0, 20.0, 0.01, false, FuelMoistureEMC::LEGACY));
}

TEST(FuelMoistureLive, LegacyDropsToTheDeadClampThenTheLiveFloor)
{
    // The defect "fixed" avoids. One-minute steps at 20 C.
    const amrex::Real T_C = 20.0, dt_h = amrex::Real(1.0 / 60.0);
    for (int emc : EMC_MODELS) {
        // Even saturated air cannot keep 0.90: the first step starts from the
        // dead-fuel clamp, so 0.90 and 0.40 give the same result.
        for (amrex::Real RH : {0.0, 30.0, 99.0}) {
            const amrex::Real first = advance_live_fuel_moisture(0.90, RH, T_C, dt_h, false, emc);
            EXPECT_LE(first, amrex::Real(FuelMoistureConst::M_MAX)) << "emc " << emc << " RH " << RH;
            EXPECT_EQ(first, advance_live_fuel_moisture(0.40, RH, T_C, dt_h, false, emc)) << "emc " << emc << " RH " << RH;
        }
        // In 30 % RH both curves sit far below 0.30, so the class falls
        // monotonically and rests on the live floor within 200 h (100-h lag).
        amrex::Real M = 0.90, prev = M;
        for (int n = 0; n < 200 * 60; ++n) {
            M = advance_live_fuel_moisture(M, 30.0, T_C, dt_h, false, emc);
            ASSERT_LE(M, prev) << "emc " << emc << " step " << n;
            prev = M;
        }
        EXPECT_NEAR(M, FuelMoistureConst::LIVE_M_MIN, TOL) << "emc " << emc;
    }
}
