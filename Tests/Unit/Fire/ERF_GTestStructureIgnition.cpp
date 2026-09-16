#include <gtest/gtest.h>
#include <cmath>
#include <AMReX_REAL.H>
#include <AMReX_Math.H>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

#include "ERF_StructureIgnition.H"

using namespace erf_structure_ignition;

/**
 * @file ERF_GTestStructureIgnition.cpp
 * @brief The EN 1991-1-2 burn curve (shape, energy budget, rejected inputs),
 *        the ground point-source radiation kernel and the ignition rule.
 */

TEST(StructureIgnition, BurnCurveShapeAndEnergy)
{
    // The documented defaults: 250 kW/m2 peak, 780 MJ/m2, 600 s to the peak.
    BurnCurve c;
    ASSERT_TRUE(make_burn_curve(2.5e5, 7.8e8, 600.0, c));
    EXPECT_NEAR(c.q_peak, 2.5e5, TOL * 2.5e5);
    EXPECT_NEAR(c.t_growth, 600.0, TOL);
    // growth releases q t/3 = 50 MJ/m2, the plateau 70 % - 50 = 496 MJ/m2, the decay 30 %
    EXPECT_NEAR(c.t_plateau, (0.7 * 7.8e8 - 2.5e5 * 600.0 / 3.0) / 2.5e5, 1e3 * TOL);
    EXPECT_NEAR(c.t_decay, 2.0 * 0.3 * 7.8e8 / 2.5e5, 1e3 * TOL);

    // zero before ignition, quadratic growth, the peak at t_growth, the plateau, zero after
    EXPECT_NEAR(c.flux_at(-1.0), 0.0, TOL);
    EXPECT_NEAR(c.flux_at(300.0), 2.5e5 * 0.25, TOL * 2.5e5);
    EXPECT_NEAR(c.flux_at(600.0), 2.5e5, TOL * 2.5e5);
    EXPECT_NEAR(c.flux_at(600.0 + 0.5 * c.t_plateau), 2.5e5, TOL * 2.5e5);
    EXPECT_NEAR(c.flux_at(c.duration() - 0.5 * c.t_decay), 0.5 * 2.5e5, 1e3 * TOL * 2.5e5);
    EXPECT_NEAR(c.flux_at(c.duration()), 0.0, TOL);
    EXPECT_NEAR(c.flux_at(c.duration() + 1000.0), 0.0, TOL);

    // The integral of the curve is the fuel load (midpoint rule, fine steps).
    const int n = 200000;
    const amrex::Real h = c.duration() / n;
    double e = 0.0;
    for (int k = 0; k < n; ++k) { e += c.flux_at((k + 0.5) * h) * h; }
    EXPECT_NEAR(e / 7.8e8, 1.0, 1e-6);
}

TEST(StructureIgnition, BurnCurveInstantGrowth)
{
    // growth_time_s = 0: the peak from the first instant, no division by zero
    BurnCurve c;
    ASSERT_TRUE(make_burn_curve(1.0e5, 1.0e8, 0.0, c));
    EXPECT_NEAR(c.flux_at(0.0), 1.0e5, TOL * 1.0e5);
    EXPECT_NEAR(c.t_plateau, 0.7e8 / 1.0e5, 1e3 * TOL);
    EXPECT_NEAR(c.t_decay, 0.6e8 / 1.0e5, 1e3 * TOL);
}

TEST(StructureIgnition, BurnCurveRejectsImpossibleInputs)
{
    BurnCurve c;
    EXPECT_FALSE(make_burn_curve(0.0, 1.0e8, 100.0, c));       // no peak
    EXPECT_FALSE(make_burn_curve(1.0e5, 0.0, 100.0, c));       // no fuel
    EXPECT_FALSE(make_burn_curve(1.0e5, 1.0e8, -1.0, c));      // negative growth
    // growth alone releases q t / 3 = 1e5 * 3000 / 3 = 1e8 > 0.7e8
    EXPECT_FALSE(make_burn_curve(1.0e5, 1.0e8, 3000.0, c));
    // right at the limit: growth releases exactly 70 %, so no plateau
    ASSERT_TRUE(make_burn_curve(1.0e5, 1.0e8, 2100.0, c));
    EXPECT_NEAR(c.t_plateau, 0.0, 1e3 * TOL);
}

TEST(StructureIgnition, PointSourceFlux)
{
    // P / (2 pi r^2) over the hemisphere; the floor keeps the source cell finite
    const amrex::Real P = 1.0e6;
    EXPECT_NEAR(point_source_incident_flux(P, 100.0, 6.25), P / (2.0 * amrex::Math::pi<double>() * 100.0), 1e3 * TOL);
    EXPECT_NEAR(point_source_incident_flux(P, 0.0, 6.25),   P / (2.0 * amrex::Math::pi<double>() * 6.25),  1e3 * TOL);
    // inverse square: four times the distance, a sixteenth of the flux
    EXPECT_NEAR(point_source_incident_flux(P, 1600.0, 6.25) * 16.0,
                point_source_incident_flux(P, 100.0, 6.25), 1e3 * TOL);
}

TEST(StructureIgnition, IgnitionRule)
{
    // nothing met
    EXPECT_EQ(ignition_cause(1.0e6, 6.0e6, 3.0, 50, 10.0, 1000.0, 60.0), None);
    // each criterion on its own, in the documented order of precedence
    EXPECT_EQ(ignition_cause(6.0e6, 6.0e6, 0.0, 50, 0.0, 1000.0, 60.0), HeatLoad);
    EXPECT_EQ(ignition_cause(0.0, 6.0e6, 50.0, 50, 0.0, 1000.0, 60.0), Embers);
    EXPECT_EQ(ignition_cause(0.0, 6.0e6, 0.0, 50, 60.0, 1000.0, 60.0), Intensity);
    EXPECT_EQ(ignition_cause(7.0e6, 6.0e6, 80.0, 50, 90.0, 1000.0, 60.0), HeatLoad);
    EXPECT_EQ(ignition_cause(0.0, 6.0e6, 80.0, 50, 90.0, 1000.0, 60.0), Embers);
    // a threshold at or below zero turns its criterion off
    EXPECT_EQ(ignition_cause(1.0e9, 0.0, 0.0, 50, 0.0, 1000.0, 60.0), None);
    EXPECT_EQ(ignition_cause(0.0, 6.0e6, 1.0e4, 0, 0.0, 1000.0, 60.0), None);
    EXPECT_EQ(ignition_cause(0.0, 6.0e6, 0.0, 50, 1.0e4, 0.0, 60.0), None);
    // residence 0 s still needs some time above the threshold
    EXPECT_EQ(ignition_cause(0.0, 6.0e6, 0.0, 50, 0.0, 1000.0, 0.0), None);
    EXPECT_EQ(ignition_cause(0.0, 6.0e6, 0.0, 50, 0.25, 1000.0, 0.0), Intensity);
}

TEST(StructureIgnition, Names)
{
    EXPECT_STREQ(cause_name(HeatLoad), "heat");
    EXPECT_STREQ(cause_name(Embers), "ember");
    EXPECT_STREQ(cause_name(Intensity), "intensity");
    EXPECT_STREQ(cause_name(None), "none");
    EXPECT_STREQ(state_name(Unignited), "unignited");
    EXPECT_STREQ(state_name(Burning), "burning");
    EXPECT_STREQ(state_name(BurnedOut), "burned_out");
}
