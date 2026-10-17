// The moving bodies' timing summary (erf.moving_bodies.timing_int): a timed phase records the time it
// takes only while the timers are on, into its own phase alone; the summary gives each phase's minimum,
// mean (the sum over the ranks over their number) and maximum, leaves out a phase no rank spent time in,
// and names the fastest and the slowest turbine of the OpenFAST step.

#include <array>
#include <sstream>
#include <string>
#include <vector>

#include <AMReX_ParallelDescriptor.H>

#include <gtest/gtest.h>

#include "ERF_PhaseTimers.H"

using erf_actuator::PhaseTimers;
using erf_actuator::ScopedPhase;

namespace {

// wait until the clock has moved on, so that a timed phase takes a time greater than zero
void spin ()
{
    const double t0 = amrex::ParallelDescriptor::second();
    while (amrex::ParallelDescriptor::second() - t0 < 1.0e-4) {}
}

// the row of phase p in a summary: its three numbers, or false when the summary has no such row
bool summary_row (const std::string& text, const std::string& p, std::array<double,3>& v)
{
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string name;
        if ((ls >> name) && name == p) { return static_cast<bool>(ls >> v[0] >> v[1] >> v[2]); }
    }
    return false;
}

} // namespace

TEST(PhaseTimers, APhaseIsTimedOnlyWhenTheTimersAreOn)
{
    PhaseTimers t;
    { ScopedPhase s(t, PhaseTimers::Spread); spin(); }
    for (double s : t.seconds()) { EXPECT_EQ(s, 0.0); }

    t.enable(true);
    { ScopedPhase s(t, PhaseTimers::Spread); spin(); }
    { ScopedPhase s(t, PhaseTimers::Spread); spin(); }
    for (int p = 0; p < PhaseTimers::NumPhases; ++p) {
        if (p == PhaseTimers::Spread) {
            EXPECT_GE(t.seconds()[p], 2.0e-4);
        } else {
            EXPECT_EQ(t.seconds()[p], 0.0) << PhaseTimers::name(p);
        }
    }
    t.reset();
    for (double s : t.seconds()) { EXPECT_EQ(s, 0.0); }
}

TEST(PhaseTimers, TheSummaryGivesEachPhasesMinimumMeanAndMaximumOverTheRanks)
{
    // two ranks: sample took 0.1 s on one and 0.3 s on the other, spread 0.5 s on both; no rank checkpointed
    std::array<double, PhaseTimers::NumPhases> mn{}, sum{}, mx{};
    mn[PhaseTimers::Sample] = 0.1; sum[PhaseTimers::Sample] = 0.4; mx[PhaseTimers::Sample] = 0.3;
    mn[PhaseTimers::Spread] = 0.5; sum[PhaseTimers::Spread] = 1.0; mx[PhaseTimers::Spread] = 0.5;
    const std::string text = PhaseTimers::summary(11, 20, 2, mn.data(), sum.data(), mx.data());
    EXPECT_NE(text.find("Moving bodies timing, steps 11 to 20 (10 steps)"), std::string::npos) << text;
    EXPECT_NE(text.find("over 2 ranks"), std::string::npos) << text;
    std::array<double,3> v{};
    ASSERT_TRUE(summary_row(text, "sample", v)) << text;
    EXPECT_DOUBLE_EQ(v[0], 0.1);
    EXPECT_DOUBLE_EQ(v[1], 0.2);
    EXPECT_DOUBLE_EQ(v[2], 0.3);
    ASSERT_TRUE(summary_row(text, "spread", v)) << text;
    EXPECT_DOUBLE_EQ(v[1], 0.5);
    EXPECT_FALSE(summary_row(text, "checkpoint", v)) << text;
    EXPECT_FALSE(summary_row(text, "openfast", v)) << text;

    // one step on one rank
    const std::string one = PhaseTimers::summary(3, 3, 1, mn.data(), sum.data(), mx.data());
    EXPECT_NE(one.find("steps 3 to 3 (1 step)"), std::string::npos) << one;
    EXPECT_NE(one.find("over 1 rank\n"), std::string::npos) << one;
}

TEST(PhaseTimers, TheTurbineLineNamesTheFastestAndTheSlowestTurbine)
{
    const std::string text = PhaseTimers::turbine_summary({"T1", "T2", "T3"}, {0.2, 0.1, 0.3});
    EXPECT_NE(text.find("(3 turbines): min 1.000e-01 (T2), mean 2.000e-01, max 3.000e-01 (T3)"), std::string::npos) << text;
    EXPECT_TRUE(PhaseTimers::turbine_summary({}, {}).empty());
}
