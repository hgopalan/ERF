// Running statistics of a body's diagnostics: mean, root mean square, minimum and maximum of
// the samples since the averaging start, exact for a known sequence, written as a CSV, and
// carried across a checkpoint round trip.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

#include <gtest/gtest.h>

#include "ERF_RunningStats.H"

namespace {
using amrex::Real;
using erf_actuator::RunningStats;
Real tol () { return (sizeof(Real) == 8) ? Real(1.0e-12) : Real(1.0e-5); }
}

TEST(RunningStats, MeanRmsMinMaxOfAKnownSequence)
{
    RunningStats s("T1", "out/T1", {"a", "b"});
    EXPECT_EQ(s.num_samples(), 0);
    EXPECT_EQ(s.mean(0), 0.0);
    // a: 1, 2, 3, 4  -> mean 2.5, rms sqrt(7.5), min 1, max 4; b: constant 7
    for (int k = 1; k <= 4; ++k) { s.accumulate(10.0 * k, {static_cast<Real>(k), Real(7.0)}); }
    EXPECT_EQ(s.num_samples(), 4);
    EXPECT_DOUBLE_EQ(s.t_first(), 10.0);
    EXPECT_DOUBLE_EQ(s.t_last(), 40.0);
    EXPECT_NEAR(s.mean(0), 2.5, tol());
    EXPECT_NEAR(s.rms(0), std::sqrt(7.5), tol());
    EXPECT_EQ(s.min(0), 1.0);
    EXPECT_EQ(s.max(0), 4.0);
    EXPECT_NEAR(s.mean(1), 7.0, tol());
    EXPECT_NEAR(s.rms(1), 7.0, tol());
    EXPECT_EQ(s.min(1), 7.0);
    EXPECT_EQ(s.max(1), 7.0);
}

TEST(RunningStats, StateRoundTripsThroughACheckpointAndTheFileHasTheRows)
{
    const auto dir = std::filesystem::temp_directory_path() / "erf_gtest_stats";
    std::filesystem::create_directories(dir);
    RunningStats a("T1", (dir / "T1").string(), {"thrust", "power"});
    a.accumulate(1.0, {Real(2.0e6), Real(1.0e7)});
    a.accumulate(2.0, {Real(3.0e6), Real(1.5e7)});
    a.write_state(dir.string());
    ASSERT_TRUE(std::filesystem::exists(dir / "T1_stats.dat"));

    RunningStats b("T1", (dir / "T1").string(), {"thrust", "power"});
    EXPECT_TRUE(b.read_state(dir.string()));
    EXPECT_EQ(b.num_samples(), 2);
    EXPECT_EQ(b.mean(0), a.mean(0));
    EXPECT_EQ(b.rms(1), a.rms(1));
    EXPECT_EQ(b.min(0), a.min(0));
    EXPECT_EQ(b.max(1), a.max(1));
    EXPECT_DOUBLE_EQ(b.t_first(), 1.0);
    // a third sample continues the same statistics
    a.accumulate(3.0, {Real(4.0e6), Real(2.0e7)});
    b.accumulate(3.0, {Real(4.0e6), Real(2.0e7)});
    EXPECT_EQ(b.mean(0), a.mean(0));
    EXPECT_EQ(b.rms(0), a.rms(0));
    EXPECT_NEAR(a.mean(0), 3.0e6, tol() * 3.0e6);
    // no state for another body
    RunningStats c("T2", (dir / "T2").string(), {"thrust", "power"});
    EXPECT_FALSE(c.read_state(dir.string()));

    a.write();
    std::ifstream csv(dir / "T1_stats.csv");
    std::string line;
    ASSERT_TRUE(std::getline(csv, line));
    EXPECT_EQ(line, "samples,t_first,t_last,quantity,mean,rms,min,max");
    ASSERT_TRUE(std::getline(csv, line));
    EXPECT_EQ(line.rfind("3,1,3,thrust,3000000,", 0), 0u) << line;
    ASSERT_TRUE(std::getline(csv, line));
    EXPECT_EQ(line.rfind("3,1,3,power,", 0), 0u) << line;
}
