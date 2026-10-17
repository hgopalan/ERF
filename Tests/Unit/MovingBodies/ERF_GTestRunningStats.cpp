// Running statistics of a body's diagnostics: mean, root mean square, minimum and maximum of
// the samples since the averaging start, exact for a known sequence, written as a CSV, carried
// across a checkpoint round trip; a non-finite sample or a malformed checkpoint aborts, naming
// the statistics or the file; a quantity renamed since the checkpoint (an older span checkpoint's
// drag_y) continues when it is the same quantity and starts afresh when it is not.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

#ifndef _WIN32
#include <sys/resource.h>
#endif

#include <gtest/gtest.h>

#include "../ERF_GTestTempDir.H"
#include "ERF_DiagnosticsLog.H"
#include "ERF_GTestThrowOnAbort.H"
#include "ERF_RunningStats.H"

namespace {

// a scratch root drawn once per test process (ERF_GTestTempDir.H): the fixed names below it are this
// process's alone, so ctest -j and the shuffled rerun never share them; removed when the process exits
const std::filesystem::path& gtest_scratch_root ()
{
    struct Root {
        std::filesystem::path p;
        ~Root () { std::error_code ec; std::filesystem::remove_all(p, ec); }
    };
    static const Root root{[] {
        const std::filesystem::path p = erf_gtest_temp_path("erf_gtest_runningstats");
        std::filesystem::create_directories(p);
        return p;
    }()};
    return root.p;
}
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
    const auto dir = gtest_scratch_root() / "erf_gtest_stats";
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

TEST(RunningStats, ANonFiniteSampleOrAMalformedCheckpointIsRefused)
{
    RunningStats s("L1_span1", "out/L1_span1", {"swing", "tension"});
    s.accumulate(1.0, {Real(3.0), Real(1.0e4)});
    std::string msg = erf_gtest::abort_message([&] { s.accumulate(2.0, {Real(3.0), std::numeric_limits<Real>::quiet_NaN()}); });
    EXPECT_NE(msg.find("RunningStats L1_span1"), std::string::npos) << msg;
    EXPECT_NE(msg.find("tension"), std::string::npos) << msg;
    EXPECT_EQ(s.num_samples(), 1) << "a refused sample is not counted";

    const auto dir = gtest_scratch_root() / "erf_gtest_stats_bad";
    std::filesystem::create_directories(dir);
    s.write_state(dir.string());
    std::string good;
    {
        std::ifstream in(dir / "L1_span1_stats.dat");
        std::stringstream ss;
        ss << in.rdbuf();
        good = ss.str();
    }
    auto refused = [&] (const std::string& from, const std::string& to) {
        std::string text = good;
        const auto at = text.find(from);
        EXPECT_NE(at, std::string::npos) << from;
        if (at != std::string::npos) { text.replace(at, from.size(), to); }
        std::ofstream(dir / "L1_span1_stats.dat", std::ios::trunc) << text;
        RunningStats b("L1_span1", "out/L1_span1", {"swing", "tension"});
        const std::string m = erf_gtest::abort_message([&] { b.read_state(dir.string()); });
        EXPECT_EQ(b.num_samples(), 0) << "a refused checkpoint changes nothing";
        EXPECT_EQ(b.mean(0), 0.0);
        return m;
    };
    EXPECT_NE(refused("count = 1", "count = -1").find("malformed statistics checkpoint"), std::string::npos);
    EXPECT_NE(refused("t_last = 1", "t_last = 0.5").find("t_first <= t_last"), std::string::npos);
    EXPECT_NE(refused("tension", "power").find("does not list tension"), std::string::npos);
    EXPECT_NE(refused("size = 2", "size = 3").find("holds 3 quantities"), std::string::npos);
}

// A checkpoint may list a span's ERF-frame y drag (drag_y) where its drag across the span (drag_normal)
// stands, the same quantity only for a span along +x: a checkpoint holding drag_y continues as
// drag_normal where the renaming is accepted as the same quantity, starts afresh where it is accepted as
// another, and is refused where it is not accepted, as is any other renaming.
TEST(RunningStats, ARenamedQuantityContinuesOnlyWhenItIsTheSame)
{
    const auto dir = erf_gtest_temp_path("erf_gtest_stats_drag");
    std::filesystem::create_directories(dir);
    RunningStats old_names("S1", (dir / "S1").string(), {"swing_deg", "drag_y"});
    old_names.accumulate(1.0, {Real(10.0), Real(250.0)});
    old_names.accumulate(2.0, {Real(12.0), Real(270.0)});
    old_names.write_state(dir.string());
    RunningStats same("S1", (dir / "S1").string(), {"swing_deg", "drag_normal"});
    same.renamed_from("drag_y", "drag_normal", true);
    EXPECT_TRUE(same.read_state(dir.string()));
    EXPECT_EQ(same.num_samples(), 2);
    EXPECT_EQ(same.mean(1), old_names.mean(1));
    RunningStats another("S1", (dir / "S1").string(), {"swing_deg", "drag_normal"});
    another.renamed_from("drag_y", "drag_normal", false);
    EXPECT_TRUE(another.read_state(dir.string()));
    EXPECT_EQ(another.num_samples(), 0) << "drag_y is another quantity here: the statistics start afresh";
    RunningStats unaccepted("S1", (dir / "S1").string(), {"swing_deg", "drag_normal"});
    std::string msg = erf_gtest::abort_message([&] { unaccepted.read_state(dir.string()); });
    EXPECT_NE(msg.find("does not list drag_normal where expected"), std::string::npos) << msg;
    RunningStats other("S1", (dir / "S1").string(), {"swing_deg", "drag_x"});
    other.renamed_from("drag_y", "drag_normal", true);
    msg = erf_gtest::abort_message([&] { other.read_state(dir.string()); });
    EXPECT_NE(msg.find("does not list drag_x where expected"), std::string::npos) << msg;
    std::filesystem::remove_all(dir);
}

TEST(RunningStats, TheRowsWithABodyColumnAreTheFilesRowsAfterIt)
{
    RunningStats a("T1", (gtest_scratch_root() / "T1rows").string(), {"thrust", "power"});
    a.accumulate(1.0, {Real(2.0e6), Real(1.0e7)});
    a.accumulate(2.0, {Real(3.0e6), Real(1.5e7)});
    std::ostringstream own, prefixed, packed;
    a.write_rows(own, "");
    a.write_rows(prefixed, "T1,");
    // the farm format's rows, from the values gathered to the I/O rank
    const std::vector<Real> v{a.mean(0), a.rms(0), a.min(0), a.max(0), a.mean(1), a.rms(1), a.min(1), a.max(1)};
    RunningStats::write_rows(packed, "T1,", a.quantities(), a.num_samples(), a.t_first(), a.t_last(), v.data());
    EXPECT_EQ(prefixed.str(), packed.str());
    std::istringstream o(own.str()), p(prefixed.str());
    std::string lo, lp;
    int n = 0;
    while (std::getline(o, lo)) {
        ASSERT_TRUE(std::getline(p, lp));
        EXPECT_EQ(lp, "T1," + lo);
        ++n;
    }
    EXPECT_EQ(n, 2);
}

TEST(LogStreams, AFileIsOpenedOnceAndAppendedToAfterARestart)
{
    const auto dir = gtest_scratch_root() / "erf_gtest_logstreams";
    const std::string f = (dir / "farm_erf.csv").string();
    const Real r1[2] = {Real(1.5), Real(-2.0)};
    const Real r2[2] = {Real(2.5), Real(-3.0)};
    const std::string body = "T1";
    {
        erf_actuator::LogStreams logs;
        // a fresh run: the file starts afresh with its header (and its directory is made)
        erf_actuator::write_log_row(logs.get(f, true, "time,body,a,b"), 0.1, &body, r1, 2);
        // later calls return the open stream: no second truncation, no second header
        erf_actuator::write_log_row(logs.get(f, true, "time,body,a,b"), 0.2, &body, r2, 2);
        EXPECT_TRUE(logs.is_open(f));
        EXPECT_EQ(logs.size(), 1u);
        logs.flush();
        // flushed: the rows are in the file while the stream stays open
        std::ifstream in(f);
        std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        EXPECT_EQ(all, "time,body,a,b\n0.1,T1,1.5,-2\n0.2,T1,2.5,-3\n");
    }
    {
        // a restarted run appends, without a header, to the file it continues
        erf_actuator::LogStreams logs;
        erf_actuator::write_log_row(logs.get(f, false, "time,body,a,b"), 0.3, nullptr, r1, 2);
    }   // flushed when destroyed
    std::ifstream in(f);
    std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(all, "time,body,a,b\n0.1,T1,1.5,-2\n0.2,T1,2.5,-3\n0.3,1.5,-2\n");
}

namespace {
std::string read_all (const std::string& f)
{
    std::ifstream in(f);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}
} // namespace

TEST(LogStreams, CappedStreamsWriteTheSameBytesAsUncapped)
{
    // seven files written in turn, the way an owner of seven bodies writes one log row for each per
    // step, with at most two open: each later row closes another file and opens its own again
    const auto dir = gtest_scratch_root() / "erf_gtest_logstreams_cap";
    const int nf = 7, nrow = 4;
    auto name = [&dir] (const char* run, int f) { return (dir / run / ("T" + std::to_string(f) + "_erf.csv")).string(); };
    // a fresh run truncates: each capped file holds stale rows first, which its first open must drop and its reopenings keep
    std::filesystem::create_directories(dir / "capped");
    for (int f = 0; f < nf; ++f) { std::ofstream(name("capped", f)) << "stale\n"; }
    auto write = [&] (erf_actuator::LogStreams& logs, const char* run) {
        for (int r = 0; r < nrow; ++r) {
            for (int f = 0; f < nf; ++f) {
                const Real v[2] = {Real(r), Real(10 * f + r)};
                erf_actuator::write_log_row(logs.get(name(run, f), true, "time,a,b"), 0.5 * r, nullptr, v, 2);
                EXPECT_LE(logs.size(), logs.max_open());
            }
        }
    };
    {
        erf_actuator::LogStreams all(100);
        erf_actuator::LogStreams two(2);
        write(all, "uncapped");
        write(two, "capped");
        EXPECT_EQ(all.size(), static_cast<std::size_t>(nf));
        EXPECT_EQ(two.size(), 2u);
    }
    for (int f = 0; f < nf; ++f) {
        std::string expect = "time,a,b\n";
        for (int r = 0; r < nrow; ++r) {
            std::ostringstream row;
            row << 0.5 * r << "," << r << "," << 10 * f + r << "\n";
            expect += row.str();
        }
        EXPECT_EQ(read_all(name("uncapped", f)), expect) << "file " << f;
        EXPECT_EQ(read_all(name("capped", f)), expect) << "file " << f;
    }
    // a restarted run's capped streams append to every file, closed and opened again as above, with no second header
    {
        erf_actuator::LogStreams two(2);
        for (int f = 0; f < nf; ++f) {
            const Real v[2] = {Real(9), Real(9)};
            erf_actuator::write_log_row(two.get(name("capped", f), false, "time,a,b"), 9.0, nullptr, v, 2);
            erf_actuator::write_log_row(two.get(name("capped", (f + 3) % nf), false, "time,a,b"), 9.5, nullptr, v, 2);
        }
    }
    EXPECT_EQ(read_all(name("capped", 0)), read_all(name("uncapped", 0)) + "9,9,9\n9.5,9,9\n");
}

TEST(LogStreams, TheDefaultCapIsAQuarterOfTheOpenFileLimit)
{
    const std::size_t n = erf_actuator::default_max_open_logs();
#ifndef _WIN32
    rlimit rl{};
    ASSERT_EQ(getrlimit(RLIMIT_NOFILE, &rl), 0);
    if (rl.rlim_cur != RLIM_INFINITY) {
        const auto quarter = static_cast<std::size_t>(rl.rlim_cur / 4);
        EXPECT_EQ(n, std::min<std::size_t>(4096, std::max<std::size_t>(16, quarter))) << "soft limit " << rl.rlim_cur;
    }
#endif
    EXPECT_GE(n, 16u);
    EXPECT_LE(n, 4096u);
    EXPECT_EQ(erf_actuator::LogStreams().max_open(), n);
    EXPECT_EQ(erf_actuator::LogStreams(0).max_open(), 1u);
}

TEST(LogStreams, OpeningOneMoreClosesTheFileWrittenToMostRecently)
{
    // logs written in the same order every step: closing the most recent one keeps the others open
    const auto dir = gtest_scratch_root() / "erf_gtest_logstreams_order";
    auto f = [&dir] (const char* n) { return (dir / n).string(); };
    erf_actuator::LogStreams two(2);
    two.get(f("A.csv"), true, "h");
    two.get(f("B.csv"), true, "h");
    two.get(f("A.csv"), true, "h");   // A is now the most recent
    two.get(f("C.csv"), true, "h");   // closes A, keeps B
    EXPECT_TRUE(two.is_open(f("B.csv")));
    EXPECT_TRUE(two.is_open(f("C.csv")));
    EXPECT_FALSE(two.is_open(f("A.csv")));
    // cycles over four files with two open: on average one file is still open at its next row, so 8
    // cycles open at most 8 x 3 files (2, 3 or 3 a cycle); closing the least recent instead opens all 4 every cycle
    for (const char* n : {"B.csv", "A.csv", "C.csv", "D.csv"}) { two.get(f(n), true, "h"); }
    const std::size_t before = two.opens();
    for (int cycle = 0; cycle < 8; ++cycle) {
        for (const char* n : {"B.csv", "A.csv", "C.csv", "D.csv"}) { two.get(f(n), true, "h"); }
    }
    EXPECT_LE(two.opens() - before, 24u);
}

namespace {

// the trim as the I/O rank did it before it cut files from their end: read the whole file and
// rewrite the rows to keep (the reference the new trim must reproduce, byte for byte)
std::size_t trim_by_rewrite (const std::string& fname, double t_chk, bool keep_at_chk, double tol)
{
    std::ifstream in(fname);
    if (!in) { return 0; }
    std::vector<std::string> keep;
    std::string line;
    bool header = true;
    std::size_t dropped = 0;
    while (std::getline(in, line)) {
        if (header) { keep.push_back(line); header = false; continue; }
        if (line.empty()) { continue; }
        double t = 0.0;
        try { t = std::stod(line.substr(0, line.find(','))); } catch (...) { keep.push_back(line); continue; }
        const bool before = keep_at_chk ? (t <= t_chk + tol) : (t < t_chk - tol);
        if (before) { keep.push_back(line); } else { ++dropped; }
    }
    in.close();
    if (dropped == 0) { return 0; }
    std::ofstream out(fname, std::ios::trunc);
    for (const auto& l : keep) { out << l << "\n"; }
    return dropped;
}

void write_file (const std::string& f, const std::string& text)
{
    std::ofstream out(f, std::ios::binary | std::ios::trunc);
    out << text;
}

} // namespace

TEST(TrimLog, CuttingFromTheEndLeavesTheBytesOfTheRewrite)
{
    // logs in time order as ERF writes them: a header, rows whole steps apart (several bodies per time in
    // the rank and farm files), rows stamped at the checkpoint time, and some files longer than the first
    // block read from the end (64 KiB). (A row that does not parse, which ERF does not write, is kept
    // with the rows before it; the rewrite also dropped the rows before it that lie past the checkpoint.)
    const auto dir = gtest_scratch_root() / "erf_gtest_trim";
    std::filesystem::create_directories(dir);
    std::mt19937 gen(20261017);
    int cases = 0, cut_cases = 0, long_cases = 0;
    for (int c = 0; c < 400; ++c) {
        const double dt = 0.5;
        const int nrow = (c % 10 == 0) ? 4000 + static_cast<int>(gen() % 3000) : static_cast<int>(gen() % 40);
        const int per_time = 1 + static_cast<int>(gen() % 3);
        std::ostringstream text;
        text << "time,body,a,b";
        if (gen() % 4 != 0) { text << "\n"; }
        else if (nrow > 0) { text << "\n"; }
        for (int r = 0; r < nrow; ++r) {
            const double t = dt * (r / per_time);
            text << std::setprecision(10) << t << ",T" << r % per_time << "," << 1.5 * r << ",-" << r << ",padding-to-make-rows-longer";
            // the last row may lack its newline (a run stopped while writing)
            if (r + 1 < nrow || gen() % 3 != 0) { text << "\n"; }
        }
        const double t_chk = dt * static_cast<double>(gen() % (nrow / per_time + 3));
        const bool keep_at_chk = (gen() % 2 == 0);
        const double tol = 0.25 * dt;
        const std::string a = (dir / ("a" + std::to_string(c) + ".csv")).string();
        const std::string b = (dir / ("b" + std::to_string(c) + ".csv")).string();
        write_file(a, text.str());
        write_file(b, text.str());
        const std::size_t na = trim_by_rewrite(a, t_chk, keep_at_chk, tol);
        const std::size_t nb = erf_actuator::trim_log_after(b, t_chk, keep_at_chk, tol);
        EXPECT_EQ(nb, na) << "case " << c;
        EXPECT_EQ(read_all(b), read_all(a)) << "case " << c << ": " << nrow << " rows, t_chk " << t_chk << ", keep_at_chk " << keep_at_chk;
        ++cases;
        if (na > 0) { ++cut_cases; }
        if (text.str().size() - read_all(a).size() > 2 * 65536) { ++long_cases; }
    }
    // the sweep reaches the cases it is for: files cut, and cuts more than two blocks back from the end
    EXPECT_GT(cut_cases, 100);
    EXPECT_GT(long_cases, 5);
    EXPECT_EQ(cases, 400);
}

TEST(TrimLog, EdgeCases)
{
    const auto dir = gtest_scratch_root() / "erf_gtest_trim_edges";
    std::filesystem::create_directories(dir);
    const std::string f = (dir / "log.csv").string();
    // a missing file is left alone, and not made
    EXPECT_EQ(erf_actuator::trim_log_after(f, 1.0, true, 0.1), 0u);
    EXPECT_FALSE(std::filesystem::exists(f));
    // an empty file, and a header alone
    write_file(f, "");
    EXPECT_EQ(erf_actuator::trim_log_after(f, 1.0, true, 0.1), 0u);
    EXPECT_EQ(read_all(f), "");
    write_file(f, "time,a\n");
    EXPECT_EQ(erf_actuator::trim_log_after(f, 1.0, true, 0.1), 0u);
    EXPECT_EQ(read_all(f), "time,a\n");
    // no row past the checkpoint: the file is not touched
    write_file(f, "time,a\n0.5,1\n1,2\n");
    EXPECT_EQ(erf_actuator::trim_log_after(f, 1.0, true, 0.1), 0u);
    EXPECT_EQ(read_all(f), "time,a\n0.5,1\n1,2\n");
    // a row at the checkpoint time: kept when stamped at the end of its step, dropped when stamped at its start
    EXPECT_EQ(erf_actuator::trim_log_after(f, 1.0, false, 0.1), 1u);
    EXPECT_EQ(read_all(f), "time,a\n0.5,1\n");
    // every row past it: the header stays; empty lines past the last row kept go too, uncounted
    write_file(f, "time,a\n2,1\n\n3,2\n\n");
    EXPECT_EQ(erf_actuator::trim_log_after(f, 1.0, true, 0.1), 2u);
    EXPECT_EQ(read_all(f), "time,a\n");
    // a row that does not parse is kept, and the rows before it
    write_file(f, "time,a\n1,1\nnote\n3,2\n4,3");
    EXPECT_EQ(erf_actuator::trim_log_after(f, 1.0, true, 0.1), 2u);
    EXPECT_EQ(read_all(f), "time,a\n1,1\nnote\n");
    // an empty line among the rows kept stays (the earlier rewrite removed it)
    write_file(f, "time,a\n1,1\n\n1.5,2\n3,3\n");
    EXPECT_EQ(erf_actuator::trim_log_after(f, 2.0, true, 0.1), 1u);
    EXPECT_EQ(read_all(f), "time,a\n1,1\n\n1.5,2\n");
    // rows longer than the first block read from the end (64 KiB): the block grows until a row fits
    const std::string wide(200000, '7');
    write_file(f, "time,a\n1," + wide + "\n2," + wide + "\n3," + wide + "\n");
    EXPECT_EQ(erf_actuator::trim_log_after(f, 1.0, true, 0.1), 2u);
    EXPECT_EQ(read_all(f), "time,a\n1," + wide + "\n");
}
