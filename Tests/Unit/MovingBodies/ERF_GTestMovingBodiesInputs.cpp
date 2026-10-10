// Contracts of the moving-bodies input reader: the actuator-disk sampling mode defaults to
// the corrected disk for mode = adm and to plain disk sampling otherwise, an explicit choice
// is kept, the turbine's upstream sampling distance defaults to two diameters, and the filtered
// lifting-line correction is on by default for an actuator line only.

#include <string>
#include <vector>

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <gtest/gtest.h>

#include "ERF_GTestThrowOnAbort.H"
#include "ERF_MovingBodiesInputs.H"

namespace {

void set_turbine (const std::string& name, const std::string& mode, const std::string& sampling)
{
    amrex::ParmParse pp("erf.moving_bodies");
    pp.add("bodies", name);
    amrex::ParmParse pb("erf.moving_bodies." + name);
    pb.add("type", std::string("openfast_turbine"));
    pb.add("fst_file", std::string("stub_turbine.fst"));
    pb.add("mode", mode);
    if (!sampling.empty()) { pb.add("sampling", sampling); }
    pb.addarr("base_pos", std::vector<amrex::Real>{750.0, 600.0, 0.0});
}

const MovingBodyInputs& only_body (const MovingBodiesInputs& in)
{
    EXPECT_EQ(in.bodies.size(), 1u);
    return in.bodies.front();
}

} // namespace

TEST(MovingBodiesInputs, DiskSamplingDefaultsToTheCorrectedDiskForAnActuatorDisk)
{
    set_turbine("Tadm", "adm", "");
    MovingBodiesInputs in = MovingBodiesInputs::read();
    const MovingBodyInputs& b = only_body(in);
    EXPECT_EQ(b.sampling, "disk_corrected");
    EXPECT_DOUBLE_EQ(b.sample_diameters_upstream, 2.0);
}

TEST(MovingBodiesInputs, SamplingDefaultsToPlainDiskForAnActuatorLine)
{
    set_turbine("Talm", "alm", "");
    MovingBodiesInputs in = MovingBodiesInputs::read();
    EXPECT_EQ(only_body(in).sampling, "disk");
}

TEST(MovingBodiesInputs, TheLiftingLineCorrectionIsOnByDefaultForAnActuatorLineOnly)
{
    set_turbine("Talm2", "alm", "");
    EXPECT_TRUE(only_body(MovingBodiesInputs::read()).fllc);
    set_turbine("Tadm2", "adm", "");
    EXPECT_FALSE(only_body(MovingBodiesInputs::read()).fllc);
    set_turbine("Talm3", "alm", "");
    { amrex::ParmParse pb("erf.moving_bodies.Talm3"); pb.add("fllc", false); }
    EXPECT_FALSE(only_body(MovingBodiesInputs::read()).fllc);
}

TEST(MovingBodiesInputs, AnExplicitSamplingChoiceIsKept)
{
    set_turbine("Tup", "adm", "upstream");
    MovingBodiesInputs in = MovingBodiesInputs::read();
    EXPECT_EQ(only_body(in).sampling, "upstream");
    set_turbine("Tdisk", "adm", "disk");
    MovingBodiesInputs in2 = MovingBodiesInputs::read();
    EXPECT_EQ(only_body(in2).sampling, "disk");
}

// Each check below aborts with a message naming the input; the old reader let these through
// (NaN passes a check written as x < 0, one ring point was accepted, two bodies could share files).
TEST(MovingBodiesInputs, NonFiniteAndOutOfRangeInputsAbortNamingTheKey)
{
    auto msg_for = [] (const std::string& name, const std::string& key, const std::string& value) {
        set_turbine(name, "adm", "");
        amrex::ParmParse pb("erf.moving_bodies." + name);
        pb.add(key.c_str(), value);
        return erf_gtest::abort_message([] { MovingBodiesInputs::read(); });
    };
    EXPECT_NE(msg_for("Tnan1", "nacelle_cd", "nan").find("nacelle_cd"), std::string::npos);
    EXPECT_NE(msg_for("Tnan2", "epsilon", "inf").find("epsilon"), std::string::npos);
    EXPECT_NE(msg_for("Tnpt", "num_points_t", "1").find("num_points_t"), std::string::npos);
    EXPECT_NE(msg_for("Trel", "correction_relax", "1.5").find("correction_relax"), std::string::npos);
    EXPECT_NE(msg_for("Trel0", "correction_relax", "0").find("correction_relax"), std::string::npos);
    EXPECT_NE(msg_for("Tct", "correction_time", "-2").find("correction_time"), std::string::npos);
    EXPECT_NE(msg_for("Tct2", "correction_time", "inf").find("correction_time"), std::string::npos);
    {
        set_turbine("Tbase", "adm", "");
        amrex::ParmParse pb("erf.moving_bodies.Tbase");
        pb.addarr("base_pos", std::vector<std::string>{"750.0", "nan", "0.0"});
        EXPECT_NE(erf_gtest::abort_message([] { MovingBodiesInputs::read(); }).find("base_pos"), std::string::npos);
    }
    {
        amrex::ParmParse pp("erf.moving_bodies");
        pp.add("avg_start", std::string("nan"));
        set_turbine("Tavg", "adm", "");
        EXPECT_NE(erf_gtest::abort_message([] { MovingBodiesInputs::read(); }).find("avg_start"), std::string::npos);
        pp.add("avg_start", std::string("0"));
    }
}

TEST(MovingBodiesInputs, TheCorrectionRelaxationDefaultsToTheGainAndAcceptsAFixedValue)
{
    // a fresh body name on every run, so a repeated test does not find the value it set before
    static int run = 0;
    const std::string name = "Trelax" + std::to_string(++run);
    set_turbine(name, "adm", "");
    EXPECT_DOUBLE_EQ(only_body(MovingBodiesInputs::read()).correction_relax, -1.0);
    EXPECT_DOUBLE_EQ(only_body(MovingBodiesInputs::read()).correction_time, -1.0);
    { amrex::ParmParse pb("erf.moving_bodies." + name); pb.add("correction_relax", 1.0); }
    EXPECT_DOUBLE_EQ(only_body(MovingBodiesInputs::read()).correction_relax, 1.0);
}

TEST(MovingBodiesInputs, TwoBodiesWithOneOutputRootAbort)
{
    amrex::ParmParse pp("erf.moving_bodies");
    pp.addarr("bodies", std::vector<std::string>{"Ta", "Tb"});
    for (const std::string n : {"Ta", "Tb"}) {
        amrex::ParmParse pb("erf.moving_bodies." + n);
        pb.add("type", std::string("openfast_turbine"));
        pb.add("fst_file", std::string("stub_turbine.fst"));
        pb.addarr("base_pos", std::vector<amrex::Real>{750.0, 600.0, 0.0});
        pb.add("output_root", std::string("same_root"));
    }
    EXPECT_NE(erf_gtest::abort_message([] { MovingBodiesInputs::read(); }).find("output_root"), std::string::npos);
}
