// Contracts of the moving-bodies input reader: the actuator-disk sampling mode defaults to
// the corrected disk for mode = adm and to plain disk sampling otherwise, an explicit choice
// is kept, the turbine's upstream sampling distance defaults to two diameters, and the filtered
// lifting-line correction is on by default for an actuator line only.

#include <string>
#include <vector>

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <gtest/gtest.h>

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
