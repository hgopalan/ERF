// Contracts of the moving-bodies input reader: the actuator-disk sampling mode defaults to
// the corrected disk for mode = adm and to plain disk sampling otherwise, an explicit choice
// is kept, the turbine's upstream sampling distance defaults to two diameters, and the filtered
// lifting-line correction is on by default for an actuator line only. The tower's and the nacelle's
// kernel widths default to epsilon, may differ from it, need their component on and a turbine, and
// enter the restart record, where a checkpoint without them stands for epsilon.

#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <AMReX_ParallelDescriptor.H>
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

namespace {

// a fresh turbine name per call, so repeated or shuffled tests never find keys an earlier run left
std::string fresh_turbine (const std::string& tag, const std::string& mode = "alm")
{
    static int run = 0;
    const std::string name = "Tw" + tag + std::to_string(++run);
    set_turbine(name, mode, "");
    return name;
}

// what f prints to the standard output (amrex::Print on the I/O rank), and its abort message if it aborts;
// the standard output gets its buffer back however f ends
std::string printed (const std::function<void()>& f, std::string& abort_msg)
{
    struct Restore {
        std::streambuf* old;
        ~Restore () { std::cout.rdbuf(old); }
    };
    std::ostringstream out;
    Restore restore{std::cout.rdbuf(out.rdbuf())};
    abort_msg = erf_gtest::abort_message(f);
    return out.str();
}

std::string record_value (const std::vector<std::pair<std::string,std::string>>& r, const std::string& key)
{
    for (const auto& kv : r) { if (kv.first == key) { return kv.second; } }
    return {};
}

} // namespace

TEST(MovingBodiesInputs, TowerAndNacelleWidthsDefaultToEpsilonAndMayDiffer)
{
    std::string name = fresh_turbine("def");
    {
        amrex::ParmParse pb("erf.moving_bodies." + name);
        pb.add("epsilon", 1.5);
        pb.add("num_force_points_tower", 10);
        pb.add("nacelle_cd", 1.0);
        pb.add("nacelle_area", 60.0);
    }
    KernelWidths w = only_body(MovingBodiesInputs::read()).kernel_widths(4.0);
    EXPECT_DOUBLE_EQ(w.rotor, 6.0);
    EXPECT_DOUBLE_EQ(w.tower, 6.0);
    EXPECT_DOUBLE_EQ(w.nacelle, 6.0);
    {
        amrex::ParmParse pb("erf.moving_bodies." + name);
        pb.add("epsilon_tower", 3.0);
        pb.add("epsilon_nacelle", 0.75);
    }
    w = only_body(MovingBodiesInputs::read()).kernel_widths(4.0);
    EXPECT_DOUBLE_EQ(w.rotor, 6.0);
    EXPECT_DOUBLE_EQ(w.tower, 12.0);
    EXPECT_DOUBLE_EQ(w.nacelle, 3.0);
}

// A width for a component that is off: an abort naming both keys when the component's switch was
// never set, a warning naming the width (the run goes on) when it was set off or mode = none forces
// nothing; on a ct_disk, which has no tower or nacelle, a warning; a width that is not positive and
// finite, an abort naming the key.
TEST(MovingBodiesInputs, AComponentWidthNeedsItsComponent)
{
    auto read_msg = [] { return erf_gtest::abort_message([] { MovingBodiesInputs::read(); }); };
    std::string msg;
    std::string name = fresh_turbine("tw");
    { amrex::ParmParse pb("erf.moving_bodies." + name); pb.add("epsilon_tower", 1.0); }
    msg = read_msg();
    EXPECT_NE(msg.find(name + ".epsilon_tower is given but erf.moving_bodies." + name + ".num_force_points_tower is not"),
              std::string::npos) << msg;
    { amrex::ParmParse pb("erf.moving_bodies." + name); pb.add("num_force_points_tower", 0); }
    std::string out = printed([] { MovingBodiesInputs::read(); }, msg);
    EXPECT_EQ(msg, "");
    if (amrex::ParallelDescriptor::IOProcessor()) {
        EXPECT_NE(out.find("Warning: erf.moving_bodies." + name + ".epsilon_tower is given but not used: erf.moving_bodies." + name +
                           ".num_force_points_tower = 0"), std::string::npos) << out;
    }

    name = fresh_turbine("nc");
    { amrex::ParmParse pb("erf.moving_bodies." + name); pb.add("epsilon_nacelle", 1.0); }
    msg = read_msg();
    EXPECT_NE(msg.find(name + ".epsilon_nacelle is given but erf.moving_bodies." + name + ".nacelle_cd is not"),
              std::string::npos) << msg;
    { amrex::ParmParse pb("erf.moving_bodies." + name); pb.add("nacelle_cd", 0.0); }
    out = printed([] { MovingBodiesInputs::read(); }, msg);
    EXPECT_EQ(msg, "");
    if (amrex::ParallelDescriptor::IOProcessor()) {
        EXPECT_NE(out.find(name + ".epsilon_nacelle is given but not used"), std::string::npos) << out;
    }

    // mode = none forces nothing: a warning only
    name = fresh_turbine("none", "none");
    { amrex::ParmParse pb("erf.moving_bodies." + name); pb.add("epsilon_tower", 1.0); pb.add("epsilon_nacelle", 1.0); }
    out = printed([] { MovingBodiesInputs::read(); }, msg);
    EXPECT_EQ(msg, "");
    if (amrex::ParallelDescriptor::IOProcessor()) {
        EXPECT_NE(out.find(name + ".epsilon_tower is given but not used: mode = none"), std::string::npos) << out;
        EXPECT_NE(out.find(name + ".epsilon_nacelle is given but not used: mode = none"), std::string::npos) << out;
    }

    for (const std::string bad : {"0", "-1", "nan", "inf"}) {
        name = fresh_turbine("bad");
        { amrex::ParmParse pb("erf.moving_bodies." + name); pb.add("num_force_points_tower", 4); pb.add("epsilon_tower", bad); }
        msg = read_msg();
        EXPECT_NE(msg.find(name + ".epsilon_tower must be positive and finite"), std::string::npos) << bad << ": " << msg;
    }

    {
        static int run = 0;
        const std::string disk = "Dw" + std::to_string(++run);
        amrex::ParmParse pp("erf.moving_bodies");
        pp.add("bodies", disk);
        amrex::ParmParse pb("erf.moving_bodies." + disk);
        pb.add("type", std::string("ct_disk"));
        pb.addarr("base_pos", std::vector<amrex::Real>{750.0, 600.0, 0.0});
        pb.add("rotor_radius", 120.0);
        pb.add("hub_height", 150.0);
        pb.add("ct", 0.75);
        pb.add("epsilon_nacelle", 1.0);
        pb.add("nacelle_cd", 1.0);
        pb.add("adm_spreading", std::string("merged_rings"));
        out = printed([] { MovingBodiesInputs::read(); }, msg);
        EXPECT_EQ(msg, "");
        if (amrex::ParallelDescriptor::IOProcessor()) {
            for (const std::string k : {"epsilon_nacelle", "nacelle_cd", "adm_spreading"}) {
                EXPECT_NE(out.find(disk + "." + k + " is given but not used: it is for an openfast_turbine"), std::string::npos) << out;
            }
        }
    }
}

// The record holds the width of every kind of point the body forces, as resolved, and no other. A
// checkpoint record without epsilon_tower or epsilon_nacelle spread those points with its epsilon:
// the same width now is no change, another is.
TEST(MovingBodiesInputs, TheRestartRecordHoldsTheWidthsInUse)
{
    MovingBodyInputs b;
    b.type = "openfast_turbine";
    b.mode = "alm";
    b.epsilon = 2.0;
    auto r = MovingBodiesInputs::restart_record(b);
    EXPECT_EQ(record_value(r, "epsilon_tower"), "");
    EXPECT_EQ(record_value(r, "epsilon_nacelle"), "");
    b.num_force_points_tower = 10;
    b.nacelle_cd = 1.0;
    b.nacelle_area = 60.0;
    r = MovingBodiesInputs::restart_record(b);
    EXPECT_EQ(record_value(r, "epsilon_tower"), "2");
    EXPECT_EQ(record_value(r, "epsilon_nacelle"), "2");
    b.epsilon_tower = 3.0;
    r = MovingBodiesInputs::restart_record(b);
    EXPECT_EQ(record_value(r, "epsilon_tower"), "3");

    // mode = none forces nothing: no width but epsilon is recorded
    {
        MovingBodyInputs none = b;
        none.mode = "none";
        const auto rn = MovingBodiesInputs::restart_record(none);
        EXPECT_EQ(record_value(rn, "epsilon_tower"), "");
        EXPECT_EQ(record_value(rn, "epsilon_nacelle"), "");
        EXPECT_EQ(record_value(rn, "epsilon"), "2");
    }

    // a checkpoint record without the component widths, its epsilon 2
    std::map<std::string,std::string> chk;
    for (const auto& kv : MovingBodiesInputs::restart_record(b)) {
        if (kv.first != "epsilon_tower" && kv.first != "epsilon_nacelle") { chk[kv.first] = kv.second; }
    }
    std::string key, was, now;
    EXPECT_TRUE(MovingBodiesInputs::changed_record_entry(chk, r, key, was, now));
    EXPECT_EQ(key, "epsilon_tower"); EXPECT_EQ(was, "2"); EXPECT_EQ(now, "3");
    b.epsilon_tower = 2.0;
    EXPECT_FALSE(MovingBodiesInputs::changed_record_entry(chk, MovingBodiesInputs::restart_record(b), key, was, now));
    b.epsilon_nacelle = 0.5;
    EXPECT_TRUE(MovingBodiesInputs::changed_record_entry(chk, MovingBodiesInputs::restart_record(b), key, was, now));
    EXPECT_EQ(key, "epsilon_nacelle");
    // with the widths recorded, the record's own value counts, not epsilon
    chk["epsilon_nacelle"] = "0.5";
    EXPECT_FALSE(MovingBodiesInputs::changed_record_entry(chk, MovingBodiesInputs::restart_record(b), key, was, now));
    // any other entry the checkpoint lacks is taken as unchanged
    chk.erase("fllc");
    EXPECT_FALSE(MovingBodiesInputs::changed_record_entry(chk, MovingBodiesInputs::restart_record(b), key, was, now));
}

TEST(MovingBodiesInputs, TheEnvelopeCheckIntervalDefaultsToEveryStepAndMustBePositive)
{
    set_turbine("Tenv", "adm", "");
    EXPECT_EQ(MovingBodiesInputs::read().envelope_check_interval, 1);
    amrex::ParmParse pp("erf.moving_bodies");
    pp.add("envelope_check_interval", 5);
    EXPECT_EQ(MovingBodiesInputs::read().envelope_check_interval, 5);
    pp.add("envelope_check_interval", 0);
    EXPECT_NE(erf_gtest::abort_message([] { MovingBodiesInputs::read(); }).find("envelope_check_interval"), std::string::npos);
    pp.add("envelope_check_interval", 1);
}

TEST(MovingBodiesInputs, TheTimingSummaryIsOffByDefaultAndItsIntervalMustNotBeNegative)
{
    set_turbine("Ttim", "adm", "");
    EXPECT_EQ(MovingBodiesInputs::read().timing_int, 0);
    amrex::ParmParse pp("erf.moving_bodies");
    pp.add("timing_int", 10);
    EXPECT_EQ(MovingBodiesInputs::read().timing_int, 10);
    pp.add("timing_int", -1);
    EXPECT_NE(erf_gtest::abort_message([] { MovingBodiesInputs::read(); }).find("erf.moving_bodies.timing_int must be >= 0"),
              std::string::npos);
    pp.add("timing_int", 0);
}

TEST(MovingBodiesInputs, TheDiagnosticsFormatDefaultsToPerTurbineAndMustBeKnown)
{
    set_turbine("Tfmt", "adm", "");
    EXPECT_EQ(MovingBodiesInputs::read().diagnostics_format, "per_turbine");
    for (const char* f : {"per_turbine", "per_rank", "farm"}) {
        EXPECT_TRUE(MovingBodiesInputs::valid_diagnostics_format(f)) << f;
    }
    EXPECT_FALSE(MovingBodiesInputs::valid_diagnostics_format("per_body"));
    EXPECT_FALSE(MovingBodiesInputs::valid_diagnostics_format(""));
    amrex::ParmParse pp("erf.moving_bodies");
    pp.add("diagnostics_format", std::string("farm"));
    EXPECT_EQ(MovingBodiesInputs::read().diagnostics_format, "farm");
    pp.add("diagnostics_format", std::string("per_body"));
    EXPECT_NE(erf_gtest::abort_message([] { MovingBodiesInputs::read(); }).find("diagnostics_format"), std::string::npos);
    pp.add("diagnostics_format", std::string("per_turbine"));
}

// adm_spreading picks how a disk's rings are laid out: rings (one per blade node, the default) or
// merged_rings (one per radial station for all the blades); any other value stops the run naming
// the key. The choice is in an actuator disk's restart record, and a checkpoint written before the
// key existed spread its disks with rings: a restart with merged_rings from it is a change.
TEST(MovingBodiesInputs, TheAdmSpreadingDefaultsToRingsAndIsInTheRestartRecord)
{
    set_turbine("Tspr", "adm", "");
    EXPECT_EQ(only_body(MovingBodiesInputs::read()).adm_spreading, "rings");
    {
        amrex::ParmParse pb("erf.moving_bodies.Tspr");
        pb.add("adm_spreading", std::string("merged_rings"));
    }
    MovingBodyInputs b = only_body(MovingBodiesInputs::read());
    EXPECT_EQ(b.adm_spreading, "merged_rings");
    {
        set_turbine("Tspr2", "adm", "");
        amrex::ParmParse pb("erf.moving_bodies.Tspr2");
        pb.add("adm_spreading", std::string("merged"));
        const std::string msg = erf_gtest::abort_message([] { MovingBodiesInputs::read(); });
        EXPECT_NE(msg.find("Tspr2.adm_spreading must be rings or merged_rings"), std::string::npos) << msg;
    }

    // recorded for an actuator disk only
    auto r = MovingBodiesInputs::restart_record(b);
    EXPECT_EQ(record_value(r, "adm_spreading"), "merged_rings");
    {
        MovingBodyInputs alm = b;
        alm.mode = "alm";
        EXPECT_EQ(record_value(MovingBodiesInputs::restart_record(alm), "adm_spreading"), "");
    }

    // a checkpoint without the entry spread with rings
    std::map<std::string,std::string> chk;
    for (const auto& kv : r) {
        if (kv.first != "adm_spreading") { chk[kv.first] = kv.second; }
    }
    std::string key, was, now;
    EXPECT_TRUE(MovingBodiesInputs::changed_record_entry(chk, r, key, was, now));
    EXPECT_EQ(key, "adm_spreading"); EXPECT_EQ(was, "rings"); EXPECT_EQ(now, "merged_rings");
    b.adm_spreading = "rings";
    EXPECT_FALSE(MovingBodiesInputs::changed_record_entry(chk, MovingBodiesInputs::restart_record(b), key, was, now));
    // with the entry recorded, a change either way is one
    chk["adm_spreading"] = "merged_rings";
    EXPECT_TRUE(MovingBodiesInputs::changed_record_entry(chk, MovingBodiesInputs::restart_record(b), key, was, now));
    EXPECT_EQ(key, "adm_spreading"); EXPECT_EQ(was, "merged_rings"); EXPECT_EQ(now, "rings");
}
