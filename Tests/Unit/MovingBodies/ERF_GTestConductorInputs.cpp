// Contract of erf_conductors::ConductorInputs: a span block is read with its defaults and its
// derived chord and catenary sag; every value outside its documented range is refused with a
// message naming the key; the shared settings and the solver settings are checked the same way.

#include <string>
#include <vector>

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <gtest/gtest.h>

#include "ERF_ConductorInputs.H"

using erf_conductors::ConductorInputs;
using erf_conductors::SpanInputs;

namespace {

void set_span (const std::string& name)
{
    amrex::ParmParse pp("erf.conductors");
    pp.add("spans", name);
    amrex::ParmParse ps("erf.conductors." + name);
    ps.addarr("end_a", std::vector<amrex::Real>{100.0, 500.0, 30.0});
    ps.addarr("end_b", std::vector<amrex::Real>{400.0, 500.0, 30.0});
    ps.add("length", 301.5);
    ps.add("diameter", 0.0281);
    ps.add("mass_per_length", 1.628);
    ps.add("axial_stiffness", 3.0e7);
}

SpanInputs good_span ()
{
    SpanInputs s;
    s.name = "S";
    s.end_a = {{100.0, 500.0, 30.0}};
    s.end_b = {{400.0, 500.0, 30.0}};
    s.length = 301.5; s.diameter = 0.0281; s.mass_per_length = 1.628; s.axial_stiffness = 3.0e7;
    return s;
}

} // namespace

TEST(ConductorInputs, ASpanIsReadWithItsDefaultsAndDerivedGeometry)
{
    set_span("S1");
    const ConductorInputs in = ConductorInputs::read();
    ASSERT_TRUE(in.active);
    ASSERT_EQ(in.spans.size(), 1u);
    const SpanInputs& s = in.spans.front();
    EXPECT_EQ(s.name, "S1");
    EXPECT_DOUBLE_EQ(s.chord(), 300.0);
    EXPECT_NEAR(s.catenary_sag(), 12.99, 0.01);
    EXPECT_DOUBLE_EQ(s.drag_coefficient, 1.0);
    EXPECT_DOUBLE_EQ(s.damping_ratio, 0.5);
    EXPECT_EQ(s.segments, 20);
    EXPECT_EQ(s.output_root, "conductors/S1");
    EXPECT_EQ(in.diagnostics_int, 1);
    EXPECT_EQ(in.anchor_level, -1);
    EXPECT_DOUBLE_EQ(in.air_density, 1.225);
    EXPECT_EQ(in.substeps, 1);
    EXPECT_DOUBLE_EQ(in.moordyn_dt, 0.0);
    EXPECT_DOUBLE_EQ(in.moordyn_cfl, 0.1);
    EXPECT_EQ(in.moordyn_log_level, 2);
    EXPECT_DOUBLE_EQ(in.surface_offset, 10000.0);
    EXPECT_FALSE(in.has_prescribed_velocity);
}

TEST(ConductorInputs, NothingIsReadWithoutSpans)
{
    // the spans key of the previous test still exists; an empty list switches the feature off
    amrex::ParmParse pp("erf.conductors");
    pp.addarr("spans", std::vector<std::string>{});
    EXPECT_FALSE(ConductorInputs::read().active);
}

TEST(ConductorInputs, EverySpanValueOutsideItsRangeIsRefusedByName)
{
    EXPECT_TRUE(ConductorInputs::validate_span(good_span()).empty());
    auto bad = [](auto mutate, const std::string& key) {
        SpanInputs s = good_span();
        mutate(s);
        const std::string err = ConductorInputs::validate_span(s);
        EXPECT_FALSE(err.empty()) << key;
        EXPECT_NE(err.find("erf.conductors.S." + key), std::string::npos) << err;
    };
    bad([](SpanInputs& s) { s.end_b = s.end_a; }, "end_a");
    bad([](SpanInputs& s) { s.length = 300.0; }, "length");      // equal to the chord: no slack
    bad([](SpanInputs& s) { s.length = 250.0; }, "length");
    bad([](SpanInputs& s) { s.diameter = 0.0; }, "diameter");
    bad([](SpanInputs& s) { s.mass_per_length = -1.0; }, "mass_per_length");
    bad([](SpanInputs& s) { s.axial_stiffness = 0.0; }, "axial_stiffness");
    bad([](SpanInputs& s) { s.drag_coefficient = -0.1; }, "drag_coefficient");
    bad([](SpanInputs& s) { s.damping_ratio = 0.0; }, "damping_ratio");
    bad([](SpanInputs& s) { s.damping_ratio = 1.5; }, "damping_ratio");
    bad([](SpanInputs& s) { s.segments = 1; }, "segments");
}

TEST(ConductorInputs, SharedSettingsOutsideTheirRangeAreRefusedByName)
{
    ConductorInputs in;
    in.spans.push_back(good_span());
    EXPECT_TRUE(ConductorInputs::validate_settings(in).empty());
    auto bad = [&](auto mutate, const std::string& key) {
        ConductorInputs c = in;
        mutate(c);
        const std::string err = ConductorInputs::validate_settings(c);
        EXPECT_FALSE(err.empty()) << key;
        EXPECT_NE(err.find(key), std::string::npos) << err;
    };
    bad([](ConductorInputs& c) { c.diagnostics_int = 0; }, "diagnostics_int");
    bad([](ConductorInputs& c) { c.anchor_level = -2; }, "anchor_level");
    bad([](ConductorInputs& c) { c.air_density = 0.0; }, "air_density");
    bad([](ConductorInputs& c) { c.substeps = 0; }, "substeps");
    bad([](ConductorInputs& c) { c.moordyn_dt = -0.01; }, "moordyn_dt");
    bad([](ConductorInputs& c) { c.moordyn_cfl = 0.0; }, "moordyn_cfl");
    bad([](ConductorInputs& c) { c.moordyn_cfl = 0.2; }, "moordyn_cfl");   // runs, but more than doubles the sag
    {
        ConductorInputs c = in;
        c.moordyn_cfl = ConductorInputs::max_moordyn_cfl;
        EXPECT_TRUE(ConductorInputs::validate_settings(c).empty()) << "the bound itself is accepted";
    }
    bad([](ConductorInputs& c) { c.moordyn_log_level = 4; }, "moordyn_log_level");
    bad([](ConductorInputs& c) { c.surface_offset = 0.0; }, "surface_offset");
    bad([](ConductorInputs& c) { c.surface_offset = 20.0; }, "surface_offset");   // below the attachments
}

TEST(ConductorInputs, SolverSettingsAreChecked)
{
    EXPECT_TRUE(ConductorInputs::validate_solver(1, 1, false).empty());
    EXPECT_TRUE(ConductorInputs::validate_solver(1, 0, false).empty());
    EXPECT_NE(ConductorInputs::validate_solver(1, 2, false).find("anchor_level"), std::string::npos);
    EXPECT_NE(ConductorInputs::validate_solver(0, 0, true).find("fpe_trap"), std::string::npos);
    EXPECT_EQ(ConductorInputs::resolve_anchor_level(-1, 2), 2);
    EXPECT_EQ(ConductorInputs::resolve_anchor_level(1, 2), 1);
}
