// Unit tests of erf_conductors::ConductorInputs, the erf.conductors.* inputs and their checks.
//
// ASingleSpanLineIsReadWithItsDefaultsAndDerivedGeometry: the defaults, the chord and the catenary sag.
// ElasticCatenaryOfALevelSpan: the elastic catenary against an independent solution.
// NothingIsReadWithoutLines: an empty erf.conductors.lines switches the module off.
// EveryLineValueOutsideItsRangeIsRefusedByName: validate_line names the key of every bad value,
//     non-finite values included.
// SharedSettingsOutsideTheirRangeAreRefusedByName: validate_settings names the key, non-finite
//     values and a conductor lighter than the air it displaces included.
// AnchorLevelMustExistAndFpeTrapsAreRefused: validate_solver and resolve_anchor_level.
// SurfaceOffsetMustHoldTheWholeDomain: validate_frame against the domain's top and bottom.
// ASectionIsReadWithItsTowersLengthsAndInsulatorStrings: a section's towers, lengths and strings.
// SectionValuesOutsideTheirRangeAreRefusedByName: a section's bad values, a line that turns back at a
//     string, and string keys given without strings.
// TransformersAreReadWithTheirFootprintsAndDefaults: transformer blocks and the footprint test.
// TransformerValuesOutsideTheirRangeAreRefusedByName: validate_transformer, non-finite values included.
// TheSlackIsCheckedBetweenTheEndsWhereTheyStandOnTheTerrain: validate_slack on placed heights.
// TheSlackIsCheckedBetweenTheBottomsOfTheStrings: a span hangs between the conductor points.
// AStringingTensionSetsEachSpansLengthSoThatItHangsWithThatTension: lengths_from_stringing_tension.
// TowerTypesAreReadAndALinesTowerTypeMustNameOne: tower-type blocks and validate_tower_type.
// ALineSharesTheTowersOfAnotherLineWithATowerType: validate_shared_towers and tower_owner.
// EveryLineNeedsAnOutputRootOfItsOwn: validate_output_roots.
// TheFirstNonFiniteValueIsNamedWithItsPoint: first_nonfinite, used at the module's boundaries.
// ReadRefusesMalformedInputsNamingTheKey: every abort of read() fires and names its key.

#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <gtest/gtest.h>

#include "ERF_ConductorInputs.H"
#include "ERF_GTestThrowOnAbort.H"

using erf_conductors::ConductorInputs;
using erf_conductors::LineInputs;
using erf_conductors::TransformerInputs;

namespace {

void set_span (const std::string& name)
{
    amrex::ParmParse pp("erf.conductors");
    pp.add("lines", name);
    amrex::ParmParse ps("erf.conductors." + name);
    ps.addarr("end_a", std::vector<amrex::Real>{100.0, 500.0, 30.0});
    ps.addarr("end_b", std::vector<amrex::Real>{400.0, 500.0, 30.0});
    ps.add("length", 301.5);
    ps.add("diameter", 0.0281);
    ps.add("mass_per_length", 1.628);
    ps.add("axial_stiffness", 3.0e7);
}

LineInputs good_span ()
{
    LineInputs s;
    s.name = "S";
    s.end_a = {{100.0, 500.0, 30.0}};
    s.end_b = {{400.0, 500.0, 30.0}};
    s.lengths = {301.5}; s.diameter = 0.0281; s.mass_per_length = 1.628; s.axial_stiffness = 3.0e7;
    return s;
}

} // namespace

TEST(ConductorInputs, ASingleSpanLineIsReadWithItsDefaultsAndDerivedGeometry)
{
    set_span("S1");
    const ConductorInputs in = ConductorInputs::read();
    ASSERT_TRUE(in.active);
    ASSERT_EQ(in.lines.size(), 1u);
    const LineInputs& s = in.lines.front();
    EXPECT_EQ(s.name, "S1");
    EXPECT_DOUBLE_EQ(s.chord(), 300.0);
    EXPECT_NEAR(s.catenary_sag(), 12.99, 0.01);
    EXPECT_DOUBLE_EQ(s.drag_coefficient, 1.0);
    EXPECT_DOUBLE_EQ(s.damping_ratio, 0.5);
    EXPECT_EQ(s.segments, 20);
    EXPECT_EQ(s.output_root, "conductors/S1");
    EXPECT_EQ(in.diagnostics_int, 1);
    EXPECT_EQ(in.anchor_level, -1);
    EXPECT_EQ(in.air_density, amrex::Real(1.225));
    EXPECT_EQ(in.substeps, 1);
    EXPECT_DOUBLE_EQ(in.moordyn_dt, 0.0);
    EXPECT_EQ(in.moordyn_cfl, amrex::Real(0.1));
    EXPECT_EQ(in.moordyn_log_level, 2);
    EXPECT_DOUBLE_EQ(in.surface_offset, 10000.0);
    EXPECT_FALSE(in.has_prescribed_velocity);
    EXPECT_DOUBLE_EQ(in.stats_start, 0.0);
    EXPECT_EQ(in.node_output_int, 0);
    EXPECT_FALSE(in.drag_on_flow);
    EXPECT_DOUBLE_EQ(in.epsilon, 2.0);
}

// The elastic catenary of the 300 m Drake span with 1.5 m of slack (net weight 15.9634 N/m in air of
// 1.2 kg/m^3, EA 3e7 N): reference values from an independent solution of the same equations, which
// MoorDyn-C reproduces to 0.002 % with 160 segments (sag 13.5838 m, stretched length 301.6339 m)
TEST(ConductorInputs, ElasticCatenaryOfALevelSpan)
{
    const double w = (1.628 - 1.2 * 0.25 * 3.14159265358979323846 * 0.0281 * 0.0281) * 9.81;
    const auto cat = erf_conductors::elastic_catenary(300.0, 301.5, w, 3.0e7);
    EXPECT_NEAR(cat.sag, 13.5841, 1.0e-4 * 13.5841);
    EXPECT_NEAR(cat.horizontal_tension, 13256.4, 1.0e-4 * 13256.4);
    EXPECT_NEAR(cat.end_tension, 13473.3, 1.0e-4 * 13473.3);
    EXPECT_NEAR(cat.stretched_length, 301.6340, 1.0e-6 * 301.634);
    // inextensible in the limit of a stiff line: sag 13.0131 m, more than the parabola's 12.9904 m
    const auto stiff = erf_conductors::elastic_catenary(300.0, 301.5, w, 1.0e15);
    EXPECT_NEAR(stiff.sag, 13.0131, 1.0e-4 * 13.0131);
    // H L / EA = 4e-9 m; a float holds 301.5 to 3e-5 m
    EXPECT_NEAR(stiff.stretched_length, 301.5, (std::is_same<amrex::Real, float>::value) ? 1.0e-4 : 1.0e-6);
    // no slack: the line is held by its stretch alone, which matches the parabola's extra length
    // 8 d^2 / 3c to the stretch H c / EA with H = w c^2 / 8d, so d^3 = 3 w c^4 / (64 EA): 5.868 m
    const auto taut = erf_conductors::elastic_catenary(300.0, 300.0, w, 3.0e7);
    EXPECT_NEAR(taut.sag, std::cbrt(3.0 * w * std::pow(300.0, 4) / (64.0 * 3.0e7)), 0.01 * taut.sag);
    EXPECT_GT(taut.stretched_length, 300.0);
}

TEST(ConductorInputs, NothingIsReadWithoutLines)
{
    // a line listed, then an empty list after it: the last list counts, and an empty one switches the module off
    set_span("S0");
    amrex::ParmParse pp("erf.conductors");
    pp.addarr("lines", std::vector<std::string>{});
    EXPECT_FALSE(ConductorInputs::read().active);
}

TEST(ConductorInputs, EveryLineValueOutsideItsRangeIsRefusedByName)
{
    EXPECT_TRUE(ConductorInputs::validate_line(good_span()).empty());
    auto bad = [](auto mutate, const std::string& key) {
        LineInputs s = good_span();
        mutate(s);
        const std::string err = ConductorInputs::validate_line(s);
        EXPECT_FALSE(err.empty()) << key;
        EXPECT_NE(err.find("erf.conductors.S." + key), std::string::npos) << err;
    };
    bad([](LineInputs& s) { s.end_b = s.end_a; }, "end_a");
    bad([](LineInputs& s) { s.lengths = {300.0}; }, "length");      // equal to the chord: no slack
    bad([](LineInputs& s) { s.lengths = {250.0}; }, "length");
    bad([](LineInputs& s) { s.diameter = 0.0; }, "diameter");
    bad([](LineInputs& s) { s.mass_per_length = -1.0; }, "mass_per_length");
    bad([](LineInputs& s) { s.axial_stiffness = 0.0; }, "axial_stiffness");
    bad([](LineInputs& s) { s.drag_coefficient = -0.1; }, "drag_coefficient");
    bad([](LineInputs& s) { s.damping_ratio = 0.0; }, "damping_ratio");
    bad([](LineInputs& s) { s.damping_ratio = 1.5; }, "damping_ratio");
    bad([](LineInputs& s) { s.segments = 1; }, "segments");
    // every Real input must be finite
    const amrex::Real nan = std::numeric_limits<amrex::Real>::quiet_NaN();
    const amrex::Real inf = std::numeric_limits<amrex::Real>::infinity();
    bad([nan](LineInputs& s) { s.end_a[1] = nan; }, "end_a must be finite");
    bad([inf](LineInputs& s) { s.end_b[2] = inf; }, "end_b must be finite");
    bad([nan](LineInputs& s) { s.lengths = {nan}; }, "length must be finite");
    bad([inf](LineInputs& s) { s.stringing_tension = inf; }, "stringing_tension must be finite");
    bad([inf](LineInputs& s) { s.diameter = inf; }, "diameter must be finite");
    bad([nan](LineInputs& s) { s.mass_per_length = nan; }, "mass_per_length must be finite");
    bad([inf](LineInputs& s) { s.axial_stiffness = inf; }, "axial_stiffness must be finite");
    bad([nan](LineInputs& s) { s.drag_coefficient = nan; }, "drag_coefficient must be finite");
    bad([nan](LineInputs& s) { s.damping_ratio = nan; }, "damping_ratio must be finite");
    bad([nan](LineInputs& s) { s.insulator_length = nan; }, "insulator_length must be finite");
    bad([nan](LineInputs& s) { s.insulator_mass = nan; }, "insulator_mass must be finite");
    bad([inf](LineInputs& s) { s.insulator_diameter = inf; }, "insulator_diameter must be finite");
}

TEST(ConductorInputs, SharedSettingsOutsideTheirRangeAreRefusedByName)
{
    ConductorInputs in;
    in.lines.push_back(good_span());
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
    bad([](ConductorInputs& c) { c.flashover_distance = 0.0; }, "flashover_distance");
    bad([](ConductorInputs& c) { c.moordyn_cfl = 0.2; }, "moordyn_cfl");   // runs, but more than doubles the sag
    {
        ConductorInputs c = in;
        c.moordyn_cfl = ConductorInputs::max_moordyn_cfl;
        EXPECT_TRUE(ConductorInputs::validate_settings(c).empty()) << "the bound itself is accepted";
    }
    bad([](ConductorInputs& c) { c.moordyn_log_level = 4; }, "moordyn_log_level");
    bad([](ConductorInputs& c) { c.moordyn_log_level = -1; }, "moordyn_log_level");
    bad([](ConductorInputs& c) { c.surface_offset = 0.0; }, "surface_offset");
    bad([](ConductorInputs& c) { c.stats_start = -1.0; }, "stats_start");
    bad([](ConductorInputs& c) { c.node_output_int = -1; }, "node_output_int");
    bad([](ConductorInputs& c) { c.epsilon = 0.0; }, "epsilon");
    // every Real input must be finite
    const amrex::Real nan = std::numeric_limits<amrex::Real>::quiet_NaN();
    const amrex::Real inf = std::numeric_limits<amrex::Real>::infinity();
    bad([inf](ConductorInputs& c) { c.air_density = inf; }, "erf.conductors.air_density must be finite");
    bad([nan](ConductorInputs& c) { c.moordyn_dt = nan; }, "erf.conductors.moordyn_dt must be finite");
    bad([inf](ConductorInputs& c) { c.surface_offset = inf; }, "erf.conductors.surface_offset must be finite");
    bad([nan](ConductorInputs& c) { c.stats_start = nan; }, "erf.conductors.stats_start must be finite");
    bad([inf](ConductorInputs& c) { c.epsilon = inf; }, "erf.conductors.epsilon must be finite");
    bad([inf](ConductorInputs& c) { c.flashover_distance = inf; }, "erf.conductors.flashover_distance must be finite");
    bad([nan](ConductorInputs& c) { c.has_prescribed_velocity = true; c.prescribed_velocity[1] = nan; },
        "erf.conductors.prescribed_velocity must be finite");
    {
        ConductorInputs c = in;
        c.prescribed_velocity[1] = nan;
        EXPECT_TRUE(ConductorInputs::validate_settings(c).empty()) << "an unused prescribed velocity is not read";
    }
    // a conductor lighter than the air it displaces (7.6e-4 kg/m for 28.1 mm in 1.225 kg/m^3)
    bad([](ConductorInputs& c) { c.lines[0].mass_per_length = 5.0e-4; }, "erf.conductors.S.mass_per_length");
}

TEST(ConductorInputs, AnchorLevelMustExistAndFpeTrapsAreRefused)
{
    EXPECT_TRUE(ConductorInputs::validate_solver(1, 1, false).empty());
    EXPECT_TRUE(ConductorInputs::validate_solver(1, 0, false).empty());
    EXPECT_NE(ConductorInputs::validate_solver(1, 2, false).find("anchor_level"), std::string::npos);
    EXPECT_NE(ConductorInputs::validate_solver(0, 0, true).find("fpe_trap"), std::string::npos);
    EXPECT_EQ(ConductorInputs::resolve_anchor_level(-1, 2), 2);
    EXPECT_EQ(ConductorInputs::resolve_anchor_level(1, 2), 1);
}

TEST(ConductorInputs, SurfaceOffsetMustHoldTheWholeDomain)
{
    // MoorDyn's free surface at ERF z = surface_offset above the top, its bottom at -surface_offset below the floor
    EXPECT_TRUE(ConductorInputs::validate_frame(10000.0, 0.0, 400.0).empty());
    EXPECT_TRUE(ConductorInputs::validate_frame(500.0, -100.0, 400.0).empty());
    const std::string top = ConductorInputs::validate_frame(300.0, 0.0, 400.0);
    EXPECT_NE(top.find("erf.conductors.surface_offset"), std::string::npos) << top;
    EXPECT_NE(top.find("geometry.prob_hi[2]"), std::string::npos) << top;
    EXPECT_FALSE(ConductorInputs::validate_frame(400.0, 0.0, 400.0).empty()) << "the top itself is refused";
    const std::string bottom = ConductorInputs::validate_frame(500.0, -600.0, 400.0);
    EXPECT_NE(bottom.find("erf.conductors.surface_offset"), std::string::npos) << bottom;
    EXPECT_NE(bottom.find("geometry.prob_lo[2]"), std::string::npos) << bottom;
}

namespace {
LineInputs good_section ()
{
    LineInputs s = good_span();
    s.end_b = {{1000.0, 500.0, 30.0}};
    s.towers = {{{400.0, 500.0, 30.0}}, {{700.0, 500.0, 30.0}}};
    s.lengths = {301.5, 301.5, 301.5};
    s.insulator_length = 2.5;
    s.insulator_mass = 60.0;
    return s;
}
} // namespace

TEST(ConductorInputs, ASectionIsReadWithItsTowersLengthsAndInsulatorStrings)
{
    amrex::ParmParse pp("erf.conductors");
    pp.add("lines", std::string("C1"));
    amrex::ParmParse ps("erf.conductors.C1");
    ps.addarr("end_a", std::vector<amrex::Real>{100.0, 500.0, 30.0});
    ps.addarr("end_b", std::vector<amrex::Real>{1000.0, 500.0, 30.0});
    ps.addarr("towers", std::vector<amrex::Real>{400.0, 500.0, 30.0, 700.0, 500.0, 32.0});
    ps.addarr("length", std::vector<amrex::Real>{301.5, 301.6, 301.7});
    ps.add("diameter", 0.0281);
    ps.add("mass_per_length", 1.628);
    ps.add("axial_stiffness", 3.0e7);
    ps.add("insulator_length", 2.5);
    ps.add("insulator_mass", 60.0);
    const ConductorInputs in = ConductorInputs::read();
    ASSERT_EQ(in.lines.size(), 1u);
    const LineInputs& s = in.lines[0];
    EXPECT_EQ(s.num_spans(), 3);
    ASSERT_EQ(s.towers.size(), 2u);
    EXPECT_DOUBLE_EQ(s.point(2)[2], 32.0);
    EXPECT_DOUBLE_EQ(s.point(3)[0], 1000.0);
    EXPECT_DOUBLE_EQ(s.lengths[2], amrex::Real(301.7));
    constexpr double chord_tol = (std::is_same<amrex::Real, float>::value) ? 1.0e-4 : 1.0e-9;
    EXPECT_NEAR(s.chord(1), std::sqrt(300.0 * 300.0 + 4.0), chord_tol);
    EXPECT_TRUE(s.has_insulators());
    EXPECT_DOUBLE_EQ(s.insulator_diameter, amrex::Real(0.254));
    EXPECT_EQ(s.num_line_nodes(), 3 * 21 + 2 * (LineInputs::insulator_segments + 1));
    EXPECT_EQ(s.span_root(1), "conductors/C1_span2");
    EXPECT_EQ(s.span_name(0), "C1_span1");
    EXPECT_DOUBLE_EQ(in.flashover_distance, 1.0);
    pp.addarr("lines", std::vector<std::string>{});
    // a single span keeps its plain names
    EXPECT_EQ(good_span().span_root(0), good_span().output_root);
    EXPECT_EQ(good_span().span_name(0), "S");
}

TEST(ConductorInputs, SectionValuesOutsideTheirRangeAreRefusedByName)
{
    EXPECT_TRUE(ConductorInputs::validate_line(good_section()).empty());
    auto bad = [](auto mutate, const std::string& key) {
        LineInputs s = good_section();
        mutate(s);
        const std::string err = ConductorInputs::validate_line(s);
        EXPECT_FALSE(err.empty()) << key;
        EXPECT_NE(err.find("erf.conductors.S." + key), std::string::npos) << err;
    };
    bad([](LineInputs& s) { s.lengths = {301.5, 301.5}; }, "length");        // one per span
    bad([](LineInputs& s) { s.lengths[1] = 299.0; }, "length");               // span 2 without slack
    bad([](LineInputs& s) { s.towers[0] = s.end_a; }, "the attachment points");
    bad([](LineInputs& s) { s.insulator_length = -1.0; }, "insulator_length");
    bad([](LineInputs& s) { s.insulator_mass = 0.0; }, "insulator_mass");
    bad([](LineInputs& s) { s.insulator_diameter = 0.0; }, "insulator_diameter");
    bad([](LineInputs& s) { s.insulator_length = 31.0; }, "insulator_length");  // longer than the tower is high
    // the attachment points either side of a tower at the same x, y: a string's across-line direction is undefined
    bad([](LineInputs& s) { s.towers[1] = {{100.0, 500.0, 30.0}}; s.lengths = {301.5, 301.5, 901.0}; },
        "towers: the attachment points either side of tower 1");
    // string keys given for a line clamped at its towers
    bad([](LineInputs& s) { s.insulator_length = 0.0; s.insulator_mass_given = true; }, "insulator_mass is given but insulator_length is 0");
    bad([](LineInputs& s) { s.insulator_length = 0.0; s.insulator_mass = 0.0; s.insulator_diameter_given = true; },
        "insulator_diameter is given but insulator_length is 0");
    // strings hang only at towers: a single span is dead-ended at both ends
    LineInputs single = good_span();
    single.insulator_length = 2.5;
    single.insulator_mass = 60.0;
    const std::string err = ConductorInputs::validate_line(single);
    EXPECT_NE(err.find("insulator_length needs towers"), std::string::npos) << err;
    // clamped at the towers: no strings, no mass needed
    LineInputs clamped = good_section();
    clamped.insulator_length = 0.0;
    clamped.insulator_mass = 0.0;
    EXPECT_TRUE(ConductorInputs::validate_line(clamped).empty());
    EXPECT_FALSE(clamped.has_insulators());
    EXPECT_EQ(clamped.num_line_nodes(), 3 * 21);
}

TEST(ConductorInputs, TransformersAreReadWithTheirFootprintsAndDefaults)
{
    set_span("L1");
    amrex::ParmParse pp("erf.conductors");
    pp.addarr("transformers", std::vector<std::string>{"T1", "T2"});
    amrex::ParmParse t1("erf.conductors.T1");
    t1.addarr("position", std::vector<amrex::Real>{100.0, 500.0});
    t1.addarr("size", std::vector<amrex::Real>{8.0, 5.0, 6.0});
    t1.add("allowable_force", 2.0e4);
    t1.add("allowable_moment", 1.5e5);
    amrex::ParmParse t2("erf.conductors.T2");
    t2.addarr("position", std::vector<amrex::Real>{400.0, 500.0});
    t2.addarr("size", std::vector<amrex::Real>{6.0, 4.0, 5.0});
    const ConductorInputs in = ConductorInputs::read();
    ASSERT_EQ(in.transformers.size(), 2u);
    const TransformerInputs& a = in.transformers[0];
    EXPECT_EQ(a.name, "T1");
    EXPECT_DOUBLE_EQ(a.size[2], 6.0);
    EXPECT_DOUBLE_EQ(a.allowable_force, amrex::Real(2.0e4));
    EXPECT_DOUBLE_EQ(a.allowable_moment, amrex::Real(1.5e5));
    // unchecked unless given
    EXPECT_DOUBLE_EQ(in.transformers[1].allowable_force, 0.0);
    EXPECT_DOUBLE_EQ(in.transformers[1].allowable_moment, 0.0);
    // the footprint, its edges included
    EXPECT_TRUE(a.on_footprint(100.0, 500.0));
    EXPECT_TRUE(a.on_footprint(104.0, 502.5));
    EXPECT_TRUE(a.on_footprint(96.0, 497.5));
    EXPECT_FALSE(a.on_footprint(104.1, 500.0));
    EXPECT_FALSE(a.on_footprint(100.0, 497.4));
    pp.addarr("transformers", std::vector<std::string>{});
    pp.addarr("lines", std::vector<std::string>{});
}

TEST(ConductorInputs, TransformerValuesOutsideTheirRangeAreRefusedByName)
{
    TransformerInputs good;
    good.name = "T";
    good.position = {{100.0, 500.0}};
    good.size = {{8.0, 5.0, 6.0}};
    EXPECT_TRUE(ConductorInputs::validate_transformer(good).empty());
    auto bad = [&good](auto mutate, const std::string& key) {
        TransformerInputs t = good;
        mutate(t);
        const std::string err = ConductorInputs::validate_transformer(t);
        EXPECT_FALSE(err.empty()) << key;
        EXPECT_NE(err.find("erf.conductors.T." + key), std::string::npos) << err;
    };
    bad([](TransformerInputs& t) { t.size[0] = 0.0; }, "size");
    bad([](TransformerInputs& t) { t.size[1] = -1.0; }, "size");
    bad([](TransformerInputs& t) { t.size[2] = 0.0; }, "size");
    bad([](TransformerInputs& t) { t.allowable_force = -1.0; }, "allowable_force");
    bad([](TransformerInputs& t) { t.allowable_moment = -1.0; }, "allowable_moment");
    const amrex::Real nan = std::numeric_limits<amrex::Real>::quiet_NaN();
    bad([nan](TransformerInputs& t) { t.position[0] = nan; }, "position must be finite");
    bad([nan](TransformerInputs& t) { t.size[2] = nan; }, "size must be finite");
    bad([nan](TransformerInputs& t) { t.allowable_force = nan; }, "allowable_force must be finite");
    bad([nan](TransformerInputs& t) { t.allowable_moment = nan; }, "allowable_moment must be finite");
}

TEST(ConductorInputs, TheSlackIsCheckedBetweenTheEndsWhereTheyStandOnTheTerrain)
{
    // a short span down a hillside: a dead end 10 m above a summit at 60 m and the first tower,
    // 30 m tall, 40 m away on ground 40 m lower. Above the terrain the ends are 20 m apart in
    // height, on the terrain 20 m the other way: the same 44.7 m chord, and 44.9 m has slack
    LineInputs s = good_span();
    s.end_a = {{100.0, 500.0, 10.0}};
    s.end_b = {{140.0, 500.0, 30.0}};
    s.lengths = {44.9};
    EXPECT_TRUE(ConductorInputs::validate_line(s, false).empty());
    LineInputs placed = s;
    placed.end_a[2] += 60.0;   // the summit
    placed.end_b[2] += 20.0;   // the hillside
    EXPECT_TRUE(ConductorInputs::validate_slack(placed, true).empty());
    // the tower on ground 30 m below the summit instead: 10 m apart in height on the terrain, a
    // 41.2 m chord, so 42 m has slack there although the heights above the terrain give 44.7 m
    s.lengths = {42.0};
    EXPECT_FALSE(ConductorInputs::validate_slack(s).empty()) << "the heights above the terrain alone refuse it";
    EXPECT_TRUE(ConductorInputs::validate_line(s, false).empty()) << "read() leaves the slack to the placement";
    placed = s;
    placed.end_a[2] += 60.0;
    placed.end_b[2] += 30.0;
    EXPECT_TRUE(ConductorInputs::validate_slack(placed, true).empty());
    // and a span with too little slack on the terrain is refused there, naming it
    placed.lengths = {41.0};
    const std::string err = ConductorInputs::validate_slack(placed, true);
    EXPECT_NE(err.find("erf.conductors.S.length"), std::string::npos) << err;
    EXPECT_NE(err.find("where they stand on the terrain"), std::string::npos) << err;
}

TEST(ConductorInputs, TheSlackIsCheckedBetweenTheBottomsOfTheStrings)
{
    // the first span runs from a dead end at 30 m to the bottom of a 2.5 m string under a tower top
    // at 30 m: 300 m across and 2.5 m down, 300.0104 m. 300.005 m has slack between the attachment
    // points but none between the points the conductor hangs from
    LineInputs s = good_section();
    s.lengths = {amrex::Real(300.005), 301.5, 301.5};
    EXPECT_GT(s.lengths[0], s.chord(0)) << "longer than the chord between the attachment points";
    const std::string err = ConductorInputs::validate_slack(s);
    EXPECT_NE(err.find("erf.conductors.S.length of span 1"), std::string::npos) << err;
    EXPECT_NE(err.find("the bottoms of the insulator strings"), std::string::npos) << err;
    s.lengths[0] = 300.1;
    EXPECT_TRUE(ConductorInputs::validate_slack(s).empty()) << ConductorInputs::validate_slack(s);
}

TEST(ConductorInputs, AStringingTensionSetsEachSpansLengthSoThatItHangsWithThatTension)
{
    LineInputs s = good_section();
    s.lengths.clear();
    s.stringing_tension = 2.0e4;
    EXPECT_TRUE(ConductorInputs::validate_line(s).empty()) << "no lengths needed with a stringing tension";
    // spans of 300 m, 100 m (a tower moved up the line) and 500 m: each must hang with H = 20 kN
    s.towers[0][0] = 400.0;
    s.towers[1][0] = 500.0;
    const double w = s.mass_per_length * 9.81;
    s.lengths_from_stringing_tension(amrex::Real(w));
    ASSERT_EQ(static_cast<int>(s.lengths.size()), s.num_spans());
    for (int k = 0; k < s.num_spans(); ++k) {
        const double c = s.chord(k);
        const double L = s.lengths[static_cast<std::size_t>(k)];
        // level spans: the parabola's 8 d^2 / 3c with d = w c^2 / 8H, stretched by H / EA
        const double d = w * c * c / (8.0 * 2.0e4);
        EXPECT_NEAR(L, (c + 8.0 * d * d / (3.0 * c)) / (1.0 + 2.0e4 / s.axial_stiffness), 1.0e-4 * c) << k;
        // the elastic catenary of that unstretched length carries the stringing tension (the 100 m
        // span is taut: shorter than its chord unstretched, held by its stretch)
        const auto cat = erf_conductors::elastic_catenary(s.chord(k), s.lengths[static_cast<std::size_t>(k)],
                                                          amrex::Real(w), s.axial_stiffness);
        EXPECT_NEAR(cat.horizontal_tension / 2.0e4, 1.0, 0.01) << "span " << k << " of " << c << " m";
    }
    EXPECT_TRUE(ConductorInputs::validate_slack(s, true).empty()) << "a strung line is not held to slack";
    // on strings the conductor hangs from their bottoms: a span from a dead end up to a tower is
    // measured to 2.5 m below the tower top, which on a steep short span changes the chord by more
    // than the span's slack
    LineInputs hung = s;
    hung.end_a = {{100.0, 500.0, 10.0}};
    hung.towers[0] = {{140.0, 500.0, 40.0}};
    hung.insulator_length = 2.5;
    hung.insulator_mass = 60.0;
    hung.lengths_from_stringing_tension(amrex::Real(w));
    EXPECT_EQ(hung.conductor_point(1)[2], amrex::Real(37.5));
    EXPECT_EQ(hung.conductor_point(0)[2], amrex::Real(10.0)) << "a dead end has no string";
    const double c0 = std::sqrt(40.0 * 40.0 + 27.5 * 27.5);
    EXPECT_NEAR(hung.lengths[0], (c0 + w * w * std::pow(40.0, 4) / (24.0 * 4.0e8 * c0)) / (1.0 + 2.0e4 * c0 / (s.axial_stiffness * 40.0)),
                1.0e-5 * c0);
    auto bad = [](auto mutate, const std::string& key) {
        LineInputs t = good_section();
        t.lengths.clear();
        t.stringing_tension = 2.0e4;
        mutate(t);
        const std::string err = ConductorInputs::validate_line(t);
        EXPECT_FALSE(err.empty()) << key;
        EXPECT_NE(err.find("erf.conductors.S." + key), std::string::npos) << err;
    };
    bad([](LineInputs& t) { t.stringing_tension = -1.0; }, "stringing_tension");
    bad([](LineInputs& t) { t.lengths = {301.5, 301.5, 301.5}; }, "length and stringing_tension");
    bad([](LineInputs& t) { t.stringing_tension = 0.0; }, "length");   // neither given
    // without a stringing tension the lengths are left alone
    LineInputs fixed = good_section();
    const auto before = fixed.lengths;
    fixed.lengths_from_stringing_tension(amrex::Real(w));
    EXPECT_EQ(fixed.lengths, before);
}

TEST(ConductorInputs, TowerTypesAreReadAndALinesTowerTypeMustNameOne)
{
    amrex::ParmParse pp("erf.conductors");
    pp.add("lines", std::string("C2"));
    amrex::ParmParse ps("erf.conductors.C2");
    ps.addarr("end_a", std::vector<amrex::Real>{100.0, 500.0, 30.0});
    ps.addarr("end_b", std::vector<amrex::Real>{700.0, 500.0, 30.0});
    ps.addarr("towers", std::vector<amrex::Real>{400.0, 500.0, 30.0});
    ps.addarr("length", std::vector<amrex::Real>{301.5, 301.5});
    ps.add("diameter", 0.0281);
    ps.add("mass_per_length", 1.628);
    ps.add("axial_stiffness", 3.0e7);
    ps.add("tower_type", std::string("suspension"));
    pp.addarr("tower_types", std::vector<std::string>{"suspension"});
    amrex::ParmParse pt("erf.conductors.suspension");
    pt.add("base_width", 6.0);
    pt.add("top_width", 1.5);
    pt.add("solidity", 0.2);
    pt.add("arm_length", 12.0);
    pt.add("weight", 9.0e4);
    pt.add("allowable_uplift", 1.0e5);
    const ConductorInputs in = ConductorInputs::read();
    ASSERT_EQ(in.tower_types.size(), 1u);
    const auto& t = in.tower_types[0];
    EXPECT_EQ(t.name, "suspension");
    EXPECT_DOUBLE_EQ(t.arm_length, 12.0);
    EXPECT_EQ(t.segments, 10);
    EXPECT_DOUBLE_EQ(t.peak, 0.0);
    EXPECT_DOUBLE_EQ(t.arm_face(), amrex::Real(1.5)) << "arm_depth defaults to top_width";
    EXPECT_DOUBLE_EQ(t.weight, amrex::Real(9.0e4));
    EXPECT_DOUBLE_EQ(t.allowable_uplift, amrex::Real(1.0e5));
    EXPECT_DOUBLE_EQ(t.allowable_compression, 0.0);
    EXPECT_DOUBLE_EQ(t.legs(), amrex::Real(6.0)) << "the legs at the base width";
    EXPECT_EQ(in.lines[0].tower_type, "suspension");
    // a line's tower type must be one of the types, and the line must have towers
    LineInputs s = in.lines[0];
    EXPECT_TRUE(ConductorInputs::validate_tower_type(s, in.tower_types).empty());
    s.tower_type = "dead_end";
    EXPECT_NE(ConductorInputs::validate_tower_type(s, in.tower_types).find("erf.conductors.C2.tower_type = dead_end is not one of"),
              std::string::npos);
    LineInputs single = good_span();
    single.tower_type = "suspension";
    EXPECT_NE(ConductorInputs::validate_tower_type(single, in.tower_types).find("erf.conductors.S.tower_type needs towers"),
              std::string::npos);
    LineInputs none = good_span();
    EXPECT_TRUE(ConductorInputs::validate_tower_type(none, in.tower_types).empty()) << "no tower type: points only";
    pp.addarr("lines", std::vector<std::string>{});
    pp.addarr("tower_types", std::vector<std::string>{});
    ps.remove("tower_type");
}

TEST(ConductorInputs, ALineSharesTheTowersOfAnotherLineWithATowerType)
{
    erf_towers::TowerType lat;
    lat.name = "lat";
    lat.base_width = 6.0; lat.top_width = 1.5; lat.solidity = 0.2; lat.arm_length = 12.0;
    ConductorInputs in;
    in.tower_types = {lat};
    LineInputs owner = good_section();
    owner.name = "P2";
    owner.tower_type = "lat";
    LineInputs phase = good_section();
    phase.name = "P1";
    phase.share_towers = "P2";
    in.lines = {owner, phase};
    EXPECT_TRUE(ConductorInputs::validate_shared_towers(in.lines).empty()) << ConductorInputs::validate_shared_towers(in.lines);
    // the sharing line's towers are the owner's: their type, and whether they move
    EXPECT_EQ(&in.tower_owner(in.lines[1]), &in.lines[0]);
    ASSERT_NE(in.tower_type(in.lines[1]), nullptr);
    EXPECT_EQ(in.tower_type(in.lines[1])->name, "lat");
    EXPECT_FALSE(in.towers_move(in.lines[1]));
    in.tower_types[0].frequency = 2.0;
    in.tower_types[0].weight = 9.0e4;
    EXPECT_TRUE(in.towers_move(in.lines[1])) << "a line on moving towers moves with them";
    auto refused = [&](std::vector<LineInputs> spans, const std::string& what) {
        const std::string err = ConductorInputs::validate_shared_towers(spans);
        EXPECT_NE(err.find("erf.conductors.P1.share_towers"), std::string::npos) << what << ": " << err;
        EXPECT_NE(err.find(what), std::string::npos) << err;
    };
    LineInputs p = phase;
    p.share_towers = "P9";
    refused({owner, p}, "is not another line");
    p.share_towers = "P1";
    refused({owner, p}, "is not another line");
    LineInputs bare = owner;
    bare.tower_type.clear();
    refused({bare, phase}, "has no tower_type");
    p = phase;
    p.tower_type = "lat";
    refused({owner, p}, "drop its own tower_type");
    p = phase;
    p.towers.pop_back();
    refused({owner, p}, "hangs from every tower");
    LineInputs chained = owner;
    chained.tower_type.clear();
    chained.share_towers = "P3";
    LineInputs third = good_section();
    third.name = "P3";
    third.tower_type = "lat";
    refused({chained, phase, third}, "name the line the towers belong to");
}

TEST(ConductorInputs, EveryLineNeedsAnOutputRootOfItsOwn)
{
    LineInputs a = good_span();
    a.name = "A";
    a.output_root = "conductors/A";
    LineInputs b = good_span();
    b.name = "B";
    b.output_root = "conductors/B";
    EXPECT_TRUE(ConductorInputs::validate_output_roots({a, b}).empty());
    b.output_root = a.output_root;
    const std::string err = ConductorInputs::validate_output_roots({a, b});
    EXPECT_NE(err.find("erf.conductors.B.output_root = conductors/A is also A's"), std::string::npos) << err;
}

TEST(ConductorInputs, TheFirstNonFiniteValueIsNamedWithItsPoint)
{
    std::vector<amrex::Real> v{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_TRUE(erf_conductors::first_nonfinite(v, 3, "the wind (m/s) at point").empty());
    v[4] = std::numeric_limits<amrex::Real>::quiet_NaN();
    const std::string err = erf_conductors::first_nonfinite(v, 3, "the wind (m/s) at point");
    EXPECT_EQ(err.rfind("the wind (m/s) at point 1 is not finite (4", 0), 0u) << err;
    const std::vector<double> d{0.0, std::numeric_limits<double>::infinity()};
    const std::string derr = erf_conductors::first_nonfinite(d, 1, "entry");
    EXPECT_NE(derr.find("entry 1 is not finite (inf)"), std::string::npos) << derr;
}

namespace {
// the required keys of a single-span line under erf.conductors.<name>
void add_line_block (const std::string& name)
{
    amrex::ParmParse ps("erf.conductors." + name);
    ps.addarr("end_a", std::vector<amrex::Real>{100.0, 500.0, 30.0});
    ps.addarr("end_b", std::vector<amrex::Real>{400.0, 500.0, 30.0});
    ps.add("length", 301.5);
    ps.add("diameter", 0.0281);
    ps.add("mass_per_length", 1.628);
    ps.add("axial_stiffness", 3.0e7);
}

// read() after setup, with aborts turned into exceptions: the abort's message, empty when read()
// returned. The shared keys a setup may add are removed again, so that the next test starts clean.
std::string read_after (const std::function<void()>& setup)
{
    setup();
    const std::string msg = erf_gtest::abort_message([] { ConductorInputs::read(); });
    amrex::ParmParse pp("erf.conductors");
    for (const char* key : {"lines", "transformers", "tower_types", "prescribed_velocity"}) { pp.remove(key); }
    return msg;
}
} // namespace

TEST(ConductorInputs, ReadRefusesMalformedInputsNamingTheKey)
{
    amrex::ParmParse pp("erf.conductors");
    // start without the shared keys other tests may have left
    for (const char* key : {"lines", "transformers", "tower_types", "prescribed_velocity"}) { pp.remove(key); }
    struct Case { const char* expected; std::function<void()> setup; };
    const std::vector<Case> cases = {
        {"erf.conductors.tower_types needs lines", [&] { pp.addarr("tower_types", std::vector<std::string>{"RT"}); }},
        {"erf.conductors.transformers needs lines", [&] { pp.addarr("transformers", std::vector<std::string>{"RX"}); }},
        {"erf.conductors.lines lists 'RA' twice", [&] {
            add_line_block("RA");
            pp.addarr("lines", std::vector<std::string>{"RA", "RA"});
        }},
        {"erf.conductors.prescribed_velocity needs three components", [&] {
            add_line_block("RB");
            pp.add("lines", std::string("RB"));
            pp.addarr("prescribed_velocity", std::vector<amrex::Real>{0.0, 15.0});
        }},
        {"erf.conductors.RC.end_a and end_b need three components", [&] {
            add_line_block("RC");
            amrex::ParmParse("erf.conductors.RC").addarr("end_a", std::vector<amrex::Real>{100.0, 500.0});
            pp.add("lines", std::string("RC"));
        }},
        {"erf.conductors.RD.towers needs three components (m) per tower", [&] {
            add_line_block("RD");
            amrex::ParmParse("erf.conductors.RD").addarr("towers", std::vector<amrex::Real>{250.0, 500.0, 30.0, 300.0});
            pp.add("lines", std::string("RD"));
        }},
        {"erf.conductors.RE.insulator_mass is given but insulator_length is 0", [&] {
            add_line_block("RE");
            amrex::ParmParse("erf.conductors.RE").add("insulator_mass", 60.0);
            pp.add("lines", std::string("RE"));
        }},
        {"erf.conductors.RG.output_root = shared is also RF's", [&] {
            add_line_block("RF");
            add_line_block("RG");
            amrex::ParmParse("erf.conductors.RF").add("output_root", std::string("shared"));
            amrex::ParmParse("erf.conductors.RG").add("output_root", std::string("shared"));
            pp.addarr("lines", std::vector<std::string>{"RF", "RG"});
        }},
        {"erf.conductors: 'RH' names two lines, transformers or tower types", [&] {
            add_line_block("RH");
            pp.add("lines", std::string("RH"));
            pp.addarr("tower_types", std::vector<std::string>{"RH"});
        }},
        {"erf.conductors.RJ.position needs two components", [&] {
            add_line_block("RI");
            pp.add("lines", std::string("RI"));
            pp.addarr("transformers", std::vector<std::string>{"RJ"});
            amrex::ParmParse pt("erf.conductors.RJ");
            pt.addarr("position", std::vector<amrex::Real>{100.0});
            pt.addarr("size", std::vector<amrex::Real>{8.0, 5.0, 6.0});
        }},
        {"erf.conductors: 'RM' names two lines, transformers or tower types", [&] {
            add_line_block("RM");
            pp.add("lines", std::string("RM"));
            pp.addarr("transformers", std::vector<std::string>{"RM"});
        }},
        {"erf.conductors.RO.size needs three components", [&] {
            add_line_block("RN");
            pp.add("lines", std::string("RN"));
            pp.addarr("transformers", std::vector<std::string>{"RO"});
            amrex::ParmParse pt("erf.conductors.RO");
            pt.addarr("position", std::vector<amrex::Real>{100.0, 500.0});
            pt.addarr("size", std::vector<amrex::Real>{8.0, 5.0});
        }},
        {"erf.conductors.RK.diameter must be finite", [&] {
            add_line_block("RK");
            amrex::ParmParse("erf.conductors.RK").add("diameter", std::numeric_limits<double>::quiet_NaN());
            pp.add("lines", std::string("RK"));
        }},
    };
    for (const auto& c : cases) {
        const std::string msg = read_after(c.setup);
        EXPECT_NE(msg.find(c.expected), std::string::npos) << "expected \"" << c.expected << "\", got \"" << msg << "\"";
    }
    // a well-formed line is read without an abort
    EXPECT_TRUE(read_after([&] { add_line_block("RL"); pp.add("lines", std::string("RL")); }).empty());
}
