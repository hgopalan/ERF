// Contract of erf_conductors::ConductorInputs: a span block is read with its defaults and its
// derived chord and catenary sag; a section over towers is read with a length per span and its
// insulator strings; every value outside its documented range is refused with a message naming
// the key; the shared settings and the solver settings are checked the same way; and the
// transformers the lines end on are read with their footprints, and refused by name out of range.

#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <gtest/gtest.h>

#include "ERF_ConductorInputs.H"

using erf_conductors::ConductorInputs;
using erf_conductors::SpanInputs;
using erf_conductors::TransformerInputs;

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
    s.lengths = {301.5}; s.diameter = 0.0281; s.mass_per_length = 1.628; s.axial_stiffness = 3.0e7;
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
    EXPECT_NEAR(stiff.stretched_length, 301.5, 1.0e-6);   // H L / EA = 4e-9 m
    // no slack: the line is held by its stretch alone, which matches the parabola's extra length
    // 8 d^2 / 3c to the stretch H c / EA with H = w c^2 / 8d, so d^3 = 3 w c^4 / (64 EA): 5.874 m
    const auto taut = erf_conductors::elastic_catenary(300.0, 300.0, w, 3.0e7);
    EXPECT_NEAR(taut.sag, std::cbrt(3.0 * w * std::pow(300.0, 4) / (64.0 * 3.0e7)), 0.01 * taut.sag);
    EXPECT_GT(taut.stretched_length, 300.0);
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
    bad([](SpanInputs& s) { s.lengths = {300.0}; }, "length");      // equal to the chord: no slack
    bad([](SpanInputs& s) { s.lengths = {250.0}; }, "length");
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
    bad([](ConductorInputs& c) { c.flashover_distance = 0.0; }, "flashover_distance");
    bad([](ConductorInputs& c) { c.moordyn_cfl = 0.2; }, "moordyn_cfl");   // runs, but more than doubles the sag
    {
        ConductorInputs c = in;
        c.moordyn_cfl = ConductorInputs::max_moordyn_cfl;
        EXPECT_TRUE(ConductorInputs::validate_settings(c).empty()) << "the bound itself is accepted";
    }
    bad([](ConductorInputs& c) { c.moordyn_log_level = 4; }, "moordyn_log_level");
    bad([](ConductorInputs& c) { c.surface_offset = 0.0; }, "surface_offset");
    bad([](ConductorInputs& c) { c.surface_offset = 20.0; }, "surface_offset");   // below the attachments
    bad([](ConductorInputs& c) { c.stats_start = -1.0; }, "stats_start");
    bad([](ConductorInputs& c) { c.node_output_int = -1; }, "node_output_int");
    bad([](ConductorInputs& c) { c.epsilon = 0.0; }, "epsilon");
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

namespace {
SpanInputs good_section ()
{
    SpanInputs s = good_span();
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
    pp.add("spans", std::string("C1"));
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
    ASSERT_EQ(in.spans.size(), 1u);
    const SpanInputs& s = in.spans[0];
    EXPECT_EQ(s.num_spans(), 3);
    ASSERT_EQ(s.towers.size(), 2u);
    EXPECT_DOUBLE_EQ(s.point(2)[2], 32.0);
    EXPECT_DOUBLE_EQ(s.point(3)[0], 1000.0);
    EXPECT_DOUBLE_EQ(s.lengths[2], amrex::Real(301.7));
    constexpr double chord_tol = std::is_same<amrex::Real, float>::value ? 1.0e-4 : 1.0e-9;
    EXPECT_NEAR(s.chord(1), std::sqrt(300.0 * 300.0 + 4.0), chord_tol);
    EXPECT_TRUE(s.has_insulators());
    EXPECT_DOUBLE_EQ(s.insulator_diameter, amrex::Real(0.254));
    EXPECT_EQ(s.num_line_nodes(), 3 * 21 + 2 * (SpanInputs::insulator_segments + 1));
    EXPECT_EQ(s.span_root(1), "conductors/C1_span2");
    EXPECT_EQ(s.span_name(0), "C1_span1");
    EXPECT_DOUBLE_EQ(in.flashover_distance, 1.0);
    pp.addarr("spans", std::vector<std::string>{});
    // a single span keeps its plain names
    EXPECT_EQ(good_span().span_root(0), good_span().output_root);
    EXPECT_EQ(good_span().span_name(0), "S");
}

TEST(ConductorInputs, SectionValuesOutsideTheirRangeAreRefusedByName)
{
    EXPECT_TRUE(ConductorInputs::validate_span(good_section()).empty());
    auto bad = [](auto mutate, const std::string& key) {
        SpanInputs s = good_section();
        mutate(s);
        const std::string err = ConductorInputs::validate_span(s);
        EXPECT_FALSE(err.empty()) << key;
        EXPECT_NE(err.find("erf.conductors.S." + key), std::string::npos) << err;
    };
    bad([](SpanInputs& s) { s.lengths = {301.5, 301.5}; }, "length");        // one per span
    bad([](SpanInputs& s) { s.lengths[1] = 299.0; }, "length");               // span 2 without slack
    bad([](SpanInputs& s) { s.towers[0] = s.end_a; }, "the attachment points");
    bad([](SpanInputs& s) { s.insulator_length = -1.0; }, "insulator_length");
    bad([](SpanInputs& s) { s.insulator_mass = 0.0; }, "insulator_mass");
    bad([](SpanInputs& s) { s.insulator_diameter = 0.0; }, "insulator_diameter");
    bad([](SpanInputs& s) { s.insulator_length = 31.0; }, "insulator_length");  // longer than the tower is high
    // strings hang only at towers: a single span is dead-ended at both ends
    SpanInputs single = good_span();
    single.insulator_length = 2.5;
    single.insulator_mass = 60.0;
    const std::string err = ConductorInputs::validate_span(single);
    EXPECT_NE(err.find("insulator_length needs towers"), std::string::npos) << err;
    // clamped at the towers: no strings, no mass needed
    SpanInputs clamped = good_section();
    clamped.insulator_length = 0.0;
    clamped.insulator_mass = 0.0;
    EXPECT_TRUE(ConductorInputs::validate_span(clamped).empty());
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
    pp.addarr("spans", std::vector<std::string>{});
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
}

TEST(ConductorInputs, TheSlackIsCheckedBetweenTheEndsWhereTheyStandOnTheTerrain)
{
    // a short span down a hillside: a dead end 10 m above a summit at 60 m and the first tower,
    // 30 m tall, 40 m away on ground 40 m lower. Above the terrain the ends are 20 m apart in
    // height, on the terrain 20 m the other way: the same 44.7 m chord, and 44.9 m has slack
    SpanInputs s = good_span();
    s.end_a = {{100.0, 500.0, 10.0}};
    s.end_b = {{140.0, 500.0, 30.0}};
    s.lengths = {44.9};
    EXPECT_TRUE(ConductorInputs::validate_span(s, false).empty());
    SpanInputs placed = s;
    placed.end_a[2] += 60.0;   // the summit
    placed.end_b[2] += 20.0;   // the hillside
    EXPECT_TRUE(ConductorInputs::validate_slack(placed, true).empty());
    // the tower on ground 30 m below the summit instead: 10 m apart in height on the terrain, a
    // 41.2 m chord, so 42 m has slack there although the heights above the terrain give 44.7 m
    s.lengths = {42.0};
    EXPECT_FALSE(ConductorInputs::validate_slack(s).empty()) << "the heights above the terrain alone refuse it";
    EXPECT_TRUE(ConductorInputs::validate_span(s, false).empty()) << "read() leaves the slack to the placement";
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

TEST(ConductorInputs, AStringingTensionSetsEachSpansLengthSoThatItHangsWithThatTension)
{
    SpanInputs s = good_section();
    s.lengths.clear();
    s.stringing_tension = 2.0e4;
    EXPECT_TRUE(ConductorInputs::validate_span(s).empty()) << "no lengths needed with a stringing tension";
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
    SpanInputs hung = s;
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
        SpanInputs t = good_section();
        t.lengths.clear();
        t.stringing_tension = 2.0e4;
        mutate(t);
        const std::string err = ConductorInputs::validate_span(t);
        EXPECT_FALSE(err.empty()) << key;
        EXPECT_NE(err.find("erf.conductors.S." + key), std::string::npos) << err;
    };
    bad([](SpanInputs& t) { t.stringing_tension = -1.0; }, "stringing_tension");
    bad([](SpanInputs& t) { t.lengths = {301.5, 301.5, 301.5}; }, "length and stringing_tension");
    bad([](SpanInputs& t) { t.stringing_tension = 0.0; }, "length");   // neither given
    // without a stringing tension the lengths are left alone
    SpanInputs fixed = good_section();
    const auto before = fixed.lengths;
    fixed.lengths_from_stringing_tension(amrex::Real(w));
    EXPECT_EQ(fixed.lengths, before);
}
