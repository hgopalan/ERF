// Contract of erf_conductors::moordyn_input_text: the span is written in MoorDyn's frame
// (heights lowered by the surface offset), with the air density, gravity, external kinematics,
// a flat bottom below everything, both attachments fixed, the line's properties and segments;
// MoorDyn (the stub or the real library) accepts the file and puts the end nodes on the
// attachments, and to_erf_frame brings the positions back.

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "ERF_ConductorInputs.H"
#include "ERF_MoorDynInputWriter.H"
#include "ERF_MoorDynSystem.H"

using erf_conductors::ConductorInputs;
using erf_conductors::SpanInputs;

namespace {

SpanInputs span ()
{
    SpanInputs s;
    s.name = "S1";
    s.end_a = {{100.0, 500.0, 30.0}};
    s.end_b = {{400.0, 500.0, 40.0}};
    s.length = 301.5; s.diameter = 0.0281; s.mass_per_length = 1.628; s.axial_stiffness = 3.0e7;
    s.drag_coefficient = 1.1; s.damping_ratio = 0.4; s.segments = 16;
    return s;
}

ConductorInputs settings ()
{
    ConductorInputs in;
    in.air_density = 1.15;
    in.surface_offset = 5000.0;
    in.moordyn_dt = 0.002;
    in.moordyn_cfl = 0.12;
    return in;
}

bool has_row (const std::string& text, const std::string& row) { return text.find(row) != std::string::npos; }

} // namespace

TEST(MoorDynInputWriter, TheSpanIsWrittenInMoorDynsFrameWithAirAndExternalKinematics)
{
    const std::string text = erf_conductors::moordyn_input_text(span(), settings(), 9.81);
    EXPECT_TRUE(has_row(text, "LINE TYPES"));
    EXPECT_TRUE(has_row(text, "POINT PROPERTIES"));
    EXPECT_TRUE(has_row(text, "LINES"));
    EXPECT_TRUE(has_row(text, "OPTIONS"));
    // the line type row: diameter, mass, EA, -damping ratio (MoorDyn's "-zeta" form), EI 0, Cd
    EXPECT_TRUE(has_row(text, "S1   0.0281   1.628   30000000   -0.4   0   1.1")) << text;
    // the attachments, fixed, lowered by the surface offset
    EXPECT_TRUE(has_row(text, "1     Fixed     100   500   -4970")) << text;
    EXPECT_TRUE(has_row(text, "2     Fixed     400   500   -4960")) << text;
    // the line: type, attachments 1 and 2, length, segments
    EXPECT_TRUE(has_row(text, "1     S1      1        2         301.5   16")) << text;
    // the options
    EXPECT_TRUE(has_row(text, "0.002   dtM")) << text;
    EXPECT_TRUE(has_row(text, "0.12   CFL")) << text;
    EXPECT_TRUE(has_row(text, "9.81   g")) << text;
    EXPECT_TRUE(has_row(text, "1.15   WtrDnsty")) << text;
    EXPECT_TRUE(has_row(text, "10000   WtrDpth")) << text;
    EXPECT_TRUE(has_row(text, "1             WaveKin")) << text;
    EXPECT_TRUE(has_row(text, "0             ICgenDynamic")) << text;
    // no MoorDyn step when none is set: MoorDyn chooses from its CFL
    ConductorInputs no_dt = settings();
    no_dt.moordyn_dt = 0.0;
    EXPECT_FALSE(has_row(erf_conductors::moordyn_input_text(span(), no_dt, 9.81), "dtM"));
}

TEST(MoorDynInputWriter, MoorDynAcceptsTheFileAndTheEndsComeBackInERFsFrame)
{
    const auto dir = std::filesystem::temp_directory_path() / "erf_gtest_moordyn_writer";
    const std::string fname = (dir / "S1.moordyn.txt").string();
    const SpanInputs s = span();
    const ConductorInputs in = settings();
    erf_conductors::write_moordyn_input(fname, s, in, 9.81);
    ASSERT_TRUE(std::filesystem::exists(fname));

    std::string err;
    auto sys = erf_moordyn::MoorDynSystem::create(fname, "", MOORDYN_ERR_LEVEL, err);
    ASSERT_TRUE(sys) << err;
    EXPECT_EQ(sys->num_coupled_dof(), 0u);
    err = sys->init({}, {});
    ASSERT_TRUE(err.empty()) << err;
    ASSERT_EQ(sys->num_lines(), 1u);
    const unsigned nn = sys->line_num_nodes(1);
    EXPECT_EQ(nn, 17u);
    const auto a = erf_conductors::to_erf_frame(sys->line_node_position(1, 0), in.surface_offset);
    const auto b = erf_conductors::to_erf_frame(sys->line_node_position(1, nn - 1), in.surface_offset);
    for (int d = 0; d < 3; ++d) {
        EXPECT_NEAR(a[static_cast<std::size_t>(d)], s.end_a[static_cast<std::size_t>(d)], 1.0e-6) << "end a, dir " << d;
        EXPECT_NEAR(b[static_cast<std::size_t>(d)], s.end_b[static_cast<std::size_t>(d)], 1.0e-6) << "end b, dir " << d;
    }
    ASSERT_GT(sys->external_kinematics_init(err), 0u) << err;
}
