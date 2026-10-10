// The start-up audit of an OpenFAST turbine against the ERF set-up: the model's air density,
// aerodynamics, gravity and tower aerodynamics from its files; the rotor against the domain
// and the mesh; overlapping rotors; ERF's density at the hub; the rotor facing the wind. Fatal
// findings are those a run must not start with.

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

#include <gtest/gtest.h>

#include "ERF_MovingBodiesInputs.H"
#include "ERF_OpenFASTAudit.H"
#include "ERF_OpenFASTDriver.H"

namespace {

using amrex::Real;
using erf_openfast::AuditFinding;

std::filesystem::path scratch_dir (const std::string& tag)
{
    const auto dir = std::filesystem::temp_directory_path() / ("erf_gtest_audit_" + tag);
    std::filesystem::create_directories(dir);
    return dir;
}

std::string write (const std::filesystem::path& dir, const std::string& name, const std::string& body)
{
    std::ofstream out(dir / name);
    out << body;
    return (dir / name).string();
}

int fatal_count (const std::vector<AuditFinding>& f)
{
    int n = 0;
    for (const auto& x : f) { if (x.fatal) { ++n; } }
    return n;
}

bool mentions (const std::vector<AuditFinding>& f, const std::string& text)
{
    for (const auto& x : f) { if (x.message.find(text) != std::string::npos) { return true; } }
    return false;
}

// a rotor of tip radius R at hub, axis along +x, three blades of `nodes` force nodes
erf_openfast::TurbineState rotor (const std::string& name, const std::array<Real,3>& hub, Real R, int nodes)
{
    erf_openfast::TurbineState t;
    t.name = name;
    t.num_blades = 3;
    t.num_force_pts_blade = nodes;
    t.num_force_nodes = 1 + 3 * nodes;
    t.hub_pos = hub;
    t.hub_axis = {{1.0, 0.0, 0.0}};
    t.force_pos.assign(3 * t.num_force_nodes, 0.0);
    t.force.assign(3 * t.num_force_nodes, 0.0);
    for (int d = 0; d < 3; ++d) { t.force_pos[d] = hub[d]; }
    for (int b = 0; b < 3; ++b) {
        for (int i = 0; i < nodes; ++i) {
            const int nd = 1 + b * nodes + i;
            const Real r = (i + 1.0) / nodes * R;   // the last node at the tip
            const Real th = 2.0 * 3.14159265358979323846 * b / 3.0;
            t.force_pos[3*nd]   = hub[0];
            t.force_pos[3*nd+1] = hub[1] + r * std::cos(th);
            t.force_pos[3*nd+2] = hub[2] + r * std::sin(th);
        }
    }
    return t;
}

MovingBodyInputs body (const std::string& fst, const std::string& mode = "adm")
{
    MovingBodyInputs b;
    b.name = "T1";
    b.type = "openfast_turbine";
    b.fst_file = fst;
    b.base_pos = {{500.0, 500.0, 0.0}};
    b.mode = mode;
    b.num_force_points_blade = 10;
    b.air_density = 1.225;
    return b;
}

} // namespace

TEST(OpenFASTAudit, ReadsDensityGravityAndModuleFilesFromTheModel)
{
    const auto dir = scratch_dir("files");
    write(dir, "ad_num.dat", "1.1      AirDens  - Air density (kg/m^3)\nTrue  TwrAero  - x\n");
    write(dir, "ad_def.dat", "\"default\"  AirDens  - Air density\nFalse  TwrAero  - x\n");
    const std::string fst_num = write(dir, "num.fst", "9.80665  Gravity  - g\n1.225  AirDens  - rho\n2  CompAero  - AeroDyn\n\"ad_num.dat\"  AeroFile  - x\n");
    const std::string fst_def = write(dir, "def.fst", "9.80665  Gravity  - g\n\"default\"  AirDens  - rho\n2  CompAero  - AeroDyn\n\"ad_num.dat\"  AeroFile  - x\n");
    const std::string fst_none = write(dir, "none.fst", "2  CompAero  - AeroDyn\n\"ad_def.dat\"  AeroFile  - x\n");
    const std::string stub = write(dir, "stub.fst", "dt = 0.01\nnum_blades = 3\n");
    Real rho = 0.0, g = 0.0;
    EXPECT_TRUE(erf_openfast::openfast_air_density(fst_num, rho)); EXPECT_NEAR(rho, 1.225, 1.0e-6);
    EXPECT_TRUE(erf_openfast::openfast_air_density(fst_def, rho)); EXPECT_NEAR(rho, 1.1, 1.0e-6);   // AeroDyn's when the primary says default
    EXPECT_FALSE(erf_openfast::openfast_air_density(fst_none, rho));                                    // neither gives a number
    EXPECT_TRUE(erf_openfast::openfast_gravity(fst_num, g)); EXPECT_NEAR(g, 9.80665, 1.0e-5);
    EXPECT_FALSE(erf_openfast::openfast_gravity(stub, g));
    EXPECT_TRUE(erf_openfast::is_openfast_model(fst_num));
    EXPECT_TRUE(erf_openfast::is_openfast_model(fst_none));
    EXPECT_FALSE(erf_openfast::is_openfast_model(stub));
    EXPECT_EQ(erf_openfast::openfast_module_path(fst_num, "AeroFile"), (dir / "ad_num.dat").string());
    EXPECT_TRUE(erf_openfast::openfast_module_path(stub, "AeroFile").empty());
}

TEST(OpenFASTAudit, ModelChecksFlagDensityAerodynamicsGravityAndTowerAero)
{
    const auto dir = scratch_dir("model");
    write(dir, "ad_on.dat", "1.225  AirDens  - rho\nTrue  TwrAero  - x\n");
    write(dir, "ad_off.dat", "1.225  AirDens  - rho\nFalse  TwrAero  - x\n");
    const std::string good = write(dir, "good.fst", "9.81  Gravity  - g\n1.225  AirDens  - rho\n2  CompAero  - AeroDyn\n\"ad_on.dat\"  AeroFile  - x\n");
    const std::string lowg = write(dir, "lowg.fst", "9.0  Gravity  - g\n1.225  AirDens  - rho\n2  CompAero  - AeroDyn\n\"ad_on.dat\"  AeroFile  - x\n");
    const std::string noaero = write(dir, "noaero.fst", "9.81  Gravity  - g\n1.225  AirDens  - rho\n0  CompAero  - off\n\"ad_on.dat\"  AeroFile  - x\n");
    const std::string twroff = write(dir, "twroff.fst", "9.81  Gravity  - g\n1.225  AirDens  - rho\n2  CompAero  - AeroDyn\n\"ad_off.dat\"  AeroFile  - x\n");
    const std::string stub = write(dir, "stub.fst", "dt = 0.01\n");
    // a consistent model: nothing
    EXPECT_TRUE(erf_openfast::audit_model(body(good), 9.81).empty());
    // the body's air_density must be the model's
    MovingBodyInputs b = body(good);
    b.air_density = 1.0;
    auto f = erf_openfast::audit_model(b, 9.81);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "air_density"));
    // aerodynamics off while the loads go into the flow: fatal; with mode = none: nothing
    f = erf_openfast::audit_model(body(noaero), 9.81);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "CompAero = 0"));
    EXPECT_TRUE(erf_openfast::audit_model(body(noaero, "none"), 9.81).empty());
    // gravity: a warning only
    f = erf_openfast::audit_model(body(lowg), 9.81);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_EQ(f.size(), 1u); EXPECT_TRUE(mentions(f, "Gravity"));
    // a forced tower with TwrAero off: a warning; without tower points: nothing
    b = body(twroff);
    b.num_force_points_tower = 10;
    f = erf_openfast::audit_model(b, 9.81);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "TwrAero"));
    EXPECT_TRUE(erf_openfast::audit_model(body(twroff), 9.81).empty());
    // the stub: one note, nothing fatal, even with a wrong density
    b = body(stub);
    b.air_density = 1.0;
    f = erf_openfast::audit_model(b, 9.81);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_EQ(f.size(), 1u); EXPECT_TRUE(mentions(f, "stub"));
}

TEST(OpenFASTAudit, GeometryChecksTheRotorAgainstTheDomainAndTheMesh)
{
    const std::array<Real,3> plo{{0.0, 0.0, 0.0}}, phi{{3000.0, 1200.0, 600.0}}, dx{{50.0, 50.0, 50.0}};
    const std::array<int,3> per{{1, 1, 0}};
    MovingBodyInputs b = body("stub.fst");
    b.base_pos = {{750.0, 600.0, 0.0}};
    // a 240 m rotor at 150 m on 50 m cells with a 100 m kernel: fine but coarse (4.8 cells across)
    auto t = rotor("T1", {{750.0, 600.0, 150.0}}, 120.0, 10);
    auto f = erf_openfast::audit_geometry(t, b, 100.0, plo, phi, dx, per, 0.0);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "cells across the rotor diameter"));
    // on 10 m cells the default 16 ring points sit 47 m apart at the tip, over two 20 m kernel widths: a note
    f = erf_openfast::audit_geometry(t, b, 20.0, plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "ring points"));
    // with 48 points nothing is left to say
    b.num_points_t = 48;
    EXPECT_TRUE(erf_openfast::audit_geometry(t, b, 20.0, plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0).empty());
    b.num_points_t = 16;
    // the rotor cuts the ground / the top: fatal
    f = erf_openfast::audit_geometry(rotor("T1", {{750.0, 600.0, 100.0}}, 120.0, 10), b, 20.0, plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "below the ground"));
    // on terrain the ground under the hub is the terrain surface: a 150 m hub with a 120 m rotor clears
    // flat ground but not a 100 m hill beneath it
    EXPECT_EQ(fatal_count(erf_openfast::audit_geometry(t, b, 20.0, plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0)), 0);
    f = erf_openfast::audit_geometry(t, b, 20.0, plo, phi, {{10.0, 10.0, 10.0}}, per, 100.0);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "terrain surface at z = 100"));
    f = erf_openfast::audit_geometry(rotor("T1", {{750.0, 600.0, 500.0}}, 120.0, 10), b, 20.0, plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "above the domain top"));
    // the base outside the domain: fatal
    b.base_pos = {{-10.0, 600.0, 0.0}};
    f = erf_openfast::audit_geometry(t, b, 20.0, plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "outside the domain in x"));
    b.base_pos = {{750.0, 600.0, 0.0}};
    // a non-periodic y boundary: the rotor near it warns about the kernel, across it is fatal
    const std::array<int,3> per_y{{1, 0, 0}};
    f = erf_openfast::audit_geometry(rotor("T1", {{750.0, 150.0, 150.0}}, 120.0, 10), b, 20.0, plo, phi, {{10.0, 10.0, 10.0}}, per_y, 0.0);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "kernel"));
    f = erf_openfast::audit_geometry(rotor("T1", {{750.0, 100.0, 150.0}}, 120.0, 10), b, 20.0, plo, phi, {{10.0, 10.0, 10.0}}, per_y, 0.0);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "non-periodic y boundary"));
    // an actuator line whose points are farther apart than the kernel; a kernel under a cell
    b.mode = "alm";
    f = erf_openfast::audit_geometry(rotor("T1", {{750.0, 600.0, 150.0}}, 120.0, 4), b, 20.0, plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "farther apart than the kernel"));
    f = erf_openfast::audit_geometry(rotor("T1", {{750.0, 600.0, 150.0}}, 120.0, 50), b, 5.0, plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "narrower than the smallest cell"));
}

TEST(OpenFASTAudit, OverlapDensityAndFacing)
{
    // two rotors 7 D apart: nothing; side by side 1 D apart: overlap
    std::vector<erf_openfast::TurbineState> farm{rotor("T1", {{800.0, 800.0, 150.0}}, 120.0, 10), rotor("T2", {{2480.0, 800.0, 150.0}}, 120.0, 10)};
    EXPECT_TRUE(erf_openfast::audit_overlap(farm).empty());
    farm[1] = rotor("T2", {{800.0, 1000.0, 150.0}}, 120.0, 10);
    auto f = erf_openfast::audit_overlap(farm);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "overlap"));
    // one rotor 1.5 D behind another on the same axis: the discs do not intersect; 0.4 D behind
    // they do (the second sits in the first's kernel)
    farm[1] = rotor("T2", {{1160.0, 800.0, 150.0}}, 120.0, 10);
    EXPECT_TRUE(erf_openfast::audit_overlap(farm).empty());
    farm[1] = rotor("T2", {{900.0, 800.0, 150.0}}, 120.0, 10);
    EXPECT_EQ(fatal_count(erf_openfast::audit_overlap(farm)), 1);
    // side by side but 3 D apart: nothing
    farm[1] = rotor("T2", {{800.0, 1520.0, 150.0}}, 120.0, 10);
    EXPECT_TRUE(erf_openfast::audit_overlap(farm).empty());
    // density: within the tolerance nothing, outside fatal
    EXPECT_TRUE(erf_openfast::audit_density("T1", 1.225, 1.20, 0.05).empty());
    f = erf_openfast::audit_density("T1", 1.225, 1.0, 0.05);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "density_tolerance"));
    EXPECT_TRUE(erf_openfast::audit_density("T1", 1.225, 1.0, 0.25).empty());
    // facing: into the wind nothing, 10 degrees nothing, 40 degrees a warning, away a warning
    const auto t = rotor("T1", {{750.0, 600.0, 150.0}}, 120.0, 10);
    EXPECT_TRUE(erf_openfast::audit_facing(t, {{10.0, 0.0, 0.0}}).empty());
    EXPECT_TRUE(erf_openfast::audit_facing(t, {{static_cast<Real>(10.0 * std::cos(0.17)), static_cast<Real>(10.0 * std::sin(0.17)), Real(0.0)}}).empty());
    f = erf_openfast::audit_facing(t, {{static_cast<Real>(10.0 * std::cos(0.7)), static_cast<Real>(10.0 * std::sin(0.7)), Real(0.0)}});
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "yawed"));
    f = erf_openfast::audit_facing(t, {{-10.0, 0.0, 0.0}});
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "faces away"));
    // still air: nothing to judge
    EXPECT_TRUE(erf_openfast::audit_facing(t, {{0.0, 0.0, 0.0}}).empty());
}
