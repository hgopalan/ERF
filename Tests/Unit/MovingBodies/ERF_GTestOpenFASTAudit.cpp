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

// every kind of point spread with one width (m)
KernelWidths same (Real eps) { return {eps, eps, eps}; }

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
    auto f = erf_openfast::audit_geometry(t, b, same(100.0), plo, phi, dx, per, 0.0);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "cells across the rotor diameter"));
    // on 10 m cells the default 16 ring points sit 47 m apart at the tip, over two 20 m kernel widths: a note
    f = erf_openfast::audit_geometry(t, b, same(20.0), plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "ring points"));
    // with 48 points nothing is left to say
    b.num_points_t = 48;
    EXPECT_TRUE(erf_openfast::audit_geometry(t, b, same(20.0), plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0).empty());
    b.num_points_t = 16;
    // the rotor cuts the ground / the top: fatal
    f = erf_openfast::audit_geometry(rotor("T1", {{750.0, 600.0, 100.0}}, 120.0, 10), b, same(20.0), plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "below the ground"));
    // on terrain the ground under the hub is the terrain surface: a 150 m hub with a 120 m rotor clears
    // flat ground but not a 100 m hill beneath it
    EXPECT_EQ(fatal_count(erf_openfast::audit_geometry(t, b, same(20.0), plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0)), 0);
    f = erf_openfast::audit_geometry(t, b, same(20.0), plo, phi, {{10.0, 10.0, 10.0}}, per, 100.0);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "terrain surface at z = 100"));
    f = erf_openfast::audit_geometry(rotor("T1", {{750.0, 600.0, 500.0}}, 120.0, 10), b, same(20.0), plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "above the domain top"));
    // the base outside the domain: fatal
    b.base_pos = {{-10.0, 600.0, 0.0}};
    f = erf_openfast::audit_geometry(t, b, same(20.0), plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "outside the domain in x"));
    b.base_pos = {{750.0, 600.0, 0.0}};
    // a non-periodic y boundary: the rotor near it warns about the kernel, across it is fatal
    const std::array<int,3> per_y{{1, 0, 0}};
    f = erf_openfast::audit_geometry(rotor("T1", {{750.0, 150.0, 150.0}}, 120.0, 10), b, same(20.0), plo, phi, {{10.0, 10.0, 10.0}}, per_y, 0.0);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "kernel"));
    f = erf_openfast::audit_geometry(rotor("T1", {{750.0, 100.0, 150.0}}, 120.0, 10), b, same(20.0), plo, phi, {{10.0, 10.0, 10.0}}, per_y, 0.0);
    EXPECT_EQ(fatal_count(f), 1); EXPECT_TRUE(mentions(f, "non-periodic y boundary"));
    // an actuator line whose points are farther apart than the kernel; a kernel under a cell
    b.mode = "alm";
    f = erf_openfast::audit_geometry(rotor("T1", {{750.0, 600.0, 150.0}}, 120.0, 4), b, same(20.0), plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "farther apart than the kernel"));
    f = erf_openfast::audit_geometry(rotor("T1", {{750.0, 600.0, 150.0}}, 120.0, 50), b, same(5.0), plo, phi, {{10.0, 10.0, 10.0}}, per, 0.0);
    EXPECT_EQ(fatal_count(f), 0); EXPECT_TRUE(mentions(f, "narrower than the smallest cell"));
}

// Every kernel the body spreads with is audited under its own key: a tower or nacelle kernel under a
// cell warns, the tower's force points farther apart than its kernel warn, and a nacelle drag area
// above 2 pi epsilon_nacelle^2 is fatal (its kernel correction factor falls under 0.5, where the linear
// estimate fails); widths of points that are not forced, or of tower nodes the model lacks, say nothing.
TEST(OpenFASTAudit, EveryKernelInUseIsAuditedUnderItsKey)
{
    const std::array<Real,3> plo{{0.0, 0.0, 0.0}}, phi{{3000.0, 1200.0, 600.0}}, dx{{10.0, 10.0, 10.0}};
    const std::array<int,3> per{{1, 1, 0}};
    MovingBodyInputs b = body("stub.fst", "alm");
    b.base_pos = {{750.0, 600.0, 0.0}};
    // 50 blade nodes 2.4 m apart under a 20 m kernel; 10 tower nodes 15 m apart from the base to the hub
    auto t = rotor("T1", {{750.0, 600.0, 150.0}}, 120.0, 50);
    const int ntw = 10;
    t.num_force_pts_tower = ntw;
    t.num_force_nodes += ntw;
    for (int k = 0; k < ntw; ++k) { t.force_pos.insert(t.force_pos.end(), {Real(750.0), Real(600.0), Real(15.0 * k)}); }
    t.force.resize(t.force_pos.size(), 0.0);
    EXPECT_TRUE(erf_openfast::audit_geometry(t, b, {20.0, 5.0, 2.0}, plo, phi, dx, per, 0.0).empty());   // tower and nacelle off
    b.num_force_points_tower = ntw;
    b.nacelle_cd = 1.0;
    b.nacelle_area = 60.0;
    EXPECT_TRUE(erf_openfast::audit_geometry(t, b, {20.0, 20.0, 20.0}, plo, phi, dx, per, 0.0).empty());
    auto f = erf_openfast::audit_geometry(t, b, {20.0, 5.0, 20.0}, plo, phi, dx, per, 0.0);
    EXPECT_EQ(fatal_count(f), 0);
    EXPECT_TRUE(mentions(f, "epsilon_tower = 5 m is narrower than the smallest cell"));
    EXPECT_TRUE(mentions(f, "tower force points up to 15 m apart"));
    EXPECT_FALSE(mentions(f, "the kernel width epsilon ="));
    // cd area 60 m^2 against 2 pi eps^2: 39.3 m^2 at 2.5 m (factor 0.24, fatal), 56.5 m^2 at 3 m (0.47, fatal),
    // 61.6 m^2 at 3.13 m (0.51, fine)
    f = erf_openfast::audit_geometry(t, b, {20.0, 20.0, 2.5}, plo, phi, dx, per, 0.0);
    EXPECT_EQ(fatal_count(f), 1);
    EXPECT_TRUE(mentions(f, "exceeds 2 pi epsilon_nacelle^2"));
    EXPECT_TRUE(mentions(f, "erf.moving_bodies.T1.epsilon_nacelle"));
    EXPECT_TRUE(mentions(f, "epsilon_nacelle = 2.5 m is narrower than the smallest cell"));
    EXPECT_EQ(fatal_count(erf_openfast::audit_geometry(t, b, {20.0, 20.0, 3.0}, plo, phi, dx, per, 0.0)), 1);
    EXPECT_EQ(fatal_count(erf_openfast::audit_geometry(t, b, {20.0, 20.0, 3.13}, plo, phi, dx, per, 0.0)), 0);
    // a model without tower force nodes spreads none, whatever num_force_points_tower asked for: its width
    // is not audited, and the run is told the tower is not forced
    {
        auto t0 = rotor("T1", {{750.0, 600.0, 150.0}}, 120.0, 50);
        f = erf_openfast::audit_geometry(t0, b, {20.0, 5.0, 20.0}, plo, phi, dx, per, 0.0);
        EXPECT_FALSE(mentions(f, "epsilon_tower = "));
        EXPECT_TRUE(mentions(f, "the model has no tower force nodes"));
    }
    // mode = none forces nothing: only the rotor's width is looked at
    b.mode = "none";
    f = erf_openfast::audit_geometry(t, b, {20.0, 5.0, 2.0}, plo, phi, dx, per, 0.0);
    EXPECT_EQ(fatal_count(f), 0);
    EXPECT_FALSE(mentions(f, "epsilon_tower"));
    EXPECT_FALSE(mentions(f, "epsilon_nacelle"));
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

// The overlap audit of the discs alone (as the I/O rank gets them from the owners) is the audit of
// the turbines; the text a non-I/O owner sends is what report_audit prints; and the rotor radius
// read before OpenFAST starts (for the owner's footprint) comes from TipRad, or the stub's
// rotor_radius, or is negative when the files give none.
TEST(OpenFASTAudit, DiscsTextsAndTheRadiusBeforeOpenFASTStarts)
{
    std::vector<erf_openfast::TurbineState> farm{rotor("T1", {{800.0, 800.0, 150.0}}, 120.0, 10),
                                                 rotor("T2", {{800.0, 1000.0, 150.0}}, 120.0, 10)};
    std::vector<erf_openfast::RotorDisc> discs;
    for (const auto& t : farm) { discs.push_back({t.name, t.hub_pos, t.hub_axis, Real(120.0)}); }
    const auto a = erf_openfast::audit_overlap(farm);
    const auto b = erf_openfast::audit_overlap(discs);
    ASSERT_EQ(a.size(), 1u);
    ASSERT_EQ(b.size(), 1u);
    EXPECT_EQ(a[0].message, b[0].message);
    discs[1].hub = {{800.0, 1520.0, 150.0}};
    EXPECT_TRUE(erf_openfast::audit_overlap(discs).empty());
    bool fatal = false;
    const std::string text = erf_openfast::format_audit("T1 (model and geometry)", a, fatal);
    EXPECT_TRUE(fatal);
    EXPECT_EQ(text, "OpenFAST input audit for T1 (model and geometry): 1 finding(s)\n  ERROR: " + a[0].message + "\n");
    fatal = false;
    erf_openfast::format_audit("T2", {{false, "a note"}}, fatal);
    EXPECT_FALSE(fatal);

    const auto dir = scratch_dir("radius");
    write(dir, "ed.dat", "------- ELASTODYN INPUT FILE -------\n      120.97   TipRad      - The distance from the rotor apex to the blade tip (meters)\n");
    const std::string fst = write(dir, "model.fst", "------- OpenFAST INPUT FILE -------\n\"ed.dat\"    EDFile      - ElastoDyn input file\n");
    EXPECT_NEAR(erf_openfast::model_rotor_radius(fst), Real(120.97), Real(1.0e-4));
    const std::string stub = write(dir, "stub.fst", "dt = 0.01\nrotor_radius = 63.0\nhub_height = 90.0\n");
    EXPECT_NEAR(erf_openfast::model_rotor_radius(stub), Real(63.0), Real(1.0e-6));
    const std::string none = write(dir, "none.fst", "dt = 0.01\n");
    EXPECT_LT(erf_openfast::model_rotor_radius(none), Real(0.0));
    EXPECT_LT(erf_openfast::model_rotor_radius((dir / "missing.fst").string()), Real(0.0));
    // the time step: DT of an OpenFAST primary file, the stub's dt, or negative
    const std::string dt_fst = write(dir, "dt.fst", "------- OpenFAST INPUT FILE -------\n         60   TMax        - Total run time (s)\n"
                                                    "      0.005   DT          - Recommended module time step (s)\n");
    EXPECT_NEAR(erf_openfast::model_time_step(dt_fst), Real(0.005), Real(1.0e-9));
    const std::string stub_dt = write(dir, "stub_dt.fst", "num_blades = 3\ndt = 0.02\n");
    EXPECT_NEAR(erf_openfast::model_time_step(stub_dt), Real(0.02), Real(1.0e-9));
    EXPECT_LT(erf_openfast::model_time_step(write(dir, "nodt.fst", "num_blades = 3\n")), Real(0.0));
    EXPECT_LT(erf_openfast::model_time_step((dir / "missing.fst").string()), Real(0.0));
}
