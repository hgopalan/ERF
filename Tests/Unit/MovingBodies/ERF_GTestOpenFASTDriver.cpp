// Contracts of the OpenFAST driver and the erf.moving_bodies inputs, checked against the
// bundled stub library (Source/MovingBodies/OpenFAST/Stub), whose geometry and disk model are
// known in closed form.

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <gtest/gtest.h>

#include "ERF_MovingBodiesInputs.H"
#include "ERF_OpenFASTDriver.H"

namespace {

using amrex::Real;

constexpr double pi = 3.14159265358979323846;

// The stub deck used by the driver tests: an IEA-15-MW-sized rigid rotor.
struct StubDeck {
    double dt = 0.01;
    int num_blades = 3;
    int num_blade_nodes = 4;
    int num_tower_nodes = 0;
    double rotor_radius = 120.0;
    double hub_height = 150.0;
    double rotor_speed_rpm = 7.55;
    double ct = 0.8;
    double cp = 0.45;
    double air_density = 1.225;
    double tower_diameter = 0.0;   // > 0 with tower_cd: the stub puts a cylinder drag on its tower force nodes
    double tower_cd = 0.0;

    double omega () const { return rotor_speed_rpm * 2.0 * pi / 60.0; }
    double area () const { return pi * rotor_radius * rotor_radius; }
};

std::filesystem::path scratch_dir (const std::string& tag)
{
    const auto dir = std::filesystem::temp_directory_path() / ("erf_gtest_openfast_" + tag);
    std::filesystem::create_directories(dir);
    return dir;
}

std::string write_stub_deck (const std::filesystem::path& dir, const StubDeck& d)
{
    const auto fname = dir / "stub_turbine.fst";
    std::ofstream out(fname);
    out << "dt = " << d.dt << "\n"
        << "num_blades = " << d.num_blades << "\n"
        << "num_blade_nodes = " << d.num_blade_nodes << "\n"
        << "num_tower_nodes = " << d.num_tower_nodes << "\n"
        << "rotor_radius = " << d.rotor_radius << "\n"
        << "hub_height = " << d.hub_height << "\n"
        << "rotor_speed_rpm = " << d.rotor_speed_rpm << "\n"
        << "ct = " << d.ct << "\n"
        << "cp = " << d.cp << "\n"
        << "air_density = " << d.air_density << "\n"
        << "tower_diameter = " << d.tower_diameter << "\n"
        << "tower_cd = " << d.tower_cd << "\n";
    return fname.string();
}

MovingBodyInputs one_turbine (const std::string& fst, const std::filesystem::path& dir)
{
    MovingBodyInputs b;
    b.name = "T1";
    b.type = "openfast_turbine";
    b.fst_file = fst;
    b.base_pos = {{500.0, 500.0, 0.0}};
    b.mode = "adm";
    b.num_force_points_blade = 6;
    b.num_force_points_tower = 0;
    b.output_root = (dir / "T1").string();
    return b;
}

} // namespace

TEST(OpenFASTDriver, SubstepCountIsTheWholeRatio)
{
    std::string err;
    EXPECT_EQ(erf_openfast::substep_count(0.05, 0.01, err), 5);
    EXPECT_TRUE(err.empty()) << err;
    EXPECT_EQ(erf_openfast::substep_count(0.01, 0.01, err), 1);
    EXPECT_TRUE(err.empty()) << err;
    // a ratio that is whole up to roundoff is accepted: 0.3/0.1 is 2.9999999999999996 in binary
    EXPECT_EQ(erf_openfast::substep_count(0.3, 0.1, err), 3);
    EXPECT_TRUE(err.empty()) << err;
}

TEST(OpenFASTDriver, SubstepCountRefusesANonMultipleStep)
{
    std::string err;
    EXPECT_EQ(erf_openfast::substep_count(0.052, 0.01, err), 0);
    EXPECT_NE(err.find("whole multiple"), std::string::npos) << err;
    // an ERF step shorter than the OpenFAST step rounds to zero substeps
    err.clear();
    EXPECT_EQ(erf_openfast::substep_count(0.004, 0.01, err), 0);
    EXPECT_FALSE(err.empty());
    err.clear();
    EXPECT_EQ(erf_openfast::substep_count(0.05, 0.0, err), 0);
    EXPECT_NE(err.find("non-positive"), std::string::npos) << err;
    err.clear();
    EXPECT_EQ(erf_openfast::substep_count(-0.05, 0.01, err), 0);
    EXPECT_FALSE(err.empty());
}

TEST(OpenFASTDriver, InductionCheckReadsWakeModFromTheAeroDynFile)
{
    const auto dir = scratch_dir("induction");
    auto write = [&](const std::string& name, const std::string& text) {
        std::ofstream out(dir / name);
        out << text;
        return (dir / name).string();
    };
    const std::string ad_bem = write("ad_bem.dat", "------- AERODYN INPUT -------\n1   Wake_Mod  - Wake/induction model (switch)\n");
    const std::string ad_off = write("ad_off.dat", "------- AERODYN INPUT -------\n0   Wake_Mod  - Wake/induction model (switch)\n");
    const std::string ad_old = write("ad_old.dat", "1   WakeMod   - Type of wake/induction model\n");
    const std::string ad_none = write("ad_none.dat", "no such line here\n");
    auto fst = [&](const std::string& name, const std::string& comp, const std::string& ad) {
        return write(name, "2   CompAero  - Compute aerodynamic loads\n" + comp +
                     "\"" + ad + "\"  AeroFile  - Name of file containing aerodynamic input parameters\n");
    };
    // relative AeroFile names resolve against the .fst directory; absolute ones are used as given
    EXPECT_TRUE(erf_openfast::check_induction_off(fst("bem.fst", "", "ad_bem.dat")).find("Wake_Mod = 1") != std::string::npos);
    EXPECT_TRUE(erf_openfast::check_induction_off(fst("off.fst", "", ad_off)).empty());
    EXPECT_TRUE(erf_openfast::check_induction_off(fst("old.fst", "", "ad_old.dat")).find("Wake_Mod = 1") != std::string::npos);
    EXPECT_TRUE(erf_openfast::check_induction_off(fst("none.fst", "", "ad_none.dat")).find("no Wake_Mod") != std::string::npos);
    // no aerodynamics, or no AeroFile line (the stub deck): nothing to check
    EXPECT_TRUE(erf_openfast::check_induction_off(write("noaero.fst", "0   CompAero  - off\n\"ad_bem.dat\"  AeroFile  - x\n")).empty());
    EXPECT_TRUE(erf_openfast::check_induction_off(write("stub.fst", "dt = 0.01\nnum_blades = 3\n")).empty());
}

TEST(MovingBodiesInputs, ValidateSolverAcceptsAnelasticFixedStepNoTrapsOnAnyAnchorLevel)
{
    EXPECT_TRUE(MovingBodiesInputs::validate_solver(true, true, 0, 0, false).empty());
    EXPECT_NE(MovingBodiesInputs::validate_solver(false, true, 0, 0, false).find("anelastic"), std::string::npos);
    EXPECT_NE(MovingBodiesInputs::validate_solver(true, false, 0, 0, false).find("fixed"), std::string::npos);
    // two levels: the anchor may be either; a level that does not exist is refused
    EXPECT_TRUE(MovingBodiesInputs::validate_solver(true, true, 1, 1, false).empty());
    EXPECT_TRUE(MovingBodiesInputs::validate_solver(true, true, 1, 0, false).empty());
    EXPECT_NE(MovingBodiesInputs::validate_solver(true, true, 1, 2, false).find("anchor_level"), std::string::npos);
    EXPECT_NE(MovingBodiesInputs::validate_solver(true, true, 1, -1, false).find("anchor_level"), std::string::npos);
    // the default anchor is the finest level
    EXPECT_EQ(MovingBodiesInputs::resolve_anchor_level(-1, 0), 0);
    EXPECT_EQ(MovingBodiesInputs::resolve_anchor_level(-1, 2), 2);
    EXPECT_EQ(MovingBodiesInputs::resolve_anchor_level(1, 2), 1);
    // OpenFAST 4.2.1 raises floating-point exceptions in FAST_ProgStart; a trapped run dies there
    EXPECT_NE(MovingBodiesInputs::validate_solver(true, true, 0, 0, true).find("fpe_trap"), std::string::npos);
}

TEST(MovingBodiesInputs, ReadFillsEveryBodyFromItsBlock)
{
    {
        amrex::ParmParse pp("erf.moving_bodies");
        pp.addarr("bodies", std::vector<std::string>{"TA", "TB"});
        pp.add("diagnostics_int", 4);
        pp.addarr("prescribed_velocity", std::vector<Real>{10.59, 0.0, 0.0});
    }
    {
        amrex::ParmParse pa("erf.moving_bodies.TA");
        pa.add("type", std::string("openfast_turbine"));
        pa.add("fst_file", std::string("TA/turbine.fst"));
        pa.addarr("base_pos", std::vector<Real>{1000.0, 1000.0, 0.0});
        pa.add("mode", std::string("none"));
        pa.add("num_force_points_blade", 40);
        pa.add("num_points_t", 24);
        pa.add("output_root", std::string("out/TA"));
    }
    {
        amrex::ParmParse pb("erf.moving_bodies.TB");
        pb.add("type", std::string("openfast_turbine"));
        pb.add("fst_file", std::string("TB/turbine.fst"));
        pb.addarr("base_pos", std::vector<Real>{2680.0, 1000.0, 0.0});
    }

    const MovingBodiesInputs in = MovingBodiesInputs::read();
    ASSERT_TRUE(in.active);
    ASSERT_EQ(in.bodies.size(), 2u);
    EXPECT_EQ(in.diagnostics_int, 4);
    ASSERT_TRUE(in.has_prescribed_velocity);
    EXPECT_EQ(in.prescribed_velocity[0], static_cast<Real>(10.59));  // exact in float and double
    EXPECT_DOUBLE_EQ(in.prescribed_velocity[1], 0.0);

    const MovingBodyInputs& a = in.bodies[0];
    EXPECT_EQ(a.name, "TA");
    EXPECT_EQ(a.type, "openfast_turbine");
    EXPECT_EQ(a.fst_file, "TA/turbine.fst");
    EXPECT_DOUBLE_EQ(a.base_pos[0], 1000.0);
    EXPECT_DOUBLE_EQ(a.base_pos[2], 0.0);
    EXPECT_EQ(a.mode, "none");
    EXPECT_EQ(a.num_force_points_blade, 40);
    EXPECT_EQ(a.num_force_points_tower, 0);
    EXPECT_EQ(a.num_points_t, 24);
    EXPECT_EQ(a.output_root, "out/TA");

    // defaults: adm, 50 blade points, no tower, 16 ring points, output under the diagnostics directory
    const MovingBodyInputs& b = in.bodies[1];
    EXPECT_EQ(b.name, "TB");
    EXPECT_DOUBLE_EQ(b.base_pos[0], 2680.0);
    EXPECT_EQ(b.mode, "adm");
    EXPECT_EQ(b.num_force_points_blade, 50);
    EXPECT_EQ(b.num_force_points_tower, 0);
    EXPECT_EQ(b.num_points_t, 16);
    EXPECT_EQ(b.output_root, "moving_bodies/TB");
}

TEST(OpenFASTDriver, InitReportsTheStubNodeLayout)
{
    const StubDeck d;
    const auto dir = scratch_dir("layout");
    const std::string fst = write_stub_deck(dir, d);
    erf_openfast::OpenFASTDriver driver({one_turbine(fst, dir)});
    driver.init(0.05, 1.0);
    ASSERT_TRUE(driver.initialized());
    // the node layout is known before the first solution, so the flow can be sampled at it
    EXPECT_FALSE(driver.solved0());
    driver.set_uniform_velocity({{10.0, 0.0, 0.0}});
    driver.solution0();
    ASSERT_TRUE(driver.solved0());
    ASSERT_EQ(driver.turbines().size(), 1u);
    const erf_openfast::TurbineState& t = driver.turbines()[0];

    EXPECT_EQ(t.num_blades, d.num_blades);
    EXPECT_EQ(t.num_blade_elem, d.num_blade_nodes);
    EXPECT_EQ(t.num_tower_elem, 0);
    EXPECT_EQ(t.num_substeps, 5);
    EXPECT_DOUBLE_EQ(t.dt_fast, d.dt);
    // hub, then num_blades * nodes per blade; no tower nodes were requested
    EXPECT_EQ(t.num_vel_nodes, 1 + d.num_blades * d.num_blade_nodes);
    EXPECT_EQ(t.num_force_nodes, 1 + d.num_blades * 6);
    ASSERT_EQ(t.vel_pos.size(), 3u * t.num_vel_nodes);
    ASSERT_EQ(t.force_pos.size(), 3u * t.num_force_nodes);
    ASSERT_EQ(t.force.size(), 3u * t.num_force_nodes);

    // hub at the base plus the hub height; the interface carries float positions
    EXPECT_NEAR(t.hub_pos[0], 500.0, 1.0e-4);
    EXPECT_NEAR(t.hub_pos[1], 500.0, 1.0e-4);
    EXPECT_NEAR(t.hub_pos[2], 150.0, 1.0e-4);
    EXPECT_NEAR(t.vel_pos[0], 500.0, 1.0e-4);
    EXPECT_NEAR(t.vel_pos[2], 150.0, 1.0e-4);
    // blade 0 root node sits half a segment out along +y at zero azimuth
    const double r_root = 0.5 / d.num_blade_nodes * d.rotor_radius;
    EXPECT_NEAR(t.vel_pos[3 * 1 + 1] - t.hub_pos[1], r_root, 1.0e-3);
    EXPECT_NEAR(t.vel_pos[3 * 1 + 2] - t.hub_pos[2], 0.0, 1.0e-3);
    // every node lies in the rotor plane x = hub x
    for (int n = 0; n < t.num_vel_nodes; ++n) {
        EXPECT_NEAR(t.vel_pos[3 * n], 500.0, 1.0e-4) << "velocity node " << n;
    }
    EXPECT_NEAR(t.rotor_speed, d.omega(), 1.0e-6);
}

TEST(OpenFASTDriver, CheckpointAndRestartReproduceTheUninterruptedRun)
{
    // a run of five ERF steps, checkpointed after three; a second driver restored from that
    // checkpoint and stepped twice must report exactly the state of the uninterrupted run
    // (the stub is deterministic, and the interface stores the same floats)
    const StubDeck d;
    const auto dir = scratch_dir("restart");
    const std::string fst = write_stub_deck(dir, d);
    const std::string prefix = (dir / "chk_").string();
    const std::array<Real,3> vel{{static_cast<Real>(10.0), static_cast<Real>(0.5), 0.0}};

    std::vector<Real> pos_ref, force_ref;
    std::array<Real,3> hub_ref;
    int index_ref = 0;
    {
        erf_openfast::OpenFASTDriver a(std::vector<MovingBodyInputs>{one_turbine(fst, dir)});
        a.init(0.05, 1.0);
        a.set_uniform_velocity(vel);
        a.solution0();
        for (int n = 0; n < 3; ++n) { a.step(); }
        a.create_checkpoint(prefix);
        EXPECT_TRUE(std::filesystem::exists(dir / "chk_T1.chkp"));
        for (int n = 0; n < 2; ++n) { a.step(); }
        const erf_openfast::TurbineState& t = a.turbines()[0];
        pos_ref = t.force_pos;
        force_ref = t.force;
        hub_ref = t.hub_pos;
        index_ref = t.time_index;
    }
    EXPECT_EQ(index_ref, 25);

    // the stub's turbines are re-allocated by the second driver, so the first is gone by now
    erf_openfast::OpenFASTDriver b(std::vector<MovingBodyInputs>{one_turbine(fst, dir)});
    b.restart(prefix, 0.05);
    EXPECT_TRUE(b.initialized());
    EXPECT_TRUE(b.solved0());
    EXPECT_EQ(b.turbines()[0].time_index, 15);
    EXPECT_EQ(b.turbines()[0].num_substeps, 5);
    b.set_uniform_velocity(vel);
    for (int n = 0; n < 2; ++n) { b.step(); }
    const erf_openfast::TurbineState& t = b.turbines()[0];
    EXPECT_EQ(t.time_index, index_ref);
    ASSERT_EQ(t.force_pos.size(), pos_ref.size());
    ASSERT_EQ(t.force.size(), force_ref.size());
    for (std::size_t i = 0; i < pos_ref.size(); ++i) {
        EXPECT_EQ(t.force_pos[i], pos_ref[i]) << "position entry " << i;
        EXPECT_EQ(t.force[i], force_ref[i]) << "force entry " << i;
    }
    for (int k = 0; k < 3; ++k) { EXPECT_EQ(t.hub_pos[k], hub_ref[k]); }
    // the logs were appended to, not truncated: the header is still the first line
    std::ifstream csv((dir / "T1_erf.csv"));
    std::string line;
    ASSERT_TRUE(std::getline(csv, line));
    EXPECT_EQ(line.rfind("time,", 0), 0u) << line;
}

TEST(OpenFASTDriver, StepAdvancesTheRotorAndReturnsTheDiskLoads)
{
    const StubDeck d;
    const auto dir = scratch_dir("step");
    const std::string fst = write_stub_deck(dir, d);
    erf_openfast::OpenFASTDriver driver({one_turbine(fst, dir)});
    const double u_inf = 10.0;
    driver.init(0.05, 1.0);
    driver.set_uniform_velocity({{static_cast<Real>(u_inf), 0.0, 0.0}});
    driver.solution0();
    driver.step();
    driver.step();
    const erf_openfast::TurbineState& t = driver.turbines()[0];

    // two ERF steps of five OpenFAST steps each
    EXPECT_EQ(t.time_index, 10);
    // blade 0 turned by omega * 10 dt about +x
    const double az = d.omega() * 10 * d.dt;
    const double r_root = 0.5 / 6 * d.rotor_radius;
    EXPECT_NEAR(t.force_pos[3 * 1 + 1] - t.hub_pos[1], r_root * std::cos(az), 1.0e-3);
    EXPECT_NEAR(t.force_pos[3 * 1 + 2] - t.hub_pos[2], r_root * std::sin(az), 1.0e-3);

    // OpenFAST reports the force on the structure, so the thrust points along the +x inflow
    // and equals the disk value; torque times omega is the power. The interface stores float
    // forces, so compare to 1e-5 relative.
    const double thrust = 0.5 * d.air_density * d.ct * u_inf * u_inf * d.area();
    const double power = 0.5 * d.air_density * d.cp * u_inf * u_inf * u_inf * d.area();
    const std::array<Real,3> f = driver.thrust(t);
    EXPECT_NEAR(f[0], thrust, 1.0e-5 * thrust);
    EXPECT_NEAR(f[1], 0.0, 1.0e-5 * thrust);
    EXPECT_NEAR(f[2], 0.0, 1.0e-5 * thrust);
    EXPECT_NEAR(driver.torque(t) * t.rotor_speed, power, 1.0e-4 * power);
    // the hub node carries no force
    EXPECT_DOUBLE_EQ(t.force[0], 0.0);
    EXPECT_DOUBLE_EQ(t.force[1], 0.0);
    EXPECT_DOUBLE_EQ(t.force[2], 0.0);
    for (Real v : t.force) { EXPECT_TRUE(std::isfinite(v)); }

    // the diagnostics file has the header and one row per call
    driver.write_diagnostics(0.1);
    std::ifstream csv((dir / "T1_erf.csv"));
    ASSERT_TRUE(csv.good());
    std::string line;
    ASSERT_TRUE(std::getline(csv, line));
    EXPECT_EQ(line, "time,rotor_speed,thrust_x,thrust_y,thrust_z,torque,power,axis_x,axis_y,axis_z,"
                    "tower_x,tower_y,tower_z,nacelle_x,nacelle_y,nacelle_z,load_x,load_y,load_z");
    ASSERT_TRUE(std::getline(csv, line));
    EXPECT_EQ(line.rfind("0.1,", 0), 0u) << line;
    std::vector<std::string> fields;
    {
        std::istringstream ls(line);
        std::string fld;
        while (std::getline(ls, fld, ',')) { fields.push_back(fld); }
    }
    ASSERT_EQ(fields.size(), 19u) << line;
    EXPECT_EQ(fields[7], "1") << line;   // the stub's hub axis is x
    EXPECT_EQ(fields[8], "0") << line;
    EXPECT_EQ(fields[9], "0") << line;
    // no tower force nodes and no nacelle set: those columns are zero and the load is the thrust
    for (int k = 10; k < 16; ++k) { EXPECT_EQ(fields[k], "0") << "column " << k << ": " << line; }
    EXPECT_EQ(fields[16], fields[2]) << line;
    EXPECT_EQ(fields[17], fields[3]) << line;
    EXPECT_EQ(fields[18], fields[4]) << line;
    // the flow file carries the velocities the nodes were given: the uniform 10 m/s here
    std::ifstream flow((dir / "T1_flow.csv"));
    ASSERT_TRUE(flow.good());
    ASSERT_TRUE(std::getline(flow, line));
    EXPECT_EQ(line, "time,hub_u,hub_v,hub_w,blade_mean_u,blade_mean_v,blade_mean_w");
    ASSERT_TRUE(std::getline(flow, line));
    EXPECT_EQ(line, "0.1,10,0,0,10,0,0") << line;
}

// The stub's tower: with tower nodes in the deck and tower force points requested, every tower
// force node carries the cylinder drag of its length for the velocity the flow supplied, along
// the flow (on the structure), and the rotor's loads are unchanged by the tower.
TEST(OpenFASTDriver, TowerNodesCarryTheCylinderDragForAPrescribedVelocity)
{
    StubDeck d;
    d.num_tower_nodes = 6;
    d.tower_diameter = 8.0;
    d.tower_cd = 1.0;
    const auto dir = scratch_dir("tower");
    const std::string fst = write_stub_deck(dir, d);
    MovingBodyInputs b = one_turbine(fst, dir);
    b.num_force_points_tower = 10;
    erf_openfast::OpenFASTDriver driver({b});
    driver.init(0.05, 1.0);
    const std::array<Real,3> wind{{10.0, -2.0, 0.0}};
    driver.set_uniform_velocity(wind);
    driver.solution0();
    const erf_openfast::TurbineState& t = driver.turbines()[0];
    EXPECT_EQ(t.num_tower_elem, 6);
    EXPECT_EQ(t.num_force_pts_tower, 10);
    ASSERT_EQ(t.num_vel_nodes, 1 + d.num_blades * d.num_blade_nodes + 6);
    ASSERT_EQ(t.num_force_nodes, 1 + d.num_blades * 6 + 10);
    // tower force nodes: on the tower axis (the base), base to top at half-segment heights
    const int first = 1 + d.num_blades * 6;
    for (int k = 0; k < 10; ++k) {
        const int n = first + k;
        EXPECT_NEAR(t.force_pos[3*n],   500.0, 1.0e-4) << "tower node " << k;
        EXPECT_NEAR(t.force_pos[3*n+1], 500.0, 1.0e-4) << "tower node " << k;
        EXPECT_NEAR(t.force_pos[3*n+2], (k + 0.5) / 10.0 * d.hub_height, 1.0e-3) << "tower node " << k;
    }
    // each node: 1/2 rho Cd D dz |u_h| u_h along the horizontal wind, nothing vertical
    const double speed = std::sqrt(wind[0]*wind[0] + wind[1]*wind[1]);
    const double per_node = 0.5 * d.air_density * d.tower_cd * d.tower_diameter * (d.hub_height / 10.0) * speed;
    for (int k = 0; k < 10; ++k) {
        const int n = first + k;
        EXPECT_NEAR(t.force[3*n],   per_node * wind[0], 1.0e-3 * per_node * speed) << "tower node " << k;
        EXPECT_NEAR(t.force[3*n+1], per_node * wind[1], 1.0e-3 * per_node * speed) << "tower node " << k;
        EXPECT_NEAR(t.force[3*n+2], 0.0, 1.0e-6) << "tower node " << k;
    }
    const auto ft = driver.tower_force(t);
    EXPECT_NEAR(ft[0], 10 * per_node * wind[0], 1.0e-2 * per_node * speed);
    EXPECT_NEAR(ft[1], 10 * per_node * wind[1], 1.0e-2 * per_node * speed);
    // the rotor's thrust is the disk model's, untouched by the tower
    const auto f = driver.thrust(t);
    const double u_axial = wind[0];
    EXPECT_NEAR(f[0], 0.5 * d.air_density * d.ct * u_axial * u_axial * d.area(), 1.0e-3 * 0.5 * d.air_density * d.ct * u_axial * u_axial * d.area());
    // the nacelle force is the caller's: zero until set, then reported as given
    EXPECT_EQ(t.nacelle_force[0], Real(0.0));
    driver.set_nacelle_force(0, {{123.0, -4.0, 0.5}});
    EXPECT_EQ(driver.turbines()[0].nacelle_force[0], Real(123.0));
    EXPECT_EQ(driver.turbines()[0].nacelle_force[1], Real(-4.0));
    // without tower force points the deck's tower nodes carry no force nodes at all
    MovingBodyInputs b0 = one_turbine(fst, dir);
    b0.num_force_points_tower = 0;
    erf_openfast::OpenFASTDriver bare({b0});
    bare.init(0.05, 1.0);
    EXPECT_EQ(bare.turbines()[0].num_force_nodes, 1 + d.num_blades * 6);
    EXPECT_EQ(bare.tower_force(bare.turbines()[0])[0], Real(0.0));
}

TEST(OpenFASTDriver, TowerShadowCheckReadsTwrShadowFromTheAeroDynFile)
{
    const auto dir = scratch_dir("shadow");
    auto write = [&](const std::string& name, const std::string& body) {
        std::ofstream out(dir / name);
        out << body;
        return (dir / name).string();
    };
    write("ad_shadow.dat", "0   Wake_Mod  - none\n1   TwrShadow  - Powles\n");
    write("ad_noshadow.dat", "0   Wake_Mod  - none\n0   TwrShadow  - off\n");
    write("ad_old.dat", "0   WakeMod  - none\n");
    auto fst = [&](const std::string& name, const std::string& ad) {
        return write(name, "1   CompAero  - on\n\"" + ad + "\"  AeroFile  - x\n");
    };
    EXPECT_TRUE(erf_openfast::check_tower_shadow_off(fst("s.fst", "ad_shadow.dat")).find("TwrShadow = 1") != std::string::npos);
    EXPECT_TRUE(erf_openfast::check_tower_shadow_off(fst("n.fst", "ad_noshadow.dat")).empty());
    EXPECT_TRUE(erf_openfast::check_tower_shadow_off(fst("o.fst", "ad_old.dat")).empty());   // no such line: nothing to warn about
    EXPECT_TRUE(erf_openfast::check_tower_shadow_off(write("noaero.fst", "0   CompAero  - off\n\"ad_shadow.dat\"  AeroFile  - x\n")).empty());
    EXPECT_TRUE(erf_openfast::check_tower_shadow_off(write("stub.fst", "dt = 0.01\n")).empty());
}

// A farm: turbine i is owned by rank i modulo the rank count, every rank computes the same
// assignment, and on one rank two turbines run as two OpenFAST instances with their own
// identities, positions, loads and logs.
TEST(OpenFASTDriver, TurbinesAreDealtRoundRobinOverTheRanks)
{
    // one rank: everything on rank 0; more ranks than turbines: one turbine per rank; fewer:
    // the deal wraps around
    for (int i = 0; i < 5; ++i) { EXPECT_EQ(erf_openfast::owner_rank_for(i, 1), 0); }
    EXPECT_EQ(erf_openfast::owner_rank_for(0, 8), 0);
    EXPECT_EQ(erf_openfast::owner_rank_for(1, 8), 1);
    EXPECT_EQ(erf_openfast::owner_rank_for(7, 8), 7);
    EXPECT_EQ(erf_openfast::owner_rank_for(8, 8), 0);
    EXPECT_EQ(erf_openfast::owner_rank_for(9, 8), 1);
    // every turbine has a valid rank and, up to the rank count, a distinct one
    for (int np = 1; np <= 6; ++np) {
        std::vector<int> seen(np, 0);
        for (int i = 0; i < np; ++i) {
            const int r = erf_openfast::owner_rank_for(i, np);
            ASSERT_GE(r, 0); ASSERT_LT(r, np);
            ++seen[r];
        }
        for (int r = 0; r < np; ++r) { EXPECT_EQ(seen[r], 1) << "np " << np << " rank " << r; }
    }
}

TEST(OpenFASTDriver, TwoTurbinesRunAsTwoInstancesWithTheirOwnLoads)
{
    StubDeck d;
    const auto dir = scratch_dir("farm");
    const std::string fst = write_stub_deck(dir, d);
    MovingBodyInputs a = one_turbine(fst, dir);
    MovingBodyInputs b = one_turbine(fst, dir);
    b.name = "T2";
    b.base_pos = {{1230.0, 500.0, 0.0}};
    b.output_root = (dir / "T2").string();
    erf_openfast::OpenFASTDriver driver({a, b});
    driver.init(0.05, 1.0);
    ASSERT_EQ(driver.turbines().size(), 2u);
    // on the single rank of this test both are owned here, as two OpenFAST instances
    EXPECT_TRUE(driver.is_owner(0));
    EXPECT_TRUE(driver.is_owner(1));
    EXPECT_EQ(driver.turbines()[0].owner_rank, 0);
    EXPECT_EQ(driver.turbines()[1].owner_rank, 0);
    EXPECT_NE(driver.turbines()[0].tid_local, driver.turbines()[1].tid_local);
    // their own places in the domain
    EXPECT_NEAR(driver.turbines()[0].hub_pos[0], 500.0, 1.0e-4);
    EXPECT_NEAR(driver.turbines()[1].hub_pos[0], 1230.0, 1.0e-4);
    // different winds at the two rotors give different loads, each the disk model's own
    std::vector<Real> u1(3 * driver.turbines()[0].num_vel_nodes), u2(3 * driver.turbines()[1].num_vel_nodes);
    for (std::size_t k = 0; k < u1.size(); k += 3) { u1[k] = 10.0; u1[k+1] = 0.0; u1[k+2] = 0.0; }
    for (std::size_t k = 0; k < u2.size(); k += 3) { u2[k] = 8.0;  u2[k+1] = 0.0; u2[k+2] = 0.0; }
    driver.set_node_velocities(0, u1);
    driver.set_node_velocities(1, u2);
    driver.solution0();
    const auto f1 = driver.thrust(driver.turbines()[0]);
    const auto f2 = driver.thrust(driver.turbines()[1]);
    const double t1 = 0.5 * d.air_density * d.ct * 100.0 * d.area();
    const double t2 = 0.5 * d.air_density * d.ct * 64.0 * d.area();
    EXPECT_NEAR(f1[0], t1, 1.0e-5 * t1);
    EXPECT_NEAR(f2[0], t2, 1.0e-5 * t2);
    // a step advances both; their logs are separate files
    driver.step();
    driver.write_diagnostics(0.05);
    EXPECT_TRUE(std::filesystem::exists(dir / "T1_erf.csv"));
    EXPECT_TRUE(std::filesystem::exists(dir / "T2_erf.csv"));
    EXPECT_EQ(driver.turbines()[0].time_index, driver.turbines()[1].time_index);
    EXPECT_EQ(driver.turbines()[0].time_index, 5);
}
