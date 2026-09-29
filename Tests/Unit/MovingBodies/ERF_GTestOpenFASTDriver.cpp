// Contracts of the OpenFAST driver and the erf.moving_bodies inputs, checked against the
// bundled stub library (Source/MovingBodies/OpenFAST/Stub), whose geometry and disk model are
// known in closed form.

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
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
        << "air_density = " << d.air_density << "\n";
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

TEST(MovingBodiesInputs, ValidateSolverAcceptsOnlyAnelasticFixedStepSingleLevelNoTraps)
{
    EXPECT_TRUE(MovingBodiesInputs::validate_solver(true, true, 0, false).empty());
    EXPECT_NE(MovingBodiesInputs::validate_solver(false, true, 0, false).find("anelastic"), std::string::npos);
    EXPECT_NE(MovingBodiesInputs::validate_solver(true, false, 0, false).find("fixed"), std::string::npos);
    EXPECT_NE(MovingBodiesInputs::validate_solver(true, true, 1, false).find("max_level"), std::string::npos);
    // OpenFAST 4.2.1 raises floating-point exceptions in FAST_ProgStart; a trapped run dies there
    EXPECT_NE(MovingBodiesInputs::validate_solver(true, true, 0, true).find("fpe_trap"), std::string::npos);
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
    EXPECT_EQ(line, "time,rotor_speed,thrust_x,thrust_y,thrust_z,torque,power,axis_x,axis_y,axis_z");
    ASSERT_TRUE(std::getline(csv, line));
    EXPECT_EQ(line.rfind("0.1,", 0), 0u) << line;
    EXPECT_EQ(line.substr(line.size() - 6), ",1,0,0") << line;   // the stub's hub axis is x
    // the flow file carries the velocities the nodes were given: the uniform 10 m/s here
    std::ifstream flow((dir / "T1_flow.csv"));
    ASSERT_TRUE(flow.good());
    ASSERT_TRUE(std::getline(flow, line));
    EXPECT_EQ(line, "time,hub_u,hub_v,hub_w,blade_mean_u,blade_mean_v,blade_mean_w");
    ASSERT_TRUE(std::getline(flow, line));
    EXPECT_EQ(line, "0.1,10,0,0,10,0,0") << line;
}
