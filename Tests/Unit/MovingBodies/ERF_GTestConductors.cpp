// Unit tests of Conductors, the manager ERF holds for the conductor lines, transformers and towers.
//
// AttachmentsArePlacedAboveTheTerrainUnderEachEnd: the attachment points stand at their height above
//     the k = 0 node plane of z_phys_nd (bilinear between the nodes), as ground.dat records.
// OnAUniformMeshTheHeightsAreAbsolute: without terrain the given heights are absolute.
// LinesStepOnTheAnchorLevelOnlyAndLogEveryStep: other levels return at once; one row per step.
// TheFlowIsSampledAtTheLinesCurrentPosition: without a prescribed velocity, MoorDyn gets ERF's
//     velocity at the line's current position, not where it hung.
// ClearanceIsTheHeightAboveTheTerrainUnderEachNode: the clearance against the terrain under each node.
// TheSpreadDragIntegratesToMinusTheDragOnTheLines: the momentum source integrates to minus the drag.
// WithoutDragOnFlowNothingIsPutIntoTheFlow: drag_on_flow off leaves the sources empty.
// ARestartContinuesTheLinesTheirSourcesStatisticsAndLogs: a restart continues where the checkpoint left off.
// TrimmingALogKeepsTheHeaderAndTheRowsUpToTheCheckpoint: trim_log_after.
// ASectionIsPlacedOnTheTerrainAtEveryTowerAndLogsEachSpan: a section's towers and per-span logs.
// TheClosestApproachOfTwoLinesIsFlaggedAgainstTheFlashoverDistance: the exact separation and the flag.
// ARestartContinuesASectionItsStringsAndTheSeparationOfTheLines: a section's restart.
// TransformersTakeThePullOfTheLinesEndingOnThemAndContinueAcrossARestart: transformer loads,
//     allowables, clearances, and their restart.
// AStringingTensionSetsTheLengthsFromTheChordsOnTheTerrain: strung lengths from the placed chords.
// LatticeTowersStandAtTheSuspensionPointsAndCarryTheWindsDrag: tower placement, member drag, the
//     line's pull and the footing checks.
// TheTowersDragGoesIntoTheFlowWithTheLinesAndSurvivesARestart: the towers' drag in the sources.
// MovingTowersSettleWhereTheirStiffnessBalancesTheWindAndTheLine: bending towers at rest.
// MovingTowersContinueAcrossARestart: bending towers' restart.
// ACircuitHangsFromOneRowOfTowersEachLineAtItsOwnPoint: shared towers, points on the tower's base.
// ACircuitOnBendingTowersMovesEveryLinesPoint: bending shared towers move every line's point.
// AShortTautSpanOnBendingTowersStaysStable: the iterated coupling keeps a stiff span stable.
// AnImmersedTerrainPlacesEverythingOnItsSurface: an immersed terrain on a flat mesh.
// SetGroundRefusesASurfaceOffsetBelowTheDomainTop: erf.conductors.surface_offset must hold the domain.
// AnAttachmentOutsideTheDomainIsRefusedNamingItsKey: the abort names end_a, end_b or the tower.
// ANonFiniteCouplingPullIsRefusedNotConverged: coupling_converged on NaN pulls.
// ARestartChecksTheTowerSwayAndTheSurfaceOffset: restart_mismatch.
// GustsComeFromTheRANSkAlongEachSpan: with gust_type = factor, per span the root-mean-square wind and normal wind
//     over its nodes and the mean k = (rho k)/rho, from stats_start, sampled where the nodes are at each step's start;
//     gusts.csv's columns for a line across the wind and one at 45 degrees to it; nothing before stats_start; the
//     set_closure and conserved-state aborts.
// GustsSwitchedOnAtARestartStartAfresh: a checkpoint written without gusts restarts with gust_type = factor, the
//     gust statistics starting at the restart.
// AnEventGustPassesEveryNodeAndTowerWithItsFront: with gust_type = event, the wind at every line node and tower drag
//     node is the flow's plus g sigma_u times the 1 - cos shape at the node's phase at the step's middle, sigma_u from
//     the mean k over the span's or tower's nodes; the origin defaults to the domain's centre; gust_series.dat.
// RandomGustsFollowTheirProcessesAndACircuitSharesThem: with gust_type = random, every span's gust is
//     sigma_u sqrt(B) z, z the Ornstein-Uhlenbeck process stepped with T = L_u / U, the phases and the shield wire on
//     shared towers taking the owner's spans' processes; a string the mean of the spans beside it; each tower
//     sigma_u sqrt(B_t) z of its own; gust_series.dat's columns.
// RandomGustsContinueAcrossARestart: a restart continues the random gusts where the checkpoint left them, and a
//     checkpoint written without gusts draws them afresh from the stationary law at the restart's first step.
// TheGustFactorRunsOnLinesWithLatticeTowers: gust_type = factor on a section with lattice towers, whose drag nodes
//     it does not sample, writes gusts.csv and no gust series.
// ALineTakesAnotherLinesGustsAndAChangedSharingDrawsAfresh: a line naming another in gust_with takes its random
//     gust, each along its span's mean horizontal wind in a wind that veers with the height; a restart whose lines
//     share differently, with as many processes, draws them afresh.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_RealBox.H>

#include <gtest/gtest.h>

#include "ERF_ActuatorSpreading.H"
#include "ERF_Conductors.H"
#include "ERF_Gusts.H"
#include "ERF_IndexDefines.H"
#include "ERF_GTestThrowOnAbort.H"
#include "ERF_MoorDynSystem.H"

namespace {

using amrex::Real;

// the roundoff of the sampling, the terrain read and the source sums: about 1e-7 relative in single precision
constexpr Real roundoff = (std::is_same<Real, float>::value) ? Real(1.0e-5) : Real(1.0e-9);
// positions and heights of a few hundred metres carry a Real's spacing there: 3e-5 m at 500 m in single precision
constexpr Real postol = (std::is_same<Real, float>::value) ? Real(1.0e-4) : Real(1.0e-6);
// the same quantity computed twice the same way: a few units of the last place
constexpr Real tight = (std::is_same<Real, float>::value) ? Real(1.0e-6) : Real(1.0e-12);

// a 1200 m x 1000 m x 400 m box of 12 x 10 x 8 cells; terrain-following nodes over a ramp
// h = slope_x x + slope_y y, so the bilinear surface under any point is exact
struct Mesh {
    int nx = 12, ny = 10, nz = 8;
    Real Lx = 1200.0, Ly = 1000.0, H = 400.0;
    Real slope_x = 0.05, slope_y = 0.02;
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    std::unique_ptr<amrex::MultiFab> znd;
    amrex::MultiFab u, v, w;

    Real h (Real x, Real y) const { return slope_x * x + slope_y * y; }

    // the crosswind v = v0 + vy (y - 500) + vz z, u = w = 0: linear, so the sampler reproduces it exactly
    Real v0 = 10.0, vy = 0.02, vz = 0.05;
    Real v_at (Real y, Real z) const { return v0 + vy * (y - 500.0) + vz * z; }
    void fill_crosswind ()
    {
        const Real dx = Lx / nx, dy = Ly / ny, dz = H / nz;
        u.setVal(0.0); w.setVal(0.0);
        for (amrex::MFIter mfi(v); mfi.isValid(); ++mfi) {
            auto va = v.array(mfi);
            amrex::LoopOnCpu(mfi.growntilebox(), [&](int i, int j, int k) {
                amrex::ignore_unused(i, dx);
                va(i,j,k) = v_at(j * dy, (k + 0.5) * dz);
            });
        }
    }

    explicit Mesh (bool terrain)
    {
        const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(nx-1, ny-1, nz-1));
        const amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(Lx, Ly, H)});
        const std::array<int,3> periodic{{0, 0, 0}};
        geom = amrex::Geometry(domain, &rb, 0, periodic.data());
        ba = amrex::BoxArray(domain);
        ba.maxSize(amrex::IntVect(4, 5, 1024));   // several boxes: the terrain read has one owner per point
        dm = amrex::DistributionMapping(ba);
        u.define(amrex::convert(ba, amrex::IntVect(1,0,0)), dm, 1, 1); u.setVal(0.0);
        v.define(amrex::convert(ba, amrex::IntVect(0,1,0)), dm, 1, 1); v.setVal(0.0);
        w.define(amrex::convert(ba, amrex::IntVect(0,0,1)), dm, 1, 1); w.setVal(0.0);
        if (terrain) {
            znd = std::make_unique<amrex::MultiFab>(amrex::convert(ba, amrex::IntVect(1,1,1)), dm, 1, 1);
            const Real dx = Lx / nx, dy = Ly / ny;
            for (amrex::MFIter mfi(*znd); mfi.isValid(); ++mfi) {
                auto za = znd->array(mfi);
                amrex::LoopOnCpu(mfi.growntilebox(), [&](int i, int j, int k) {
                    const Real hs = h(i * dx, j * dy);
                    za(i,j,k) = hs + (H - hs) * Real(k) / nz;
                });
            }
        }
    }
};

std::string scratch (const std::string& tag)
{
    const auto dir = std::filesystem::temp_directory_path() / ("erf_gtest_conductors_" + tag);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir.string();
}

// one span, 300 m along x at y = 500, ends 30 m above the local surface
void set_inputs (const std::string& dir, const std::string& name, bool prescribed = true,
                 const std::array<Real,3>& a = {{300.0, 500.0, 30.0}}, const std::array<Real,3>& b = {{600.0, 500.0, 30.0}})
{
    amrex::ParmParse pp("erf.conductors");
    pp.add("lines", name);
    pp.add("diagnostics_dir", dir);
    pp.add("air_density", 1.2);
    // ParmParse is global to the test binary: drop what another test left before setting this one's
    for (const char* key : {"prescribed_velocity", "drag_on_flow", "epsilon", "node_output_int", "stats_start", "flashover_distance"}) { pp.remove(key); }
    if (prescribed) { pp.addarr("prescribed_velocity", std::vector<Real>{0.0, 10.0, 0.0}); }
    amrex::ParmParse ps("erf.conductors." + name);
    ps.addarr("end_a", std::vector<Real>{a[0], a[1], a[2]});
    ps.addarr("end_b", std::vector<Real>{b[0], b[1], b[2]});
    ps.add("length", 301.5);
    ps.add("diameter", 0.0281);
    ps.add("mass_per_length", 1.628);
    ps.add("axial_stiffness", 3.0e7);
}

} // namespace

TEST(Conductors, AttachmentsArePlacedAboveTheTerrainUnderEachEnd)
{
    const std::string dir = scratch("terrain");
    set_inputs(dir, "Tterrain");
    Mesh m(true);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(m.znd.get(), m.geom);
    ASSERT_EQ(c->lines().size(), 1u);
    const auto& span = *c->lines().front();
    const unsigned last = span.num_nodes() - 1;
    const auto a = span.node_position(0);
    const auto b = span.node_position(last);
    // the ramp rises 15 m from x = 300 to x = 600: the two ends sit at different absolute heights
    EXPECT_NEAR(a[2], m.h(300.0, 500.0) + 30.0, postol);
    EXPECT_NEAR(b[2], m.h(600.0, 500.0) + 30.0, postol);
    EXPECT_NEAR(a[0], 300.0, postol);
    EXPECT_NEAR(b[0], 600.0, postol);
    EXPECT_GT(b[2] - a[2], 10.0);
    // ground.dat records the surface and the absolute height of each end
    std::ifstream g(dir + "/ground.dat");
    ASSERT_TRUE(g.good());
    std::string header, name, end;
    Real x, y, ground, z;
    std::getline(g, header);
    ASSERT_TRUE(static_cast<bool>(g >> name >> end >> x >> y >> ground >> z));
    EXPECT_EQ(name, "Tterrain"); EXPECT_EQ(end, "a");
    EXPECT_NEAR(ground, m.h(300.0, 500.0), postol);
    EXPECT_NEAR(z, m.h(300.0, 500.0) + 30.0, postol);
    ASSERT_TRUE(static_cast<bool>(g >> name >> end >> x >> y >> ground >> z));
    EXPECT_EQ(end, "b");
    EXPECT_NEAR(ground, m.h(600.0, 500.0), postol);
}

TEST(Conductors, OnAUniformMeshTheHeightsAreAbsolute)
{
    const std::string dir = scratch("flat");
    set_inputs(dir, "Tflat");
    Mesh m(false);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(nullptr, m.geom);
    const auto& span = *c->lines().front();
    EXPECT_NEAR(span.node_position(0)[2], 30.0, postol);
    EXPECT_NEAR(span.node_position(span.num_nodes() - 1)[2], 30.0, postol);
}

TEST(Conductors, LinesStepOnTheAnchorLevelOnlyAndLogEveryStep)
{
    const std::string dir = scratch("advance");
    set_inputs(dir, "Tadv");
    Mesh m(false);
    auto c = Conductors::create(1);   // two levels: the anchor is the finest, level 1
    ASSERT_TRUE(c);
    EXPECT_EQ(c->anchor_level(), 1);
    c->set_ground(nullptr, m.geom);
    const auto& span = *c->lines().front();
    const Real off0 = span.mid_offset();
    c->advance(0, 0.0, 0.1, m.u, m.v, m.w, nullptr, nullptr, m.geom);   // level 0: nothing happens
    EXPECT_DOUBLE_EQ(span.mid_offset(), off0);
    EXPECT_FALSE(std::filesystem::exists(dir + "/Tadv.dat"));
    for (int s = 0; s < 3; ++s) { c->advance(1, 0.1 * s, 0.1, m.u, m.v, m.w, nullptr, nullptr, m.geom); }
    EXPECT_GT(span.mid_offset(), off0) << "the prescribed +y wind must move the span towards +y";
    std::ifstream f(dir + "/Tadv.dat");
    ASSERT_TRUE(f.good());
    std::string line;
    int rows = 0;
    while (std::getline(f, line)) { if (!line.empty() && line.rfind("time", 0) != 0) { ++rows; } }
    EXPECT_EQ(rows, 4) << "the initial row and one per step";
}

TEST(Conductors, TheFlowIsSampledAtTheLinesCurrentPosition)
{
    const std::string dir = scratch("sampled");
    set_inputs(dir, "Tsampled", false);
    amrex::ParmParse pp("erf.conductors");
    ASSERT_FALSE(pp.contains("prescribed_velocity")) << "set_inputs must have removed the prescribed velocity";
    Mesh m(false);
    m.fill_crosswind();
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(nullptr, m.geom);
    const auto& span = *c->lines().front();
    Real moved = 0.0;
    for (int s = 0; s < 20; ++s) {
        const std::vector<Real> where = span.kinematics_points();   // where the line is when the wind is sampled
        c->advance(0, 0.25 * s, 0.25, m.u, m.v, m.w, nullptr, nullptr, m.geom);
        for (unsigned p = 0; p < span.num_kinematics_points(); ++p) {
            // the flow at the line nodes; no wind at MoorDyn's fixed entries after them
            const auto uvw = span.wind_at_point(p);
            const Real expect = (p < span.num_nodes()) ? m.v_at(where[3*p+1], where[3*p+2]) : Real(0.0);
            ASSERT_NEAR(uvw[0], 0.0, roundoff) << "step " << s << " point " << p;
            ASSERT_NEAR(uvw[1], expect, roundoff * std::max(Real(1.0), expect)) << "step " << s << " point " << p;
            ASSERT_NEAR(uvw[2], 0.0, roundoff) << "step " << s << " point " << p;
        }
        moved = std::max(moved, span.mid_offset());
    }
    // the line really moved, so sampling at the initial positions would have handed a different wind
    EXPECT_GT(moved, 0.5) << "the span must blow out in the sampled crosswind";
    EXPECT_GT(m.vy * moved, 1.0e-3) << "the field must change measurably over the distance the line moved";
}

TEST(Conductors, GustsComeFromTheRANSkAlongEachSpan)
{
    const std::string dir = scratch("gusts");
    set_inputs(dir, "Tgust", false);
    amrex::ParmParse pp("erf.conductors");
    // a second line at 45 degrees to the wind and 10 m higher: the normal component and the second line's height
    pp.remove("lines");
    pp.addarr("lines", std::vector<std::string>{"Tgust", "Tgust2"});
    {
        amrex::ParmParse ps("erf.conductors.Tgust2");
        ps.addarr("end_a", std::vector<Real>{300.0, 300.0, 40.0});
        ps.addarr("end_b", std::vector<Real>{513.0, 513.0, 40.0});
        ps.add("length", 301.5);
        ps.add("diameter", 0.0281);
        ps.add("mass_per_length", 1.628);
        ps.add("axial_stiffness", 3.0e7);
    }
    pp.add("gust_type", std::string("factor"));
    // the steps end at 0.25, 0.5, ...: the first is not sampled
    pp.add("stats_start", 0.5);
    Mesh m(false);
    m.fill_crosswind();
    // rho and rho k with k linear in the height, which the cell sampler reproduces exactly
    const Real rho0 = 1.2, k0 = 0.5, kz = 0.01;
    amrex::MultiFab cons(m.ba, m.dm, RhoKE_comp + 1, 1);
    cons.setVal(0.0);
    const Real dz = m.H / m.nz;
    for (amrex::MFIter mfi(cons); mfi.isValid(); ++mfi) {
        auto ca = cons.array(mfi);
        amrex::LoopOnCpu(mfi.growntilebox(), [&](int i, int j, int k) {
            ca(i,j,k,Rho_comp) = rho0;
            ca(i,j,k,RhoKE_comp) = rho0 * (k0 + kz * (k + 0.5) * dz);
        });
    }
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(nullptr, m.geom);
    {
        using erf_gtest::abort_message;
        EXPECT_NE(abort_message([&] { c->advance(0, 0.0, 0.25, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons); })
                      .find("needs set_closure() first"), std::string::npos);
        EXPECT_NE(abort_message([&] { c->set_closure(false, 0.5562); }).find("needs the k-equation RANS"), std::string::npos);
    }
    c->set_closure(true, 0.5562);
    // Cmu0 arrives as amrex::Real: a float in single precision
    EXPECT_NEAR(c->gust_sigma_factor(), 2.5 * 0.5562, 1e-6);
    EXPECT_NE(erf_gtest::abort_message([&] { c->advance(0, 0.0, 0.25, m.u, m.v, m.w, nullptr, nullptr, m.geom); })
                  .find("needs the conserved state"), std::string::npos);
    ASSERT_EQ(c->lines().size(), 2u);
    // the normals of the two chords' horizontal projections: +y for the line along x, (-1, 1)/sqrt(2) for the other
    const double ny[2] = {1.0, 1.0 / std::sqrt(2.0)};
    // the expected means: every sampled step, the root-mean-square wind and normal wind over each span's nodes and
    // the mean k, where the nodes are when the wind is sampled (u = w = 0, so the wind is |v|)
    double wind[2] = {0.0, 0.0}, normal[2] = {0.0, 0.0}, kmean[2] = {0.0, 0.0};
    const int steps = 8;
    for (int s = 0; s < steps; ++s) {
        if (s == 1) {
            EXPECT_FALSE(std::filesystem::exists(dir + "/gusts.csv")) << "no table before the first sample (stats_start)";
            EXPECT_FALSE(std::filesystem::exists(dir + "/Tgust_gusts_stats.csv")) << "no statistics before stats_start";
        }
        if (s > 0) {
            for (int l = 0; l < 2; ++l) {
                const auto& span = *c->lines()[static_cast<std::size_t>(l)];
                const std::vector<Real> where = span.kinematics_points();
                double w2 = 0.0, kk = 0.0;
                for (unsigned n = 0; n < span.num_nodes(); ++n) {
                    const double v = static_cast<double>(m.v_at(where[3*n+1], where[3*n+2]));
                    w2 += v * v;
                    kk += static_cast<double>(k0 + kz * where[3*n+2]);
                }
                wind[l] += std::sqrt(w2 / span.num_nodes());
                normal[l] += ny[l] * std::sqrt(w2 / span.num_nodes());
                kmean[l] += kk / span.num_nodes();
            }
        }
        c->advance(0, 0.25 * s, 0.25, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons);
    }
    {
        // the running statistics count the sampled steps only
        std::ifstream st(dir + "/Tgust_gusts_stats.csv");
        std::string h, r;
        ASSERT_TRUE(std::getline(st, h) && std::getline(st, r));
        EXPECT_EQ(r.substr(0, r.find(',')), std::to_string(steps - 1));
    }
    std::ifstream f(dir + "/gusts.csv");
    ASSERT_TRUE(f.good());
    std::string header, row;
    std::getline(f, header);
    EXPECT_EQ(header, "line,span,height,chord,wind,normal_wind,k,sigma,normal_sigma,intensity,gust_response,gust_wind,mean_load,"
                      "peak_load,valid");
    const double tol = (std::is_same<Real, float>::value) ? 1e-4 : 1e-8;
    const char* names[2] = {"Tgust", "Tgust2"};
    const double height[2] = {30.0, 40.0}, chord[2] = {300.0, 213.0 * std::sqrt(2.0)};
    for (int l = 0; l < 2; ++l) {
        ASSERT_TRUE(std::getline(f, row)) << "one row per span";
        std::vector<std::string> col;
        std::stringstream ss(row);
        for (std::string x; std::getline(ss, x, ',');) { col.push_back(x); }
        ASSERT_EQ(col.size(), 15u);
        EXPECT_EQ(col[0], names[l]);
        EXPECT_EQ(col[1], "1");
        const double U = wind[l] / (steps - 1), Un = normal[l] / (steps - 1), k = kmean[l] / (steps - 1);
        EXPECT_NEAR(std::stod(col[2]), height[l], tol * height[l]) << "the attachment height above the flat ground";
        EXPECT_NEAR(std::stod(col[3]), chord[l], tol * chord[l]);
        EXPECT_NEAR(std::stod(col[4]), U, tol * U);
        EXPECT_NEAR(std::stod(col[5]), Un, tol * Un);
        EXPECT_NEAR(std::stod(col[6]), k, tol * k);
        // the formulas are pinned by the Gusts unit tests; here the columns must carry them
        const auto G = erf_conductors::span_gust(U, Un, k, chord[l], 0.0281, 1.0, 1.2, 2.5 * 0.5562, 2.7, 67.056);
        EXPECT_NEAR(std::stod(col[7]), G.sigma, tol * G.sigma);
        EXPECT_NEAR(std::stod(col[8]), G.normal_sigma, tol * G.normal_sigma);
        EXPECT_NEAR(std::stod(col[9]), G.intensity, tol * G.intensity);
        EXPECT_NEAR(std::stod(col[10]), G.gust_response, tol * G.gust_response);
        EXPECT_NEAR(std::stod(col[11]), G.gust_wind, tol * G.gust_wind);
        EXPECT_NEAR(std::stod(col[12]), 0.5 * 1.2 * 1.0 * 0.0281 * Un * Un, tol * G.mean_load);
        EXPECT_NEAR(std::stod(col[13]), G.peak_load, tol * G.peak_load);
        EXPECT_EQ(col[14], G.linear_valid ? "1" : "0");
        EXPECT_TRUE(G.linear_valid) << "about 10 and 7 m/s normal to the spans against sigma_n of about 1.2 m/s";
    }
    EXPECT_FALSE(std::getline(f, row)) << "one row per span";
}

TEST(Conductors, GustsSwitchedOnAtARestartStartAfresh)
{
    const std::string dir = scratch("gust_restart");
    set_inputs(dir, "Tfresh", false);
    Mesh m(false);
    m.fill_crosswind();
    amrex::MultiFab cons(m.ba, m.dm, RhoKE_comp + 1, 1);
    cons.setVal(0.0);
    for (amrex::MFIter mfi(cons); mfi.isValid(); ++mfi) {
        auto ca = cons.array(mfi);
        amrex::LoopOnCpu(mfi.growntilebox(), [&](int i, int j, int k) {
            ca(i,j,k,Rho_comp) = 1.2;
            ca(i,j,k,RhoKE_comp) = 1.2 * 0.8;
        });
    }
    const double dt = 0.25;
    // a run without gusts writes a checkpoint after three steps
    auto a = Conductors::create(0);
    ASSERT_TRUE(a);
    a->set_closure(true, 0.5562);
    a->set_ground(nullptr, m.geom);
    for (int s = 0; s < 3; ++s) { a->advance(0, dt * s, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons); }
    const std::string chk = dir + "/chk00003";
    std::filesystem::create_directories(chk);
    a->write_checkpoint(chk);
    EXPECT_FALSE(std::filesystem::exists(dir + "/gusts.csv"));
    // the restart turns gusts on: the lines continue, the gust statistics start at the restart
    amrex::ParmParse("erf.conductors").add("gust_type", std::string("factor"));
    auto b = Conductors::create(0);
    ASSERT_TRUE(b);
    b->set_closure(true, 0.5562);
    b->set_ground(nullptr, m.geom, chk);
    EXPECT_TRUE(b->restored());
    for (int s = 3; s < 6; ++s) { b->advance(0, dt * s, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons); }
    std::ifstream st(dir + "/Tfresh_gusts_stats.csv");
    std::string h, r;
    ASSERT_TRUE(std::getline(st, h) && std::getline(st, r));
    std::vector<std::string> col;
    std::stringstream ss(r);
    for (std::string x; std::getline(ss, x, ',');) { col.push_back(x); }
    ASSERT_GE(col.size(), 3u);
    EXPECT_EQ(col[0], "3") << "the three steps after the restart";
    EXPECT_DOUBLE_EQ(std::stod(col[1]), 0.75) << "the first sample is the restart step's start";
    EXPECT_TRUE(std::filesystem::exists(dir + "/gusts.csv"));
}

TEST(Conductors, ClearanceIsTheHeightAboveTheTerrainUnderEachNode)
{
    const std::string dir = scratch("clearance");
    set_inputs(dir, "Tclear");
    amrex::ParmParse("erf.conductors").add("node_output_int", 1);
    Mesh m(true);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(m.znd.get(), m.geom);
    for (int s = 0; s < 3; ++s) { c->advance(0, 0.2 * s, 0.2, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    const auto& span = *c->lines().front();
    Real lowest = 1.0e30;
    unsigned lowest_node = 0;
    for (unsigned n = 0; n < span.num_nodes(); ++n) {
        const auto p = span.node_position(n);
        const Real expect = p[2] - m.h(p[0], p[1]);   // the ramp: the bilinear surface is exact
        EXPECT_NEAR(span.clearance(n), expect, roundoff * std::max(Real(1.0), std::abs(expect))) << "node " << n;
        if (expect < lowest) { lowest = expect; lowest_node = n; }
    }
    unsigned node = 0;
    EXPECT_NEAR(span.min_clearance(node), lowest, roundoff * std::max(Real(1.0), std::abs(lowest)));
    EXPECT_EQ(node, lowest_node);
    EXPECT_LT(lowest, 30.0) << "the sagging middle sits lower above the ground than the attachments";
    // the node file: one row per node per write (the initial write and three steps)
    std::ifstream f(dir + "/Tclear_nodes.dat");
    ASSERT_TRUE(f.good());
    std::string line;
    int rows = 0;
    while (std::getline(f, line)) { if (!line.empty() && line.rfind("time", 0) != 0) { ++rows; } }
    EXPECT_EQ(rows, 4 * static_cast<int>(span.num_nodes()));
    // the statistics file lists every quantity
    std::ifstream st(dir + "/Tclear_stats.csv");
    ASSERT_TRUE(st.good());
    std::string all((std::istreambuf_iterator<char>(st)), std::istreambuf_iterator<char>());
    for (const char* q : {"swing_deg", "mid_offset", "tension_a", "tension_b", "max_tension", "min_clearance", "drag_y"}) {
        EXPECT_NE(all.find(q), std::string::npos) << q;
    }
}

TEST(Conductors, TheSpreadDragIntegratesToMinusTheDragOnTheLines)
{
    const std::string dir = scratch("drag");
    set_inputs(dir, "Tdrag");   // a prescribed 10 m/s crosswind along +y
    amrex::ParmParse("erf.conductors").add("drag_on_flow", true);
    Mesh m(true);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    EXPECT_TRUE(c->drag_on_flow());
    c->set_ground(m.znd.get(), m.geom);
    // a terrain-following mesh: the face volumes carry detJ, the cell's physical height over its nominal
    // height, which shrinks where the ramp rises (the columns span H - h)
    amrex::MultiFab detJ(m.ba, m.dm, 1, 1);
    for (amrex::MFIter mfi(detJ); mfi.isValid(); ++mfi) {
        const auto za = m.znd->const_array(mfi);
        auto ja = detJ.array(mfi);
        const Real dz = m.H / m.nz;
        amrex::LoopOnCpu(mfi.growntilebox(), [&](int i, int j, int k) {
            ja(i,j,k) = Real(0.25) * ((za(i,j,k+1) - za(i,j,k)) + (za(i+1,j,k+1) - za(i+1,j,k)) +
                                      (za(i,j+1,k+1) - za(i,j+1,k)) + (za(i+1,j+1,k+1) - za(i+1,j+1,k))) / dz;
        });
    }
    EXPECT_LT(detJ.min(0), 0.99 * detJ.max(0)) << "the Jacobian must vary over the ramp";
    for (int s = 0; s < 2; ++s) { c->advance(0, 0.2 * s, 0.2, m.u, m.v, m.w, m.znd.get(), &detJ, m.geom); }
    const auto& span = *c->lines().front();
    const auto drag = span.total_drag();
    EXPECT_GT(drag[1], 1.0) << "a crosswind along +y must drag the line along +y";
    const auto& integral = c->source_integral();
    for (int d = 0; d < 3; ++d) {
        EXPECT_NEAR(integral[d], -drag[d], Real(0.1) * roundoff * std::max(Real(1.0), std::abs(drag[d]))) << "component " << d;
    }
    // the sources ERF adds integrate to the same force
    amrex::MultiFab sx(amrex::convert(m.ba, amrex::IntVect(1,0,0)), m.dm, 1, 0);
    amrex::MultiFab sy(amrex::convert(m.ba, amrex::IntVect(0,1,0)), m.dm, 1, 0);
    amrex::MultiFab sz(amrex::convert(m.ba, amrex::IntVect(0,0,1)), m.dm, 1, 0);
    sx.setVal(0.0); sy.setVal(0.0); sz.setVal(0.0);
    c->add_momentum_sources(0, sx, sy, sz);
    EXPECT_NEAR(erf_actuator::integrate_source(1, sy, &detJ, m.geom), -drag[1], Real(0.1) * roundoff * std::abs(drag[1]));
    c->add_momentum_sources(1, sx, sy, sz);   // not the anchor level: nothing is added
    EXPECT_NEAR(erf_actuator::integrate_source(1, sy, &detJ, m.geom), -drag[1], Real(0.1) * roundoff * std::abs(drag[1]));
    // the cell-centred plot field
    amrex::MultiFab cells(m.ba, m.dm, 3, 0);
    c->cell_sources(0, cells, 0);
    EXPECT_LT(cells.min(1), 0.0) << "the air is pushed against the wind (-y) around the line";
    c->cell_sources(1, cells, 0);
    EXPECT_DOUBLE_EQ(cells.norm0(1), 0.0);
}

TEST(Conductors, WithoutDragOnFlowNothingIsPutIntoTheFlow)
{
    const std::string dir = scratch("nodrag");
    set_inputs(dir, "Tnodrag");
    Mesh m(false);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    EXPECT_FALSE(c->drag_on_flow());
    c->set_ground(nullptr, m.geom);
    c->advance(0, 0.0, 0.2, m.u, m.v, m.w, nullptr, nullptr, m.geom);
    EXPECT_GT(c->lines().front()->total_drag()[1], 1.0) << "the line still feels the drag";
    for (int d = 0; d < 3; ++d) { EXPECT_DOUBLE_EQ(c->source_integral()[d], 0.0); }
    amrex::MultiFab sy(amrex::convert(m.ba, amrex::IntVect(0,1,0)), m.dm, 1, 0);
    amrex::MultiFab sx(amrex::convert(m.ba, amrex::IntVect(1,0,0)), m.dm, 1, 0);
    amrex::MultiFab sz(amrex::convert(m.ba, amrex::IntVect(0,0,1)), m.dm, 1, 0);
    sx.setVal(0.0); sy.setVal(0.0); sz.setVal(0.0);
    c->add_momentum_sources(0, sx, sy, sz);
    EXPECT_DOUBLE_EQ(sy.norm0(), 0.0);
}

namespace {
std::string slurp (const std::string& fname)
{
    std::ifstream in(fname);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}
} // namespace

TEST(Conductors, ARestartContinuesTheLinesTheirSourcesStatisticsAndLogs)
{
    const std::string dir = scratch("restart");
    set_inputs(dir, "Trst");   // a prescribed 10 m/s crosswind along +y
    amrex::ParmParse pp("erf.conductors");
    pp.add("drag_on_flow", true);
    pp.add("node_output_int", 2);
    Mesh m(true);
    // the sources need only agree between the two runs here, so a unit Jacobian will do
    amrex::MultiFab detJ(m.ba, m.dm, 1, 1);
    detJ.setVal(1.0);
    const double dt = 0.2;

    // the run that writes the checkpoint after four steps and goes on for three more
    auto a = Conductors::create(0);
    ASSERT_TRUE(a);
    a->set_ground(m.znd.get(), m.geom);
    EXPECT_FALSE(a->restored());
    int step = 0;
    for (; step < 4; ++step) { a->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), &detJ, m.geom); }
    const std::string chk = dir + "/chk00004";
    std::filesystem::create_directories(chk);
    a->write_checkpoint(chk);
    // the state file records the frame MoorDyn's saved state is in
    EXPECT_NE(slurp(chk + "/conductors/state").find("surface_offset = 10000"), std::string::npos);
    amrex::MultiFab src_at_chk(m.ba, m.dm, 3, 0);
    a->cell_sources(0, src_at_chk, 0);
    ASSERT_GT(src_at_chk.norm0(1), 0.0) << "the lines must push on the air";
    const std::vector<Real> pos_at_chk = a->lines().front()->node_positions();
    for (; step < 7; ++step) { a->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), &detJ, m.geom); }
    const std::vector<Real> pos_end = a->lines().front()->node_positions();
    const std::string log_a = slurp(dir + "/Trst.dat");
    const std::string nodes_a = slurp(dir + "/Trst_nodes.dat");
    const std::string stats_a = slurp(dir + "/Trst_stats.csv");
    const std::string load_a = slurp(dir + "/total_load.dat");
    ASSERT_FALSE(log_a.empty());

    // the restarted run, in the same directory: it continues from the checkpoint
    auto b = Conductors::create(0);
    ASSERT_TRUE(b);
    b->set_ground(m.znd.get(), m.geom, chk);
    ASSERT_TRUE(b->restored());
    const auto& span_b = *b->lines().front();
    const std::vector<Real> pos_b = span_b.node_positions();
    ASSERT_EQ(pos_b.size(), pos_at_chk.size());
    for (std::size_t i = 0; i < pos_b.size(); ++i) {
        ASSERT_NEAR(pos_b[i], pos_at_chk[i], roundoff * std::max(Real(1.0), std::abs(pos_at_chk[i]))) << "component " << i;
    }
    EXPECT_GT(span_b.mid_offset(), 0.1) << "restored blown out, not hanging still";
    // the logs end at the checkpoint until the restarted run writes again
    EXPECT_LT(slurp(dir + "/Trst.dat").size(), log_a.size());
    // the momentum source is that of the step the checkpoint was written at
    b->restore_sources(0, m.u, m.znd.get(), &detJ, m.geom);
    amrex::MultiFab src_b(m.ba, m.dm, 3, 0);
    b->cell_sources(0, src_b, 0);
    amrex::MultiFab::Subtract(src_b, src_at_chk, 0, 0, 3, 0);
    EXPECT_LE(src_b.norm0(1), tight * src_at_chk.norm0(1));
    // the same three steps: the lines, the logs and the statistics come out as in the run without the restart
    for (step = 4; step < 7; ++step) { b->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), &detJ, m.geom); }
    const std::vector<Real> pos_b_end = span_b.node_positions();
    for (std::size_t i = 0; i < pos_end.size(); ++i) {
        EXPECT_NEAR(pos_b_end[i], pos_end[i], roundoff * std::max(Real(1.0), std::abs(pos_end[i]))) << "component " << i;
    }
    EXPECT_EQ(slurp(dir + "/Trst.dat"), log_a);
    EXPECT_EQ(slurp(dir + "/Trst_nodes.dat"), nodes_a);
    EXPECT_EQ(slurp(dir + "/Trst_stats.csv"), stats_a);
    EXPECT_EQ(slurp(dir + "/total_load.dat"), load_a);

    // a checkpoint without conductor state (a precursor's, say): the lines start afresh, from
    // still air, at the restart time; MoorDyn's clock starts at zero there
    const std::string bare = dir + "/chk_bare";
    std::filesystem::create_directories(bare);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(m.znd.get(), m.geom, bare);
    EXPECT_FALSE(c->restored());
    EXPECT_NEAR(c->lines().front()->mid_offset(), 0.0, 1.0e-3);
    for (step = 7; step < 9; ++step) { c->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), &detJ, m.geom); }
    EXPECT_GT(c->lines().front()->mid_offset(), 0.01) << "the span started at the restart time moves";
    // and a restart from a checkpoint of that run continues it on the same clock
    const std::string chk_c = dir + "/chk00009";
    std::filesystem::create_directories(chk_c);
    c->write_checkpoint(chk_c);
    auto d = Conductors::create(0);
    ASSERT_TRUE(d);
    d->set_ground(m.znd.get(), m.geom, chk_c);
    ASSERT_TRUE(d->restored());
    c->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), &detJ, m.geom);
    d->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), &detJ, m.geom);
    const std::vector<Real> pc = c->lines().front()->node_positions();
    const std::vector<Real> pd = d->lines().front()->node_positions();
    for (std::size_t i = 0; i < pc.size(); ++i) {
        EXPECT_NEAR(pd[i], pc[i], roundoff * std::max(Real(1.0), std::abs(pc[i]))) << "component " << i;
    }
}

TEST(Conductors, TrimmingALogKeepsTheHeaderAndTheRowsUpToTheCheckpoint)
{
    const std::string dir = scratch("trim");
    const std::string f = dir + "/log.dat";
    {
        std::ofstream out(f);
        out << "time a b\n# a comment\n0.5 1 2\n1 3 4\n1.0000000001 9 9\n1.5 5 6\n2 7 8\n";
    }
    erf_conductors::trim_log_after(f, 1.0);
    EXPECT_EQ(slurp(f), "time a b\n# a comment\n0.5 1 2\n1 3 4\n1.0000000001 9 9\n")
        << "a row printed to ten digits at the checkpoint time stays; later rows go";
    erf_conductors::trim_log_after(f, 0.0);
    EXPECT_EQ(slurp(f), "time a b\n# a comment\n");
    erf_conductors::trim_log_after(dir + "/missing.dat", 1.0);
    EXPECT_FALSE(std::filesystem::exists(dir + "/missing.dat"));
    // a log whose rows carry the time a step starts at loses the row at the checkpoint time too
    {
        std::ofstream out(f, std::ios::trunc);
        out << "time a\n0.5 1\n1 3\n1.0000000001 9\n1.5 5\n";
    }
    erf_conductors::trim_log_after(f, 1.0, true);
    EXPECT_EQ(slurp(f), "time a\n0.5 1\n");
}

TEST(Conductors, ASectionIsPlacedOnTheTerrainAtEveryTowerAndLogsEachSpan)
{
    const std::string dir = scratch("section");
    set_inputs(dir, "Tsec", true, {{100.0, 500.0, 30.0}}, {{1000.0, 500.0, 30.0}});
    amrex::ParmParse ps("erf.conductors.Tsec");
    ps.remove("length");
    ps.addarr("length", std::vector<Real>{301.5, 301.5, 301.5});
    ps.addarr("towers", std::vector<Real>{400.0, 500.0, 30.0, 700.0, 500.0, 30.0});
    ps.add("insulator_length", 2.5);
    ps.add("insulator_mass", 60.0);
    Mesh m(true);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(m.znd.get(), m.geom);
    const auto& line = *c->lines().front();
    ASSERT_EQ(line.num_spans(), 3);
    EXPECT_EQ(line.inputs().lengths, (std::vector<Real>{301.5, 301.5, 301.5})) << "given lengths are kept";
    // the towers stand on the ramp, and the conductor hangs 2.5 m under each
    for (int k = 1; k <= 2; ++k) {
        const Real x = 100.0 + 300.0 * k;
        const auto p = line.node_position(line.span_first_node(k));
        EXPECT_NEAR(p[0], x, 0.05) << "tower " << k;
        EXPECT_NEAR(p[2], m.h(x, 500.0) + 30.0 - 2.5, 0.05) << "tower " << k;
    }
    std::ifstream g(dir + "/ground.dat");
    std::string header, name, label;
    Real x, y, ground, z;
    std::getline(g, header);
    std::vector<std::string> labels;
    while (g >> name >> label >> x >> y >> ground >> z) {
        labels.push_back(label);
        EXPECT_NEAR(ground, m.h(x, y), postol) << label;
        EXPECT_NEAR(z, m.h(x, y) + 30.0, postol) << label;
    }
    EXPECT_EQ(labels, (std::vector<std::string>{"a", "t1", "t2", "b"}));
    for (int s = 0; s < 3; ++s) { c->advance(0, 0.2 * s, 0.2, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    // a log per span, one for the strings, and their statistics
    for (const char* f : {"/Tsec_span1.dat", "/Tsec_span2.dat", "/Tsec_span3.dat", "/Tsec_insulators.dat",
                          "/Tsec_span2_stats.csv", "/Tsec_insulators_stats.csv"}) {
        EXPECT_TRUE(std::filesystem::exists(dir + f)) << f;
    }
    std::ifstream ins(dir + "/Tsec_insulators.dat");
    std::getline(ins, header);
    EXPECT_EQ(header, "time t1_swing_deg t1_across_deg t1_tension t2_swing_deg t2_across_deg t2_tension");
    // the strings' statistics: a sample per step, the strings swinging with the +y wind
    std::ifstream st(dir + "/Tsec_insulators_stats.csv");
    std::getline(st, header);
    std::string row;
    std::getline(st, row);
    EXPECT_EQ(row.substr(0, row.find(',')), "3") << row;
    EXPECT_NE(row.find(",t1_swing_deg,"), std::string::npos) << row;
    const Real mean_swing = std::stod(row.substr(row.find("t1_swing_deg,") + 13));
    EXPECT_GT(mean_swing, 0.1) << row;
    EXPECT_FALSE(std::filesystem::exists(dir + "/separation.dat")) << "one line: no pairs to watch";
    ps.remove("towers");
    ps.remove("insulator_length");
    ps.remove("insulator_mass");
}

TEST(Conductors, TheClosestApproachOfTwoLinesIsFlaggedAgainstTheFlashoverDistance)
{
    const std::string dir = scratch("clash");
    // two parallel spans 6 m apart across a +y wind; the upwind one is lighter, swings further and
    // closes on the other
    set_inputs(dir, "Pa", true, {{300.0, 500.0, 30.0}}, {{600.0, 500.0, 30.0}});
    set_inputs(dir, "Pb", true, {{300.0, 506.0, 30.0}}, {{600.0, 506.0, 30.0}});
    amrex::ParmParse pp("erf.conductors");
    pp.addarr("lines", std::vector<std::string>{"Pa", "Pb"});
    pp.add("flashover_distance", 5.5);
    amrex::ParmParse("erf.conductors.Pa").add("mass_per_length", 1.0);
    amrex::ParmParse("erf.conductors.Pb").add("mass_per_length", 3.0);
    Mesh m(false);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(nullptr, m.geom);
    ASSERT_EQ(c->pairs().size(), 1u);
    EXPECT_NEAR(c->separations()[0].distance, 6.0, 0.1) << "at rest the spans hang side by side";
    Real closest = 1.0e30;
    for (int s = 0; s < 40; ++s) {
        c->advance(0, 0.25 * s, 0.25, m.u, m.v, m.w, nullptr, nullptr, m.geom);
        // the distance is the exact minimum between the two conductors at their current positions
        const auto exact = erf_conductors::closest_polylines(c->lines()[0]->conductor_path(), c->lines()[1]->conductor_path());
        ASSERT_NEAR(c->separations()[0].distance, exact.distance, tight * exact.distance) << "step " << s;
        closest = std::min(closest, c->separations()[0].distance);
    }
    EXPECT_LT(closest, 5.5) << "the lighter line must close on the heavier one";
    // separation.dat: the pair's columns, the flag off at rest and on once inside the flashover distance
    std::ifstream f(dir + "/separation.dat");
    std::string header;
    std::getline(f, header);
    EXPECT_EQ(header, "time Pa-Pb_distance Pa-Pb_x Pa-Pb_y Pa-Pb_z Pa-Pb_clash");
    std::vector<std::array<Real,6>> rows;
    std::array<Real,6> r{};
    while (f >> r[0] >> r[1] >> r[2] >> r[3] >> r[4] >> r[5]) { rows.push_back(r); }
    ASSERT_EQ(rows.size(), 41u) << "the initial row and one per step";
    for (const auto& row : rows) { EXPECT_EQ(row[5], (row[1] < 5.5) ? 1.0 : 0.0) << "t = " << row[0]; }
    EXPECT_EQ(rows.front()[5], 0.0);
    EXPECT_EQ(rows.back()[5], 1.0);
    EXPECT_GT(rows.back()[3], 500.0);
    EXPECT_LT(rows.back()[3], 506.0) << "the closest point lies between the lines";
    // the statistics: the fraction of samples in a clash is the mean of the flag
    const std::string stats = slurp(dir + "/separation_Pa-Pb_stats.csv");
    EXPECT_NE(stats.find(",clash,"), std::string::npos) << stats;
    pp.addarr("lines", std::vector<std::string>{});
    amrex::ParmParse("erf.conductors.Pa").remove("mass_per_length");
    amrex::ParmParse("erf.conductors.Pb").remove("mass_per_length");
}

TEST(Conductors, ARestartContinuesASectionItsStringsAndTheSeparationOfTheLines)
{
    const std::string dir = scratch("restart_circuit");
    // a section on strings and a single span beside its middle span
    set_inputs(dir, "Rs", true, {{100.0, 500.0, 30.0}}, {{1000.0, 500.0, 30.0}});
    set_inputs(dir, "Rp", true, {{400.0, 506.0, 30.0}}, {{700.0, 506.0, 30.0}});
    amrex::ParmParse pp("erf.conductors");
    pp.addarr("lines", std::vector<std::string>{"Rs", "Rp"});
    amrex::ParmParse ps("erf.conductors.Rs");
    ps.remove("length");
    ps.addarr("length", std::vector<Real>{301.5, 301.5, 301.5});
    ps.addarr("towers", std::vector<Real>{400.0, 500.0, 30.0, 700.0, 500.0, 30.0});
    ps.add("insulator_length", 2.5);
    ps.add("insulator_mass", 60.0);
    Mesh m(false);
    const double dt = 0.25;
    auto a = Conductors::create(0);
    ASSERT_TRUE(a);
    a->set_ground(nullptr, m.geom);
    int step = 0;
    for (; step < 4; ++step) { a->advance(0, dt * step, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom); }
    const std::string chk = dir + "/chk00004";
    std::filesystem::create_directories(chk);
    a->write_checkpoint(chk);
    for (; step < 7; ++step) { a->advance(0, dt * step, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom); }
    const auto files = std::vector<std::string>{"/separation.dat", "/separation_Rs-Rp_stats.csv", "/Rs_span2.dat",
                                                "/Rs_insulators.dat", "/Rs_insulators_stats.csv", "/Rp.dat"};
    std::vector<std::string> before;
    for (const auto& f : files) { before.push_back(slurp(dir + f)); ASSERT_FALSE(before.back().empty()) << f; }
    const Real sep = a->separations()[0].distance;

    auto b = Conductors::create(0);
    ASSERT_TRUE(b);
    b->set_ground(nullptr, m.geom, chk);
    ASSERT_TRUE(b->restored());
    EXPECT_LT(slurp(dir + "/separation.dat").size(), before[0].size()) << "the separation log ends at the checkpoint";
    for (step = 4; step < 7; ++step) { b->advance(0, dt * step, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom); }
    for (std::size_t i = 0; i < files.size(); ++i) { EXPECT_EQ(slurp(dir + files[i]), before[i]) << files[i]; }
    EXPECT_NEAR(b->separations()[0].distance, sep, roundoff * sep);
    EXPECT_NEAR(b->lines()[0]->insulator_swing(0), a->lines()[0]->insulator_swing(0), roundoff);
    pp.addarr("lines", std::vector<std::string>{});
    ps.remove("towers");
    ps.remove("insulator_length");
    ps.remove("insulator_mass");
}

TEST(Conductors, TransformersTakeThePullOfTheLinesEndingOnThemAndContinueAcrossARestart)
{
    const std::string dir = scratch("transformers");
    // a span from T1 to T2 on the ramp, 12 m above the ground at its ends (6 m over the tops), and
    // a second line dead-ended on open ground 2 m beside T2 and 3 m higher than its top; a +y wind
    set_inputs(dir, "Lt", true, {{302.0, 500.0, 12.0}}, {{598.0, 500.0, 12.0}});
    set_inputs(dir, "Lo", true, {{606.0, 500.0, 9.0}}, {{606.0, 800.0, 9.0}});
    amrex::ParmParse("erf.conductors.Lt").remove("length");
    amrex::ParmParse("erf.conductors.Lt").add("length", 297.5);
    amrex::ParmParse pp("erf.conductors");
    pp.addarr("lines", std::vector<std::string>{"Lt", "Lo"});
    pp.addarr("transformers", std::vector<std::string>{"T1", "T2"});
    for (const char* t : {"T1", "T2"}) {
        amrex::ParmParse pt(std::string("erf.conductors.") + t);
        pt.addarr("position", std::vector<Real>{t[1] == '1' ? Real(300.0) : Real(600.0), Real(500.0)});
        pt.addarr("size", std::vector<Real>{8.0, 5.0, 6.0});
    }
    amrex::ParmParse("erf.conductors.T1").add("allowable_force", 1.0);   // any pull is over it
    Mesh m(true);
    const double dt = 0.25;
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(m.znd.get(), m.geom);
    ASSERT_EQ(c->transformers().size(), 2u);
    const auto& T1 = c->transformers()[0];
    const auto& T2 = c->transformers()[1];
    // the boxes stand on the ramp under their centres; Lt ends on both, Lo on neither
    EXPECT_NEAR(T1.base()[2], m.h(300.0, 500.0), roundoff * 1000);
    EXPECT_NEAR(T2.box_hi()[2], m.h(600.0, 500.0) + 6.0, roundoff * 1000);
    ASSERT_EQ(T1.ends().size(), 1u);
    ASSERT_EQ(T2.ends().size(), 1u);
    EXPECT_EQ(T1.ends()[0].end, 0);
    EXPECT_EQ(T2.ends()[0].end, 1);
    std::ifstream g(dir + "/ground.dat");
    std::string row, name, label;
    int boxes = 0;
    while (std::getline(g, row)) {
        std::istringstream ls(row);
        Real x, y, ground, top;
        if (ls >> name >> label >> x >> y >> ground >> top && label == "transformer") {
            ++boxes;
            EXPECT_NEAR(ground, m.h(x, y), postol) << name;
            EXPECT_NEAR(top, m.h(x, y) + 6.0, postol) << name;
        }
    }
    EXPECT_EQ(boxes, 2);

    auto check = [&](const Conductors& cc, const char* when) {
        const auto& span = *cc.lines()[0];
        const auto& L1 = cc.transformer_loads()[0];
        const auto& L2 = cc.transformer_loads()[1];
        const auto fa = span.end_force(0);
        const auto fb = span.end_force(1);
        for (int d = 0; d < 3; ++d) {
            EXPECT_EQ(L1.force[d], fa[d]) << when << " T1 carries Lt's end_a alone, component " << d;
            EXPECT_EQ(L2.force[d], fb[d]) << when << " T2 carries Lt's end_b alone, component " << d;
        }
        // the moment about the base centre of the pull at the end
        const auto& e = span.inputs().end_a;
        const auto b = cc.transformers()[0].base();
        const Real rx = e[0] - b[0], ry = e[1] - b[1], rz = e[2] - b[2];
        EXPECT_NEAR(L1.moment[0], ry * fa[2] - rz * fa[1], roundoff * 1.0e3 * std::abs(L1.moment[1])) << when;
        EXPECT_NEAR(L1.moment[1], rz * fa[0] - rx * fa[2], roundoff * 1.0e3 * std::abs(L1.moment[1])) << when;
        EXPECT_GT(L1.horizontal_force, 0.0);
        EXPECT_TRUE(L1.over_allowable) << when;
        EXPECT_FALSE(L2.over_allowable) << when << " no allowable on T2";
        // the closest conductor to each box: the exact distance over every line
        for (std::size_t t = 0; t < 2; ++t) {
            const auto& tr = cc.transformers()[t];
            Real best = 1.0e30;
            for (const auto& sp : cc.lines()) {
                best = std::min(best, erf_conductors::closest_polyline_box(sp->conductor_path(), tr.box_lo(), tr.box_hi()).distance);
            }
            EXPECT_EQ(cc.transformer_clearances()[t].distance, best) << when << " " << tr.name();
        }
    };
    check(*c, "at rest");
    // Lo's end is about 3.9 m from T2's top edge, nearer than Lt's 6 m standoff; only Lt comes near T1
    EXPECT_GT(c->transformer_clearances()[1].distance, 3.0);
    EXPECT_LT(c->transformer_clearances()[1].distance, 4.5);
    EXPECT_LE(c->transformer_clearances()[0].distance, 6.0 + 0.2);

    int step = 0;
    for (; step < 3; ++step) { c->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    check(*c, "blown out");
    const std::string chk = dir + "/chk00003";
    std::filesystem::create_directories(chk);
    c->write_checkpoint(chk);
    for (; step < 6; ++step) { c->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    const std::string log = slurp(dir + "/transformers.dat");
    const std::string stats1 = slurp(dir + "/transformer_T1_stats.csv");
    const std::string stats2 = slurp(dir + "/transformer_T2_stats.csv");
    // the log: a row at the start and one per step, T1 over its allowable throughout
    std::istringstream lg(log);
    std::string header;
    std::getline(lg, header);
    EXPECT_EQ(header.rfind("time T1_Fx T1_Fy T1_Fz T1_Fh T1_Mx T1_My T1_Mh T1_over T1_clearance T1_clash T2_Fx ", 0), 0u) << header;
    int rows = 0;
    while (std::getline(lg, row)) {
        std::istringstream ls(row);
        std::vector<Real> v;
        Real x;
        while (ls >> x) { v.push_back(x); }
        ASSERT_EQ(v.size(), 21u) << row;
        EXPECT_EQ(v[8], 1.0) << "T1_over";
        EXPECT_EQ(v[18], 0.0) << "T2_over";
        EXPECT_EQ(v[20], v[19] < 1.0 ? 1.0 : 0.0) << "T2_clash against the 1 m flashover distance";
        ++rows;
    }
    EXPECT_EQ(rows, 7);
    EXPECT_NE(stats1.find(",overturning_moment,"), std::string::npos) << stats1;

    // restarted from the checkpoint: the same log, statistics and loads
    auto r = Conductors::create(0);
    ASSERT_TRUE(r);
    r->set_ground(m.znd.get(), m.geom, chk);
    ASSERT_TRUE(r->restored());
    for (step = 3; step < 6; ++step) { r->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    EXPECT_EQ(slurp(dir + "/transformers.dat"), log);
    EXPECT_EQ(slurp(dir + "/transformer_T1_stats.csv"), stats1);
    EXPECT_EQ(slurp(dir + "/transformer_T2_stats.csv"), stats2);
    for (int d = 0; d < 3; ++d) { EXPECT_EQ(r->transformer_loads()[0].force[d], c->transformer_loads()[0].force[d]); }

    pp.addarr("lines", std::vector<std::string>{});
    pp.addarr("transformers", std::vector<std::string>{});
    amrex::ParmParse("erf.conductors.T1").remove("allowable_force");
}

TEST(Conductors, AStringingTensionSetsTheLengthsFromTheChordsOnTheTerrain)
{
    const std::string dir = scratch("stringing");
    // a section up the ramp, clamped at the towers, with spans of 300, 150 and 450 m strung to 20 kN
    set_inputs(dir, "Tstr", true, {{100.0, 500.0, 30.0}}, {{1000.0, 500.0, 30.0}});
    amrex::ParmParse ps("erf.conductors.Tstr");
    ps.remove("length");
    ps.add("stringing_tension", 2.0e4);
    ps.addarr("towers", std::vector<Real>{400.0, 500.0, 30.0, 550.0, 500.0, 30.0});
    Mesh m(true);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(m.znd.get(), m.geom);
    const auto& line = *c->lines().front();
    const auto& s = line.inputs();
    ASSERT_EQ(s.lengths.size(), 3u);
    const std::array<Real,3> h{{300.0, 150.0, 450.0}};
    for (int k = 0; k < 3; ++k) {
        // the chord between the attachments on the ramp, which rises 5 m per 100 m
        const Real c_on_terrain = h[k] * std::sqrt(Real(1.0) + m.slope_x * m.slope_x);
        EXPECT_NEAR(s.chord(k), c_on_terrain, roundoff * 1.0e4) << k;
    }
    // every span pulls on its towers with about the stringing tension along the line: the real
    // MoorDyn within the parabola's few per cent, the stub (no stretch) within its sag error
    const Real tol = erf_moordyn::is_stub() ? Real(0.25) : Real(0.04);
    for (int k = 0; k < 3; ++k) {
        const Real T = line.tension_a(k);
        EXPECT_GT(T / 2.0e4, 1.0 - tol) << "span " << k;
        EXPECT_LT(T / 2.0e4, 1.0 + 2.0 * tol) << "span " << k;
    }
    const auto fa = line.end_force(0);
    EXPECT_NEAR(std::sqrt(fa[0] * fa[0] + fa[1] * fa[1]) / 2.0e4, 1.0, tol) << "the dead end takes the stringing tension";
    ps.remove("stringing_tension");
    ps.remove("towers");
}

namespace {
// a section of three spans along x on the ramp with lattice towers at x = 400 and 700
void set_towered_section (const std::string& dir, const std::string& name)
{
    set_inputs(dir, name, true, {{100.0, 500.0, 30.0}}, {{1000.0, 500.0, 30.0}});
    amrex::ParmParse ps("erf.conductors." + name);
    ps.remove("length");
    ps.addarr("length", std::vector<Real>{301.5, 301.5, 301.5});
    ps.addarr("towers", std::vector<Real>{400.0, 500.0, 30.0, 700.0, 500.0, 30.0});
    ps.add("tower_type", std::string("lat"));
    amrex::ParmParse pp("erf.conductors");
    pp.addarr("tower_types", std::vector<std::string>{"lat"});
    amrex::ParmParse pt("erf.conductors.lat");
    for (const char* k : {"base_width", "top_width", "solidity", "arm_length", "arm_depth"}) { pt.remove(k); }
    pt.add("base_width", 6.0);
    pt.add("top_width", 1.5);
    pt.add("solidity", 0.2);
    pt.add("arm_length", 12.0);
    pt.add("arm_depth", 1.2);
    for (const char* k : {"weight", "allowable_uplift", "allowable_compression"}) { pt.remove(k); }
    pt.add("weight", 9.0e4);
}
void clear_towered_section (const std::string& name)
{
    amrex::ParmParse pt("erf.conductors.lat");
    for (const char* k : {"frequency", "damping_ratio"}) { pt.remove(k); }
    amrex::ParmParse ps("erf.conductors." + name);
    ps.remove("towers"); ps.remove("tower_type");
    amrex::ParmParse pp("erf.conductors");
    pp.addarr("tower_types", std::vector<std::string>{});
    pp.remove("drag_on_flow");
    pp.addarr("lines", std::vector<std::string>{});
}
} // namespace

TEST(Conductors, LatticeTowersStandAtTheSuspensionPointsAndCarryTheWindsDrag)
{
    const std::string dir = scratch("towers");
    set_towered_section(dir, "Tw");
    Mesh m(true);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(m.znd.get(), m.geom);
    ASSERT_EQ(c->towers().size(), 2u);
    for (int k = 0; k < 2; ++k) {
        const auto& tw = c->towers()[static_cast<std::size_t>(k)];
        const Real x = 400.0 + 300.0 * k;
        EXPECT_EQ(tw.name(), "Tw_t" + std::to_string(k + 1));
        EXPECT_NEAR(tw.base()[0], x, (std::is_same<Real, float>::value) ? Real(1.0e-4) : Real(1.0e-9));
        EXPECT_NEAR(tw.base()[2], m.h(x, 500.0), roundoff * 1000) << "the base stands on the ramp";
        EXPECT_NEAR(tw.arm_height(), 30.0, roundoff * 1000) << "the cross-arm at the conductor's height";
        EXPECT_NEAR(std::abs(tw.across()[1]), 1.0, tight) << "the cross-arm across a line along x";
    }
    // the prescribed +y wind runs along the cross-arms: only the bodies carry it; the body's drag by
    // hand, q Cf phi (b0 + b1)/2 H (N), with Cf = 4 phi^2 - 5.9 phi + 4 for the solidity phi = 0.2
    const Real U = 10.0, q = 0.5 * 1.2 * U * U, cf = 4.0 * 0.04 - 5.9 * 0.2 + 4.0;
    const Real body = q * cf * 0.2 * 0.5 * (6.0 + 1.5) * 30.0;
    // the line's pull on each tower at the start of the last step, which the towers carry over it
    std::vector<std::array<Real,3>> pull;
    for (int s = 0; s < 3; ++s) {
        pull.clear();
        for (int t = 0; t < 2; ++t) { pull.push_back(c->lines()[0]->tower_force(t)); }
        c->advance(0, 0.25 * s, 0.25, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom);
    }
    for (const auto& tw : c->towers()) {
        const auto F = tw.total_force();
        EXPECT_NEAR(F[1], body, 1.0e-6 * body) << tw.name();
        EXPECT_NEAR(F[0], 0.0, roundoff * body) << tw.name();
        // M_x = -z F_y: the body's q cf phi int w(z) z dz, short by the midpoint sum's H^3 / (12 n^2) on z^2
        const Real H = 30.0, n = 10.0;
        const Real mom = q * cf * 0.2 * (6.0 * H * H / 2.0 + (1.5 - 6.0) / H * (H * H * H / 3.0 - H * H * H / (12.0 * n * n)));
        EXPECT_NEAR(tw.base_moment()[0], -mom, 1.0e-6 * mom) << tw.name();
    }
    // each tower carries its line's pull where the line hangs from it, and its footings the lot
    for (std::size_t t = 0; t < 2; ++t) {
        const auto& tw = c->towers()[t];
        const auto& F = pull[t];
        for (int d = 0; d < 3; ++d) { EXPECT_EQ(tw.line_force()[d], F[d]) << tw.name() << " " << d; }
        EXPECT_LT(F[2], 0.0) << "the clamped conductor weighs on the tower";
        const auto L = tw.foundation();
        EXPECT_NEAR(L.vertical, 9.0e4 - F[2] - tw.total_force()[2], roundoff * L.vertical);
        EXPECT_NEAR(L.legs[0] + L.legs[1] + L.legs[2] + L.legs[3], L.vertical, roundoff * L.vertical);
        EXPECT_NEAR(L.shear, std::hypot(F[0] + tw.total_force()[0], F[1] + tw.total_force()[1]), roundoff * L.shear);
    }
    // towers.dat: a row at the start of every step with every tower's drag, line pull and foundation load
    std::ifstream f(dir + "/towers.dat");
    std::string header, row;
    std::getline(f, header);
    EXPECT_EQ(header.rfind("time Tw_t1_drag_Fx Tw_t1_drag_Fy Tw_t1_drag_Fz Tw_t1_line_Fx Tw_t1_line_Fy Tw_t1_line_Fz Tw_t1_shear "
                           "Tw_t1_overturning Tw_t1_vertical Tw_t1_max_compression Tw_t1_max_uplift Tw_t1_over Tw_t2_drag_Fx", 0), 0u)
        << header;
    int rows = 0;
    while (std::getline(f, row)) {
        std::istringstream ls(row);
        std::vector<Real> v;
        Real x;
        while (ls >> x) { v.push_back(x); }
        ASSERT_EQ(v.size(), 25u);
        EXPECT_NEAR(v[2], body, 1.0e-6 * body);
        EXPECT_EQ(v[12], 0.0) << "no allowable given";
        ++rows;
    }
    EXPECT_EQ(rows, 3);
    for (const char* q : {",drag_h,", ",line_h,", ",shear,", ",overturning,", ",max_compression,", ",max_uplift,", ",over_allowable,"}) {
        EXPECT_NE(slurp(dir + "/tower_Tw_t1_stats.csv").find(q), std::string::npos) << q;
    }
    clear_towered_section("Tw");
}

TEST(Conductors, TheTowersDragGoesIntoTheFlowWithTheLinesAndSurvivesARestart)
{
    const std::string dir = scratch("towers_flow");
    set_towered_section(dir, "Tf");
    amrex::ParmParse pp("erf.conductors");
    pp.add("drag_on_flow", true);
    pp.remove("prescribed_velocity");
    Mesh m(true);
    m.fill_crosswind();
    const double dt = 0.25;
    auto a = Conductors::create(0);
    ASSERT_TRUE(a);
    a->set_ground(m.znd.get(), m.geom);
    int step = 0;
    for (; step < 3; ++step) { a->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    // the source integrates to minus the drag on the lines and on the towers
    std::array<Real,3> drag{{0.0, 0.0, 0.0}};
    for (const auto& s : a->lines()) { const auto f = s->total_drag(); for (int d = 0; d < 3; ++d) { drag[d] += f[d]; } }
    Real towers_y = 0.0;
    for (const auto& t : a->towers()) { const auto f = t.total_force(); towers_y += f[1]; for (int d = 0; d < 3; ++d) { drag[d] += f[d]; } }
    EXPECT_GT(towers_y, 0.0);
    for (int d = 0; d < 3; ++d) { EXPECT_NEAR(a->source_integral()[d], -drag[d], 1.0e-6 * std::abs(drag[1]) + roundoff) << d; }
    const std::string chk = dir + "/chk00003";
    std::filesystem::create_directories(chk);
    a->write_checkpoint(chk);
    const auto src = a->source_integral();
    for (; step < 6; ++step) { a->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    const std::string log = slurp(dir + "/towers.dat");
    const std::string stats = slurp(dir + "/tower_Tf_t2_stats.csv");

    auto b = Conductors::create(0);
    ASSERT_TRUE(b);
    b->set_ground(m.znd.get(), m.geom, chk);
    ASSERT_TRUE(b->restored());
    // the restored towers' drag is the checkpointed step's, and so is the source spread from it
    b->restore_sources(0, m.u, m.znd.get(), nullptr, m.geom);
    for (int d = 0; d < 3; ++d) { EXPECT_EQ(b->source_integral()[d], src[d]) << d; }
    for (step = 3; step < 6; ++step) { b->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    EXPECT_EQ(slurp(dir + "/towers.dat"), log);
    EXPECT_EQ(slurp(dir + "/tower_Tf_t2_stats.csv"), stats);
    clear_towered_section("Tf");
}

namespace {
// the section's towers bend at 2 Hz with the damping ratio given
void make_towers_move (Real damping)
{
    amrex::ParmParse pt("erf.conductors.lat");
    for (const char* k : {"frequency", "damping_ratio"}) { pt.remove(k); }
    pt.add("frequency", 2.0);
    pt.add("damping_ratio", damping);
}
} // namespace

TEST(Conductors, MovingTowersSettleWhereTheirStiffnessBalancesTheWindAndTheLine)
{
    const double dt = 0.25;
    // the same section with towers that stand still, for the line's pull on rigid towers
    const std::string rdir = scratch("still_towers");
    set_towered_section(rdir, "Ms");
    Mesh m(true);
    std::vector<std::array<Real,3>> rigid_pull;
    {
        auto c = Conductors::create(0);
        ASSERT_TRUE(c);
        c->set_ground(m.znd.get(), m.geom);
        for (int s = 0; s < 40; ++s) { c->advance(0, dt * s, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
        for (const auto& tw : c->towers()) { rigid_pull.push_back(tw.line_force()); }
        EXPECT_EQ(c->tower_models()[0], nullptr);
    }
    clear_towered_section("Ms");
    const std::string dir = scratch("moving_towers");
    set_towered_section(dir, "Mv");
    make_towers_move(0.3);   // heavily damped, so that they settle in a few periods
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(m.znd.get(), m.geom);
    ASSERT_TRUE(c->lines()[0]->towers_move());
    for (int s = 0; s < 40; ++s) { c->advance(0, dt * s, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    for (std::size_t t = 0; t < 2; ++t) {
        const auto& tw = c->towers()[t];
        const auto* model = dynamic_cast<const erf_towers::OneModeTower*>(c->tower_models()[t].get());
        ASSERT_NE(model, nullptr);
        // at rest, the stiffness carries the members' drag by the mode shape and the line's pull
        std::array<double,2> Q{{tw.line_force()[0], tw.line_force()[1]}};
        for (std::size_t i = 0; i < tw.nodes().size(); ++i) {
            for (int d = 0; d < 2; ++d) { Q[d] += model->mode_shape(i) * tw.loads()[3*i+d]; }
        }
        // (the real MoorDyn's conductors still swing slowly after 10 s, lightly damped, and the tower
        // follows their pull, hence the 2 % tolerance below)
        const auto x = tw.arm_displacement();
        EXPECT_GT(x[1], 1.0e-3) << tw.name() << " leans with the +y wind";
        EXPECT_NEAR(x[1] * model->stiffness() / Q[1], 1.0, 0.02) << tw.name();
        EXPECT_NEAR(x[0] * model->stiffness(), Q[0], 0.02 * Q[1]) << tw.name();
        EXPECT_LT(std::abs(model->v()[1]), 0.02 * 2.0 * 3.14159265 * model->frequency() * x[1]) << tw.name() << " has settled";
        RecordProperty(tw.name() + "_lean_m", std::to_string(x[1]));
        // a few millimetres of lean barely change the line's pull from the rigid towers'
        const auto& Fr = rigid_pull[t];
        for (int d = 0; d < 3; ++d) {
            EXPECT_NEAR(tw.line_force()[d], Fr[d], 0.02 * std::abs(Fr[2])) << tw.name() << " dir " << d;
        }
        // the line hangs from the cross-arm where the tower has taken it
        const auto p = c->lines()[0]->node_position(c->lines()[0]->span_first_node(static_cast<int>(t) + 1));
        const Real ptol = (std::is_same<Real, float>::value) ? Real(1.0e-4) : Real(1.0e-6);   // a Real's spacing at 500 m
        EXPECT_NEAR(p[0], 400.0 + 300.0 * t + x[0], ptol);
        EXPECT_NEAR(p[1], 500.0 + x[1], ptol);
    }
    // towers.dat carries each moving tower's cross-arm displacement, and the statistics its size
    std::ifstream f(dir + "/towers.dat");
    std::string header, row, last;
    std::getline(f, header);
    EXPECT_NE(header.find("Mv_t1_over Mv_t1_arm_dx Mv_t1_arm_dy Mv_t2_drag_Fx"), std::string::npos) << header;
    while (std::getline(f, row)) { last = row; }
    std::istringstream ls(last);
    std::vector<Real> v;
    Real x;
    while (ls >> x) { v.push_back(x); }
    ASSERT_EQ(v.size(), 29u);
    EXPECT_GT(v[14], 1.0e-3) << "the row's cross-arm displacement across";
    EXPECT_NE(slurp(dir + "/tower_Mv_t1_stats.csv").find(",arm_displacement"), std::string::npos);
    clear_towered_section("Mv");
}

TEST(Conductors, MovingTowersContinueAcrossARestart)
{
    const std::string dir = scratch("moving_restart");
    set_towered_section(dir, "Mr");
    make_towers_move(0.02);   // still swaying at the checkpoint
    Mesh m(true);
    const double dt = 0.25;
    auto a = Conductors::create(0);
    ASSERT_TRUE(a);
    a->set_ground(m.znd.get(), m.geom);
    int step = 0;
    for (; step < 3; ++step) { a->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    const std::string chk = dir + "/chk00003";
    std::filesystem::create_directories(chk);
    a->write_checkpoint(chk);
    for (; step < 6; ++step) { a->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    const auto* ma = dynamic_cast<const erf_towers::OneModeTower*>(a->tower_models()[1].get());
    ASSERT_NE(ma, nullptr);
    EXPECT_GT(std::abs(ma->v()[1]), 1.0e-4) << "the test needs the towers moving";
    const std::string towers = slurp(dir + "/towers.dat");
    const std::string stats = slurp(dir + "/tower_Mr_t2_stats.csv");
    const std::string span = slurp(dir + "/Mr_span2.dat");

    auto b = Conductors::create(0);
    ASSERT_TRUE(b);
    b->set_ground(m.znd.get(), m.geom, chk);
    ASSERT_TRUE(b->restored());
    for (step = 3; step < 6; ++step) { b->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    const auto* mb = dynamic_cast<const erf_towers::OneModeTower*>(b->tower_models()[1].get());
    ASSERT_NE(mb, nullptr);
    EXPECT_EQ(mb->q()[1], ma->q()[1]);
    EXPECT_EQ(mb->v()[1], ma->v()[1]);
    EXPECT_EQ(slurp(dir + "/towers.dat"), towers);
    EXPECT_EQ(slurp(dir + "/tower_Mr_t2_stats.csv"), stats);
    EXPECT_EQ(slurp(dir + "/Mr_span2.dat"), span);
    clear_towered_section("Mr");
}

namespace {
// a circuit along x on the ramp: the middle phase C2 owns the lattice towers at x = 400 and 700,
// the outer phases C1 and C3 hang 5.5 m to either side of it on the cross-arm (1.5 m at the dead
// ends) and the shield wire CS 7 m above the cross-arm on the peak; the phases on strings
void set_circuit (const std::string& dir, Real damping = 0.0)
{
    amrex::ParmParse pp("erf.conductors");
    pp.addarr("lines", std::vector<std::string>{"C1", "C2", "C3", "CS"});
    pp.add("diagnostics_dir", dir);
    pp.add("air_density", 1.2);
    for (const char* key : {"prescribed_velocity", "drag_on_flow", "epsilon", "node_output_int", "stats_start", "flashover_distance"}) { pp.remove(key); }
    pp.addarr("prescribed_velocity", std::vector<Real>{0.0, 10.0, 0.0});
    pp.addarr("tower_types", std::vector<std::string>{"lat"});
    amrex::ParmParse pt("erf.conductors.lat");
    for (const char* k : {"base_width", "top_width", "solidity", "arm_length", "arm_depth", "peak", "weight", "allowable_uplift",
                          "allowable_compression", "frequency", "damping_ratio"}) { pt.remove(k); }
    pt.add("base_width", 6.0); pt.add("top_width", 1.5); pt.add("solidity", 0.2); pt.add("arm_length", 12.0);
    pt.add("arm_depth", 1.2); pt.add("peak", 8.0); pt.add("weight", 9.0e4);
    if (damping > 0.0) { pt.add("frequency", 2.0); pt.add("damping_ratio", damping); }
    const struct { const char* name; Real tower_y, end_y, z_tower, z_end; bool phase; } lines[] = {
        {"C1", 494.5, 498.5, 30.0, 30.0, true}, {"C2", 500.0, 500.0, 30.0, 30.0, true},
        {"C3", 505.5, 501.5, 30.0, 30.0, true}, {"CS", 500.0, 500.0, 37.0, 32.0, false}};
    for (const auto& l : lines) {
        amrex::ParmParse ps(std::string("erf.conductors.") + l.name);
        for (const char* k : {"end_a", "end_b", "towers", "length", "stringing_tension", "diameter", "mass_per_length", "axial_stiffness",
                              "tower_type", "share_towers", "insulator_length", "insulator_mass", "output_root"}) { ps.remove(k); }
        ps.addarr("end_a", std::vector<Real>{100.0, l.end_y, l.z_end});
        ps.addarr("end_b", std::vector<Real>{1000.0, l.end_y, l.z_end});
        ps.addarr("towers", std::vector<Real>{400.0, l.tower_y, l.z_tower, 700.0, l.tower_y, l.z_tower});
        ps.add("stringing_tension", l.phase ? 20000.0 : 10000.0);
        ps.add("diameter", l.phase ? 0.0281 : 0.0111);
        ps.add("mass_per_length", l.phase ? 1.628 : 0.406);
        ps.add("axial_stiffness", l.phase ? 3.0e7 : 9.7e6);
        if (std::string(l.name) == "C2") { ps.add("tower_type", std::string("lat")); }
        else { ps.add("share_towers", std::string("C2")); }
        if (l.phase) { ps.add("insulator_length", 2.5); ps.add("insulator_mass", 60.0); }
        ps.add("output_root", dir + "/" + l.name);
    }
}
void clear_circuit ()
{
    amrex::ParmParse pp("erf.conductors");
    pp.addarr("lines", std::vector<std::string>{});
    pp.addarr("tower_types", std::vector<std::string>{});
    amrex::ParmParse pt("erf.conductors.lat");
    for (const char* k : {"peak", "frequency", "damping_ratio"}) { pt.remove(k); }
    for (const char* n : {"C1", "C2", "C3", "CS"}) {
        amrex::ParmParse ps(std::string("erf.conductors.") + n);
        for (const char* k : {"tower_type", "share_towers", "stringing_tension", "insulator_length", "insulator_mass"}) { ps.remove(k); }
    }
}
} // namespace

TEST(Conductors, ACircuitHangsFromOneRowOfTowersEachLineAtItsOwnPoint)
{
    const std::string dir = scratch("circuit_towers");
    set_circuit(dir);
    Mesh m(true);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(m.znd.get(), m.geom);
    ASSERT_EQ(c->towers().size(), 2u) << "one row of towers for the four lines";
    for (std::size_t t = 0; t < 2; ++t) {
        const auto& tw = c->towers()[t];
        EXPECT_EQ(tw.name(), "C2_t" + std::to_string(t + 1));
        ASSERT_EQ(c->tower_lines()[t].size(), 4u);
        ASSERT_EQ(tw.attachments().size(), 4u);
        // the owner first, at the centre of the cross-arm, then the others in the order of erf.conductors.lines
        EXPECT_EQ(c->lines()[c->tower_lines()[t][0].first]->name(), "C2");
        EXPECT_EQ(c->lines()[c->tower_lines()[t][1].first]->name(), "C1");
        EXPECT_EQ(c->lines()[c->tower_lines()[t][2].first]->name(), "C3");
        EXPECT_EQ(c->lines()[c->tower_lines()[t][3].first]->name(), "CS");
        // every line's point stands on the tower's base: the ramp rises 0.11 m across the 5.5 m to C1,
        // which the cross-arm does not follow
        for (std::size_t a = 0; a < 4; ++a) {
            const auto [line, j] = c->tower_lines()[t][a];
            const auto& p = c->lines()[line]->inputs().point(j + 1);
            const Real above = (c->lines()[line]->name() == "CS") ? Real(37.0) : Real(30.0);
            EXPECT_NEAR(p[2], tw.base()[2] + above, roundoff * 1000) << c->lines()[line]->name();
            EXPECT_EQ(tw.attachments()[a], p);
        }
        const auto& c1 = c->lines()[0]->inputs().point(static_cast<int>(t) + 1);
        EXPECT_GT(std::abs(c1[2] - (m.h(c1[0], c1[1]) + 30.0)), 0.1) << "the test must see the level cross-arm";
    }
    std::vector<std::array<Real,3>> pull(2, {{0.0, 0.0, 0.0}});
    for (int s = 0; s < 3; ++s) {
        for (std::size_t t = 0; t < 2; ++t) {
            pull[t] = {{0.0, 0.0, 0.0}};
            for (const auto& [line, j] : c->tower_lines()[t]) {
                const auto f = c->lines()[line]->tower_force(j);
                for (int d = 0; d < 3; ++d) { pull[t][d] += f[d]; }
            }
        }
        c->advance(0, 0.25 * s, 0.25, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom);
    }
    for (std::size_t t = 0; t < 2; ++t) {
        const auto& tw = c->towers()[t];
        for (int d = 0; d < 3; ++d) { EXPECT_NEAR(tw.line_force()[d], pull[t][d], roundoff * 1.0e5) << "the four lines' pull, dir " << d; }
        EXPECT_LT(pull[t][2], -3.0 * 4000.0) << "three phases and a shield wire weigh on the tower";
        const auto L = tw.foundation();
        EXPECT_NEAR(L.vertical, 9.0e4 - pull[t][2] - tw.total_force()[2], roundoff * L.vertical);
    }
    clear_circuit();
}

TEST(Conductors, ACircuitOnBendingTowersMovesEveryLinesPoint)
{
    const std::string dir = scratch("circuit_moving");
    set_circuit(dir, 0.3);
    Mesh m(true);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(m.znd.get(), m.geom);
    for (const auto& span : c->lines()) { EXPECT_TRUE(span->towers_move()) << span->name() << " moves with the towers it shares"; }
    const double dt = 0.25;
    int most = 0, unconverged = 0;
    for (int s = 0; s < 40; ++s) {
        c->advance(0, dt * s, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom);
        most = std::max(most, c->coupling_iterations());
        unconverged += c->coupling_unconverged();
    }
    EXPECT_GE(most, 2) << "the coupling steps are iterated while the towers move";
    EXPECT_EQ(unconverged, 0);
    for (std::size_t t = 0; t < 2; ++t) {
        const auto& tw = c->towers()[t];
        const auto* model = dynamic_cast<const erf_towers::OneModeTower*>(c->tower_models()[t].get());
        ASSERT_NE(model, nullptr);
        // at rest the stiffness carries the drag by the shape and every line's pull by the shape where it hangs
        double Q = 0.0;
        for (std::size_t a = 0; a < tw.line_forces().size(); ++a) { Q += model->attachment_shape(a) * tw.line_forces()[a][1]; }
        for (std::size_t i = 0; i < tw.nodes().size(); ++i) { Q += model->mode_shape(i) * tw.loads()[3*i+1]; }
        EXPECT_GT(model->q()[1], 1.0e-3) << tw.name() << " leans with the +y wind";
        EXPECT_NEAR(model->q()[1] * model->stiffness() / Q, 1.0, 0.02) << tw.name();
        // each line's point has moved with the tower by the shape at its height
        const Real ptol = (std::is_same<Real, float>::value) ? Real(1.0e-4) : Real(1.0e-6);
        for (std::size_t a = 0; a < tw.attachments().size(); ++a) {
            const auto [line, j] = c->tower_lines()[t][a];
            const erf_conductors::ConductorLine& span = *c->lines()[line];
            // the point the line hangs from: the top of its string, or the first node of the span after the tower
            const unsigned node = span.num_insulators() > 0
                ? span.span_first_node(span.num_spans()) + static_cast<unsigned>((erf_conductors::LineInputs::insulator_segments + 1) * j)
                : span.span_first_node(j + 1);
            const auto p = span.node_position(node);
            const auto x = model->attachment_displacement(a);
            EXPECT_NEAR(p[1], tw.attachments()[a][1] + x[1], ptol) << span.name() << " on " << tw.name();
        }
    }
    clear_circuit();
}

TEST(Conductors, AShortTautSpanOnBendingTowersStaysStable)
{
    // the shield wire's first span is 40 m long, clamped to the peak and strung tight: its pull
    // changes by ~2.4e5 N/m of the peak's travel, 1.5 times the tower's own stiffness once weighted
    // by the shape there; with the pull lagging a coupling step behind, the exchange pumps energy in
    const std::string dir = scratch("short_span");
    set_circuit(dir, 0.02);
    for (const char* n : {"C1", "C2", "C3", "CS"}) {
        amrex::ParmParse ps(std::string("erf.conductors.") + n);
        std::vector<Real> t;
        ps.getarr("towers", t);
        t[0] = 140.0;
        ps.remove("towers");
        ps.addarr("towers", t);
    }
    Mesh m(true);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(m.znd.get(), m.geom);
    const double dt = 0.25;
    Real peak = 0.0, last = 0.0;
    int unconverged = 0;
    for (int s = 0; s < 60; ++s) {
        c->advance(0, dt * s, dt, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom);
        unconverged += c->coupling_unconverged();
        const auto x = c->towers()[0].arm_displacement();
        last = std::hypot(x[0], x[1]);
        peak = std::max(peak, last);
    }
    EXPECT_EQ(unconverged, 0);
    EXPECT_LT(peak, Real(0.05)) << "the first tower's cross-arm stays within a few centimetres";
    EXPECT_LT(last, peak) << "and settles";
    // about 4 mm, where the stiffness holds the drag and the pulls; without the iteration of each
    // coupling step the exchange pumps energy into the cross-arm and this bound fails
    EXPECT_LT(last, Real(0.006));
    clear_circuit();
}

TEST(Conductors, AnImmersedTerrainPlacesEverythingOnItsSurface)
{
    // the same section and towers on the ramp, once on the fitted mesh and once on a flat mesh with
    // the ramp given as the immersed terrain's surface
    std::vector<std::array<Real,3>> fitted;
    std::vector<Real> fitted_clearance;
    for (int pass = 0; pass < 2; ++pass) {
        const std::string dir = scratch(pass == 0 ? "ib_fitted" : "ib_surface");
        set_circuit(dir);
        Mesh m(pass == 0);
        auto c = Conductors::create(0);
        ASSERT_TRUE(c);
        if (pass == 1) {
            const amrex::Box nodes = amrex::surroundingNodes(m.geom.Domain());
            amrex::FArrayBox h(amrex::makeSlab(amrex::grow(nodes, 3), 2, 0), 1);
            const Real dx = m.Lx / m.nx, dy = m.Ly / m.ny;
            const auto a = h.array();
            amrex::LoopOnCpu(h.box(), [&](int i, int j, int k) { a(i,j,k) = m.h(i * dx, j * dy); });
            c->set_ground_surface(h, m.geom);
        }
        c->set_ground(pass == 0 ? m.znd.get() : nullptr, m.geom);
        std::vector<std::array<Real,3>> placed;
        std::vector<Real> clear;
        for (const auto& span : c->lines()) {
            for (int k = 0; k <= span->num_spans(); ++k) { placed.push_back(span->inputs().point(k)); }
            for (unsigned i = 0; i < span->num_nodes(); ++i) { clear.push_back(span->clearance(i)); }
        }
        for (const auto& tw : c->towers()) { placed.push_back(tw.base()); }
        if (pass == 0) { fitted = placed; fitted_clearance = clear; continue; }
        ASSERT_EQ(placed.size(), fitted.size());
        for (std::size_t p = 0; p < placed.size(); ++p) {
            for (int d = 0; d < 3; ++d) { EXPECT_NEAR(placed[p][d], fitted[p][d], roundoff * 1000) << "point " << p << " dir " << d; }
        }
        ASSERT_EQ(clear.size(), fitted_clearance.size());
        for (std::size_t i = 0; i < clear.size(); ++i) { EXPECT_NEAR(clear[i], fitted_clearance[i], roundoff * 1000) << "node " << i; }
        // the flat mesh's own bottom is z = 0: without the surface the hills would not be seen
        EXPECT_GT(fitted[0][2], 30.0 + 1.0);
    }
    clear_circuit();
}

TEST(Conductors, SetGroundRefusesASurfaceOffsetBelowTheDomainTop)
{
    const std::string dir = scratch("frame");
    set_inputs(dir, "Tframe");
    amrex::ParmParse pp("erf.conductors");
    pp.add("surface_offset", 300.0);   // below the 400 m domain top
    Mesh m(false);
    auto c = Conductors::create(0);
    pp.remove("surface_offset");
    ASSERT_TRUE(c);
    const std::string msg = erf_gtest::abort_message([&] { c->set_ground(nullptr, m.geom); });
    EXPECT_NE(msg.find("erf.conductors.surface_offset"), std::string::npos) << msg;
    EXPECT_NE(msg.find("geometry.prob_hi[2]"), std::string::npos) << msg;
    EXPECT_FALSE(c->ground_set()) << "the check comes before anything is placed";
}

TEST(Conductors, AnAttachmentOutsideTheDomainIsRefusedNamingItsKey)
{
    const std::string dir = scratch("outside");
    // the domain ends at x = 1200 m
    set_inputs(dir, "Tout", true, {{300.0, 500.0, 30.0}}, {{1300.0, 500.0, 30.0}});
    Mesh m(false);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    const std::string msg = erf_gtest::abort_message([&] { c->set_ground(nullptr, m.geom); });
    EXPECT_NE(msg.find("erf.conductors.Tout.end_b at (1300"), std::string::npos) << msg;
    EXPECT_NE(msg.find("lies outside the domain horizontally"), std::string::npos) << msg;
}

TEST(Conductors, ANonFiniteCouplingPullIsRefusedNotConverged)
{
    const std::vector<double> F0{1000.0, 0.0, -2000.0}, F{1000.0, 0.0, -2000.0};
    bool converged = false;
    // the tolerance is 1e-4 of the largest pull at the step's start, 2000 N: 0.2 N
    EXPECT_TRUE(erf_conductors::coupling_converged(F0, F, {1000.1, 0.0, -2000.0}, 1.0e-4, converged).empty());
    EXPECT_TRUE(converged) << "0.1 N within 0.2 N";
    EXPECT_TRUE(erf_conductors::coupling_converged(F0, F, {1001.0, 0.0, -2000.0}, 1.0e-4, converged).empty());
    EXPECT_FALSE(converged) << "1 N beyond 0.2 N";
    const std::string err = erf_conductors::coupling_converged(F0, F, {std::numeric_limits<double>::quiet_NaN(), 0.0, -2000.0},
                                                               1.0e-4, converged);
    EXPECT_NE(err.find("is not finite"), std::string::npos) << err;
    EXPECT_FALSE(converged) << "a pull that is not finite never counts as converged";
}

TEST(Conductors, ARestartChecksTheTowerSwayAndTheSurfaceOffset)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(erf_conductors::restart_mismatch(false, false, 10000.0, 10000.0).empty());
    EXPECT_TRUE(erf_conductors::restart_mismatch(true, true, 10000.0, 10000.0).empty());
    EXPECT_TRUE(erf_conductors::restart_mismatch(false, false, nan, 5000.0).empty()) << "a checkpoint without the record";
    EXPECT_NE(erf_conductors::restart_mismatch(false, true, nan, 10000.0).find("holds no tower sway"), std::string::npos);
    EXPECT_NE(erf_conductors::restart_mismatch(true, false, nan, 10000.0).find("holds tower sway, but no tower type"), std::string::npos);
    EXPECT_NE(erf_conductors::restart_mismatch(false, false, 10000.0, 5000.0).find("erf.conductors.surface_offset"), std::string::npos);
}

namespace {
// rho and rho k = rho (k0 + kz z) in every cell of a flat mesh: k linear in the height, which the cell sampler reproduces
void fill_k (amrex::MultiFab& cons, const Mesh& m, Real k0, Real kz)
{
    cons.setVal(0.0);
    const Real dz = m.H / m.nz;
    for (amrex::MFIter mfi(cons); mfi.isValid(); ++mfi) {
        auto ca = cons.array(mfi);
        amrex::LoopOnCpu(mfi.growntilebox(), [&](int i, int j, int k) {
            amrex::ignore_unused(i, j);
            ca(i,j,k,Rho_comp) = 1.2;
            ca(i,j,k,RhoKE_comp) = Real(1.2) * (k0 + kz * (k + Real(0.5)) * dz);
        });
    }
}

// the mean of k = k0 + kz z over the nodes [n0, n0 + nn) of the points pos (3 per point)
double mean_k (const std::vector<Real>& pos, unsigned n0, unsigned nn, Real k0, Real kz)
{
    double k = 0.0;
    for (unsigned n = n0; n < n0 + nn; ++n) { k += static_cast<double>(k0 + kz * pos[3*n+2]); }
    return k / nn;
}

// the rows of a whitespace-separated log after its header, as numbers
std::vector<std::vector<double>> read_rows (const std::string& fname, std::string& header)
{
    std::ifstream f(fname);
    std::getline(f, header);
    std::vector<std::vector<double>> rows;
    for (std::string line; std::getline(f, line);) {
        std::istringstream ls(line);
        rows.emplace_back();
        for (double x; ls >> x;) { rows.back().push_back(x); }
    }
    return rows;
}
} // namespace

TEST(Conductors, AnEventGustPassesEveryNodeAndTowerWithItsFront)
{
    const std::string dir = scratch("gust_event");
    set_towered_section(dir, "Tev");   // three spans along x from 100 to 1000 m, towers at 400 and 700 m
    amrex::ParmParse pp("erf.conductors");
    pp.remove("prescribed_velocity");
    pp.add("gust_type", std::string("event"));
    pp.add("gust_event_time", 1.0);
    pp.add("gust_event_speed", 150.0);
    pp.add("gust_event_duration", 2.0);
    Mesh m(false);
    m.fill_crosswind();
    const Real k0 = 0.5, kz = 0.01;
    amrex::MultiFab cons(m.ba, m.dm, RhoKE_comp + 1, 1);
    fill_k(cons, m, k0, kz);
    {
        // without an origin the front crosses the domain's centre, moving along +x
        auto c0 = Conductors::create(0);
        ASSERT_TRUE(c0);
        c0->set_closure(true, 0.5562);
        c0->set_ground(nullptr, m.geom);
        EXPECT_NEAR(c0->gust_event().x0, 600.0, 1e-9);
        EXPECT_NEAR(c0->gust_event().y0, 500.0, 1e-9);
        EXPECT_NEAR(c0->gust_event().ex, 1.0, 1e-15);
        EXPECT_NEAR(c0->gust_event().ey, 0.0, 1e-15);
    }
    {
        // a direction in degrees counterclockwise from +x: 90 moves the front along +y
        pp.add("gust_event_direction", 90.0);
        auto c90 = Conductors::create(0);
        ASSERT_TRUE(c90);
        c90->set_closure(true, 0.5562);
        c90->set_ground(nullptr, m.geom);
        EXPECT_NEAR(c90->gust_event().ex, 0.0, 1e-6);
        EXPECT_NEAR(c90->gust_event().ey, 1.0, 1e-12);
        pp.add("gust_event_direction", 0.0);
    }
    // the front crosses the line's first end at t = 1 s and its last at 7 s; it lasts 2 s at a point
    pp.addarr("gust_event_origin", std::vector<Real>{100.0, 500.0});
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_closure(true, 0.5562);
    c->set_ground(nullptr, m.geom);
    EXPECT_NE(erf_gtest::abort_message([&] { c->advance(0, 0.0, 0.25, m.u, m.v, m.w, nullptr, nullptr, m.geom); })
                  .find("gust_type = event needs the conserved state"), std::string::npos);
    const auto& e = c->gust_event();
    EXPECT_NEAR(e.x0, 100.0, 1e-9);
    EXPECT_EQ(e.time, 1.0);
    EXPECT_EQ(e.speed, 150.0);
    EXPECT_EQ(e.duration, 2.0);
    const double sf = c->gust_sigma_factor(), g = 2.7, dt = 0.25;
    const auto& line = *c->lines().front();
    ASSERT_EQ(c->towers().size(), 2u);
    int during = 0, before = 0;
    std::vector<double> mid_gust;   // the gust at span 1's middle node, step by step
    const int steps = 40;
    for (int s = 0; s < steps; ++s) {
        const double tm = dt * s + 0.5 * dt;   // the wind is held at the step's middle
        const std::vector<Real> where = line.kinematics_points();
        c->advance(0, dt * s, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons);
        for (int sp = 0; sp < line.num_spans(); ++sp) {
            const unsigned n0 = line.span_first_node(sp), nn = line.span_num_nodes(sp);
            const double A = g * sf * std::sqrt(mean_k(where, n0, nn, k0, kz));
            for (unsigned n = n0; n < n0 + nn; ++n) {
                const double phase = erf_conductors::gust_event_phase(e, where[3*n], where[3*n+1], tm);
                const double gust = A * erf_conductors::gust_event_shape(phase);
                const double expect = static_cast<double>(m.v_at(where[3*n+1], where[3*n+2])) + gust;
                const auto uvw = line.wind_at_point(n);
                ASSERT_NEAR(uvw[1], expect, roundoff * std::max(1.0, expect)) << "step " << s << " node " << n;
                ASSERT_NEAR(uvw[0], 0.0, roundoff) << "along the +y wind only";
                if (gust > 0.5) { ++during; } else if (phase < 0.0) { ++before; }
                if (sp == 0 && n == n0 + (nn - 1) / 2) { mid_gust.push_back(gust); }
            }
        }
        // the towers stand still: their drag nodes where they were built
        for (std::size_t t = 0; t < 2; ++t) {
            const auto& nodes = c->towers()[t].nodes();
            double kk = 0.0;
            for (const auto& nd : nodes) { kk += static_cast<double>(k0 + kz * nd.pos[2]); }
            const double A = g * sf * std::sqrt(kk / nodes.size());
            const auto& w = c->tower_winds()[t];
            ASSERT_EQ(w.size(), 3 * nodes.size());
            for (std::size_t j = 0; j < nodes.size(); ++j) {
                const double phase = erf_conductors::gust_event_phase(e, nodes[j].pos[0], nodes[j].pos[1], tm);
                const double gust = A * erf_conductors::gust_event_shape(phase);
                const double expect = static_cast<double>(m.v_at(nodes[j].pos[1], nodes[j].pos[2])) + gust;
                ASSERT_NEAR(w[3*j+1], expect, roundoff * std::max(1.0, expect)) << "tower " << t << " node " << j << " step " << s;
            }
        }
    }
    EXPECT_GT(during, 100) << "the gust must blow on the line for a while";
    EXPECT_GT(before, 100) << "and not before its front";
    // gust_series.dat: a row per step, the gust at each span's middle node and at each tower's highest drag node
    std::string header;
    const auto rows = read_rows(dir + "/gust_series.dat", header);
    EXPECT_EQ(header, "time Tev_span1 Tev_span2 Tev_span3 Tev_t1 Tev_t2");
    ASSERT_EQ(rows.size(), static_cast<std::size_t>(steps));
    for (int s = 0; s < steps; ++s) {
        const auto ss = static_cast<std::size_t>(s);
        ASSERT_EQ(rows[ss].size(), 6u);
        EXPECT_NEAR(rows[ss][0], dt * s, 1e-12) << "stamped with the step's start";
        EXPECT_NEAR(rows[ss][1], mid_gust[ss], 1e-6 * std::max(1.0, mid_gust[ss]) + roundoff) << "step " << s;
    }
    clear_towered_section("Tev");
}

TEST(Conductors, RandomGustsFollowTheirProcessesAndACircuitSharesThem)
{
    const std::string dir = scratch("gust_random");
    set_circuit(dir);   // C2 owns the towers; C1, C3 and the shield wire CS share them
    amrex::ParmParse pp("erf.conductors");
    pp.remove("prescribed_velocity");
    pp.add("gust_type", std::string("random"));
    pp.add("gust_seed", 11);
    Mesh m(false);
    m.fill_crosswind();
    const Real k0 = 0.5, kz = 0.01;
    amrex::MultiFab cons(m.ba, m.dm, RhoKE_comp + 1, 1);
    fill_k(cons, m, k0, kz);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_closure(true, 0.5562);
    c->set_ground(nullptr, m.geom);
    ASSERT_EQ(c->lines().size(), 4u);
    ASSERT_EQ(c->towers().size(), 2u);
    EXPECT_NE(erf_gtest::abort_message([&] { c->advance(0, 0.0, 0.5, m.u, m.v, m.w, nullptr, nullptr, m.geom); })
                  .find("gust_type = random needs the conserved state"), std::string::npos);
    const double sf = c->gust_sigma_factor(), Ls = 67.056, dt = 0.5;
    // the processes: C2's three spans (0 to 2), whose time scales come from C2's nodes, then the two towers (3, 4)
    std::vector<double> z(5, 0.0);
    const auto& C2 = *c->lines()[1];
    double zmin = 0.0, zmax = 0.0;
    const int steps = 12;
    for (int s = 0; s < steps; ++s) {
        const auto step = static_cast<std::uint64_t>(s + 1);   // the step count the draws take, from 1
        std::vector<std::vector<Real>> where(4);
        for (std::size_t i = 0; i < 4; ++i) { where[i] = c->lines()[i]->kinematics_points(); }
        auto draw = [&] (std::size_t p, double height, double U) {
            const double T = erf_conductors::gust_integral_length(height) / U;
            const double xi = erf_conductors::gust_normal(11, p, step);
            z[p] = (s == 0) ? xi : erf_conductors::gust_ou_step(z[p], dt, T, xi);
        };
        for (int sp = 0; sp < 3; ++sp) {
            const unsigned n0 = C2.span_first_node(sp), nn = C2.span_num_nodes(sp);
            double u2 = 0.0;
            for (unsigned n = n0; n < n0 + nn; ++n) {
                const double v = static_cast<double>(m.v_at(where[1][3*n+1], where[1][3*n+2]));
                u2 += v * v;
            }
            // the span's height: the mean of the heights it hangs from (the strings' lower ends at the towers), the ground flat at 0
            const double height = 0.5 * static_cast<double>(C2.inputs().conductor_point(sp)[2] + C2.inputs().conductor_point(sp + 1)[2]);
            draw(static_cast<std::size_t>(sp), height, std::sqrt(u2 / nn));
        }
        for (std::size_t t = 0; t < 2; ++t) {
            const auto& tw = c->towers()[t];
            double u2 = 0.0, zm = 0.0;
            for (const auto& nd : tw.nodes()) {
                const double v = static_cast<double>(m.v_at(nd.pos[1], nd.pos[2]));
                u2 += v * v;
                zm += static_cast<double>(nd.pos[2] - tw.base()[2]);
            }
            draw(3 + t, zm / tw.nodes().size(), std::sqrt(u2 / tw.nodes().size()));
        }
        c->advance(0, dt * s, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons);
        for (std::size_t i = 0; i < 4; ++i) {
            const auto& L = *c->lines()[i];
            std::vector<double> gust(3);
            for (int sp = 0; sp < 3; ++sp) {
                const unsigned n0 = L.span_first_node(sp), nn = L.span_num_nodes(sp);
                const double sigma = sf * std::sqrt(mean_k(where[i], n0, nn, k0, kz));
                const double B = erf_conductors::gust_background_factor(static_cast<double>(L.inputs().chord(sp)), Ls);
                // every line's span takes C2's process of that span: the circuit's phases move together
                gust[static_cast<std::size_t>(sp)] = sigma * std::sqrt(B) * z[static_cast<std::size_t>(sp)];
                for (unsigned n = n0; n < n0 + nn; ++n) {
                    const double expect = static_cast<double>(m.v_at(where[i][3*n+1], where[i][3*n+2])) +
                                          gust[static_cast<std::size_t>(sp)];
                    ASSERT_NEAR(L.wind_at_point(n)[1], expect, roundoff * std::max(1.0, expect))
                        << L.name() << " step " << s << " node " << n;
                }
            }
            // a string takes the mean of the spans beside it
            for (int j = 0; j < L.num_insulators(); ++j) {
                const double A = 0.5 * (gust[static_cast<std::size_t>(j)] + gust[static_cast<std::size_t>(j) + 1]);
                for (unsigned n = L.string_first_node(j); n < L.string_first_node(j) + L.string_num_nodes(j); ++n) {
                    const double expect = static_cast<double>(m.v_at(where[i][3*n+1], where[i][3*n+2])) + A;
                    ASSERT_NEAR(L.wind_at_point(n)[1], expect, roundoff * std::max(1.0, expect))
                        << L.name() << " string " << j << " node " << n;
                }
            }
        }
        for (std::size_t t = 0; t < 2; ++t) {
            const auto& tw = c->towers()[t];
            double kk = 0.0, top = 0.0;
            for (const auto& nd : tw.nodes()) {
                kk += static_cast<double>(k0 + kz * nd.pos[2]);
                top = std::max(top, static_cast<double>(nd.pos[2] - tw.base()[2]));
            }
            const double gust = sf * std::sqrt(kk / tw.nodes().size()) *
                                std::sqrt(erf_conductors::tower_background_factor(top, Ls)) * z[3 + t];
            for (std::size_t j = 0; j < tw.nodes().size(); ++j) {
                const double expect = static_cast<double>(m.v_at(tw.nodes()[j].pos[1], tw.nodes()[j].pos[2])) + gust;
                ASSERT_NEAR(c->tower_winds()[t][3*j+1], expect, roundoff * std::max(1.0, expect)) << "tower " << t << " node " << j;
            }
        }
        for (const double v : z) { zmin = std::min(zmin, v); zmax = std::max(zmax, v); }
    }
    EXPECT_LT(zmin, -0.3) << "the processes must swing both ways for the test to see the sign";
    EXPECT_GT(zmax, 0.3);
    std::string header;
    const auto rows = read_rows(dir + "/gust_series.dat", header);
    EXPECT_EQ(header, "time C1_span1 C1_span2 C1_span3 C2_span1 C2_span2 C2_span3 C3_span1 C3_span2 C3_span3 CS_span1 CS_span2 "
                      "CS_span3 C2_t1 C2_t2");
    ASSERT_EQ(rows.size(), static_cast<std::size_t>(steps));
    for (const auto& r : rows) { EXPECT_EQ(r.size(), 15u); }
    clear_circuit();
}

TEST(Conductors, RandomGustsContinueAcrossARestart)
{
    Mesh m(false);
    m.fill_crosswind();
    const Real k0 = 0.5, kz = 0.01;
    amrex::MultiFab cons(m.ba, m.dm, RhoKE_comp + 1, 1);
    fill_k(cons, m, k0, kz);
    const double dt = 0.25;
    auto setup = [] (const std::string& dir, const char* type) {
        set_inputs(dir, "Trg", false);
        amrex::ParmParse pp("erf.conductors");
        pp.add("gust_type", std::string(type));
        pp.remove("gust_seed");
        if (std::string(type) == "random") { pp.add("gust_seed", 3); }
    };
    auto winds = [] (const Conductors& c) {
        const auto& line = *c.lines().front();
        std::vector<Real> w;
        for (unsigned n = 0; n < line.num_nodes(); ++n) { const auto u = line.wind_at_point(n); w.insert(w.end(), {u[0], u[1], u[2]}); }
        return w;
    };
    // six steps straight through
    const std::string dir_a = scratch("gust_restart_a");
    setup(dir_a, "random");
    auto a = Conductors::create(0);
    ASSERT_TRUE(a);
    a->set_closure(true, 0.5562);
    a->set_ground(nullptr, m.geom);
    for (int s = 0; s < 6; ++s) { a->advance(0, dt * s, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons); }
    const std::vector<Real> wind_a = winds(*a);
    // three steps, a checkpoint, and a restart for the other three
    const std::string dir_b = scratch("gust_restart_b");
    setup(dir_b, "random");
    auto b = Conductors::create(0);
    ASSERT_TRUE(b);
    b->set_closure(true, 0.5562);
    b->set_ground(nullptr, m.geom);
    for (int s = 0; s < 3; ++s) { b->advance(0, dt * s, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons); }
    const std::string chk = dir_b + "/chk00003";
    std::filesystem::create_directories(chk);
    b->write_checkpoint(chk);
    EXPECT_NE(slurp(chk + "/conductors/gust_processes").find("processes 1 set 1"), std::string::npos);
    auto b2 = Conductors::create(0);
    ASSERT_TRUE(b2);
    b2->set_closure(true, 0.5562);
    b2->set_ground(nullptr, m.geom, chk);
    ASSERT_TRUE(b2->restored());
    for (int s = 3; s < 6; ++s) { b2->advance(0, dt * s, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons); }
    const std::vector<Real> wind_b = winds(*b2);
    ASSERT_EQ(wind_a.size(), wind_b.size());
    for (std::size_t i = 0; i < wind_a.size(); ++i) {
        EXPECT_NEAR(wind_b[i], wind_a[i], roundoff * std::max(Real(1.0), std::abs(wind_a[i]))) << "component " << i;
    }
    std::string ha, hb;
    const auto rows_a = read_rows(dir_a + "/gust_series.dat", ha);
    const auto rows_b = read_rows(dir_b + "/gust_series.dat", hb);
    EXPECT_EQ(ha, hb);
    ASSERT_EQ(rows_b.size(), rows_a.size()) << "the restarted run continues the series from the checkpoint";
    for (std::size_t r = 0; r < rows_a.size(); ++r) {
        ASSERT_EQ(rows_b[r].size(), 2u);
        EXPECT_NEAR(rows_b[r][1], rows_a[r][1], 1e-6) << "row " << r;
    }
    EXPECT_GT(std::abs(rows_a.back()[1]), 1.0e-3) << "a gust to compare";
    // a checkpoint written without gusts: the restart draws the process afresh, from its stationary law, at its first step
    const std::string dir_c = scratch("gust_restart_c");
    setup(dir_c, "none");
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(nullptr, m.geom);
    for (int s = 0; s < 3; ++s) { c->advance(0, dt * s, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom); }
    const std::string chk_c = dir_c + "/chk00003";
    std::filesystem::create_directories(chk_c);
    c->write_checkpoint(chk_c);
    EXPECT_FALSE(std::filesystem::exists(chk_c + "/conductors/gust_processes"));
    {
        amrex::ParmParse pp("erf.conductors");
        pp.add("gust_type", std::string("random"));
        pp.add("gust_seed", 3);
    }
    auto c2 = Conductors::create(0);
    ASSERT_TRUE(c2);
    c2->set_closure(true, 0.5562);
    c2->set_ground(nullptr, m.geom, chk_c);
    ASSERT_TRUE(c2->restored());
    const auto& line = *c2->lines().front();
    const std::vector<Real> where = line.kinematics_points();
    c2->advance(0, dt * 3, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons);
    const unsigned nn = line.span_num_nodes(0);
    const double gust = c2->gust_sigma_factor() * std::sqrt(mean_k(where, 0, nn, k0, kz)) *
                        std::sqrt(erf_conductors::gust_background_factor(static_cast<double>(line.inputs().chord(0)), 67.056)) *
                        erf_conductors::gust_normal(3, 0, 4);
    for (unsigned n = 0; n < nn; ++n) {
        const double expect = static_cast<double>(m.v_at(where[3*n+1], where[3*n+2])) + gust;
        ASSERT_NEAR(line.wind_at_point(n)[1], expect, roundoff * std::max(1.0, expect)) << "node " << n;
    }
}

TEST(Conductors, TheGustFactorRunsOnLinesWithLatticeTowers)
{
    // gust_type = factor samples k at the line nodes only; the towers' drag nodes are not sampled
    const std::string dir = scratch("gust_factor_towers");
    set_towered_section(dir, "Tft");
    amrex::ParmParse pp("erf.conductors");
    pp.remove("prescribed_velocity");
    pp.add("gust_type", std::string("factor"));
    Mesh m(false);
    m.fill_crosswind();
    amrex::MultiFab cons(m.ba, m.dm, RhoKE_comp + 1, 1);
    fill_k(cons, m, 0.5, 0.01);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_closure(true, 0.5562);
    c->set_ground(nullptr, m.geom);
    ASSERT_EQ(c->towers().size(), 2u);
    for (int s = 0; s < 3; ++s) { c->advance(0, 0.25 * s, 0.25, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons); }
    std::ifstream f(dir + "/gusts.csv");
    std::string row;
    int rows = 0;
    while (std::getline(f, row)) { ++rows; }
    EXPECT_EQ(rows, 4) << "the header and one row per span";
    EXPECT_FALSE(std::filesystem::exists(dir + "/gust_series.dat")) << "the factor adds no gust to the wind";
    clear_towered_section("Tft");
}

TEST(Conductors, ALineTakesAnotherLinesGustsAndAChangedSharingDrawsAfresh)
{
    // three parallel lines 6 m apart without lattice towers: Tgb takes Tga's random gusts (gust_with), Tgc its own
    const std::string dir = scratch("gust_with");
    set_inputs(dir, "Tga", false);
    amrex::ParmParse pp("erf.conductors");
    pp.addarr("lines", std::vector<std::string>{"Tga", "Tgb", "Tgc"});
    for (const char* name : {"Tgb", "Tgc"}) {
        amrex::ParmParse ps(std::string("erf.conductors.") + name);
        const Real y = (std::string(name) == "Tgb") ? Real(506.0) : Real(512.0);
        ps.addarr("end_a", std::vector<Real>{300.0, y, 30.0});
        ps.addarr("end_b", std::vector<Real>{600.0, y, 30.0});
        ps.add("length", 301.5);
        ps.add("diameter", 0.0281);
        ps.add("mass_per_length", 1.628);
        ps.add("axial_stiffness", 3.0e7);
    }
    amrex::ParmParse("erf.conductors.Tgb").add("gust_with", std::string("Tga"));
    pp.add("gust_type", std::string("random"));
    pp.add("gust_seed", 5);
    Mesh m(false);
    m.fill_crosswind();
    // a wind along x that grows with the height as well: the wind veers with the height, so a node's own direction
    // differs from its span's mean direction
    auto u_at = [] (Real z) { return Real(2.0) + Real(0.1) * z; };
    {
        const Real dz = m.H / m.nz;
        for (amrex::MFIter mfi(m.u); mfi.isValid(); ++mfi) {
            auto ua = m.u.array(mfi);
            amrex::LoopOnCpu(mfi.growntilebox(), [&](int i, int j, int k) {
                amrex::ignore_unused(i, j);
                ua(i,j,k) = u_at((k + Real(0.5)) * dz);
            });
        }
    }
    const Real k0 = 0.5, kz = 0.01;
    amrex::MultiFab cons(m.ba, m.dm, RhoKE_comp + 1, 1);
    fill_k(cons, m, k0, kz);
    const double dt = 0.25, Ls = 67.056;
    // the expected wind at a line's nodes, the flow's plus gust along the span's mean horizontal wind
    auto check = [&] (const Conductors& c, std::size_t i, const std::vector<Real>& where, double z, const std::string& what) {
        const auto& L = *c.lines()[i];
        const unsigned nn = L.span_num_nodes(0);
        double mu = 0.0, mv = 0.0;
        for (unsigned n = 0; n < nn; ++n) {
            mu += static_cast<double>(u_at(where[3*n+2]));
            mv += static_cast<double>(m.v_at(where[3*n+1], where[3*n+2]));
        }
        const double h = std::hypot(mu, mv);
        const double gust = c.gust_sigma_factor() * std::sqrt(mean_k(where, 0, nn, k0, kz)) *
                            std::sqrt(erf_conductors::gust_background_factor(static_cast<double>(L.inputs().chord(0)), Ls)) * z;
        for (unsigned n = 0; n < nn; ++n) {
            const double eu = static_cast<double>(u_at(where[3*n+2])) + gust * mu / h;
            const double ev = static_cast<double>(m.v_at(where[3*n+1], where[3*n+2])) + gust * mv / h;
            ASSERT_NEAR(L.wind_at_point(n)[0], eu, roundoff * std::max(1.0, std::abs(eu))) << what << " node " << n;
            ASSERT_NEAR(L.wind_at_point(n)[1], ev, roundoff * std::max(1.0, std::abs(ev))) << what << " node " << n;
        }
    };
    auto a = Conductors::create(0);
    ASSERT_TRUE(a);
    a->set_closure(true, 0.5562);
    a->set_ground(nullptr, m.geom);
    double z = 0.0;
    for (int s = 0; s < 3; ++s) {
        std::vector<std::vector<Real>> where{a->lines()[0]->kinematics_points(), a->lines()[1]->kinematics_points()};
        // one process, Tga's span, its time scale from Tga's nodes
        const unsigned nn = a->lines()[0]->span_num_nodes(0);
        double u2 = 0.0;
        for (unsigned n = 0; n < nn; ++n) {
            const double u = static_cast<double>(u_at(where[0][3*n+2])), v = static_cast<double>(m.v_at(where[0][3*n+1], where[0][3*n+2]));
            u2 += u * u + v * v;
        }
        const double T = erf_conductors::gust_integral_length(30.0) / std::sqrt(u2 / nn);
        const double xi = erf_conductors::gust_normal(5, 0, static_cast<std::uint64_t>(s + 1));
        z = (s == 0) ? xi : erf_conductors::gust_ou_step(z, dt, T, xi);
        a->advance(0, dt * s, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons);
        check(*a, 0, where[0], z, "Tga step " + std::to_string(s));
        check(*a, 1, where[1], z, "Tgb step " + std::to_string(s));
    }
    const std::string chk = dir + "/chk00003";
    std::filesystem::create_directories(chk);
    a->write_checkpoint(chk);
    EXPECT_NE(slurp(chk + "/conductors/gust_processes").find("Tgc_span1 "), std::string::npos);
    // the restart has Tgc take Tga's gusts instead of Tgb: two processes again, but not the same ones (Tga's and
    // Tgb's where the checkpoint holds Tga's and Tgc's), so both are drawn afresh, from the stationary law, at the
    // restart's first step
    amrex::ParmParse("erf.conductors.Tgb").remove("gust_with");
    amrex::ParmParse("erf.conductors.Tgc").add("gust_with", std::string("Tga"));
    auto b = Conductors::create(0);
    ASSERT_TRUE(b);
    b->set_closure(true, 0.5562);
    b->set_ground(nullptr, m.geom, chk);
    ASSERT_TRUE(b->restored());
    std::vector<std::vector<Real>> where{b->lines()[0]->kinematics_points(), b->lines()[1]->kinematics_points(),
                                         b->lines()[2]->kinematics_points()};
    b->advance(0, dt * 3, dt, m.u, m.v, m.w, nullptr, nullptr, m.geom, &cons);
    check(*b, 0, where[0], erf_conductors::gust_normal(5, 0, 4), "Tga after the restart");
    check(*b, 1, where[1], erf_conductors::gust_normal(5, 1, 4), "Tgb after the restart");
    check(*b, 2, where[2], erf_conductors::gust_normal(5, 0, 4), "Tgc after the restart, with Tga's gusts");
}
