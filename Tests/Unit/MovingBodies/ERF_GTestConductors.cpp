// Contract of Conductors, the manager ERF holds: the attachments are placed at their height above
// the terrain surface under each end (the k = 0 node plane of z_phys_nd, bilinear between the
// nodes) and ground.dat records it; on a uniform-dz mesh the given heights are absolute; the
// spans step only on the anchor level and write one diagnostics row per step; and without a
// prescribed velocity the wind handed to MoorDyn is ERF's velocity sampled where the line is
// now, not where it hung; a restart from a checkpoint continues the lines, their drag on the
// air, their statistics and their logs exactly where the checkpoint left them; a section is placed
// on the terrain at every tower and logs each span and its strings; and the closest approach of two
// lines is the exact distance between their conductors, flagged against the flashover distance;
// and a transformer stands on the terrain, takes the pull of the lines ending on it with their
// moment about its base, flags them against its allowable values, watches how close every
// conductor comes to its box, and continues its log and statistics across a restart; and a line's
// suspension towers of a tower type stand on the terrain with their cross-arms across the line,
// carry the members' drag of the wind at their nodes, put it into the flow with the lines', and
// continue their log, statistics and drag on the flow across a restart.

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
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
#include "ERF_MoorDynSystem.H"

namespace {

using amrex::Real;

// the roundoff of the sampling, the terrain read and the source sums: about 1e-7 relative in single precision
constexpr Real roundoff = std::is_same<Real, float>::value ? Real(1.0e-5) : Real(1.0e-9);

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
    pp.add("spans", name);
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
    ASSERT_EQ(c->spans().size(), 1u);
    const auto& span = *c->spans().front();
    const unsigned last = span.num_nodes() - 1;
    const auto a = span.node_position(0);
    const auto b = span.node_position(last);
    // the ramp rises 15 m from x = 300 to x = 600: the two ends sit at different absolute heights
    EXPECT_NEAR(a[2], m.h(300.0, 500.0) + 30.0, 1.0e-6);
    EXPECT_NEAR(b[2], m.h(600.0, 500.0) + 30.0, 1.0e-6);
    EXPECT_NEAR(a[0], 300.0, 1.0e-6);
    EXPECT_NEAR(b[0], 600.0, 1.0e-6);
    EXPECT_GT(b[2] - a[2], 10.0);
    // ground.dat records the surface and the absolute height of each end
    std::ifstream g(dir + "/ground.dat");
    ASSERT_TRUE(g.good());
    std::string header, name, end;
    Real x, y, ground, z;
    std::getline(g, header);
    ASSERT_TRUE(static_cast<bool>(g >> name >> end >> x >> y >> ground >> z));
    EXPECT_EQ(name, "Tterrain"); EXPECT_EQ(end, "a");
    EXPECT_NEAR(ground, m.h(300.0, 500.0), 1.0e-6);
    EXPECT_NEAR(z, m.h(300.0, 500.0) + 30.0, 1.0e-6);
    ASSERT_TRUE(static_cast<bool>(g >> name >> end >> x >> y >> ground >> z));
    EXPECT_EQ(end, "b");
    EXPECT_NEAR(ground, m.h(600.0, 500.0), 1.0e-6);
}

TEST(Conductors, OnAUniformMeshTheHeightsAreAbsolute)
{
    const std::string dir = scratch("flat");
    set_inputs(dir, "Tflat");
    Mesh m(false);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(nullptr, m.geom);
    const auto& span = *c->spans().front();
    EXPECT_NEAR(span.node_position(0)[2], 30.0, 1.0e-6);
    EXPECT_NEAR(span.node_position(span.num_nodes() - 1)[2], 30.0, 1.0e-6);
}

TEST(Conductors, SpansStepOnTheAnchorLevelOnlyAndLogEveryStep)
{
    const std::string dir = scratch("advance");
    set_inputs(dir, "Tadv");
    Mesh m(false);
    auto c = Conductors::create(1);   // two levels: the anchor is the finest, level 1
    ASSERT_TRUE(c);
    EXPECT_EQ(c->anchor_level(), 1);
    c->set_ground(nullptr, m.geom);
    const auto& span = *c->spans().front();
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

TEST(Conductors, TheFlowIsSampledWhereTheLineIsNow)
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
    const auto& span = *c->spans().front();
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
    const auto& span = *c->spans().front();
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
    const auto& span = *c->spans().front();
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
    EXPECT_GT(c->spans().front()->total_drag()[1], 1.0) << "the line still feels the drag";
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
    amrex::MultiFab src_at_chk(m.ba, m.dm, 3, 0);
    a->cell_sources(0, src_at_chk, 0);
    ASSERT_GT(src_at_chk.norm0(1), 0.0) << "the lines must push on the air";
    const std::vector<Real> pos_at_chk = a->spans().front()->node_positions();
    for (; step < 7; ++step) { a->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), &detJ, m.geom); }
    const std::vector<Real> pos_end = a->spans().front()->node_positions();
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
    const auto& span_b = *b->spans().front();
    const std::vector<Real> pos_b = span_b.node_positions();
    ASSERT_EQ(pos_b.size(), pos_at_chk.size());
    for (std::size_t i = 0; i < pos_b.size(); ++i) {
        ASSERT_NEAR(pos_b[i], pos_at_chk[i], 1.0e-9 * std::max(Real(1.0), std::abs(pos_at_chk[i]))) << "component " << i;
    }
    EXPECT_GT(span_b.mid_offset(), 0.1) << "restored blown out, not hanging still";
    // the logs end at the checkpoint until the restarted run writes again
    EXPECT_LT(slurp(dir + "/Trst.dat").size(), log_a.size());
    // the momentum source is that of the step the checkpoint was written at
    b->restore_sources(0, m.u, m.znd.get(), &detJ, m.geom);
    amrex::MultiFab src_b(m.ba, m.dm, 3, 0);
    b->cell_sources(0, src_b, 0);
    amrex::MultiFab::Subtract(src_b, src_at_chk, 0, 0, 3, 0);
    EXPECT_LE(src_b.norm0(1), 1.0e-12 * src_at_chk.norm0(1));
    // the same three steps: the lines, the logs and the statistics come out as in the run without the restart
    for (step = 4; step < 7; ++step) { b->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), &detJ, m.geom); }
    const std::vector<Real> pos_b_end = span_b.node_positions();
    for (std::size_t i = 0; i < pos_end.size(); ++i) {
        EXPECT_NEAR(pos_b_end[i], pos_end[i], 1.0e-9 * std::max(Real(1.0), std::abs(pos_end[i]))) << "component " << i;
    }
    EXPECT_EQ(slurp(dir + "/Trst.dat"), log_a);
    EXPECT_EQ(slurp(dir + "/Trst_nodes.dat"), nodes_a);
    EXPECT_EQ(slurp(dir + "/Trst_stats.csv"), stats_a);
    EXPECT_EQ(slurp(dir + "/total_load.dat"), load_a);

    // a checkpoint without conductor state (a precursor's, say): the spans start afresh, from
    // still air, at the restart time; MoorDyn's clock starts at zero there
    const std::string bare = dir + "/chk_bare";
    std::filesystem::create_directories(bare);
    auto c = Conductors::create(0);
    ASSERT_TRUE(c);
    c->set_ground(m.znd.get(), m.geom, bare);
    EXPECT_FALSE(c->restored());
    EXPECT_NEAR(c->spans().front()->mid_offset(), 0.0, 1.0e-3);
    for (step = 7; step < 9; ++step) { c->advance(0, dt * step, dt, m.u, m.v, m.w, m.znd.get(), &detJ, m.geom); }
    EXPECT_GT(c->spans().front()->mid_offset(), 0.01) << "the span started at the restart time moves";
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
    const std::vector<Real> pc = c->spans().front()->node_positions();
    const std::vector<Real> pd = d->spans().front()->node_positions();
    for (std::size_t i = 0; i < pc.size(); ++i) {
        EXPECT_NEAR(pd[i], pc[i], 1.0e-9 * std::max(Real(1.0), std::abs(pc[i]))) << "component " << i;
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
    const auto& line = *c->spans().front();
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
        EXPECT_NEAR(ground, m.h(x, y), 1.0e-6) << label;
        EXPECT_NEAR(z, m.h(x, y) + 30.0, 1.0e-6) << label;
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
    pp.addarr("spans", std::vector<std::string>{"Pa", "Pb"});
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
        // the distance is the exact minimum between the two conductors where they are now
        const auto exact = erf_conductors::closest_polylines(c->spans()[0]->conductor_path(), c->spans()[1]->conductor_path());
        ASSERT_NEAR(c->separations()[0].distance, exact.distance, 1.0e-12 * exact.distance) << "step " << s;
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
    pp.addarr("spans", std::vector<std::string>{});
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
    pp.addarr("spans", std::vector<std::string>{"Rs", "Rp"});
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
    EXPECT_NEAR(b->separations()[0].distance, sep, 1.0e-9 * sep);
    EXPECT_NEAR(b->spans()[0]->insulator_swing(0), a->spans()[0]->insulator_swing(0), roundoff);
    pp.addarr("spans", std::vector<std::string>{});
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
    pp.addarr("spans", std::vector<std::string>{"Lt", "Lo"});
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
            EXPECT_NEAR(ground, m.h(x, y), 1.0e-6) << name;
            EXPECT_NEAR(top, m.h(x, y) + 6.0, 1.0e-6) << name;
        }
    }
    EXPECT_EQ(boxes, 2);

    auto check = [&](const Conductors& cc, const char* when) {
        const auto& span = *cc.spans()[0];
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
            for (const auto& sp : cc.spans()) {
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

    pp.addarr("spans", std::vector<std::string>{});
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
    const auto& line = *c->spans().front();
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
}
void clear_towered_section (const std::string& name)
{
    amrex::ParmParse ps("erf.conductors." + name);
    ps.remove("towers"); ps.remove("tower_type");
    amrex::ParmParse pp("erf.conductors");
    pp.addarr("tower_types", std::vector<std::string>{});
    pp.remove("drag_on_flow");
    pp.addarr("spans", std::vector<std::string>{});
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
        EXPECT_NEAR(tw.base()[0], x, 1.0e-9);
        EXPECT_NEAR(tw.base()[2], m.h(x, 500.0), roundoff * 1000) << "the base stands on the ramp";
        EXPECT_NEAR(tw.arm_height(), 30.0, roundoff * 1000) << "the cross-arm at the conductor's height";
        EXPECT_NEAR(std::abs(tw.across()[1]), 1.0, 1.0e-12) << "the cross-arm across a line along x";
    }
    // the prescribed +y wind runs along the cross-arms: only the bodies carry it, the hand value
    const Real U = 10.0, q = 0.5 * 1.2 * U * U, cf = 4.0 * 0.04 - 5.9 * 0.2 + 4.0;
    const Real body = q * cf * 0.2 * 0.5 * (6.0 + 1.5) * 30.0;
    for (int s = 0; s < 3; ++s) { c->advance(0, 0.25 * s, 0.25, m.u, m.v, m.w, m.znd.get(), nullptr, m.geom); }
    for (const auto& tw : c->towers()) {
        const auto F = tw.total_force();
        EXPECT_NEAR(F[1], body, 1.0e-6 * body) << tw.name();
        EXPECT_NEAR(F[0], 0.0, roundoff * body) << tw.name();
        // M_x = -z F_y: the body's q cf phi int w(z) z dz, short by the midpoint sum's H^3 / (12 n^2) on z^2
        const Real H = 30.0, n = 10.0;
        const Real mom = q * cf * 0.2 * (6.0 * H * H / 2.0 + (1.5 - 6.0) / H * (H * H * H / 3.0 - H * H * H / (12.0 * n * n)));
        EXPECT_NEAR(tw.base_moment()[0], -mom, 1.0e-6 * mom) << tw.name();
    }
    // towers.dat: a row at the start of every step with every tower's force and base moment
    std::ifstream f(dir + "/towers.dat");
    std::string header, row;
    std::getline(f, header);
    EXPECT_EQ(header, "time Tw_t1_Fx Tw_t1_Fy Tw_t1_Fz Tw_t1_Mx Tw_t1_My Tw_t2_Fx Tw_t2_Fy Tw_t2_Fz Tw_t2_Mx Tw_t2_My");
    int rows = 0;
    while (std::getline(f, row)) {
        std::istringstream ls(row);
        std::vector<Real> v;
        Real x;
        while (ls >> x) { v.push_back(x); }
        ASSERT_EQ(v.size(), 11u);
        EXPECT_NEAR(v[2], body, 1.0e-6 * body);
        ++rows;
    }
    EXPECT_EQ(rows, 3);
    EXPECT_NE(slurp(dir + "/tower_Tw_t1_stats.csv").find(",drag_h,"), std::string::npos);
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
    for (const auto& s : a->spans()) { const auto f = s->total_drag(); for (int d = 0; d < 3; ++d) { drag[d] += f[d]; } }
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
