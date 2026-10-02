// Contract of Conductors, the manager ERF holds: the attachments are placed at their height above
// the terrain surface under each end (the k = 0 node plane of z_phys_nd, bilinear between the
// nodes) and ground.dat records it; on a uniform-dz mesh the given heights are absolute; the
// spans step only on the anchor level and write one diagnostics row per step; and without a
// prescribed velocity the wind handed to MoorDyn is ERF's velocity sampled where the line is
// now, not where it hung; and a restart from a checkpoint continues the lines, their drag on the
// air, their statistics and their logs exactly where the checkpoint left them.

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
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
    for (const char* key : {"prescribed_velocity", "drag_on_flow", "epsilon", "node_output_int", "stats_start"}) { pp.remove(key); }
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
}
