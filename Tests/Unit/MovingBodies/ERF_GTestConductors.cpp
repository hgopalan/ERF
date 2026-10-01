// Contract of Conductors, the manager ERF holds: the attachments are placed at their height above
// the terrain surface under each end (the k = 0 node plane of z_phys_nd, bilinear between the
// nodes) and ground.dat records it; on a uniform-dz mesh the given heights are absolute; the
// spans step only on the anchor level and write one diagnostics row per step; and without a
// prescribed velocity the wind handed to MoorDyn is ERF's velocity sampled where the line is
// now, not where it hung.

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_RealBox.H>

#include <gtest/gtest.h>

#include "ERF_Conductors.H"

namespace {

using amrex::Real;

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
    pp.remove("prescribed_velocity");
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
    c->advance(0, 0.0, 0.1, m.u, m.v, m.w, nullptr, m.geom);   // level 0: nothing happens
    EXPECT_DOUBLE_EQ(span.mid_offset(), off0);
    EXPECT_FALSE(std::filesystem::exists(dir + "/Tadv.dat"));
    for (int s = 0; s < 3; ++s) { c->advance(1, 0.1 * s, 0.1, m.u, m.v, m.w, nullptr, m.geom); }
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
        c->advance(0, 0.25 * s, 0.25, m.u, m.v, m.w, nullptr, m.geom);
        for (unsigned p = 0; p < span.num_kinematics_points(); ++p) {
            // the flow at the line nodes; no wind at MoorDyn's fixed entries after them
            const auto uvw = span.wind_at_point(p);
            const Real expect = (p < span.num_nodes()) ? m.v_at(where[3*p+1], where[3*p+2]) : Real(0.0);
            ASSERT_NEAR(uvw[0], 0.0, 1.0e-9) << "step " << s << " point " << p;
            ASSERT_NEAR(uvw[1], expect, 1.0e-9 * std::max(Real(1.0), expect)) << "step " << s << " point " << p;
            ASSERT_NEAR(uvw[2], 0.0, 1.0e-9) << "step " << s << " point " << p;
        }
        moved = std::max(moved, span.mid_offset());
    }
    // the line really moved, so sampling at the initial positions would have handed a different wind
    EXPECT_GT(moved, 0.5) << "the span must blow out in the sampled crosswind";
    EXPECT_GT(m.vy * moved, 1.0e-3) << "the field must change measurably over the distance the line moved";
}
