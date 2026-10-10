#include <gtest/gtest.h>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <type_traits>

// Absolute round-off allowance on distances of 1 to 100 m: 1e-9 m in double,
// 1e-4 m in a single-precision build (float Real), the suite's REAL_RTOL
// convention scaled to the magnitudes compared.
constexpr double TOL = std::is_same<amrex::Real, float>::value ? 1.0e-4 : 1.0e-9;

#include "ERF_FireParams.H"
#include "ERF_PolygonIgnition.H"

/**
 * @file ERF_GTestPolygonIgnition.cpp
 * @brief Several perimeter files in one deck: erf.fire.ignition.polygon_file
 *        takes a list, every file is read, and each file is stamped into the
 *        level set with the min(phi, new) rule, so two parallel line fires
 *        give the signed distance to the nearer line and two polygons the
 *        union of their interiors.
 *
 * The fire grid is 200 x 100 m of 2 m cells in two boxes. The exact level set
 * of a polyline of half width w is d - w with d the distance to the nearest
 * segment; for two files it is min(d1, d2) - w, which only holds if the second
 * stamp merges with the first instead of overwriting it.
 */

using namespace amrex;

namespace {

constexpr int  NX = 100;
constexpr int  NY = 50;
constexpr Real DX = 2.0;
constexpr Real W  = 4.0;   // polyline half width [m]

struct GTestFireGrid
{
    BoxArray ba; DistributionMapping dm; Geometry geom;
    GTestFireGrid ()
    {
        Box domain(IntVect(0, 0, 0), IntVect(NX - 1, NY - 1, 0));
        ba = BoxArray(domain);
        ba.maxSize(IntVect(NX / 2, NY, 1));
        dm = DistributionMapping(ba);
        RealBox rb({0.0, 0.0, 0.0}, {NX * DX, NY * DX, 1.0});
        geom = Geometry(domain, rb, CoordSys::cartesian, {0, 0, 0});
    }
};

Real segment_dist (Real px, Real py, Real ax, Real ay, Real bx, Real by)
{
    const Real ux = bx - ax, uy = by - ay;
    Real t = ((px - ax) * ux + (py - ay) * uy) / (ux * ux + uy * uy);
    t = std::max(Real(0.0), std::min(Real(1.0), t));
    return std::hypot(px - (ax + t * ux), py - (ay + t * uy));
}

void write_vertices (const std::string& name, const std::vector<std::pair<Real, Real>>& pts)
{
    std::ofstream f(name);
    f << "# written by ERF_GTestPolygonIgnition\n";
    for (const auto& p : pts) { f << p.first << " " << p.second << "\n"; }
}

/// Adds erf.fire.ignition.polygon_file and removes it again, so one test's
/// deck cannot leak into the next through ParmParse's global table.
struct ScopedPolygonFiles
{
    explicit ScopedPolygonFiles (const std::vector<std::string>& files)
    {
        ParmParse pp("erf.fire");
        pp.addarr("ignition.polygon_file", files);
    }
    ~ScopedPolygonFiles ()
    {
        ParmParse pp("erf.fire");
        pp.remove("ignition.polygon_file");
    }
    ScopedPolygonFiles (const ScopedPolygonFiles&) = delete;
    ScopedPolygonFiles& operator= (const ScopedPolygonFiles&) = delete;
};

} // namespace

// The key takes a list: two names give two files in the order written, and an
// empty entry (the "" of a template deck) is no file at all.
TEST(PolygonIgnition, DeckListsSeveralFiles)
{
    {
        ScopedPolygonFiles deck({"south.csv", "north.csv"});
        FireParams p;
        ASSERT_EQ(p.ignition.polygon_files.size(), 2u);
        EXPECT_EQ(p.ignition.polygon_files[0], "south.csv");
        EXPECT_EQ(p.ignition.polygon_files[1], "north.csv");
        EXPECT_TRUE(p.ignition.has_polygon());
    }
    {
        ScopedPolygonFiles deck({""});
        FireParams p;
        EXPECT_TRUE(p.ignition.polygon_files.empty());
        EXPECT_FALSE(p.ignition.has_polygon());
    }
    {
        FireParams p;
        EXPECT_FALSE(p.ignition.has_polygon());
    }
}

// Each file is read on its own: the vertices come back as written.
TEST(PolygonIgnition, EachFileIsRead)
{
    const std::string a = "erf_gtest_polygon_a.csv", b = "erf_gtest_polygon_b.csv";
    write_vertices(a, {{40.0, 30.0}, {160.0, 30.0}});
    write_vertices(b, {{40.0, 70.0}, {100.0, 70.0}, {160.0, 70.0}});
    std::vector<Real> xs, ys;
    read_polygon_vertices(a, xs, ys);
    ASSERT_EQ(xs.size(), 2u);
    EXPECT_NEAR(xs[1], 160.0, TOL);
    EXPECT_NEAR(ys[1], 30.0, TOL);
    read_polygon_vertices(b, xs, ys);
    ASSERT_EQ(xs.size(), 3u);
    EXPECT_NEAR(xs[1], 100.0, TOL);
    EXPECT_NEAR(ys[2], 70.0, TOL);
    std::remove(a.c_str());
    std::remove(b.c_str());
}

// Two parallel lines stamped one after the other: the level set is the signed
// distance to the nearer line, min(d1, d2) - w, in every cell. A second stamp
// that overwrote the first would leave the cells on the first line at
// d2 - w = 36 m, unburned.
TEST(PolygonIgnition, TwoPolylinesMergeToTheNearerLine)
{
    GTestFireGrid g;
    MultiFab phi(g.ba, g.dm, 1, 0);
    phi.setVal(Real(1.0e10));

    const std::vector<Real> xa = {40.0, 160.0}, ya = {30.0, 30.0};
    const std::vector<Real> xb = {40.0, 160.0}, yb = {70.0, 70.0};
    init_phi_from_polyline(phi, g.geom, xa, ya, W);
    init_phi_from_polyline(phi, g.geom, xb, yb, W);

    int n_checked = 0;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const& p = phi.const_array(mfi);
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                const Real x = (i + Real(0.5)) * DX, y = (j + Real(0.5)) * DX;
                const Real d1 = segment_dist(x, y, xa[0], ya[0], xa[1], ya[1]);
                const Real d2 = segment_dist(x, y, xb[0], yb[0], xb[1], yb[1]);
                EXPECT_NEAR(p(i, j, 0), std::min(d1, d2) - W, TOL) << "cell " << i << "," << j;
                ++n_checked;
            }
        }
    }
    EXPECT_EQ(n_checked, NX * NY);

    // the first line is still burning after the second stamp (phi = -w on it)
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const& p = phi.const_array(mfi);
        const int j = 14;   // y = 29 m, on the first line
        if (j < bx.smallEnd(1) || j > bx.bigEnd(1)) { continue; }
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
            const Real x = (i + Real(0.5)) * DX;
            if (x > 40.0 && x < 160.0) { EXPECT_NEAR(p(i, j, 0), 1.0 - W, TOL); }
        }
    }
}

// Two closed polygons: a cell inside either square is burned, a cell between
// them is unburned at the distance to the nearer edge.
TEST(PolygonIgnition, TwoPolygonsBurnTheUnion)
{
    GTestFireGrid g;
    MultiFab phi(g.ba, g.dm, 1, 0);
    phi.setVal(Real(1.0e10));
    init_phi_from_polygon(phi, g.geom, {20.0, 60.0, 60.0, 20.0}, {30.0, 30.0, 70.0, 70.0});
    init_phi_from_polygon(phi, g.geom, {140.0, 180.0, 180.0, 140.0}, {30.0, 30.0, 70.0, 70.0});
    int n_a = 0, n_b = 0, n_mid = 0;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const& p = phi.const_array(mfi);
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                const Real x = (i + Real(0.5)) * DX, y = (j + Real(0.5)) * DX;
                if (y > 30.0 && y < 70.0 && x > 20.0 && x < 60.0)   { EXPECT_LT(p(i, j, 0), 0.0); ++n_a; }
                if (y > 30.0 && y < 70.0 && x > 140.0 && x < 180.0) { EXPECT_LT(p(i, j, 0), 0.0); ++n_b; }
                // the cell centred at (101, 49) is 39 m from the east square, 41 m from the west one
                if (i == 50 && j == 24) { EXPECT_NEAR(p(i, j, 0), 39.0, TOL); ++n_mid; }
            }
        }
    }
    EXPECT_EQ(n_a, 20 * 20);
    EXPECT_EQ(n_b, 20 * 20);
    EXPECT_EQ(n_mid, 1);
}
