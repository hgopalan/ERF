#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Math.H>
#include <cmath>
#include <string>
#include <vector>

#include "ERF_FireSuppression.H"

/**
 * @file ERF_GTestFireSuppression.cpp
 * @brief The action-file parser and the geometry kernels behind
 *        erf.fire.suppression: good lines of every type, malformed lines,
 *        duplicate and unknown ids, the polygon inside test and the cells a
 *        built polyline covers.
 */

using namespace amrex;
using namespace fire_suppression;

namespace {

constexpr Real TOL = (sizeof(Real) == 8) ? 1.0e-12 : 1.0e-5;

const std::string GOOD =
    "# id type start geometry param expiry flame_limit\n"
    "L1  line     3600  500 200 900 600 950 650   rate=0.5        -1    2.4\n"
    "D1  drop     5400  poly:100,100;200,100;200,200;100,200  ros_factor=0.0  1800  -\n"
    "D2  drop     0     poly:0,0;10,0;10,10  ros_factor=0.3  -1  -   # trailing comment\n"
    "B1  burnout  4000  ref=L1  offset=20  -  -\n";

} // namespace

TEST(FireSuppression, ParsesEveryActionType)
{
    std::vector<SuppressionAction> a;
    std::string err;
    ASSERT_TRUE(parse_suppression_text(GOOD, {}, {}, a, err)) << err;
    ASSERT_EQ(a.size(), 4u);

    EXPECT_EQ(a[0].id, "L1");
    EXPECT_EQ(a[0].type, line);
    EXPECT_NEAR(a[0].start_s, 3600.0, TOL);
    EXPECT_EQ(a[0].nverts(), 3);
    EXPECT_NEAR(a[0].rate, 0.5, TOL);
    EXPECT_NEAR(a[0].expiry_s, -1.0, TOL);
    EXPECT_NEAR(a[0].flame_limit_m, 2.4, TOL);
    // 500,200 -> 900,600 is 400 sqrt(2); 900,600 -> 950,650 is 50 sqrt(2)
    EXPECT_NEAR(a[0].length(), 450.0 * std::sqrt(2.0), TOL * 450.0);   // precision-aware: float sums to 1.5e-5

    EXPECT_EQ(a[1].type, drop);
    EXPECT_EQ(a[1].nverts(), 4);
    EXPECT_NEAR(a[1].ros_factor, 0.0, TOL);
    EXPECT_NEAR(a[1].expiry_s, 1800.0, TOL);
    EXPECT_NEAR(a[1].flame_limit_m, -1.0, TOL);

    EXPECT_EQ(a[2].type, drop);
    EXPECT_NEAR(a[2].ros_factor, 0.3, TOL);
    EXPECT_NEAR(a[2].expiry_s, -1.0, TOL);

    EXPECT_EQ(a[3].type, burnout);
    EXPECT_EQ(a[3].ref, "L1");
    EXPECT_NEAR(a[3].offset, 20.0, TOL);
    EXPECT_NEAR(a[3].start_s, 4000.0, TOL);
    for (const auto& x : a) {
        EXPECT_FALSE(x.applied);
        EXPECT_FALSE(x.expired);
        EXPECT_NEAR(x.built_m, 0.0, TOL);
    }
}

TEST(FireSuppression, RejectsMalformedLines)
{
    const std::vector<std::pair<std::string, std::string>> bad = {
        {"L1 line 0 0 0 100 rate=1 -1 -",                    "even number of coordinates"},   // three coordinates
        {"L1 line 0 0 0 100 100 rate=0 -1 -",                "rate must be"},
        {"L1 line 0 0 0 100 100 -1 -",                       "is not a number"},   // the '-' is read as a vertex
        {"L1 line 0 0 0 100 100 speed=1 -1 -",               "rate=<m/s>"},
        {"L1 line -5 0 0 100 100 rate=1 -1 -",               "start_s"},
        {"L1 line 0 0 0 100 100 rate=1 0 -",                 "expiry_s"},
        {"L1 line 0 0 0 100 100 rate=1 -1 0",                "flame_limit_m"},
        {"L1 line 0 0 0 100 100 rate=1 -1",                  "exactly two fields"},
        {"L1 line 0 0 0",                                    "fewer than six"},
        {"L1 line 0 0 0 100 100 rate=1 -1 - extra",          "exactly two fields"},
        {"X1 wall 0 0 0 100 100 rate=1 -1 -",                "unknown type"},
        {"D1 drop 0 poly:0,0;1,0 ros_factor=0 -1 -",         "at least three"},
        {"D1 drop 0 poly:0,0;1,0;1,1 ros_factor=1.0 -1 -",   "ros_factor"},
        {"D1 drop 0 poly:0,0;1,0;1,1 ros_factor=0.5 -1 2.0", "lines only"},
        {"D1 drop 0 0 0 1 0 1 1 ros_factor=0.5 -1 -",        "poly:"},
        {"B1 burnout 0 ref=L1 offset=-1 - -",                "offset"},
        {"B1 burnout 0 ref=L1 offset=5 100 -",               "burnout takes -"},
        {"B1 burnout 0 L1 offset=5 - -",                     "ref="},
    };
    for (const auto& [text, why] : bad) {
        SuppressionAction a;
        std::string err;
        EXPECT_FALSE(parse_suppression_line(text, a, err)) << text;
        EXPECT_NE(err.find(why), std::string::npos) << text << " -> " << err;
    }
}

TEST(FireSuppression, TextParserQuotesTheLineAndRejectsDuplicates)
{
    std::vector<SuppressionAction> a;
    std::string err;
    const std::string dup = "L1 line 0 0 0 100 100 rate=1 -1 -\nL1 line 5 0 0 50 50 rate=1 -1 -\n";
    EXPECT_FALSE(parse_suppression_text(dup, {}, {}, a, err));
    EXPECT_NE(err.find("line 2"), std::string::npos) << err;
    EXPECT_NE(err.find("duplicate id \"L1\""), std::string::npos) << err;
    EXPECT_NE(err.find("\"L1 line 5 0 0 50 50 rate=1 -1 -\""), std::string::npos) << err;

    // A duplicate against the ids read earlier
    EXPECT_FALSE(parse_suppression_text("L2 line 0 0 0 100 100 rate=1 -1 -\n", {"L2"}, {"L2"}, a, err));
    EXPECT_NE(err.find("duplicate id \"L2\""), std::string::npos) << err;

    // A malformed line is reported with its number and text
    const std::string bad = "# ok\nL1 line 0 0 0 100 100 rate=1 -1 -\n\nD9 drop 0 poly:0,0 ros_factor=0 -1 -\n";
    EXPECT_FALSE(parse_suppression_text(bad, {}, {}, a, err));
    EXPECT_NE(err.find("line 4"), std::string::npos) << err;
    EXPECT_NE(err.find("\"D9 drop 0 poly:0,0 ros_factor=0 -1 -\""), std::string::npos) << err;

    // A burnout must reference a line: in the text or read earlier
    EXPECT_FALSE(parse_suppression_text("B1 burnout 0 ref=L7 offset=5 - -\n", {}, {}, a, err));
    EXPECT_NE(err.find("unknown line id \"L7\""), std::string::npos) << err;
    EXPECT_TRUE(parse_suppression_text("B1 burnout 0 ref=L7 offset=5 - -\n", {"L7"}, {"L7"}, a, err)) << err;
    ASSERT_EQ(a.size(), 1u);
    EXPECT_EQ(a[0].ref, "L7");
}

TEST(FireSuppression, PointInPolygon)
{
    // A concave L shape: (0,0) (4,0) (4,1) (1,1) (1,4) (0,4)
    const Real v[] = {0, 0, 4, 0, 4, 1, 1, 1, 1, 4, 0, 4};
    const int nv = 6;
    EXPECT_TRUE (point_in_polygon(0.5, 0.5, v, nv));
    EXPECT_TRUE (point_in_polygon(3.5, 0.5, v, nv));
    EXPECT_TRUE (point_in_polygon(0.5, 3.5, v, nv));
    EXPECT_FALSE(point_in_polygon(2.5, 2.5, v, nv));   // the notch
    EXPECT_FALSE(point_in_polygon(5.0, 0.5, v, nv));
    EXPECT_FALSE(point_in_polygon(-1.0, -1.0, v, nv));
    // A horizontal edge at the test height does not trap (yj == yi guarded)
    EXPECT_TRUE (point_in_polygon(0.5, 1.0, v, nv));
}

namespace {

/// Flood fill over the uncovered cells of an n x n block from (i0, j0) with
/// 4-neighbour steps; true when (i1, j1) is reached.
bool four_connected (const std::vector<std::vector<bool>>& cov, int n, int i0, int j0, int i1, int j1)
{
    std::vector<std::vector<bool>> seen(n, std::vector<bool>(n, false));
    std::vector<std::pair<int, int>> stack{{i0, j0}};
    seen[i0][j0] = true;
    while (!stack.empty()) {
        auto [i, j] = stack.back();
        stack.pop_back();
        if (i == i1 && j == j1) { return true; }
        const int di[4] = {1, -1, 0, 0}, dj[4] = {0, 0, 1, -1};
        for (int d = 0; d < 4; ++d) {
            const int ii = i + di[d], jj = j + dj[d];
            if (ii < 0 || jj < 0 || ii >= n || jj >= n || seen[ii][jj] || cov[ii][jj]) { continue; }
            seen[ii][jj] = true;
            stack.push_back({ii, jj});
        }
    }
    return false;
}

} // namespace

TEST(FireSuppression, BuiltPolylineCoversACellWideBarrier)
{
    const Real dx = 2.0, half = 1.0, hx = 1.0, hy = 1.0;
    auto centre = [&] (int i) { return (i + 0.5) * dx; };

    // A line along x at y = 21 (cell centres of row j = 10) from x = 100 to 140,
    // built 20 m: cells i = 50..59 of row 10 by the box test, plus cells 49
    // and 60, whose centres are exactly half a cell from the two ends (the
    // distance rule is inclusive), and nothing in rows 9 and 11.
    const Real vx[] = {100, 21, 140, 21};
    for (int i = 48; i < 72; ++i) {
        const bool expect = (i >= 49 && i <= 60);
        EXPECT_EQ(polyline_covers_cell(centre(i), centre(10), hx, hy, half, vx, 2, 20.0), expect) << "i=" << i;
        EXPECT_FALSE(polyline_covers_cell(centre(i), centre(9),  hx, hy, half, vx, 2, 20.0)) << "i=" << i;
        EXPECT_FALSE(polyline_covers_cell(centre(i), centre(11), hx, hy, half, vx, 2, 20.0)) << "i=" << i;
    }
    // Nothing built: nothing covered
    EXPECT_FALSE(polyline_covers_cell(centre(50), centre(10), hx, hy, half, vx, 2, 0.0));
    // Fully built: the whole run of cells up to the far end's half-cell touch
    EXPECT_TRUE(polyline_covers_cell(centre(69), centre(10), hx, hy, half, vx, 2, 40.0));
    EXPECT_TRUE(polyline_covers_cell(centre(70), centre(10), hx, hy, half, vx, 2, 40.0));
    EXPECT_FALSE(polyline_covers_cell(centre(71), centre(10), hx, hy, half, vx, 2, 40.0));
    // A line through cell centres, ending inside cell 59: cells 49..59 only
    const Real vi[] = {99, 21, 119, 21};
    EXPECT_TRUE(polyline_covers_cell(centre(49), centre(10), hx, hy, half, vi, 2, 100.0));
    EXPECT_TRUE(polyline_covers_cell(centre(59), centre(10), hx, hy, half, vi, 2, 100.0));
    EXPECT_FALSE(polyline_covers_cell(centre(48), centre(10), hx, hy, half, vi, 2, 100.0));
    EXPECT_FALSE(polyline_covers_cell(centre(60), centre(10), hx, hy, half, vi, 2, 100.0));

    // A line along a cell edge (y = 20, between rows 9 and 10) covers both rows
    // by the distance test, so a front cannot slip along the edge.
    const Real ve[] = {100, 20, 140, 20};
    EXPECT_TRUE(polyline_covers_cell(centre(55), centre(9),  hx, hy, half, ve, 2, 40.0));
    EXPECT_TRUE(polyline_covers_cell(centre(55), centre(10), hx, hy, half, ve, 2, 40.0));
    EXPECT_FALSE(polyline_covers_cell(centre(55), centre(8),  hx, hy, half, ve, 2, 40.0));

    // A diagonal through cell corners, (0,0) to (20,20): only the ten diagonal
    // cells (the corner touches of their neighbours do not count), and the
    // chain still blocks every 4-connected path across it.
    const int n = 10;
    const Real vd[] = {0, 0, 20, 20};
    std::vector<std::vector<bool>> cov(n, std::vector<bool>(n, false));
    int covered = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cov[i][j] = polyline_covers_cell(centre(i), centre(j), hx, hy, half, vd, 2, 100.0);
            if (cov[i][j]) { ++covered; }
            EXPECT_EQ(cov[i][j], i == j) << i << "," << j;
        }
    }
    EXPECT_EQ(covered, n);
    EXPECT_FALSE(four_connected(cov, n, 9, 0, 0, 9));
    EXPECT_TRUE (four_connected(cov, n, 9, 0, 5, 1));   // the same side stays connected

    // Lines at 15, 30 and 60 degrees across a 20 x 20 block: whatever cells
    // the rules pick, no 4-connected path joins the two sides, and a line
    // built only halfway leaves the far half open.
    const int m = 20;
    for (const Real deg : {15.0, 30.0, 60.0}) {
        const Real t = std::tan(deg * amrex::Math::pi<Real>() / 180.0);
        const Real vs[] = {0, 1, 40, 1 + 40 * t};
        std::vector<std::vector<bool>> c(m, std::vector<bool>(m, false));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                c[i][j] = polyline_covers_cell(centre(i), centre(j), hx, hy, half, vs, 2, 200.0);
            }
        }
        // (19, 0) is below the line, (0, 19) above it
        EXPECT_FALSE(c[19][0]) << deg;
        EXPECT_FALSE(c[0][19]) << deg;
        EXPECT_FALSE(four_connected(c, m, 19, 0, 0, 19)) << deg;
        std::vector<std::vector<bool>> h(m, std::vector<bool>(m, false));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                h[i][j] = polyline_covers_cell(centre(i), centre(j), hx, hy, half, vs, 2, 10.0);
            }
        }
        EXPECT_TRUE(four_connected(h, m, 19, 0, 0, 19)) << deg;
    }
}

TEST(FireSuppression, SegmentGeometry)
{
    EXPECT_NEAR(segment_distance(5.0, 3.0, 0.0, 0.0, 10.0, 0.0), 3.0, TOL);
    EXPECT_NEAR(segment_distance(13.0, 4.0, 0.0, 0.0, 10.0, 0.0), 5.0, TOL);
    EXPECT_NEAR(segment_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0), 5.0, TOL);   // degenerate segment
    EXPECT_TRUE (segment_crosses_box(-5.0, 0.5, 5.0, 0.5, 0.0, 0.0, 1.0, 1.0));
    EXPECT_FALSE(segment_crosses_box(-5.0, 2.0, 5.0, 2.0, 0.0, 0.0, 1.0, 1.0));
    EXPECT_TRUE (segment_crosses_box(0.2, 0.2, 0.8, 0.8, 0.0, 0.0, 1.0, 1.0));   // inside
    EXPECT_TRUE (segment_crosses_box(-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0)); // corner to corner
    EXPECT_FALSE(segment_crosses_box(2.0, -1.0, 2.0, 5.0, 0.0, 0.0, 1.0, 1.0));  // parallel, outside
}
