// Contract of the conductor geometry: the closest points of two segments, for crossing, skew,
// parallel, end-to-end and degenerate segments, and of two polylines, are the exact minimisers,
// with the distance between them.

#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ERF_ConductorGeometry.H"

using amrex::Real;
using erf_conductors::closest_polylines;
using erf_conductors::closest_segments;
using P3 = std::array<Real,3>;

namespace {
constexpr Real tol = std::is_same<Real, float>::value ? Real(1.0e-5) : Real(1.0e-12);
void expect_point (const P3& got, const P3& want, const char* what)
{
    for (int d = 0; d < 3; ++d) { EXPECT_NEAR(got[d], want[d], tol * 100) << what << " component " << d; }
}
}

TEST(ConductorGeometry, SkewSegmentsMeetAtTheCommonPerpendicular)
{
    // one along x at z = 0, the other along y at z = 3, crossing over (1, 2)
    const auto c = closest_segments({{0, 2, 0}}, {{4, 2, 0}}, {{1, 0, 3}}, {{1, 5, 3}});
    EXPECT_NEAR(c.distance, 3.0, tol);
    expect_point(c.a, {{1, 2, 0}}, "a");
    expect_point(c.b, {{1, 2, 3}}, "b");
}

TEST(ConductorGeometry, TheClosestPointsAreClampedToTheSegments)
{
    // the infinite lines cross at (5, 0, 0), beyond the end of the first segment
    const auto c = closest_segments({{0, 0, 0}}, {{2, 0, 0}}, {{5, -1, 0}}, {{5, 1, 0}});
    EXPECT_NEAR(c.distance, 3.0, tol);
    expect_point(c.a, {{2, 0, 0}}, "a");
    expect_point(c.b, {{5, 0, 0}}, "b");
    // end to end, collinear
    const auto e = closest_segments({{0, 0, 0}}, {{1, 0, 0}}, {{3, 0, 0}}, {{4, 0, 0}});
    EXPECT_NEAR(e.distance, 2.0, tol);
    expect_point(e.a, {{1, 0, 0}}, "a");
    expect_point(e.b, {{3, 0, 0}}, "b");
}

TEST(ConductorGeometry, ParallelAndDegenerateSegments)
{
    // parallel, overlapping: the separation of the lines
    EXPECT_NEAR(closest_segments({{0, 0, 0}}, {{10, 0, 0}}, {{3, 6, 0}}, {{12, 6, 0}}).distance, 6.0, tol);
    // parallel, not overlapping: the gap between the nearest ends
    EXPECT_NEAR(closest_segments({{0, 0, 0}}, {{1, 0, 0}}, {{4, 4, 0}}, {{6, 4, 0}}).distance, 5.0, tol);
    // a point against a segment, and two points
    EXPECT_NEAR(closest_segments({{1, 1, 0}}, {{1, 1, 0}}, {{0, 0, 0}}, {{4, 0, 0}}).distance, 1.0, tol);
    EXPECT_NEAR(closest_segments({{0, 0, 0}}, {{0, 0, 0}}, {{0, 3, 4}}, {{0, 3, 4}}).distance, 5.0, tol);
}

TEST(ConductorGeometry, TwoPolylinesComeClosestWhereTheirSegmentsDo)
{
    // two sagging chains side by side, 6 m apart, one dipping towards the other in the middle
    std::vector<Real> P, Q;
    for (int i = 0; i <= 10; ++i) {
        const Real x = Real(30.0) * i;
        const Real dip = (i == 5) ? Real(4.5) : Real(0.0);
        const Real z = Real(30.0) - Real(0.1) * i * (10 - i);
        P.insert(P.end(), {x, dip, z});
        Q.insert(Q.end(), {x, Real(6.0), z});
    }
    const auto c = closest_polylines(P, Q);
    EXPECT_NEAR(c.distance, 1.5, tol * 100);
    EXPECT_NEAR(c.a[0], 150.0, tol * 1000);
    EXPECT_NEAR(c.b[1], 6.0, tol * 100);
}
