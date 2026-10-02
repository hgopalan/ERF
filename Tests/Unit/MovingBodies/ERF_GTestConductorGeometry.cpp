// Contract of the conductor geometry: the closest points of two segments, for crossing, skew,
// parallel, end-to-end and degenerate segments, and of two polylines, are the exact minimisers,
// with the distance between them; and the closest approach of a segment or a polyline to a box is
// the exact distance, over the box, past an edge, through it, and no larger than any sampled point's.

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ERF_ConductorGeometry.H"

using amrex::Real;
using erf_conductors::closest_polyline_box;
using erf_conductors::closest_polylines;
using erf_conductors::closest_segment_box;
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

// a transformer-sized box: 8 m x 5 m x 6 m with its base at z = 100
namespace {
const P3 lo{{-4.0, -2.5, 100.0}};
const P3 hi{{4.0, 2.5, 106.0}};
// the search fixes the distance to round-off, but where a stretch of the segment is equally close
// (alongside a face) the squared distance is flat to round-off over about sqrt(epsilon) of the
// segment, so the closest point is only that close: 8e-6 x 80 m in double, 3e-4 x 80 m in single
constexpr Real ptol = std::is_same<Real, float>::value ? Real(0.05) : Real(1.0e-5);
void expect_near_point (const P3& got, const P3& want, const char* what)
{
    for (int d = 0; d < 3; ++d) { EXPECT_NEAR(got[d], want[d], ptol) << what << " component " << d; }
}
}

TEST(ConductorGeometry, ASegmentOverABoxClearsItsTop)
{
    // a conductor running along x 2 m above the top, across the whole box and beyond
    const auto c = closest_segment_box({{-50.0, 1.0, 108.0}}, {{50.0, 1.0, 108.0}}, lo, hi);
    EXPECT_NEAR(c.distance, 2.0, tol * 100);
    EXPECT_NEAR(c.a[2], 108.0, tol * 100);
    EXPECT_NEAR(c.b[2], 106.0, tol * 100);
    EXPECT_GE(c.a[0], -4.0 - ptol); EXPECT_LE(c.a[0], 4.0 + ptol);
}

TEST(ConductorGeometry, ASegmentPastAnEdgeIsMeasuredToThatEdge)
{
    // along y, 3 m out from the x-high face and 4 m above the top: 5 m from the top edge
    const auto c = closest_segment_box({{7.0, -40.0, 110.0}}, {{7.0, 40.0, 110.0}}, lo, hi);
    EXPECT_NEAR(c.distance, 5.0, tol * 100);
    expect_near_point(c.b, {{4.0, c.a[1], 106.0}}, "on the edge");
    // and one ending short of the box is measured from its end: a corner 3-4-12 away
    const auto e = closest_segment_box({{7.0, 6.5, 118.0}}, {{7.0, 60.0, 118.0}}, lo, hi);
    EXPECT_NEAR(e.distance, std::sqrt(Real(9.0 + 16.0 + 144.0)), tol * 1000);
    expect_near_point(e.a, {{7.0, 6.5, 118.0}}, "the near end");
    expect_near_point(e.b, {{4.0, 2.5, 106.0}}, "the corner");
}

TEST(ConductorGeometry, ASegmentThroughABoxTouchesIt)
{
    const auto c = closest_segment_box({{-10.0, 0.0, 103.0}}, {{10.0, 0.0, 103.0}}, lo, hi);
    EXPECT_NEAR(c.distance, 0.0, tol * 100);
    const auto d = closest_segment_box({{0.0, 0.0, 120.0}}, {{1.0, 1.0, 50.0}}, lo, hi);
    EXPECT_NEAR(d.distance, 0.0, tol * 100) << "a steep segment through the top";
}

TEST(ConductorGeometry, NoPointOfASegmentIsCloserToABoxThanItsClosestApproach)
{
    // random segments around the box: the search's minimum must not lie above any sampled point's
    // distance, and must be attained at its own closest point
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> u(-30.0, 30.0);
    for (int n = 0; n < 200; ++n) {
        const P3 p0{{Real(u(gen)), Real(u(gen)), Real(103.0 + u(gen))}};
        const P3 p1{{Real(u(gen)), Real(u(gen)), Real(103.0 + u(gen))}};
        const auto c = closest_segment_box(p0, p1, lo, hi);
        Real sampled = 1.0e30;
        for (int i = 0; i <= 2000; ++i) {
            const Real t = Real(i) / 2000;
            P3 q, b;
            for (int d = 0; d < 3; ++d) { q[d] = p0[d] + t * (p1[d] - p0[d]); b[d] = std::clamp(q[d], lo[d], hi[d]); }
            sampled = std::min(sampled, std::sqrt((q[0]-b[0])*(q[0]-b[0]) + (q[1]-b[1])*(q[1]-b[1]) + (q[2]-b[2])*(q[2]-b[2])));
        }
        ASSERT_LE(c.distance, sampled + Real(1.0e-4)) << "segment " << n;
        Real at = 0.0;
        for (int d = 0; d < 3; ++d) { at += (c.a[d] - c.b[d]) * (c.a[d] - c.b[d]); }
        ASSERT_NEAR(std::sqrt(at), c.distance, Real(1.0e-4)) << "segment " << n;
    }
}

TEST(ConductorGeometry, APolylineComesClosestToABoxWhereItsNearestSegmentDoes)
{
    // a sagging conductor: down from 112 m over the box to 104 m beside it and up again
    const std::vector<Real> P{-60.0, 0.0, 112.0,   9.0, 0.0, 104.0,   60.0, 0.0, 112.0};
    const auto c = closest_polyline_box(P, lo, hi);
    const auto first = closest_segment_box({{-60.0, 0.0, 112.0}}, {{9.0, 0.0, 104.0}}, lo, hi);
    const auto second = closest_segment_box({{9.0, 0.0, 104.0}}, {{60.0, 0.0, 112.0}}, lo, hi);
    EXPECT_NEAR(c.distance, std::min(first.distance, second.distance), tol * 100);
    EXPECT_LT(c.distance, 5.0);
}
