#ifndef ERF_CONDUCTOR_GEOMETRY_H
#define ERF_CONDUCTOR_GEOMETRY_H

/**
 * Closest-approach geometry of the conductor module: segment-segment, polyline-polyline,
 * segment-box and polyline-box distances, used to flag lines that clash and conductors that come
 * close to a transformer. Coordinates and distances are in ERF's frame (m).
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_REAL.H>

namespace erf_conductors {

/** The closest approach of two curves, or of a curve and a box: the distance and the closest point on each. */
struct Closest
{
    amrex::Real distance{std::numeric_limits<amrex::Real>::max()};   //!< |a - b| (m); the largest Real when nothing was compared
    std::array<amrex::Real,3> a{{0.0, 0.0, 0.0}};                    //!< the closest point on the first curve (m, ERF frame)
    std::array<amrex::Real,3> b{{0.0, 0.0, 0.0}};                    //!< the closest point on the second curve or on the box (m, ERF frame)
};

/**
 * The closest points of the segments p0-p1 and q0-q1: the minimum of |p(s) - q(t)| over s, t in
 * [0, 1], by clamping the unconstrained minimiser onto the square (Ericson, Real-Time Collision
 * Detection, 5.1.9), exact for parallel and degenerate segments as well.
 */
inline Closest closest_segments (const std::array<amrex::Real,3>& p0, const std::array<amrex::Real,3>& p1,
                                 const std::array<amrex::Real,3>& q0, const std::array<amrex::Real,3>& q1)
{
    using amrex::Real;
    std::array<Real,3> d1, d2, r;
    for (int i = 0; i < 3; ++i) { d1[i] = p1[i] - p0[i]; d2[i] = q1[i] - q0[i]; r[i] = p0[i] - q0[i]; }
    auto dot = [] (const std::array<Real,3>& x, const std::array<Real,3>& y) { return x[0]*y[0] + x[1]*y[1] + x[2]*y[2]; };
    const Real a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    const Real tiny = std::numeric_limits<Real>::epsilon() * std::max({a, e, Real(1.0)});
    Real s = 0.0, t = 0.0;
    if (a <= tiny && e <= tiny) {
        s = t = 0.0;
    } else if (a <= tiny) {
        t = std::clamp(f / e, Real(0.0), Real(1.0));
    } else {
        const Real c = dot(d1, r);
        if (e <= tiny) {
            s = std::clamp(-c / a, Real(0.0), Real(1.0));
        } else {
            const Real b = dot(d1, d2);
            const Real denom = a * e - b * b;
            s = (denom > tiny * std::max(a, e)) ? std::clamp((b * f - c * e) / denom, Real(0.0), Real(1.0)) : Real(0.0);
            t = (b * s + f) / e;
            if (t < 0.0) { t = 0.0; s = std::clamp(-c / a, Real(0.0), Real(1.0)); }
            else if (t > 1.0) { t = 1.0; s = std::clamp((b - c) / a, Real(0.0), Real(1.0)); }
        }
    }
    Closest out;
    for (int i = 0; i < 3; ++i) { out.a[i] = p0[i] + s * d1[i]; out.b[i] = q0[i] + t * d2[i]; }
    Real d2sum = 0.0;
    for (int i = 0; i < 3; ++i) { d2sum += (out.a[i] - out.b[i]) * (out.a[i] - out.b[i]); }
    out.distance = std::sqrt(d2sum);
    return out;
}

/**
 * The closest approach of two polylines given as 3 coordinates per vertex.
 * Precondition: each has at least two vertices (aborts otherwise).
 */
inline Closest closest_polylines (const std::vector<amrex::Real>& P, const std::vector<amrex::Real>& Q)
{
    using amrex::Real;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(P.size() % 3 == 0 && Q.size() % 3 == 0 && P.size() >= 6 && Q.size() >= 6,
                                     "closest_polylines: each polyline needs 3 coordinates per vertex and at least two vertices");
    Closest best;
    const std::size_t np = P.size() / 3, nq = Q.size() / 3;
    auto at = [] (const std::vector<Real>& v, std::size_t i) { return std::array<Real,3>{{v[3*i], v[3*i+1], v[3*i+2]}}; };
    // a segment pair whose bounding boxes lie farther apart than the closest approach found so far cannot
    // come closer: skipped, without changing the answer (lines far apart cost one box test per pair)
    auto box_gap = [&] (std::size_t i, std::size_t j) {
        Real g2 = 0.0;
        for (std::size_t d = 0; d < 3; ++d) {
            const Real plo = std::min(P[3*i+d], P[3*(i+1)+d]), phi = std::max(P[3*i+d], P[3*(i+1)+d]);
            const Real qlo = std::min(Q[3*j+d], Q[3*(j+1)+d]), qhi = std::max(Q[3*j+d], Q[3*(j+1)+d]);
            const Real g = std::max({Real(0.0), qlo - phi, plo - qhi});
            g2 += g * g;
        }
        return std::sqrt(g2);
    };
    for (std::size_t i = 0; i + 1 < np; ++i) {
        for (std::size_t j = 0; j + 1 < nq; ++j) {
            if (box_gap(i, j) > best.distance) { continue; }
            const Closest c = closest_segments(at(P, i), at(P, i + 1), at(Q, j), at(Q, j + 1));
            if (c.distance < best.distance) { best = c; }
        }
    }
    return best;
}

/** The point of the axis-aligned box [lo, hi] closest to p (p itself when inside). */
inline std::array<amrex::Real,3> closest_box_point (const std::array<amrex::Real,3>& p,
                                                    const std::array<amrex::Real,3>& lo, const std::array<amrex::Real,3>& hi)
{
    return {{std::clamp(p[0], lo[0], hi[0]), std::clamp(p[1], lo[1], hi[1]), std::clamp(p[2], lo[2], hi[2])}};
}

/**
 * The closest approach of the segment p0-p1 and the axis-aligned box [lo, hi] (distance zero when
 * they meet). The distance from a point moving along the segment to a convex set is a convex
 * function of the point's parameter, so a golden-section search on [0, 1] finds its minimum to
 * round-off, at an end of the segment as well (the bracket closes on it).
 */
inline Closest closest_segment_box (const std::array<amrex::Real,3>& p0, const std::array<amrex::Real,3>& p1,
                                    const std::array<amrex::Real,3>& lo, const std::array<amrex::Real,3>& hi)
{
    using amrex::Real;
    auto at = [&] (Real t) { return std::array<Real,3>{{p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]),
                                                         p0[2] + t * (p1[2] - p0[2])}}; };
    auto dist2 = [&] (Real t) {
        const auto p = at(t);
        const auto q = closest_box_point(p, lo, hi);
        return (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]);
    };
    const Real g = Real(0.5) * (std::sqrt(Real(5.0)) - Real(1.0));
    Real a = 0.0, b = 1.0;
    Real c = b - g * (b - a), d = a + g * (b - a);
    Real fc = dist2(c), fd = dist2(d);
    // the bracket shrinks by g per pass: 0.618^120 is far below any precision's epsilon
    for (int it = 0; it < 120 && b - a > std::numeric_limits<Real>::epsilon(); ++it) {
        if (fc <= fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = dist2(c); }
        else          { a = c; c = d; fc = fd; d = a + g * (b - a); fd = dist2(d); }
    }
    const Real t = Real(0.5) * (a + b);
    Closest out;
    out.a = at(t);
    out.b = closest_box_point(out.a, lo, hi);
    out.distance = std::sqrt(dist2(t));
    return out;
}

/**
 * The closest approach of a polyline (3 coordinates per vertex) and the axis-aligned box [lo, hi].
 * Precondition: the polyline has at least two vertices (aborts otherwise).
 */
inline Closest closest_polyline_box (const std::vector<amrex::Real>& P,
                                     const std::array<amrex::Real,3>& lo, const std::array<amrex::Real,3>& hi)
{
    using amrex::Real;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(P.size() % 3 == 0 && P.size() >= 6,
                                     "closest_polyline_box: the polyline needs 3 coordinates per vertex and at least two vertices");
    Closest best;
    auto at = [&] (std::size_t i) { return std::array<Real,3>{{P[3*i], P[3*i+1], P[3*i+2]}}; };
    for (std::size_t i = 0; i + 1 < P.size() / 3; ++i) {
        // a segment whose bounding box lies farther from the box than the closest approach so far is skipped
        Real g2 = 0.0;
        for (std::size_t d = 0; d < 3; ++d) {
            const Real slo = std::min(P[3*i+d], P[3*(i+1)+d]), shi = std::max(P[3*i+d], P[3*(i+1)+d]);
            const Real g = std::max({Real(0.0), lo[d] - shi, slo - hi[d]});
            g2 += g * g;
        }
        if (std::sqrt(g2) > best.distance) { continue; }
        const Closest c = closest_segment_box(at(i), at(i + 1), lo, hi);
        if (c.distance < best.distance) { best = c; }
    }
    return best;
}

} // namespace erf_conductors

#endif
