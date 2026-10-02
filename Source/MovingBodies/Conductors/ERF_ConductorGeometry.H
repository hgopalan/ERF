#ifndef ERF_CONDUCTOR_GEOMETRY_H
#define ERF_CONDUCTOR_GEOMETRY_H

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <AMReX_REAL.H>

namespace erf_conductors {

/** The closest approach of two curves: the distance and the closest point on each. */
struct Closest
{
    amrex::Real distance{std::numeric_limits<amrex::Real>::max()};
    std::array<amrex::Real,3> a{{0.0, 0.0, 0.0}};
    std::array<amrex::Real,3> b{{0.0, 0.0, 0.0}};
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

/** The closest approach of two polylines given as 3 coordinates per vertex (each at least two vertices). */
inline Closest closest_polylines (const std::vector<amrex::Real>& P, const std::vector<amrex::Real>& Q)
{
    using amrex::Real;
    Closest best;
    const std::size_t np = P.size() / 3, nq = Q.size() / 3;
    auto at = [] (const std::vector<Real>& v, std::size_t i) { return std::array<Real,3>{{v[3*i], v[3*i+1], v[3*i+2]}}; };
    for (std::size_t i = 0; i + 1 < np; ++i) {
        for (std::size_t j = 0; j + 1 < nq; ++j) {
            const Closest c = closest_segments(at(P, i), at(P, i + 1), at(Q, j), at(Q, j + 1));
            if (c.distance < best.distance) { best = c; }
        }
    }
    return best;
}

} // namespace erf_conductors

#endif
