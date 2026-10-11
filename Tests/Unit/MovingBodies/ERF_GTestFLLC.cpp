// The filtered lifting-line correction: its kernel, the span interpolation, the induced
// velocity of an elliptic lift distribution (the classical uniform downwash in the narrow-kernel
// limit), a zero correction when the run's kernel is the optimal one, a downwash correction
// when it is wider, the relaxation, and the state round trip.

#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <AMReX_REAL.H>

#include <gtest/gtest.h>

#include "ERF_FLLC.H"

namespace {

using amrex::Real;
using erf_actuator::FLLC;

Real tol () { return (sizeof(Real) == 8) ? Real(1.0e-10) : Real(1.0e-4); }

// A blade of n nodes over the span [0, b] with an elliptic circulation Gamma0 sqrt(1 - (2r/b - 1)^2):
// the force on the fluid per node is -rho V Gamma dr along the lift direction e_L, the relative
// velocity V along e_v (normal to e_L). The inputs are per unit density.
struct Wing {
    std::vector<Real> r, chord, force, vel;
    Real b, Gamma0, V;
    std::array<Real,3> e_L{{0.0, 0.0, 1.0}}, e_v{{1.0, 0.0, 0.0}};
    Wing (int n, Real span, Real gamma0, Real speed, Real c)
        : b(span), Gamma0(gamma0), V(speed)
    {
        for (int i = 0; i < n; ++i) {
            const Real ri = (i + Real(0.5)) / n * b;
            r.push_back(ri);
            chord.push_back(c);
        }
        const std::vector<Real> dr = FLLC::node_widths(r);
        for (int i = 0; i < n; ++i) {
            const Real x = 2.0 * r[i] / b - 1.0;
            const Real gamma = Gamma0 * std::sqrt(std::max(Real(0.0), Real(1.0) - x * x));
            for (int d = 0; d < 3; ++d) {
                force.push_back(-V * gamma * dr[i] * e_L[d]);   // lift per unit span times the node width, on the fluid
                vel.push_back(V * e_v[d]);
            }
        }
    }
};

} // namespace

TEST(FLLC, KernelHasTheRightLimits)
{
    const Real eps = 2.0;
    EXPECT_NEAR(FLLC::kernel(0.0, eps), 0.5 / (eps * eps), tol());
    // small r: 1/(2 eps^2) - 3/4 r^2/eps^4
    const Real r = 0.05;
    EXPECT_NEAR(FLLC::kernel(r, eps), 0.5 / (eps * eps) - 0.75 * r * r / std::pow(eps, 4), 1.0e-5);
    // far away: -1/(2 r^2), the unfiltered vortex sheet
    EXPECT_NEAR(FLLC::kernel(50.0, eps), -0.5 / 2500.0, tol() * 1.0e-3);
    // a rounding residue of a coincident point must give the r = 0 value, not garbage
    EXPECT_NEAR(FLLC::kernel(1.0e-14, eps), 0.5 / (eps * eps), 1.0e-9);
    const Real rel = (sizeof(Real) == 8) ? Real(1.0e-12) : Real(1.0e-6);   // a few roundoffs of the arithmetic
    EXPECT_NEAR(FLLC::kernel(3.0e-13, 0.13), 0.5 / (0.13 * 0.13), rel * 0.5 / (0.13 * 0.13));
    // the kernel is the derivative of (1 - exp(-r^2/eps^2)) / r over 2
    const Real h = 1.0e-4, r0 = 1.3;
    auto F = [&](Real x) { return (1.0 - std::exp(-x * x / (eps * eps))) / x; };
    // the difference quotient loses about roundoff / h, 1e-3 of the value in single precision
    EXPECT_NEAR(FLLC::kernel(r0, eps), 0.5 * (F(r0 + h) - F(r0 - h)) / (2 * h), (sizeof(Real) == 8) ? 1.0e-6 : 5.0e-4);
}

TEST(FLLC, InterpolationAlongTheSpanIsLinearAndHoldsTheEnds)
{
    const std::vector<Real> x{1.0, 2.0, 4.0};
    const std::vector<Real> v{1.0, 10.0, 100.0,  2.0, 20.0, 200.0,  4.0, 40.0, 400.0};   // linear in x
    std::vector<Real> out;
    erf_actuator::interpolate_along(x, v, {0.0, 1.5, 3.0, 4.0, 9.0}, out);
    ASSERT_EQ(out.size(), 15u);
    EXPECT_NEAR(out[0], 1.0, tol()); EXPECT_NEAR(out[2], 100.0, tol());     // held below the range
    EXPECT_NEAR(out[3], 1.5, tol()); EXPECT_NEAR(out[4], 15.0, tol());
    EXPECT_NEAR(out[6], 3.0, tol()); EXPECT_NEAR(out[8], 300.0, tol());
    EXPECT_NEAR(out[9], 4.0, tol());
    EXPECT_NEAR(out[12], 4.0, tol()); EXPECT_NEAR(out[14], 400.0, tol());   // held above the range
    // the fine grid keeps its spacing at or below epsilon_opt / eps_dr and ends at the tip
    const std::vector<Real> r{0.0, 10.0, 20.0}, eps{2.0, 1.0, 0.5};
    const auto rf = FLLC::fine_grid(r, eps, 1.0);
    EXPECT_EQ(rf.front(), 0.0);
    EXPECT_EQ(rf.back(), 20.0);
    for (std::size_t j = 1; j < rf.size(); ++j) {
        EXPECT_GT(rf[j], rf[j-1]);
        const Real e_lo = (rf[j-1] < 10.0) ? 2.0 - 0.1 * rf[j-1] : 1.0 - 0.05 * (rf[j-1] - 10.0);   // epsilon_opt at the left point
        EXPECT_LE(rf[j] - rf[j-1], e_lo + 1.0e-9) << "segment " << j;
    }
    EXPECT_GT(static_cast<int>(rf.size()), 16);   // at least sum(dr / eps) ~ 10/1.5 + 10/0.75
}

TEST(FLLC, EllipticLiftGivesTheUniformDownwashInTheNarrowKernelLimit)
{
    // b = 100 m, Gamma0 = 40 m^2/s: the classical downwash is Gamma0 / (2 b) = 0.2 m/s, opposite
    // to the lift. With a kernel of 1 m (1 % of the span) the filtered result is within a few
    // per cent of it over the middle of the span.
    const Wing w(200, 100.0, 40.0, 10.0, 4.0);
    const std::vector<Real> dr = FLLC::node_widths(w.r);
    // G / |v| with G the lift per unit span per unit density: -Gamma e_L
    std::vector<Real> g(3 * w.r.size());
    for (std::size_t i = 0; i < w.r.size(); ++i) {
        for (int d = 0; d < 3; ++d) { g[3*i+d] = w.force[3*i+d] / (2 * dr[i]) / w.V; }
    }
    const std::vector<Real> eps(w.r.size(), 1.0);
    std::vector<Real> u;
    FLLC::induced_velocity(w.r, w.r, g, std::vector<Real>(dr.begin(), dr.end()), eps, u);
    // node_widths are half the neighbour distance; the integration weights are the full node
    // widths, so induced_velocity was given dr: rescale by 2 here for the check
    const Real expected = -w.Gamma0 / (2 * w.b);
    for (std::size_t i = 60; i < 140; ++i) {
        EXPECT_NEAR(2 * u[3*i+2], expected, 0.05 * std::abs(expected)) << "node " << i;
        EXPECT_NEAR(u[3*i], 0.0, 1.0e-12);   // nothing along the flow
        EXPECT_NEAR(u[3*i+1], 0.0, 1.0e-12);
    }
    // symmetric about the mid-span, and linear in the lift
    EXPECT_NEAR(u[3*20+2], u[3*179+2], 1.0e-9);
    std::vector<Real> g2(g.size());
    for (std::size_t k = 0; k < g.size(); ++k) { g2[k] = 2 * g[k]; }
    std::vector<Real> u2;
    FLLC::induced_velocity(w.r, w.r, g2, dr, eps, u2);
    EXPECT_NEAR(u2[3*100+2], 2 * u[3*100+2], 1.0e-9);
}

TEST(FLLC, CorrectionVanishesAtTheOptimalKernelAndIsADownwashForAWiderOne)
{
    const Real chord = 4.0, eps_chord = 0.25;   // optimal kernel 1 m
    const Wing w(50, 100.0, 40.0, 10.0, chord);
    // the run's kernel equal to the optimal one: the two induced velocities coincide exactly
    FLLC same("b1", w.r, w.chord, eps_chord * chord, eps_chord, 1.0, 1.0);
    same.update(w.force, w.vel);
    EXPECT_EQ(same.max_correction(), Real(0.0));
    // a kernel of 10 m: the wide kernel under-predicts the downwash, so the correction points
    // against the lift (negative along e_L) over the span, and it is larger where the lift is
    FLLC wide("b1", w.r, w.chord, 10.0, eps_chord, 1.0, 1.0);
    wide.update(w.force, w.vel);
    const auto& du = wide.correction();
    Real interior_max = 0.0;
    for (std::size_t i = 5; i + 5 < w.r.size(); ++i) {
        EXPECT_LT(du[3*i+2], 0.0) << "node " << i;
        EXPECT_NEAR(du[3*i], 0.0, 1.0e-12);
        interior_max = std::max(interior_max, std::abs(du[3*i+2]));
    }
    // over the span the correction is a fraction of the downwash itself (0.2 m/s); at the tip
    // nodes, where the discrete elliptic circulation drops to zero within one node, the narrow
    // optimal kernel resolves a strong tip vortex and the correction is largest
    EXPECT_GT(interior_max, 0.01);
    EXPECT_LT(interior_max, 0.3);
    EXPECT_GT(std::abs(du[2]), interior_max);
    EXPECT_GT(std::abs(du[3*(w.r.size()-1)+2]), interior_max);
    EXPECT_EQ(wide.max_correction(), std::max(std::abs(du[2]), std::abs(du[3*(w.r.size()-1)+2])));
    EXPECT_GT(wide.num_fine_points(), wide.num_points());
    EXPECT_EQ(static_cast<int>(wide.target().size()), 3 * wide.num_points());
    // relaxation: with f = 0.1 and a fixed input, du after n updates is (1 - 0.9^n) of the target
    FLLC relaxed("b1", w.r, w.chord, 10.0, eps_chord, 1.0, 0.1);
    for (int n = 1; n <= 5; ++n) {
        relaxed.update(w.force, w.vel);
        const Real want = (1.0 - std::pow(0.9, n)) * relaxed.target()[3*25+2];
        EXPECT_NEAR(relaxed.correction()[3*25+2], want, 1.0e-12 + ((sizeof(Real) == 8) ? 1.0e-9 : 1.0e-5) * std::abs(want))
            << "update " << n;
    }
    // the state round trip restores the relaxed correction; a different blade is refused by name
    std::stringstream ss;
    relaxed.write_state(ss);
    FLLC fresh("b1", w.r, w.chord, 10.0, eps_chord, 1.0, 0.1);
    EXPECT_EQ(fresh.max_correction(), Real(0.0));
    ASSERT_TRUE(fresh.read_state(ss));
    for (std::size_t k = 0; k < fresh.correction().size(); ++k) {
        EXPECT_DOUBLE_EQ(fresh.correction()[k], relaxed.correction()[k]);
    }
    std::stringstream empty;
    EXPECT_FALSE(fresh.read_state(empty));
    // the geometry travels with the state: a blade built from slightly different node positions
    // takes over the written span coordinates and chords, so its grids equal the writer's
    std::vector<Real> r_shift(w.r);
    for (auto& x : r_shift) { x += 1.0e-6; }
    FLLC shifted("b1", r_shift, w.chord, 10.0, eps_chord, 1.0, 0.1);
    EXPECT_NE(shifted.span()[3], relaxed.span()[3]);
    std::stringstream ss2;
    relaxed.write_state(ss2);
    ASSERT_TRUE(shifted.read_state(ss2));
    EXPECT_EQ(shifted.span()[3], relaxed.span()[3]);
    EXPECT_EQ(shifted.num_fine_points(), relaxed.num_fine_points());
    EXPECT_EQ(shifted.fine_span().back(), relaxed.fine_span().back());
}
