#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <ERF_Constants.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>
#include <limits>
#include <string>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

#include "ERF_NumericalSchemes.H"
#include "ERF_Reinitialize.H"
#include "ERF_LevelSetAdvection.H"

/**
 * @file ERF_GTestLevelSetAdvection.cpp
 * @brief The level-set path of the fire module on a fire grid: the Godunov
 *        norm of the gradient with the first-order and HJ-WENO5-Z one-sided
 *        derivatives, the terrain projection and the viscosity term of the
 *        right-hand side, one SSP-RK3 step, an expanding disc over many steps
 *        with the production reinitialisation, and the reinitialisation
 *        itself (unit gradient restored, zero contour kept).
 *
 * Every field lives on a 400 m square fire grid of 80 x 80 cells (dx = 5 m),
 * non-periodic, split into four boxes so that the ghost exchange and the
 * copy-out fill of fire_fill_boundary are both exercised. A planar front
 * phi = x - x0 gives exact expectations (its one-sided differences are all
 * one), the disc phi = r - R0 gives the geometric ones.
 */

using namespace amrex;
using namespace fire_levelset;

namespace {

constexpr int  NCELL = 80;
constexpr Real LDOM  = 400.0;
constexpr Real DX    = LDOM / NCELL;   // 5 m

struct FireGridFixture
{
    BoxArray            ba;
    DistributionMapping dm;
    Geometry            geom;

    FireGridFixture ()
    {
        Box domain(IntVect(0, 0, 0), IntVect(NCELL - 1, NCELL - 1, 0));
        ba = BoxArray(domain);
        ba.maxSize(IntVect(NCELL / 2, NCELL / 2, 1));
        dm = DistributionMapping(ba);
        RealBox rb({0.0, 0.0, 0.0}, {LDOM, LDOM, 1.0});
        geom = Geometry(domain, rb, CoordSys::cartesian, {0, 0, 0});
    }

    Real xc (int i) const { return (i + Real(0.5)) * DX; }

    /// phi = scale * (x - x0): a straight front normal to x, burned on the left
    void planar (MultiFab& phi, Real x0, Real scale = 1.0) const
    {
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            auto p = phi.array(mfi);
            const Box& gbx = mfi.growntilebox();
            ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                p(i, j, k) = scale * ((i + Real(0.5)) * DX - x0);
            });
        }
    }

    /// phi = scale * (r - R0): a disc of radius R0 about the domain centre
    void disc (MultiFab& phi, Real R0, Real scale = 1.0) const
    {
        const Real c = Real(0.5) * LDOM;
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            auto p = phi.array(mfi);
            const Box& gbx = mfi.growntilebox();
            ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real x = (i + Real(0.5)) * DX - c;
                const Real y = (j + Real(0.5)) * DX - c;
                p(i, j, k) = scale * (std::sqrt(x * x + y * y) - R0);
            });
        }
    }

    Real radius (int i, int j) const
    {
        const Real x = xc(i) - Real(0.5) * LDOM;
        const Real y = xc(j) - Real(0.5) * LDOM;
        return std::sqrt(x * x + y * y);
    }
};

/// Number of valid cells with phi < 0, summed over ranks
amrex::Long burned_cells (const MultiFab& phi)
{
    amrex::Long n = 0;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if (p(i, j, 0) < Real(0.0)) { ++n; }
            }
        }
    }
    ParallelDescriptor::ReduceLongSum(n);
    return n;
}

/// Largest |a - b| over the valid cells selected by keep(i, j), over ranks
template <class Keep>
Real max_abs_diff (const MultiFab& a, const MultiFab& b, Keep&& keep)
{
    Real m = 0.0;
    for (MFIter mfi(a); mfi.isValid(); ++mfi) {
        auto pa = a.const_array(mfi);
        auto pb = b.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if (keep(i, j)) { m = amrex::max(m, std::abs(pa(i, j, 0) - pb(i, j, 0))); }
            }
        }
    }
    ParallelDescriptor::ReduceRealMax(m);
    return m;
}

LevelSetGradient scheme (int s, Real band = -1.0)
{
    LevelSetGradient g;
    g.scheme = s;
    g.band   = band;
    return g;
}

/// Number of valid cells holding a NaN or an infinity, over ranks. The
/// max-based checks skip NaN (a comparison with NaN is false), so every field
/// that a kernel wrote is checked here as well.
amrex::Long nonfinite_cells (const MultiFab& a)
{
    amrex::Long n = 0;
    for (MFIter mfi(a); mfi.isValid(); ++mfi) {
        auto pa = a.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if (!std::isfinite(pa(i, j, 0))) { ++n; }
            }
        }
    }
    ParallelDescriptor::ReduceLongSum(n);
    return n;
}

/// Number of valid cells whose sign differs between a and b, over ranks
amrex::Long sign_changes (const MultiFab& a, const MultiFab& b)
{
    amrex::Long n = 0;
    for (MFIter mfi(a); mfi.isValid(); ++mfi) {
        auto pa = a.const_array(mfi);
        auto pb = b.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if ((pa(i, j, 0) < Real(0.0)) != (pb(i, j, 0) < Real(0.0))) { ++n; }
            }
        }
    }
    ParallelDescriptor::ReduceLongSum(n);
    return n;
}

} // namespace

/// On phi = x - x0 every one-sided difference is exactly one, so the RHS with
/// R = 1 and no viscosity is -1 with either scheme. The three cells next to
/// each x edge are excluded: the copy-out ghost fill puts a kink there that
/// the WENO stencil reads. Along y the field is constant, so no edge effect.
TEST(LevelSetAdvection, PlanarGradientNormIsOne)
{
    FireGridFixture f;
    MultiFab phi(f.ba, f.dm, 1, 3), R(f.ba, f.dm, 1, 0), rhs(f.ba, f.dm, 1, 0), ref(f.ba, f.dm, 1, 0);
    f.planar(phi, 200.0);
    R.setVal(1.0);
    ref.setVal(-1.0);
    const Real tol = 100.0 * TOL;
    for (int s : {LEVELSET_GRAD_UPWIND1, LEVELSET_GRAD_WENO5Z, LEVELSET_GRAD_WENO5Z_FRONT}) {
        compute_levelset_rhs(rhs, phi, R, DX, DX, 0.0, nullptr, nullptr, false, scheme(s, 3.0 * DX));
        // along y the stencil is constant: every difference zero, which the
        // WENO weights must survive in either precision
        EXPECT_EQ(nonfinite_cells(rhs), 0) << "scheme " << s;
        const Real err = max_abs_diff(rhs, ref, [] (int i, int) { return i >= 3 && i < NCELL - 3; });
        EXPECT_LT(err, tol) << "scheme " << s;
    }
    // WENO only within the band about x0, first order elsewhere: away from the
    // front the hybrid reads one cell each side, so only the low-x edge cell
    // (whose backward difference is zero through the copied ghost) differs
    compute_levelset_rhs(rhs, phi, R, DX, DX, 0.0, nullptr, nullptr, false, scheme(LEVELSET_GRAD_WENO5Z_FRONT, 3.0 * DX));
    const Real err = max_abs_diff(rhs, ref, [] (int i, int) { return i >= 1; });
    EXPECT_LT(err, tol);
}

/// With terrain slopes the spread rate is along the ground: the map-view
/// |grad phi| of a planar front normal to x on a slope s_x is 1/sqrt(1+s_x^2),
/// and a slope across the front (s_y) does not enter.
TEST(LevelSetAdvection, TerrainProjectionOfPlanarFront)
{
    FireGridFixture f;
    MultiFab phi(f.ba, f.dm, 1, 3), R(f.ba, f.dm, 1, 0), rhs(f.ba, f.dm, 1, 0), ref(f.ba, f.dm, 1, 0);
    MultiFab slopes(f.ba, f.dm, 2, 0);
    f.planar(phi, 200.0);
    R.setVal(1.0);
    auto keep = [] (int i, int) { return i >= 3 && i < NCELL - 3; };

    slopes.setVal(0.75, 0, 1);   // dz/dx = 0.75: 1 + s^2 = 25/16
    slopes.setVal(0.0,  1, 1);
    ref.setVal(-0.8);
    compute_levelset_rhs(rhs, phi, R, DX, DX, 0.0, &slopes, nullptr, false, scheme(LEVELSET_GRAD_UPWIND1));
    EXPECT_LT(max_abs_diff(rhs, ref, keep), 100.0 * TOL);

    slopes.setVal(0.0, 0, 1);
    slopes.setVal(2.0, 1, 1);    // a slope along the front leaves |grad phi| alone
    ref.setVal(-1.0);
    compute_levelset_rhs(rhs, phi, R, DX, DX, 0.0, &slopes, nullptr, false, scheme(LEVELSET_GRAD_WENO5Z));
    EXPECT_LT(max_abs_diff(rhs, ref, keep), 100.0 * TOL);
}

/// The viscosity term is -R eps lap(phi): nothing on a planar front, and
/// R eps / r on the disc (lap r = 1/r in two dimensions), to the accuracy of
/// the central difference well away from the centre and the edges.
TEST(LevelSetAdvection, ViscosityTermIsLaplacian)
{
    FireGridFixture f;
    MultiFab phi(f.ba, f.dm, 1, 3), R(f.ba, f.dm, 1, 0), rhs0(f.ba, f.dm, 1, 0), rhs1(f.ba, f.dm, 1, 0);
    R.setVal(0.5);
    const Real eps = 0.4;

    f.planar(phi, 200.0);
    compute_levelset_rhs(rhs0, phi, R, DX, DX, 0.0, nullptr, nullptr, false, scheme(LEVELSET_GRAD_UPWIND1));
    compute_levelset_rhs(rhs1, phi, R, DX, DX, eps, nullptr, nullptr, false, scheme(LEVELSET_GRAD_UPWIND1));
    EXPECT_LT(max_abs_diff(rhs0, rhs1, [] (int i, int) { return i >= 1 && i < NCELL - 1; }), 100.0 * TOL);

    f.disc(phi, 50.0);
    compute_levelset_rhs(rhs0, phi, R, DX, DX, 0.0, nullptr, nullptr, false, scheme(LEVELSET_GRAD_UPWIND1));
    compute_levelset_rhs(rhs1, phi, R, DX, DX, eps, nullptr, nullptr, false, scheme(LEVELSET_GRAD_UPWIND1));
    Real worst = 0.0;
    for (MFIter mfi(rhs0); mfi.isValid(); ++mfi) {
        auto a = rhs0.const_array(mfi);
        auto b = rhs1.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                const Real r = f.radius(i, j);
                if (r < 40.0 || r > 150.0) { continue; }
                const Real expected = Real(0.5) * eps / r;
                worst = amrex::max(worst, std::abs((b(i, j, 0) - a(i, j, 0)) / expected - Real(1.0)));
            }
        }
    }
    ParallelDescriptor::ReduceRealMax(worst);
    EXPECT_LT(worst, 0.02) << "relative error of the Laplacian of r";
}

/// One SSP-RK3 step moves a planar front by exactly R dt: every cell of
/// phi = x - x0 drops by R dt, up to round-off. Cells within a few stencils
/// of the x edges see the ghost kink propagate one stencil per stage and are
/// left out. With R = 0 the field is returned untouched.
TEST(LevelSetAdvection, PlanarFrontAdvancesAtRos)
{
    FireGridFixture f;
    MultiFab phi(f.ba, f.dm, 1, 3), vel(f.ba, f.dm, 2, 0), R(f.ba, f.dm, 1, 0), ref(f.ba, f.dm, 1, 3);
    vel.setVal(0.0);
    const Real ros = 0.5, dt = 2.0;
    auto keep = [] (int i, int) { return i >= 12 && i < NCELL - 12; };

    for (int s : {LEVELSET_GRAD_UPWIND1, LEVELSET_GRAD_WENO5Z, LEVELSET_GRAD_WENO5Z_FRONT}) {
        f.planar(phi, 200.0);
        f.planar(ref, 200.0 + ros * dt);
        R.setVal(ros);
        advect_levelset_weno5z_rk3(phi, vel, R, f.geom, dt, 0.0, nullptr, nullptr, false, scheme(s, 3.0 * DX));
        EXPECT_EQ(nonfinite_cells(phi), 0) << "scheme " << s;
        EXPECT_LT(max_abs_diff(phi, ref, keep), 100.0 * TOL) << "scheme " << s;
        // the zero contour moved from x0 to x0 + R dt: one more column burned
        EXPECT_EQ(burned_cells(phi), burned_cells(ref)) << "scheme " << s;
    }

    f.planar(phi, 200.0);
    f.planar(ref, 200.0);
    R.setVal(0.0);
    advect_levelset_weno5z_rk3(phi, vel, R, f.geom, dt, 0.4, nullptr, nullptr, false, scheme(LEVELSET_GRAD_WENO5Z_FRONT, 3.0 * DX));
    EXPECT_LT(max_abs_diff(phi, ref, [] (int, int) { return true; }), 100.0 * TOL);
}

/// A disc of radius 50 m spreading at 0.5 m/s for 40 s, with the production
/// reinitialisation every fifth step, burns the disc of radius 70 m: the
/// burned-cell count grows every step and gives that radius to within one
/// cell. The first-order scheme lags the exact rate by up to dx/(2 sqrt2 r)
/// off the axes, the hybrid WENO scheme by much less.
TEST(LevelSetAdvection, DiscExpandsAtRos)
{
    FireGridFixture f;
    MultiFab phi(f.ba, f.dm, 1, 3), vel(f.ba, f.dm, 2, 0), R(f.ba, f.dm, 1, 0);
    vel.setVal(0.0);
    const Real ros = 0.5, dt = 1.0, R0 = 50.0;
    const int  nsteps = 40;
    const Real r_final = R0 + ros * dt * nsteps;   // 70 m

    for (int s : {LEVELSET_GRAD_UPWIND1, LEVELSET_GRAD_WENO5Z_FRONT}) {
        f.disc(phi, R0);
        R.setVal(ros);
        const LevelSetGradient g = scheme(s, 3.0 * DX);
        amrex::Long n_prev = burned_cells(phi);
        EXPECT_NEAR(std::sqrt(n_prev * DX * DX / PI), R0, 0.5 * DX);
        for (int step = 0; step < nsteps; ++step) {
            advect_levelset_weno5z_rk3(phi, vel, R, f.geom, dt, 0.4, nullptr, nullptr, false, g);
            fire_fill_boundary(phi, f.geom);
            if ((step + 1) % 5 == 0) {
                reinitialize_phi(phi, f.geom, 10, 0.25 * DX, 4.0 * DX);
                fire_fill_boundary(phi, f.geom);
            }
            const amrex::Long n = burned_cells(phi);
            EXPECT_GE(n, n_prev) << "scheme " << s << " step " << step;
            n_prev = n;
        }
        const Real r_est = std::sqrt(n_prev * DX * DX / PI);
        EXPECT_NEAR(r_est, r_final, DX) << "scheme " << s << " burned cells " << n_prev;
        EXPECT_EQ(nonfinite_cells(phi), 0) << "scheme " << s;
    }
}

/// Reinitialisation on the signed-distance path ends with
/// min(phi_out, phi_in) ("fire area can only increase"), so phi can only fall.
/// The gradient is therefore restored where that lowers phi -- outside the
/// front of a field that is too steep (|grad phi| = 1.5), inside it of a field
/// that is too flat (|grad phi| = 0.5) -- and the other side is left exactly as
/// it was. In both cases the zero contour (burned-cell count) does not move,
/// the sign is kept everywhere, phi never rises and the field stays finite.
TEST(LevelSetAdvection, ReinitialisationRestoresUnitGradient)
{
    FireGridFixture f;
    MultiFab phi(f.ba, f.dm, 1, 3), phi0(f.ba, f.dm, 1, 3), R(f.ba, f.dm, 1, 0), rhs(f.ba, f.dm, 1, 0), ref(f.ba, f.dm, 1, 0);
    MultiFab rise(f.ba, f.dm, 1, 0);
    R.setVal(1.0);
    ref.setVal(-1.0);
    const Real R0 = 50.0;
    const int  iters = 40;
    const Real dtau  = 0.25 * DX;
    const LevelSetGradient g = scheme(LEVELSET_GRAD_WENO5Z_FRONT, 3.0 * DX);

    struct Case { Real stretch; bool restore_outside; };
    for (const Case c : {Case{1.5, true}, Case{0.5, false}}) {
        f.disc(phi, R0, c.stretch);
        f.disc(phi0, R0, c.stretch);
        const amrex::Long n0 = burned_cells(phi);

        // Cells more than a cell from the front, on the side the clamp allows
        // to change (restored) and on the side it holds (held).
        auto restored = [&f, R0, c] (int i, int j) {
            const Real d = f.radius(i, j) - R0;
            return c.restore_outside ? (d > DX && d < 3.0 * DX) : (d < -DX && d > -3.0 * DX);
        };
        auto held = [&f, R0, c] (int i, int j) {
            const Real d = f.radius(i, j) - R0;
            return c.restore_outside ? (d < -DX) : (d > DX);
        };

        // |grad phi| = stretch before: the RHS with R = 1 is -stretch off the axes too
        compute_levelset_rhs(rhs, phi, R, DX, DX, 0.0, nullptr, nullptr, false, g);
        EXPECT_GT(max_abs_diff(rhs, ref, restored), 0.4) << "stretch " << c.stretch;

        reinitialize_phi(phi, f.geom, iters, dtau, 4.0 * DX);
        fire_fill_boundary(phi, f.geom);

        compute_levelset_rhs(rhs, phi, R, DX, DX, 0.0, nullptr, nullptr, false, g);
        EXPECT_LT(max_abs_diff(rhs, ref, restored), 0.1) << "stretch " << c.stretch;
        // the side the clamp holds is untouched
        EXPECT_EQ(max_abs_diff(phi, phi0, held), Real(0.0)) << "stretch " << c.stretch;

        MultiFab::Copy(rise, phi, 0, 0, 1, 0);
        MultiFab::Subtract(rise, phi0, 0, 0, 1, 0);
        EXPECT_LE(rise.max(0), Real(0.0)) << "phi rose somewhere, stretch " << c.stretch;
        EXPECT_EQ(burned_cells(phi), n0) << "stretch " << c.stretch;
        // the sign is kept everywhere, not only at the front
        EXPECT_EQ(sign_changes(phi, phi0), 0) << "stretch " << c.stretch;
        EXPECT_EQ(nonfinite_cells(phi), 0) << "stretch " << c.stretch;
    }
}

/**
 * At the production setting (erf.fire.levelset.reinit_dtau auto = 0.01 dx,
 * reinit_iters = 1, WRF-Fire's own) one call moves phi by at most 0.01 dx S (1
 * - |grad phi|), and the correction of the gradient spreads from the front at
 * unit pseudo-speed, 0.01 dx per call: a cell a few cells from the front
 * keeps its stretched gradient. It is a nudge, which the doc now says, not a
 * restoration of the signed distance; the test above restores it with dtau =
 * 0.25 dx over 40 outer steps.
 */
TEST(LevelSetAdvection, ReinitialisationAtTheProductionStepIsANudge)
{
    FireGridFixture f;
    MultiFab phi(f.ba, f.dm, 1, 3), phi0(f.ba, f.dm, 1, 3), R(f.ba, f.dm, 1, 0), rhs(f.ba, f.dm, 1, 0), ref(f.ba, f.dm, 1, 0);
    R.setVal(1.0);
    ref.setVal(-1.0);
    const Real R0 = 50.0, stretch = 1.5;
    const LevelSetGradient g = scheme(LEVELSET_GRAD_WENO5Z_FRONT, 3.0 * DX);
    auto band = [&f, R0] (int i, int j) { const Real d = f.radius(i, j) - R0; return d > DX && d < 3.0 * DX; };

    f.disc(phi, R0, stretch);
    f.disc(phi0, R0, stretch);
    compute_levelset_rhs(rhs, phi, R, DX, DX, 0.0, nullptr, nullptr, false, g);
    const Real before = max_abs_diff(rhs, ref, band);   // about stretch - 1
    EXPECT_NEAR(before, stretch - 1.0, 0.05);

    reinitialize_phi(phi, f.geom, 1, 0.01 * DX, 4.0 * DX);
    fire_fill_boundary(phi, f.geom);
    compute_levelset_rhs(rhs, phi, R, DX, DX, 0.0, nullptr, nullptr, false, g);
    const Real after = max_abs_diff(rhs, ref, band);
    EXPECT_NEAR(after, before, 0.02 * before) << "the band's gradient error is unchanged to a few percent";
    MultiFab moved(f.ba, f.dm, 1, 0);
    MultiFab::Copy(moved, phi, 0, 0, 1, 0);
    MultiFab::Subtract(moved, phi0, 0, 0, 1, 0);
    EXPECT_LE(moved.norm0(), 0.01 * DX * (stretch - 1.0) * 1.0001) << "|dphi| <= dtau (|grad phi| - 1)";
    EXPECT_EQ(burned_cells(phi), burned_cells(phi0));
}

#include <ERF_FireIgnition.H>

/**
 * erf.fire.ignition_r = 0 means "no disc": a scheduled, polygon or threshold
 * ignition starts the fire. On the level-set path the initialiser wrote the
 * distance to the ignition point, a disc of zero radius that the advection
 * grew from the first step (the idealized Marshall decks burned 2 cells at
 * 150 s of a 1200 s spin-up); it must leave every cell unburned, at least
 * the domain diagonal from the front.
 */
TEST(LevelSetAdvection, ANoDiscIgnitionLeavesTheLevelSetUnburned)
{
    Box domain(IntVect(0, 0, 0), IntVect(19, 19, 0));
    BoxArray ba(domain);
    DistributionMapping dm(ba);
    Geometry geom(domain, RealBox(0.0, 0.0, 0.0, 200.0, 200.0, 1.0), CoordSys::cartesian, {0, 0, 0});
    MultiFab phi(ba, dm, 1, 1);
    initialize_ignition(phi, geom, 100.0_rt, 100.0_rt, 0.0_rt, /*normalized=*/ false);
    const Real diag = std::sqrt(Real(200.0) * Real(200.0) + Real(200.0) * Real(200.0));
    // within the precision-aware TOL: the float sqrt sits 7e-8 below the double one
    EXPECT_GE(phi.min(0), diag * (Real(1.0) - TOL)) << "no disc: every cell at least the domain diagonal from a front (the distance to the point until 2026-10)";
    EXPECT_NEAR(phi.max(0), diag, diag * TOL) << "and the field is uniform";
    // the advection and the reinitialisation keep it there: 20 substeps of the
    // production scheme at R = 1 m/s with a reinitialisation every fifth,
    // nothing burns and the field is the diagonal to the bit (the reinit's
    // +dtau rise is removed by its min clamp)
    {
        MultiFab vel(ba, dm, 2, 0), R(ba, dm, 1, 0);
        vel.setVal(0.0_rt); R.setVal(1.0_rt);
        fire_fill_boundary(phi, geom);
        const Real dx10 = 10.0_rt;
        for (int step = 0; step < 20; ++step) {
            advect_levelset_weno5z_rk3(phi, vel, R, geom, 0.4_rt * dx10, 0.4_rt, nullptr, nullptr, false,
                                       scheme(LEVELSET_GRAD_WENO5Z_FRONT, 3.0 * dx10));
            fire_fill_boundary(phi, geom);
            if ((step + 1) % 5 == 0) {
                reinitialize_phi(phi, geom, 1, 0.01 * dx10, 4.0 * dx10);
                fire_fill_boundary(phi, geom);
            }
        }
        EXPECT_EQ(nonfinite_cells(phi), 0);
        EXPECT_NEAR(phi.min(0), diag, diag * TOL) << "no cell moved off the diagonal (a zero-radius disc burned by now)";
        EXPECT_NEAR(phi.max(0), diag, diag * TOL);
    }
    initialize_ignition(phi, geom, 100.0_rt, 100.0_rt, 0.0_rt, /*normalized=*/ true);
    EXPECT_NEAR(phi.min(0), 1.0, 1.0e-12) << "the FARSITE indicator stays +1";
    EXPECT_NEAR(phi.max(0), 1.0, 1.0e-12);
    initialize_ignition(phi, geom, 100.0_rt, 100.0_rt, 30.0_rt, /*normalized=*/ false);
    EXPECT_LT(phi.min(0), 0.0) << "a disc still burns";
}

#include <ERF_PolygonIgnition.H>

/**
 * On a fire grid wider than 100 km the no-disc start must still be the full
 * domain diagonal. A cap there (100 km in the first form of the fix above)
 * left every cell beyond it at the same value once a polygon's distance field
 * was min-merged over it: a flat plateau, which the upwind scheme still
 * crosses, but late. Copilot's review of hgopalan/ERF#501 found it.
 */
TEST(LevelSetAdvection, ANoDiscStartIsTheFullDiagonalOnAWideGrid)
{
    // 600 km on 20 cells of 30 km: the diagonal is 848.5 km
    Box domain(IntVect(0, 0, 0), IntVect(19, 19, 0));
    BoxArray ba(domain);
    DistributionMapping dm(ba);
    const Real L = 6.0e5;
    Geometry geom(domain, RealBox(0.0, 0.0, 0.0, L, L, 1.0), CoordSys::cartesian, {0, 0, 0});
    MultiFab phi(ba, dm, 1, 1);
    initialize_ignition(phi, geom, 0.5_rt * L, 0.5_rt * L, 0.0_rt, /*normalized=*/ false);
    const Real diag = std::sqrt(Real(2.0)) * L;
    EXPECT_NEAR(phi.min(0), diag, diag * TOL) << "no cap: every cell starts the full diagonal from a front (1e5 m with the cap)";
    EXPECT_NEAR(phi.max(0), diag, diag * TOL);
    // a t = 0 polygon (a 200 m square at the south-west corner) min-merged
    // over it by the production stamp: every cell holds its own distance to
    // the square, the farthest the distance from (200, 200) m to the
    // opposite corner cell, not a value shared with its neighbours
    init_phi_from_polygon(phi, geom, {0.0_rt, 200.0_rt, 200.0_rt, 0.0_rt}, {0.0_rt, 0.0_rt, 200.0_rt, 200.0_rt});
    const Real c = L - Real(0.5) * geom.CellSize(0) - Real(200.0);
    const Real far_corner = std::sqrt(Real(2.0)) * c;
    EXPECT_NEAR(phi.max(0), far_corner, far_corner * TOL) << "the merged field is the distance to the polygon in every cell (1e5 m beyond 100 km with the cap)";
}

#include <ERF_FireBreak.H>

namespace {
/// Valid cells with a >= thr, over ranks (a free function: nvcc rejects
/// device lambdas inside gtest's private TestBody).
int count_at_least (const MultiFab& mf, Real thr)
{
    ReduceOps<ReduceOpSum> op; ReduceData<int> data(op);
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto a = mf.const_array(mfi);
        op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> GpuTuple<int> {
            return { a(i,j,k) >= thr ? 1 : 0 };
        });
    }
    int n = amrex::get<0>(data.value());
    ParallelDescriptor::ReduceIntSum(n);
    return n;
}
}  // namespace

/**
 * The non-burnable mask marks the firebreak cells from the barrier shapes
 * (mark_firebreak_cells), the cells apply_firebreaks() stamps the sentinel
 * into. Until 2026-10 it read them back as phi >= half the sentinel, which
 * an unburned level set whose domain diagonal exceeds 500 km also passes:
 * every cell of this 600 km grid would have been non-burnable.
 */
TEST(FireBreak, TheMaskMarksTheBarrierCellsWhateverTheLevelSet)
{
    Box domain(IntVect(0, 0, 0), IntVect(19, 19, 0));
    BoxArray ba(domain);
    ba.maxSize(8);   // several boxes: the test spans box edges
    DistributionMapping dm(ba);
    const Real L = 6.0e5;
    Geometry geom(domain, RealBox(0.0, 0.0, 0.0, L, L, 1.0), CoordSys::cartesian, {0, 0, 0});
    // a 150 km square from 150 to 300 km: the centres of cells 5 to 9 (x = 165 ... 285 km)
    FireBreak brk; brk.type = "rect"; brk.x_lo = 1.5e5; brk.x_hi = 3.0e5; brk.y_lo = 1.5e5; brk.y_hi = 3.0e5;
    const std::vector<FireBreak> breaks{brk};

    MultiFab phi(ba, dm, 1, 1), mask(ba, dm, 1, 0);
    initialize_ignition(phi, geom, 0.5_rt * L, 0.5_rt * L, 0.0_rt, /*normalized=*/ false);
    apply_firebreaks(phi, breaks, geom);
    mask.setVal(0.0_rt);
    mark_firebreak_cells(mask, breaks, geom, 1.0_rt);

    EXPECT_EQ(count_at_least(mask, 0.5_rt), 25) << "the mask is the 5 x 5 barrier cells";
    EXPECT_EQ(count_at_least(phi, FIREBREAK_PHI_SENTINEL), 25) << "the same cells carry the sentinel";
    // the deck CTest FireMergingFronts_firebreak_mask runs the mask through
    // FireLayer::build_nonburnable_mask; this pins the shared cell test
    EXPECT_EQ(count_at_least(phi, 0.5_rt * FIREBREAK_PHI_SENTINEL), 400) << "the former rule (phi >= half the sentinel) marks the whole grid here";
}

namespace {
/// The production level-set scheme of FireLayer (weno5z_front, a 4-cell
/// WENO band, eps 0.1 m within 2 cells of the front blended to 0.4 m over 2
/// more) on a cell size h.
LevelSetGradient production_scheme (Real h)
{
    LevelSetGradient g;
    g.scheme = LEVELSET_GRAD_WENO5Z_FRONT;
    g.band = 4.0_rt * h;
    g.eps_visc_front = 0.1_rt;
    g.visc_d0 = 2.0_rt * h;
    g.visc_d1 = 4.0_rt * h;
    return g;
}

/// phi = the distance to (px, py) on every valid cell (the level-set start
/// before 2026-10 with ignition_r = 0, and the shape of any background far
/// from a front).
void fill_distance (MultiFab& phi, const Geometry& geom, Real px, Real py)
{
    const auto lo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real x = lo[0] + (i + 0.5_rt) * dx[0] - px, y = lo[1] + (j + 0.5_rt) * dx[1] - py;
            p(i,j,k) = std::sqrt(x * x + y * y);
        });
    }
    Gpu::streamSynchronize();
}

/// T = T_hot in the one cell (ih, jh), T_cold elsewhere.
void fill_hot_cell (MultiFab& T, int ih, int jh, Real T_hot, Real T_cold)
{
    for (MFIter mfi(T); mfi.isValid(); ++mfi) {
        auto t = T.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            t(i,j,k) = (i == ih && j == jh) ? T_hot : T_cold;
        });
    }
    Gpu::streamSynchronize();
}

/// phi at cell (i, j), over ranks.
Real value_at (const MultiFab& phi, int i, int j)
{
    Real v = std::numeric_limits<Real>::lowest();
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(IntVect(i, j, 0))) {
            const auto h = phi.const_array(mfi);
#ifdef AMREX_USE_GPU
            Gpu::DeviceScalar<Real> d(0.0_rt);
            Real* dp = d.dataPtr();
            ParallelFor(1, [=] AMREX_GPU_DEVICE (int) noexcept { *dp = h(i, j, 0); });
            v = d.dataValue();
#else
            v = h(i, j, 0);
#endif
        }
    }
    ParallelDescriptor::ReduceRealMax(v);
    return v;
}

/// One threshold ignition of the hot cell at the default radius (one cell,
/// threshold_radius = 0), then nsub substeps of the production scheme at
/// R = 0.1 m/s and CFL 0.4; returns the hot cell's phi after them.
Real hot_cell_after_substeps (MultiFab& phi, const Geometry& geom, int ih, int jh, int nsub)
{
    const Real h = geom.CellSize(0);
    MultiFab T(phi.boxArray(), phi.DistributionMap(), 1, 0);
    fill_hot_cell(T, ih, jh, 400.0_rt, 300.0_rt);
    const amrex::Long n = apply_threshold_ignition(phi, T, nullptr, geom, 350.0_rt, h, /*normalized=*/ false);
    EXPECT_EQ(n, 1) << "the hot cell ignites";
    fire_fill_boundary(phi, geom);
    EXPECT_LT(value_at(phi, ih, jh), 0.0_rt) << "stamped burning";
    MultiFab vel(phi.boxArray(), phi.DistributionMap(), 2, 0), R(phi.boxArray(), phi.DistributionMap(), 1, 0);
    vel.setVal(0.0_rt); R.setVal(0.1_rt);
    for (int s = 0; s < nsub; ++s) {
        advect_levelset_weno5z_rk3(phi, vel, R, geom, 0.4_rt * h / 0.1_rt, 0.4_rt, nullptr, nullptr, false,
                                   production_scheme(h));
        fire_fill_boundary(phi, geom);
    }
    EXPECT_EQ(nonfinite_cells(phi), 0);
    return value_at(phi, ih, jh);
}
}  // namespace

/**
 * A threshold ignition at the default radius stamps the hot cell alone. Its
 * neighbours keep whatever phi they had: the no-disc start (the domain
 * diagonal), or on any grid the distance to a front or ignition point far
 * away. The artificial viscosity of the next substep then reads that jump
 * as a Laplacian and lifts the cell above zero, and the new fire goes out:
 * on 30 m cells over a 14 km background the hot cell went from -30 m to
 * +31.6 m in one substep. On fine cells the viscosity carries the lift in
 * from further out over a few substeps, so the stamp's distance band must
 * be wide in metres, not only in cells (a three-cell band let a 1 m-cell
 * ignition go out within a few substeps).
 */
TEST(LevelSetAdvection, AThresholdIgnitionSurvivesTheNextSubstep)
{
    // 64 cells of 30 m: 1.92 km
    Box domain(IntVect(0, 0, 0), IntVect(63, 63, 0));
    BoxArray ba(domain);
    ba.maxSize(32);
    DistributionMapping dm(ba);
    const Real L = 1920.0;
    Geometry geom(domain, RealBox(0.0, 0.0, 0.0, L, L, 1.0), CoordSys::cartesian, {0, 0, 0});
    MultiFab phi(ba, dm, 1, 3);

    // (a) the no-disc start of a wide grid (14 km, a 10 km square)
    phi.setVal(1.4e4_rt);
    EXPECT_LT(hot_cell_after_substeps(phi, geom, 32, 32, 1), 0.0_rt)
        << "the hot cell still burns after one substep on the no-disc start";
    // and keeps burning while the jump three cells out collapses (20 substeps
    // move a front 8 cells at CFL 0.4)
    phi.setVal(1.4e4_rt);
    EXPECT_LT(hot_cell_after_substeps(phi, geom, 32, 32, 20), 0.0_rt)
        << "the hot cell still burns 20 substeps later";

    // (b) the distance to a point 8 km away (outside the grid): the start
    // before 2026-10, and the shape of the field near a fire far from the hot
    // cell. The one-cell stamp went out here too, which predates the no-disc
    // start.
    fill_distance(phi, geom, 32.5_rt * 30.0_rt - 8000.0_rt, 32.5_rt * 30.0_rt);
    fire_fill_boundary(phi, geom);
    EXPECT_LT(hot_cell_after_substeps(phi, geom, 32, 32, 1), 0.0_rt)
        << "and on a distance field 8 km from its source";

    // (c) 1 m cells under the same 14 km background, 20 substeps: the band
    // must be wide in metres (three cells, 3 m, let it go out)
    Geometry geom1(domain, RealBox(0.0, 0.0, 0.0, 64.0, 64.0, 1.0), CoordSys::cartesian, {0, 0, 0});
    phi.setVal(1.4e4_rt);
    EXPECT_LT(hot_cell_after_substeps(phi, geom1, 32, 32, 20), 0.0_rt)
        << "on 1 m cells the hot cell still burns 20 substeps later";
}

namespace {
/// mask = 1 in the one cell (im, jm), 0 elsewhere.
void fill_one_cell (MultiFab& mask, int im, int jm)
{
    for (MFIter mfi(mask); mfi.isValid(); ++mfi) {
        auto m = mask.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            m(i,j,k) = (i == im && j == jm) ? 1.0_rt : 0.0_rt;
        });
    }
    Gpu::streamSynchronize();
}
}  // namespace

/**
 * A cell of the non-burnable mask next to a hot cell takes no stamp: neither
 * the disc nor the distance band beyond it is written there.
 */
TEST(LevelSetAdvection, AThresholdStampSkipsNonBurnableCells)
{
    Box domain(IntVect(0, 0, 0), IntVect(31, 31, 0));
    BoxArray ba(domain);
    DistributionMapping dm(ba);
    Geometry geom(domain, RealBox(0.0, 0.0, 0.0, 320.0, 320.0, 1.0), CoordSys::cartesian, {0, 0, 0});
    MultiFab phi(ba, dm, 1, 3), T(ba, dm, 1, 0), mask(ba, dm, 1, 0);
    phi.setVal(1.0e3_rt);
    fill_hot_cell(T, 16, 16, 400.0_rt, 300.0_rt);
    fill_one_cell(mask, 17, 16);   // the hot cell's east neighbour, inside the band
    const amrex::Long n = apply_threshold_ignition(phi, T, &mask, geom, 350.0_rt, 10.0_rt, /*normalized=*/ false);
    EXPECT_EQ(n, 1);
    EXPECT_LT(value_at(phi, 16, 16), 0.0_rt) << "the hot cell burns";
    EXPECT_NEAR(value_at(phi, 15, 16), 0.0_rt, 1.0e-6) << "its west neighbour takes the band distance (10 m - 10 m)";
    EXPECT_NEAR(value_at(phi, 17, 16), 1.0e3_rt, 1.0e-6) << "the masked east neighbour is untouched";
}

namespace {
/// phi = s_in (r - R0) inside the disc of radius R0 about the domain centre
/// and r - R0 outside: a front whose burned side is s_in times too steep.
void stretched_disc (MultiFab& phi, Real R0, Real s_in)
{
    const Real c = Real(0.5) * LDOM;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.array(mfi);
        ParallelFor(mfi.growntilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real x = (i + Real(0.5)) * DX - c;
            const Real y = (j + Real(0.5)) * DX - c;
            const Real d = std::sqrt(x * x + y * y) - R0;
            p(i, j, k) = (d < Real(0.0)) ? s_in * d : d;
        });
    }
    Gpu::streamSynchronize();
}

/// One production subcycle of FireLayer::advance with the reinitialisation
/// in it: keep the level set at its start, advect, reinitialise with the
/// final clamp against that start ("step_start") or against the advected
/// field ("post_advection"), and hold the non-burnable cells at zero or
/// above (enforce_nonburnable_phi), with the fills in between. With
/// `reinit` false the subcycle only advects (reinit_every > 1).
void subcycle_with_reinit (MultiFab& phi, MultiFab& phi_start, const MultiFab& R, const Geometry& geom,
                           Real dt, int iters, Real dtau, bool step_start,
                           const MultiFab* mask = nullptr, bool wall_extrap = false,
                           bool jiang_peng = false, bool reinit = true)
{
    MultiFab vel(phi.boxArray(), phi.DistributionMap(), 2, 0);
    vel.setVal(0.0_rt);
    const Real h = geom.CellSize(0);
    LevelSetGradient g;
    g.scheme = LEVELSET_GRAD_WENO5Z_FRONT;
    g.band = 4.0_rt * h;
    MultiFab::Copy(phi_start, phi, 0, 0, 1, 0);
    advect_levelset_weno5z_rk3(phi, vel, R, geom, dt, 0.0_rt, nullptr, mask, wall_extrap, g);
    hold_phi_in_mask(phi, mask, 0.0_rt);
    fire_fill_boundary(phi, geom);
    if (!reinit) { return; }
    if (jiang_peng) {
        reinitialize_phi_jiang_peng(phi, geom, iters, dtau, mask, wall_extrap, -1.0_rt, nullptr,
                                    step_start ? &phi_start : nullptr);
    } else {
        reinitialize_phi(phi, geom, iters, dtau, 4.0_rt * h, mask, wall_extrap, nullptr,
                         step_start ? &phi_start : nullptr);
    }
    hold_phi_in_mask(phi, mask, 0.0_rt);
    fire_fill_boundary(phi, geom);
}

/// Mean |grad phi| (the Godunov norm of the production scheme, which reads
/// the burned side of an expanding front) over the valid cells whose
/// distance from the domain centre is in (r_lo, r_hi), over ranks.
Real mean_grad_in_ring (const MultiFab& phi, const FireGridFixture& f, Real r_lo, Real r_hi)
{
    MultiFab R(f.ba, f.dm, 1, 0), rhs(f.ba, f.dm, 1, 0);
    R.setVal(1.0_rt);
    LevelSetGradient g;
    g.scheme = LEVELSET_GRAD_WENO5Z_FRONT;
    g.band = 4.0_rt * DX;
    compute_levelset_rhs(rhs, phi, R, DX, DX, 0.0_rt, nullptr, nullptr, false, g);
    Gpu::streamSynchronize();
    Real sum = 0.0;
    amrex::Long n = 0;
    for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        auto r = rhs.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                const Real d = f.radius(i, j);
                if (d > r_lo && d < r_hi) { sum -= r(i, j, 0); ++n; }
            }
        }
    }
    ParallelDescriptor::ReduceRealSum(sum);
    ParallelDescriptor::ReduceLongSum(n);
    return (n > 0) ? sum / Real(n) : Real(0.0);
}

/// mask = 1 in the cells i0..i1 x j0..j1, ghost cells included, 0 elsewhere
/// (a free function: nvcc rejects device lambdas inside gtest's private
/// TestBody).
void fill_block (MultiFab& mask, int i0, int i1, int j0, int j1)
{
    for (MFIter mfi(mask); mfi.isValid(); ++mfi) {
        auto m = mask.array(mfi);
        ParallelFor(mfi.growntilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            m(i, j, k) = (i >= i0 && i <= i1 && j >= j0 && j <= j1) ? 1.0_rt : 0.0_rt;
        });
    }
    Gpu::streamSynchronize();
}

/// x of the zero crossing of phi along row j (linear between the two cells
/// that bracket it), over ranks; -1 when the row has none.
Real zero_crossing_x (const MultiFab& phi, int j)
{
    Real x = -1.0;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.const_array(mfi);
        const Box& bx = mfi.validbox();
        if (j < bx.smallEnd(1) || j > bx.bigEnd(1)) { continue; }
        // the crossing between i and i + 1 is found by the box holding i
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0) && i + 1 < NCELL; ++i) {
            const Real a = p(i, j, 0), b = p(i + 1, j, 0);
            if (a < Real(0.0) && b >= Real(0.0)) { x = (i + Real(0.5)) * DX + DX * a / (a - b); }
        }
    }
    ParallelDescriptor::ReduceRealMax(x);
    return x;
}
} // namespace

/**
 * erf.fire.levelset.reinit_clamp: the burned side of a disc stretched to
 * |grad phi| = 3 (as a falling rate of spread leaves it behind a front).
 * WRF-Fire clamps the reinitialisation against the level set at the start
 * of the step ("step_start", the default), so the reinitialisation may lift
 * phi by the step's own advance and the burned side relaxes toward
 * |grad phi| = 1 behind the moving front. Clamping against the advected
 * field ("post_advection", ERF's form until 2026-10) forbids any rise, and
 * the burned side keeps |grad phi| = 3 however strong the reinitialisation.
 * 30 subcycles at R = 0.5 m/s and dt = 2 s (cfl 0.2) move the front 30 m;
 * each reinitialises with 2 iterations of dtau = 0.25 dx.
 */
TEST(LevelSetAdvection, ReinitClampStepStartRelaxesASteepBurnedSide)
{
    FireGridFixture f;
    MultiFab phi(f.ba, f.dm, 1, 3), phi_start(f.ba, f.dm, 1, 0), R(f.ba, f.dm, 1, 0);
    R.setVal(0.5_rt);
    const Real R0 = 50.0, ros = 0.5, dt = 2.0;
    const int  nsteps = 30;
    Real grad_after[2] = {0.0, 0.0};
    Real r_front[2]    = {0.0, 0.0};
    for (int step_start = 0; step_start <= 1; ++step_start) {
        stretched_disc(phi, R0, 3.0_rt);
        fire_fill_boundary(phi, f.geom);
        EXPECT_NEAR(mean_grad_in_ring(phi, f, R0 - 3.0 * DX, R0 - DX), 3.0, 0.1) << "the start is stretched";
        for (int s = 0; s < nsteps; ++s) {
            subcycle_with_reinit(phi, phi_start, R, f.geom, dt, 2, 0.25 * DX, step_start == 1);
        }
        EXPECT_EQ(nonfinite_cells(phi), 0);
        r_front[step_start] = std::sqrt(Real(burned_cells(phi)) * DX * DX / PI);
        // the band one to three cells behind the front where it now stands
        grad_after[step_start] = mean_grad_in_ring(phi, f, r_front[step_start] - 3.0 * DX,
                                                   r_front[step_start] - DX);
    }
    // post_advection: nothing on the burned side may rise, so it stays
    // steep (2.53 measured; the slope-3 region travels with the front)
    EXPECT_GT(grad_after[0], 2.0) << "post_advection, front at r = " << r_front[0] << " m";
    // step_start: lifted to the signed distance (1.00002 measured)
    EXPECT_LT(grad_after[1], 1.1) << "step_start, front at r = " << r_front[1] << " m";
    // the step_start front is at R0 + R t = 80 m to the resolution of the
    // burned-cell count (80.8 m measured; post_advection 83.7 m, see the
    // next test)
    EXPECT_NEAR(r_front[1], R0 + ros * dt * nsteps, DX);
}

/**
 * A straight front with |grad phi| = 3 on both sides, R = 0.5 m/s, and a
 * strong reinitialisation (2 iterations of dtau = 0.25 dx) after every
 * subcycle, with both reinitialisation schemes. The exact front moves at
 * R, 1 m per subcycle. With the clamp against the start of the step both
 * sides relax to the signed distance within the first ten subcycles, which
 * leaves the zero crossing a fixed 0.36 m ("wrf") or 0.48 m ("jiang_peng")
 * ahead, and from then on it moves at R exactly (0.002 m and 0.000 m
 * gained between subcycles 20 and 40). With the clamp against the advected
 * field only the unburned side can be lowered, every correction lowers the
 * cells just ahead of the front, and the front keeps running ahead: 4.0 m
 * ("wrf") and 4.5 m ("jiang_peng") after 40 subcycles, 0.95 m and 1.15 m of
 * it gained between subcycles 20 and 40 (5 % of R).
 */
TEST(LevelSetAdvection, ReinitClampStepStartKeepsAStraightFrontAtRos)
{
    FireGridFixture f;
    MultiFab phi(f.ba, f.dm, 1, 3), phi_start(f.ba, f.dm, 1, 0), R(f.ba, f.dm, 1, 0);
    R.setVal(0.5_rt);
    const Real x0 = 200.0, ros = 0.5, dt = 2.0;
    const int  nsteps = 40;
    for (int jp = 0; jp <= 1; ++jp) {
        Real err20[2] = {0.0, 0.0}, err40[2] = {0.0, 0.0};
        for (int step_start = 0; step_start <= 1; ++step_start) {
            f.planar(phi, x0, 3.0);
            EXPECT_NEAR(zero_crossing_x(phi, NCELL / 2), x0, 100.0 * TOL);
            for (int s = 1; s <= nsteps; ++s) {
                subcycle_with_reinit(phi, phi_start, R, f.geom, dt, 2, 0.25 * DX, step_start == 1,
                                     nullptr, false, jp == 1);
                if (s == nsteps / 2) { err20[step_start] = zero_crossing_x(phi, NCELL / 2) - (x0 + ros * dt * s); }
            }
            EXPECT_EQ(nonfinite_cells(phi), 0);
            // every row holds the same front
            const Real xa = zero_crossing_x(phi, NCELL / 2);
            EXPECT_NEAR(zero_crossing_x(phi, 3), xa, 100.0 * TOL * xa);
            err40[step_start] = xa - (x0 + ros * dt * nsteps);
        }
        const std::string tag = (jp == 1) ? "jiang_peng" : "wrf";
        // step_start: a fixed offset from the first relaxation, then R exactly
        EXPECT_LT(std::abs(err40[1]), 0.15 * DX) << "step_start, " << tag << ": front " << err40[1] << " m from R t";
        EXPECT_LT(std::abs(err40[1] - err20[1]), 0.01 * DX) << "step_start, " << tag << ": the front gained "
            << err40[1] - err20[1] << " m on R t between subcycles 20 and 40";
        // post_advection: ahead, and still gaining
        EXPECT_GT(err40[0], 0.5 * DX) << "post_advection, " << tag << ": front only " << err40[0] << " m ahead of R t";
        EXPECT_GT(err40[0] - err20[0], 0.1 * DX) << "post_advection, " << tag << ": the front gained only "
            << err40[0] - err20[0] << " m on R t between subcycles 20 and 40";
    }
}

/**
 * The production default: the reinitialisation every 5th subcycle
 * (reinit_every = 5), with the "step_start" reference the level set at the
 * start of that 5th subcycle, as FireLayer::advance copies it (m_phi_prev,
 * every subcycle, before the advection). The same straight front three times
 * too steep and the same strong reinitialisation as above, 40 subcycles.
 * "step_start" leaves the front a fixed 0.76 m ahead of R t (the first
 * relaxations; 0.36 m when every subcycle reinitialises), "post_advection"
 * 3.16 m. A reference copied only at the first subcycle of each 5-subcycle
 * window, which lets the reinitialisation lift phi by up to five subcycles
 * of advance, gives 0.78 m on this front: too close to tell apart here, so
 * the gtest pins the production window and the FireReinitClamp regression
 * test checks FireLayer's own copy.
 */
TEST(LevelSetAdvection, ReinitClampStepStartEveryFifthSubcycle)
{
    FireGridFixture f;
    MultiFab phi(f.ba, f.dm, 1, 3), phi_start(f.ba, f.dm, 1, 0), R(f.ba, f.dm, 1, 0);
    R.setVal(0.5_rt);
    const Real x0 = 200.0, ros = 0.5, dt = 2.0;
    const int  nsteps = 40, every = 5;
    Real err[2] = {0.0, 0.0};
    for (int step_start = 0; step_start <= 1; ++step_start) {
        f.planar(phi, x0, 3.0);
        for (int s = 1; s <= nsteps; ++s) {
            subcycle_with_reinit(phi, phi_start, R, f.geom, dt, 2, 0.25 * DX, step_start == 1,
                                 nullptr, false, false, s % every == 0);
        }
        EXPECT_EQ(nonfinite_cells(phi), 0);
        err[step_start] = zero_crossing_x(phi, NCELL / 2) - (x0 + ros * dt * nsteps);
    }
    // 0.764 m and 3.161 m measured in double
    EXPECT_NEAR(err[1], 0.764, 0.1) << "step_start, every 5th subcycle";
    EXPECT_GT(err[0] - err[1], 0.4 * DX) << "post_advection runs " << err[0] << " m ahead of R t";
}

/**
 * Non-burnable cells under either clamp: with the wall stencil the
 * reinitialisation passes them through, the rate of spread is zero there
 * (zero_ros_in_mask), so the advection leaves them too, and the clamp's
 * reference holds their own value whichever field it is. A 3 x 3 block
 * ahead of the stretched disc's front, which reaches it during the run,
 * keeps its values to the bit.
 */
TEST(LevelSetAdvection, ReinitClampLeavesNonBurnableCellsUntouched)
{
    FireGridFixture f;
    MultiFab phi(f.ba, f.dm, 1, 3), phi0(f.ba, f.dm, 1, 0), phi_start(f.ba, f.dm, 1, 0);
    MultiFab R(f.ba, f.dm, 1, 0), mask(f.ba, f.dm, 1, 3);
    // cells 51-53 in x and 39-41 in y: x from 255 to 270 m, 55 to 70 m from the centre
    fill_block(mask, 51, 53, 39, 41);
    R.setVal(0.5_rt);
    zero_ros_in_mask(R, &mask);
    auto masked = [] (int i, int j) { return i >= 51 && i <= 53 && j >= 39 && j <= 41; };
    for (int step_start = 0; step_start <= 1; ++step_start) {
        stretched_disc(phi, 50.0_rt, 3.0_rt);
        fire_fill_boundary(phi, f.geom);
        MultiFab::Copy(phi0, phi, 0, 0, 1, 0);
        for (int s = 0; s < 30; ++s) {
            subcycle_with_reinit(phi, phi_start, R, f.geom, 2.0, 2, 0.25 * DX, step_start == 1, &mask, true);
        }
        EXPECT_EQ(nonfinite_cells(phi), 0);
        // to round-off: the advection's RK3 stages recombine 3/4 + 1/4 and
        // 1/3 + 2/3 of an unchanged value (1.8e-15 m measured in double)
        EXPECT_LT(max_abs_diff(phi, phi0, masked), 100.0 * TOL * 70.0) << "step_start " << step_start;
        // the front got there: a neighbour of the block burned
        EXPECT_LT(value_at(phi, 50, 40), 0.0_rt) << "step_start " << step_start;
    }
}

namespace {
/// Growth of a wave cos(theta (i + j)) of amplitude `seed` laid on a planar
/// front of unit normal at angle `ang` to x, R = 1 m/s everywhere, after
/// `nsteps` substeps of the production advection at dt = cfl h / R with
/// viscosity eps and no near-front value: the RMS of the departure from
/// the exact front over the central 40 x 40 cells of a 120 x 120 grid of
/// dx x dy cells, divided by its start. theta = pi is the two-cell wave
/// (-1)^(i+j). Far enough from the edges that their copy-out fill does not
/// reach the measurement in that many substeps.
Real wave_gain (int scheme, Real eps, Real cfl, Real ang, Real dx, Real dy, int nsteps,
                Real theta = PI, Real seed = 1.0e-3)
{
    constexpr int N = 120;
    Box domain(IntVect(0, 0, 0), IntVect(N - 1, N - 1, 0));
    BoxArray ba(domain);
    ba.maxSize(IntVect(N / 2, N / 2, 1));
    DistributionMapping dm(ba);
    Geometry geom(domain, RealBox({0.0, 0.0, 0.0}, {N * dx, N * dy, 1.0}), CoordSys::cartesian, {0, 0, 0});
    MultiFab phi(ba, dm, 1, 3), vel(ba, dm, 2, 0), R(ba, dm, 1, 0);
    vel.setVal(0.0_rt);
    R.setVal(1.0_rt);
    const Real nx = std::cos(ang), ny = std::sin(ang);
    const Real xc = Real(0.5) * N * dx, yc = Real(0.5) * N * dy;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.array(mfi);
        ParallelFor(mfi.growntilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real x = (i + Real(0.5)) * dx - xc, y = (j + Real(0.5)) * dy - yc;
            p(i, j, k) = nx * x + ny * y + seed * std::cos(theta * Real(i + j));
        });
    }
    LevelSetGradient g;
    g.scheme = scheme;
    const Real dt = cfl * amrex::min(dx, dy);
    for (int s = 0; s < nsteps; ++s) {
        advect_levelset_weno5z_rk3(phi, vel, R, geom, dt, eps, nullptr, nullptr, false, g);
        fire_fill_boundary(phi, geom);
    }
    Gpu::streamSynchronize();
    const Real t = nsteps * dt;
    Real s2 = 0.0, s2_0 = 0.0;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if (i < 40 || i >= 80 || j < 40 || j >= 80) { continue; }
                const Real x = (i + Real(0.5)) * dx - xc, y = (j + Real(0.5)) * dy - yc;
                const Real d = p(i, j, 0) - (nx * x + ny * y - t);
                const Real d0 = seed * std::cos(theta * Real(i + j));
                s2 += d * d;
                s2_0 += d0 * d0;
            }
        }
    }
    ParallelDescriptor::ReduceRealSum(s2);
    ParallelDescriptor::ReduceRealSum(s2_0);
    return std::isfinite(s2) ? std::sqrt(s2 / s2_0) : std::numeric_limits<Real>::max();
}
} // namespace

/**
 * The start-up limit on the level-set artificial viscosity
 * (levelset_viscous_stability, checked by verify_levelset_viscous_stability)
 * against the substep itself. At cfl 0.4 on square cells the limit is
 * eps/dx <= 0.4317 with first-order differences and 0.5967 with WENO5, both
 * on a front along the diagonal (the closed form of the two-cell wave gives
 * 0.5352 and 0.6519 on a front facing an axis). The production advection at
 * 0.95 of each limit damps a two-cell wave over 30 substeps and at 1.05 of it
 * grows the wave, on square cells and on cells twice as long in y as in x
 * (whose least favourable front has its normal along (1/dx, 1/dy)). Between
 * the diagonal and the axis limits a diagonal front grows the wave and an
 * axis-facing one damps it: start-up aborts there, since every curved front
 * has a part facing the diagonal.
 */
TEST(LevelSetAdvection, ViscousStabilityLimitIsTheSubstepsLimit)
{
    const Real cfl = 0.4, h = 5.0;
    const Real pi4 = Real(0.25) * PI;
    {
        const LevelSetViscousStability up = levelset_viscous_stability(cfl, 0.0, h, h, false);
        const LevelSetViscousStability w5 = levelset_viscous_stability(cfl, 0.0, h, h, true);
        EXPECT_NEAR(up.eps_axis / h, 0.5352, 1.0e-4);
        EXPECT_NEAR(up.eps_max / h,  0.4317, 1.0e-4);
        EXPECT_NEAR(w5.eps_axis / h, 0.6519, 1.0e-4);
        EXPECT_NEAR(w5.eps_max / h,  0.5967, 1.0e-4);
        // without viscosity first-order differences need cfl <= 0.888 on the diagonal
        EXPECT_NEAR(up.cfl_max, 0.8884, 1.0e-3) << "first order, eps = 0";
        EXPECT_LT(levelset_substep_gain(0.88, 0.0, h, h, false), 1.0 + LEVELSET_GAIN_TOL);
        EXPECT_GT(levelset_substep_gain(0.89, 0.0, h, h, false), 1.0 + LEVELSET_GAIN_TOL);
    }
    struct Case { int scheme; Real dy; bool worst; };
    for (const Case c : {Case{LEVELSET_GRAD_UPWIND1, h, false}, Case{LEVELSET_GRAD_UPWIND1, h, true},
                         Case{LEVELSET_GRAD_WENO5Z, h, false}, Case{LEVELSET_GRAD_WENO5Z, h, true},
                         Case{LEVELSET_GRAD_UPWIND1, Real(2.0) * h, false}, Case{LEVELSET_GRAD_UPWIND1, Real(2.0) * h, true}}) {
        const bool weno = (c.scheme == LEVELSET_GRAD_WENO5Z);
        const LevelSetViscousStability st = levelset_viscous_stability(cfl, 0.0, h, c.dy, weno);
        const Real eps_lim = c.worst ? st.eps_max : st.eps_axis;
        // best: a front facing the coarser axis (y when dy > dx); worst:
        // the normal along (1/dx, 1/dy)
        const Real ang = c.worst ? std::atan2(Real(1.0) / c.dy, Real(1.0) / h)
                                 : ((c.dy > h) ? Real(2.0) * pi4 : Real(0.0));
        const std::string tag = std::string(weno ? "weno5z" : "upwind") + (c.worst ? " worst" : " best")
                              + ((c.dy > h) ? ", dy = 2 dx" : ", square cells");
        const Real below = wave_gain(c.scheme, 0.95 * eps_lim, cfl, ang, h, c.dy, 30);
        const Real above = wave_gain(c.scheme, 1.05 * eps_lim, cfl, ang, h, c.dy, 30);
        EXPECT_LT(below, 1.0) << tag << ": eps = 0.95 x " << eps_lim << " m";
        EXPECT_GT(above, 10.0) << tag << ": eps = 1.05 x " << eps_lim << " m";
    }
    // between the two limits a front along the diagonal grows the wave and
    // one facing an axis damps it
    for (const int scheme : {LEVELSET_GRAD_UPWIND1, LEVELSET_GRAD_WENO5Z}) {
        const LevelSetViscousStability st = levelset_viscous_stability(cfl, 0.0, h, h, scheme == LEVELSET_GRAD_WENO5Z);
        const Real eps_mid = Real(0.5) * (st.eps_axis + st.eps_max);
        const Real diag = wave_gain(scheme, eps_mid, cfl, pi4, h, h, 30);
        const Real axis = wave_gain(scheme, eps_mid, cfl, Real(0.0), h, h, 30);
        EXPECT_GT(diag, 10.0) << "scheme " << scheme << ", diagonal front at eps " << eps_mid << " m";
        EXPECT_LT(axis, 1.0) << "scheme " << scheme << ", axis front at eps " << eps_mid << " m";
        EXPECT_GT(levelset_substep_gain(cfl, eps_mid, h, h, scheme == LEVELSET_GRAD_WENO5Z), 1.0 + LEVELSET_GAIN_TOL)
            << "start-up aborts there";
    }
}

/**
 * Above cfl of about 0.55 the WENO5 substep is limited by a complex wave
 * about 2.4 cells long (theta near 2.58) on a front near the diagonal, not by
 * the two-cell wave. At cfl 0.8 on square cells the scan gives eps/dx <=
 * 0.1732, the two-cell formula 0.2041 (which the check used before). The
 * production advection with that wave seeded: at the formula's value it
 * grows (|G| = 1.20 per substep from the scan), at 0.95 of the scan's limit
 * it does not. Without viscosity WENO5 needs cfl <= 1.015 on square cells,
 * where the formula allowed 1.666 on the diagonal.
 */
TEST(LevelSetAdvection, ViscousStabilityLimitWenoAboveTheTwoCellRegime)
{
    const Real h = 5.0, cfl = 0.8;
    const Real pi4 = Real(0.25) * PI;
    const LevelSetViscousStability st = levelset_viscous_stability(cfl, 0.0, h, h, true);
    EXPECT_NEAR(st.eps_max / h, 0.1732, 1.0e-3);
    // the two-cell wave alone, the old formula: eps = (2.513/(cfl h) - (16/15) sqrt(2)/h) / (8/h^2)
    const Real eps_two_cell = (Real(LEVELSET_RK3_REAL_LIMIT) / (cfl * h) - LEVELSET_WENO5_TWO_CELL_GAIN * std::sqrt(Real(2.0)) / h)
                            / (Real(8.0) / (h * h));
    EXPECT_NEAR(eps_two_cell / h, 0.2041, 1.0e-3);
    EXPECT_GT(levelset_substep_gain(cfl, eps_two_cell, h, h, true), 1.15) << "the formula's eps is unstable";
    EXPECT_NEAR(levelset_viscous_stability(cfl, 0.0, h, h, true).cfl_max, 1.0149, 2.0e-3);
    EXPECT_NEAR(levelset_viscous_stability(Real(0.4), 0.0, h, h, true).cfl_max, 1.0149, 2.0e-3);
    // the production substep on that wave, a front along the diagonal
    const Real theta = 2.58;
    const Real seed  = (sizeof(Real) == 8) ? 1.0e-5 : 1.0e-3;
    const Real at_formula = wave_gain(LEVELSET_GRAD_WENO5Z, eps_two_cell, cfl, pi4, h, h, 30, theta, seed);
    const Real below      = wave_gain(LEVELSET_GRAD_WENO5Z, 0.95 * st.eps_max, cfl, pi4, h, h, 30, theta, seed);
    EXPECT_GT(at_formula, 10.0) << "eps = " << eps_two_cell << " m, the two-cell formula's limit";
    EXPECT_LT(below, 1.0) << "eps = 0.95 x " << st.eps_max << " m";
}
