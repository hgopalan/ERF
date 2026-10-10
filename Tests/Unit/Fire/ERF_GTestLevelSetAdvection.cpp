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
