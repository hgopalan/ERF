#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <cmath>

#include "ERF_FarsiteEllipse.H"

/**
 * @file ERF_GTestFarsiteShape.cpp
 * @brief The spread shape of the FARSITE front-cell update: with
 *        erf.fire.farsite.shape = ellipse (the default) a cell spreads with the
 *        Richards (1990) / Alexander (1985) Huygens ellipse of its wind, so a
 *        point source burns the first upwind cell at h HB / R and the first
 *        cell across the wind at h LB sqrt(HB) / R, with HB = (LB + sqrt(LB^2 -
 *        1))^2; the rectangle kept as shape = rectangle burns them at 5 h / R
 *        and 2 LB h / (1.2 R). The ellipse is the disc at zero wind, so a
 *        vanishing wind leaves the arrival times unchanged, where the
 *        rectangle's fixed head-to-back ratio of 5 made them jump.
 */

using namespace amrex;

namespace {

const Real tol_t = (sizeof(Real) == 8) ? 1.0e-9 : 1.0e-3;

/// Anderson (1983) as fitted in FARSITE (Finney 1998, eq. 8), U in mi/h
double anderson_lb (double U_mph)
{
    return std::max(1.0, std::min(8.0, 0.936 * std::exp(0.2566 * U_mph) + 0.461 * std::exp(-0.1548 * U_mph) - 0.397));
}

double hb_of (double LB) { const double s = std::sqrt(LB * LB - 1.0); return (LB + s) * (LB + s); }

Real arrival_at (const MultiFab& at, int i, int j)
{
    for (MFIter mfi(at); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(IntVect(i, j, 0))) { return at.const_array(mfi)(i, j, 0); }
    }
    return -2.0_rt;
}

/// 21x21 cells of 10 m, one burned cell at the centre, R = 0.1 m/s, wind U along +x, n steps of 10 s
MultiFab point_source (Real U, const FarsiteParams& fp, int nsteps)
{
    Box domain(IntVect(0, 0, 0), IntVect(20, 20, 0));
    BoxArray ba(domain);
    DistributionMapping dm(ba);
    Geometry geom(domain, RealBox(0.0, 0.0, 0.0, 210.0, 210.0, 1.0), CoordSys::cartesian, {false, false, false});
    MultiFab phi(ba, dm, 1, 1), work(ba, dm, 2, 0), disp(ba, dm, 4, 0), at(ba, dm, 1, 0);
    MultiFab vel(ba, dm, 2, 0), ros(ba, dm, 1, 0);
    phi.setVal(1.0_rt); work.setVal(0.0_rt); disp.setVal(0.0_rt); at.setVal(-1.0_rt);
    vel.setVal(U, 0, 1); vel.setVal(0.0_rt, 1, 1); ros.setVal(0.1_rt);
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        phi.array(mfi)(10, 10, 0) = -1.0_rt;
        at.array(mfi)(10, 10, 0)  = 0.0_rt;
    }
    for (int n = 0; n < nsteps; ++n) {
        advance_fire_subcycle(phi, work, disp, at, vel, ros, geom, 10.0_rt, n * 10.0_rt, fp);
    }
    return at;
}

} // namespace

TEST(FarsiteShape, EllipseRatesFromTheLengthToWidthRatio)
{
    const Real R = 0.2_rt;
    for (double LB : {1.0, 1.5, 3.0, 8.0}) {
        const SpreadEllipse e = spread_ellipse_from_lw(R, Real(LB));
        const double HB = hb_of(LB);
        EXPECT_NEAR(e.b + e.c, R, 1.0e3 * tol_t) << "head rate, LB " << LB;
        EXPECT_NEAR(e.b - e.c, R / HB, 1.0e3 * tol_t) << "back rate, LB " << LB;
        EXPECT_NEAR(e.a, e.b / LB, 1.0e3 * tol_t) << "flank semi-axis, LB " << LB;
        EXPECT_NEAR(e.HB, HB, 1.0e3 * tol_t * HB) << "LB " << LB;
    }
    const SpreadEllipse disc = spread_ellipse_from_lw(R, 1.0_rt);
    EXPECT_NEAR(disc.a, R, tol_t);
    EXPECT_NEAR(disc.c, 0.0, tol_t);
    EXPECT_NEAR(ellipse_normal_speed(disc, 0.3_rt, std::sqrt(0.91_rt)), R, 1.0e3 * tol_t);
}

TEST(FarsiteShape, GaugeIsTheEllipseTravelTime)
{
    const double LB = 2.0, HB = hb_of(LB);
    const SpreadEllipse e = spread_ellipse_from_lw(1.0_rt, Real(LB));
    farsite_front::Shape s;
    s.wx = 1.0_rt; s.wy = 0.0_rt; s.a = e.a; s.b = e.b; s.c = e.c; s.sx = 0.0_rt; s.sy = 0.0_rt;
    s.isotropic = false; s.ellipse = true;
    EXPECT_NEAR(farsite_front::gauge( 7.0_rt, 0.0_rt, s), 7.0, 1.0e3 * tol_t) << "straight ahead at the head rate";
    EXPECT_NEAR(farsite_front::gauge(-7.0_rt, 0.0_rt, s), 7.0 * HB, 1.0e3 * tol_t) << "straight behind at R / HB";
    EXPECT_NEAR(farsite_front::gauge(0.0_rt, 7.0_rt, s), 7.0 * LB * std::sqrt(HB), 1.0e3 * tol_t) << "across the wind";
    // a point on the ellipse grown for time t: (c + b cos th, a sin th) t
    const double t = 3.0, th = 1.0;
    const Real px = Real(t * (e.c + e.b * std::cos(th))), py = Real(t * e.a * std::sin(th));
    EXPECT_NEAR(farsite_front::gauge(px, py, s), t, 1.0e3 * tol_t);
    EXPECT_NEAR(farsite_front::gauge(px, -py, s), t, 1.0e3 * tol_t);
    // a slope lengthens the travel by the ground distance
    s.sx = 0.75_rt;
    EXPECT_NEAR(farsite_front::gauge(7.0_rt, 0.0_rt, s), 7.0 * 1.25, 1.0e3 * tol_t);
}

TEST(FarsiteShape, HeadBackAndFlankFromAPointSource)
{
    const Real U = 0.5_rt, R = 0.1_rt, h = 10.0_rt;
    const double LB = anderson_lb(U * 2.237), HB = hb_of(LB);
    FarsiteParams fp;
    ASSERT_EQ(fp.shape, farsite_shape::ellipse) << "the ellipse is the default";
    const MultiFab at = point_source(U, fp, 50);
    for (int m = 1; m <= 5; ++m) {
        EXPECT_NEAR(arrival_at(at, 10 + m, 10), m * h / R, m * 100.0 * tol_t) << "head cell " << m;
    }
    EXPECT_NEAR(arrival_at(at, 9, 10),  h * HB / R, 0.5) << "backing cell at h HB / R";
    EXPECT_NEAR(arrival_at(at, 10, 11), h * LB * std::sqrt(HB) / R, 0.5) << "flank cell +y";
    EXPECT_NEAR(arrival_at(at, 10, 9),  h * LB * std::sqrt(HB) / R, 0.5) << "flank cell -y";
    // the rectangle of the same wind: back at h / (0.2 R), flank at 2 LB h / (1.2 R)
    fp.shape = farsite_shape::rectangle;
    const MultiFab rect = point_source(U, fp, 60);
    EXPECT_NEAR(arrival_at(rect, 9, 10),  h / (0.2_rt * R), 0.5);
    EXPECT_NEAR(arrival_at(rect, 10, 11), 2.0 * LB * h / (1.2 * R), 0.5);
    EXPECT_NEAR(arrival_at(rect, 11, 10), h / R, 100.0 * tol_t);
}

/**
 * From a single burning cell under wind every cell within the nine-cell source
 * stencil is reached by the direct Huygens path, so its arrival time is the
 * point-source value gauge(x, y) / R exactly (a chain through another cell
 * can only be later, by the triangle inequality of the convex gauge); the
 * quadrant update alone lagged the flank by a factor of three at L/W = 2.1.
 * 41 x 41 cells of 10 m, R = 0.1 m/s, a 1.5 m/s wind (L/W = 2.09), 1200 s:
 * the head reaches 12 cells, the flank 3 rows.
 */
void point_source_arrivals_are_exact (bool stamp_by_update)
{
    const Real U = 1.5_rt, R = 0.1_rt, h = 10.0_rt;
    Box domain(IntVect(0, 0, 0), IntVect(40, 40, 0));
    BoxArray ba(domain);
    ba.maxSize(IntVect(14, 14, 1));   // nine boxes: the sources cross box edges
    DistributionMapping dm(ba);
    Geometry geom(domain, RealBox(0.0, 0.0, 0.0, 410.0, 410.0, 1.0), CoordSys::cartesian, {false, false, false});
    MultiFab phi(ba, dm, 1, 1), work(ba, dm, 2, 0), disp(ba, dm, 4, 0), at(ba, dm, 1, 0);
    MultiFab vel(ba, dm, 2, 0), ros(ba, dm, 1, 0);
    phi.setVal(1.0_rt); work.setVal(0.0_rt); disp.setVal(0.0_rt); at.setVal(-1.0_rt);
    vel.setVal(U, 0, 1); vel.setVal(0.0_rt, 1, 1); ros.setVal(R);
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(IntVect(20, 20, 0))) {
            phi.array(mfi)(20, 20, 0) = -1.0_rt;
            // a spot landing or a scheduled ignition leaves phi < 0 without an
            // arrival time; the update dates it t0 and takes the clock there
            if (!stamp_by_update) { at.array(mfi)(20, 20, 0) = 0.0_rt; }
        }
    }
    FarsiteParams fp;
    for (int n = 0; n < 120; ++n) {
        advance_fire_subcycle(phi, work, disp, at, vel, ros, geom, 10.0_rt, n * 10.0_rt, fp);
    }
    // the shape of the test's own wind, as the update builds it
    const double LB = anderson_lb(U * 2.237);
    const SpreadEllipse e = spread_ellipse_from_lw(1.0_rt, Real(LB));
    farsite_front::Shape s;
    s.wx = 1.0_rt; s.wy = 0.0_rt; s.a = e.a; s.b = e.b; s.c = e.c; s.sx = 0.0_rt; s.sy = 0.0_rt;
    s.isotropic = false; s.ellipse = true;

    BoxArray one(domain);
    MultiFab all(one, DistributionMapping(one), 1, 0);
    all.ParallelCopy(at, 0, 0, 1);
    int n_checked = 0, n_flank = 0;
    Real err = 0.0_rt;
    for (MFIter mfi(all); mfi.isValid(); ++mfi) {
        auto a = all.const_array(mfi);
        for (int j = 0; j <= 40; ++j) {
            for (int i = 0; i <= 40; ++i) {
                const int di = i - 20, dj = j - 20;
                if (a(i, j, 0) < 0.0_rt) { continue; }
                if (std::abs(dj) == 3) { ++n_flank; }
                if (std::abs(di) > 9 || std::abs(dj) > 9) { continue; }
                const Real T_exact = farsite_front::gauge(di * h, dj * h, s) / R;
                err = amrex::max(err, std::abs(a(i, j, 0) - T_exact) / amrex::max(T_exact, 1.0_rt));
                ++n_checked;
            }
        }
    }
    EXPECT_GT(n_checked, 40);
    EXPECT_LT(err, 1.0e-6) << "every burned cell within the stencil carries its point-source arrival";
    // the flank: the third row across the wind is reached (semi-minor rate
    // a = 0.255 R puts it at 1180 s; the across-wind rate alone would need 4200 s)
    EXPECT_GE(n_flank, 1) << "the ellipse's flank grows at its semi-minor rate";
    EXPECT_NEAR(e.a, 0.255, 0.01);
}

TEST(FarsiteShape, PointSourceArrivalsAreExactWithinTheStencil) { point_source_arrivals_are_exact(false); }

/// The same from a cell the update itself dates (phi < 0, no arrival time):
/// its clock at burn is the clock at t0, so the direct paths from it are exact
TEST(FarsiteShape, ACellTheUpdateDatesIsAnExactSource) { point_source_arrivals_are_exact(true); }

/**
 * A fuel boundary across the wind: slow fuel (R = 0.25 m/s) up to x = 300 m,
 * fast fuel (0.5 m/s) beyond. The head row's cells in the fast fuel burn at
 * T_boundary + k h / R_fast, the planar-front rate, and no earlier: a direct
 * source in the slow fuel may not lend the fast cell its own clock (which
 * would credit the slow fuel's history at the fast rate and burn the ninth
 * cell 100 s early).
 */
TEST(FarsiteShape, AFuelBoundaryDoesNotLendTheSlowFuelsClock)
{
    const int nx = 60, ny = 5;
    const Real h = 10.0_rt, U = 1.0_rt, R_slow = 0.25_rt, R_fast = 0.5_rt;
    Box domain(IntVect(0, 0, 0), IntVect(nx - 1, ny - 1, 0));
    BoxArray ba(domain);
    ba.maxSize(IntVect(20, ny, 1));
    DistributionMapping dm(ba);
    Geometry geom(domain, RealBox(0.0, 0.0, 0.0, nx * h, ny * h, 1.0), CoordSys::cartesian, {false, false, false});
    MultiFab phi(ba, dm, 1, 1), work(ba, dm, 2, 0), disp(ba, dm, 4, 0), at(ba, dm, 1, 0);
    MultiFab vel(ba, dm, 2, 0), ros(ba, dm, 1, 0);
    phi.setVal(1.0_rt); work.setVal(0.0_rt); disp.setVal(0.0_rt); at.setVal(-1.0_rt);
    vel.setVal(U, 0, 1); vel.setVal(0.0_rt, 1, 1);
    for (MFIter mfi(ros); mfi.isValid(); ++mfi) {
        auto r = ros.array(mfi);
        auto p = phi.array(mfi);
        auto a = at.array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
            for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i) {
                r(i, j, 0) = (i < 30) ? R_slow : R_fast;
                if (i <= 2) { p(i, j, 0) = -1.0_rt; a(i, j, 0) = 0.0_rt; }   // a line ignition
            }
    }
    FarsiteParams fp;
    for (int n = 0; n < 160; ++n) {
        advance_fire_subcycle(phi, work, disp, at, vel, ros, geom, 10.0_rt, n * 10.0_rt, fp);
    }
    const Real T30 = arrival_at(at, 30, 2);
    ASSERT_GT(T30, 1000.0_rt) << "the boundary cell burns at about (27.5 cells) 10 m / 0.25 m/s";
    for (int k = 1; k <= 12; ++k) {
        const Real T = arrival_at(at, 30 + k, 2);
        ASSERT_GE(T, 0.0_rt) << "cell " << 30 + k << " burned";
        EXPECT_NEAR(T, T30 + k * h / R_fast, 1.0e3 * tol_t) << "fast cell " << k << " runs at the fast fuel's rate from the boundary";
    }
}

TEST(FarsiteShape, AVanishingWindIsTheDisc)
{
    FarsiteParams fp;
    const MultiFab calm = point_source(0.0_rt, fp, 40);
    // Alexander's HB - 1 grows as sqrt(LB - 1), so the ellipse's rates differ
    // from the disc's by about sqrt(U): 5e-5 at 1e-9 m/s (6e-4 at 1e-7)
    const MultiFab tiny = point_source(1.0e-9_rt, fp, 40);
    Real max_diff = 0.0_rt;
    int n_burned = 0, n_differ = 0;
    for (MFIter mfi(calm); mfi.isValid(); ++mfi) {
        auto a = calm.const_array(mfi);
        auto b = tiny.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
            for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i) {
                const bool ba = a(i, j, 0) >= 0.0_rt, bb = b(i, j, 0) >= 0.0_rt;
                if (ba) { ++n_burned; }
                if (ba != bb) { ++n_differ; continue; }   // a cell burning at the final instant may flip
                if (ba) { max_diff = amrex::max(max_diff, std::abs(a(i, j, 0) - b(i, j, 0))); }
            }
    }
    EXPECT_GT(n_burned, 30);
    EXPECT_LE(n_differ, 3);
    EXPECT_LT(max_diff, 0.05) << "ellipse: a 1e-9 m/s wind leaves the arrival times (about 100-400 s) unchanged";
    EXPECT_NEAR(arrival_at(calm, 9, 10), 100.0, 100.0 * tol_t) << "the disc's back cell at h / R";

    fp.shape = farsite_shape::rectangle;
    const MultiFab rect = point_source(1.0e-9_rt, fp, 60);
    EXPECT_NEAR(arrival_at(rect, 9, 10), 500.0, 0.5)
        << "rectangle: the back cell jumps from h / R to 5 h / R";
}
