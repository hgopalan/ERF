#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>

#include "ERF_FireGrid.H"
#include "ERF_FireAcceleration.H"
#include "ERF_FuelBlending.H"
#include "ERF_CrownFire.H"

/**
 * @file ERF_GTestFireBoxParity.cpp
 * @brief Three neighbour stencils of the fire grid that must not depend on the
 *        box decomposition: the burned-perimeter count of the temporal
 *        acceleration (it stopped at each box's edge, so the A_point/A_line
 *        switch moved with the rank count), the fuel-boundary blending (it
 *        read neighbours the same pass had already rewritten) and the crown
 *        rate carried into the unburned band ahead of a crowned cell.
 *        Each is run on one box and on many, against a value computed here.
 */

using namespace amrex;

namespace {

constexpr int  N  = 24;
constexpr Real DX = 10.0;
/// Round-off tolerance of the build precision
constexpr double TOLP = (sizeof(Real) == 8) ? 1.0e-12 : 1.0e-5;

struct Grid
{
    BoxArray            ba;
    DistributionMapping dm;
    Geometry            geom;
    Grid (int max_grid, bool periodic)
    {
        Box domain(IntVect(0, 0, 0), IntVect(N - 1, N - 1, 0));
        ba = BoxArray(domain);
        ba.maxSize(IntVect(max_grid, max_grid, 1));
        dm = DistributionMapping(ba);
        geom = Geometry(domain, RealBox(0.0, 0.0, 0.0, N * DX, N * DX, 1.0), CoordSys::cartesian,
                        {periodic ? 1 : 0, periodic ? 1 : 0, 0});
    }
};

/// Fill a field from a host function of (i, j)
template <class F>
void fill (MultiFab& mf, F f)
{
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto a = mf.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { a(i, j, k) = f(i, j); });
    }
}

/// Copy a field to one box, so every rank can read any cell
MultiFab gather (const MultiFab& mf, const Geometry& geom)
{
    BoxArray one(geom.Domain());
    MultiFab out(one, DistributionMapping(one), 1, 0);
    out.ParallelCopy(mf, 0, 0, 1);
    return out;
}

Real max_abs_diff (const MultiFab& a, const MultiFab& b)
{
    Real m = 0.0;
    for (MFIter mfi(a); mfi.isValid(); ++mfi) {
        auto pa = a.const_array(mfi);
        auto pb = b.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i)
                m = amrex::max(m, std::abs(pa(i, j, 0) - pb(i, j, 0)));
    }
    ParallelDescriptor::ReduceRealMax(m);
    return m;
}

/// phi of a disc of radius r0 cells about the domain centre
AMREX_GPU_HOST_DEVICE Real disc_phi (int i, int j, Real r0) noexcept
{
    const Real x = i + 0.5_rt - 0.5_rt * N, y = j + 0.5_rt - 0.5_rt * N;
    return std::sqrt(x * x + y * y) - r0;
}

} // namespace

// nvcc refuses an extended __device__ lambda whose enclosing function is
// private, as gtest's TestBody is: the bodies that launch kernels are
// namespace-scope functions, and each TEST calls its own.
namespace {
void PerimeterCount_DiscOnOneBoxAndOnMany ()
{
    const Real r0 = 6.0_rt;
    amrex::Long counts[2];
    int n = 0;
    for (int max_grid : {N, 4}) {
        Grid g(max_grid, true);
        MultiFab phi(g.ba, g.dm, 1, 0);
        fill(phi, [=] AMREX_GPU_DEVICE (int i, int j) { return disc_phi(i, j, r0); });
        counts[n++] = count_perimeter_cells(phi, g.geom);
    }
    // the hand count: burned cells with an unburned 4-neighbour
    amrex::Long expect = 0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            if (disc_phi(i, j, r0) >= 0.0_rt) { continue; }
            const bool per = disc_phi(i + 1, j, r0) >= 0.0_rt || disc_phi(i - 1, j, r0) >= 0.0_rt
                          || disc_phi(i, j + 1, r0) >= 0.0_rt || disc_phi(i, j - 1, r0) >= 0.0_rt;
            if (per) { ++expect; }
        }
    }
    EXPECT_EQ(counts[0], expect);
    EXPECT_EQ(counts[1], expect) << "36 boxes of 4x4: the stencil crosses the box edges";
    EXPECT_GT(expect, 20);
}
}  // namespace

TEST(PerimeterCount, DiscOnOneBoxAndOnMany)
{
    PerimeterCount_DiscOnOneBoxAndOnMany();
}

namespace {
void PerimeterCount_DomainEdgeAndPeriodicSeam ()
{
    // a strip burned for i < 4: only i = 3 faces unburned cells in a
    // non-periodic domain; with periodic x, i = 0 sees i = 23 as well
    for (bool periodic : {false, true}) {
        Grid g(4, periodic);
        MultiFab phi(g.ba, g.dm, 1, 0);
        fill(phi, [=] AMREX_GPU_DEVICE (int i, int) { return (i < 4) ? -1.0_rt : 1.0_rt; });
        EXPECT_EQ(count_perimeter_cells(phi, g.geom), periodic ? 2 * N : N) << "periodic " << periodic;
    }
}
}  // namespace

TEST(PerimeterCount, DomainEdgeAndPeriodicSeam)
{
    PerimeterCount_DomainEdgeAndPeriodicSeam();
}

namespace {
void FuelBlendingParity_OneBoxManyBoxesAndTheFormula ()
{
    // fuel 1 on the left half, 2 on the right, a 3x3 patch of 3 at (6..8, 6..8);
    // host and device: the kernels fill the fields with them and the host loop
    // below evaluates the formula with them (HIP refuses a device-only call)
    auto code = [] AMREX_GPU_HOST_DEVICE (int i, int j) -> Real {
        if (i >= 6 && i <= 8 && j >= 6 && j <= 8) { return 3.0_rt; }
        return (i < 12) ? 1.0_rt : 2.0_rt;
    };
    auto rate = [=] AMREX_GPU_HOST_DEVICE (int i, int j) -> Real {
        const Real c = code(i, j);
        return 0.1_rt * c + 0.001_rt * j + 0.0005_rt * i;
    };
    const Real frac = 0.4_rt;
    MultiFab results[2];
    int n = 0;
    for (int max_grid : {N, 6}) {
        Grid g(max_grid, false);
        MultiFab ros(g.ba, g.dm, 1, 0), fuel(g.ba, g.dm, 1, 0);
        fill(ros, rate);
        fill(fuel, code);
        apply_fuel_boundary_blending(ros, fuel, frac, g.geom);
        results[n++] = gather(ros, g.geom);
    }
    EXPECT_EQ(max_abs_diff(results[0], results[1]), 0.0_rt) << "16 boxes of 6x6 must give one box's result";

    // the formula from the unblended field: (1 - f) R + f mean(R of the
    // differently coded 4-neighbours inside the domain)
    Real err = 0.0;
    int n_blended = 0;
    for (MFIter mfi(results[0]); mfi.isValid(); ++mfi) {
        auto r = results[0].const_array(mfi);
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                const Real c = code(i, j);
                Real sum = 0.0; int nd = 0;
                const int di[4] = {-1, 1, 0, 0}, dj[4] = {0, 0, -1, 1};
                for (int m = 0; m < 4; ++m) {
                    const int ii = i + di[m], jj = j + dj[m];
                    if (ii < 0 || ii >= N || jj < 0 || jj >= N) { continue; }
                    if (code(ii, jj) != c) { sum += rate(ii, jj); ++nd; }
                }
                Real expect = rate(i, j);
                if (nd > 0) { expect = (1.0_rt - frac) * expect + frac * sum / nd; ++n_blended; }
                err = amrex::max(err, std::abs(r(i, j, 0) - expect));
            }
        }
    }
    EXPECT_LT(err, TOLP);
    EXPECT_GT(n_blended, 40);
}
}  // namespace

TEST(FuelBlendingParity, OneBoxManyBoxesAndTheFormula)
{
    FuelBlendingParity_OneBoxManyBoxesAndTheFormula();
}

namespace {
void CrownFront_CrownRateReachesTheBandAhead ()
{
    // a strip 40 x 4 of 10 m, burned for i < 10, crowned at i = 8, 9 with a
    // crown rate of 1 m/s over a surface rate of 0.1 m/s; dt = 10 s puts
    // n_ext = 2 + ceil(1 x 10 / 10) = 3 cells of the unburned band on the
    // crown rate (factor 10), the rest stays on the surface rate
    const int nx = 40, ny = 4;
    for (int max_grid : {nx, 8}) {
        Box domain(IntVect(0, 0, 0), IntVect(nx - 1, ny - 1, 0));
        BoxArray ba(domain);
        ba.maxSize(IntVect(max_grid, ny, 1));
        DistributionMapping dm(ba);
        Geometry geom(domain, RealBox(0.0, 0.0, 0.0, nx * DX, ny * DX, 1.0), CoordSys::cartesian, {0, 1, 0});
        MultiFab ros(ba, dm, 1, 0), surf(ba, dm, 1, 0), phi(ba, dm, 1, 0), crown_ros(ba, dm, 1, 0), cact(ba, dm, 1, 0), fac(ba, dm, 1, 0);
        fill(phi,  [] AMREX_GPU_DEVICE (int i, int) { return (i < 10) ? -1.0_rt : 1.0_rt; });
        fill(surf, [] AMREX_GPU_DEVICE (int, int)   { return 0.1_rt; });
        fill(cact, [] AMREX_GPU_DEVICE (int i, int) { return (i == 8 || i == 9) ? 1.0_rt : 0.0_rt; });
        fill(crown_ros, [] AMREX_GPU_DEVICE (int i, int) { return (i == 8 || i == 9) ? 1.0_rt : 0.0_rt; });
        fill(ros,  [] AMREX_GPU_DEVICE (int i, int) { return (i == 8 || i == 9) ? 1.0_rt : 0.1_rt; });
        const amrex::Long took = extend_crown_ros_to_front(ros, surf, phi, crown_ros, cact, geom, 10.0_rt, fac);
        EXPECT_EQ(took, 3 * ny) << "max_grid " << max_grid;
        MultiFab r = gather(ros, geom), f = gather(fac, geom);
        for (MFIter mfi(r); mfi.isValid(); ++mfi) {
            auto ra = r.const_array(mfi);
            auto fa = f.const_array(mfi);
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const bool crown = (i >= 8 && i <= 12);
                    EXPECT_NEAR(ra(i, j, 0), crown ? 1.0 : 0.1, TOLP) << "cell " << i << ", max_grid " << max_grid;
                    EXPECT_NEAR(fa(i, j, 0), crown ? 10.0 : 1.0, 1.0e3 * TOLP) << "factor at " << i << ", max_grid " << max_grid;
                }
            }
        }
    }
}
}  // namespace

TEST(CrownFront, CrownRateReachesTheBandAhead)
{
    CrownFront_CrownRateReachesTheBandAhead();
}

namespace {
void CrownFront_AMaskStopsTheExtension ()
{
    // the strip of the test above with a non-burnable column at i = 11: the
    // crown rate reaches i = 10 and stops, so i = 12 keeps its surface rate
    const int nx = 40, ny = 4;
    Box domain(IntVect(0, 0, 0), IntVect(nx - 1, ny - 1, 0));
    BoxArray ba(domain);
    DistributionMapping dm(ba);
    Geometry geom(domain, RealBox(0.0, 0.0, 0.0, nx * DX, ny * DX, 1.0), CoordSys::cartesian, {0, 1, 0});
    MultiFab ros(ba, dm, 1, 0), surf(ba, dm, 1, 0), phi(ba, dm, 1, 0), crown_ros(ba, dm, 1, 0), cact(ba, dm, 1, 0), fac(ba, dm, 1, 0), msk(ba, dm, 1, 0);
    fill(phi,  [] AMREX_GPU_DEVICE (int i, int) { return (i < 10) ? -1.0_rt : 1.0_rt; });
    // the mask column carries no surface rate (a non-burnable code): the
    // factor there is 1 from a floored divide, not 0/0
    fill(surf, [] AMREX_GPU_DEVICE (int i, int) { return (i == 11) ? 0.0_rt : 0.1_rt; });
    fill(cact, [] AMREX_GPU_DEVICE (int i, int) { return (i == 8 || i == 9) ? 1.0_rt : 0.0_rt; });
    fill(crown_ros, [] AMREX_GPU_DEVICE (int i, int) { return (i == 8 || i == 9) ? 1.0_rt : 0.0_rt; });
    fill(ros,  [] AMREX_GPU_DEVICE (int i, int) { return (i == 8 || i == 9) ? 1.0_rt : (i == 11) ? 0.0_rt : 0.1_rt; });
    fill(msk,  [] AMREX_GPU_DEVICE (int i, int) { return (i == 11) ? 1.0_rt : 0.0_rt; });
    const amrex::Long took = extend_crown_ros_to_front(ros, surf, phi, crown_ros, cact, geom, 10.0_rt, fac, &msk);
    EXPECT_EQ(took, ny) << "only i = 10";
    MultiFab r = gather(ros, geom), f = gather(fac, geom);
    for (MFIter mfi(r); mfi.isValid(); ++mfi) {
        auto ra = r.const_array(mfi);
        auto fa = f.const_array(mfi);
        for (int j = 0; j < ny; ++j) {
            EXPECT_NEAR(ra(10, j, 0), 1.0, TOLP);
            EXPECT_NEAR(ra(11, j, 0), 0.0, TOLP) << "the mask cell keeps its (zero) rate";
            EXPECT_NEAR(fa(11, j, 0), 1.0, TOLP) << "no factor where there is no surface rate";
            EXPECT_NEAR(ra(12, j, 0), 0.1, TOLP) << "nothing beyond the mask";
        }
    }
}
}  // namespace

TEST(CrownFront, AMaskStopsTheExtension)
{
    CrownFront_AMaskStopsTheExtension();
}

namespace {
void CrownFront_NoCrownedCellLeavesEverythingAlone ()
{
    Grid g(N, false);
    MultiFab ros(g.ba, g.dm, 1, 0), surf(g.ba, g.dm, 1, 0), phi(g.ba, g.dm, 1, 0), crown_ros(g.ba, g.dm, 1, 0), cact(g.ba, g.dm, 1, 0), fac(g.ba, g.dm, 1, 0);
    fill(phi,  [] AMREX_GPU_DEVICE (int i, int) { return (i < 10) ? -1.0_rt : 1.0_rt; });
    fill(surf, [] AMREX_GPU_DEVICE (int, int) { return 0.1_rt; });
    fill(ros,  [] AMREX_GPU_DEVICE (int, int) { return 0.1_rt; });
    crown_ros.setVal(2.0_rt);
    cact.setVal(0.0_rt);
    EXPECT_EQ(extend_crown_ros_to_front(ros, surf, phi, crown_ros, cact, g.geom, 10.0_rt, fac), 0);
    EXPECT_NEAR(ros.max(0), 0.1, TOLP);
    EXPECT_NEAR(fac.min(0), 1.0, TOLP);
    EXPECT_NEAR(fac.max(0), 1.0, TOLP);
}
}  // namespace

TEST(CrownFront, NoCrownedCellLeavesEverythingAlone)
{
    CrownFront_NoCrownedCellLeavesEverythingAlone();
}
