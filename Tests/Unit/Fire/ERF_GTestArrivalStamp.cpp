#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <cmath>

#include "ERF_NumericalSchemes.H"
#include "ERF_LevelSetAdvection.H"

/**
 * @file ERF_GTestArrivalStamp.cpp
 * @brief The arrival time of a cell the level set burns during a substep is
 *        the time the front crossed its centre, t0 + dt phi_old / (phi_old -
 *        phi_new), interpolated in time, not the start of the substep (a bias
 *        of up to one substep, -0.2 h/R on average at cfl 0.4).
 */

using namespace amrex;
using namespace fire_levelset;

/// Absolute tolerance on times of order 100 s at the build precision
static constexpr double TOL_T = (sizeof(Real) == 8) ? 1.0e-9 : 1.0e-3;

TEST(ArrivalStamp, CrossingTimeInterpolatesInTime)
{
    EXPECT_NEAR(levelset_crossing_time(100.0_rt, 10.0_rt, 0.75_rt, -0.25_rt), 107.5, TOL_T);
    EXPECT_NEAR(levelset_crossing_time(100.0_rt, 10.0_rt, 1.0_rt, 0.0_rt),   110.0, TOL_T) << "reaches zero at the end";
    EXPECT_NEAR(levelset_crossing_time(100.0_rt, 10.0_rt, 1.0_rt, -3.0_rt),  102.5, TOL_T);
    EXPECT_NEAR(levelset_crossing_time(100.0_rt, 10.0_rt, 0.0_rt, -1.0_rt),  100.0, TOL_T) << "was at the front: the start";
    EXPECT_NEAR(levelset_crossing_time(100.0_rt, 10.0_rt, -0.5_rt, -1.5_rt), 100.0, TOL_T) << "already burned: the start";
    EXPECT_NEAR(levelset_crossing_time(100.0_rt, 10.0_rt, 1.0_rt, 1.0_rt),   100.0, TOL_T) << "no drop: the start, no division";
}

// nvcc refuses an extended __device__ lambda whose enclosing function is
// private, as gtest's TestBody is: the bodies that launch kernels are
// namespace-scope functions, and each TEST calls its own.
namespace {
void ArrivalStamp_NewlyBurnedCellsTakeTheCrossingTime ()
{
    // a strip of 40 x 4 cells of 5 m in two boxes; a planar front at x0 =
    // 100 m moves R dt = 10 m during a substep of 20 s starting at t0 = 100 s
    const int nx = 40, ny = 4;
    const Real dx = 5.0_rt, x0 = 100.0_rt, R = 0.5_rt, dt = 20.0_rt, t0 = 100.0_rt;
    Box domain(IntVect(0, 0, 0), IntVect(nx - 1, ny - 1, 0));
    BoxArray ba(domain);
    ba.maxSize(IntVect(nx / 2, ny, 1));
    DistributionMapping dm(ba);
    MultiFab po(ba, dm, 1, 0), pn(ba, dm, 1, 0), at(ba, dm, 1, 0);
    for (MFIter mfi(at); mfi.isValid(); ++mfi) {
        auto o = po.array(mfi);
        auto n = pn.array(mfi);
        auto a = at.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real x = (i + 0.5_rt) * dx;
            o(i, j, k) = x - x0;
            n(i, j, k) = x - x0 - R * dt;
            a(i, j, k) = (o(i, j, k) < 0.0_rt) ? 0.0_rt : -1.0_rt;   // burned before: stamped earlier
        });
    }
    stamp_arrival_time(at, po, pn, t0, dt);
    for (MFIter mfi(at); mfi.isValid(); ++mfi) {
        auto a = at.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                const Real x = (i + 0.5_rt) * dx;
                if (x < x0) {
                    EXPECT_EQ(a(i, j, 0), 0.0_rt) << "cell " << i << " burned before the substep keeps its stamp";
                } else if (x < x0 + R * dt) {
                    EXPECT_NEAR(a(i, j, 0), t0 + (x - x0) / R, TOL_T) << "cell " << i << " burns when the front crosses it";
                } else {
                    EXPECT_EQ(a(i, j, 0), -1.0_rt) << "cell " << i << " is still unburned";
                }
            }
        }
    }
}
}  // namespace

TEST(ArrivalStamp, NewlyBurnedCellsTakeTheCrossingTime)
{
    ArrivalStamp_NewlyBurnedCellsTakeTheCrossingTime();
}
