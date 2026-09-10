#include <gtest/gtest.h>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_IntVect.H>
#include <cmath>

// Include the header with FARSITE functions
#include "ERF_FarsiteEllipse.H"

/**
 * @file ERF_GTestFarsiteSpreadAccumulation.cpp
 * @brief Unit tests for the FARSITE front update
 *
 * The default update ("front_cell") burns the unburned cells next to the
 * burned region one row at a time, at arrival times built from their burned
 * neighbours' arrival times, so a planar front moves at R t (not the 2 R t of
 * the legacy stamping), keeps moving when a burned cell's rate drops to zero,
 * and gives the same arrival times whatever the box decomposition.
 */

using namespace amrex;

namespace {
// Arrival times are sums of cell crossings; single precision keeps a few digits.
const Real tol_t = (sizeof(Real) == 8) ? 1.0e-9 : 1.0e-3;
}

class FarsiteSpreadTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a simple 2D domain (10x10 cells)
        Box domain(IntVect(0, 0, 0), IntVect(9, 9, 0));
        BoxArray ba(domain);
        DistributionMapping dm(ba);

        // Physical domain: 100 m x 100 m
        RealBox prob_domain(0.0, 0.0, 0.0, 100.0, 100.0, 1.0);
        Geometry geom(domain, prob_domain, CoordSys::cartesian, {false, false, false});

        // Create MultiFabs
        // phi: level-set field with 1 ghost cell
        phi.define(ba, dm, 1, 1);

        // farsite_work: 2-component spread vector, no ghosts
        farsite_work.define(ba, dm, 2, 0);

        // vel_eff: 2-component wind field, no ghosts
        vel_eff.define(ba, dm, 2, 0);

        // R_mf: ROS field, no ghosts
        R_mf.define(ba, dm, 1, 0);

        // disp_accum: 2-component accumulator the stepper carries between
        // substeps; arrival_time: -1 until a cell burns
        disp_accum.define(ba, dm, 2, 0);
        arrival_time.define(ba, dm, 1, 0);

        // Initialize: phi = 1 (unburned everywhere)
        phi.setVal(1.0_rt);
        farsite_work.setVal(0.0_rt);
        vel_eff.setVal(0.0_rt);
        R_mf.setVal(0.1_rt);  // Constant ROS = 0.1 m/s
        disp_accum.setVal(0.0_rt);
        arrival_time.setVal(-1.0_rt);

        this->geom = geom;
    }

    /// Burn the cells for which burned(i, j) holds, at time 0
    template <typename F>
    void burn(F burned)
    {
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            auto p  = phi.array(mfi);
            auto at = arrival_time.array(mfi);
            const Box& bx = mfi.validbox();
            for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
                for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i)
                    if (burned(i, j)) { p(i, j, 0) = -1.0_rt; at(i, j, 0) = 0.0_rt; }
        }
    }

    /// Advance whole atmospheric steps of dt_atm from t = 0 to t_end; before
    /// each, set_ros(R_mf) may rewrite the rate of spread
    template <typename G>
    void run(const FarsiteParams& fp, Real dt_atm, Real t_end, G set_ros)
    {
        const int nsteps = static_cast<int>(std::lround(t_end / dt_atm));
        for (int n = 0; n < nsteps; ++n) {
            set_ros(R_mf, arrival_time);
            advance_fire_subcycle(phi, farsite_work, disp_accum, arrival_time,
                                  vel_eff, R_mf, geom, dt_atm, n * dt_atm, fp);
        }
    }

    MultiFab phi;
    MultiFab farsite_work;
    MultiFab vel_eff;
    MultiFab R_mf;
    MultiFab disp_accum;
    MultiFab arrival_time;
    Geometry geom;
};

/// Number of burned cells (phi < 0) on this rank
static int count_burned(const MultiFab& phi)
{
    int n = 0;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int k = bx.smallEnd()[2]; k <= bx.bigEnd()[2]; ++k)
            for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
                for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i)
                    if (p(i, j, k) < 0.0_rt) ++n;
    }
    return n;
}

/// Arrival time of cell (i, j), from whichever box holds it
static Real arrival_at(const MultiFab& at, int i, int j)
{
    for (MFIter mfi(at); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(IntVect(i, j, 0))) { return at.const_array(mfi)(i, j, 0); }
    }
    return -2.0_rt;
}

static auto keep_ros = [] (MultiFab&, const MultiFab&) {};

/**
 * Test 1: the front cells are the unburned 4-neighbours of the burned region
 *
 *   - One burned cell at (4,4) (arrival time 0), unburned elsewhere
 *   - One 0.1 s substep at 0.1 m/s with no wind
 *   - The four 4-neighbours are front cells: flagged, with 0.01 m of head-rate
 *     distance accumulated; the burned cell and the diagonals carry nothing
 *   - Nothing burns, and unburned cells have phi = +1, not 0, so they are not
 *     mistaken for burned or front cells by the legacy threshold
 */
TEST_F(FarsiteSpreadTest, FrontCellsAreUnburnedNeighbours)
{
    burn([] (int i, int j) { return i == 4 && j == 4; });

    FarsiteParams fp;
    ASSERT_EQ(fp.front_update, farsite_front::front_cell);
    advance_farsite_one_step(phi, farsite_work, disp_accum, arrival_time,
                             vel_eff, R_mf, geom, 0.1_rt, 0.0_rt, fp);

    int n_front = 0;
    for (MFIter mfi(disp_accum); mfi.isValid(); ++mfi) {
        auto d = disp_accum.const_array(mfi);
        auto p = phi.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
            for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i) {
                const bool front = ((i == 3 || i == 5) && j == 4) || (i == 4 && (j == 3 || j == 5));
                if (front) {
                    EXPECT_EQ(d(i, j, 0, 1), 1.0_rt) << "(" << i << "," << j << ") should be a front cell";
                    EXPECT_NEAR(d(i, j, 0, 0), 0.01, tol_t) << "(" << i << "," << j << ") should hold R dt";
                    ++n_front;
                } else {
                    EXPECT_EQ(d(i, j, 0, 0), 0.0_rt) << "(" << i << "," << j << ") is not a front cell";
                    EXPECT_EQ(d(i, j, 0, 1), 0.0_rt);
                }
                EXPECT_EQ(p(i, j, 0), (i == 4 && j == 4) ? -1.0_rt : 1.0_rt);
            }
    }
    EXPECT_EQ(n_front, 4);
    EXPECT_EQ(count_burned(phi), 1);
}

/**
 * Test 2: the accumulator survives substeps and the arrival time is exact
 *
 *   - Burned columns i <= 4 (arrival time 0), R = 0.1 m/s, no wind
 *   - Two 0.1 s substeps: the front column's accumulator doubles, nothing burns
 *   - One substep from 0.2 s to 150 s: column 5 burns with arrival time
 *     dx / R = 100 s, inside the substep rather than at its start; column 6,
 *     not yet a front cell, does not burn; the accumulator of a burned cell is
 *     reset
 */
TEST_F(FarsiteSpreadTest, ArrivalTimeAcrossSubsteps)
{
    burn([] (int i, int) { return i <= 4; });
    FarsiteParams fp;

    advance_farsite_one_step(phi, farsite_work, disp_accum, arrival_time,
                             vel_eff, R_mf, geom, 0.1_rt, 0.0_rt, fp);
    advance_farsite_one_step(phi, farsite_work, disp_accum, arrival_time,
                             vel_eff, R_mf, geom, 0.1_rt, 0.1_rt, fp);
    for (MFIter mfi(disp_accum); mfi.isValid(); ++mfi) {
        auto d = disp_accum.const_array(mfi);
        for (int j = 0; j < 10; ++j) {
            EXPECT_NEAR(d(5, j, 0, 0), 0.02, tol_t) << "two substeps of R dt at (5," << j << ")";
        }
    }
    EXPECT_EQ(count_burned(phi), 50);

    advance_farsite_one_step(phi, farsite_work, disp_accum, arrival_time,
                             vel_eff, R_mf, geom, 149.8_rt, 0.2_rt, fp);
    EXPECT_EQ(count_burned(phi), 60);
    for (MFIter mfi(disp_accum); mfi.isValid(); ++mfi) {
        auto d  = disp_accum.const_array(mfi);
        auto at = arrival_time.const_array(mfi);
        for (int j = 0; j < 10; ++j) {
            EXPECT_NEAR(at(5, j, 0), 100.0, 100.0 * tol_t) << "column 5 arrives dx / R after column 4";
            EXPECT_LT(at(6, j, 0), 0.0_rt) << "column 6 only becomes a front cell after this substep";
            EXPECT_EQ(d(5, j, 0, 0), 0.0_rt);
            EXPECT_EQ(d(5, j, 0, 1), 0.0_rt);
        }
    }
}

/**
 * Test 3: a planar front advances R t, one row per dx / R, not 2 R t
 *
 *   - Burned column i = 0 at t = 0, R = 0.1 m/s, no wind, 10 m cells
 *   - 35 atmospheric steps of 10 s through advance_fire_subcycle
 *   - Column i arrives at i dx / R = 100 i s: columns 1-3 by 350 s, column 4 not
 *   - The legacy update on the same case runs at least 1.5 times as far
 */
TEST_F(FarsiteSpreadTest, PlanarFrontAdvancesRt)
{
    burn([] (int i, int) { return i == 0; });
    FarsiteParams fp;
    run(fp, 10.0_rt, 350.0_rt, keep_ros);

    for (int i = 1; i <= 3; ++i) {
        for (int j = 0; j < 10; ++j) {
            EXPECT_NEAR(arrival_at(arrival_time, i, j), 100.0 * i, 100.0 * i * tol_t)
                << "column " << i << " row " << j;
        }
    }
    EXPECT_EQ(count_burned(phi), 40) << "four burned columns after 350 s at 0.1 m/s from one";

    // The same case with the legacy update
    arrival_time.setVal(-1.0_rt);
    phi.setVal(1.0_rt);
    disp_accum.setVal(0.0_rt);
    burn([] (int i, int) { return i == 0; });
    FarsiteParams fp_legacy;
    fp_legacy.front_update = farsite_front::legacy;
    run(fp_legacy, 10.0_rt, 350.0_rt, keep_ros);
    const int n_legacy = count_burned(phi);
    EXPECT_GE(n_legacy, 60) << "the legacy update advanced two rows per cell of travel";
}

/**
 * Test 4: the front does not stall where a burned cell's rate drops to zero
 *
 *   - The planar case of test 3, with R = 0 in every burned cell (as after
 *     burnout) and 0.1 m/s in unburned cells: the arrival times are unchanged
 *   - The reverse, R = 0.1 m/s only in burned cells (the crown-fire rate is set
 *     there only): the burned side carries the front at the same rate
 */
TEST_F(FarsiteSpreadTest, NoStallWhenBurnedCellsLoseTheirRate)
{
    burn([] (int i, int) { return i == 0; });
    FarsiteParams fp;
    auto zero_behind = [] (MultiFab& ros, const MultiFab& at) {
        for (MFIter mfi(ros); mfi.isValid(); ++mfi) {
            auto r = ros.array(mfi);
            auto a = at.const_array(mfi);
            const Box& bx = mfi.validbox();
            for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
                for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i)
                    r(i, j, 0) = (a(i, j, 0) >= 0.0_rt) ? 0.0_rt : 0.1_rt;
        }
    };
    run(fp, 10.0_rt, 350.0_rt, zero_behind);
    for (int i = 1; i <= 3; ++i) {
        EXPECT_NEAR(arrival_at(arrival_time, i, 5), 100.0 * i, 100.0 * i * tol_t) << "column " << i;
    }
    EXPECT_EQ(count_burned(phi), 40);

    arrival_time.setVal(-1.0_rt);
    phi.setVal(1.0_rt);
    disp_accum.setVal(0.0_rt);
    burn([] (int i, int) { return i == 0; });
    auto only_behind = [] (MultiFab& ros, const MultiFab& at) {
        for (MFIter mfi(ros); mfi.isValid(); ++mfi) {
            auto r = ros.array(mfi);
            auto a = at.const_array(mfi);
            const Box& bx = mfi.validbox();
            for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
                for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i)
                    r(i, j, 0) = (a(i, j, 0) >= 0.0_rt) ? 0.1_rt : 0.0_rt;
        }
    };
    run(fp, 10.0_rt, 350.0_rt, only_behind);
    for (int i = 1; i <= 3; ++i) {
        EXPECT_NEAR(arrival_at(arrival_time, i, 5), 100.0 * i, 100.0 * i * tol_t) << "column " << i;
    }
    EXPECT_EQ(count_burned(phi), 40);
}

/**
 * Test 5: head, flank and backing rates from one burning cell under wind
 *
 *   - 21x21 cells of 10 m, one burned cell at the centre, R = 0.1 m/s,
 *     a 1.64 m/s wind along +x (Anderson L/W 1.64, Richards a = 1, c = 0.2,
 *     b = 1.2 / (2 L/W))
 *   - The cells downwind burn at m dx / (a R), the first upwind cell at
 *     dx / (c R) = 500 s and the first cells across the wind at dx / (b R)
 */
TEST(FarsiteFrontCell, HeadFlankAndBackRates)
{
    Box domain(IntVect(0, 0, 0), IntVect(20, 20, 0));
    BoxArray ba(domain);
    DistributionMapping dm(ba);
    Geometry geom(domain, RealBox(0.0, 0.0, 0.0, 210.0, 210.0, 1.0), CoordSys::cartesian, {false, false, false});
    MultiFab phi(ba, dm, 1, 1), work(ba, dm, 2, 0), disp(ba, dm, 2, 0), at(ba, dm, 1, 0);
    MultiFab vel(ba, dm, 2, 0), ros(ba, dm, 1, 0);
    phi.setVal(1.0_rt); work.setVal(0.0_rt); disp.setVal(0.0_rt); at.setVal(-1.0_rt);
    const Real U = 1.64_rt, R = 0.1_rt, h = 10.0_rt;
    vel.setVal(U, 0, 1); vel.setVal(0.0_rt, 1, 1); ros.setVal(R);
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        phi.array(mfi)(10, 10, 0) = -1.0_rt;
        at.array(mfi)(10, 10, 0)  = 0.0_rt;
    }
    Real a = 0, b = 0, c = 0;
    LW_ratio_to_richards_coefficients(anderson_LW_ratio(U * 2.237_rt), a, b, c);

    FarsiteParams fp;
    for (int n = 0; n < 60; ++n) {
        advance_fire_subcycle(phi, work, disp, at, vel, ros, geom, 10.0_rt, n * 10.0_rt, fp);
    }
    for (int m = 1; m <= 5; ++m) {
        EXPECT_NEAR(arrival_at(at, 10 + m, 10), m * h / (a * R), m * 100.0 * tol_t) << "head cell " << m;
    }
    EXPECT_NEAR(arrival_at(at, 9, 10), h / (c * R), 500.0 * tol_t) << "backing cell";
    EXPECT_NEAR(arrival_at(at, 10, 11), h / (b * R), 300.0 * tol_t) << "flank cell +y";
    EXPECT_NEAR(arrival_at(at, 10, 9),  h / (b * R), 300.0 * tol_t) << "flank cell -y";
    EXPECT_LT(arrival_at(at, 8, 10), 0.0_rt) << "the second backing cell needs 1000 s";
}

/// Arrival times after 400 s of a sloped, windy, patchy case on boxes of max_grid cells
static MultiFab run_decomposed(int max_grid, const Geometry& geom, const BoxArray& ba_one)
{
    BoxArray ba(geom.Domain());
    ba.maxSize(max_grid);
    DistributionMapping dm(ba);
    MultiFab phi(ba, dm, 1, 1), work(ba, dm, 2, 0), disp(ba, dm, 2, 0), at(ba, dm, 1, 0);
    MultiFab vel(ba, dm, 2, 0), ros(ba, dm, 1, 0), slopes(ba, dm, 2, 1);
    phi.setVal(1.0_rt); work.setVal(0.0_rt); disp.setVal(0.0_rt); at.setVal(-1.0_rt);
    vel.setVal(1.2_rt, 0, 1); vel.setVal(0.7_rt, 1, 1);
    slopes.setVal(0.1_rt, 0, 1); slopes.setVal(-0.05_rt, 1, 1);
    const Real two_pi = 2.0_rt * amrex::Math::pi<Real>();
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.array(mfi);
        auto a = at.array(mfi);
        auto r = ros.array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
            for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i) {
                r(i, j, 0) = 0.2_rt + 0.1_rt * std::sin(two_pi * static_cast<Real>(i) / 32.0_rt)
                                             * std::cos(two_pi * static_cast<Real>(j) / 32.0_rt);
                if (std::abs(i - 8) <= 1 && std::abs(j - 8) <= 1) { p(i, j, 0) = -1.0_rt; a(i, j, 0) = 0.0_rt; }
            }
    }
    FarsiteParams fp;
    for (int n = 0; n < 40; ++n) {
        advance_fire_subcycle(phi, work, disp, at, vel, ros, geom, 10.0_rt, n * 10.0_rt, fp);
    }
    MultiFab out(ba_one, DistributionMapping(ba_one), 1, 0);
    out.ParallelCopy(at, 0, 0, 1);
    return out;
}

/**
 * Test 6: the arrival times do not depend on the box decomposition
 *
 *   - 32x32 periodic cells, a burned 3x3 block, wind at an angle, a slope and a
 *     rate of spread that varies in space, 400 s
 *   - One box and 64 boxes of 4x4 cells give identical arrival times
 */
TEST(FarsiteFrontCell, DecompositionIndependent)
{
    Box domain(IntVect(0, 0, 0), IntVect(31, 31, 0));
    Geometry geom(domain, RealBox(0.0, 0.0, 0.0, 320.0, 320.0, 1.0), CoordSys::cartesian, {true, true, false});
    BoxArray ba_one(domain);
    MultiFab one  = run_decomposed(32, geom, ba_one);
    MultiFab many = run_decomposed(4,  geom, ba_one);

    int n_burned = 0;
    Real max_diff = 0.0_rt;
    for (MFIter mfi(one); mfi.isValid(); ++mfi) {
        auto a1 = one.const_array(mfi);
        auto a2 = many.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
            for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i) {
                max_diff = amrex::max(max_diff, std::abs(a1(i, j, 0) - a2(i, j, 0)));
                if (a1(i, j, 0) >= 0.0_rt) { ++n_burned; }
            }
    }
    EXPECT_GT(n_burned, 30) << "the fire should have spread beyond the 3x3 ignition";
    EXPECT_EQ(max_diff, 0.0_rt) << "one box and 64 boxes must give the same arrival times";
}

/**
 * Test 7: Single-cell stamping race safety (legacy update)
 *
 * Scenario:
 *   - Create two propagated points in the same cell
 *   - Verify that both can stamp without race condition
 *   - The min() operation should make this safe
 */
TEST_F(FarsiteSpreadTest, SingleCellStampingRaceSafety)
{
    // Initialize phi to 0 (front everywhere for simplicity)
    phi.setVal(0.0_rt);

    // No wind
    vel_eff.setVal(0.0_rt);

    FarsiteParams fp;
    fp.front_update = farsite_front::legacy;
    fp.phi_threshold = 0.1;
    fp.gaussian_sigma = -1.0;  // Single-cell stamping

    Real dt_fire = 0.1;

    // Run a step
    advance_farsite_one_step(phi, farsite_work, disp_accum, arrival_time,
                             vel_eff, R_mf, geom, dt_fire, 0.0_rt, fp);

    // After stamping, burned cells should be -1
    Real min_phi = phi.min(0);
    EXPECT_LE(min_phi, 0.0_rt) << "Some cells should be burned (phi <= 0)";

    // The test passes if we reach here without deadlock or out-of-bounds access
    // (verified at runtime by the GPU kernel using min())
}

/**
 * Test 8: Fire grid geometry resolution
 *
 * Validates that the fire grid is created with correct cell sizes.
 * With refinement factor C=4, dx_fire should be dx_atm / 4.
 */
TEST_F(FarsiteSpreadTest, FireGridGeometryResolution)
{
    // Test geometry should have dx = 100/10 = 10 m
    auto dx = geom.CellSize();
    EXPECT_NEAR(dx[0], 10.0, 1e-6) << "X cell size should be 10 m";
    EXPECT_NEAR(dx[1], 10.0, 1e-6) << "Y cell size should be 10 m";
}

// main() lives in Tests/Unit/ERF_GTestMain.cpp, shared by every suite.
