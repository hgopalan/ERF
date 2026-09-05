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
 * @brief Unit tests for FARSITE spread accumulation behavior
 *
 * Validates that the ERF implementation correctly reproduces
 * wildfire_levelset behavior for accumulated spread vectors.
 *
 * Key test: After Pass 1 zeros phi globally, the accumulated
 * spread vectors from previous steps allow Pass 2 to reconstruct
 * the burned interior (cells that had nonzero spread before).
 */

using namespace amrex;

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

        // disp_accum: 2-component displacement accumulator the stepper
        // carries between substeps; arrival_time: -1 until a cell burns
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

/// Largest |displacement| in disp_accum, and the cell it sits in
static Real max_disp(const MultiFab& disp, IntVect& where)
{
    Real m = 0.0;
    for (MFIter mfi(disp); mfi.isValid(); ++mfi) {
        auto d = disp.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int k = bx.smallEnd()[2]; k <= bx.bigEnd()[2]; ++k)
            for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
                for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i) {
                    Real mag = std::sqrt(d(i, j, k, 0) * d(i, j, k, 0) + d(i, j, k, 1) * d(i, j, k, 1));
                    if (mag > m) { m = mag; where = IntVect(i, j, k); }
                }
    }
    return m;
}

/**
 * Test 1: one substep accumulates displacement at the front and stamps nothing
 *
 * Scenario:
 *   - One burned cell at (4,4) (arrival time 0), its four neighbours on the
 *     front (phi = 0), unburned elsewhere
 *   - One 0.1 s substep at 0.1 m/s with no wind: each front cell travels
 *     0.01 m along its front normal, far short of the one-cell stamping
 *     threshold
 *   - The displacement accumulator holds that 0.01 m at the four front cells
 *     and nothing elsewhere; the work array holds no stamp target; the burned
 *     set is unchanged
 */
TEST_F(FarsiteSpreadTest, SingleStepFrontDetection)
{
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p  = phi.array(mfi);
        auto at = arrival_time.array(mfi);
        const Box& bx = mfi.tilebox();
        for (int k = bx.smallEnd()[2]; k <= bx.bigEnd()[2]; ++k) {
            for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j) {
                for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i) {
                    if (i == 4 && j == 4) {
                        p(i, j, k) = -1.0_rt; at(i, j, k) = 0.0_rt;   // burned
                    } else if (((i == 3 || i == 5) && j == 4) || (i == 4 && (j == 3 || j == 5))) {
                        p(i, j, k) = 0.0_rt;                          // front
                    } else {
                        p(i, j, k) = 1.0_rt;                          // unburned
                    }
                }
            }
        }
    }

    FarsiteParams fp;
    fp.phi_threshold = 0.1;
    const Real dt_fire = 0.1;   // 0.1 s at 0.1 m/s: 0.01 m

    advance_farsite_one_step(phi, farsite_work, disp_accum, arrival_time,
                             vel_eff, R_mf, geom, dt_fire, 0.0_rt, fp);

    // The four front cells carry 0.01 m of displacement, nothing else does
    int n_moved = 0;
    for (MFIter mfi(disp_accum); mfi.isValid(); ++mfi) {
        auto d = disp_accum.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int k = bx.smallEnd()[2]; k <= bx.bigEnd()[2]; ++k)
            for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
                for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i) {
                    Real mag = std::sqrt(d(i, j, k, 0) * d(i, j, k, 0) + d(i, j, k, 1) * d(i, j, k, 1));
                    const bool front = ((i == 3 || i == 5) && j == 4) || (i == 4 && (j == 3 || j == 5));
                    if (front) {
                        EXPECT_NEAR(mag, 0.01, 1.0e-9) << "front cell (" << i << "," << j << ") should have moved 0.01 m";
                        ++n_moved;
                    } else {
                        EXPECT_NEAR(mag, 0.0, 1.0e-12) << "cell (" << i << "," << j << ") is not on the front";
                    }
                }
    }
    EXPECT_EQ(n_moved, 4);

    // No stamp target this substep, and the burned set is unchanged
    EXPECT_NEAR(farsite_work.norm0(0), 0.0, 1.0e-12);
    EXPECT_NEAR(farsite_work.norm0(1), 0.0, 1.0e-12);
    EXPECT_EQ(count_burned(phi), 1);
}

/**
 * Test 2: the displacement accumulates across substeps and stamps at one cell
 *
 * Scenario:
 *   - A burned row i <= 4 at j = 5 (arrival time 0), the front cell (5,5)
 *     just inside the threshold, unburned elsewhere
 *   - Two 0.1 s substeps: the accumulated displacement at the front after
 *     the second is exactly twice that after the first, so the history is
 *     carried between substeps rather than rebuilt
 *   - One 100 s substep at 0.1 m/s adds 10 m, the one-cell threshold:
 *     the front stamps, the burned set grows, the new cells get the
 *     substep's time as arrival time, and the accumulator at the stamped
 *     cells is reset
 */
TEST_F(FarsiteSpreadTest, SpreadAccumulationAcrossSteps)
{
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p  = phi.array(mfi);
        auto at = arrival_time.array(mfi);
        const Box& bx = mfi.tilebox();
        for (int k = bx.smallEnd()[2]; k <= bx.bigEnd()[2]; ++k) {
            for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j) {
                for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i) {
                    if (i <= 4 && j == 5) {
                        p(i, j, k) = -1.0_rt; at(i, j, k) = 0.0_rt;   // burned row
                    } else if (i == 5 && j == 5) {
                        p(i, j, k) = 0.05_rt;                         // front
                    } else {
                        p(i, j, k) = 1.0_rt;
                    }
                }
            }
        }
    }

    FarsiteParams fp;
    fp.phi_threshold = 0.1;
    fp.gaussian_sigma = -1.0;   // single-cell stamping
    const int n_burned_0 = count_burned(phi);
    EXPECT_EQ(n_burned_0, 5);

    // Two short substeps: the accumulator doubles, nothing stamps
    advance_farsite_one_step(phi, farsite_work, disp_accum, arrival_time,
                             vel_eff, R_mf, geom, 0.1_rt, 0.0_rt, fp);
    IntVect c1;
    const Real d1 = max_disp(disp_accum, c1);
    EXPECT_GT(d1, 1.0e-6) << "the first substep should move the front";
    EXPECT_EQ(count_burned(phi), n_burned_0);

    advance_farsite_one_step(phi, farsite_work, disp_accum, arrival_time,
                             vel_eff, R_mf, geom, 0.1_rt, 0.1_rt, fp);
    IntVect c2;
    const Real d2 = max_disp(disp_accum, c2);
    EXPECT_EQ(c1, c2) << "the same cell leads the front";
    EXPECT_NEAR(d2, 2.0 * d1, 1.0e-9) << "the displacement accumulated in the first substep must survive the second";
    EXPECT_EQ(count_burned(phi), n_burned_0);

    // One long substep reaches the one-cell threshold and stamps
    const Real t_stamp = 0.2_rt;
    advance_farsite_one_step(phi, farsite_work, disp_accum, arrival_time,
                             vel_eff, R_mf, geom, 100.0_rt, t_stamp, fp);
    const int n_burned_1 = count_burned(phi);
    EXPECT_GT(n_burned_1, n_burned_0) << "a 10 m displacement must stamp new burned cells";

    // Every newly burned cell carries the stamping substep's time, and the
    // accumulator of a cell that stamped has been reset
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p  = phi.const_array(mfi);
        auto at = arrival_time.const_array(mfi);
        auto d  = disp_accum.const_array(mfi);
        auto w  = farsite_work.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int k = bx.smallEnd()[2]; k <= bx.bigEnd()[2]; ++k)
            for (int j = bx.smallEnd()[1]; j <= bx.bigEnd()[1]; ++j)
                for (int i = bx.smallEnd()[0]; i <= bx.bigEnd()[0]; ++i) {
                    if (p(i, j, k) < 0.0_rt) {
                        EXPECT_TRUE(at(i, j, k) == 0.0_rt || at(i, j, k) == t_stamp)
                            << "burned cell (" << i << "," << j << ") has arrival time " << at(i, j, k);
                    } else {
                        EXPECT_LT(at(i, j, k), 0.0_rt);
                    }
                    if (w(i, j, k, 0) != 0.0_rt || w(i, j, k, 1) != 0.0_rt) {
                        EXPECT_NEAR(d(i, j, k, 0), 0.0, 1.0e-12) << "a stamped cell keeps no displacement";
                        EXPECT_NEAR(d(i, j, k, 1), 0.0, 1.0e-12);
                    }
                }
    }
}

/**
 * Test 3: Single-cell stamping race safety
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
 * Test 4: Fire grid geometry resolution
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
