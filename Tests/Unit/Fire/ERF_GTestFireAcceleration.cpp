#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>
#include <string>

/// Round-off tolerance of the build precision: 1e-12 in double, 1e-5 in single.
static constexpr double TOL = (sizeof(amrex::Real) == 8) ? 1e-12 : 1e-5;

#include "ERF_FireAcceleration.H"
#include "ERF_FarsiteEllipse.H"
#include "ERF_NumericalSchemes.H"
#include "ERF_Reinitialize.H"
#include "ERF_LevelSetAdvection.H"

/**
 * @file ERF_GTestFireAcceleration.cpp
 * @brief The temporal fire acceleration, R = R_E (1 - exp(-A t)), with its two
 *        clocks (erf.fire.accel.clock).
 *
 * "front": the clock starts at ignition and travels with the front, a cell the
 * front reaches continues the clock of the cells that ignited it, a separate
 * ignition starts its own, and the accelerated rate is written to the unburned
 * cells the front moves into as well as to the burned ones. A planar front
 * from a line of burned cells at a constant R_E then covers
 * R_E (t - (1 - exp(-A t)) / A) on both propagation paths.
 *
 * "legacy": a clock per burned cell, restarted when the cell burns and on any
 * change of R_E, written to burned cells only. Its tests pin that behaviour
 * and show that the front does not follow the curve with it.
 *
 * Fields live on a strip NY = 4 cells wide, periodic across it, non-periodic
 * along it and split into boxes along x, so the extension of the clock into
 * the unburned cells crosses box boundaries.
 */

using namespace amrex;

namespace {

constexpr int NY = 4;

struct Strip
{
    BoxArray            ba;
    DistributionMapping dm;
    Geometry            geom;
    Real                h;

    Strip (int nx, Real dx, int max_grid) : h(dx)
    {
        Box domain(IntVect(0, 0, 0), IntVect(nx - 1, NY - 1, 0));
        ba = BoxArray(domain);
        ba.maxSize(IntVect(max_grid, NY, 1));
        dm = DistributionMapping(ba);
        RealBox rb(0.0_rt, 0.0_rt, 0.0_rt, Real(nx) * dx, Real(NY) * dx, 1.0_rt);
        geom = Geometry(domain, rb, CoordSys::cartesian, {0, 1, 0});
    }
};

/// phi = -1 on columns i < ib (burned), +1 elsewhere, ghosts included
void set_burned_columns (MultiFab& phi, int ib)
{
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.array(mfi);
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            p(i, j, k) = (i < ib) ? -1.0_rt : 1.0_rt;
        });
    }
}

/// Burn column ic (phi = -1)
void burn_column (MultiFab& phi, int ic)
{
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.array(mfi);
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (i == ic) { p(i, j, k) = -1.0_rt; }
        });
    }
}

/// Value of component comp at valid cell (i, j), from whichever rank owns it
Real value_at (const MultiFab& mf, int i, int j, int comp = 0)
{
    Real v = 0.0;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(IntVect(i, j, 0))) { v += mf.const_array(mfi)(i, j, 0, comp); }
    }
    ParallelDescriptor::ReduceRealSum(v);
    return v;
}

AccelerationParams temporal_params (int clock)
{
    AccelerationParams ap;
    ap.enable       = true;
    ap.use_temporal = true;
    ap.clock        = clock;
    ap.A_point      = 6.0;      // 1/min = 0.1 1/s
    ap.A_line       = 6.0;
    ap.perim_limit  = 1.0e12;   // always the point constant
    return ap;
}

/// Distance R_E (t - (1 - exp(-A t)) / A) over R_E t: the accelerated share of the travel
Real accelerated_share (Real A, Real t)
{
    return (t + std::expm1(-A * t) / A) / t;
}

} // namespace

TEST(FireAcceleration, StepMeanFactorIntegratesExactly)
{
    // Steps of any length: the rate integrates to R_E (t - (1 - exp(-A t)) / A).
    const Real A = 0.02;
    const Real dts[] = {0.3_rt, 1.7_rt, 5.0_rt, 0.01_rt, 12.0_rt};
    Real s = 0.0, t = 0.0, dist = 0.0;
    for (int n = 0; n < 200; ++n) {
        const Real dt = dts[n % 5];
        dist += accel_step_mean_factor(s, A * dt) * dt;
        s += A * dt;
        t += dt;
    }
    const Real exact = t + std::expm1(-A * t) / A;
    EXPECT_NEAR(dist, exact, (sizeof(Real) == 8 ? 1.0e-10 : 1.0e-4) * t);
    EXPECT_NEAR(accel_step_mean_factor(2.0_rt, 0.0_rt), 1.0 - std::exp(-2.0), TOL);
}

TEST(FireAcceleration, FrontClockIsZeroBeforeIgnition)
{
    Strip s(60, 10.0, 20);
    MultiFab phi(s.ba, s.dm, 1, 1), R(s.ba, s.dm, 1, 0), state(s.ba, s.dm, 3, 0);
    set_burned_columns(phi, 0);
    R.setVal(0.5);
    state.setVal(0.0);
    apply_fire_acceleration(R, phi, s.geom, temporal_params(accel_clock::front), 1.0, &state, false);
    EXPECT_NEAR(R.max(0), 0.0, TOL);
    EXPECT_NEAR(state.max(1), 0.0, TOL);
}

TEST(FireAcceleration, FrontClockCarriedIntoCellsTheFrontReaches)
{
    // 10 m cells, R_E 0.5 m/s, 1 s steps: the band is 2 + ceil(0.05) = 3 cells.
    Strip s(60, 10.0, 20);
    MultiFab phi(s.ba, s.dm, 1, 1), R(s.ba, s.dm, 1, 0), state(s.ba, s.dm, 3, 0);
    const AccelerationParams ap = temporal_params(accel_clock::front);
    const Real dt = 1.0;
    const Real a  = 0.1 * dt;
    state.setVal(0.0);

    set_burned_columns(phi, 30);
    R.setVal(0.5);
    apply_fire_acceleration(R, phi, s.geom, ap, dt, &state, false);
    const Real f0 = accel_step_mean_factor(0.0, a);
    for (int j = 0; j < NY; ++j) {
        for (int i : {0, 25, 29, 30, 31, 32}) {
            EXPECT_NEAR(value_at(state, i, j, 1), a, TOL) << "i = " << i;
            EXPECT_NEAR(value_at(R, i, j), 0.5 * f0, TOL) << "i = " << i;
        }
        // Past the band: no clock stored, the rate takes the largest progress
        EXPECT_NEAR(value_at(state, 33, j, 1), 0.0, TOL);
        EXPECT_NEAR(value_at(R, 33, j), 0.5 * f0, TOL);
    }

    // The front reaches column 30: it continues the clock, it does not restart it.
    burn_column(phi, 30);
    R.setVal(0.5);
    apply_fire_acceleration(R, phi, s.geom, ap, dt, &state, false);
    const Real f1 = accel_step_mean_factor(a, a);
    for (int j = 0; j < NY; ++j) {
        for (int i : {29, 30, 31, 33}) {
            EXPECT_NEAR(value_at(state, i, j, 1), 2.0 * a, TOL) << "i = " << i;
            EXPECT_NEAR(value_at(R, i, j), 0.5 * f1, TOL) << "i = " << i;
            EXPECT_NEAR(value_at(state, i, j, 2), f1, TOL) << "i = " << i;
        }
        EXPECT_NEAR(value_at(state, 34, j, 1), 0.0, TOL);
    }
}

TEST(FireAcceleration, LegacyClockRestartsInEveryCellThatBurns)
{
    Strip s(60, 10.0, 20);
    MultiFab phi(s.ba, s.dm, 1, 1), R(s.ba, s.dm, 1, 0), state(s.ba, s.dm, 3, 0);
    const AccelerationParams ap = temporal_params(accel_clock::legacy);
    const Real dt = 1.0;
    state.setVal(0.0);

    set_burned_columns(phi, 30);
    for (int n = 0; n < 20; ++n) {
        R.setVal(0.5);
        apply_fire_acceleration(R, phi, s.geom, ap, dt, &state, false);
    }
    EXPECT_NEAR(value_at(R, 29, 0), 0.5 * (1.0 - std::exp(-2.0)), TOL);
    EXPECT_NEAR(value_at(R, 30, 0), 0.5, TOL);   // unburned cells keep R_E

    burn_column(phi, 30);
    R.setVal(0.5);
    apply_fire_acceleration(R, phi, s.geom, ap, dt, &state, false);
    EXPECT_NEAR(value_at(R, 29, 0), 0.5 * (1.0 - std::exp(-2.1)), TOL);
    EXPECT_NEAR(value_at(R, 30, 0), 0.5 * (1.0 - std::exp(-0.1)), TOL);   // restarted
    EXPECT_NEAR(value_at(R, 31, 0), 0.5, TOL);
}

TEST(FireAcceleration, NewIgnitionStartsItsOwnFrontClock)
{
    Strip s(120, 10.0, 40);
    MultiFab phi(s.ba, s.dm, 1, 1), R(s.ba, s.dm, 1, 0), state(s.ba, s.dm, 3, 0);
    const AccelerationParams ap = temporal_params(accel_clock::front);
    const Real dt = 1.0;
    const Real a  = 0.1 * dt;
    state.setVal(0.0);

    set_burned_columns(phi, 30);
    for (int n = 0; n < 50; ++n) {
        R.setVal(0.5);
        apply_fire_acceleration(R, phi, s.geom, ap, dt, &state, false);
    }
    // A spot lands at column 90, far from the front.
    burn_column(phi, 90);
    R.setVal(0.5);
    apply_fire_acceleration(R, phi, s.geom, ap, dt, &state, false);
    EXPECT_NEAR(value_at(state, 29, 1, 1), 51.0 * a, 100.0 * TOL);
    EXPECT_NEAR(value_at(state, 90, 1, 1), a, TOL);
    EXPECT_NEAR(value_at(state, 91, 1, 1), a, TOL);
    EXPECT_NEAR(value_at(R, 90, 1), 0.5 * accel_step_mean_factor(0.0, a), TOL);
    // Away from both fires the rate follows the older one
    EXPECT_NEAR(value_at(R, 60, 1), 0.5 * accel_step_mean_factor(50.0 * a, a), 100.0 * TOL);
}

TEST(FireAcceleration, EquilibriumRateChangeKeepsTheFrontClock)
{
    Strip s(60, 10.0, 20);
    MultiFab phi(s.ba, s.dm, 1, 1), R(s.ba, s.dm, 1, 0), state(s.ba, s.dm, 3, 0);
    const Real dt = 1.0;
    set_burned_columns(phi, 30);

    for (int clock : {accel_clock::front, accel_clock::legacy}) {
        const AccelerationParams ap = temporal_params(clock);
        state.setVal(0.0);
        for (int n = 0; n < 10; ++n) {
            R.setVal(0.5);
            apply_fire_acceleration(R, phi, s.geom, ap, dt, &state, false);
        }
        R.setVal(1.0);   // the wind picks up
        apply_fire_acceleration(R, phi, s.geom, ap, dt, &state, false);
        if (clock == accel_clock::front) {
            EXPECT_NEAR(value_at(state, 10, 0, 1), 1.1, 100.0 * TOL);
            EXPECT_NEAR(value_at(R, 10, 0), accel_step_mean_factor(1.0, 0.1), 100.0 * TOL);
        } else {
            EXPECT_NEAR(value_at(state, 10, 0, 1), dt, TOL);
            EXPECT_NEAR(value_at(R, 10, 0), 1.0 - std::exp(-0.1), TOL);
        }
    }
}

TEST(FireAcceleration, FrontClockContinuesAcrossTheLineConstant)
{
    Strip s(60, 10.0, 20);
    MultiFab phi(s.ba, s.dm, 1, 1), R(s.ba, s.dm, 1, 0), state(s.ba, s.dm, 3, 0);
    AccelerationParams ap = temporal_params(accel_clock::front);
    const Real dt = 1.0;
    set_burned_columns(phi, 30);
    state.setVal(0.0);
    for (int n = 0; n < 10; ++n) {
        R.setVal(0.5);
        apply_fire_acceleration(R, phi, s.geom, ap, dt, &state, false);
    }
    // The perimeter passes the limit: A goes from 0.1 to 1 1/s from the
    // progress reached, not from zero.
    ap.perim_limit = 0.0;
    ap.A_line      = 60.0;
    R.setVal(0.5);
    apply_fire_acceleration(R, phi, s.geom, ap, dt, &state, false);
    EXPECT_NEAR(value_at(state, 10, 0, 1), 2.0, 100.0 * TOL);
    EXPECT_NEAR(value_at(R, 10, 0), 0.5 * accel_step_mean_factor(1.0, 1.0), 100.0 * TOL);
}

namespace {

/// Head advance [m] of a planar FARSITE front from 10 burned columns after nsteps 1 s steps
Real farsite_head_advance (const AccelerationParams& ap, Real R_E, int nsteps, int front_update)
{
    constexpr int  NX = 120;
    constexpr int  IB = 10;
    constexpr Real H  = 5.0;
    Strip s(NX, H, 40);
    MultiFab phi(s.ba, s.dm, 1, 1), work(s.ba, s.dm, 2, 0), disp(s.ba, s.dm, 2, 0);
    MultiFab arrival(s.ba, s.dm, 1, 0), vel(s.ba, s.dm, 2, 0), R(s.ba, s.dm, 1, 0);
    MultiFab state(s.ba, s.dm, 3, 0);
    set_burned_columns(phi, IB);
    for (MFIter mfi(arrival); mfi.isValid(); ++mfi) {
        auto at = arrival.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            at(i, j, k) = (i < IB) ? 0.0_rt : -1.0_rt;
        });
    }
    work.setVal(0.0); disp.setVal(0.0); vel.setVal(0.0); state.setVal(0.0);
    fire_fill_boundary(phi, s.geom);

    FarsiteParams fp;
    fp.front_update = front_update;
    const Real dt = 1.0;
    Real t = 0.0;
    for (int n = 0; n < nsteps; ++n) {
        R.setVal(R_E);
        apply_fire_acceleration(R, phi, s.geom, ap, dt, &state, false);
        advance_fire_subcycle(phi, work, disp, arrival, vel, R, s.geom, dt, t, fp);
        fire_fill_boundary(phi, s.geom);
        t += dt;
    }
    int i_head = -1;
    for (int i = 0; i < NX; ++i) {
        if (value_at(phi, i, 1) < 0.0) { i_head = i; }
    }
    return Real(i_head + 1 - IB) * H;
}

/// Advance [m] of the zero contour of phi = x - 50 m after nsteps 1 s level-set steps
Real levelset_front_advance (const AccelerationParams& ap, Real R_E, int nsteps)
{
    constexpr int  NX = 120;
    constexpr Real H  = 5.0;
    constexpr Real X0 = 50.0;
    Strip s(NX, H, 40);
    MultiFab phi(s.ba, s.dm, 1, 3), vel(s.ba, s.dm, 2, 0), R(s.ba, s.dm, 1, 0);
    MultiFab state(s.ba, s.dm, 3, 0);
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.array(mfi);
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            p(i, j, k) = (i + 0.5_rt) * H - X0;
        });
    }
    vel.setVal(0.0); state.setVal(0.0);

    const Real dt = 1.0;
    for (int n = 0; n < nsteps; ++n) {
        R.setVal(R_E);
        apply_fire_acceleration(R, phi, s.geom, ap, dt, &state, false);
        fire_levelset::advect_levelset_weno5z_rk3(phi, vel, R, s.geom, dt, 0.0);
    }
    for (int i = 0; i + 1 < NX; ++i) {
        const Real p0 = value_at(phi, i, 1);
        const Real p1 = value_at(phi, i + 1, 1);
        if (p0 < 0.0 && p1 >= 0.0) {
            return (i + 0.5_rt) * H + H * (-p0) / (p1 - p0) - X0;
        }
    }
    return -1.0;
}

} // namespace

TEST(FireAcceleration, FarsiteFrontCoversTheAcceleratedDistance)
{
    // A = 0.02 1/s: after 200 s the front has covered 75 % of what it would
    // have without acceleration, on both FARSITE front updates.
    const Real R_E = 0.5;
    const int  nsteps = 200;
    AccelerationParams off;
    AccelerationParams front  = temporal_params(accel_clock::front);
    AccelerationParams legacy = temporal_params(accel_clock::legacy);
    front.A_point = 1.2;
    legacy.A_point = 1.2;
    const Real share = accelerated_share(0.02, Real(nsteps));

    for (int update : {farsite_front::front_cell, farsite_front::legacy}) {
        const std::string tag = (update == farsite_front::front_cell) ? "front_cell_" : "legacy_update_";
        const Real d_off    = farsite_head_advance(off, R_E, nsteps, update);
        const Real d_front  = farsite_head_advance(front, R_E, nsteps, update);
        const Real d_legacy = farsite_head_advance(legacy, R_E, nsteps, update);
        RecordProperty(tag + "d_off",    std::to_string(d_off));
        RecordProperty(tag + "d_front",  std::to_string(d_front));
        RecordProperty(tag + "d_legacy", std::to_string(d_legacy));
        RecordProperty(tag + "expected_front", std::to_string(share * d_off));

        ASSERT_GT(d_off, 80.0) << tag;
        // front_cell: within one and a half 5 m cells of the accelerated share of
        // the unaccelerated advance. legacy update: within 7 %; it resets a cell's
        // accumulated travel when it stamps, dropping the overshoot, on average
        // half a step's travel per cell: nothing without acceleration, where a
        // step's 0.5 m divides the 5 m cell, and up to 5 % with it.
        const Real tol = (update == farsite_front::front_cell) ? 7.5 : 0.07 * d_off;
        EXPECT_NEAR(d_front, share * d_off, tol) << tag << "off " << d_off << " front " << d_front;
        // The legacy clock does not follow it
        EXPECT_GT(std::abs(d_legacy - share * d_off), 15.0) << tag << "off " << d_off << " legacy " << d_legacy;
    }
}

TEST(FireAcceleration, LevelSetFrontCoversTheAcceleratedDistance)
{
    const Real R_E = 0.5;
    const int  nsteps = 100;
    AccelerationParams off;
    AccelerationParams front  = temporal_params(accel_clock::front);
    AccelerationParams legacy = temporal_params(accel_clock::legacy);
    front.A_point = 3.0;
    legacy.A_point = 3.0;

    const Real d_off    = levelset_front_advance(off, R_E, nsteps);
    const Real d_front  = levelset_front_advance(front, R_E, nsteps);
    const Real d_legacy = levelset_front_advance(legacy, R_E, nsteps);
    const Real share    = accelerated_share(0.05, Real(nsteps));
    RecordProperty("d_off",    std::to_string(d_off));
    RecordProperty("d_front",  std::to_string(d_front));
    RecordProperty("d_legacy", std::to_string(d_legacy));
    RecordProperty("expected_front", std::to_string(share * R_E * nsteps));

    EXPECT_NEAR(d_off, R_E * nsteps, 0.05);
    EXPECT_NEAR(d_front, share * R_E * nsteps, 0.05) << "front " << d_front;
    // The legacy clock restarts every cell the front burns and does not follow it
    EXPECT_GT(std::abs(d_legacy - share * R_E * nsteps), 2.0) << "legacy " << d_legacy;
}

TEST(FireAcceleration, LevelSetRosScaleMultipliesTheRebuiltRate)
{
    // The directional level-set paths rebuild the rate in every RK stage; the
    // front clock's factor has to reach them through ros_scale.
    constexpr int  NX = 80;
    constexpr Real H  = 5.0;
    Strip s(NX, H, 40);
    MultiFab slopes(s.ba, s.dm, 2, 0), scale(s.ba, s.dm, 1, 0);
    slopes.setVal(0.0);
    scale.setVal(0.25);

    auto advance = [&](const MultiFab* ros_scale) {
        MultiFab phi(s.ba, s.dm, 1, 3);
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            auto p = phi.array(mfi);
            ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                p(i, j, k) = (i + 0.5_rt) * H - 100.0_rt;
            });
        }
        for (int n = 0; n < 40; ++n) {
            fire_levelset::advect_levelset_rk3_with_fill(phi, slopes, s.geom, 1.0_rt, 0.0_rt,
                [](MultiFab& Rf, const MultiFab&) { Rf.setVal(0.8); },
                nullptr, false, fire_levelset::LevelSetGradient{}, ros_scale);
        }
        return value_at(phi, 20, 1);
    };
    const Real phi_full   = advance(nullptr);
    const Real phi_scaled = advance(&scale);
    // phi at a fixed cell drops by the distance the front covered (to 1e-3 m:
    // the one-sided derivatives feel the domain edge 20 cells away)
    const Real p_init = 20.5 * H - 100.0;
    EXPECT_NEAR(p_init - phi_full, 32.0, 1.0e-3);
    EXPECT_NEAR(p_init - phi_scaled, 8.0, 1.0e-3);
}
