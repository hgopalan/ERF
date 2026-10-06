#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Math.H>
#include <cmath>

#include "ERF_FireParams.H"   // ERF_CheneyGouldModel.H reads FireParams without including it
#include "ERF_NumericalSchemes.H"
#include "ERF_Reinitialize.H"
#include "ERF_LevelSetAdvection.H"
#include "ERF_DirectionalRos.H"

/**
 * @file ERF_GTestSplitHamiltonianAdvection.cpp
 * @brief Regression guard for the Hamiltonian-splitting fix
 *        (erf.fire.directional_split_hamiltonian): with
 *        directional_wind_coupling = "advective", R(n)|grad phi| is exactly
 *        the two-term Hamiltonian R0|grad phi| + max(V.grad phi, 0), and
 *        folding it into one Godunov flux built from an estimated front
 *        normal upwinds the wind-driven term along that normal instead of
 *        along V, where its information actually travels (a corner/curvature
 *        -seeded rotation-variance bulge, worst at oblique angles). The split
 *        path (advect_levelset_directional_rk3_split, ERF_DirectionalRos.H)
 *        upwinds the two terms separately, each by its own correct rule.
 *
 * A short, obliquely-oriented (34deg)
 * capsule ignition line is advanced under a Rothermel wind tuned to a head
 * rate of 6 m/s -- deliberately high so a large, easily measured head
 * advance accumulates in a modest number of steps -- and the head's actual
 * position (found by a 1D bilinear-sampled bisection for the zero crossing
 * along the wind ray through the capsule's own centre, i.e. away from its
 * two rounded end corners) is compared to the analytic Rf * T. One test runs
 * the split path alone (advection only, no reinitialisation); a second
 * interleaves the Jiang-Peng reinitialisation every step to confirm the fix
 * holds up with reinit active too, the combination that showed the
 * multi-hundred-metre wing with the baseline (un-split) scheme. The other
 * tests check that per-fuel Rothermel coefficients, a mixed fuel map and the
 * per-cell ROS scale reach the split path.
 */

using namespace amrex;
using namespace fire_levelset;

namespace {

constexpr int  NCELL = 120;
constexpr Real LDOM  = 600.0;
constexpr Real DX    = LDOM / NCELL;   // 5 m

constexpr Real WIND_DEG   = 34.0;   // math convention, 0 = +x
constexpr Real TARGET_RF  = 6.0;    // deliberately high head rate [m/s]
constexpr Real HALF_WIDTH = 25.0;   // capsule half-width [m]
constexpr Real HALF_LEN   = 75.0;   // capsule half-length [m] (150 m line)

/// Rothermel coefficients close to fuel model 1 at 5.5% moisture, wind cap
/// lifted -- identical fixture to ERF_GTestDirectionalShape.cpp's
/// rothermel_state(), reused here rather than re-derived so both tests draw
/// on the same validated constants.
DirectionalRosState rothermel_state ()
{
    DirectionalRosState st;
    st.model = DIRECTIONAL_ROS_ROTHERMEL;
    st.rc = RothermelComputed{};
    st.rc.R0           = 0.0239;
    st.rc.C            = 5.4e-5;
    st.rc.B            = 2.07;
    st.rc.beta_ratio_E = 1.32;
    st.rc.beta         = 0.0010625;
    st.rc.phi_s_const  = 41.1;
    st.rc.U_max_ftmin  = 1.0e6;
    st.rc.wind_conv    = 196.85;
    st.rc.ros_conv     = 1.0;
    st.rc.I_R          = 0.0;
    return st;
}

/// Two-fuel map split at x_boundary. A free function: nvcc rejects an extended
/// device lambda in a gtest case body (the generated TestBody is private).
void fill_two_fuel_map (MultiFab& fuel, Real x_boundary, int slow_code, int fast_code)
{
    for (MFIter mfi(fuel); mfi.isValid(); ++mfi) {
        auto fa = fuel.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real x = (i + Real(0.5)) * DX;
            fa(i, j, k) = (x < x_boundary) ? Real(slow_code) : Real(fast_code);
        });
    }
}

/// Planar front phi = x - x_front0 on every grown cell (free function, as above).
void fill_planar_front (MultiFab& phi, Real x_front0)
{
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.array(mfi);
        ParallelFor(mfi.growntilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            p(i, j, k) = (i + Real(0.5)) * DX - x_front0;
        });
    }
}

/// Wind speed [m/s] whose head-on (cos_theta_wind=1) advective-coupling rate
/// is exactly target_rf, found by bisection against directional_ros_cell
/// itself -- not a hand-derived constant, so this stays correct if the
/// Rothermel coefficients above ever change.
Real wind_speed_for_head_rate (const DirectionalRosState& st, Real target_rf)
{
    Real lo = 0.0, hi = 30.0;
    for (int it = 0; it < 60; ++it) {
        const Real mid = 0.5 * (lo + hi);
        const Real r = directional_ros_cell(st, mid, Real(0.0),
                                             DIRECTIONAL_WIND_COUPLING_ADVECTIVE, mid, Real(0.0));
        if (r < target_rf) { lo = mid; } else { hi = mid; }
    }
    return 0.5 * (lo + hi);
}

struct CapsuleFixture
{
    BoxArray            ba;
    DistributionMapping dm;
    Geometry            geom;
    Real wind_deg, wind_speed, wx, wy;
    Real cx, cy;      // capsule centre (upwind of the domain centre)
    Real px, py;      // capsule line direction (perpendicular to the wind)

    CapsuleFixture (Real target_rf)
    {
        Box domain(IntVect(0, 0, 0), IntVect(NCELL - 1, NCELL - 1, 0));
        ba = BoxArray(domain);   // single box: no ghost-exchange concern here, already covered elsewhere
        dm = DistributionMapping(ba);
        RealBox rb({0.0, 0.0, 0.0}, {LDOM, LDOM, 1.0});
        geom = Geometry(domain, rb, CoordSys::cartesian, {0, 0, 0});

        wind_deg = WIND_DEG;
        const DirectionalRosState st = rothermel_state();
        wind_speed = wind_speed_for_head_rate(st, target_rf);
        const Real th = wind_deg * Math::pi<Real>() / 180.0;
        wx = wind_speed * std::cos(th);
        wy = wind_speed * std::sin(th);

        // Capsule centred at the domain centre, shifted upwind so the head
        // has room to advance without leaving the domain.
        const Real cxh = std::cos(th), cyh = std::sin(th);
        cx = 0.5 * LDOM - 200.0 * cxh;
        cy = 0.5 * LDOM - 200.0 * cyh;
        // Line direction perpendicular to the wind
        px = -cyh; py = cxh;
    }

    /// Signed distance to the capsule (stadium): negative (burned) within
    /// HALF_WIDTH of the line segment [centre - HALF_LEN*p, centre + HALF_LEN*p].
    void capsule (MultiFab& phi) const
    {
        const Real cxL = cx, cyL = cy, pxL = px, pyL = py;
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            auto p = phi.array(mfi);
            const Box& gbx = mfi.growntilebox();
            ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real x = (i + Real(0.5)) * DX;
                const Real y = (j + Real(0.5)) * DX;
                const Real rx = x - cxL, ry = y - cyL;
                const Real t = amrex::Clamp(rx * pxL + ry * pyL, -HALF_LEN, HALF_LEN);
                const Real dx0 = rx - t * pxL, dy0 = ry - t * pyL;
                p(i, j, k) = std::sqrt(dx0 * dx0 + dy0 * dy0) - HALF_WIDTH;
            });
        }
    }

    /// Bilinear sample of a (single-box) cell-centred MultiFab at (x, y).
    Real sample (const MultiFab& phi, Real x, Real y) const
    {
        const Real fi = x / DX - 0.5, fj = y / DX - 0.5;
        int i0 = static_cast<int>(std::floor(fi));
        int j0 = static_cast<int>(std::floor(fj));
        const Real tx = fi - i0, ty = fj - j0;
        Real result = 0.0;
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.fabbox();   // includes ghosts
            if (!bx.contains(IntVect(i0, j0, 0)) || !bx.contains(IntVect(i0 + 1, j0 + 1, 0))) { continue; }
            auto p = phi.const_array(mfi);
            const Real v00 = p(i0, j0, 0),     v10 = p(i0 + 1, j0, 0);
            const Real v01 = p(i0, j0 + 1, 0), v11 = p(i0 + 1, j0 + 1, 0);
            result = v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty)
                   + v01 * (1 - tx) * ty       + v11 * tx * ty;
        }
        ParallelDescriptor::ReduceRealSum(result);   // exactly one rank contributes
        return result;
    }

    /// Distance from the capsule centre, along the wind ray, to phi's zero
    /// crossing -- bisection on the bilinear-sampled field. The ray starts
    /// at the capsule's own centre (far from both rounded end corners), so
    /// this is the head position, not a flank or corner effect.
    Real head_position (const MultiFab& phi) const
    {
        Real lo = 0.0, hi = 350.0;   // must stay inside the domain (see ctor: 200 m upwind margin)
        const Real th = wind_deg * Math::pi<Real>() / 180.0;
        const Real ux = std::cos(th), uy = std::sin(th);
        auto phi_at = [&](Real d) { return sample(phi, cx + d * ux, cy + d * uy); };
        EXPECT_LT(phi_at(lo), 0.0) << "capsule centre should start burned";
        EXPECT_GT(phi_at(hi), 0.0) << "search bound should stay ahead of the front";
        for (int it = 0; it < 40; ++it) {
            const Real mid = 0.5 * (lo + hi);
            if (phi_at(mid) < 0.0) { lo = mid; } else { hi = mid; }
        }
        return 0.5 * (lo + hi);
    }
};

} // namespace

/// Advection only (no reinitialisation), split-Hamiltonian path: the head
/// advances at the model's own analytic Rothermel rate (6 m/s, tuned above)
/// to within 2% over 30 s (180 m) at 34deg -- an oblique angle where the
/// baseline (un-split) scheme's front-normal estimate is least accurate.
TEST(SplitHamiltonianAdvection, HeadRateAt34DegreesMatchesAnalyticRf)
{
    const Real target_rf = TARGET_RF;
    CapsuleFixture f(target_rf);
    const DirectionalRosState st = rothermel_state();

    MultiFab phi(f.ba, f.dm, 1, 3), wind(f.ba, f.dm, 2, 0), slopes(f.ba, f.dm, 2, 0);
    f.capsule(phi);
    wind.setVal(f.wx, 0, 1);
    wind.setVal(f.wy, 1, 1);
    slopes.setVal(0.0);

    const Real dt = 0.1;
    const int  nsteps = 300;   // 30 s simulated -> ~180 m head advance
    const Real eps_visc = 0.4;

    const Real d0 = f.head_position(phi);
    for (int step = 0; step < nsteps; ++step) {
        advect_levelset_directional_rk3_split(phi, wind, slopes, f.geom, dt, eps_visc, st);
        fire_fill_boundary(phi, f.geom);
    }
    const Real d1 = f.head_position(phi);

    const Real traveled = d1 - d0;
    const Real expected = target_rf * (nsteps * dt);
    EXPECT_NEAR(traveled, expected, 0.02 * expected)
        << "traveled=" << traveled << " expected=" << expected;
}

/// Same setup, but with the Jiang-Peng reinitialisation
/// (erf.fire.levelset.reinit_scheme = "jiang_peng") applied every step -- the
/// combination that showed a multi-hundred-metre wing at 34deg with the
/// baseline (un-split) advection scheme. With the split path, head rate still
/// matches the analytic Rf to within 3% (a slightly looser tolerance than the
/// pure advection test, since reinit has its own small bias, which this test
/// does not aim to eliminate -- only to confirm the Hamiltonian-splitting bug
/// is not reintroducing a much larger error on top of it).
TEST(SplitHamiltonianAdvection, ReinitHeadRateAt34DegreesMatchesAnalyticRf)
{
    const Real target_rf = TARGET_RF;
    CapsuleFixture f(target_rf);
    const DirectionalRosState st = rothermel_state();

    MultiFab phi(f.ba, f.dm, 1, 3), wind(f.ba, f.dm, 2, 0), slopes(f.ba, f.dm, 2, 0);
    f.capsule(phi);
    wind.setVal(f.wx, 0, 1);
    wind.setVal(f.wy, 1, 1);
    slopes.setVal(0.0);

    const Real dt   = 0.015;
    const int  nsteps = 2000;  // same 30s/180m as the advection-only test above
    const int  reinit_every = 40;  // 50 reinit calls
    const Real eps_visc = 0.4;
    const Real dtau  = 0.01 * DX;   // matches WRF-Fire's own default, ERF_FireLayer.cpp

    const Real d0 = f.head_position(phi);
    for (int step = 0; step < nsteps; ++step) {
        advect_levelset_directional_rk3_split(phi, wind, slopes, f.geom, dt, eps_visc, st);
        fire_fill_boundary(phi, f.geom);
        if ((step + 1) % reinit_every == 0) {
            reinitialize_phi_jiang_peng(phi, f.geom, 1, dtau);
            fire_fill_boundary(phi, f.geom);
        }
    }
    const Real d1 = f.head_position(phi);

    const Real traveled = d1 - d0;
    const Real expected = target_rf * (nsteps * dt);
    EXPECT_NEAR(traveled, expected, 0.03 * expected)
        << "traveled=" << traveled << " expected=" << expected;
}

/// Per-fuel Rothermel coefficients (erf.fire.rothermel_per_fuel with a spatial
/// fuel map) reach the split path's advective velocities and its isotropic R0.
/// A map holding one fuel code, whose table entry differs from the uniform
/// state, must reproduce a uniform-state run with that entry's coefficients
/// exactly, and must differ from a run that ignores the table.
TEST(SplitHamiltonianAdvection, PerFuelTableIsUsedForVelocityAndR0)
{
    CapsuleFixture f(TARGET_RF);
    const DirectionalRosState st_uniform = rothermel_state();   // what the map must override

    // Table entry for fuel code 2: three times the isotropic rate, same wind response.
    constexpr int FUEL_CODE = 2;
    DirectionalRosState st_fuel = rothermel_state();
    st_fuel.rc.R0 *= Real(3.0);

    std::vector<RothermelComputed> h_tbl(14, st_uniform.rc);
    h_tbl[FUEL_CODE] = st_fuel.rc;
    amrex::Gpu::DeviceVector<RothermelComputed> d_tbl(h_tbl.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_tbl.begin(), h_tbl.end(), d_tbl.begin());

    MultiFab fuel(f.ba, f.dm, 1, 0);
    fuel.setVal(Real(FUEL_CODE));

    MultiFab wind(f.ba, f.dm, 2, 0), slopes(f.ba, f.dm, 2, 0);
    wind.setVal(f.wx, 0, 1);
    wind.setVal(f.wy, 1, 1);
    slopes.setVal(0.0);

    const Real dt = 0.1, eps_visc = 0.4;
    const int  nsteps = 100;   // 10 s: the faster fuel advances ~120 m, inside the domain margin

    auto run = [&](const DirectionalRosState& st, bool use_map) {
        MultiFab phi(f.ba, f.dm, 1, 3);
        f.capsule(phi);
        for (int step = 0; step < nsteps; ++step) {
            if (use_map) {
                advect_levelset_directional_rk3_split(phi, wind, slopes, f.geom, dt, eps_visc, st,
                                                      nullptr, false, fire_levelset::LevelSetGradient{},
                                                      &fuel, d_tbl.data(), static_cast<int>(d_tbl.size()),
                                                      FUEL_SET_ANDERSON13);
            } else {
                advect_levelset_directional_rk3_split(phi, wind, slopes, f.geom, dt, eps_visc, st);
            }
            fire_fill_boundary(phi, f.geom);
        }
        return phi;
    };

    MultiFab phi_map    = run(st_uniform, true);    // uniform state says slow, the map says fast
    MultiFab phi_ref    = run(st_fuel,    false);   // reference: fast coefficients everywhere
    MultiFab phi_ignore = run(st_uniform, false);   // reference: table ignored

    MultiFab diff(f.ba, f.dm, 1, 0);
    MultiFab::Copy(diff, phi_map, 0, 0, 1, 0);
    MultiFab::Subtract(diff, phi_ref, 0, 0, 1, 0);
    EXPECT_EQ(diff.norminf(0), Real(0.0)) << "per-fuel path should match the uniform run with the same coefficients";

    const Real d_map    = f.head_position(phi_map);
    const Real d_ignore = f.head_position(phi_ignore);
    EXPECT_GT(d_map - d_ignore, Real(20.0))
        << "map head=" << d_map << " table-ignored head=" << d_ignore;
}

/// The per-cell ROS scale (front acceleration clock, suppression drops) that
/// the baseline applies to its whole rate must reach the split path too. A
/// uniform 0.5 scale is exactly a halved R0 (and so halved Vw, Vs) -- both are
/// power-of-two multiplies, so the two runs must agree, and must differ from
/// the unscaled run.
TEST(SplitHamiltonianAdvection, RosScaleMatchesScaledCoefficients)
{
    CapsuleFixture f(TARGET_RF);
    const DirectionalRosState st = rothermel_state();
    DirectionalRosState st_half = rothermel_state();
    st_half.rc.R0 *= Real(0.5);

    MultiFab wind(f.ba, f.dm, 2, 0), slopes(f.ba, f.dm, 2, 0);
    wind.setVal(f.wx, 0, 1);
    wind.setVal(f.wy, 1, 1);
    slopes.setVal(0.0);
    MultiFab scale(f.ba, f.dm, 1, 0);
    scale.setVal(Real(0.5));

    const Real dt = 0.1, eps_visc = 0.4;
    const int  nsteps = 100;

    auto run = [&](const DirectionalRosState& s, const MultiFab* ros_scale) {
        MultiFab phi(f.ba, f.dm, 1, 3);
        f.capsule(phi);
        for (int step = 0; step < nsteps; ++step) {
            advect_levelset_directional_rk3_split(phi, wind, slopes, f.geom, dt, eps_visc, s,
                                                  nullptr, false, fire_levelset::LevelSetGradient{},
                                                  nullptr, nullptr, 0, 0, ros_scale);
            fire_fill_boundary(phi, f.geom);
        }
        return phi;
    };

    MultiFab phi_scaled = run(st,      &scale);
    MultiFab phi_half   = run(st_half, nullptr);
    MultiFab phi_full   = run(st,      nullptr);

    MultiFab diff(f.ba, f.dm, 1, 0);
    MultiFab::Copy(diff, phi_scaled, 0, 0, 1, 0);
    MultiFab::Subtract(diff, phi_half, 0, 0, 1, 0);
    EXPECT_LT(diff.norminf(0), Real(1.0e-8)) << "ros_scale should equal scaling R0, Vw and Vs together";

    const Real d_scaled = f.head_position(phi_scaled);
    const Real d_full   = f.head_position(phi_full);
    EXPECT_GT(d_full - d_scaled, Real(10.0))
        << "scaled head=" << d_scaled << " unscaled head=" << d_full;
}

/// Two fuels side by side: a planar front crosses from a slow fuel (fuel code
/// 1, table entry with the fixture's Rothermel coefficients, head rate 6 m/s)
/// into a fast one (code 2, three times the isotropic rate, head rate 18 m/s)
/// at x = 300 m. Each side must advance at its own analytic head rate, and the
/// jump in R0 and V across the fuel boundary must not distort the front:
/// it stays planar to within a cell.
TEST(SplitHamiltonianAdvection, MixedFuelMapAdvancesEachRegionAtItsOwnRate)
{
    CapsuleFixture f(TARGET_RF);
    const DirectionalRosState st_slow = rothermel_state();
    DirectionalRosState st_fast = rothermel_state();
    st_fast.rc.R0 *= Real(3.0);

    constexpr int  SLOW_CODE = 1, FAST_CODE = 2;
    constexpr Real X_BOUNDARY = 300.0, X_FRONT0 = 100.0;
    const Real rate_slow = TARGET_RF;
    const Real rate_fast = Real(3.0) * TARGET_RF;

    std::vector<RothermelComputed> h_tbl(14, st_slow.rc);
    h_tbl[FAST_CODE] = st_fast.rc;
    amrex::Gpu::DeviceVector<RothermelComputed> d_tbl(h_tbl.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_tbl.begin(), h_tbl.end(), d_tbl.begin());

    MultiFab fuel(f.ba, f.dm, 1, 0);
    fill_two_fuel_map(fuel, X_BOUNDARY, SLOW_CODE, FAST_CODE);

    // Wind along +x with the fixture's speed; slow-fuel head rate is then TARGET_RF.
    MultiFab wind(f.ba, f.dm, 2, 0), slopes(f.ba, f.dm, 2, 0);
    wind.setVal(f.wind_speed, 0, 1);
    wind.setVal(Real(0.0), 1, 1);
    slopes.setVal(0.0);

    // Planar front, burned behind it: phi = x - X_FRONT0.
    MultiFab phi(f.ba, f.dm, 1, 3);
    fill_planar_front(phi, X_FRONT0);

    auto front_x = [&](Real y) {
        Real lo = 20.0, hi = LDOM - 20.0;
        for (int it = 0; it < 40; ++it) {
            const Real mid = 0.5 * (lo + hi);
            if (f.sample(phi, mid, y) < 0.0) { lo = mid; } else { hi = mid; }
        }
        return 0.5 * (lo + hi);
    };

    const Real dt = 0.05, eps_visc = 0.4;
    const int  max_steps = 1500;
    const Real y_mid = 0.5 * LDOM;
    std::vector<Real> ts, xs;
    for (int step = 0; step < max_steps; ++step) {
        advect_levelset_directional_rk3_split(phi, wind, slopes, f.geom, dt, eps_visc, st_slow,
                                              nullptr, false, fire_levelset::LevelSetGradient{},
                                              &fuel, d_tbl.data(), static_cast<int>(d_tbl.size()),
                                              FUEL_SET_ANDERSON13);
        fire_fill_boundary(phi, f.geom);
        const Real x = front_x(y_mid);
        ts.push_back((step + 1) * dt);
        xs.push_back(x);
        if (x > 520.0) { break; }
    }
    ASSERT_GT(xs.back(), Real(520.0)) << "front never reached the end of the fast fuel";

    // Least-squares slope of x(t) over the samples whose x lies in [x0, x1].
    auto slope = [&](Real x0, Real x1) {
        Real n = 0, st = 0, sx = 0, stt = 0, stx = 0;
        for (size_t i = 0; i < xs.size(); ++i) {
            if (xs[i] < x0 || xs[i] > x1) { continue; }
            n += 1; st += ts[i]; sx += xs[i]; stt += ts[i] * ts[i]; stx += ts[i] * xs[i];
        }
        EXPECT_GT(n, Real(10.0));
        return (n * stx - st * sx) / (n * stt - st * st);
    };
    const Real r_slow = slope(150.0, 250.0);   // well inside each region, away from the boundary
    const Real r_fast = slope(360.0, 520.0);
    EXPECT_NEAR(r_slow, rate_slow, 0.03 * rate_slow) << "slow fuel rate " << r_slow << " expected " << rate_slow;
    EXPECT_NEAR(r_fast, rate_fast, 0.03 * rate_fast) << "fast fuel rate " << r_fast << " expected " << rate_fast;

    // The front is uniform in y, so it must stay planar across the fuel boundary.
    const Real x_lo = front_x(0.2 * LDOM), x_hi = front_x(0.8 * LDOM);
    EXPECT_LT(std::abs(x_lo - x_hi), DX) << "front skewed: x(0.2L)=" << x_lo << " x(0.8L)=" << x_hi;
}
