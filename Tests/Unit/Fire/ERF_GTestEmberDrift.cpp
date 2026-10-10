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

#include "ERF_FireParams.H"
#include "ERF_AlbiniSpotting.H"

/**
 * @file ERF_GTestEmberDrift.cpp
 * @brief Firebrands drift on the wind at their height: the reference wind of
 *        wind_ref_ht scaled by the neutral log profile ln(z/z0)/ln(z_ref/z0)
 *        (before October 2026 they drifted on the WAF-reduced midflame wind
 *        throughout a fall of 50-300 m). The landing distance is the forward
 *        Euler sum of that profile over the fall, replicated here; the cap is
 *        the source cell's own fuel on a fuel map; and the launch list is
 *        gathered and sorted, so the landings do not depend on the box
 *        decomposition.
 */

using namespace amrex;

namespace {

constexpr int  NX = 100, NY = 20;
constexpr Real DX = 10.0;
constexpr Real Z_REF = 6.1, Z0 = 0.1, V_T = 0.5, I_B = 100.0, U = 2.0;
constexpr int  N_TRAJ = 20;
constexpr double TOLP = (sizeof(Real) == 8) ? 1.0e-12 : 1.0e-5;

/// Launch block: cells 2-4 by 8-11, intensity I_B (exactly the launch threshold)
bool launches (int i, int j) { return i >= 2 && i <= 4 && j >= 8 && j <= 11; }

struct SpotCase
{
    BoxArray ba; DistributionMapping dm; Geometry geom;
    MultiFab phi, data, wind, intensity, fuel;
    SpotCase (int max_x, int max_y)
    {
        Box domain(IntVect(0, 0, 0), IntVect(NX - 1, NY - 1, 0));
        ba = BoxArray(domain);
        ba.maxSize(IntVect(max_x, max_y, 1));
        dm = DistributionMapping(ba);
        geom = Geometry(domain, RealBox(0.0, 0.0, 0.0, NX * DX, NY * DX, 1.0), CoordSys::cartesian, {0, 0, 0});
        phi.define(ba, dm, 1, 1); data.define(ba, dm, 4, 0); wind.define(ba, dm, 2, 0);
        intensity.define(ba, dm, 1, 0); fuel.define(ba, dm, 1, 0);
        phi.setVal(1.0_rt); data.setVal(0.0_rt); wind.setVal(U, 0, 1); wind.setVal(0.0_rt, 1, 1);
        intensity.setVal(0.0_rt); fuel.setVal(13.0_rt);
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            auto p = phi.array(mfi);
            auto I = intensity.array(mfi);
            const Box& bx = mfi.validbox();
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j)
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i)
                    if (launches(i, j)) { p(i, j, 0) = -1.0_rt; I(i, j, 0) = I_B; }
        }
    }
    FireParams::SpottingParams params () const
    {
        FireParams::SpottingParams sp;
        sp.enable = true; sp.I_B_min = 100.0; sp.P_base = 1.0; sp.terminal_velocity = V_T;
        sp.n_traj_steps = N_TRAJ; sp.spot_radius = 10.0; sp.P_catch = 1.0; sp.random_seed = 7;
        sp.launch_from = "burned";
        return sp;
    }
    /// phi gathered to one box
    MultiFab gathered () const
    {
        BoxArray one(geom.Domain());
        MultiFab out(one, DistributionMapping(one), 1, 0);
        out.ParallelCopy(phi, 0, 0, 1);
        return out;
    }
};

/// The forward Euler drift of a brand lofted to H_z on flat ground
Real euler_drift (Real H_z)
{
    const Real dt = (H_z / V_T) / N_TRAJ, dz = V_T * dt;
    Real z = H_z, d = 0.0;
    for (int n = 0; n < N_TRAJ; ++n) {
        d += albini_wind_profile_factor(z, Z_REF, Z0) * U * dt;
        z -= dz;
        if (z <= 0.0_rt) { break; }
    }
    return d;
}

/// Range of x of the burned cells beyond the launch block (the landings)
void landing_x_range (const MultiFab& phi_one, Real& xmin, Real& xmax, int& n)
{
    xmin = 1.0e30; xmax = -1.0e30; n = 0;
    for (MFIter mfi(phi_one); mfi.isValid(); ++mfi) {
        auto p = phi_one.const_array(mfi);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i)
                if (p(i, j, 0) < 0.0_rt && !launches(i, j)) {
                    const Real x = (i + 0.5_rt) * DX;
                    xmin = amrex::min(xmin, x); xmax = amrex::max(xmax, x); ++n;
                    EXPECT_GE(j, 6); EXPECT_LE(j, 13);   // no drift across the wind
                }
    }
}

} // namespace

TEST(EmberDrift, ProfileFactor)
{
    EXPECT_NEAR(albini_wind_profile_factor(Z_REF, Z_REF, Z0), 1.0, TOLP);
    EXPECT_NEAR(albini_wind_profile_factor(Z0, Z_REF, Z0), 0.0, TOLP);
    EXPECT_NEAR(albini_wind_profile_factor(0.0, Z_REF, Z0), 0.0, TOLP) << "floored at z0";
    EXPECT_NEAR(albini_wind_profile_factor(61.0, Z_REF, Z0), std::log(610.0) / std::log(61.0), TOLP);
    EXPECT_NEAR(albini_wind_profile_factor(200.0, Z_REF, Z0), 1.85, 0.01) << "about twice the 6.1 m wind at 200 m";
}

TEST(EmberDrift, LandingDistanceIsTheLogProfileIntegral)
{
    // 100 kW/m lofts a brand to 12.2 100^{1/3} = 56.6 m; at 0.5 m/s it falls
    // 113 s, drifting 227 m on the uniform 2 m/s wind and about 290 m on the
    // log profile. FM13's cap (1000 m) does not bind.
    const Real H_z = 12.2 * std::cbrt(I_B);
    const Real D = euler_drift(H_z);
    EXPECT_GT(D, 1.2 * U * H_z / V_T) << "the profile carries the brand further than the reference wind";
    SpotCase r(NX, NY);
    const FireParams::SpottingParams sp = r.params();
    compute_albini_spotting(r.phi, r.data, r.wind, r.intensity, r.geom, sp, 1,
                            nullptr, nullptr, 0.5_rt, nullptr, "13", 13, Z_REF, Z0);
    MultiFab one = r.gathered();
    Real xmin, xmax; int n;
    landing_x_range(one, xmin, xmax, n);
    EXPECT_GE(n, 4) << "the brands landed and stamped cells";
    // the launch columns span x = 25..45; landings within spot_radius + a cell
    EXPECT_GE(xmin, 25.0 + D - sp.spot_radius - DX);
    EXPECT_LE(xmax, 45.0 + D + sp.spot_radius + DX);
    EXPECT_LT(xmax - xmin, 20.0 + 2.0 * sp.spot_radius + 2.0 * DX);
}

TEST(EmberDrift, TheCapIsTheSourceCellsFuel)
{
    // on a fuel map the launch cells are FM1 (cap 200 m) while the uniform
    // fuel is FM13: the 290 m drift is cut to 200 m
    SpotCase r(NX, NY);
    r.fuel.setVal(1.0_rt);
    const FireParams::SpottingParams sp = r.params();
    compute_albini_spotting(r.phi, r.data, r.wind, r.intensity, r.geom, sp, 1,
                            nullptr, nullptr, 0.5_rt, &r.fuel, "13", 13, Z_REF, Z0);
    MultiFab one = r.gathered();
    Real xmin, xmax; int n;
    landing_x_range(one, xmin, xmax, n);
    EXPECT_GE(n, 4);
    EXPECT_GE(xmin, 25.0 + 200.0 - sp.spot_radius - DX);
    EXPECT_LE(xmax, 45.0 + 200.0 + sp.spot_radius + DX);
}

TEST(EmberDrift, LandingsDoNotDependOnTheDecomposition)
{
    SpotCase a(NX, NY), b(25, 10);
    const FireParams::SpottingParams sp = a.params();
    compute_albini_spotting(a.phi, a.data, a.wind, a.intensity, a.geom, sp, 3,
                            nullptr, nullptr, 0.5_rt, nullptr, "13", 13, Z_REF, Z0);
    compute_albini_spotting(b.phi, b.data, b.wind, b.intensity, b.geom, sp, 3,
                            nullptr, nullptr, 0.5_rt, nullptr, "13", 13, Z_REF, Z0);
    MultiFab pa = a.gathered(), pb = b.gathered();
    Real diff = 0.0;
    int n = 0;
    for (MFIter mfi(pa); mfi.isValid(); ++mfi) {
        auto x = pa.const_array(mfi);
        auto y = pb.const_array(mfi);
        for (int j = 0; j < NY; ++j)
            for (int i = 0; i < NX; ++i) {
                diff = amrex::max(diff, std::abs(x(i, j, 0) - y(i, j, 0)));
                if (x(i, j, 0) < 0.0_rt && !launches(i, j)) { ++n; }
            }
    }
    EXPECT_GE(n, 4);
    EXPECT_EQ(diff, 0.0_rt) << "one box and eight boxes stamp the same landings";
}
