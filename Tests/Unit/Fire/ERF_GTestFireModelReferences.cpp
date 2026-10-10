#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>
#include <type_traits>
#include <vector>

#include "ERF_FireParams.H"   // ERF_CheneyGouldModel.H reads FireParams without including it
#include "ERF_Rothermel.H"
#include "ERF_FuelModels.H"
#include "ERF_BehaveModel.H"
#include "ERF_FbpModel.H"
#include "ERF_CheneyGouldModel.H"
#include "ERF_IgnitionSchedule.H"
#include "ERF_FireWindExtract.H"
#include "ERF_FireUtils.H"

/**
 * @file ERF_GTestFireModelReferences.cpp
 * @brief Fire-model values checked against their published sources, and the
 *        defects of the October 2026 audit, each test written to fail on the
 *        code before the fix:
 *  - Anderson (1982) Table 1 loads, 1-h SAV, depths and M_x of the 13 models
 *    (FM2 1-h/10-h loads were half, FM6 100-h 0.115, FM7 live load counted
 *    twice, FM4/FM5 1-h SAV 1739);
 *  - BEHAVE live moisture of extinction with Rothermel's (1972) Eq. 88
 *    live heating number exp(-500/sigma) (the code used -138);
 *  - FBP ISI above 40 km/h (Wotton et al. 2009 eq. 53a; the code kept the
 *    exponential, 7.5x too fast at 100 km/h relative to 60);
 *  - the isotropic Cheney-Gould path reads the deck moisture and curing (it
 *    passed placeholders 10 % and 1);
 *  - a scheduled ignition on the level-set path stamps a signed distance in
 *    metres and leaves the field away from the disc alone (it clamped the
 *    whole domain to <= 1 m);
 *  - the valley wind deflection turns the wind by k_deflect asin(sin theta)
 *    (the sine was divided by the speed factor);
 *  - the k = 0 temperature handed to the fuel-moisture model is the air
 *    temperature theta * Exner (it was theta).
 */

using namespace amrex;

// Relative round-off allowance for amrex::Real: the double-precision checks
// stay at 1e-9, a single-precision build (float Real) gets 1e-6.
constexpr double REAL_RTOL = std::is_same<amrex::Real, float>::value ? 1.0e-6 : 1.0e-9;

namespace {

/// Anderson (1982) GTR INT-122, Table 1, in t/ac: 1-h, 10-h, 100-h, live
struct AndersonRow { int id; double t1, t10, t100, tlive, sav1, depth_ft, mx_pct; };
const AndersonRow anderson1982[13] = {
    { 1, 0.74,  0.00,  0.00, 0.00, 3500, 1.0, 12},
    { 2, 2.00,  1.00,  0.50, 0.50, 3000, 1.0, 15},
    { 3, 3.01,  0.00,  0.00, 0.00, 1500, 2.5, 25},
    { 4, 5.01,  4.01,  2.00, 5.01, 2000, 6.0, 20},
    { 5, 1.00,  0.50,  0.00, 2.00, 2000, 2.0, 20},
    { 6, 1.50,  2.50,  2.00, 0.00, 1750, 2.5, 25},
    { 7, 1.13,  1.87,  1.50, 0.37, 1750, 2.5, 40},
    { 8, 1.50,  1.00,  2.50, 0.00, 2000, 0.2, 30},
    { 9, 2.92,  0.41,  0.15, 0.00, 2500, 0.2, 25},
    {10, 3.01,  2.00,  5.01, 2.00, 2000, 1.0, 25},
    {11, 1.50,  4.51,  5.51, 0.00, 1500, 1.0, 15},
    {12, 4.01, 14.03, 16.53, 0.00, 1500, 2.3, 20},
    {13, 7.01, 23.04, 28.05, 0.00, 1500, 3.0, 25},
};
constexpr double TPA_TO_LBFT2 = 2000.0 / 43560.0;

/// One-cell 2D fields for the per-cell kernels
struct OneCell {
    Box domain{IntVect(0, 0, 0), IntVect(0, 0, 0)};
    BoxArray ba{domain};
    DistributionMapping dm{ba};
};

/// Fill a 1-component field with a planar phi = x + offset (metres)
void fill_phi_plus_x (MultiFab& phi, const Geometry& geom, Real offset)
{
    const auto plo = geom.ProbLoArray();
    const auto dx  = geom.CellSizeArray();
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            p(i, j, k) = plo[0] + (i + Real(0.5)) * dx[0] + offset;
        });
    }
}

/// Copy cell (i,j) of a field to the host (one rank: the box is local)
Real host_value (const MultiFab& mf, int i, int j, int comp = 0)
{
    Real v = 0.0;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        if (!mfi.validbox().contains(IntVect(i, j, 0))) { continue; }
        FArrayBox h(Box(IntVect(i, j, 0), IntVect(i, j, 0)), 1, The_Pinned_Arena());
        h.copy<RunOn::Device>(mf[mfi], Box(IntVect(i, j, 0), IntVect(i, j, 0)), comp,
                              Box(IntVect(i, j, 0), IntVect(i, j, 0)), 0, 1);
        Gpu::streamSynchronize();
        v = h(IntVect(i, j, 0));
    }
    ParallelDescriptor::ReduceRealSum(v);
    return v;
}

/// Set a 2-component one-cell field
void set2 (MultiFab& mf, Real a, Real b) { mf.setVal(a, 0, 1); mf.setVal(b, 1, 1); }

} // namespace

// --------------------------------------------------------------------------
TEST(FireModelReferences, AndersonTableOneMatchesTheSource)
{
    for (const auto& r : anderson1982) {
        const FuelModelParams fp = get_anderson_fuel_params(r.id);
        const double tol = 6e-4;   // the table rounds the converted loads to 0.001 lb/ft2
        EXPECT_NEAR(fp.w_d1,   r.t1   * TPA_TO_LBFT2, tol) << "FM" << r.id << " 1-h load";
        EXPECT_NEAR(fp.w_d10,  r.t10  * TPA_TO_LBFT2, tol) << "FM" << r.id << " 10-h load";
        EXPECT_NEAR(fp.w_d100, r.t100 * TPA_TO_LBFT2, tol) << "FM" << r.id << " 100-h load";
        // one live load in Anderson's table, whichever class carries it
        EXPECT_NEAR(fp.w_lh + fp.w_lw, r.tlive * TPA_TO_LBFT2, tol) << "FM" << r.id << " live load";
        EXPECT_NEAR(fp.sigma_d1, r.sav1, 0.5) << "FM" << r.id << " 1-h SAV";
        EXPECT_NEAR(fp.delta, r.depth_ft, REAL_RTOL * std::max(1.0, r.depth_ft)) << "FM" << r.id << " depth";
        EXPECT_NEAR(fp.Mx, r.mx_pct / 100.0, REAL_RTOL) << "FM" << r.id << " moisture of extinction";
    }
}

// --------------------------------------------------------------------------
TEST(FireModelReferences, BehaveLiveExtinctionUsesTheLiveHeatingNumber)
{
    // One dead (1-h) and one live (woody) class, so the category-mean SAVs
    // equal the class SAVs and Rothermel's W' is exact:
    //   W' = sum_dead w exp(-138/sigma) / sum_live w exp(-500/sigma)   (Eq. 88)
    //   M_x,live = 2.9 W' (1 - M_f'/M_x,dead) - 0.226                  (Eq. 88)
    FuelModelParams fp{};
    fp.w_d1 = 0.10; fp.w_d10 = 0.0; fp.w_d100 = 0.0; fp.w_lh = 0.0; fp.w_lw = 0.10;
    fp.sigma_d1 = 2000.0; fp.sigma_lh = 1500.0; fp.sigma_lw = 1500.0;
    fp.delta = 2.0; fp.Mx = 0.25; fp.heat_content = 8000.0; fp.rho_p = 32.0;
    const double M1 = 0.06, Mlive = 1.0;

    const double Wp   = (0.10 * std::exp(-138.0 / 2000.0)) / (0.10 * std::exp(-500.0 / 1500.0));
    const double mx_l = std::max(2.9 * Wp * (1.0 - M1 / 0.25) - 0.226, 0.25);
    const double r    = std::min(1.0, Mlive / mx_l);
    const double eta  = std::max(0.0, 1.0 - 2.59*r + 5.11*r*r - 3.52*r*r*r);

    const BehaveState bs = compute_behave_state(fp, Real(M1), Real(0.08), Real(0.10), Real(Mlive), Real(Mlive),
                                                Real(0.30), Real(1.20));
    // the -138 exponent on the live side gives 0.543 here, -500 gives 0.561
    EXPECT_NEAR(bs.etam_live, eta, 1e-6) << "live moisture damping";
    EXPECT_GT(bs.etam_live, Real(0.555));
}

// --------------------------------------------------------------------------
TEST(FireModelReferences, BehaveDeadOnlyFuelIsFinite)
{
    // FM1 has no live load: every live weight is 0/0 if evaluated, which the
    // fpe-trap CTest catches; here the state must at least be finite.
    const BehaveState bs = compute_behave_state(get_anderson_fuel_params(1), Real(0.06), Real(0.08),
                                                Real(0.10), Real(0.9), Real(0.9), Real(0.3), Real(1.2));
    EXPECT_TRUE(std::isfinite(static_cast<double>(bs.r_0)));
    EXPECT_GT(bs.r_0, Real(0.0));
}

// --------------------------------------------------------------------------
TEST(FireModelReferences, FbpIsiSaturatesAboveFortyKmPerHour)
{
    const Real fF = fbp_ffmc_function(Real(90.0));
    // the two branches of f(W) meet at 40 km/h (FBP 1992 eq. 53 / Wotton 2009 eq. 53a)
    const Real below = fbp_isi(fF, Real(39.999));
    const Real above = fbp_isi(fF, Real(40.001));
    EXPECT_NEAR(above / below, 1.0, 2e-3);
    // above 40 km/h the wind function saturates at 12: ISI(100)/ISI(60) is
    // 1.07 (the plain exponential gives exp(0.05039*40) = 7.5)
    const Real r = fbp_isi(fF, Real(100.0)) / fbp_isi(fF, Real(60.0));
    EXPECT_LT(r, Real(1.2));
    EXPECT_GT(r, Real(1.0));
    EXPECT_LT(fbp_isi(fF, Real(200.0)), Real(0.208 * fF * 12.0 * 1.0001));
}

// --------------------------------------------------------------------------
TEST(FireModelReferences, CheneyGouldIsotropicPathUsesTheDeckMoisture)
{
    FireParams::CheneyGouldParams cgp;
    cgp.moisture = 20.0;
    cgp.curing   = 0.8;
    const CheneyGouldComputed cgc = compute_cheney_gould_params(cgp);

    OneCell c;
    MultiFab ros(c.ba, c.dm, 1, 0), wind(c.ba, c.dm, 2, 0);
    set2(wind, Real(4.0), Real(3.0));   // |U| = 5 m/s
    fill_cheney_gould_ros(ros, wind, cgc);

    const Real expect = cheney_gould_ros(Real(5.0), cgc.ros_backing, Real(20.0), Real(0.8));
    const Real placeholder = cheney_gould_ros(Real(5.0), cgc.ros_backing, Real(10.0), Real(1.0));
    ASSERT_GT(std::abs(expect - placeholder), Real(0.05) * expect) << "moisture must change the rate";
    EXPECT_NEAR(host_value(ros, 0, 0), expect, 1e-6 * expect);
}

// --------------------------------------------------------------------------
TEST(FireModelReferences, ScheduledIgnitionOnTheLevelSetKeepsTheSignedDistance)
{
    // 100 x 100 m, 1 m cells, unburned planar field phi = x + 50 (metres)
    const Box domain(IntVect(0, 0, 0), IntVect(99, 99, 0));
    RealBox rb({0.0, 0.0, 0.0}, {100.0, 100.0, 1.0});
    Geometry geom(domain, rb, 0, {0, 0, 0});
    BoxArray ba(domain); ba.maxSize(32);
    DistributionMapping dm(ba);
    MultiFab phi(ba, dm, 1, 0);
    fill_phi_plus_x(phi, geom, Real(50.0));

    IgnitionSchedule sched;
    IgnitionEvent ev{};
    ev.time_s = 1.0; ev.cx = 80.0; ev.cy = 50.0; ev.radius = 5.0;
    ev.source_type = 0; ev.priority = 5; ev.suppress_if_burning = false; ev.fired = false;
    sched.events.push_back(ev);

    const int n = apply_scheduled_ignitions(phi, geom, sched, Real(1.0), Real(0.0), /*normalized=*/false);
    EXPECT_EQ(n, 1);
    // every cell is the union of the two signed distances, min(x + 50, |x - c| - r),
    // in metres; the clamped stamp turned both of these into <= 1 m
    auto expect_at = [](int i, int j) {
        const double x = i + 0.5, y = j + 0.5;
        return std::min(x + 50.0, std::sqrt((x-80.0)*(x-80.0) + (y-50.0)*(y-50.0)) - 5.0);
    };
    EXPECT_NEAR(host_value(phi, 10, 50), expect_at(10, 50), REAL_RTOL * 100.0);   // 60.5: the plane
    EXPECT_NEAR(host_value(phi, 60, 20), expect_at(60, 20), REAL_RTOL * 100.0);   // 30.36: the disc's distance
    EXPECT_GT(host_value(phi, 60, 20), Real(30.0));
    // the disc centre cell carries dist - r in metres
    const Real d = std::sqrt(0.5*0.5 + 0.5*0.5);
    EXPECT_NEAR(host_value(phi, 80, 50), d - 5.0, REAL_RTOL * 100.0);
    // the event does not fire twice
    EXPECT_EQ(apply_scheduled_ignitions(phi, geom, sched, Real(2.0), Real(1.0), false), 0);
}

// --------------------------------------------------------------------------
TEST(FireModelReferences, ValleyDeflectionTurnsByTheWindSlopeAngle)
{
    // A valley cell (curvature < -0.01) on a 0.2 slope along x, wind (-1, -3):
    // cos(wind, upslope) = -0.316 (neither windward nor lee), so the valley
    // speed factor applies and the wind turns by k_deflect * asin(sin theta),
    // sin theta = (s x U)_z / (|s| |U|) = -3/sqrt(10).
    OneCell c;
    MultiFab wind(c.ba, c.dm, 2, 0), slopes(c.ba, c.dm, 2, 0), curv(c.ba, c.dm, 1, 0);
    set2(wind, Real(-1.0), Real(-3.0));
    set2(slopes, Real(0.2), Real(0.0));
    curv.setVal(-0.02);
    const Real k_valley = 0.8, k_deflect = 0.3;
    apply_farsite_terrain_wind(wind, slopes, curv, Real(1.5), Real(0.6), k_valley, k_deflect);

    const double factor = k_valley + (1.0 - k_valley) * (1.0 - 0.2);
    const double ang0   = std::atan2(-3.0, -1.0);
    const double turn   = k_deflect * std::asin(-3.0 / std::sqrt(10.0));
    const double ux = host_value(wind, 0, 0, 0), uy = host_value(wind, 0, 0, 1);
    EXPECT_NEAR(std::sqrt(ux*ux + uy*uy), factor * std::sqrt(10.0), REAL_RTOL * 10.0) << "valley speed factor";
    const double pi = amrex::Math::pi<double>();
    double dang = std::atan2(uy, ux) - ang0;
    if (dang >  pi) { dang -= 2.0 * pi; }
    if (dang < -pi) { dang += 2.0 * pi; }
    // the sine divided by the factor (0.96) gave a turn 0.050 rad larger
    EXPECT_NEAR(dang, turn, 1e-6) << "deflection angle";
}

// --------------------------------------------------------------------------
TEST(FireModelReferences, MoistureDriverTemperatureIsTheAirTemperature)
{
    // Dry air, rho = 1 kg/m3, theta = 300 K: the air temperature satisfies
    // the ideal gas law with the pressure of the state, and lies below theta
    // (the pressure is below p0). theta itself fails both.
    OneCell c;
    MultiFab S(c.ba, c.dm, RhoTheta_comp + 1, 0), T(c.ba, c.dm, 1, 0);
    S.setVal(1.0, Rho_comp, 1);
    S.setVal(300.0, RhoTheta_comp, 1);
    compute_t_from_conservative(T, S);
    const Real Tk = host_value(T, 0, 0);
    const Real p  = getPgivenRTh(Real(300.0), Real(0.0));
    EXPECT_NEAR(p, 1.0 * R_d * Tk, REAL_RTOL * p) << "p = rho R_d T";
    EXPECT_LT(Tk, Real(299.0));
}
