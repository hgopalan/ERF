#include <gtest/gtest.h>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>
#include <cmath>
#include <vector>

#include "ERF_FireDiagnostics.H"
#include "ERF_FireWindExtract.H"
#include "ERF_FuelMap.H"

/**
 * @file ERF_GTestFireDiagnosticsPerFuel.cpp
 * @brief Byram's fireline intensity and the wind adjustment factor on a fuel
 *        map read the cell's own fuel model, not erf.fire.fuel_model_id.
 *
 * Byram is I_B = h (w_0 - w) R. Under erf.fire.fuel_map.load_from_map each cell
 * starts at its own model's load, so a uniform w_0 differenced against a
 * per-cell w is a different fire: where the uniform model is the lighter of the
 * two the difference clamps and the intensity is exactly zero. The wind
 * adjustment factor is a function of the fuel bed depth alone, so on a map it
 * is a field rather than one number.
 */

using namespace amrex;

namespace {

constexpr int NX = 6;
constexpr int NY = 4;

// Scott-Burgan codes with loads and bed depths far enough apart that a value
// taken from the wrong one cannot pass as the right one.
constexpr int CODE_LIGHT = 101;  // GR1, a light grass
constexpr int CODE_HEAVY = 165;  // TU5, a heavy timber-understory bed
constexpr int CODE_NB    = 98;   // NB8, open water: no load at all

constexpr Real M_LIVE   = 0.6;
constexpr Real ROS_MS   = 0.25;
constexpr Real BURNED   = -1.0;  // phi < 0

int code_at (int i, int /*j*/)
{
    if (i == NX - 1) { return CODE_NB; }
    return (i < NX / 2) ? CODE_LIGHT : CODE_HEAVY;
}

FuelModelParams params_of (int code)
{
    return get_fuel_params(code, FUEL_SET_SCOTT_BURGAN40, M_LIVE);
}

Real load_of (int code)
{
    return sb40_nonburnable(code) ? 0.0 : fuel_total_load_kg_m2(params_of(code));
}

Real tol () { return (sizeof(Real) == 8) ? 1.0e-11 : 1.0e-4; }

// A burned grid with each cell's own fuel code, a uniform rate of spread, and a
// remaining load a fixed fraction of what the cell started with.
struct Grid
{
    static constexpr Real REMAIN_FRACTION = 0.25;

    BoxArray ba;
    DistributionMapping dm;
    MultiFab model, phi, ros, load, I_B, L;

    Grid ()
    {
        const Box domain(IntVect(0, 0, 0), IntVect(NX - 1, NY - 1, 0));
        ba.define(domain);
        ba.maxSize(IntVect(2, 2, 1));   // several boxes: the lookup is per cell
        dm.define(ba);
        model.define(ba, dm, 1, 0);
        phi.define(ba, dm, 1, 0);
        ros.define(ba, dm, 1, 0);
        load.define(ba, dm, 1, 0);
        I_B.define(ba, dm, 1, 0);
        L.define(ba, dm, 1, 0);

        phi.setVal(BURNED);
        ros.setVal(ROS_MS);
        I_B.setVal(-1.0);
        L.setVal(-1.0);

        for (MFIter mfi(model); mfi.isValid(); ++mfi) {
            auto const& m = model.array(mfi);
            auto const& w = load.array(mfi);
            const Box& bx = mfi.validbox();
            for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
                for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                    const int c = code_at(i, j);
                    m(i, j, 0) = static_cast<Real>(c);
                    w(i, j, 0) = REMAIN_FRACTION * load_of(c);
                }
            }
        }
    }
};

}  // namespace

// The three codes have to be far enough apart for the checks below to mean
// something; if the published table ever changes they stop being a test.
TEST(FireDiagnosticsPerFuel, TheCodesAreDistinguishable)
{
    EXPECT_GT(load_of(CODE_HEAVY), 4.0 * load_of(CODE_LIGHT));
    EXPECT_EQ(load_of(CODE_NB), 0.0);
    EXPECT_GT(std::abs(params_of(CODE_HEAVY).delta - params_of(CODE_LIGHT).delta), 0.2);
}

TEST(FireDiagnosticsPerFuel, ByramUsesTheCellsOwnLoadAndHeat)
{
    Grid g;
    fill_fire_diagnostics(g.I_B, g.L, g.phi, g.ros, g.load,
                          /*fuel_load_initial_kg_m2=*/load_of(CODE_LIGHT),
                          /*h_fuel_kJ_per_kg=*/params_of(CODE_LIGHT).heat_content * 2.326,
                          &g.model, FUEL_SET_SCOTT_BURGAN40, M_LIVE,
                          /*fp_tbl=*/nullptr, /*fp_tbl_size=*/0,
                          /*load_from_map=*/true, /*sb40=*/true);

    for (MFIter mfi(g.I_B); mfi.isValid(); ++mfi) {
        auto const& I = g.I_B.const_array(mfi);
        auto const& Lf = g.L.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                const int c = code_at(i, j);
                const Real w0 = load_of(c);
                const Real h = params_of(c).heat_content * 2.326;
                const Real want = h * (w0 - Grid::REMAIN_FRACTION * w0) * ROS_MS;
                EXPECT_NEAR(I(i, j, 0), want, tol() * std::max(want, Real(1.0)))
                    << "cell (" << i << ", " << j << "), code " << c;
                EXPECT_NEAR(Lf(i, j, 0), compute_flame_length_m(want),
                            tol() * std::max(compute_flame_length_m(want), Real(1.0)));
            }
        }
    }
}

// The defect this guards: with the uniform initial load the heavy cells consume
// a load smaller than what they still hold, the difference clamps, and the
// intensity of the heavier half of the grid is exactly zero.
TEST(FireDiagnosticsPerFuel, TheUniformInitialLoadClampsTheHeavyCellsToZero)
{
    Grid g;
    fill_fire_diagnostics(g.I_B, g.L, g.phi, g.ros, g.load,
                          load_of(CODE_LIGHT),
                          params_of(CODE_LIGHT).heat_content * 2.326,
                          &g.model, FUEL_SET_SCOTT_BURGAN40, M_LIVE,
                          nullptr, 0,
                          /*load_from_map=*/false, /*sb40=*/true);

    int zero_heavy = 0, positive_heavy = 0;
    for (MFIter mfi(g.I_B); mfi.isValid(); ++mfi) {
        auto const& I = g.I_B.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if (code_at(i, j) != CODE_HEAVY) { continue; }
                if (I(i, j, 0) == 0.0) { ++zero_heavy; } else { ++positive_heavy; }
            }
        }
    }
    EXPECT_GT(zero_heavy, 0);
    EXPECT_EQ(positive_heavy, 0);
}

// Non-burnable Scott-Burgan codes start with no load, exactly as
// fill_fuel_load_from_map() starts them, so they can raise no intensity even
// though get_fuel_params() hands back a bed for them.
TEST(FireDiagnosticsPerFuel, NonBurnableCodesRaiseNoIntensity)
{
    Grid g;
    fill_fire_diagnostics(g.I_B, g.L, g.phi, g.ros, g.load,
                          load_of(CODE_LIGHT),
                          params_of(CODE_LIGHT).heat_content * 2.326,
                          &g.model, FUEL_SET_SCOTT_BURGAN40, M_LIVE,
                          nullptr, 0, /*load_from_map=*/true, /*sb40=*/true);

    int nb_cells = 0;
    for (MFIter mfi(g.I_B); mfi.isValid(); ++mfi) {
        auto const& I = g.I_B.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if (code_at(i, j) != CODE_NB) { continue; }
                EXPECT_EQ(I(i, j, 0), 0.0) << "cell (" << i << ", " << j << ")";
                ++nb_cells;
            }
        }
    }
    EXPECT_EQ(nb_cells, NY);
}

// erf.fire.fuel_map.sb40_crosswalk reads the Scott-Burgan raster but resolves
// its codes in the Anderson set, where 98 is not a code at all and falls through
// to model 1 with a real load. fill_fuel_load_from_map() still starts those
// cells at zero, on the sb40 flag rather than the set, and the initial load
// here has to agree with it or the water would burn.
TEST(FireDiagnosticsPerFuel, CrosswalkStillZeroesTheNonBurnableCodes)
{
    ASSERT_GT(fuel_total_load_kg_m2(get_fuel_params(CODE_NB, FUEL_SET_ANDERSON13)), 0.0)
        << "code " << CODE_NB << " no longer falls through to a loaded Anderson model; "
        << "this test needs another non-burnable code";

    Grid g;
    fill_fire_diagnostics(g.I_B, g.L, g.phi, g.ros, g.load,
                          load_of(CODE_LIGHT),
                          params_of(CODE_LIGHT).heat_content * 2.326,
                          &g.model, FUEL_SET_ANDERSON13, M_LIVE,
                          nullptr, 0, /*load_from_map=*/true, /*sb40=*/true);

    for (MFIter mfi(g.I_B); mfi.isValid(); ++mfi) {
        auto const& I = g.I_B.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if (code_at(i, j) != CODE_NB) { continue; }
                EXPECT_EQ(I(i, j, 0), 0.0) << "cell (" << i << ", " << j << ")";
            }
        }
    }
}

// Without a fuel field nothing changes: the uniform scalars are still what a
// deck with no map gets.
TEST(FireDiagnosticsPerFuel, NoFuelMapKeepsTheUniformScalars)
{
    Grid g;
    const Real w0 = load_of(CODE_LIGHT);
    const Real h = params_of(CODE_LIGHT).heat_content * 2.326;
    fill_fire_diagnostics(g.I_B, g.L, g.phi, g.ros, g.load, w0, h);

    for (MFIter mfi(g.I_B); mfi.isValid(); ++mfi) {
        auto const& I = g.I_B.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                const Real w = Grid::REMAIN_FRACTION * load_of(code_at(i, j));
                const Real want = (w0 > w) ? h * (w0 - w) * ROS_MS : 0.0;
                EXPECT_NEAR(I(i, j, 0), want, tol() * std::max(want, Real(1.0)))
                    << "cell (" << i << ", " << j << ")";
            }
        }
    }
}

// The wind adjustment factor of the two formulas separates these beds, which is
// what apply_waf_to_wind() evaluates per cell on a map. A factor built from one
// model would apply the left column's number to the right column's fuel.
TEST(FireDiagnosticsPerFuel, TheWafSeparatesTheBedDepths)
{
    const Real d_light = params_of(CODE_LIGHT).delta;
    const Real d_heavy = params_of(CODE_HEAVY).delta;

    const Real a_light = compute_waf_unsheltered(d_light);
    const Real a_heavy = compute_waf_unsheltered(d_heavy);
    EXPECT_GT(std::abs(a_heavy - a_light), 0.02 * a_light);

    const Real b_light = compute_waf_behaviorplus(d_light);
    const Real b_heavy = compute_waf_behaviorplus(d_heavy);
    EXPECT_GT(std::abs(b_heavy - b_light), 0.0);

    // Anderson 1 against Anderson 13, the pair the FireCustomFuel identity deck
    // uses: 27 % of the midflame wind.
    const Real a1 = compute_waf_unsheltered(get_fuel_params(1, FUEL_SET_ANDERSON13).delta);
    const Real a13 = compute_waf_unsheltered(get_fuel_params(13, FUEL_SET_ANDERSON13).delta);
    EXPECT_NEAR(a13 / a1, 1.2666, 1.0e-3);
}
