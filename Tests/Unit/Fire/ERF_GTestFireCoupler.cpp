#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <cmath>
#include <memory>

#include <ERF_Constants.H>
#include <ERF_IndexDefines.H>
#include <ERF_EOS.H>
#include "ERF_FireCoupler.H"
#include "ERF_FireEngine.H"
#include "ERF_FireGrid.H"

/**
 * @file ERF_GTestFireCoupler.cpp
 * @brief The shared fire coupling with an engine that is not the native one
 *
 * A stand-in engine plugs into FireCoupler through the FireEngine interface
 * alone and reports a sensible flux linear in x, a uniform latent flux and a
 * heat per kilogram of its own. The coupling must (closed-form references,
 * nothing taken from the coupling's own code):
 * - average the flux onto the atmospheric columns it came from: a linear
 *   field's mean over a column's C x C fire cells is its value at the column
 *   centre;
 * - hand the engine the wind before the dycore with lagged coupling and after
 *   it with synchronous coupling, and no precipitation it did not ask for;
 * - inject heat whose column integral is the surface flux: with the density
 *   and Exner factors off, c_p sum_k d(rho theta)/dt dz = Q (1 - e^{-z_top/alfg}),
 *   and L_v sum_k d(rho q_v)/dt dz the latent flux likewise;
 * - turn the flux into smoke with the engine's heat per kilogram:
 *   d(rho s)/dt dz = e Q / h in the lowest cell;
 * - with passive coupling and no smoke, never ask the engine for fluxes and
 *   leave every source untouched.
 */

using namespace amrex;

namespace {

constexpr int  NX = 4, NZ = 50, C = 2;
constexpr Real DXY = 100.0, DZ = 10.0;
constexpr Real Q0 = 2.0e4, QX = 30.0;      // sensible flux a + b x [W/m2], x in m
constexpr Real QLAT = 3.0e3;               // latent flux [W/m2]
constexpr Real HEAT_PER_KG = 1.5e7;        // the engine's heat per kg fuel [J/kg]
constexpr Real ALFG = 10.0;                // e-folding height [m]: the profile ends inside the 500 m column
/// Relative round-off tolerance of the build precision
constexpr double REL = (sizeof(Real) == 8) ? 1.0e-9 : 1.0e-4;

/// An engine that is not FireLayer: a fixed flux field, and a record of what it was handed.
class StandInEngine : public FireEngine
{
public:
    const char* name () const override { return "stand-in"; }
    void verify_prerequisites (const ERF&, int) override {}
    void initialize (const ERF&, int, const FireGrid&, const SurfaceLayer*, const MultiFab*) override {}

    /// Fill the flux fields on the fire grid the coupler holds (the test calls it).
    void set_up (const FireGrid& fg, bool with_latent)
    {
        sensible = std::make_unique<MultiFab>(fg.ba, fg.dm, 1, 0);
        const Real dxf = fg.geom.CellSize(0);
        const Real x0  = fg.geom.ProbLo(0);
        for (MFIter mfi(*sensible); mfi.isValid(); ++mfi) {
            auto q = sensible->array(mfi);
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                q(i, j, k) = Q0 + QX * (x0 + (i + 0.5_rt) * dxf);
            });
        }
        if (with_latent) {
            latent = std::make_unique<MultiFab>(fg.ba, fg.dm, 1, 0);
            latent->setVal(QLAT);
        }
    }

    void advance (const FireForcing& forcing) override
    {
        ++n_advance;
        last_xvel   = &forcing.xvel;
        last_yvel   = &forcing.yvel;
        last_precip = forcing.precip_accum_k0;
        last_time   = forcing.time;
        last_dt     = forcing.dt;
    }
    void fill_fluxes (FireFluxes& fluxes) override
    {
        ++n_fill;
        fluxes.sensible = sensible.get();
        fluxes.latent   = latent.get();
    }
    Real smoke_heat_per_kg () const override { return HEAT_PER_KG; }

    std::unique_ptr<MultiFab> sensible, latent;
    int n_advance = 0, n_fill = 0;
    const MultiFab* last_xvel = nullptr;
    const MultiFab* last_yvel = nullptr;
    const MultiFab* last_precip = nullptr;
    Real last_time = -1.0, last_dt = -1.0;
};

/// One level of NX x NX x NZ cells, two boxes in x so the average crosses a box edge.
struct Level
{
    BoxArray ba, ba2d;
    DistributionMapping dm;
    Geometry geom;
    MultiFab S, xold, yold, xnew, ynew, zcc;
    Level ()
    {
        const Box domain(IntVect(0, 0, 0), IntVect(NX - 1, NX - 1, NZ - 1));
        ba = BoxArray(domain);
        ba.maxSize(IntVect(NX / 2, NX, NZ));
        dm = DistributionMapping(ba);
        geom = Geometry(domain, RealBox(0.0, 0.0, 0.0, NX * DXY, NX * DXY, NZ * DZ), CoordSys::cartesian, {1, 1, 0});
        ba2d = ba;
        ba2d.coarsen(IntVect(1, 1, NZ));
        S.define(ba, dm, NVAR_max, 1);
        S.setVal(0.0_rt);
        S.setVal(1.0_rt, Rho_comp, 1, 1);
        S.setVal(getRhoThetagivenP(9.0e4_rt), RhoTheta_comp, 1, 1);
        xold.define(convert(ba, IntVect(1, 0, 0)), dm, 1, 1); xold.setVal(1.0_rt);
        yold.define(convert(ba, IntVect(0, 1, 0)), dm, 1, 1); yold.setVal(1.0_rt);
        xnew.define(convert(ba, IntVect(1, 0, 0)), dm, 1, 1); xnew.setVal(2.0_rt);
        ynew.define(convert(ba, IntVect(0, 1, 0)), dm, 1, 1); ynew.setVal(2.0_rt);
        zcc.define(ba, dm, 1, 1);
        for (MFIter mfi(zcc); mfi.isValid(); ++mfi) {
            auto z = zcc.array(mfi);
            ParallelFor(mfi.growntilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { z(i, j, k) = (k + 0.5_rt) * DZ; });
        }
    }
};

FireCouplingParams coupling (const std::string& type, bool smoke)
{
    FireCouplingParams p;
    p.grid_ratio            = C;
    p.coupling_type         = type;
    p.heat_flux_alfg        = ALFG;
    p.fire_atm_feedback     = 1.0;
    p.inject_latent         = true;
    p.source_mode           = "overwrite";
    p.heat_tendency_density = false;
    p.heat_tendency_exner   = false;
    p.smoke_enable          = smoke;
    p.smoke_emission_factor = 0.02;
    return p;
}

/// The coupler with the stand-in engine, laid out on lev's fire grid.
std::unique_ptr<FireCoupler> make_coupler (const Level& lev, const FireCouplingParams& p,
                                           StandInEngine*& engine, bool with_latent)
{
    auto eng = std::make_unique<StandInEngine>();
    engine = eng.get();
    auto cpl = std::make_unique<FireCoupler>(p, std::move(eng));
    FireGrid fg = create_fire_grid(lev.ba, lev.dm, lev.geom, C);
    cpl->set_fire_grid(0, fg, lev.ba, lev.dm, NZ);
    engine->set_up(cpl->fire_grid(), with_latent);
    return cpl;
}

void advance (FireCoupler& cpl, Level& lev)
{
    cpl.advance(10.0_rt, 0.5_rt, lev.S, lev.xold, lev.yold, lev.xnew, lev.ynew, lev.zcc,
                lev.ba2d, lev.geom, SurfacePrecipAccumulationSources{});
}

void FluxesAverageOntoTheColumnsTheyCameFrom ()
{
    Level lev;
    StandInEngine* eng = nullptr;
    auto cpl = make_coupler(lev, coupling("lagged", false), eng, true);
    advance(*cpl, lev);

    EXPECT_EQ(eng->n_advance, 1);
    EXPECT_EQ(eng->n_fill, 1);
    EXPECT_EQ(eng->last_xvel, &lev.xold) << "lagged coupling hands over the wind before the dycore";
    EXPECT_EQ(eng->last_yvel, &lev.yold);
    EXPECT_EQ(eng->last_precip, nullptr) << "an engine that did not ask gets no precipitation";
    EXPECT_EQ(eng->last_time, 10.0_rt);
    EXPECT_EQ(eng->last_dt, 0.5_rt);

    // the mean of a + b x over a column's C x C fire cells is a + b x_centre;
    // the ghost columns carry the periodic neighbour's value
    const MultiFab& Q  = *cpl->get_Q_atm_prev();
    const MultiFab& QL = *cpl->get_Q_lat_atm_prev();
    for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
        const auto q  = Q.const_array(mfi);
        const auto ql = QL.const_array(mfi);
        const Box gbx = grow(mfi.validbox(), IntVect(1, 1, 0));
        for (int j = gbx.smallEnd(1); j <= gbx.bigEnd(1); ++j) {
            for (int i = gbx.smallEnd(0); i <= gbx.bigEnd(0); ++i) {
                const int ip = ((i % NX) + NX) % NX;
                const Real expect = Q0 + QX * (ip + 0.5_rt) * DXY;
                EXPECT_NEAR(q(i, j, 0), expect, REL * expect) << "column (" << i << ", " << j << ")";
                EXPECT_NEAR(ql(i, j, 0), QLAT, REL * QLAT);
            }
        }
    }
}

void SynchronousCouplingHandsOverTheWindAfterTheDycore ()
{
    Level lev;
    StandInEngine* eng = nullptr;
    auto cpl = make_coupler(lev, coupling("synchronous", false), eng, false);
    advance(*cpl, lev);
    EXPECT_EQ(eng->last_xvel, &lev.xnew);
    EXPECT_EQ(eng->last_yvel, &lev.ynew);
    // no latent flux from the engine: the lagged latent buffer stays zero
    EXPECT_EQ(cpl->get_Q_lat_atm_prev()->norm0(0, 1, IntVect(5, 5, 0)), 0.0_rt);
}

void TheInjectedHeatIsTheSurfaceFlux ()
{
    Level lev;
    StandInEngine* eng = nullptr;
    auto cpl = make_coupler(lev, coupling("lagged", true), eng, true);
    advance(*cpl, lev);

    MultiFab src(lev.ba, lev.dm, NVAR_max, 0);
    src.setVal(0.0_rt);
    cpl->add_sources(src, lev.S, lev.zcc, nullptr, lev.geom, true, 1);

    constexpr Real Cp_fire = 1004.64, L_V = 2.501e6;   // the coupling's c_p and L_v
    const Real frac = 1.0_rt - std::exp(-Real(NZ) * DZ / ALFG);
    const Real ef = 0.02_rt;
    for (MFIter mfi(src); mfi.isValid(); ++mfi) {
        const auto s = src.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                Real th = 0.0, qv = 0.0;
                for (int k = 0; k < NZ; ++k) { th += s(i, j, k, RhoTheta_comp) * DZ; qv += s(i, j, k, RhoQ1_comp) * DZ; }
                const Real q = Q0 + QX * (i + 0.5_rt) * DXY;
                EXPECT_NEAR(Cp_fire * th, q * frac, REL * q) << "sensible, column " << i;
                EXPECT_NEAR(L_V * qv, QLAT * frac, REL * QLAT) << "latent, column " << i;
                // the smoke: e Q / h in the lowest cell, spread over its thickness
                const Real smoke = ef * q / HEAT_PER_KG;
                EXPECT_NEAR(s(i, j, 0, RhoSmoke_comp) * DZ, smoke, REL * smoke) << "smoke, column " << i;
                EXPECT_EQ(s(i, j, 1, RhoSmoke_comp), 0.0_rt);
            }
        }
    }
}

void PassiveWithoutSmokeLeavesTheAtmosphereAlone ()
{
    Level lev;
    StandInEngine* eng = nullptr;
    auto cpl = make_coupler(lev, coupling("passive", false), eng, true);
    advance(*cpl, lev);
    EXPECT_EQ(eng->n_advance, 1);
    EXPECT_EQ(eng->n_fill, 0) << "no heat and no smoke reach the atmosphere: nothing to ask for";
    EXPECT_EQ(cpl->get_Q_atm_prev()->norm0(0, 1, IntVect(5, 5, 0)), 0.0_rt);

    MultiFab src(lev.ba, lev.dm, NVAR_max, 0);
    src.setVal(7.0_rt);
    cpl->add_sources(src, lev.S, lev.zcc, nullptr, lev.geom, true, 1);
    EXPECT_EQ(src.min(RhoTheta_comp), 7.0_rt);
    EXPECT_EQ(src.max(RhoTheta_comp), 7.0_rt);
    EXPECT_EQ(src.min(RhoQ1_comp), 7.0_rt);
    EXPECT_EQ(src.max(RhoSmoke_comp), 7.0_rt);
}

} // namespace

TEST(FireCoupler, FluxesAverageOntoTheColumnsTheyCameFrom)         { FluxesAverageOntoTheColumnsTheyCameFrom(); }
TEST(FireCoupler, SynchronousCouplingHandsOverTheWindAfterTheDycore) { SynchronousCouplingHandsOverTheWindAfterTheDycore(); }
TEST(FireCoupler, TheInjectedHeatIsTheSurfaceFlux)                  { TheInjectedHeatIsTheSurfaceFlux(); }
TEST(FireCoupler, PassiveWithoutSmokeLeavesTheAtmosphereAlone)      { PassiveWithoutSmokeLeavesTheAtmosphereAlone(); }
