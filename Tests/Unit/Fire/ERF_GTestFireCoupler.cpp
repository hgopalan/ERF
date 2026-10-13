#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <ERF_Constants.H>
#include <ERF_IndexDefines.H>
#include <ERF_EOS.H>
#include "ERF_FireCoupler.H"
#include "ERF_FireEngine.H"
#include "ERF_FireGrid.H"
#include "ERF_FireParams.H"

/**
 * @file ERF_GTestFireCoupler.cpp
 * @brief The shared fire coupling with an engine that is not the native one
 *
 * A stand-in engine plugs into FireCoupler through the FireEngine interface
 * alone and reports a sensible flux linear in x, a uniform latent flux and a
 * heat per kilogram of its own. Every expected value is a closed form, taken
 * from the physics and never from the coupling's code:
 * - the flux averages onto the columns it came from: a linear field's mean
 *   over a column's C x C fire cells is its value at the column centre, also
 *   for a fire region that does not start at the domain corner;
 * - the engine gets the wind before the dycore with lagged coupling and after
 *   it with synchronous coupling, and the rain only when it asks for it;
 * - the injected heat integrates to the surface flux: with the density factor
 *   on and feedback f, c_p sum_k d(rho theta)/dt dz = rho f Q (1 - e^{-z_top/alfg}),
 *   z measured from the first cell centre, and L_v sum_k d(rho q_v)/dt dz the
 *   same for the latent flux; the smoke is e Q / h in the lowest cell;
 * - source_mode = add keeps the other sources, overwrite replaces the theta
 *   and vapour sources only;
 * - passive coupling injects no heat (and, without smoke, asks for no flux),
 *   and inject_latent = false drops a latent flux the engine reports;
 * - the erf.fire.* coupling keys reach FireCouplingParams unchanged.
 */

using namespace amrex;

namespace {

constexpr int  NX = 6, NY = 4, NZ = 50, C = 2;
constexpr Real DXY = 100.0, DZ = 10.0;
constexpr Real Q0 = 2.0e4, QX = 30.0;      // sensible flux a + b x [W/m2], x in m
constexpr Real QLAT = 3.0e3;               // latent flux [W/m2]
constexpr Real HEAT_PER_KG = 1.5e7;        // the engine's heat per kg fuel [J/kg]
constexpr Real ALFG = 100.0;               // e-folding height [m]: 1 - e^{-5} of the flux stays in the 500 m column
constexpr Real RHO = 1.15;                 // air density [kg/m3]
constexpr Real FEEDBACK = 0.5;             // erf.fire.fire_atm_feedback
constexpr Real EMISSION = 0.02;            // erf.fire.smoke_emission_factor
constexpr Real BACKGROUND = 7.0;           // every other source of the cell
constexpr Real Cp_fire = 1004.64, L_V = 2.501e6;   // the coupling's c_p and L_v
/// Relative round-off tolerance of the build precision
constexpr double REL = (sizeof(Real) == 8) ? 1.0e-9 : 1.0e-4;

// nvcc refuses an extended __device__ lambda in a constructor or in a function
// it cannot take the address of, and in gtest's private TestBody: every kernel
// is in a namespace-scope function.

/// q = a + b x at the fire cell centres, x the physical position
void fill_linear_flux (MultiFab& q, const Geometry& g)
{
    const Real dxf = g.CellSize(0);
    const Real x0  = g.ProbLo(0);
    for (MFIter mfi(q); mfi.isValid(); ++mfi) {
        auto a = q.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            a(i, j, k) = Q0 + QX * (x0 + (i + 0.5_rt) * dxf);
        });
    }
}

/// Cell-centre heights of a flat mesh, ghosts included
void fill_zcc (MultiFab& zcc)
{
    for (MFIter mfi(zcc); mfi.isValid(); ++mfi) {
        auto z = zcc.array(mfi);
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { z(i, j, k) = (k + 0.5_rt) * DZ; });
    }
}

/// An engine that is not FireLayer: fixed flux fields, and a record of what it was handed.
class StandInEngine : public FireEngine
{
public:
    const char* name () const override { return "stand-in"; }
    void verify_prerequisites (const ERF&, int) override {}
    void initialize (const ERF&, int, const FireGrid&, const SurfaceLayer*, const MultiFab*) override {}

    /// Allocate and fill the flux fields on the coupler's fire grid (the test calls it).
    void set_up (const FireGrid& fg)
    {
        sensible = std::make_unique<MultiFab>(fg.ba, fg.dm, 1, 0);
        fill_linear_flux(*sensible, fg.geom);
        latent = std::make_unique<MultiFab>(fg.ba, fg.dm, 1, 0);
        latent->setVal(QLAT);
    }

    void advance (const FireForcing& forcing) override
    {
        ++n_advance;
        last_xvel   = &forcing.xvel;
        last_yvel   = &forcing.yvel;
        // the rain field lives only for the call: record what the test checks
        got_precip  = (forcing.precip_accum_k0 != nullptr);
        if (got_precip) {
            precip_ba    = forcing.precip_accum_k0->boxArray();
            precip_ncomp = forcing.precip_accum_k0->nComp();
        }
        last_time   = forcing.time;
        last_dt     = forcing.dt;
    }
    void fill_fluxes (FireFluxes& fluxes) override
    {
        ++n_fill;
        fluxes.sensible = sensible.get();
        fluxes.latent   = report_latent ? latent.get() : nullptr;
    }
    Real smoke_heat_per_kg () const override { return HEAT_PER_KG; }
    bool wants_surface_precip () const override { return wants_precip; }

    std::unique_ptr<MultiFab> sensible, latent;
    bool report_latent = true;
    bool wants_precip  = false;
    int n_advance = 0, n_fill = 0;
    const MultiFab* last_xvel = nullptr;
    const MultiFab* last_yvel = nullptr;
    bool got_precip = false;
    BoxArray precip_ba;
    int precip_ncomp = 0;
    Real last_time = -1.0, last_dt = -1.0;
};

/// The level the fire grid refines: the whole NX x NY x NZ domain in two
/// boxes of unequal width (periodic in x and y), or the region of columns 2-5
/// in two boxes, so the fire grid starts at column 2 (atm_lo = (2, 0)).
struct Level
{
    BoxArray ba, ba2d;
    DistributionMapping dm;
    Geometry geom;
    MultiFab S, xold, yold, xnew, ynew, zcc;
};

std::unique_ptr<Level> make_level (bool region)
{
    auto L = std::make_unique<Level>();
    const Box domain(IntVect(0, 0, 0), IntVect(NX - 1, NY - 1, NZ - 1));
    BoxList bl;
    if (region) {
        bl.push_back(Box(IntVect(2, 0, 0), IntVect(3, NY - 1, NZ - 1)));
        bl.push_back(Box(IntVect(4, 0, 0), IntVect(5, NY - 1, NZ - 1)));
    } else {
        bl.push_back(Box(IntVect(0, 0, 0), IntVect(3, NY - 1, NZ - 1)));
        bl.push_back(Box(IntVect(4, 0, 0), IntVect(NX - 1, NY - 1, NZ - 1)));
    }
    L->ba = BoxArray(bl);
    L->dm = DistributionMapping(L->ba);
    L->geom = Geometry(domain, RealBox(0.0, 0.0, 0.0, NX * DXY, NY * DXY, NZ * DZ), CoordSys::cartesian, {1, 1, 0});
    L->ba2d = L->ba;
    L->ba2d.coarsen(IntVect(1, 1, NZ));
    L->S.define(L->ba, L->dm, NVAR_max, 1);
    L->S.setVal(0.0_rt);
    L->S.setVal(RHO, Rho_comp, 1, 1);
    L->S.setVal(getRhoThetagivenP(9.0e4_rt), RhoTheta_comp, 1, 1);
    L->xold.define(convert(L->ba, IntVect(1, 0, 0)), L->dm, 1, 1); L->xold.setVal(1.0_rt);
    L->yold.define(convert(L->ba, IntVect(0, 1, 0)), L->dm, 1, 1); L->yold.setVal(1.0_rt);
    L->xnew.define(convert(L->ba, IntVect(1, 0, 0)), L->dm, 1, 1); L->xnew.setVal(2.0_rt);
    L->ynew.define(convert(L->ba, IntVect(0, 1, 0)), L->dm, 1, 1); L->ynew.setVal(2.0_rt);
    L->zcc.define(L->ba, L->dm, 1, 1);
    fill_zcc(L->zcc);
    return L;
}

FireCouplingParams coupling (const std::string& type, bool smoke)
{
    FireCouplingParams p;
    p.grid_ratio            = C;
    p.coupling_type         = type;
    p.heat_flux_alfg        = ALFG;
    p.fire_atm_feedback     = FEEDBACK;
    p.inject_latent         = true;
    p.source_mode           = "add";
    p.heat_tendency_density = true;
    p.heat_tendency_exner   = false;
    p.smoke_enable          = smoke;
    p.smoke_emission_factor = EMISSION;
    return p;
}

/// The coupler with the stand-in engine, laid out on the level's fire grid.
std::unique_ptr<FireCoupler> make_coupler (const Level& lev, const FireCouplingParams& p, StandInEngine*& engine)
{
    auto eng = std::make_unique<StandInEngine>();
    engine = eng.get();
    auto cpl = std::make_unique<FireCoupler>(p, std::move(eng));
    FireGrid fg = create_fire_grid(lev.ba, lev.dm, lev.geom, C);
    cpl->set_fire_grid(0, fg, lev.ba, lev.dm, NZ);
    engine->set_up(cpl->fire_grid());
    return cpl;
}

void step (FireCoupler& cpl, Level& lev)
{
    cpl.advance(10.0_rt, 0.5_rt, lev.S, lev.xold, lev.yold, lev.xnew, lev.ynew, lev.zcc,
                lev.ba2d, lev.geom, SurfacePrecipAccumulationSources{});
}

/// The cell sources after add_sources on a background of BACKGROUND everywhere
std::unique_ptr<MultiFab> sources (FireCoupler& cpl, Level& lev)
{
    auto src = std::make_unique<MultiFab>(lev.ba, lev.dm, NVAR_max, 0);
    src->setVal(BACKGROUND);
    cpl.add_sources(*src, lev.S, lev.zcc, nullptr, lev.geom, true, 1);
    return src;
}

/// Column sums sum_k (src - b) dz of one component, one value per (i, j), checked by f(i, sum)
template <typename F>
void for_each_column (const MultiFab& src, int comp, Real b, F&& f)
{
    for (MFIter mfi(src); mfi.isValid(); ++mfi) {
        const auto s = src.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                Real sum = 0.0;
                for (int k = 0; k < NZ; ++k) { sum += (s(i, j, k, comp) - b) * DZ; }
                f(i, j, sum, s);
            }
        }
    }
}

Real flux_at_column (int i) { return Q0 + QX * (i + 0.5_rt) * DXY; }

void FluxesAverageOntoTheColumnsTheyCameFrom ()
{
    for (bool region : {false, true}) {
        auto lev = make_level(region);
        StandInEngine* eng = nullptr;
        auto cpl = make_coupler(*lev, coupling("lagged", false), eng);
        EXPECT_EQ(cpl->fire_grid().atm_lo[0], region ? 2 : 0);
        step(*cpl, *lev);

        EXPECT_EQ(eng->n_advance, 1);
        EXPECT_EQ(eng->n_fill, 1);
        EXPECT_EQ(eng->last_xvel, &lev->xold) << "lagged coupling hands over the wind before the dycore";
        EXPECT_EQ(eng->last_yvel, &lev->yold);
        EXPECT_FALSE(eng->got_precip) << "an engine that did not ask gets no rain";
        EXPECT_EQ(eng->last_time, 10.0_rt);
        EXPECT_EQ(eng->last_dt, 0.5_rt);

        // the mean of a + b x over a column's C x C fire cells is a + b x_centre;
        // on the whole periodic domain the 5 ghost columns carry the periodic
        // neighbour's value
        const MultiFab& Q  = *cpl->get_Q_atm_prev();
        const MultiFab& QL = *cpl->get_Q_lat_atm_prev();
        const IntVect ng = region ? IntVect(0, 0, 0) : IntVect(5, 5, 0);
        for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
            const auto q  = Q.const_array(mfi);
            const auto ql = QL.const_array(mfi);
            const Box gbx = grow(mfi.validbox(), ng);
            for (int j = gbx.smallEnd(1); j <= gbx.bigEnd(1); ++j) {
                for (int i = gbx.smallEnd(0); i <= gbx.bigEnd(0); ++i) {
                    const int ip = ((i % NX) + NX) % NX;
                    EXPECT_NEAR(q(i, j, 0), flux_at_column(ip), REL * flux_at_column(ip))
                        << (region ? "region" : "domain") << ", column (" << i << ", " << j << ")";
                    EXPECT_NEAR(ql(i, j, 0), QLAT, REL * QLAT);
                }
            }
        }
    }
}

void SynchronousCouplingHandsOverTheNewWindAndDropsAVanishedLatentFlux ()
{
    auto lev = make_level(false);
    StandInEngine* eng = nullptr;
    auto cpl = make_coupler(*lev, coupling("synchronous", false), eng);
    step(*cpl, *lev);
    EXPECT_EQ(eng->last_xvel, &lev->xnew) << "synchronous coupling hands over the wind after the dycore";
    EXPECT_EQ(eng->last_yvel, &lev->ynew);
    EXPECT_NEAR(cpl->get_Q_lat_atm_prev()->max(0), QLAT, REL * QLAT);
    // the next step reports no latent flux: last step's must not linger
    eng->report_latent = false;
    step(*cpl, *lev);
    EXPECT_EQ(cpl->get_Q_lat_atm_prev()->norm0(0, 1, IntVect(5, 5, 0)), 0.0_rt);
    EXPECT_GT(cpl->get_Q_atm_prev()->min(0), 0.0_rt) << "the sensible flux still arrives";
}

void TheInjectedHeatIsTheSurfaceFlux ()
{
    auto lev = make_level(false);
    StandInEngine* eng = nullptr;
    auto cpl = make_coupler(*lev, coupling("lagged", true), eng);
    step(*cpl, *lev);
    auto src = sources(*cpl, *lev);

    const Real frac = 1.0_rt - std::exp(-Real(NZ) * DZ / ALFG);
    // add mode: the background stays, the tendency comes on top
    for_each_column(*src, RhoTheta_comp, BACKGROUND, [&] (int i, int, Real sum, const auto& s) {
        const Real q = flux_at_column(i);
        EXPECT_NEAR(Cp_fire * sum, RHO * FEEDBACK * q * frac, REL * q) << "sensible, column " << i;
        // the smoke: e Q / h in the lowest cell (not scaled by the feedback), spread over its thickness
        const Real smoke = EMISSION * q / HEAT_PER_KG;
        EXPECT_NEAR((s(i, 0, 0, RhoSmoke_comp) - BACKGROUND) * DZ, smoke, REL * smoke) << "smoke, column " << i;
        EXPECT_EQ(s(i, 0, 1, RhoSmoke_comp), BACKGROUND);
        EXPECT_EQ(s(i, 0, 0, Rho_comp), BACKGROUND) << "no other source is touched";
    });
    for_each_column(*src, RhoQ1_comp, BACKGROUND, [&] (int i, int, Real sum, const auto&) {
        EXPECT_NEAR(L_V * sum, RHO * FEEDBACK * QLAT * frac, REL * QLAT) << "latent, column " << i;
    });

    // overwrite mode: the theta and vapour sources are replaced, the others kept
    FireCouplingParams p = coupling("lagged", false);
    p.source_mode = "overwrite";
    StandInEngine* eng2 = nullptr;
    auto cpl2 = make_coupler(*lev, p, eng2);
    step(*cpl2, *lev);
    auto src2 = sources(*cpl2, *lev);
    for_each_column(*src2, RhoTheta_comp, 0.0_rt, [&] (int i, int, Real sum, const auto& s) {
        const Real q = flux_at_column(i);
        EXPECT_NEAR(Cp_fire * sum, RHO * FEEDBACK * q * frac, REL * q) << "overwritten theta source, column " << i;
        EXPECT_EQ(s(i, 0, 0, Rho_comp), BACKGROUND);
        EXPECT_EQ(s(i, 0, 0, RhoSmoke_comp), BACKGROUND) << "no smoke without smoke_enable";
    });
}

void PassiveCouplingInjectsNoHeat ()
{
    // with the smoke tracer: the flux is still averaged, the smoke injected, the heat not
    {
        auto lev = make_level(false);
        StandInEngine* eng = nullptr;
        auto cpl = make_coupler(*lev, coupling("passive", true), eng);
        step(*cpl, *lev);
        EXPECT_EQ(eng->n_fill, 1);
        EXPECT_NEAR(cpl->get_Q_atm_prev()->max(0), flux_at_column(NX - 1), REL * flux_at_column(NX - 1));
        auto src = sources(*cpl, *lev);
        EXPECT_EQ(src->min(RhoTheta_comp), BACKGROUND);
        EXPECT_EQ(src->max(RhoTheta_comp), BACKGROUND);
        EXPECT_EQ(src->max(RhoQ1_comp), BACKGROUND);
        EXPECT_GT(src->max(RhoSmoke_comp), BACKGROUND);
    }
    // without it: nothing to ask the engine for, nothing touched
    {
        auto lev = make_level(false);
        StandInEngine* eng = nullptr;
        auto cpl = make_coupler(*lev, coupling("passive", false), eng);
        step(*cpl, *lev);
        EXPECT_EQ(eng->n_advance, 1);
        EXPECT_EQ(eng->n_fill, 0) << "no heat and no smoke reach the atmosphere: nothing to ask for";
        EXPECT_EQ(cpl->get_Q_atm_prev()->norm0(0, 1, IntVect(5, 5, 0)), 0.0_rt);
        auto src = sources(*cpl, *lev);
        for (int comp : {static_cast<int>(RhoTheta_comp), static_cast<int>(RhoQ1_comp), static_cast<int>(RhoSmoke_comp)}) {
            EXPECT_EQ(src->min(comp), BACKGROUND);
            EXPECT_EQ(src->max(comp), BACKGROUND);
        }
    }
}

void InjectLatentOffDropsTheEnginesLatentFlux ()
{
    auto lev = make_level(false);
    FireCouplingParams p = coupling("lagged", false);
    p.inject_latent = false;
    StandInEngine* eng = nullptr;
    auto cpl = make_coupler(*lev, p, eng);
    step(*cpl, *lev);
    EXPECT_EQ(cpl->get_Q_lat_atm_prev()->norm0(0, 1, IntVect(5, 5, 0)), 0.0_rt);
    auto src = sources(*cpl, *lev);
    EXPECT_EQ(src->min(RhoQ1_comp), BACKGROUND);
    EXPECT_EQ(src->max(RhoQ1_comp), BACKGROUND);
    EXPECT_GT(src->max(RhoTheta_comp), BACKGROUND) << "the sensible heat still arrives";
}

void TheRainIsHandedOverOnlyWhenAsked ()
{
    auto lev = make_level(true);
    StandInEngine* eng = nullptr;
    auto cpl = make_coupler(*lev, coupling("lagged", false), eng);
    eng->wants_precip = true;
    step(*cpl, *lev);
    ASSERT_TRUE(eng->got_precip);
    EXPECT_EQ(eng->precip_ba, lev->ba2d) << "one k = 0 slab per box of the level";
    EXPECT_EQ(eng->precip_ncomp, 1);
}

/// Keys added to erf.fire for one test and removed on exit
class ScopedKeys
{
public:
    ScopedKeys () = default;
    ~ScopedKeys () { for (const auto& n : m_names) { m_pp.remove(n.c_str()); } }
    ScopedKeys (const ScopedKeys&) = delete;
    ScopedKeys& operator= (const ScopedKeys&) = delete;
    void add (const char* key, const std::string& v) { m_pp.add(key, v); m_names.emplace_back(key); }
    void add (const char* key, double v)             { m_pp.add(key, v); m_names.emplace_back(key); }
    void add (const char* key, int v)                { m_pp.add(key, v); m_names.emplace_back(key); }
    void add (const char* key, bool v)               { m_pp.add(key, v); m_names.emplace_back(key); }
private:
    ParmParse                m_pp{"erf.fire"};
    std::vector<std::string> m_names;
};

void TheFireInputsReachTheCoupling ()
{
    // the two tendency factors are set opposite ways in two decks, so neither
    // can stand in for the other
    for (bool density : {true, false}) {
        ScopedKeys deck;
        deck.add("engine", std::string("native"));
        deck.add("grid_ratio", 4);
        deck.add("anchor_level", 0);
        deck.add("coupling_type", std::string("synchronous"));
        deck.add("heat_flux_alfg", 30.0);
        deck.add("fire_atm_feedback", 0.25);
        deck.add("inject_latent", false);
        deck.add("source_mode", std::string("add"));
        deck.add("heat_tendency_density", density);
        deck.add("heat_tendency_exner", !density);
        deck.add("smoke_enable", true);
        deck.add("smoke_emission_factor", 0.05);
        deck.add("fire_debug", true);
        const FireCouplingParams c = FireParams().coupling_params();
        EXPECT_EQ(c.engine, "native");
        EXPECT_EQ(c.grid_ratio, 4);
        EXPECT_EQ(c.anchor_level, 0);
        EXPECT_EQ(c.coupling_type, "synchronous");
        EXPECT_TRUE(c.is_synchronous());
        EXPECT_TRUE(c.injects_flux());
        EXPECT_NEAR(c.heat_flux_alfg, 30.0, 1.0e-12);
        EXPECT_NEAR(c.fire_atm_feedback, 0.25, 1.0e-12);
        EXPECT_FALSE(c.inject_latent);
        EXPECT_EQ(c.source_mode, "add");
        EXPECT_FALSE(c.heat_open_fraction);
        EXPECT_EQ(c.heat_tendency_density, density);
        EXPECT_EQ(c.heat_tendency_exner, !density);
        EXPECT_TRUE(c.smoke_enable);
        EXPECT_NEAR(c.smoke_emission_factor, 0.05, 1.0e-12);
        EXPECT_TRUE(c.fire_debug);
    }
}

} // namespace

TEST(FireCoupler, FluxesAverageOntoTheColumnsTheyCameFrom)        { FluxesAverageOntoTheColumnsTheyCameFrom(); }
TEST(FireCoupler, SynchronousCouplingHandsOverTheNewWindAndDropsAVanishedLatentFlux)
{
    SynchronousCouplingHandsOverTheNewWindAndDropsAVanishedLatentFlux();
}
TEST(FireCoupler, TheInjectedHeatIsTheSurfaceFlux)                 { TheInjectedHeatIsTheSurfaceFlux(); }
TEST(FireCoupler, PassiveCouplingInjectsNoHeat)                    { PassiveCouplingInjectsNoHeat(); }
TEST(FireCoupler, InjectLatentOffDropsTheEnginesLatentFlux)        { InjectLatentOffDropsTheEnginesLatentFlux(); }
TEST(FireCoupler, TheRainIsHandedOverOnlyWhenAsked)                { TheRainIsHandedOverOnlyWhenAsked(); }
TEST(FireCoupler, TheFireInputsReachTheCoupling)                   { TheFireInputsReachTheCoupling(); }
