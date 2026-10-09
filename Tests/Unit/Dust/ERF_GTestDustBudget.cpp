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

#include "ERF_DustGrid.H"
#include "ERF_DustSettling.H"
#include "ERF_DustDeposition.H"
#include "ERF_DustAtmCoupling.H"
#include "ERF_DustRoadSchedule.H"
#include "ERF_DustBlastSchedule.H"

/**
 * @file ERF_GTestDustBudget.cpp
 * @brief The dust mass budget kernels, written to fail on the code before the
 *        October 2026 validation:
 *  - settling and injection divide by the cell thickness detJ dz, so a
 *    stretched column conserves mass (the centre-to-centre spacing created it);
 *  - the deposition velocity is never below the settling velocity (a 0.1 m/s cap
 *    sat below the 1 m/s settling cap, so coarse dust piled up in the lowest cell);
 *  - the lumped scalar settles at the mean of the bins' velocities;
 *  - a haul road emits the AP-42 mass rate whatever cells its box covers;
 *  - a blast delivers its whole mass whatever the time step.
 */

using namespace amrex;

constexpr double REAL_RTOL = std::is_same<amrex::Real, float>::value ? 1.0e-5 : 1.0e-10;

namespace {

constexpr int NZ = 8;
constexpr int DUST = 2;
constexpr int NCOMP = DUST + 1;
constexpr Real DZ0 = 10.0;          // first cell thickness
constexpr Real STRETCH = 1.1;       // geometric stretching ratio

/// Column with cells of thickness h_k = DZ0 STRETCH^k, uniform in index space
/// (dz = mean thickness), detJ(k) = h_k / dz.
struct StretchedColumn {
    Box domain{IntVect(0, 0, 0), IntVect(0, 0, NZ - 1)};
    Real ztop = 0.0;
    Geometry geom;
    BoxArray ba{domain};
    DistributionMapping dm{ba};
    MultiFab S{ba, dm, NCOMP, 0};
    MultiFab src{ba, dm, NCOMP, 0};
    MultiFab detJ{ba, dm, 1, 1};
    Real h[NZ];

    StretchedColumn ()
    {
        for (int k = 0; k < NZ; ++k) { h[k] = DZ0 * std::pow(STRETCH, k); ztop += h[k]; }
        geom = Geometry(domain, RealBox({0.0, 0.0, 0.0}, {10.0, 10.0, ztop}), 0, {0, 0, 0});
        const Real dz = ztop / NZ;
        S.setVal(0.0);
        S.setVal(1.225, Rho_comp, 1);
        src.setVal(0.0);
        const Real v_lo = h[0] / dz, v_hi = h[NZ - 1] / dz;
        for (MFIter mfi(detJ); mfi.isValid(); ++mfi) {
            auto dj = detJ.array(mfi);
            for (int k = 0; k < NZ; ++k) {
                const Real v = h[k] / dz;
                ParallelFor(mfi.growntilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int kk) noexcept {
                    if (kk == k) dj(i, j, kk) = v;
                    if (kk < 0) dj(i, j, kk) = v_lo;
                    if (kk >= NZ) dj(i, j, kk) = v_hi;
                });
            }
        }
    }
    void put_dust (int k_dust, Real value)
    {
        for (MFIter mfi(S); mfi.isValid(); ++mfi) {
            auto sa = S.array(mfi);
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (k == k_dust) { sa(i, j, k, DUST) = value; }
            });
        }
    }
};

Real value_at (const MultiFab& mf, int k, int comp)
{
    Real v = 0.0;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box b(IntVect(0, 0, k), IntVect(0, 0, k));
        if (!mfi.validbox().contains(b.smallEnd())) { continue; }
        FArrayBox hh(b, 1, The_Pinned_Arena());
        hh.copy<RunOn::Device>(mf[mfi], b, comp, b, 0, 1);
        Gpu::streamSynchronize();
        v = hh(b.smallEnd());
    }
    ParallelDescriptor::ReduceRealSum(v);
    return v;
}

DustBinDiameters bins_of (std::initializer_list<Real> ds)
{
    DustBinDiameters b{};
    for (int i = 0; i < DustSettlingConst::MAX_BINS; ++i) b[i] = 0.0;
    int i = 0;
    for (Real d : ds) b[i++] = d;
    return b;
}

/// Weights of a scalar that carries one bin.
inline DustBinWeights one_weight ()
{
    DustBinWeights w{};
    for (int i = 0; i < DustSettlingConst::MAX_BINS; ++i) w[i] = 0.0;
    w[0] = 1.0;
    return w;
}

} // namespace

TEST(DustBudget, SettlingConservesMassOnAStretchedColumn)
{
    StretchedColumn c;
    c.put_dust(5, 1.0);
    const Real d = 7.0e-6, rho_p = 2650.0;
    apply_dust_settling_to_cc_source(c.src, c.S, c.detJ, c.geom, bins_of({d}), one_weight(), 1, rho_p, DUST);
    const Real vs = compute_stokes_settling(d, rho_p, Real(1.225), DustSettlingConst::MU_AIR_STD);
    ASSERT_GT(vs, Real(0.0));
    // the dusty cell loses v_s rho / h_5, the cell below gains v_s rho / h_4
    EXPECT_NEAR(value_at(c.src, 5, DUST), -vs / c.h[5], REAL_RTOL * vs / c.h[5]);
    EXPECT_NEAR(value_at(c.src, 4, DUST),  vs / c.h[4], REAL_RTOL * vs / c.h[4]);
    Real column = 0.0;
    for (int k = 0; k < NZ; ++k) { column += value_at(c.src, k, DUST) * c.h[k]; }
    EXPECT_NEAR(column, 0.0, 1e-12 + REAL_RTOL * vs)
        << "the centre-to-centre spacing created 5 % of the transferred mass per step at this ratio";
}

TEST(DustBudget, InjectionPutsTheWholeFluxIntoTheLowestCell)
{
    StretchedColumn c;
    const Box slab(IntVect(0, 0, 0), IntVect(0, 0, 0));
    BoxArray ba2d(slab);
    DistributionMapping dm2d(ba2d);
    MultiFab flux(ba2d, dm2d, 1, 0);
    const Real F = 2.5e-6;
    flux.setVal(F);
    apply_dust_tendency_to_cc_source(c.src, flux, c.detJ, c.geom, DUST, Real(1.0), false);
    // d(rho s)/dt h_0 = F in the lowest cell, nothing elsewhere
    EXPECT_NEAR(value_at(c.src, 0, DUST) * c.h[0], F, REAL_RTOL * F)
        << "(h_0 + h_1)/2 instead of h_0 gave 95 % of the flux";
    EXPECT_NEAR(value_at(c.src, 1, DUST), 0.0, 1e-30);
}

TEST(DustBudget, DepositionVelocityIsNeverBelowTheSettlingVelocity)
{
    // 50 um quartz settles at ~0.2 m/s, above the former 0.1 m/s deposition cap
    const Real vs = compute_stokes_settling(Real(50.0e-6), Real(2650.0), Real(1.225), DustSettlingConst::MU_AIR_STD);
    ASSERT_GT(vs, Real(0.15));
    const Real vd = compute_deposition_velocity(vs, Real(0.5), Real(3.0e-3));
    EXPECT_GE(vd, vs) << "the surface takes at least what settles onto it (the cap returned 0.1 < 0.2)";
}

TEST(DustBudget, LumpedScalarSettlesAtTheMeanOfTheBins)
{
    const Real rho_p = 2650.0, rho_a = 1.225, mu = DustSettlingConst::MU_AIR_STD;
    const DustBinDiameters b = bins_of({7.0e-6, 2.5e-6, 50.0e-6});
    DustBinWeights w{}; for (int i = 0; i < 3; ++i) w[i] = 1.0 / 3.0;
    const Real mean = compute_mean_settling(b, w, 3, rho_p, rho_a, mu);
    Real sum = 0.0;
    for (int i = 0; i < 3; ++i) sum += compute_stokes_settling(b[i], rho_p, rho_a, mu);
    EXPECT_NEAR(mean, sum / 3.0, REAL_RTOL * mean);
    // the coarse third dominates: 17x the bin-0 velocity
    EXPECT_GT(mean, 10.0 * compute_stokes_settling(b[0], rho_p, rho_a, mu))
        << "bin 0 alone settled the 50 um third at the 7 um velocity";
    EXPECT_NEAR(compute_mean_settling(b, w, 1, rho_p, rho_a, mu),
                compute_stokes_settling(b[0], rho_p, rho_a, mu), REAL_RTOL);
    // the shares weight the mean: haul-road dust (bin 0 only) settles at the
    // 7 um velocity, not the three-bin mean (17x faster, the form until October 2026)
    DustBinWeights road{}; road[0] = 1.0;
    EXPECT_NEAR(compute_mean_settling(b, road, 3, rho_p, rho_a, mu),
                compute_stokes_settling(b[0], rho_p, rho_a, mu), REAL_RTOL)
        << "equal shares gave " << mean << " for a scalar that holds bin 0 only";
    DustBinWeights half{}; half[0] = 0.5; half[2] = 0.5;
    EXPECT_NEAR(compute_mean_settling(b, half, 3, rho_p, rho_a, mu),
                0.5 * (compute_stokes_settling(b[0], rho_p, rho_a, mu) + compute_stokes_settling(b[2], rho_p, rho_a, mu)),
                REAL_RTOL * mean);
    DustBinWeights none{};   // no shares: equal
    EXPECT_NEAR(compute_mean_settling(b, none, 3, rho_p, rho_a, mu), mean, REAL_RTOL * mean);
}

TEST(DustBudget, HaulRoadEmitsTheAP42MassRateOverTheCellsItCovers)
{
    // 4 x 4 dust cells of 100 m; a 20 m wide, 280 m long road along x covers
    // the two cells whose centres lie in its box (x 60..340, y 140..160:
    // centres at 150 and 250; the box edges are inclusive)
    const Box slab(IntVect(0, 0, 0), IntVect(3, 3, 0));
    BoxArray ba(slab);
    DistributionMapping dm(ba);
    Geometry geom(slab, RealBox({0.0, 0.0, 0.0}, {400.0, 400.0, 1.0}), 0, {0, 0, 0});
    RoadSchedule sched;
    RoadEvent ev;
    ev.name = "haul"; ev.x_lo = 60.0; ev.x_hi = 340.0; ev.y_lo = 140.0; ev.y_hi = 160.0;
    ev.road_width_m = 20.0; ev.vehicle_weight_t = 40.0; ev.silt_pct = 8.0; ev.vkt_per_h = 12.0;
    ev.start_time_s = 0.0; ev.end_time_s = -1.0;
    sched.roads.push_back(ev);
    sched.loaded = true;
    count_road_cells(sched, ba, geom);
    ASSERT_EQ(sched.roads[0].n_cells, 2);
    MultiFab flux(ba, dm, 1, 0);
    flux.setVal(0.0);
    apply_road_schedule(flux, geom, sched, Real(10.0), Real(1.0), false, "", 1);
    const Real M = road_mass_rate_kg_s(ev);
    // E = 423 (8/12)^0.9 (40/3)^0.45 g/VKT; M = 1e-3 E 12 / 3600
    const Real E = 423.0 * std::pow(8.0 / 12.0, 0.9) * std::pow(40.0 / 3.0, 0.45);
    EXPECT_NEAR(M, 1.0e-3 * E * 12.0 / 3600.0, REAL_RTOL * M);
    const Real total = flux.sum(0) * 100.0 * 100.0;   // sum of flux x cell area
    EXPECT_NEAR(total, M, REAL_RTOL * M)
        << "the flux per unit road area stamped per cell gave n A / (W L) = 3.6x this";
}

TEST(DustBudget, BlastDeliversItsMassWhateverTheTimeStep)
{
    const Box slab(IntVect(0, 0, 0), IntVect(3, 3, 0));
    BoxArray ba(slab);
    DistributionMapping dm(ba);
    DustGrid dg;
    dg.ba = ba; dg.dm = dm; dg.grid_ratio = 1;
    dg.geom = Geometry(slab, RealBox({0.0, 0.0, 0.0}, {400.0, 400.0, 1.0}), 0, {0, 0, 0});
    const int nb = 3;
    const Real mass = 0.05, r = 2.0;
    for (Real dt : {Real(0.5), Real(5.0)}) {
        BlastSchedule sched;
        BlastEvent evt;
        evt.time_s = 1.0; evt.cx = 150.0; evt.cy = 150.0; evt.radius = 60.0; evt.mass_kg_m2 = mass;
        sched.events.push_back(evt);
        MultiFab flux(ba, dm, nb, 0);
        flux.setVal(0.0);
        apply_blast_schedule(flux, dg, sched, Real(2.0), Real(0.0), dt, nb, r);
        // the cell at the centre: sum over the bins of F dt = m r
        Real sum = 0.0;
        for (int b = 0; b < nb; ++b) sum += flux.max(b) * dt;
        EXPECT_NEAR(sum, mass * r, REAL_RTOL * mass * r)
            << "dt = " << dt << ": the 1e-2 kg/m2/s per-bin cap delivered 15 % at dt = 0.5";
    }
}
