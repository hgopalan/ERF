#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "ERF_DustPM.H"
#include "ERF_DustSTEL.H"
#include "ERF_DustStatsOutput.H"
#include "ERF_DustSuppression.H"

/**
 * @file ERF_GTestDustDiagnostics.cpp
 * @brief The dust diagnostics, written to fail on the code before the October
 *        2026 validation:
 *  - the 24-hour and STEL averages are window means (the exponential running
 *    mean gave 0.632 C after one window of a constant C, so a 50 ug/m3 day never
 *    flagged the 35 ug/m3 standard);
 *  - the lumped scalar is apportioned to the PM classes by the bins' shares
 *    (the bin-0 diameter classified the whole mass: PM2.5 was identically zero);
 *  - the statistics CSV integrates the flux over every bin and the cell area;
 *  - the re-treatment flag stays on a treated cell whose coverage wore to zero.
 */

using namespace amrex;

constexpr double REAL_RTOL = std::is_same<amrex::Real, float>::value ? 1.0e-5 : 1.0e-10;

namespace {

struct Slab {
    Box domain{IntVect(0, 0, 0), IntVect(1, 1, 0)};   // 2 x 2 cells of 10 m
    Geometry geom{domain, RealBox({0.0, 0.0, 0.0}, {20.0, 20.0, 1.0}), 0, {0, 0, 0}};
    BoxArray ba{domain};
    DistributionMapping dm{ba};
};

Real first_value (const MultiFab& mf, int comp = 0)
{
    Real v = 0.0;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box b(IntVect(0, 0, 0), IntVect(0, 0, 0));
        if (!mfi.validbox().contains(b.smallEnd())) { continue; }
        FArrayBox h(b, 1, The_Pinned_Arena());
        h.copy<RunOn::Device>(mf[mfi], b, comp, b, 0, 1);
        Gpu::streamSynchronize();
        v = h(b.smallEnd());
    }
    ParallelDescriptor::ReduceRealSum(v);
    return v;
}

} // namespace

TEST(DustDiagnostics, WindowMeanOfAConstantIsTheConstant)
{
    Slab s;
    DustWindowMean w;
    w.define(s.ba, s.dm, 24, 86400.0);
    MultiFab C(s.ba, s.dm, 1, 0), mean(s.ba, s.dm, 1, 0);
    C.setVal(50.0);
    const Real dt = 600.0;
    for (int n = 0; n < 144; ++n) { w.update(C, dt, mean); }   // 24 h
    EXPECT_NEAR(first_value(mean), 50.0, REAL_RTOL * 50.0)
        << "the exponential running mean reported 0.632 C = 31.6 after 24 h";
    EXPECT_GT(first_value(mean), Real(DustPMConst::PM25_24H_NAAQS)) << "a 50 ug/m3 day flags the 35 ug/m3 standard";
    // then 24 h of clean air: the window forgets the episode entirely
    C.setVal(0.0);
    for (int n = 0; n < 144; ++n) { w.update(C, dt, mean); }
    EXPECT_NEAR(first_value(mean), 0.0, 1.0e-12) << "an exponential mean keeps e^-1 of the episode";
}

TEST(DustDiagnostics, WindowMeanIsExactMidWindow)
{
    Slab s;
    DustWindowMean w;
    w.define(s.ba, s.dm, 15, 900.0);   // the STEL ring: 15 one-minute slots
    MultiFab C(s.ba, s.dm, 1, 0), mean(s.ba, s.dm, 1, 0);
    // 5 minutes at 10, then 5 minutes at 0: the 10-minute mean is 5
    C.setVal(10.0);
    for (int n = 0; n < 60; ++n) { w.update(C, Real(5.0), mean); }
    EXPECT_NEAR(first_value(mean), 10.0, REAL_RTOL * 10.0);
    C.setVal(0.0);
    for (int n = 0; n < 60; ++n) { w.update(C, Real(5.0), mean); }
    EXPECT_NEAR(first_value(mean), 5.0, REAL_RTOL * 5.0);
    // a step longer than a slot closes that slot with the step's value
    C.setVal(10.0);
    w.update(C, Real(120.0), mean);
    EXPECT_NEAR(first_value(mean), (10.0 * 300.0 + 0.0 * 300.0 + 10.0 * 120.0) / 720.0, REAL_RTOL * 10.0);
}

TEST(DustDiagnostics, ExponentialMeanWeightIsClamped)
{
    Slab s;
    MultiFab C(s.ba, s.dm, 1, 0), avg(s.ba, s.dm, 1, 0);
    C.setVal(10.0); avg.setVal(0.0);
    update_running_average(avg, C, Real(2000.0), Real(900.0));   // dt above the window
    EXPECT_NEAR(first_value(avg), 10.0, REAL_RTOL * 10.0) << "the unclamped weight gave -1.2 * 0 + 2.2 * 10 = 22";
    update_stel_average(avg, C, Real(2000.0), Real(900.0), nullptr);   // mg/m3 path, same clamp
    EXPECT_NEAR(first_value(avg), 0.01, REAL_RTOL * 0.01);
}

TEST(DustDiagnostics, LumpedScalarIsApportionedByTheBinShares)
{
    Slab s;
    MultiFab conc(s.ba, s.dm, 1, 0), pm25(s.ba, s.dm, 1, 0), pm10(s.ba, s.dm, 1, 0);
    conc.setVal(3.0e-9);   // 3 ug/m3 of lumped dust
    const std::vector<Real> bins = {7.0e-6, 2.5e-6, 50.0e-6};
    compute_pm_concentrations(pm25, pm10, conc, bins, 1, /*lumped=*/true);
    EXPECT_NEAR(first_value(pm25), 1.0, REAL_RTOL) << "one bin of three is PM2.5 (the bin-0 rule gave 0)";
    EXPECT_NEAR(first_value(pm10), 2.0, REAL_RTOL) << "two bins of three are PM10 (the bin-0 rule gave 3)";
    // per-bin scalars classify each bin whole
    MultiFab conc3(s.ba, s.dm, 3, 0);
    conc3.setVal(1.0e-9);
    compute_pm_concentrations(pm25, pm10, conc3, bins, 3, false);
    EXPECT_NEAR(first_value(pm25), 1.0, REAL_RTOL);
    EXPECT_NEAR(first_value(pm10), 2.0, REAL_RTOL);
}

TEST(DustDiagnostics, StatsRowIntegratesEveryBinOverTheCellArea)
{
    Slab s;
    MultiFab flux(s.ba, s.dm, 3, 0), dep(s.ba, s.dm, 1, 0), us(s.ba, s.dm, 1, 0), conc(s.ba, s.dm, 1, 0);
    flux.setVal(1.0); dep.setVal(2.0); us.setVal(0.3); conc.setVal(1.0e-6);
    const std::string f = "gtest_dust_diag.dat";
    if (ParallelDescriptor::IOProcessor()) std::remove(f.c_str());
    ParallelDescriptor::Barrier();
    append_dust_stats(7, Real(3.5), f, &flux, &dep, &us, &conc, Real(100.0));
    ParallelDescriptor::Barrier();
    if (ParallelDescriptor::IOProcessor()) {
        std::ifstream in(f);
        std::string line, last;
        while (std::getline(in, line)) { if (!line.empty() && line[0] != '#' && line[0] != 's') last = line; }
        std::vector<Real> v;
        std::size_t p = 0;
        while (p <= last.size()) {
            std::size_t q = last.find(',', p);
            if (q == std::string::npos) q = last.size();
            v.push_back(std::stod(last.substr(p, q - p)));
            p = q + 1;
        }
        ASSERT_EQ(v.size(), 7u);
        EXPECT_NEAR(v[2], 3 * 4 * 1.0 * 100.0, REAL_RTOL * 1200.0) << "3 bins x 4 cells x 1 kg/m2/s x 100 m2 = 1200 kg/s (the cell sum of bin 0 gave 4)";
        EXPECT_NEAR(v[3], 4 * 2.0 * 100.0, REAL_RTOL * 800.0) << "4 cells x 2 kg/m2 x 100 m2 = 800 kg";
        std::remove(f.c_str());
    }
}

TEST(DustDiagnostics, RetreatFlagStaysOnATreatedCellWornToZero)
{
    Slab s;
    MultiFab supp(s.ba, s.dm, 1, 0), flag(s.ba, s.dm, 1, 0), treated(s.ba, s.dm, 1, 0);
    supp.setVal(0.3); treated.setVal(1.0); flag.setVal(0.0);
    // twenty e-folds of decay at the reference temperature and no wind
    for (int n = 0; n < 20; ++n) {
        advance_dust_suppression(supp, flag, treated, Real(293.15), Real(0.0), Real(3600.0), Real(3600.0));
    }
    EXPECT_NEAR(first_value(supp), 0.0, 1.0e-12);
    EXPECT_NEAR(first_value(flag), 1.0, 1.0e-12) << "a cell worn to zero needs treatment most (the flag used to drop to 0 there)";
    // a never-treated cell is not flagged
    treated.setVal(0.0); supp.setVal(0.0);
    advance_dust_suppression(supp, flag, treated, Real(293.15), Real(0.0), Real(3600.0), Real(3600.0));
    EXPECT_NEAR(first_value(flag), 0.0, 1.0e-12);
}
