#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

// Property checks on a plotfile, for regression tests that need no gold file. Modes:
//
//   sounding <plotfile> <theta_air> <tol>
//       Two levels, the fine one down to the bottom. The fine cells at the bottom must hold the air
//       theta of the sounding, and the coarse cells at the bottom the average of the fine cells
//       over them (the coarse state is averaged down from the fine one).
//   uniform <plotfile> <theta0> <tol>
//       theta must equal theta0 in every fluid cell (terrain_IB_mask < 1 when plotted) of every level.
//   mean <plotfile> <theta0> <tol>
//       The fluid-mass-weighted mean theta of level 0 (the coarse level covers the domain, its
//       covered cells are averaged down) must be within tol of theta0: no net heat source.
//   fluidavg <plotfile> <tol> <unused>
//       Immersed forcing: every coarse cell (not solid on its own level) covered by level 1 must hold
//       the fluid-mass-weighted average of the fine cells over it,
//       sum (1 - beta_f) rho_f theta_f / sum (1 - beta_f) rho_f (the average down of the state).
//   bounded <plotfile> <lo> <hi>
//       theta must be finite and within [lo, hi] in every cell of every level.

namespace {

using namespace amrex;

int fail (const std::string& message)
{
    std::cerr << "MultiLevelPropertyCheck: " << message << "\n";
    return 1;
}

bool has_variable (const PlotFileData& pf, const std::string& name)
{
    const auto& names = pf.varNames();
    return std::find(names.begin(), names.end(), name) != names.end();
}

// largest |a - b| over the cells of layer k = klo (all k when klo < 0), and over the cells where
// mask < 1 (all cells when mask is null)
Real max_abs_diff (const MultiFab& a, const MultiFab& b, int klo, const MultiFab* mask)
{
    ReduceOps<ReduceOpMax> op;
    ReduceData<Real> data(op);
    using Tuple = typename decltype(data)::Type;
    for (MFIter mfi(a); mfi.isValid(); ++mfi) {
        auto const aa = a.const_array(mfi);
        auto const bb = b.const_array(mfi);
        auto const mm = (mask) ? mask->const_array(mfi) : Array4<Real const>{};
        const bool use_mask = (mask != nullptr);
        op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple
        {
            if (klo >= 0 && k != klo) { return {Real(0.)}; }
            if (use_mask && mm(i,j,k) >= Real(1.)) { return {Real(0.)}; }
            return {std::abs(aa(i,j,k) - bb(i,j,k))};
        });
    }
    Real r = get<0>(data.value(op));
    ParallelDescriptor::ReduceRealMax(r);
    return r;
}

int check_sounding (PlotFileData& pf, Real theta_air, Real tol)
{
    if (pf.finestLevel() < 1) { return fail("sounding: needs two levels"); }
    const MultiFab crse = pf.get(0, "theta");
    const MultiFab fine = pf.get(1, "theta");
    const int kbot = pf.probDomain(1).smallEnd(2);
    if (fine.boxArray().minimalBox().smallEnd(2) != kbot) { return fail("sounding: fine level must reach the bottom"); }

    MultiFab air(fine.boxArray(), fine.DistributionMap(), 1, 0);
    air.setVal(theta_air);
    const Real err_fine = max_abs_diff(fine, air, kbot, nullptr);

    const IntVect rr(pf.refRatio(0));
    MultiFab avg(amrex::coarsen(fine.boxArray(), rr), fine.DistributionMap(), 1, 0);
    amrex::average_down(fine, avg, 0, 1, rr);
    MultiFab crse_on_fine(avg.boxArray(), avg.DistributionMap(), 1, 0);
    crse_on_fine.ParallelCopy(crse, 0, 0, 1);
    const Real err_avg = max_abs_diff(crse_on_fine, avg, pf.probDomain(0).smallEnd(2), nullptr);

    Print() << "sounding: max |theta - theta_air| in the fine bottom cells = " << err_fine
            << ", max |coarse - fine average| in the coarse bottom cells = " << err_avg
            << " (tol " << tol << ")\n";
    if (err_fine > tol) { return fail("sounding: the fine bottom cells do not hold the air theta"); }
    if (err_avg > tol)  { return fail("sounding: the coarse bottom cells are not the fine average"); }
    return 0;
}

int check_uniform (PlotFileData& pf, Real theta0, Real tol)
{
    const bool masked = has_variable(pf, "terrain_IB_mask");
    Real worst = Real(0.);
    for (int lev = 0; lev <= pf.finestLevel(); ++lev) {
        const MultiFab th = pf.get(lev, "theta");
        if (!th.is_finite()) { return fail("uniform: non-finite theta on level " + std::to_string(lev)); }
        MultiFab ref(th.boxArray(), th.DistributionMap(), 1, 0);
        ref.setVal(theta0);
        MultiFab mask;
        if (masked) { mask = pf.get(lev, "terrain_IB_mask"); }
        const Real e = max_abs_diff(th, ref, -1, masked ? &mask : nullptr);
        Print() << "uniform: level " << lev << " max |theta - theta0| over fluid cells = " << e << "\n";
        worst = std::max(worst, e);
    }
    if (worst > tol) { return fail("uniform: theta is not uniform (tol " + std::to_string(tol) + ")"); }
    return 0;
}

int check_mean (PlotFileData& pf, Real theta0, Real tol)
{
    if (!has_variable(pf, "density")) { return fail("mean: plotfile has no density"); }
    const bool masked = has_variable(pf, "terrain_IB_mask");
    const MultiFab th  = pf.get(0, "theta");
    const MultiFab rho = pf.get(0, "density");
    MultiFab mask;
    if (masked) { mask = pf.get(0, "terrain_IB_mask"); }
    if (!th.is_finite()) { return fail("mean: non-finite theta"); }
    ReduceOps<ReduceOpSum, ReduceOpSum> op;
    ReduceData<Real, Real> data(op);
    using Tuple = typename decltype(data)::Type;
    for (MFIter mfi(th); mfi.isValid(); ++mfi) {
        auto const t = th.const_array(mfi);
        auto const r = rho.const_array(mfi);
        auto const m = (masked) ? mask.const_array(mfi) : Array4<Real const>{};
        op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple
        {
            const Real fluid = (masked) ? amrex::max(Real(0.), Real(1.) - m(i,j,k)) : Real(1.);
            const Real w = fluid * r(i,j,k);
            return {w * (t(i,j,k) - theta0), w};
        });
    }
    auto sums = data.value(op);
    Real num = get<0>(sums), den = get<1>(sums);
    ParallelDescriptor::ReduceRealSum(num);
    ParallelDescriptor::ReduceRealSum(den);
    const Real drift = num / den;
    Print() << "mean: fluid-mass-weighted mean theta - theta0 = " << drift << " (tol " << tol << ")\n";
    if (std::abs(drift) > tol) { return fail("mean: the mean theta drifted (a net heat source)"); }
    return 0;
}

int check_fluidavg (PlotFileData& pf, Real tol)
{
    if (pf.finestLevel() < 1) { return fail("fluidavg: needs two levels"); }
    if (!has_variable(pf, "density") || !has_variable(pf, "terrain_IB_mask")) {
        return fail("fluidavg: needs density and terrain_IB_mask");
    }
    const MultiFab th_c = pf.get(0, "theta");
    const MultiFab b_c  = pf.get(0, "terrain_IB_mask");
    const MultiFab th_f = pf.get(1, "theta");
    const MultiFab r_f  = pf.get(1, "density");
    const MultiFab b_f  = pf.get(1, "terrain_IB_mask");
    const IntVect rr(pf.refRatio(0));
    const BoxArray cba = amrex::coarsen(th_f.boxArray(), rr);
    // coarse values on the coarsened fine layout
    MultiFab tc(cba, th_f.DistributionMap(), 1, 0), bc(cba, th_f.DistributionMap(), 1, 0);
    tc.ParallelCopy(th_c, 0, 0, 1);
    bc.ParallelCopy(b_c, 0, 0, 1);
    ReduceOps<ReduceOpMax, ReduceOpSum> op;
    ReduceData<Real, int> data(op);
    using Tuple = typename decltype(data)::Type;
    for (MFIter mfi(tc); mfi.isValid(); ++mfi) {
        auto const t  = tc.const_array(mfi);
        auto const bb = bc.const_array(mfi);
        auto const tf = th_f.const_array(mfi);
        auto const rf = r_f.const_array(mfi);
        auto const bf = b_f.const_array(mfi);
        const int rx = rr[0], ry = rr[1], rz = rr[2];
        op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Tuple
        {
            if (bb(i,j,k) >= Real(1.)) { return {Real(0.), 0}; }
            Real num = Real(0.), den = Real(0.);
            int nsolid = 0;
            for (int kk = k*rz; kk < (k+1)*rz; ++kk) {
            for (int jj = j*ry; jj < (j+1)*ry; ++jj) {
            for (int ii = i*rx; ii < (i+1)*rx; ++ii) {
                const Real w = amrex::max(Real(0.), Real(1.) - bf(ii,jj,kk)) * rf(ii,jj,kk);
                num += w * tf(ii,jj,kk);
                den += w;
                if (bf(ii,jj,kk) >= Real(1.)) { ++nsolid; }
            }}}
            if (den <= Real(0.)) { return {Real(0.), 0}; }
            return {std::abs(t(i,j,k) - num / den), (nsolid > 0) ? 1 : 0};
        });
    }
    auto v = data.value(op);
    Real err = get<0>(v);
    int nmixed = get<1>(v);
    ParallelDescriptor::ReduceRealMax(err);
    ParallelDescriptor::ReduceIntSum(nmixed);
    Print() << "fluidavg: max |coarse theta - fluid-weighted fine average| = " << err
            << " over covered coarse cells (" << nmixed << " of them over solid fine cells; tol " << tol << ")\n";
    if (nmixed == 0) { return fail("fluidavg: no covered coarse cell lies over a solid fine cell"); }
    if (err > tol) { return fail("fluidavg: coarse cells are not the fluid-weighted fine average"); }
    return 0;
}

int check_bounded (PlotFileData& pf, Real lo, Real hi)
{
    for (int lev = 0; lev <= pf.finestLevel(); ++lev) {
        const MultiFab th = pf.get(lev, "theta");
        if (!th.is_finite()) { return fail("bounded: non-finite theta on level " + std::to_string(lev)); }
        const Real tmin = th.min(0), tmax = th.max(0);
        Print() << "bounded: level " << lev << " theta in [" << tmin << ", " << tmax << "]\n";
        if (tmin < lo || tmax > hi) { return fail("bounded: theta leaves [" + std::to_string(lo) + ", " + std::to_string(hi) + "]"); }
    }
    return 0;
}

} // namespace

int main (int argc, char** argv)
{
    if (argc != 5) {
        std::cerr << "usage: checker sounding|uniform|mean|fluidavg|bounded plotfile a b\n";
        return 2;
    }
    amrex::Initialize(argc, argv, false);
    int result = 2;
    {
        const std::string mode(argv[1]);
        PlotFileData pf(argv[2]);
        const Real a = static_cast<Real>(std::atof(argv[3]));
        const Real b = static_cast<Real>(std::atof(argv[4]));
        if (!has_variable(pf, "theta")) {
            result = fail("plotfile has no theta");
        } else if (mode == "sounding") {
            result = check_sounding(pf, a, b);
        } else if (mode == "uniform") {
            result = check_uniform(pf, a, b);
        } else if (mode == "mean") {
            result = check_mean(pf, a, b);
        } else if (mode == "fluidavg") {
            result = check_fluidavg(pf, a);
            amrex::ignore_unused(b);
        } else if (mode == "bounded") {
            result = check_bounded(pf, a, b);
        } else {
            result = fail("unknown mode " + mode);
        }
    }
    amrex::Finalize();
    return result;
}
