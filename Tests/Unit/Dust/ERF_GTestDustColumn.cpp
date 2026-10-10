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

#include "ERF_DustSettling.H"
#include "ERF_DustDeposition.H"
#include "ERF_DustEmission.H"

/**
 * @file ERF_GTestDustColumn.cpp
 * @brief The dust column kernels on a single column with a known answer,
 *        written to fail on the code before the October 2026 audit:
 *  - Stokes settling moves dust DOWN, from the dust density of the state
 *    (the kernel read the tendency, so the dust never settled, and took the
 *    neighbour from below, so the flux pointed up);
 *  - dry deposition removes v_d * RhoDust(klo) and reports that flux (it
 *    multiplied v_d by the tendency);
 *  - the bins share the M&B vertical flux (each bin carried all of it, so
 *    the atmosphere received n_size_bins times the flux).
 */

using namespace amrex;

// Relative round-off allowance for amrex::Real: the double-precision checks
// stay at 1e-9, a single-precision build (float Real) gets 1e-6.
constexpr double REAL_RTOL = std::is_same<amrex::Real, float>::value ? 1.0e-6 : 1.0e-9;

namespace {

constexpr int NZ = 8;
constexpr Real DZ = 10.0;
constexpr int DUST = 2;              // a component after Rho and RhoTheta
constexpr int NCOMP = DUST + 1;

/// Cell-centre heights and a unit dust density in cell k_dust. A free
/// function: nvcc rejects an extended device lambda in a constructor.
void init_column (MultiFab& S, MultiFab& z, int k_dust)
{
    for (MFIter mfi(z); mfi.isValid(); ++mfi) {
        auto za = z.array(mfi);
        ParallelFor(mfi.growntilebox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            za(i, j, k) = (k + Real(0.5)) * DZ;
        });
    }
    for (MFIter mfi(S); mfi.isValid(); ++mfi) {
        auto sa = S.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (k == k_dust && i == 0 && j == 0) { sa(i, j, k, DUST) = Real(1.0); }
        });
    }
}

struct Column {
    Box domain{IntVect(0, 0, 0), IntVect(0, 0, NZ - 1)};
    Geometry geom{domain, RealBox({0.0, 0.0, 0.0}, {10.0, 10.0, NZ * DZ}), 0, {0, 0, 0}};
    BoxArray ba{domain};
    DistributionMapping dm{ba};
    MultiFab S{ba, dm, NCOMP, 0};
    MultiFab src{ba, dm, NCOMP, 0};
    MultiFab z{ba, dm, 1, 1};
    MultiFab detJ{ba, dm, 1, 1};

    explicit Column (int k_dust)
    {
        S.setVal(0.0);
        S.setVal(1.225, Rho_comp, 1);
        src.setVal(0.0);
        detJ.setVal(1.0);
        init_column(S, z, k_dust);
    }
};

DustBinDiameters one_bin (Real d)
{
    DustBinDiameters b{};
    for (int i = 0; i < DustSettlingConst::MAX_BINS; ++i) b[i] = 0.0;
    b[0] = d;
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

Real value_at (const MultiFab& mf, int k, int comp)
{
    Real v = 0.0;
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box b(IntVect(0, 0, k), IntVect(0, 0, k));
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

TEST(DustColumn, StokesVelocityIsTheSlipCorrectedStokesLaw)
{
    // 7 um quartz in air: no-slip Stokes (rho_p - rho_a) g d^2 / (18 mu) times
    // a Cunningham factor slightly above 1 (~1.02 at this size)
    const Real d = 7.0e-6, rho_p = 2650.0, rho_a = 1.225, mu = DustSettlingConst::MU_AIR_STD;
    const Real vs = compute_stokes_settling(d, rho_p, rho_a, mu);
    const Real vs_noslip = (rho_p - rho_a) * 9.81 * d * d / (18.0 * mu);
    EXPECT_GT(vs / vs_noslip, Real(1.0));
    EXPECT_LT(vs / vs_noslip, Real(1.05));
}

TEST(DustColumn, SettlingMovesDustDownFromTheState)
{
    Column c(5);
    const Real d = 7.0e-6, rho_p = 2650.0;
    apply_dust_settling_to_cc_source(c.src, c.S, c.detJ, c.geom, one_bin(d), one_weight(), 1, rho_p, DUST);
    const Real vs = compute_stokes_settling(d, rho_p, Real(1.225), DustSettlingConst::MU_AIR_STD);
    ASSERT_GT(vs, Real(0.0));
    // the dusty cell loses v_s rho / dz, the cell BELOW gains it, nothing else moves
    EXPECT_NEAR(value_at(c.src, 5, DUST), -vs / DZ, 1e-12 + 1e-6 * vs / DZ);
    EXPECT_NEAR(value_at(c.src, 4, DUST),  vs / DZ, 1e-12 + 1e-6 * vs / DZ);
    EXPECT_NEAR(value_at(c.src, 6, DUST), 0.0, 1e-15);
    Real column = 0.0;
    for (int k = 0; k < NZ; ++k) { column += value_at(c.src, k, DUST) * DZ; }
    EXPECT_NEAR(column, 0.0, 1e-12 + REAL_RTOL * vs) << "settling inside the column conserves its mass";
}

TEST(DustColumn, DepositionRemovesTheSurfaceDustAtTheDepositionVelocity)
{
    Column c(0);
    const Box dom2d(IntVect(0, 0, 0), IntVect(0, 0, 0));
    BoxArray ba2d(dom2d);
    DistributionMapping dm2d(ba2d);
    MultiFab ustar(ba2d, dm2d, 1, 0), dep(ba2d, dm2d, 1, 0);
    ustar.setVal(0.5);
    const Real d = 7.0e-6, rho_p = 2650.0, E0 = 3.0e-3;
    apply_dust_deposition_bc(c.src, dep, c.S, ustar, c.detJ, c.geom, one_bin(d), one_weight(), 1, rho_p, E0, DUST, false);
    const Real vs = compute_stokes_settling(d, rho_p, Real(1.225), DustSettlingConst::MU_AIR_STD);
    const Real vd = compute_deposition_velocity(vs, Real(0.5), E0);
    ASSERT_GT(vd, vs);
    EXPECT_NEAR(value_at(dep, 0, 0), vd * 1.0, REAL_RTOL * vd) << "deposition flux [kg/m2/s]";
    EXPECT_NEAR(value_at(c.src, 0, DUST), -vd / DZ, REAL_RTOL * vd / DZ);
}

TEST(DustColumn, BinsShareTheVerticalFlux)
{
    const Box dom2d(IntVect(0, 0, 0), IntVect(0, 0, 0));
    BoxArray ba(dom2d);
    DistributionMapping dm(ba);
    const int nb = 3;
    MultiFab flux(ba, dm, nb, 0), ut(ba, dm, 1, 0), us(ba, dm, 1, 0), silt(ba, dm, 1, 0);
    ut.setVal(0.25); us.setVal(0.5); silt.setVal(0.1);
    compute_dust_emission_flux(flux, ut, us, silt, nb, Real(1.225));
    const Real Qs    = compute_saltation_flux(Real(0.5), Real(0.25), Real(1.225));
    const Real total = compute_vertical_emission_flux(compute_sandblasting_efficiency(Real(0.1)), Real(0.1), Qs);
    ASSERT_GT(total, Real(0.0));
    Real sum = 0.0;
    for (int b = 0; b < nb; ++b) { sum += value_at(flux, 0, b); }
    EXPECT_NEAR(sum, total, REAL_RTOL * total) << "the bins together carry the M&B flux once";
}
