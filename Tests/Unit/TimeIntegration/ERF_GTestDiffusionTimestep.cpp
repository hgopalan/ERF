#include <limits>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_MultiFab.H>

#include <gtest/gtest.h>

#include "ERF_IndexDefines.H"
#include "TimeIntegration/ERF_DiffusionTimestep.H"

// Motivation: the anelastic integrators treat horizontal diffusion (and, without the implicit
// vertical solve, vertical diffusion) explicitly, but the time step only saw advection. With a
// k-eqn closure K_theta is three times K_m and grew past the explicit bound, and a grid-scale
// checkerboard in theta blew the runs up. The limit must take the largest diffusivity of any
// variable, horizontal and vertical separately, divided by the density, in every cell, and drop
// the directions that have no grid-scale mode (one cell wide, or implicit in the vertical).

using namespace amrex;

namespace {

constexpr int n = 8;
const double tol = 100. * std::numeric_limits<Real>::epsilon();
// the one cell with a small density and large theta and moisture diffusivities
constexpr int is = 5, js = 6, ks = 7;

void fill (MultiFab& cons, MultiFab& K)
{
    for (MFIter mfi(cons); mfi.isValid(); ++mfi) {
        auto const& s  = cons.array(mfi);
        auto const& mu = K.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const bool special = (i == is && j == js && k == ks);
            s(i,j,k,Rho_comp) = special ? Real(1.) : Real(2.);
            for (int c = 0; c < EddyDiff::NumDiffs; ++c) { mu(i,j,k,c) = Real(0.); }
            mu(i,j,k,EddyDiff::Mom_h)    = Real(1.);
            mu(i,j,k,EddyDiff::Theta_h)  = special ? Real(10.) : Real(3.);
            mu(i,j,k,EddyDiff::KE_h)     = Real(0.5);
            mu(i,j,k,EddyDiff::Mom_v)    = Real(2.);
            mu(i,j,k,EddyDiff::Theta_v)  = Real(4.);
            mu(i,j,k,EddyDiff::Q_v)      = special ? Real(20.) : Real(0.);
        });
    }
}

struct Fields {
    MultiFab cons, K;
    Fields () {
        BoxArray ba(Box(IntVect(0), IntVect(n-1)));
        ba.maxSize(4);
        DistributionMapping dm(ba);
        cons.define(ba, dm, 2, 0);   // density and rho theta
        K.define(ba, dm, EddyDiff::NumDiffs, 0);
        fill(cons, K);
    }
};

} // namespace

TEST(DiffusionTimestep, DropsDirectionsWithoutAGridScaleMode)
{
    const GpuArray<Real,3> dxinv{Real(0.1), Real(0.05), Real(0.2)};
    const auto all = erf_dt::diffusion_inv_dx2(dxinv, n, n, true);
    EXPECT_NEAR(all[0], 0.01, 0.01 * tol);
    EXPECT_NEAR(all[1], 0.0025, 0.0025 * tol);
    EXPECT_NEAR(all[2], 0.04, 0.04 * tol);

    const auto thin = erf_dt::diffusion_inv_dx2(dxinv, 1, 1, false);
    EXPECT_EQ(thin[0], Real(0.));
    EXPECT_EQ(thin[1], Real(0.));
    EXPECT_EQ(thin[2], Real(0.));
}

TEST(DiffusionTimestep, TakesTheLargestDiffusivityOverDensity)
{
    Fields f;
    const GpuArray<Real,3> dxinv{Real(0.1), Real(0.05), Real(0.2)};
    // explicit vertical: 2 (K_h (0.01 + 0.0025) + K_v 0.04) / rho, largest in the special cell
    // (K_h = 10 from theta, K_v = 20 from moisture, rho = 1): 2 (0.125 + 0.8) = 1.85; elsewhere
    // 2 (3 x 0.0125 + 4 x 0.04) / 2 = 0.1975
    const Real r = erf_dt::anelastic_diffusion_inv_dt(f.cons, f.K, erf_dt::diffusion_inv_dx2(dxinv, n, n, true));
    EXPECT_NEAR(r, 1.85, 1.85 * tol);
}

TEST(DiffusionTimestep, ImplicitVerticalLeavesTheHorizontalLimit)
{
    Fields f;
    const GpuArray<Real,3> dxinv{Real(0.1), Real(0.05), Real(0.2)};
    // implicit vertical: 2 x 10 x 0.0125 / 1 = 0.25 in the special cell
    const Real r = erf_dt::anelastic_diffusion_inv_dt(f.cons, f.K, erf_dt::diffusion_inv_dx2(dxinv, n, n, false));
    EXPECT_NEAR(r, 0.25, 0.25 * tol);
}
