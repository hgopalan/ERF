#include <cmath>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <gtest/gtest.h>

#ifdef ERF_USE_FFT
#include <AMReX_GMRES.H>
#include "ERF_ImmersedPoisson.H"
#endif

// Motivation: with immersed-forcing terrain the anelastic projection treated the solid as fluid,
// and the solid faces were zeroed after it. The faces between a solid and a fluid cell kept their
// projected flux, so the solid row under the surface became a mass sink and source (theta fell
// 3 K below its initial value in a two-level run). erf.if_implicit_projection puts the drag of
// the solid inside the projection, m = sigma (m* - grad phi) with sigma = 1 / (1 + dt rate) on
// each face, which needs div(sigma grad phi) = div(sigma m*). ImmersedPoisson is that operator,
// solved with FFT-preconditioned GMRES. The projected momenta must be divergence free in every
// cell, the solid and the surface cells included.

using namespace amrex;

namespace {

constexpr int nx = 16;
constexpr int ny = 2;
constexpr int nz = 16;
constexpr int k_surf = 6;   // cells below k_surf are solid

#ifdef ERF_USE_FFT
// sigma of the implicit drag: 1 / (1 + dt rate), a strong drag in the solid, none in the fluid,
// half the rate on the surface z-face (the face average of the cell rates), 1 on the domain faces
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real sigma_of (int dir, int k)
{
    const Real a_solid = Real(4.);
    if (dir < 2) {
        return (k < k_surf) ? Real(1.) / (Real(1.) + a_solid) : Real(1.);
    }
    if (k == 0 || k == nz) { return Real(1.); }
    if (k <  k_surf) { return Real(1.) / (Real(1.) + a_solid); }
    if (k == k_surf) { return Real(1.) / (Real(1.) + Real(0.5) * a_solid); }
    return Real(1.);
}

// A momentum field that is not divergence free, with no flux through the top and bottom walls
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real mom_of (int dir, int i, int j, int k)
{
    const Real pi = Real(3.14159265358979323846);
    const Real x = Real(i) / nx;
    const Real z = Real(k) / nz;
    if (dir == 0) { return std::sin(2*pi*x) * std::cos(pi*(k + Real(0.5))/nz) + Real(2.); }
    if (dir == 1) { return Real(0.3) * std::cos(2*pi*(j + Real(0.5))/ny); }
    return std::sin(pi*z) * std::cos(2*pi*(i + Real(0.5))/nx) + Real(0.5) * std::sin(pi*z);
}

// sigma on the faces of direction d, and sigma m* in mom
void fill_face (int d, MultiFab& sigma, MultiFab& mom)
{
    for (MFIter mfi(sigma); mfi.isValid(); ++mfi) {
        auto const& s = sigma.array(mfi);
        auto const& m = mom.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            s(i,j,k) = sigma_of(d, k);
            m(i,j,k) = mom_of(d, i, j, k) * s(i,j,k);
        });
    }
}

// Largest |value| on the z-faces at height index k
Real max_abs_on_zface (MultiFab const& mz, int k)
{
    Real mx = Real(0.);
    for (MFIter mfi(mz); mfi.isValid(); ++mfi) {
        const Box& vb = mfi.validbox();
        if (vb.smallEnd(2) <= k && k <= vb.bigEnd(2)) {
            mx = amrex::max(mx, mz[mfi].template maxabs<RunOn::Device>(makeSlab(vb, 2, k), 0));
        }
    }
    ParallelDescriptor::ReduceRealMax(mx);
    return mx;
}

// Cell divergence of the face momenta (unit cells)
void divergence (MultiFab& out, Array<MultiFab,AMREX_SPACEDIM> const& m)
{
    for (MFIter mfi(out); mfi.isValid(); ++mfi) {
        auto const& o  = out.array(mfi);
        auto const& mx = m[0].const_array(mfi);
        auto const& my = m[1].const_array(mfi);
        auto const& mz = m[2].const_array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            o(i,j,k) = (mx(i+1,j,k) - mx(i,j,k)) + (my(i,j+1,k) - my(i,j,k)) + (mz(i,j,k+1) - mz(i,j,k));
        });
    }
}
#endif

} // namespace

TEST(ImmersedPoisson, ProjectedMomentaAreDivergenceFreeInTheSolidToo)
{
#ifndef ERF_USE_FFT
    GTEST_SKIP() << "ImmersedPoisson needs a build with FFT";
#else
    const Box domain(IntVect(0), IntVect(nx-1, ny-1, nz-1));
    RealBox rb({Real(0.), Real(0.), Real(0.)}, {Real(nx), Real(ny), Real(nz)});
    Array<int,AMREX_SPACEDIM> is_per{1, 1, 0};
    Geometry geom(domain, rb, 0, is_per);

    BoxArray ba(domain);
    ba.maxSize(IntVect(8, 2, 8));
    DistributionMapping dm(ba);

    Array<std::string,2*AMREX_SPACEDIM> bcs{"Periodic", "Periodic", "SlipWall",
                                            "Periodic", "Periodic", "SlipWall"};

    Array<MultiFab,AMREX_SPACEDIM> sigma, mom, flux;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        const BoxArray fba = convert(ba, IntVect::TheDimensionVector(d));
        sigma[d].define(fba, dm, 1, 0);
        mom[d].define(fba, dm, 1, 0);
        flux[d].define(fba, dm, 1, 0);
        fill_face(d, sigma[d], mom[d]);
    }

    MultiFab rhs(ba, dm, 1, 0);
    MultiFab phi(ba, dm, 1, 1);
    divergence(rhs, mom);
    const Real rhs_max = rhs.norm0();
    ASSERT_GT(rhs_max, Real(0.1));
    phi.setVal(Real(0.));

    ImmersedPoisson ip(geom, geom, ba, dm, bcs, {&sigma[0], &sigma[1], &sigma[2]}, false);
    ip.usePrecond(true);
    GMRES<MultiFab, ImmersedPoisson> gmres;
    gmres.define(ip);
    gmres.setRestartLength(50);
    gmres.solve(phi, rhs, Real(1.e-12), Real(0.));
    ip.getFluxes(phi, flux);

    // m = sigma m* + sigma (-grad phi)
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        MultiFab::Multiply(flux[d], sigma[d], 0, 0, 1, 0);
        MultiFab::Add(mom[d], flux[d], 0, 0, 1, 0);
    }
    MultiFab div(ba, dm, 1, 0);
    divergence(div, mom);
    EXPECT_LT(div.norm0(), Real(1.e-9) * rhs_max);

    // nothing flows through the top and bottom walls (the even ghost cells of phi)
    EXPECT_LT(max_abs_on_zface(mom[2], 0),  Real(1.e-12));
    EXPECT_LT(max_abs_on_zface(mom[2], nz), Real(1.e-12));

    // the constant-coefficient FFT preconditioner is the exact inverse in the fluid, so GMRES
    // converges in a few iterations even with a fivefold contrast in sigma
    EXPECT_LT(gmres.getNumIters(), 40);
#endif
}
