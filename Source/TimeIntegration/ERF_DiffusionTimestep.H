#ifndef ERF_DIFFUSION_TIMESTEP_H_
#define ERF_DIFFUSION_TIMESTEP_H_

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include "ERF_IndexDefines.H"

/**
 * Explicit-diffusion time-step limit of the anelastic integrators (erf.diffusive_cfl).
 *
 * The anelastic integrators treat horizontal diffusion explicitly (MidPoint: explicit midpoint,
 * stable for dt * 4 K / dx^2 <= 2 per direction) and, without the implicit vertical solve,
 * vertical diffusion too; the advective dt does not see it. With a turbulence closure K can exceed
 * that bound (k-eqn RANS: K_theta = K_m / Pr_t, three times K_m) and a grid-scale checkerboard in the
 * most diffusive scalar grows while the velocity stays smooth.
 */
namespace erf_dt {

/**
 * 1/dx^2 per direction for the limit. A direction one cell wide has no grid-scale mode, and the
 * vertical has none to limit when it is solved implicitly: both give 0.
 *
 * @param[in] dxinv         inverse cell sizes (x, y and the smallest dz)
 * @param[in] nx            level-0 cells in x
 * @param[in] ny            level-0 cells in y
 * @param[in] vert_explicit true unless the vertical diffusion is implicit
 */
AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real,3>
diffusion_inv_dx2 (amrex::GpuArray<amrex::Real,3> const& dxinv, int nx, int ny, bool vert_explicit) noexcept
{
    const amrex::Real zero = amrex::Real(0.0);
    return {(nx > 1) ? dxinv[0]*dxinv[0] : zero,
            (ny > 1) ? dxinv[1]*dxinv[1] : zero,
            vert_explicit ? dxinv[2]*dxinv[2] : zero};
}

/**
 * Largest 2 (K_h (1/dx^2 + 1/dy^2) + K_v / dz^2) / rho over the valid cells and all ranks, where
 * K_h and K_v are the largest horizontal and vertical diffusivities of the momentum, theta, TKE,
 * scalar and moisture. The limit is dt = erf.diffusive_cfl / this (no limit when it is 0).
 *
 * @param[in] cons  conserved state (density in Rho_comp)
 * @param[in] K     eddy diffusivities (EddyDiff components)
 * @param[in] inv2  1/dx^2 per direction from diffusion_inv_dx2
 */
inline amrex::Real
anelastic_diffusion_inv_dt (amrex::MultiFab const& cons, amrex::MultiFab const& K,
                            amrex::GpuArray<amrex::Real,3> const& inv2)
{
    using namespace amrex;
    const Real hx2 = inv2[0], hy2 = inv2[1], hz2 = inv2[2];
    Real r = ReduceMax(cons, K, 0,
        [=] AMREX_GPU_HOST_DEVICE (Box const& b, Array4<Real const> const& s,
                                   Array4<Real const> const& mu) -> Real
    {
        Real rmax = Real(0.0);
        Loop(b, [=,&rmax] (int i, int j, int k) noexcept
        {
            const Real inv_rho = Real(1.0) / s(i,j,k,Rho_comp);
            const Real kh = amrex::max(amrex::max(mu(i,j,k,EddyDiff::Mom_h), mu(i,j,k,EddyDiff::Theta_h)),
                                       amrex::max(mu(i,j,k,EddyDiff::KE_h),
                                                  amrex::max(mu(i,j,k,EddyDiff::Scalar_h), mu(i,j,k,EddyDiff::Q_h))));
            const Real kv = amrex::max(amrex::max(mu(i,j,k,EddyDiff::Mom_v), mu(i,j,k,EddyDiff::Theta_v)),
                                       amrex::max(mu(i,j,k,EddyDiff::KE_v),
                                                  amrex::max(mu(i,j,k,EddyDiff::Scalar_v), mu(i,j,k,EddyDiff::Q_v))));
            rmax = amrex::max(rmax, Real(2.0) * inv_rho * (kh * (hx2 + hy2) + kv * hz2));
        });
        return rmax;
    });
    ParallelDescriptor::ReduceRealMax(r);
    return r;
}

} // namespace erf_dt

#endif
