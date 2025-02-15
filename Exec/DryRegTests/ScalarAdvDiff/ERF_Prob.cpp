#include "ERF_Prob.H"

using namespace amrex;

std::unique_ptr<ProblemBase>
amrex_probinit(
    const amrex_real* /*problo*/,
    const amrex_real* /*probhi*/)
{
    return std::make_unique<Problem>();
}

Problem::Problem()
{
    // Parse params
    ParmParse pp("prob");
    pp.query("rho_0", parms.rho_0);
    pp.query("T_0", parms.T_0);
    pp.query("A_0", parms.A_0);
    pp.query("B_0", parms.B_0);
    pp.query("u_0", parms.u_0);
    pp.query("v_0", parms.v_0);
    pp.query("w_0", parms.w_0);
    pp.query("rad_0", parms.rad_0);
    pp.query("z0", parms.z0);
    pp.query("zRef", parms.zRef);
    pp.query("uRef", parms.uRef);

    pp.query("xc_frac", parms.xc_frac);
    pp.query("yc_frac", parms.yc_frac);
    pp.query("zc_frac", parms.zc_frac);

    pp.query("prob_type", parms.prob_type);

    init_base_parms(parms.rho_0, parms.T_0);
}

void
Problem::erf_init_rayleigh(
    amrex::Vector<amrex::Vector<amrex::Real> >& rayleigh_ptrs,
    amrex::Geometry const& geom,
    std::unique_ptr<MultiFab>& /*z_phys_nd*/,
    amrex::Real /*zdamp*/)
{
  const int khi = geom.Domain().bigEnd()[2];

  // We just use these values to test the Rayleigh damping
  for (int k = 0; k <= khi; k++)
  {
      rayleigh_ptrs[Rayleigh::ubar][k]     = 2.0;
      rayleigh_ptrs[Rayleigh::vbar][k]     = 1.0;
      rayleigh_ptrs[Rayleigh::wbar][k]     = 0.0;
      rayleigh_ptrs[Rayleigh::thetabar][k] = parms.T_0;
  }
}

void
Problem::init_custom_pert(
    const Box& bx,
    const Box& xbx,
    const Box& ybx,
    const Box& zbx,
    Array4<Real const> const& /*state*/,
    Array4<Real      > const& state_pert,
    Array4<Real      > const& x_vel_pert,
    Array4<Real      > const& y_vel_pert,
    Array4<Real      > const& z_vel_pert,
    Array4<Real      > const& /*r_hse*/,
    Array4<Real      > const& /*p_hse*/,
    Array4<Real const> const& /*z_nd*/,
    Array4<Real const> const& /*z_cc*/,
    GeometryData const& geomdata,
    Array4<Real const> const& /*mf_m*/,
    Array4<Real const> const& /*mf_u*/,
    Array4<Real const> const& /*mf_v*/,
    const SolverChoice& sc)
{
    const bool use_moisture = (sc.moisture_type != MoistureType::None);

  ParallelFor(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
    // Geometry
    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx = geomdata.CellSize();
    const Real x = prob_lo[0] + (i + 0.5) * dx[0];
    const Real y = prob_lo[1] + (j + 0.5) * dx[1];
    const Real z = prob_lo[2] + (k + 0.5) * dx[2];

    // Define a point (xc,yc,zc) at the center of the domain
    const Real xc = parms_d.xc_frac * (prob_lo[0] + prob_hi[0]);
    const Real yc = parms_d.yc_frac * (prob_lo[1] + prob_hi[1]);
    const Real zc = parms_d.zc_frac * (prob_lo[2] + prob_hi[2]);

    // Define ellipse parameters
    const Real r0   = parms_d.rad_0 * (prob_hi[0] - prob_lo[0]);

    const Real r3d    = std::sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc));
    const Real r2d_xy = std::sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc));
    const Real r2d_xz = std::sqrt((x-xc)*(x-xc) + (z-zc)*(z-zc));

    if (parms_d.prob_type == 10)
    {
        // Set scalar = A_0*exp(-10r^2), where r is distance from center of domain,
        //            + B_0*sin(x)
        state_pert(i, j, k, RhoScalar_comp) = parms_d.A_0 * exp(-10.*r3d*r3d) + parms_d.B_0*sin(x);

    } else if (parms_d.prob_type == 11) {
        state_pert(i, j, k, RhoScalar_comp) = parms_d.A_0 * 0.25 * (1.0 + std::cos(PI * std::min(r2d_xy, r0) / r0));
    } else if (parms_d.prob_type == 12) {
        state_pert(i, j, k, RhoScalar_comp) = parms_d.A_0 * 0.25 * (1.0 + std::cos(PI * std::min(r2d_xz, r0) / r0));
    } else if (parms_d.prob_type == 13) {
        const Real r0_z = parms_d.rad_0 * (prob_hi[2] - prob_lo[2]);
        //ellipse for mapfac shear validation
        const Real r2d_xz_ell = std::sqrt((x-xc)*(x-xc)/(r0*r0) + (z-zc)*(z-zc)/(r0_z*r0_z));
        state_pert(i, j, k, RhoScalar_comp) = parms_d.A_0 * 0.25 * (1.0 + std::cos(PI * std::min(r2d_xz_ell, r0_z) / r0_z));
    } else if (parms_d.prob_type == 14) {
        state_pert(i, j, k, RhoScalar_comp) = std::cos(PI*x);
    } else {
        // Set scalar = A_0 in a ball of radius r0 and 0 elsewhere
        if (r3d < r0) {
           state_pert(i, j, k, RhoScalar_comp) = parms_d.A_0;
        } else {
           state_pert(i, j, k, RhoScalar_comp) = 0.0;
        }
    }

    state_pert(i, j, k, RhoScalar_comp) *= parms_d.rho_0;

    if (use_moisture) {
        state_pert(i, j, k, RhoQ1_comp) = 0.0;
        state_pert(i, j, k, RhoQ2_comp) = 0.0;
    }
  });

  // Set the x-velocity
  ParallelFor(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      const Real* prob_lo = geomdata.ProbLo();
      const Real*      dx = geomdata.CellSize();
      const Real        z = prob_lo[2] + (k + 0.5) * dx[2];

      // Set the x-velocity
      if (parms_d.uRef != 0.0) {
          x_vel_pert(i, j, k) = parms_d.u_0 + parms_d.uRef *
                                std::log((z + parms_d.z0)/parms_d.z0)/
                                std::log((parms_d.zRef +parms_d.z0)/parms_d.z0);
      } else {
          x_vel_pert(i, j, k) = parms_d.u_0;
      }
  });

  // Set the y-velocity
  ParallelFor(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      y_vel_pert(i, j, k) = parms_d.v_0;
  });

  // Set the z-velocity
  ParallelFor(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
  {
      z_vel_pert(i, j, k) = parms_d.w_0;
  });
}

#if 0
AMREX_GPU_DEVICE
Real
dhdt(int /*i*/, int /*j*/,
     const GpuArray<Real,AMREX_SPACEDIM> /*dx*/,
     const Real /*time_mt*/, const Real /*delta_t*/)
{
    return 0.;
}
#endif
