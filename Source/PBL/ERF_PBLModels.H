#ifndef ERF_PBLMODELS_H_
#define ERF_PBLMODELS_H_

#include <ERF_Thetav.H>

/**
 * Compute eddy diffusivities of momentum (eddy viscosity) and heat using the
 * Mellor-Yamada-Nakanishi-Niino Level 2.5 PBL scheme
 *
 * @param[in] xvel Velocity in x-dir
 * @param[in] yvel Velocity in y-dir
 * @param[in] cons_in Cell center conserved quantities
 * @param[out] eddyViscosity Holds turbulent viscosity
 * @param[in] geom Problem geometry
 * @param[in] turbChoice Container with turbulence parameters
 * @param[in] most Pointer to Monin-Obukhov class if instantiated
 * @param[in] use_moisture If we have microphysics enabled
 * @param[in] level Current level
 * @param[in] bc_ptr Pointer to array with boundary condition info
 * @param[in] vert_only Only compute vertical eddy diffusivities
 * @param[in] z_phys_nd Physical location of grid nodes, if terrain (or grid stretching) is enabled
 */
void
ComputeDiffusivityMYNN25 (const amrex::MultiFab& xvel,
                          const amrex::MultiFab& yvel,
                          const amrex::MultiFab& cons_in,
                          amrex::MultiFab& eddyViscosity,
                          const amrex::Geometry& geom,
                          const TurbChoice& turbChoice,
                          std::unique_ptr<ABLMost>& most,
                          bool use_moisture,
                          int level,
                          const amrex::BCRec* bc_ptr,
                          bool /*vert_only*/,
                          const std::unique_ptr<amrex::MultiFab>& z_phys_nd,
                          const int RhoQv_comp,
                          const int RhoQc_comp,
                          const int RhoQr_comp);

/**
 * Compute eddy diffusivities of momentum (eddy viscosity) and heat using the
 * Yonsei University PBL scheme
 *
 * @param[in] xvel Velocity in x-dir
 * @param[in] yvel Velocity in y-dir
 * @param[in] cons_in Cell center conserved quantities
 * @param[out] eddyViscosity Holds turbulent viscosity
 * @param[in] geom Problem geometry
 * @param[in] turbChoice Container with turbulence parameters
 * @param[in] most Pointer to Monin-Obukhov class if instantiated
 * @param[in] use_moisture If we have microphysics enabled
 * @param[in] level Current level
 * @param[in] bc_ptr Pointer to array with boundary condition info
 * @param[in] vert_only Only compute vertical eddy diffusivities
 * @param[in] z_phys_nd Physical location of grid nodes, if terrain (or grid stretching) is enabled
 */
void
ComputeDiffusivityYSU (const amrex::MultiFab& xvel,
                       const amrex::MultiFab& yvel,
                       const amrex::MultiFab& cons_in,
                       amrex::MultiFab& eddyViscosity,
                       const amrex::Geometry& geom,
                       const TurbChoice& turbChoice,
                       std::unique_ptr<ABLMost>& most,
                       bool use_moisture,
                       int level,
                       const amrex::BCRec* bc_ptr,
                       bool /*vert_only*/,
                       const std::unique_ptr<amrex::MultiFab>& z_phys_nd);


/**
 * Function for computing vertical derivatives for use in PBL model
 *
 * @param[in] u velocity in x-dir
 * @param[in] v velocity in y-dir
 * @param[in] cell_data conserved cell center vars
 */
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
ComputeVerticalDerivativesPBL (int i, int j, int k,
                               const amrex::Array4<const amrex::Real>& uvel,
                               const amrex::Array4<const amrex::Real>& vvel,
                               const amrex::Array4<const amrex::Real>& cell_data,
                               const int izmin,
                               const int izmax,
                               const amrex::Real& dz_inv,
                               const bool c_ext_dir_on_zlo,
                               const bool c_ext_dir_on_zhi,
                               const bool u_ext_dir_on_zlo,
                               const bool u_ext_dir_on_zhi,
                               const bool v_ext_dir_on_zlo,
                               const bool v_ext_dir_on_zhi,
                               amrex::Real& dthetadz,
                               amrex::Real& dudz,
                               amrex::Real& dvdz,
                               const int RhoQv_comp,
                               const int RhoQc_comp,
                               const int RhoQr_comp,
                               const bool use_most=true)
{
    if ( k==izmax && c_ext_dir_on_zhi ) {
        dthetadz = (1.0/3.0)*(       -Thetav(i,j,k-1,cell_data,RhoQv_comp,RhoQc_comp,RhoQr_comp)
                              - 3.0 * Thetav(i,j,k  ,cell_data,RhoQv_comp,RhoQc_comp,RhoQr_comp)
                              + 4.0 * Thetav(i,j,k+1,cell_data,RhoQv_comp,RhoQc_comp,RhoQr_comp) )*dz_inv;
    } else if ( k==izmin && c_ext_dir_on_zlo ) {
        dthetadz = (1.0/3.0)*(        Thetav(i,j,k+1,cell_data,RhoQv_comp,RhoQc_comp,RhoQr_comp)
                              + 3.0 * Thetav(i,j,k  ,cell_data,RhoQv_comp,RhoQc_comp,RhoQr_comp)
                              - 4.0 * Thetav(i,j,k-1,cell_data,RhoQv_comp,RhoQc_comp,RhoQr_comp) )*dz_inv;
    } else if ( k==izmin && use_most ) {
        dthetadz = (  Thetav(i,j,k+1,cell_data,RhoQv_comp,RhoQc_comp,RhoQr_comp)
                    - Thetav(i,j,k  ,cell_data,RhoQv_comp,RhoQc_comp,RhoQr_comp) )*dz_inv;
    } else {
        dthetadz = 0.5*(  Thetav(i,j,k+1,cell_data,RhoQv_comp,RhoQc_comp,RhoQr_comp)
                        - Thetav(i,j,k-1,cell_data,RhoQv_comp,RhoQc_comp,RhoQr_comp) )*dz_inv;
    }

    if ( k==izmax && u_ext_dir_on_zhi ) {
        dudz = (1.0/6.0)*( (-uvel(i  ,j,k-1) - 3.0 * uvel(i  ,j,k  ) + 4.0 * uvel(i  ,j,k+1))
                         + (-uvel(i+1,j,k-1) - 3.0 * uvel(i+1,j,k  ) + 4.0 * uvel(i+1,j,k+1)) )*dz_inv;
    } else if ( k==izmin && u_ext_dir_on_zlo ) {
        dudz = (1.0/6.0)*( (uvel(i  ,j,k+1) + 3.0 * uvel(i  ,j,k  ) - 4.0 * uvel(i  ,j,k-1))
                         + (uvel(i+1,j,k+1) + 3.0 * uvel(i+1,j,k  ) - 4.0 * uvel(i+1,j,k-1)) )*dz_inv;
    } else if ( k==izmin && use_most ) {
        dudz = 0.50*( uvel(i,j,k+1) - uvel(i,j,k  ) + uvel(i+1,j,k+1) - uvel(i+1,j,k  ) )*dz_inv;
    } else {
        dudz = 0.25*( uvel(i,j,k+1) - uvel(i,j,k-1) + uvel(i+1,j,k+1) - uvel(i+1,j,k-1) )*dz_inv;
    }

    if ( k==izmax && v_ext_dir_on_zhi ) {
        dvdz = (1.0/6.0)*( (-vvel(i,j  ,k-1) - 3.0 * vvel(i,j  ,k  ) + 4.0 * vvel(i,j  ,k+1))
                         + (-vvel(i,j+1,k-1) - 3.0 * vvel(i,j+1,k  ) + 4.0 * vvel(i,j+1,k+1)) )*dz_inv;
    } else if ( k==izmin && v_ext_dir_on_zlo ) {
        dvdz = (1.0/6.0)*( (vvel(i,j  ,k+1) + 3.0 * vvel(i,j  ,k  ) - 4.0 * vvel(i,j  ,k-1))
                         + (vvel(i,j+1,k+1) + 3.0 * vvel(i,j+1,k  ) - 4.0 * vvel(i,j+1,k-1)) )*dz_inv;
    } else if ( k==izmin && use_most ) {
        dvdz = 0.50*( vvel(i,j,k+1) - vvel(i,j,k  ) + vvel(i,j+1,k+1) - vvel(i,j+1,k  ) )*dz_inv;
    } else {
        dvdz = 0.25*( vvel(i,j,k+1) - vvel(i,j,k-1) + vvel(i,j+1,k+1) - vvel(i,j+1,k-1) )*dz_inv;
    }
}

/**
 * Function for computing the QKE source terms (NN09, Eqn. 5).
 *
 * @param[in] u velocity in x-dir
 * @param[in] v velocity in y-dir
 * @param[in] cell_data conserved cell center vars
 * @param[in] cell_prim primitive cell center vars
 * @param[in] K_turb turbulent viscosity
 * @param[in] cellSizeInv inverse cell size array
 * @param[in] domain box of the whole domain
 * @param[in] pbl_mynn_B1_l a parameter
 * @param[in] theta_mean average theta
 */
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
ComputeQKESourceTerms (int i, int j, int k,
                       const amrex::Array4<const amrex::Real>& uvel,
                       const amrex::Array4<const amrex::Real>& vvel,
                       const amrex::Array4<const amrex::Real>& cell_data,
                       const amrex::Array4<const amrex::Real>& cell_prim,
                       const amrex::Array4<const amrex::Real>& K_turb,
                       const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
                       const amrex::Box& domain,
                       amrex::Real pbl_mynn_B1_l,
                       const amrex::Real theta_mean,
                       const int RhoQv_comp,
                       const int RhoQc_comp,
                       const int RhoQr_comp,
                       bool c_ext_dir_on_zlo,
                       bool c_ext_dir_on_zhi,
                       bool u_ext_dir_on_zlo,
                       bool u_ext_dir_on_zhi,
                       bool v_ext_dir_on_zlo,
                       bool v_ext_dir_on_zhi,
                       const bool use_most=false,
                       const amrex::Real met_h_zeta=1.0)
{
    // Compute some relevant derivatives
    amrex::Real dthetadz, dudz, dvdz;
    amrex::Real source_term = 0.0;

    amrex::Real dz_inv = cellSizeInv[2];
    int izmin = domain.smallEnd(2);
    int izmax = domain.bigEnd(2);

    // NOTE: With MOST, the ghost cells are filled AFTER k_turb is computed
    //       so that the non-explicit pathway works. Therefore, at this
    //       point we DO have valid ghost cells from MOST. We are passing
    //       the MOST flag to use one-sided diffs here to be consistent with
    //       the explicit pathway.

    ComputeVerticalDerivativesPBL(i, j, k,
                                  uvel, vvel, cell_data, izmin, izmax, dz_inv/met_h_zeta,
                                  c_ext_dir_on_zlo, c_ext_dir_on_zhi,
                                  u_ext_dir_on_zlo, u_ext_dir_on_zhi,
                                  v_ext_dir_on_zlo, v_ext_dir_on_zhi,
                                  dthetadz, dudz, dvdz,
                                  RhoQv_comp, RhoQc_comp, RhoQr_comp, use_most);

    // Note: Transport terms due to turbulence and pressure are included when
    // DiffusionSrcForState_* is called from ERF_slow_rhs_post.

    // Shear Production
    source_term += K_turb(i,j,k,EddyDiff::Mom_v) * (dudz*dudz + dvdz*dvdz);

    // Buoyancy Production
    source_term -= (CONST_GRAV/theta_mean)*K_turb(i,j,k,EddyDiff::Theta_v)*dthetadz;

    // Dissipation
    amrex::Real qke = 2.0 * cell_prim(i,j,k,PrimKE_comp);
    if (std::abs(qke) > 0.0) {
        source_term -= cell_data(i,j,k,Rho_comp) * std::pow(qke,1.5) /
                       (pbl_mynn_B1_l * K_turb(i,j,k,EddyDiff::Turb_lengthscale));
    }

    return source_term;
}
#endif
