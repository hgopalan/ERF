#include "AMReX_BCRec.H"

#include <ERF_Advection.H>
#include <ERF_AdvectionSrcForMom_N.H>
#include <ERF_AdvectionSrcForMom_T.H>

using namespace amrex;

/**
 * Function for computing the advective tendency for the momentum equations
 * This routine has explicit expressions for all cases (terrain or not) when
 * the horizontal and vertical spatial orders are <= 2, and calls more specialized
 * functions when either (or both) spatial order(s) is greater than 2.
 *
 * @param[in] bxx box over which the x-momentum is updated
 * @param[in] bxy box over which the y-momentum is updated
 * @param[in] bxz box over which the z-momentum is updated
 * @param[out] rho_u_rhs tendency for the x-momentum equation
 * @param[out] rho_v_rhs tendency for the y-momentum equation
 * @param[out] rho_w_rhs tendency for the z-momentum equation
 * @param[in] u x-component of the velocity
 * @param[in] v y-component of the velocity
 * @param[in] w z-component of the velocity
 * @param[in] rho_u x-component of the momentum
 * @param[in] rho_v y-component of the momentum
 * @param[in] Omega component of the momentum normal to the z-coordinate surface
 * @param[in] z_nd height coordinate at nodes
 * @param[in] ax   Area fraction of x-faces
 * @param[in] ay   Area fraction of y-faces
 * @param[in] az   Area fraction of z-faces
 * @param[in] detJ Jacobian of the metric transformation (= 1 if use_terrain is false)
 * @param[in] cellSizeInv inverse of the mesh spacing
 * @param[in] mf_m map factor at cell centers
 * @param[in] mf_u map factor at x-faces
 * @param[in] mf_v map factor at y-faces
 * @param[in] horiz_adv_type sets the spatial order to be used for lateral derivatives
 * @param[in] vert_adv_type  sets the spatial order to be used for vertical derivatives
 * @param[in] use_terrain if true, use the terrain-aware derivatives (with metric terms)
 */
void
AdvectionSrcForMom (const Box& bx,
                    const Box& bxx, const Box& bxy, const Box& bxz,
                    const Array4<      Real>& rho_u_rhs,
                    const Array4<      Real>& rho_v_rhs,
                    const Array4<      Real>& rho_w_rhs,
                    const Array4<const Real>& cell_data,
                    const Array4<const Real>& u,
                    const Array4<const Real>& v,
                    const Array4<const Real>& w,
                    const Array4<const Real>& rho_u,
                    const Array4<const Real>& rho_v,
                    const Array4<const Real>& Omega,
                    const Array4<const Real>& z_nd,
                    const Array4<const Real>& ax,
                    const Array4<const Real>& ay,
                    const Array4<const Real>& az,
                    const Array4<const Real>& detJ,
                    const GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv,
                    const Array4<const Real>& mf_m,
                    const Array4<const Real>& mf_u,
                    const Array4<const Real>& mf_v,
                    const AdvType horiz_adv_type,
                    const AdvType vert_adv_type,
                    const Real horiz_upw_frac,
                    const Real vert_upw_frac,
                    const bool use_terrain,
                    const int lo_z_face, const int hi_z_face,
                    const Box& domain,
                    const BCRec* bc_ptr_h)
{
    BL_PROFILE_VAR("AdvectionSrcForMom", AdvectionSrcForMom);

    auto dxInv = cellSizeInv[0], dyInv = cellSizeInv[1], dzInv = cellSizeInv[2];

    AMREX_ALWAYS_ASSERT(bxz.smallEnd(2) > 0);

    // compute mapfactor inverses
    Box box2d_u(bxx);   box2d_u.setRange(2,0);   box2d_u.grow({3,3,0});
    Box box2d_v(bxy);   box2d_v.setRange(2,0);   box2d_v.grow({3,3,0});
    FArrayBox mf_u_invFAB(box2d_u,1,The_Async_Arena());
    FArrayBox mf_v_invFAB(box2d_v,1,The_Async_Arena());
    const Array4<Real>& mf_u_inv = mf_u_invFAB.array();
    const Array4<Real>& mf_v_inv = mf_v_invFAB.array();

    ParallelFor(box2d_u, box2d_v,
    [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
    {
        mf_u_inv(i,j,0) = 1. / mf_u(i,j,0);
    },
    [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
    {
        mf_v_inv(i,j,0) = 1. / mf_v(i,j,0);
    });

#ifdef ERF_USE_EB
    amrex::ignore_unused(use_terrain);
    ParallelFor(bxx, bxy, bxz,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        rho_u_rhs(i, j, k) = 0.0;
    },
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        rho_v_rhs(i, j, k) = 0.0;
    },
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        rho_w_rhs(i, j, k) = 0.0;
    });
#else
    if (!use_terrain) {
        // Inline with 2nd order for efficiency
        if (horiz_adv_type == AdvType::Centered_2nd && vert_adv_type == AdvType::Centered_2nd)
        {
            ParallelFor(bxx, bxy, bxz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real xflux_hi = 0.25 * (rho_u(i, j  , k) * mf_u_inv(i,j,0) + rho_u(i+1, j  , k) * mf_u_inv(i+1,j,0)) * (u(i+1,j,k) + u(i,j,k));
                Real xflux_lo = 0.25 * (rho_u(i, j  , k) * mf_u_inv(i,j,0) + rho_u(i-1, j  , k) * mf_u_inv(i-1,j,0)) * (u(i-1,j,k) + u(i,j,k));

                Real yflux_hi = 0.25 * (rho_v(i, j+1, k) * mf_v_inv(i,j+1,0) + rho_v(i-1, j+1, k) * mf_v_inv(i-1,j+1,0)) * (u(i,j+1,k) + u(i,j,k));
                Real yflux_lo = 0.25 * (rho_v(i, j  , k) * mf_v_inv(i,j  ,0) + rho_v(i-1, j  , k) * mf_v_inv(i-1,j  ,0)) * (u(i,j-1,k) + u(i,j,k));

                Real zflux_hi = 0.25 * (Omega(i, j, k+1) + Omega(i-1, j, k+1)) * (u(i,j,k+1) + u(i,j,k));
                Real zflux_lo = 0.25 * (Omega(i, j, k  ) + Omega(i-1, j, k  )) * (u(i,j,k-1) + u(i,j,k));

                Real mfsq = mf_u(i,j,0) * mf_u(i,j,0);

                Real advectionSrc = (xflux_hi - xflux_lo) * dxInv * mfsq
                                  + (yflux_hi - yflux_lo) * dyInv * mfsq
                                  + (zflux_hi - zflux_lo) * dzInv;
                rho_u_rhs(i, j, k) = -advectionSrc;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real xflux_hi = 0.25 * (rho_u(i+1, j, k) * mf_u_inv(i+1,j,0) + rho_u(i+1, j-1, k) * mf_u_inv(i+1,j-1,0)) * (v(i+1,j,k) + v(i,j,k));
                Real xflux_lo = 0.25 * (rho_u(i  , j, k) * mf_u_inv(i  ,j,0) + rho_u(i  , j-1, k) * mf_u_inv(i  ,j-1,0)) * (v(i-1,j,k) + v(i,j,k));

                Real yflux_hi = 0.25 * (rho_v(i  ,j+1,k) * mf_v_inv(i,j+1,0) + rho_v(i  ,j  ,k) * mf_v_inv(i,j  ,0)) * (v(i,j+1,k) + v(i,j,k));
                Real yflux_lo = 0.25 * (rho_v(i  ,j  ,k) * mf_v_inv(i,j  ,0) + rho_v(i  ,j-1,k) * mf_v_inv(i,j-1,0) ) * (v(i,j-1,k) + v(i,j,k));

                Real zflux_hi = 0.25 * (Omega(i, j, k+1) + Omega(i, j-1, k+1)) * (v(i,j,k+1) + v(i,j,k));
                Real zflux_lo = 0.25 * (Omega(i, j, k  ) + Omega(i, j-1, k  )) * (v(i,j,k-1) + v(i,j,k));

                Real mfsq = mf_v(i,j,0) * mf_v(i,j,0);

                Real advectionSrc = (xflux_hi - xflux_lo) * dxInv * mfsq
                                  + (yflux_hi - yflux_lo) * dyInv * mfsq
                                  + (zflux_hi - zflux_lo) * dzInv;
                rho_v_rhs(i, j, k) = -advectionSrc;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real xflux_hi = 0.25*(rho_u(i+1,j  ,k) + rho_u(i+1, j, k-1)) * mf_u_inv(i+1,j  ,0) * (w(i+1,j,k) + w(i,j,k));
                Real xflux_lo = 0.25*(rho_u(i  ,j  ,k) + rho_u(i  , j, k-1)) * mf_u_inv(i  ,j  ,0) * (w(i-1,j,k) + w(i,j,k));

                Real yflux_hi = 0.25*(rho_v(i  ,j+1,k) + rho_v(i, j+1, k-1)) * mf_v_inv(i  ,j+1,0) * (w(i,j+1,k) + w(i,j,k));
                Real yflux_lo = 0.25*(rho_v(i  ,j  ,k) + rho_v(i, j  , k-1)) * mf_v_inv(i  ,j  ,0) * (w(i,j-1,k) + w(i,j,k));

                Real zflux_lo = 0.25 * (Omega(i,j,k) + Omega(i,j,k-1)) * (w(i,j,k) + w(i,j,k-1));

                Real zflux_hi = (k == hi_z_face) ? Omega(i,j,k) * w(i,j,k) :
                    0.25 * (Omega(i,j,k) + Omega(i,j,k+1)) * (w(i,j,k) + w(i,j,k+1));

                Real mfsq = mf_m(i,j,0) * mf_m(i,j,0);

                Real advectionSrc = (xflux_hi - xflux_lo) * dxInv * mfsq
                                  + (yflux_hi - yflux_lo) * dyInv * mfsq
                                  + (zflux_hi - zflux_lo) * dzInv;
                rho_w_rhs(i, j, k) = -advectionSrc;
            });
        // Template higher order methods
        } else {
            if (horiz_adv_type == AdvType::Centered_2nd) {
                AdvectionSrcForMomVert_N<CENTERED2>(bxx, bxy, bxz,
                                                  rho_u_rhs, rho_v_rhs, rho_w_rhs,
                                                  rho_u, rho_v, Omega, u, v, w,
                                                  cellSizeInv, mf_m,
                                                  mf_u_inv, mf_v_inv,
                                                  horiz_upw_frac, vert_upw_frac,
                                                  vert_adv_type, lo_z_face, hi_z_face);
            } else if (horiz_adv_type == AdvType::Upwind_3rd) {
                AdvectionSrcForMomVert_N<UPWIND3>(bxx, bxy, bxz,
                                                  rho_u_rhs, rho_v_rhs, rho_w_rhs,
                                                  rho_u, rho_v, Omega, u, v, w,
                                                  cellSizeInv, mf_m,
                                                  mf_u_inv, mf_v_inv,
                                                  horiz_upw_frac, vert_upw_frac,
                                                  vert_adv_type, lo_z_face, hi_z_face);
            } else if (horiz_adv_type == AdvType::Centered_4th) {
                AdvectionSrcForMomVert_N<CENTERED4>(bxx, bxy, bxz,
                                                  rho_u_rhs, rho_v_rhs, rho_w_rhs,
                                                  rho_u, rho_v, Omega, u, v, w,
                                                  cellSizeInv, mf_m,
                                                  mf_u_inv, mf_v_inv,
                                                  horiz_upw_frac, vert_upw_frac,
                                                  vert_adv_type, lo_z_face, hi_z_face);
            } else if (horiz_adv_type == AdvType::Upwind_5th) {
                AdvectionSrcForMomVert_N<UPWIND5>(bxx, bxy, bxz,
                                                  rho_u_rhs, rho_v_rhs, rho_w_rhs,
                                                  rho_u, rho_v, Omega, u, v, w,
                                                  cellSizeInv, mf_m,
                                                  mf_u_inv, mf_v_inv,
                                                  horiz_upw_frac, vert_upw_frac,
                                                  vert_adv_type, lo_z_face, hi_z_face);
            } else if (horiz_adv_type == AdvType::Centered_6th) {
                AdvectionSrcForMomVert_N<CENTERED6>(bxx, bxy, bxz,
                                                  rho_u_rhs, rho_v_rhs, rho_w_rhs,
                                                  rho_u, rho_v, Omega, u, v, w,
                                                  cellSizeInv, mf_m,
                                                  mf_u_inv, mf_v_inv,
                                                  horiz_upw_frac, vert_upw_frac,
                                                  vert_adv_type, lo_z_face, hi_z_face);
            } else {
                AMREX_ASSERT_WITH_MESSAGE(false, "Unknown advection scheme!");
            }
        }
    } // end of use_terrain == false
    else
    { // now do use_terrain = true
        // Inline with 2nd order for efficiency
        if (horiz_adv_type == AdvType::Centered_2nd && vert_adv_type == AdvType::Centered_2nd)
        {
            ParallelFor(bxx, bxy, bxz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real xflux_hi = 0.25 * (rho_u(i,j,k) * mf_u_inv(i,j,0) + rho_u(i+1,j,k) * mf_u_inv(i+1,j,0)) *
                                       (u(i+1,j,k) + u(i,j,k)) * 0.5 * (ax(i,j,k) + ax(i+1,j,k));

                Real xflux_lo = 0.25 * (rho_u(i,j,k) * mf_u_inv(i,j,0) + rho_u(i-1,j,k) * mf_u_inv(i-1,j,0)) *
                                       (u(i-1,j,k) + u(i,j,k)) * 0.5 * (ax(i,j,k) + ax(i-1,j,k));

                Real met_h_zeta_yhi = Compute_h_zeta_AtEdgeCenterK(i,j+1,k,cellSizeInv,z_nd);
                Real yflux_hi = 0.25 * (rho_v(i,j+1,k)*mf_v_inv(i,j+1,0) + rho_v(i-1,j+1,k)*mf_v_inv(i-1,j+1,0)) *
                                       (u(i,j+1,k) + u(i,j,k)) * met_h_zeta_yhi;

                Real met_h_zeta_ylo = Compute_h_zeta_AtEdgeCenterK(i,j  ,k,cellSizeInv,z_nd);
                Real yflux_lo = 0.25 * (rho_v(i,j  ,k)*mf_v_inv(i,j  ,0) + rho_v(i-1,j  ,k)*mf_v_inv(i-1,j  ,0)) *
                                       (u(i,j-1,k) + u(i,j,k)) * met_h_zeta_ylo;

                Real zflux_hi = 0.25 * (Omega(i,j,k+1) + Omega(i-1,j,k+1)) * (u(i,j,k+1) + u(i,j,k)) *
                                        0.5 * (az(i,j,k+1) + az(i-1,j,k+1));
                Real zflux_lo = 0.25 * (Omega(i,j,k  ) + Omega(i-1,j,k  )) * (u(i,j,k-1) + u(i,j,k)) *
                                        0.5 * (az(i,j,k  ) + az(i-1,j,k  ));

                Real mfsq = mf_u(i,j,0) * mf_u(i,j,0);

                Real advectionSrc = (xflux_hi - xflux_lo) * dxInv * mfsq
                                  + (yflux_hi - yflux_lo) * dyInv * mfsq
                                  + (zflux_hi - zflux_lo) * dzInv;

                rho_u_rhs(i, j, k) = -advectionSrc / (0.5 * (detJ(i,j,k) + detJ(i-1,j,k)));
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {

                Real met_h_zeta_xhi = Compute_h_zeta_AtEdgeCenterK(i+1,j,k,cellSizeInv,z_nd);
                Real xflux_hi = 0.25 * (rho_u(i+1,j,k)*mf_u_inv(i+1,j,0) + rho_u(i+1,j-1,k)*mf_u_inv(i+1,j-1,0)) *
                                       (v(i+1,j,k) + v(i,j,k)) * met_h_zeta_xhi;

                Real met_h_zeta_xlo = Compute_h_zeta_AtEdgeCenterK(i  ,j,k,cellSizeInv,z_nd);
                Real xflux_lo = 0.25 * (rho_u(i, j, k)*mf_u_inv(i  ,j,0) + rho_u(i  ,j-1,k)*mf_u_inv(i-1,j  ,0)) *
                                       (v(i-1,j,k) + v(i,j,k)) * met_h_zeta_xlo;

                Real yflux_hi = 0.25 * (rho_v(i,j+1,k)*mf_v_inv(i,j+1,0) + rho_v(i,j  ,k) * mf_v_inv(i,j  ,0)) *
                                       (v(i,j+1,k) + v(i,j,k)) * 0.5 * (ay(i,j,k) + ay(i,j+1,k));

                Real yflux_lo = 0.25 * (rho_v(i,j  ,k)*mf_v_inv(i,j  ,0) + rho_v(i,j-1,k) * mf_v_inv(i,j-1,0)) *
                                       (v(i,j-1,k) + v(i,j,k)) * 0.5 * (ay(i,j,k) + ay(i,j-1,k));

                Real zflux_hi = 0.25 * (Omega(i,j,k+1) + Omega(i, j-1, k+1)) * (v(i,j,k+1) + v(i,j,k)) *
                                        0.5 * (az(i,j,k+1) + az(i,j-1,k+1));
                Real zflux_lo = 0.25 * (Omega(i,j,k  ) + Omega(i, j-1, k  )) * (v(i,j,k-1) + v(i,j,k)) *
                                        0.5 * (az(i,j,k  ) + az(i,j-1,k  ));

                Real mfsq = mf_v(i,j,0) * mf_v(i,j,0);

                Real advectionSrc = (xflux_hi - xflux_lo) * dxInv * mfsq
                                  + (yflux_hi - yflux_lo) * dyInv * mfsq
                                  + (zflux_hi - zflux_lo) * dzInv;

                rho_v_rhs(i, j, k) = -advectionSrc / (0.5 * (detJ(i,j,k) + detJ(i,j-1,k)));
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real met_h_zeta_xhi = Compute_h_zeta_AtEdgeCenterJ(i+1,j  ,k  ,cellSizeInv,z_nd);
                Real xflux_hi = 0.25*(rho_u(i+1,j  ,k) + rho_u(i+1,j,k-1)) * mf_u_inv(i+1,j,0) *
                                     (w(i+1,j,k) + w(i,j,k)) * met_h_zeta_xhi;

                Real met_h_zeta_xlo = Compute_h_zeta_AtEdgeCenterJ(i  ,j  ,k  ,cellSizeInv,z_nd);
                Real xflux_lo = 0.25*(rho_u(i  ,j  ,k) + rho_u(i  ,j,k-1)) * mf_u_inv(i  ,j,0) *
                                     (w(i-1,j,k) + w(i,j,k)) * met_h_zeta_xlo;

                Real met_h_zeta_yhi = Compute_h_zeta_AtEdgeCenterI(i  ,j+1,k  ,cellSizeInv,z_nd);
                Real yflux_hi = 0.25*(rho_v(i,j+1,k) + rho_v(i,j+1,k-1)) * mf_v_inv(i,j+1,0) *
                                     (w(i,j+1,k) + w(i,j,k)) * met_h_zeta_yhi;

                Real met_h_zeta_ylo = Compute_h_zeta_AtEdgeCenterI(i  ,j  ,k  ,cellSizeInv,z_nd);
                Real yflux_lo = 0.25*(rho_v(i,j  ,k) + rho_v(i,j  ,k-1)) * mf_v_inv(i,j  ,0) *
                                     (w(i,j-1,k) + w(i,j,k)) * met_h_zeta_ylo;

                Real zflux_lo = 0.25 * (Omega(i,j,k) + Omega(i,j,k-1)) * (w(i,j,k) + w(i,j,k-1));

                Real zflux_hi = (k == hi_z_face) ? Omega(i,j,k) * w(i,j,k)  * az(i,j,k):
                    0.25 * (Omega(i,j,k) + Omega(i,j,k+1)) * (w(i,j,k) + w(i,j,k+1)) *
                    0.5  * (az(i,j,k) + az(i,j,k+1));

                Real mfsq = mf_m(i,j,0) * mf_m(i,j,0);

                Real advectionSrc = (xflux_hi - xflux_lo) * dxInv * mfsq
                                  + (yflux_hi - yflux_lo) * dyInv * mfsq
                                  + (zflux_hi - zflux_lo) * dzInv;

                rho_w_rhs(i, j, k) = -advectionSrc / (0.5*(detJ(i,j,k) + detJ(i,j,k-1)));
            });
        // Template higher order methods
        } else {
            if (horiz_adv_type == AdvType::Centered_2nd) {
                AdvectionSrcForMomVert<CENTERED2>(bxx, bxy, bxz,
                                                rho_u_rhs, rho_v_rhs, rho_w_rhs,
                                                rho_u, rho_v, Omega, u, v, w, z_nd, ax, ay, az, detJ,
                                                cellSizeInv, mf_m, mf_u_inv, mf_v_inv,
                                                horiz_upw_frac, vert_upw_frac,
                                                vert_adv_type, lo_z_face, hi_z_face);
            } else if (horiz_adv_type == AdvType::Upwind_3rd) {
                AdvectionSrcForMomVert<UPWIND3>(bxx, bxy, bxz,
                                                rho_u_rhs, rho_v_rhs, rho_w_rhs,
                                                rho_u, rho_v, Omega, u, v, w, z_nd, ax, ay, az, detJ,
                                                cellSizeInv, mf_m, mf_u_inv, mf_v_inv,
                                                horiz_upw_frac, vert_upw_frac,
                                                vert_adv_type, lo_z_face, hi_z_face);
            } else if (horiz_adv_type == AdvType::Centered_4th) {
                AdvectionSrcForMomVert<CENTERED4>(bxx, bxy, bxz,
                                                rho_u_rhs, rho_v_rhs, rho_w_rhs,
                                                rho_u, rho_v, Omega, u, v, w, z_nd, ax, ay, az, detJ,
                                                cellSizeInv, mf_m, mf_u_inv, mf_v_inv,
                                                horiz_upw_frac, vert_upw_frac,
                                                vert_adv_type, lo_z_face, hi_z_face);
            } else if (horiz_adv_type == AdvType::Upwind_5th) {
                AdvectionSrcForMomVert<UPWIND5>(bxx, bxy, bxz,
                                                rho_u_rhs, rho_v_rhs, rho_w_rhs,
                                                rho_u, rho_v, Omega, u, v, w, z_nd, ax, ay, az, detJ,
                                                cellSizeInv, mf_m, mf_u_inv, mf_v_inv,
                                                horiz_upw_frac, vert_upw_frac,
                                                vert_adv_type, lo_z_face, hi_z_face);
            } else if (horiz_adv_type == AdvType::Centered_6th) {
                AdvectionSrcForMomVert<CENTERED6>(bxx, bxy, bxz,
                                                rho_u_rhs, rho_v_rhs, rho_w_rhs,
                                                rho_u, rho_v, Omega, u, v, w, z_nd, ax, ay, az, detJ,
                                                cellSizeInv, mf_m, mf_u_inv, mf_v_inv,
                                                horiz_upw_frac, vert_upw_frac,
                                                vert_adv_type, lo_z_face, hi_z_face);
            } else {
                    AMREX_ASSERT_WITH_MESSAGE(false, "Unknown advection scheme!");
            }
        } // higher order
    } // terrain

    // Open bc will be imposed upon all vars (we only access cons here for simplicity)
    const bool xlo_open = (bc_ptr_h[BCVars::cons_bc].lo(0) == ERFBCType::open);
    const bool xhi_open = (bc_ptr_h[BCVars::cons_bc].hi(0) == ERFBCType::open);
    const bool ylo_open = (bc_ptr_h[BCVars::cons_bc].lo(1) == ERFBCType::open);
    const bool yhi_open = (bc_ptr_h[BCVars::cons_bc].hi(1) == ERFBCType::open);

    // We recreate tbx, tbz, tbz here rather than using bxx, bxy, bxz because those
    //    have already been shrunk by one in the case of open BCs.
    Box tbx(surroundingNodes(bx,0));
    Box tby(surroundingNodes(bx,1));
    Box tbz(surroundingNodes(bx,2)); tbz.growLo(2,-1); tbz.growHi(2,-1);

    const int domhi_z = domain.bigEnd(2);

    // Special advection operator for open BC (bndry normal/tangent operations)
    if (xlo_open)
    {
        Box tbx_xlo, tby_xlo, tbz_xlo;
        if (tbx.smallEnd(0) == domain.smallEnd(0)) { tbx_xlo = makeSlab(tbx,0,domain.smallEnd(0));}
        if (tby.smallEnd(0) == domain.smallEnd(0)) { tby_xlo = makeSlab(tby,0,domain.smallEnd(0));}
        if (tbz.smallEnd(0) == domain.smallEnd(0)) { tbz_xlo = makeSlab(tbz,0,domain.smallEnd(0));}

        bool do_lo = true;

        AdvectionSrcForOpenBC_Normal(tbx_xlo, 0, rho_u_rhs, u, cell_data, cellSizeInv, do_lo);
        AdvectionSrcForOpenBC_Tangent_Ymom(tby_xlo, 0, rho_v_rhs, v,
                                           rho_u, rho_v, Omega,
                                           ay, az, detJ, cellSizeInv,
                                           do_lo);
        AdvectionSrcForOpenBC_Tangent_Zmom(tbz_xlo, 0, rho_w_rhs, w,
                                           rho_u, rho_v, Omega,
                                           ax, ay, az, detJ, cellSizeInv,
                                           domhi_z, do_lo);
    }
    if (xhi_open)
    {
        Box tbx_xhi, tby_xhi, tbz_xhi;
        if (tbx.bigEnd(0) == domain.bigEnd(0)+1)   { tbx_xhi = makeSlab(tbx,0,domain.bigEnd(0)+1);}
        if (tby.bigEnd(0) == domain.bigEnd(0))     { tby_xhi = makeSlab(tby,0,domain.bigEnd(0)  );}
        if (tbz.bigEnd(0) == domain.bigEnd(0))     { tbz_xhi = makeSlab(tbz,0,domain.bigEnd(0)  );}

        AdvectionSrcForOpenBC_Normal(tbx_xhi, 0, rho_u_rhs, u, cell_data, cellSizeInv);
        AdvectionSrcForOpenBC_Tangent_Ymom(tby_xhi, 0, rho_v_rhs, v,
                                           rho_u, rho_v, Omega,
                                           ay, az, detJ, cellSizeInv);
        AdvectionSrcForOpenBC_Tangent_Zmom(tbz_xhi, 0, rho_w_rhs, w,
                                           rho_u, rho_v, Omega,
                                           ax, ay, az, detJ, cellSizeInv,
                                           domhi_z);
    }
    if (ylo_open)
    {
        Box tbx_ylo, tby_ylo, tbz_ylo;
        if (tbx.smallEnd(1) == domain.smallEnd(1)) { tbx_ylo = makeSlab(tbx,1,domain.smallEnd(1));}
        if (tby.smallEnd(1) == domain.smallEnd(1)) { tby_ylo = makeSlab(tby,1,domain.smallEnd(1));}
        if (tbz.smallEnd(1) == domain.smallEnd(1)) { tbz_ylo = makeSlab(tbz,1,domain.smallEnd(1));}

        bool do_lo = true;
        AdvectionSrcForOpenBC_Tangent_Xmom(tbx_ylo, 1, rho_u_rhs, u,
                                           rho_u, rho_v, Omega,
                                           ax, az, detJ, cellSizeInv,
                                           do_lo);
        AdvectionSrcForOpenBC_Normal(tby_ylo, 1, rho_v_rhs, v, cell_data, cellSizeInv, do_lo);
        AdvectionSrcForOpenBC_Tangent_Zmom(tbz_ylo, 1, rho_w_rhs, w,
                                           rho_u, rho_v, Omega,
                                           ax, ay, az, detJ, cellSizeInv,
                                           domhi_z, do_lo);
    }
    if (yhi_open)
    {
        Box tbx_yhi, tby_yhi, tbz_yhi;
        if (tbx.bigEnd(1) == domain.bigEnd(1))     { tbx_yhi = makeSlab(tbx,1,domain.bigEnd(1)  );}
        if (tby.bigEnd(1) == domain.bigEnd(1)+1)   { tby_yhi = makeSlab(tby,1,domain.bigEnd(1)+1);}
        if (tbz.bigEnd(1) == domain.bigEnd(1))     { tbz_yhi = makeSlab(tbz,1,domain.bigEnd(1)  );}

        AdvectionSrcForOpenBC_Tangent_Xmom(tbx_yhi, 1, rho_u_rhs, u,
                                           rho_u, rho_v, Omega,
                                           ax, az, detJ, cellSizeInv);
        AdvectionSrcForOpenBC_Normal(tby_yhi, 1, rho_v_rhs, v, cell_data, cellSizeInv);
        AdvectionSrcForOpenBC_Tangent_Zmom(tbz_yhi, 1, rho_w_rhs, w,
                                           rho_u, rho_v, Omega,
                                           ax, ay, az, detJ, cellSizeInv,
                                           domhi_z);
    }
#endif
}

