#ifndef ERF_METGRIDUTIL_H_
#define ERF_METGRIDUTIL_H_

#include <ERF.H>
#include <ERF_EOS.H>
#include <ERF_Utils.H>
#include <ERF_ProbCommon.H>
#include <ERF_HSEUtils.H>

void
read_from_metgrid (int lev,
                   const amrex::Box& domain,
                   const std::string& fname,
                   std::string& NC_dateTime,
                   amrex::Real& NC_epochTime,
                   int& flag_psfc,
                   int& flag_msf,
                   int& flag_sst,
                   int& flag_lmask,
                   int& NC_nx,
                   int& NC_ny,
                   amrex::Real& NC_dx,
                   amrex::Real& NC_dy,
                   amrex::FArrayBox& NC_xvel_fab,
                   amrex::FArrayBox& NC_yvel_fab,
                   amrex::FArrayBox& NC_temp_fab,
                   amrex::FArrayBox& NC_rhum_fab,
                   amrex::FArrayBox& NC_pres_fab,
                   amrex::FArrayBox& NC_ght_fab,
                   amrex::FArrayBox& NC_hgt_fab,
                   amrex::FArrayBox& NC_psfc_fab,
                   amrex::FArrayBox& NC_msfu_fab,
                   amrex::FArrayBox& NC_msfv_fab,
                   amrex::FArrayBox& NC_msfm_fab,
                   amrex::FArrayBox& NC_sst_fab,
                   amrex::FArrayBox& NC_LAT_fab,
                   amrex::FArrayBox& NC_LON_fab,
                   amrex::IArrayBox& NC_lmask_iab,
                   amrex::Real& Latitude,
                   amrex::Real& Longitude,
                   amrex::Geometry& geom);



void
init_terrain_from_metgrid (amrex::FArrayBox& z_phys_nd_fab,
                           const amrex::Vector<amrex::FArrayBox>& NC_hgt_fab);

void
init_state_from_metgrid (const bool use_moisture,
                         const bool interp_theta,
                         const bool metgrid_debug_quiescent,
                         const bool metgrid_debug_isothermal,
                         const bool metgrid_debug_dry,
                         const bool metgrid_basic_linear,
                         const bool metgrid_use_below_sfc,
                         const bool metgrid_use_sfc,
                         const bool metgrid_retain_sfc,
                         const amrex::Real metgrid_proximity,
                         const int metgrid_order,
                         const int metgrid_metgrid_force_sfc_k,
                         const amrex::Real l_rdOcp,
                         amrex::Box& tbxc,
                         amrex::Box& tbxu,
                         amrex::Box& tbxv,
                         amrex::FArrayBox& state_fab,
                         amrex::FArrayBox& x_vel_fab,
                         amrex::FArrayBox& y_vel_fab,
                         amrex::FArrayBox& z_vel_fab,
                         amrex::FArrayBox& z_phys_nd_fab,
                         const amrex::Vector<amrex::FArrayBox>& NC_hgt_fab,
                         const amrex::Vector<amrex::FArrayBox>& NC_ght_fab,
                         const amrex::Vector<amrex::FArrayBox>& NC_xvel_fab,
                         const amrex::Vector<amrex::FArrayBox>& NC_yvel_fab,
                         const amrex::Vector<amrex::FArrayBox>& NC_temp_fab,
                         const amrex::Vector<amrex::FArrayBox>& NC_rhum_fab,
                         const amrex::Vector<amrex::FArrayBox>& NC_pres_fab,
                         amrex::FArrayBox& p_interp_fab,
                         amrex::FArrayBox& t_interp_fab,
                         amrex::FArrayBox& theta_fab,
                         amrex::FArrayBox& mxrat_fab,
                         amrex::Vector<amrex::Vector<amrex::FArrayBox>>& fabs_for_bcs_xlo,
                         amrex::Vector<amrex::Vector<amrex::FArrayBox>>& fabs_for_bcs_xhi,
                         amrex::Vector<amrex::Vector<amrex::FArrayBox>>& fabs_for_bcs_ylo,
                         amrex::Vector<amrex::Vector<amrex::FArrayBox>>& fabs_for_bcs_yhi,
                         const amrex::Array4<const int>& mask_c_arr,
                         const amrex::Array4<const int>& mask_u_arr,
                         const amrex::Array4<const int>& mask_v_arr);

void
init_msfs_from_metgrid (const bool metgrid_debug_msf,
                        amrex::FArrayBox& msfu_fab,
                        amrex::FArrayBox& msfv_fab,
                        amrex::FArrayBox& msfm_fab,
                        const int& flag_msf,
                        const amrex::Vector<amrex::FArrayBox>& NC_MSFU_fab,
                        const amrex::Vector<amrex::FArrayBox>& NC_MSFV_fab,
                        const amrex::Vector<amrex::FArrayBox>& NC_MSFM_fab);

void
init_base_state_from_metgrid (const bool use_moisture,
                              const bool metgrid_debug_psfc,
                              const amrex::Real l_rdOcp,
                              const amrex::Box& valid_bx,
                              const amrex::Vector<int>& flag_psfc,
                              amrex::FArrayBox& state,
                              amrex::FArrayBox& r_hse_fab,
                              amrex::FArrayBox& p_hse_fab,
                              amrex::FArrayBox& pi_hse_fab,
                              amrex::FArrayBox& th_hse_fab,
                              amrex::FArrayBox& z_phys_cc_fab,
                              const amrex::Vector<amrex::FArrayBox>& NC_psfc_fab);

AMREX_FORCE_INLINE
AMREX_GPU_DEVICE
void
lagrange_interp (const int& order,
                 amrex::Real* x,
                 amrex::Real* y,
                 amrex::Real& new_x,
                 amrex::Real& new_y)
{
    // Interpolation using Lagrange polynomials.
    // P(x) = f(x0)Ln0(x) + ... + f(xn)Lnn(x)
    // where Lnk(x) = (x -x0)(x -x1)...(x -xk-1)(x -xk+1)...(x -xn)
    //                ---------------------------------------------
    //                (xk-x0)(xk-x1)...(xk-xk-1)(xk-xk+1)...(xk-xn)
    amrex::Real Px = 0.0;
    for (int i=0; i <= order; i++) {
        amrex::Real n = 1.0;
        amrex::Real d = 1.0;
        for (int k=0; k <= order; k++) {
            if (k == i) continue;
            n *= new_x-x[k];
            d *= x[i]-x[k];
        }
        if (d != 0.0) {
            Px += y[i]*n/d;
        }
    }
    new_y = Px;
}

AMREX_FORCE_INLINE
AMREX_GPU_DEVICE
void
lagrange_setup (char var_type,
                const bool& exp_interp,
                const int& orig_n,
                const int& new_n,
                const int& order,
                amrex::Real* orig_x_z,
                amrex::Real* orig_x_p,
                amrex::Real* orig_y,
                amrex::Real* new_x_z,
                amrex::Real* new_x_p,
                amrex::Real* new_y)
{
    if (order < 1) amrex::Abort("metgrid initialization, we cannot go lower order than linear");

    amrex::Real CRC_const1 = 11880.516; // m
    amrex::Real CRC_const2 = 0.1902632;
    amrex::Real CRC_const3 = 0.0065; // K km-1
    int vboundb = 4;
    int vboundt = 0;

#ifndef AMREX_USE_GPU
    bool debug = false;
#endif

    for (int new_k=0; new_k < new_n; new_k++) {
#ifndef AMREX_USE_GPU
        if (debug) amrex::Print() << "new_k=" << new_k;
#endif
        // Find bounding x values and store the indices.
        bool extrapolating = true;
        int kl, kr;
        for (int ko=0; ko < orig_n-1; ko++) {
            amrex::Real a = new_x_z[new_k]-orig_x_z[ko];
            amrex::Real b = new_x_z[new_k]-orig_x_z[ko+1];
            if (a*b <= 0.0) {
                kl = ko;
                kr = ko+1;
                extrapolating = false;
                break;
            }
        }

        if (extrapolating) {
            if (var_type == 'T') {
                // Assume a standard atmosphere -6.5 K km-1 lapse rate.
                // Comparable to the WRF default, t_extrap_type=2.
                amrex::Real depth_of_extrap_in_p = new_x_p[new_k]-orig_x_p[1];
                amrex::Real avg_of_extrap_p = 0.5*(new_x_p[new_k]+orig_x_p[1]);
                amrex::Real temp_extrap_starting_point = orig_y[1]*std::pow(orig_x_p[1]/100000.0, R_d/Cp_d);
                amrex::Real dZdP = CRC_const1*CRC_const2*std::pow(avg_of_extrap_p/100.0, CRC_const2-1.0);
                amrex::Real dZ = dZdP*(depth_of_extrap_in_p/100.0);
                amrex::Real dT = dZ*CRC_const3;
                new_y[new_k] = (temp_extrap_starting_point+dT)*std::pow(100000.0/new_x_p[new_k], R_d/Cp_d);
            } else {
                // Use a constant value below ground.
                // Comparable to the WRF default, extrap_type=2.
                new_y[new_k] = orig_y[1];
            }
            continue;
        }

        if (order%2 != 0) {
            if ((kl-((order+1)/2-1) >= 0) && (kr+((order+1)/2-1) <= orig_n-1)) {
                int ksta = kl-((order/2)-1);
                int kend = ksta+order;
                amrex::Real new_x;
                amrex::GpuArray<amrex::Real,2> orig_x_sub;
                amrex::Real* orig_x_sub_p = orig_x_sub.data();
                if (exp_interp) {
                    new_x = new_x_p[new_k];
                    orig_x_sub_p[0] = orig_x_p[ksta];
                    orig_x_sub_p[1] = orig_x_p[kend];
                } else {
                    new_x = new_x_z[new_k];
                    orig_x_sub_p[0] = orig_x_z[ksta];
                    orig_x_sub_p[1] = orig_x_z[kend];
                }
                amrex::GpuArray<amrex::Real,2> orig_y_sub;
                amrex::Real* orig_y_sub_p = orig_y_sub.data();
                orig_y_sub_p[0] = orig_y[ksta];
                orig_y_sub_p[1] = orig_y[kend];
                lagrange_interp(order, orig_x_sub_p, orig_y_sub_p, new_x, new_y[new_k]);
            } else {
                amrex::Abort("metgrid initialization, lost in lagrange_setup (odd order)");
            }
        } else if ((order%2 == 0) && (new_k >= 1+vboundb) && (new_k < new_n-vboundt)) {
            if ((kl-(order/2) >= 0) && (kr+order/2 <= orig_n-1)) {
                amrex::Real new_y_l, new_y_r;
                {
                    int ksta = kl-(order/2-1);
                    int kend = ksta+order;
#ifndef AMREX_USE_GPU
                    int ksize = kend-ksta;
                    if (debug) amrex::Print() << "   (1a)   new_x=" << new_x_z[new_k] << "   new_x_p=" << new_x_p[new_k] << "   kl=" << kl << "   kr=" << kr <<"   ksta=" << ksta << "   kend=" << kend << std::endl;
#endif
                    amrex::Real new_x;
                    amrex::GpuArray<amrex::Real,256> orig_x_sub;
                    amrex::GpuArray<amrex::Real,256> orig_y_sub;
                    amrex::Real* orig_x_sub_p = orig_x_sub.data();
                    amrex::Real* orig_y_sub_p = orig_y_sub.data();
                    if (exp_interp) {
                        new_x = new_x_p[new_k];
                        for (int k=ksta; k <= kend; k++) { orig_x_sub_p[k-ksta] = orig_x_p[k]; }
                    } else {
                        new_x = new_x_z[new_k];
                        for (int k=ksta; k <= kend; k++) { orig_x_sub_p[k-ksta] = orig_x_z[k]; }
                    }
                    for (int k=ksta; k <= kend; k++) { orig_y_sub_p[k-ksta] = orig_y[k]; }
#ifndef AMREX_USE_GPU
                    if (debug) {
                        amrex::Print() << "    orig_x_sub = [";
                        for (int k=0; k < ksize; k++) amrex::Print() << " " << orig_x_sub_p[k];
                        amrex::Print() << "]" << std::endl;
                        amrex::Print() << "    orig_y_sub = [";
                        for (int k=0; k < ksize; k++) amrex::Print() << " " << orig_y_sub_p[k];
                        amrex::Print() << "]" << std::endl;
                    }
#endif
                    lagrange_interp(order, orig_x_sub_p, orig_y_sub_p, new_x, new_y_l);
                }
                {
                    int ksta = kl-order/2;
                    int kend = ksta+order;
#ifndef AMREX_USE_GPU
                    int ksize = kend-ksta;
                    if (debug) amrex::Print() << "new_k=" << new_k << "   (1b)   new_x=" << new_x_z[new_k] << "   new_x_p=" << new_x_p[new_k] << "   kl=" << kl << "   kr=" << kr <<"   ksta=" << ksta << "   kend=" << kend << std::endl;
#endif
                    amrex::Real new_x;
                    amrex::GpuArray<amrex::Real,256> orig_x_sub;
                    amrex::GpuArray<amrex::Real,256> orig_y_sub;
                    amrex::Real* orig_x_sub_p = orig_x_sub.data();
                    amrex::Real* orig_y_sub_p = orig_y_sub.data();
                    if (exp_interp) {
                        new_x = new_x_p[new_k];
                        for (int k=ksta; k <= kend; k++) { orig_x_sub_p[k-ksta] = orig_x_p[k]; }
                    } else {
                        new_x = new_x_z[new_k];
                        for (int k=ksta; k <= kend; k++) { orig_x_sub_p[k-ksta] = orig_x_z[k]; }
                    }
                    for (int k=ksta; k <= kend; k++) { orig_y_sub_p[k-ksta] = orig_y[k]; }
#ifndef AMREX_USE_GPU
                    if (debug) {
                        amrex::Print() << "    orig_x_sub = [";
                        for (int k=0; k < ksize; k++) amrex::Print() << " " << orig_x_sub_p[k];
                        amrex::Print() << "]" << std::endl;
                        amrex::Print() << "    orig_y_sub = [";
                        for (int k=0; k < ksize; k++) amrex::Print() << " " << orig_y_sub_p[k];
                        amrex::Print() << "]" << std::endl;
                    }
#endif
                    lagrange_interp(order, orig_x_sub_p, orig_y_sub_p, new_x, new_y_r);
                }
                new_y[new_k] = 0.5*(new_y_l+new_y_r);
#ifndef AMREX_USE_GPU
                if (debug) amrex::Print() << "    new_y=" << new_y[new_k] << std::endl;
#endif
            } else if ((kl-(order/2-1) >= 0) && (kr+order/2 <= orig_n-1)) {
                int ksta = kl-(order/2-1);
                int kend = ksta+order;
#ifndef AMREX_USE_GPU
                int ksize = kend-ksta;
                if (debug) amrex::Print() << "   (2)   new_x=" << new_x_z[new_k] << "   new_x_p=" << new_x_p[new_k] << "   kl=" << kl << "   kr=" << kr <<"   ksta=" << ksta << "   kend=" << kend << std::endl;
#endif
                amrex::Real new_x;
                amrex::GpuArray<amrex::Real,256> orig_x_sub;
                amrex::GpuArray<amrex::Real,256> orig_y_sub;
                amrex::Real* orig_x_sub_p = orig_x_sub.data();
                amrex::Real* orig_y_sub_p = orig_y_sub.data();
                if (exp_interp) {
                    new_x = new_x_p[new_k];
                    for (int k=ksta; k <= kend; k++) { orig_x_sub_p[k-ksta] = orig_x_p[k]; }
                } else {
                    new_x = new_x_z[new_k];
                    for (int k=ksta; k <= kend; k++) { orig_x_sub_p[k-ksta] = orig_x_z[k]; }
                }
                for (int k=ksta; k <= kend; k++) { orig_y_sub_p[k-ksta] = orig_y[k]; }
#ifndef AMREX_USE_GPU
                if (debug) {
                    amrex::Print() << "    orig_x_sub = [";
                    for (int k=0; k < ksize; k++) amrex::Print() << " " << orig_x_sub_p[k];
                    amrex::Print() << "]" << std::endl;
                    amrex::Print() << "    orig_y_sub = [";
                    for (int k=0; k < ksize; k++) amrex::Print() << " " << orig_y_sub_p[k];
                    amrex::Print() << "]" << std::endl;
                }
#endif
                lagrange_interp(order, orig_x_sub_p, orig_y_sub_p, new_x, new_y[new_k]);
#ifndef AMREX_USE_GPU
                if (debug) amrex::Print() << "    new_y=" << new_y[new_k] << std::endl;
#endif
            } else if ((kl-order/2 >= 0) && (kr+(order/2-1) <= orig_n-1)) {
                int ksta = kl-order/2;
                int kend = ksta+order;
#ifndef AMREX_USE_GPU
                int ksize = kend-ksta;
                if (debug) amrex::Print() << "   (3)   new_x=" << new_x_z[new_k] << "   new_x_p=" << new_x_p[new_k] << "   kl=" << kl << "   kr=" << kr <<"   ksta=" << ksta << "   kend=" << kend << std::endl;
#endif
                amrex::Real new_x;
                amrex::GpuArray<amrex::Real,256> orig_x_sub;
                amrex::GpuArray<amrex::Real,256> orig_y_sub;
                amrex::Real* orig_x_sub_p = orig_x_sub.data();
                amrex::Real* orig_y_sub_p = orig_y_sub.data();
                if (exp_interp) {
                    new_x = new_x_p[new_k];
                    for (int k=ksta; k <= kend; k++) { orig_x_sub_p[k-ksta] = orig_x_p[k]; }
                } else {
                    new_x = new_x_z[new_k];
                    for (int k=ksta; k <= kend; k++) { orig_x_sub_p[k-ksta] = orig_x_z[k]; }
                }
                for (int k=ksta; k <= kend; k++) { orig_y_sub_p[k-ksta] = orig_y[k]; }
#ifndef AMREX_USE_GPU
                if (debug) {
                    amrex::Print() << "    orig_x_sub = [";
                    for (int k=0; k < ksize; k++) amrex::Print() << " " << orig_x_sub_p[k];
                    amrex::Print() << "]" << std::endl;
                    amrex::Print() << "    orig_y_sub = [";
                    for (int k=0; k < ksize; k++) amrex::Print() << " " << orig_y_sub_p[k];
                    amrex::Print() << "]" << std::endl;
                }
#endif
                lagrange_interp(order, orig_x_sub_p, orig_y_sub_p, new_x, new_y[new_k]);
#ifndef AMREX_USE_GPU
                if (debug) amrex::Print() << "    new_y=" << new_y[new_k] << std::endl;
#endif
            } else {
                amrex::Abort("metgrid initialization, lost in lagrange_setup (even order)");
            }
        } else {
            // Linear interpolation.
            int ksta = kl;
            int kend = kr;
#ifndef AMREX_USE_GPU
            int ksize = kend-ksta;
            if (debug) amrex::Print() << "   (4)   new_x=" << new_x_z[new_k] << "   new_x_p=" << new_x_p[new_k] << "   kl=" << kl << "   kr=" << kr <<"   ksta=" << ksta << "   kend=" << kend << std::endl;
#endif
            amrex::Real new_x;
            amrex::GpuArray<amrex::Real,256> orig_x_sub;
            amrex::GpuArray<amrex::Real,256> orig_y_sub;
            amrex::Real* orig_x_sub_p = orig_x_sub.data();
            amrex::Real* orig_y_sub_p = orig_y_sub.data();
            if (exp_interp) {
                new_x = new_x_p[new_k];
                for (int k=ksta; k <= kend; k++) { orig_x_sub_p[k-ksta] = orig_x_p[k]; }
            } else {
                new_x = new_x_z[new_k];
                for (int k=ksta; k <= kend; k++) { orig_x_sub_p[k-ksta] = orig_x_z[k]; }
            }
            for (int k=ksta; k <= kend; k++) { orig_y_sub_p[k-ksta] = orig_y[k]; }
#ifndef AMREX_USE_GPU
            if (debug) {
                amrex::Print() << "    orig_x_sub = [";
                for (int k=0; k < ksize; k++) amrex::Print() << " " << orig_x_sub_p[k];
                amrex::Print() << "]" << std::endl;
                amrex::Print() << "    orig_y_sub = [";
                for (int k=0; k < ksize; k++) amrex::Print() << " " << orig_y_sub_p[k];
                amrex::Print() << "]" << std::endl;
            }
#endif
            lagrange_interp(1, orig_x_sub_p, orig_y_sub_p, new_x, new_y[new_k]);
#ifndef AMREX_USE_GPU
            if (debug) amrex::Print() << "    new_y=" << new_y[new_k] << std::endl;
#endif
        }
    }
}

AMREX_FORCE_INLINE
AMREX_GPU_DEVICE
void
calc_p_isothermal (const amrex::Real& z,
                   amrex::Real& p)
{
    p = p_0*exp(-CONST_GRAV*z/(290.0*R_d));
}

AMREX_FORCE_INLINE
AMREX_GPU_DEVICE
void
interpolate_column_metgrid (const bool& metgrid_use_below_sfc,
                            const bool& metgrid_use_sfc,
                            const bool& exp_interp,
                            const bool& metgrid_retain_sfc,
                            const amrex::Real& metgrid_proximity,
                            const int& metgrid_order,
                            const int& metgrid_force_sfc_k,
                            const int& i,
                            const int& j,
                            const int& src_comp,
                            const int& itime,
                            char var_type,
                            char stag,
                            const amrex::Array4<amrex::Real const>& orig_z_full,
                            const amrex::Array4<amrex::Real const>& orig_data,
                            const amrex::Array4<amrex::Real const>& new_z_full,
                            const amrex::Array4<amrex::Real>& new_data_full,
                            const bool& update_bc_data,
                            const amrex::Array4<amrex::Real>& bc_data_xlo,
                            const amrex::Array4<amrex::Real>& bc_data_xhi,
                            const amrex::Array4<amrex::Real>& bc_data_ylo,
                            const amrex::Array4<amrex::Real>& bc_data_yhi,
                            const amrex::Box& bx_xlo,
                            const amrex::Box& bx_xhi,
                            const amrex::Box& bx_ylo,
                            const amrex::Box& bx_yhi,
                            const amrex::Array4<const int>& mask)
{
    // Here we closely follow WRF's vert_interp from
    // dyn_em/module_initialize_real.F, although changes have been
    // made to accommodate interpolation relative to height instead of
    // pressure.
    int imax_orig = amrex::ubound(amrex::Box(orig_data)).x;
    int jmax_orig = amrex::ubound(amrex::Box(orig_data)).y;
    int kmax_orig = amrex::ubound(amrex::Box(orig_data)).z;
    int kmax_new  = amrex::ubound(amrex::Box(new_z_full)).z;

    AMREX_ASSERT(kmax_orig < 256);
    AMREX_ASSERT(kmax_new  < 256);

    amrex::GpuArray<amrex::Real,256> new_z;
    amrex::GpuArray<amrex::Real,256> new_p;
    amrex::GpuArray<amrex::Real,256> new_data;
    amrex::Real* new_z_p = new_z.data();
    amrex::Real* new_p_p = new_p.data();
    amrex::Real* new_data_p = new_data.data();
    for (int k=0; k < kmax_new; k++) {
        if (stag == 'X') {
            new_z_p[k] = 0.25*(new_z_full(i,j,k)+new_z_full(i,j+1,k)+new_z_full(i,j,k+1)+new_z_full(i,j+1,k+1));
        } else if (stag == 'Y') {
            new_z_p[k] = 0.25*(new_z_full(i,j,k)+new_z_full(i+1,j,k)+new_z_full(i,j,k+1)+new_z_full(i+1,j,k+1));
        } else if (stag == 'M') {
            new_z_p[k] = 0.125*(new_z_full(i,j,k  )+new_z_full(i,j+1,k  )+new_z_full(i+1,j,k  )+new_z_full(i+1,j+1,k  )+
                              new_z_full(i,j,k+1)+new_z_full(i,j+1,k+1)+new_z_full(i+1,j,k+1)+new_z_full(i+1,j+1,k+1));
        }
        calc_p_isothermal(new_z_p[k], new_p_p[k]);
    }

    amrex::GpuArray<amrex::Real,256> orig_z;
    amrex::Real* orig_z_p = orig_z.data();
    for (int k=0; k < kmax_orig; k++) {
        if (stag == 'M') {
            orig_z_p[k] = orig_z_full(i,j,k);
        } else if (stag == 'X') {
            if (i == 0) {
                orig_z_p[k] = orig_z_full(i,j,k);
            } else if (i == imax_orig) {
                orig_z_p[k] = orig_z_full(imax_orig-1,j,k);
            } else {
                orig_z_p[k] = 0.5*(orig_z_full(i,j,k)+orig_z_full(i-1,j,k));
            }
        } else if (stag == 'Y') {
            if (j == 0) {
                orig_z_p[k] = orig_z_full(i,j,k);
            } else if (j == jmax_orig) {
                orig_z_p[k] = orig_z_full(i,jmax_orig-1,k);
            } else {
                orig_z_p[k] = 0.5*(orig_z_full(i,j,k)+orig_z_full(i,j-1,k));
            }
        }
    }

    // Check if the data is top-down instead of bottom-up.
    bool flip_data_required = false;
    if (orig_z[1] > orig_z[kmax_orig-1]) flip_data_required = true;
    if (flip_data_required) amrex::Abort("metgrid initialization flip_data_required. Not yet implemented.");

    // Search for the first level above the surface in the origin data.
    // This is needed since the origin model topography will be
    // different than the topography processed by WPS.
    int k_above_sfc = 0;
    for (int k=1; k < kmax_orig; k++) {
        if (orig_z_p[k] > orig_z_p[0]) {
            k_above_sfc = k;
            break;
        }
    }

    int zap = 0;
    int zap_below = 0;
    int zap_above = 0;
    int kend_order;
    amrex::ignore_unused(zap,zap_above);
    amrex::GpuArray<amrex::Real,256> ordered_z;
    amrex::GpuArray<amrex::Real,256> ordered_data;
    amrex::Real* ordered_z_p = ordered_z.data();
    amrex::Real* ordered_data_p = ordered_data.data();
    if (k_above_sfc > 1) {
        // The levels are not monotonically increasing in height, so
        // we sort and then make "artistic" quality control choices.
        int count = 0;

        // Insert levels that are below the surface.
        for (int k=1; k < k_above_sfc; k++) {
            ordered_z_p[count] = orig_z_p[k];
            ordered_data_p[count] = orig_data(i,j,k);
            count++;
        }

        // Check if the level that is nearest to and below the surface
        // is "too close". If so, we'll ignore the upper level and keep
        // the lower. Origin data is likely to be on pressure levels
        // with higher spatial resolution near-surface, which supports
        // the choice of eliminating levels that are "too close" in
        // pressure-space. For simplicity, calculate delta P assuming a
        // baroclinic atmosphere.
        amrex::Real Pu, Pl;
        calc_p_isothermal(orig_z_p[0], Pu);
        calc_p_isothermal(ordered_z_p[count-1], Pl);
        if (Pl-Pu < metgrid_proximity) {
            count--;
            zap = 1;
            zap_below = 1;
        }

        // Insert the surface level.
        ordered_z_p[count] = orig_z_p[0];
        ordered_data_p[count] = orig_data(i,j,0);
        count++;

        // Quoting WRF's comments, the next level to use is at,
        // "... ta da, the first level above the surface. I know, wow."
        int knext = k_above_sfc;
        // Conditionally more strongly use the surface data by removing
        // levels between the surface and the height corresponding to a
        // set number of ERF levels from the surface. This forces the
        // interpolator to use the surface data up through a number of
        // ERF levels from the surface.
        if (metgrid_force_sfc_k > 0) {
            for (int k=k_above_sfc; k < kmax_orig; k++) {
                if (orig_z_p[k] > new_z_p[metgrid_force_sfc_k-1]) {
                    knext = k;
                    break;
                } else {
                    zap++;
                    zap_above++;
                }
            }
        }

        // Check if the level that is nearest to and above the surface
        // is "too close". If so, we'll ignore that level.
        calc_p_isothermal(orig_z_p[knext], Pu);
        calc_p_isothermal(ordered_z_p[count-1], Pl);
        if (Pl-Pu < metgrid_proximity) {
            knext++;
            zap++;
            zap_above++;
        }

        // Insert levels that are above the surface.
        for (int k=knext; k < kmax_orig; k++) {
            ordered_z_p[count] = orig_z_p[k];
            ordered_data_p[count] = orig_data(i,j,k);
            count++;
        }

        kend_order = count;
    } else {
        // The surface is the lowest level in the origin data.

        // Insert the surface.
        ordered_z_p[0] = orig_z[0];
        ordered_data_p[0] = orig_data(i,j,0);

        // Similar to above, conditionally more strongly use the
        // surface data.
        int count = 1;
        int knext = count;
        if (metgrid_force_sfc_k > 0) {
            for (int k=knext; k < kmax_orig; k++) {
                if (orig_z_p[k] > new_z_p[metgrid_force_sfc_k]) {
                    knext = k;
                    break;
                } else {
                    zap++;
                    zap_above++;
                }
            }
        }

        // Insert the remaining levels, again ignoring levels that are
        // "too close" to the prior valid level.
        for (int k=knext; k < kmax_orig; k++) {
            amrex::Real Pu, Pl;
            calc_p_isothermal(orig_z_p[k], Pu);
            calc_p_isothermal(ordered_z_p[count-1], Pl);
            if (Pl-Pu < metgrid_proximity) {
                zap++;
                zap_above++;
                continue;
            }
            ordered_z_p[count] = orig_z_p[k];
            ordered_data_p[count] = orig_data(i,j,k);
            count++;
        }
        kend_order = count;
    }

    int ksta(0), kend(0);
    if (metgrid_use_below_sfc && metgrid_use_sfc) {
        // Use all levels.
        ksta = 0;
        kend = ksta+kend_order-1;
    } else if (metgrid_use_below_sfc && !metgrid_use_sfc) {
        // Use all levels except for the surface.
        int ksfc = 0;
        for (int k=0; k < kmax_orig; k++) {
            if (ordered_z_p[k] == orig_z_p[0]) {
                ksfc = k;
                break;
            }
        }
        for (int k=ksfc; k < kmax_orig-1; k++) {
            ordered_z_p[k] = ordered_z_p[k+1];
            ordered_data_p[k] = ordered_data_p[k+1];
        }
        ksta = 0;
        kend = ksta+kend_order-2;
    } else if (!metgrid_use_below_sfc && metgrid_use_sfc) {
        // Use all levels above and including the surface.
        int kcount = k_above_sfc-1-zap_below;
        int  count = 0;
        for (int k=0; k < kmax_orig; k++) {
            if (ordered_z_p[kcount] == orig_z_p[k]) {
                kcount++;
                count++;
            }
        }
        ksta = k_above_sfc-1-zap_below;
        kend = ksta+count-1;
    } else {
        // We shouldn't be in here!
        amrex::Abort("metgrid initialization, !use_levels below_ground && !metgrid_use_sfc");
    }

    // Insert the level of maximum winds.
//    amrex::Real maxw_above_this_level = 30000.0;
//    amrex::Real maxw_horiz_pres_diff = 5000.0;
//    if ((flag_maxw == 1) && (use_maxw)) {
//        amrex::Abort("metgrid initialization, use_maxw not yet implemented");
//    }

    // Insert the level of the tropopause.
//    amrex::Real trop_horiz_pres_diff = 5000.0;
//    if ((flag_trop == 1) && (use_trop)) {
//        amrex::Abort("metgrid initialization, use_trop not yet implemented");
//    }

    amrex::GpuArray<amrex::Real,256> ordered_p;
    amrex::Real* ordered_p_p = ordered_p.data();
    for (int k=0; k < kend_order; k++) {
        calc_p_isothermal(ordered_z_p[k], ordered_p_p[k]);
    }

    int new_n = 0;
    int zap_final = 0;
    amrex::GpuArray<amrex::Real,256> final_z;
    amrex::GpuArray<amrex::Real,256> final_p;
    amrex::GpuArray<amrex::Real,256> final_data;
    amrex::Real* final_z_p = final_z.data();
    amrex::Real* final_p_p = final_p.data();
    amrex::Real* final_data_p = final_data.data();
    final_z_p[0] = ordered_z[ksta];
    final_p_p[0] = ordered_p[ksta];
    final_data_p[0] = ordered_data[ksta];
    for (int k=ksta+1; k <= kend; k++) {
        if ((final_p_p[new_n]-ordered_p_p[k]) < metgrid_proximity) {
            zap_final++;
        } else {
            new_n++;
            final_z_p[new_n] = ordered_z_p[k];
            final_p_p[new_n] = ordered_p_p[k];
            final_data_p[new_n] = ordered_data_p[k];
        }
    }
    kend -= zap_final;

    // Call the interpolator.
    lagrange_setup(var_type,
                   exp_interp,
                   kend-ksta,
                   kmax_new,
                   metgrid_order,
                   final_z_p,
                   final_p_p,
                   final_data_p,
                   new_z_p,
                   new_p_p,
                   new_data_p);

    // Optionally replace the lowest level of data with the surface
    // field from the origin data.
    if (metgrid_retain_sfc) new_data[0] = ordered_data[0];

    // Save the interpolated data.
    for (int k=0; k < kmax_new; k++) {
        if (mask(i,j,k) && update_bc_data && bx_xlo.contains(i,j,k)) bc_data_xlo(i,j,k,0) = new_data[k];
        if (mask(i,j,k) && update_bc_data && bx_xhi.contains(i,j,k)) bc_data_xhi(i,j,k,0) = new_data[k];
        if (mask(i,j,k) && update_bc_data && bx_ylo.contains(i,j,k)) bc_data_ylo(i,j,k,0) = new_data[k];
        if (mask(i,j,k) && update_bc_data && bx_yhi.contains(i,j,k)) bc_data_yhi(i,j,k,0) = new_data[k];
        if (itime == 0) new_data_full(i,j,k,src_comp) = new_data[k];
    }

}

AMREX_FORCE_INLINE
AMREX_GPU_DEVICE
amrex::Real
interpolate_column_metgrid_linear (const int& i,
                                   const int& j,
                                   const int& k,
                                   char stag,
                                   int src_comp,
                                   const amrex::Array4<amrex::Real const>& orig_z,
                                   const amrex::Array4<amrex::Real const>& orig_data,
                                   const amrex::Array4<amrex::Real const>&  new_z)
{
    // This subroutine is a bit ham-handed and can be cleaned up later.
    int imax_orig = amrex::ubound(amrex::Box(orig_data)).x;
    int jmax_orig = amrex::ubound(amrex::Box(orig_data)).y;
    int kmax_orig = amrex::ubound(amrex::Box(orig_data)).z;

    amrex::Real z;
    if (stag == 'X') {
        z = 0.25*(new_z(i,j,k)+new_z(i,j+1,k)+new_z(i,j,k+1)+new_z(i,j+1,k+1));
    }
    else if (stag == 'Y') {
        z = 0.25*(new_z(i,j,k)+new_z(i+1,j,k)+new_z(i,j,k+1)+new_z(i+1,j,k+1));
    }
    else if (stag == 'M') {
        z = 0.125*(new_z(i,j,k  )+new_z(i,j+1,k  )+new_z(i+1,j,k  )+new_z(i+1,j+1,k  )+
                   new_z(i,j,k+1)+new_z(i,j+1,k+1)+new_z(i+1,j,k+1)+new_z(i+1,j+1,k+1));
    }

    amrex::Real z0, z1;
    int klow   = -1;
    int khi0   = -1;
    amrex::Real dzlow =  1.0e12;
    amrex::Real dzhi0 = -1.0e12;
    for (int kk = 0; kk < kmax_orig; kk++) {
        amrex::Real orig_z_stag = 0.0;
        if (stag == 'M') {
            orig_z_stag = orig_z(i,j,kk);
        }
        if (stag == 'X') {
            if (i == 0) {
                orig_z_stag = orig_z(i,j,kk);
            }
            else if (i == imax_orig) {
                orig_z_stag = orig_z(imax_orig-1,j,kk);
            }
            else {
                orig_z_stag = 0.5*(orig_z(i,j,kk)+orig_z(i-1,j,kk));
            }
        }
        else if (stag == 'Y') {
            if (j == 0) {
                orig_z_stag = orig_z(i,j,kk);
            }
            else if (j == jmax_orig) {
                orig_z_stag = orig_z(i,jmax_orig-1,kk);
            }
            else {
                orig_z_stag = 0.5*(orig_z(i,j,kk)+orig_z(i,j-1,kk));
            }
        }

        amrex::Real dz = z - orig_z_stag;
        if ((dz < 0.0) && (dz > dzhi0)) {
            dzhi0 = dz;
            khi0  = kk;
            z1    = orig_z_stag;
        }
        if ((dz >= 0.0) && (dz < dzlow)) {
            dzlow = dz;
            klow  = kk;
            z0    = orig_z_stag;
        }
    } // kk

    // extrapolate below the bottom surface
    if (klow == -1) {
        int khi1   = -1;
        amrex::Real dzhi1 = -1.0e12;
        for (int kk = 0; kk < kmax_orig; kk++) {
            amrex::Real orig_z_stag = 0.0;
            if (stag == 'M') {
                orig_z_stag = orig_z(i,j,kk);
            }
            else if (stag == 'X') {
                if (i == 0) {
                    orig_z_stag = orig_z(i,j,kk);
                }
                else if (i == imax_orig) {
                    orig_z_stag = orig_z(imax_orig-1,j,kk);
                }
                else {
                    orig_z_stag = 0.5*(orig_z(i,j,kk)+orig_z(i-1,j,kk));
                }
            }
            else if (stag == 'Y') {
                if (j == 0) {
                    orig_z_stag = orig_z(i,j,kk);
                }
                else if (j == jmax_orig) {
                    orig_z_stag = orig_z(i,jmax_orig-1,kk);
                }
                else {
                    orig_z_stag = 0.5*(orig_z(i,j,kk)+orig_z(i,j-1,kk));
                }
            }
            amrex::Real dz = z - orig_z_stag;
            if ((dz < 0.0) && (dz > dzhi1) && (kk != khi0)) {
                dzhi1 = dz;
                khi1  = kk;
                z1    = orig_z_stag;
            }
        }
        amrex::Real y0 = orig_data(i,j,khi0,src_comp);
        amrex::Real y1 = orig_data(i,j,khi1,src_comp);
        return ( y0-(y1-y0)/(z1-z0)*(z0-z) );

    // Extrapolate above the top surface
    } else if (khi0 == -1) {
        khi0 = klow - 1;
        int khi1 = klow;
        if (stag == 'M') {
            z0 = orig_z(i,j,khi0);
        }
        else if (stag == 'X') {
            if (i == 0) {
                z0 = orig_z(i,j,khi0);
            }
            else if (i == imax_orig) {
                z0 = orig_z(imax_orig-1,j,khi0);
            }
            else {
                z0 = 0.5*(orig_z(i,j,khi0)+orig_z(i-1,j,khi0));
            }
        }
        else if (stag == 'Y') {
            if (j == 0) {
                z0 = orig_z(i,j,khi0);
            }
            else if (j == jmax_orig) {
                z0 = orig_z(i,jmax_orig-1,khi0);
            }
            else {
                z0 = 0.5*(orig_z(i,j,khi0)+orig_z(i,j-1,khi0));
            }
        }
        amrex::Real y0 = orig_data(i,j,khi0,src_comp);
        amrex::Real y1 = orig_data(i,j,khi1,src_comp);
        return ( y0+(y1-y0)/(z1-z0)*(z-z0) );
    } else {
        // interpolate
        amrex::Real y0 = orig_data(i,j,klow,src_comp);
        amrex::Real y1 = orig_data(i,j,khi0,src_comp);
        return ( y0+(y1-y0)/(z1-z0)*(z-z0) );

    }
}

AMREX_FORCE_INLINE
AMREX_GPU_DEVICE
void
rh_to_mxrat (int i,
             int j,
             int k,
             const amrex::Array4<amrex::Real const>& rhum,
             const amrex::Array4<amrex::Real const>& temp,
             const amrex::Array4<amrex::Real const>& pres,
             const amrex::Array4<amrex::Real>& mxrat)
{
    amrex::Real qv_max_p_safe = 10000.0; // WRF default value
    amrex::Real qv_max_flag   = 1.0e-5; // WRF default value
    amrex::Real qv_max_value  = 3.0e-6; // WRF default value
    amrex::Real qv_min_p_safe = 110000.0; // WRF default value
    amrex::Real qv_min_flag   = 1.0e-6; // WRF default value
    amrex::Real qv_min_value  = 1.0e-6; // WRF default value
    amrex::Real eps   = 0.622;
    amrex::Real svp1  = 0.6112;
    amrex::Real svp2  = 17.67;
    amrex::Real svp3  = 29.65;
    amrex::Real svpt0 = 273.15;
    // WRF's method when model_config_rec%rh2qv_wrt_liquid=.true. (default behavior)
    if (temp(i,j,k) != 0.0) {
        amrex::Real es=0.01*rhum(i,j,k)*svp1*10.0*exp(svp2*(temp(i,j,k)-svpt0)/(temp(i,j,k)-svp3));
        if (es >= pres(i,j,k)/100.0) {
            // vapor pressure exceeds total pressure
            mxrat(i,j,k) = std::pow(10.0, -6);
        }
        else {
            mxrat(i,j,k) = amrex::max(eps*es/(pres(i,j,k)/100.0-es), 1.0e-6);
        }
    }
    else {
        // I don't know why there's a fringe case handled in WRF where T is absolute zero...
        // Let's just deal with it here in case we also end up needing it.
        mxrat(i,j,k) = 1.0e-6;
    }
    // See the below comment from WRF dyn_em/module_initialize_real.F rh_to_mxrat1.
    // For pressures above a defined level, reasonable Qv values should be
    // a certain value or smaller. If they are larger than this, the input data
    // probably had "missing" RH, and we filled in some values. This is an
    // attempt to catch those. Also, set the minimum value for the entire
    // domain that is above the selected pressure level.
    if (pres(i,j,k) < qv_max_p_safe) {
        if (mxrat(i,j,k) > qv_max_flag) {
            mxrat(i,j,k) = qv_max_value;
        }
    }
    if (pres(i,j,k) < qv_min_p_safe) {
        if (mxrat(i,j,k) < qv_min_flag) {
            mxrat(i,j,k) = qv_min_value;
        }
    }
}
#endif
