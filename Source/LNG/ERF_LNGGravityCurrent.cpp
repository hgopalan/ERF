/**
 * @file ERF_LNGGravityCurrent.cpp
 * @brief Implementation of shallow-water gravity current PDEs for LNG Phase 5
 * 
 * Solves the depth-averaged shallow-water equations using explicit first-order
 * time-stepping. See ERF_LNGGravityCurrent.H for governing equations and references.
 */

#include "ERF_LNGGravityCurrent.H"

#ifdef ERF_USE_LNG

#include <AMReX_MFIter.H>
#include <AMReX_Reduce.H>
#include <AMReX_Print.H>
#include <cmath>

/**
 * @brief Advance shallow-water gravity current one timestep (explicit Euler)
 */
void advance_gravity_current(amrex::MultiFab&       lng_gc_h,
                              amrex::MultiFab&       lng_gc_u,
                              amrex::MultiFab&       lng_gc_v,
                              amrex::MultiFab&       lng_gc_ri_flag,
                              const amrex::MultiFab& lng_evap_flux,
                              const amrex::MultiFab& lng_ustar,
                              const amrex::Geometry& geom_lng,
                              amrex::Real            rho_vapor,
                              amrex::Real            rho_air,
                              amrex::Real            Cd,
                              amrex::Real            dt,
                              bool                   lng_debug)
{
    // Compute reduced gravity g'
    amrex::Real g_prime = compute_reduced_gravity(rho_vapor, rho_air);
    
    // Mesh spacing
    const auto& dx = geom_lng.CellSize();
    amrex::Real dx_inv = 1.0 / dx[0];
    
    // Time step coefficients
    amrex::Real dt_over_rho = dt / rho_vapor;
    
    // MFIter loop over all boxes
    for (amrex::MFIter mfi(lng_gc_h, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        
        auto h_arr    = lng_gc_h.array(mfi);
        auto u_arr    = lng_gc_u.array(mfi);
        auto v_arr    = lng_gc_v.array(mfi);
        auto ri_flag  = lng_gc_ri_flag.array(mfi);
        auto evap     = lng_evap_flux.const_array(mfi);
        auto ustar    = lng_ustar.const_array(mfi);
        
        // Temporary arrays for updated state
        amrex::FArrayBox h_new(bx, 1, amrex::The_Async_Arena());
        amrex::FArrayBox u_new(bx, 1, amrex::The_Async_Arena());
        amrex::FArrayBox v_new(bx, 1, amrex::The_Async_Arena());
        auto h_new_arr = h_new.array();
        auto u_new_arr = u_new.array();
        auto v_new_arr = v_new.array();
        
        // Step 1: Update from evaporation source and pressure gradients
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real h_old = h_arr(i, j, k);
            amrex::Real u_old = u_arr(i, j, k);
            amrex::Real v_old = v_arr(i, j, k);
            
            amrex::Real hu_old = h_old * u_old;
            amrex::Real hv_old = h_old * v_old;
            
            amrex::Real evap_src = evap(i, j, k) * dt_over_rho;
            
            amrex::Real h_new_val = amrex::max(h_old + evap_src, 0.0);
            
            // Pressure gradient using central/one-sided differences
            amrex::Real dhx = 0.0;
            amrex::Real dhy = 0.0;
            
            if (i > bx.loVect()[0] && i < bx.hiVect()[0]) {
                dhx = (h_arr(i+1, j, k) - h_arr(i-1, j, k)) * 0.5 * dx_inv;
            } else if (i < bx.hiVect()[0]) {
                dhx = (h_arr(i+1, j, k) - h_arr(i, j, k)) * dx_inv;
            } else if (i > bx.loVect()[0]) {
                dhx = (h_arr(i, j, k) - h_arr(i-1, j, k)) * dx_inv;
            }
            
            if (j > bx.loVect()[1] && j < bx.hiVect()[1]) {
                dhy = (h_arr(i, j+1, k) - h_arr(i, j-1, k)) * 0.5 * dx_inv;
            } else if (j < bx.hiVect()[1]) {
                dhy = (h_arr(i, j+1, k) - h_arr(i, j, k)) * dx_inv;
            } else if (j > bx.loVect()[1]) {
                dhy = (h_arr(i, j, k) - h_arr(i, j-1, k)) * dx_inv;
            }
            
            amrex::Real h_avg = 0.5 * (h_old + h_new_val);
            amrex::Real pressure_grad_x = -g_prime * h_avg * dhx;
            amrex::Real pressure_grad_y = -g_prime * h_avg * dhy;
            
            amrex::Real drag_x = -Cd * u_old * std::abs(u_old);
            amrex::Real drag_y = -Cd * v_old * std::abs(v_old);
            
            amrex::Real hu_new_val = hu_old + (pressure_grad_x + drag_x) * dt;
            amrex::Real hv_new_val = hv_old + (pressure_grad_y + drag_y) * dt;
            
            amrex::Real u_new_val = 0.0;
            amrex::Real v_new_val = 0.0;
            
            if (h_new_val > LNGGravityConst::H_MIN) {
                u_new_val = hu_new_val / h_new_val;
                v_new_val = hv_new_val / h_new_val;
            } else {
                h_new_val = 0.0;
            }
            
            u_new_val = amrex::max(amrex::min(u_new_val,  LNGGravityConst::U_MAX),
                                               -LNGGravityConst::U_MAX);
            v_new_val = amrex::max(amrex::min(v_new_val,  LNGGravityConst::U_MAX),
                                               -LNGGravityConst::U_MAX);
            
            h_new_arr(i, j, k) = h_new_val;
            u_new_arr(i, j, k) = u_new_val;
            v_new_arr(i, j, k) = v_new_val;
        });
        
        // Step 2: Copy updated state back and compute Richardson number
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real h_val     = h_new_arr(i, j, k);
            amrex::Real u_val     = u_new_arr(i, j, k);
            amrex::Real v_val     = v_new_arr(i, j, k);
            amrex::Real ustar_val = ustar(i, j, k);
            
            h_arr(i, j, k) = h_val;
            u_arr(i, j, k) = u_val;
            v_arr(i, j, k) = v_val;
            
            amrex::Real Ri = compute_richardson(g_prime, h_val, ustar_val);
            ri_flag(i, j, k) = is_gravity_current_active(Ri) ? 0.0 : 1.0;
        });
    }
    
    // Optional debug output — use standalone AMReX reduce functions
    if (lng_debug) {
        amrex::Real h_max_val = lng_gc_h.max(0);
        amrex::Real u_max_val = lng_gc_u.max(0);
        amrex::Real v_max_val = lng_gc_v.max(0);

        // Count gravity-current-active cells (ri_flag == 0) using ReduceSum
        amrex::Long gc_active_cells = amrex::ReduceSum(lng_gc_ri_flag, 0,
            [=] (amrex::Box const& bx,
                 amrex::Array4<amrex::Real const> const& flag) -> amrex::Long
            {
                amrex::Long cnt = 0;
                amrex::Loop(bx, [&](int i, int j, int k) {
                    if (flag(i,j,k) < 0.5) ++cnt;
                });
                return cnt;
            });

        // Compute Ri max and min over all cells
        amrex::Real ri_max_val = amrex::ReduceMax(lng_gc_h, lng_ustar, 0,
            [=] (amrex::Box const& bx,
                 amrex::Array4<amrex::Real const> const& h,
                 amrex::Array4<amrex::Real const> const& us) -> amrex::Real
            {
                amrex::Real mx = -1.0e30;
                amrex::Loop(bx, [&](int i, int j, int k) {
                    amrex::Real Ri = compute_richardson(g_prime, h(i,j,k), us(i,j,k));
                    mx = amrex::max(mx, Ri);
                });
                return mx;
            });

        amrex::Real ri_min_val = amrex::ReduceMin(lng_gc_h, lng_ustar, 0,
            [=] (amrex::Box const& bx,
                 amrex::Array4<amrex::Real const> const& h,
                 amrex::Array4<amrex::Real const> const& us) -> amrex::Real
            {
                amrex::Real mn = 1.0e30;
                amrex::Loop(bx, [&](int i, int j, int k) {
                    amrex::Real Ri = compute_richardson(g_prime, h(i,j,k), us(i,j,k));
                    mn = amrex::min(mn, Ri);
                });
                return mn;
            });

        amrex::Long total_cells  = static_cast<amrex::Long>(lng_gc_ri_flag.boxArray().numPts());
        amrex::Long mixed_cells  = total_cells - gc_active_cells;

        amrex::Print() << "[LNG DEBUG] Phase 5: gravity_current"
                       << "  h_max=" << h_max_val << " m"
                       << "  u_max=" << u_max_val << " m/s"
                       << "  v_max=" << v_max_val << " m/s\n"
                       << "[LNG DEBUG] Phase 5:"
                       << "  g_prime=" << g_prime << " m/s^2"
                       << "  Ri_max=" << ri_max_val
                       << "  Ri_min=" << ri_min_val
                       << "  gc_active_cells=" << gc_active_cells
                       << "  mixed_cells=" << mixed_cells << "\n";
    }
}

#endif /* ERF_USE_LNG */