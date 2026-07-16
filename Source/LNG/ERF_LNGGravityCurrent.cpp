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
 * 
 * The algorithm:
 * 1. Compute g' = g*(rho_vapor - rho_air)/rho_air
 * 2. For each cell in a ParallelFor:
 *    a) Compute source from evaporation: F_evap/rho_vapor
 *    b) Update depth h: h_new = h + (F_evap/rho_vapor)*dt
 *    c) Compute gradients ∂h/∂x, ∂h/∂y using central differences (upwind fallback)
 *    d) Update x-momentum: hu_new = hu + (-g'*h*∂h/∂x - Cd*u*|u|) * dt
 *    e) Update y-momentum: hv_new = hv + (-g'*h*∂h/∂y - Cd*v*|v|) * dt
 *    f) Recover u_new = hu_new/h_new, v_new = hv_new/h_new
 *    g) Clamp velocities to [-U_MAX, U_MAX]
 *    h) Zero out fields where h < H_MIN
 * 3. Compute Richardson number and set ri_flag for each cell
 * 4. Optional debug output
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
    amrex::Real dx_inv = 1.0 / dx[0];  // Assuming square grid: dx[0] = dx[1]
    
    // Statistics for debug output
    amrex::Real h_max_val = 0.0;
    amrex::Real u_max_val = 0.0;
    amrex::Real ri_max_val = 0.0;
    amrex::Real ri_min_val = 1e10;
    amrex::Long gc_active_cells = 0;
    amrex::Long mixed_cells = 0;
    
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
        amrex::FArrayBox h_new(bx);
        amrex::FArrayBox u_new(bx);
        amrex::FArrayBox v_new(bx);
        auto h_new_arr = h_new.array();
        auto u_new_arr = u_new.array();
        auto v_new_arr = v_new.array();
        
        // Step 1: Update from evaporation source and pressure gradients
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            // Current state
            amrex::Real h_old = h_arr(i, j, k);
            amrex::Real u_old = u_arr(i, j, k);
            amrex::Real v_old = v_arr(i, j, k);
            
            // Momentum (depth-integrated)
            amrex::Real hu_old = h_old * u_old;
            amrex::Real hv_old = h_old * v_old;
            
            // Source from evaporation
            amrex::Real evap_src = evap(i, j, k) * dt_over_rho;
            
            // Update depth
            amrex::Real h_new_val = h_old + evap_src;
            h_new_val = amrex::max(h_new_val, 0.0);
            
            // Compute pressure gradient using central differences with boundaries
            // ∂h/∂x = (h(i+1,j) - h(i-1,j)) / (2*dx)
            // At boundaries, fall back to one-sided
            amrex::Real dhx = 0.0;
            amrex::Real dhy = 0.0;
            
            // For now, use simple upwind on available neighbors
            // This is a first-order explicit scheme
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
            
            // Pressure gradient force: -g'*h*∂h/∂x
            // Use average depth (old + new) / 2 for stability
            amrex::Real h_avg = 0.5 * (h_old + h_new_val);
            amrex::Real pressure_grad_x = -g_prime * h_avg * dhx;
            amrex::Real pressure_grad_y = -g_prime * h_avg * dhy;
            
            // Drag force: -Cd*u*|u| and -Cd*v*|v|
            amrex::Real drag_x = -Cd * u_old * std::abs(u_old);
            amrex::Real drag_y = -Cd * v_old * std::abs(v_old);
            
            // Update momentum
            amrex::Real hu_new_val = hu_old + (pressure_grad_x + drag_x) * dt;
            amrex::Real hv_new_val = hv_old + (pressure_grad_y + drag_y) * dt;
            
            // Recover velocities (avoid division by small h)
            amrex::Real u_new_val = 0.0;
            amrex::Real v_new_val = 0.0;
            
            if (h_new_val > LNGGravityConst::H_MIN) {
                u_new_val = hu_new_val / h_new_val;
                v_new_val = hv_new_val / h_new_val;
            } else {
                // Cloud has evaporated away
                h_new_val = 0.0;
                u_new_val = 0.0;
                v_new_val = 0.0;
            }
            
            // Clamp velocities to prevent runaway
            u_new_val = amrex::max(amrex::min(u_new_val, LNGGravityConst::U_MAX),
                                    -LNGGravityConst::U_MAX);
            v_new_val = amrex::max(amrex::min(v_new_val, LNGGravityConst::U_MAX),
                                    -LNGGravityConst::U_MAX);
            
            // Store updated state
            h_new_arr(i, j, k) = h_new_val;
            u_new_arr(i, j, k) = u_new_val;
            v_new_arr(i, j, k) = v_new_val;
        });
        
        // Step 2: Copy updated state back to arrays and compute Richardson number
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real h_val = h_new_arr(i, j, k);
            amrex::Real u_val = u_new_arr(i, j, k);
            amrex::Real v_val = v_new_arr(i, j, k);
            amrex::Real ustar_val = ustar(i, j, k);
            
            // Update the arrays
            h_arr(i, j, k) = h_val;
            u_arr(i, j, k) = u_val;
            v_arr(i, j, k) = v_val;
            
            // Compute Richardson number for this cell
            amrex::Real Ri = compute_richardson(g_prime, h_val, ustar_val);
            
            // Set flag: 0 = gravity current active, 1 = mixed regime
            ri_flag(i, j, k) = (is_gravity_current_active(Ri)) ? 0.0 : 1.0;
        });
    }
    
    // Optional debug output
    if (lng_debug) {
        // Compute statistics across all boxes
        h_max_val = lng_gc_h.max(0);
        u_max_val = lng_gc_u.max(0);
        
        // Count active vs. mixed cells
        for (amrex::MFIter mfi(lng_gc_ri_flag, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto ri_flag = lng_gc_ri_flag.const_array(mfi);
            auto ustar = lng_ustar.const_array(mfi);
            auto h = lng_gc_h.const_array(mfi);
            
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpMax, amrex::ReduceOpMin> 
                reduce_op;
            amrex::ReduceData<amrex::Long, amrex::Real, amrex::Real> reduce_data(reduce_op);
            
            using ReduceTuple = typename decltype(reduce_data)::Type;
            
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::Real Ri = compute_richardson(g_prime, h(i, j, k), ustar(i, j, k));
                reduce_op.join(
                    ReduceTuple(
                        (ri_flag(i, j, k) < 0.5) ? 1L : 0L,  // Count active cells
                        Ri,                                   // Max Ri
                        Ri                                    // Min Ri
                    )
                );
            });
            
            auto final_data = reduce_data.value();
            gc_active_cells += amrex::get<0>(final_data);
            ri_max_val = amrex::max(ri_max_val, amrex::get<1>(final_data));
            ri_min_val = amrex::min(ri_min_val, amrex::get<2>(final_data));
        }
        
        mixed_cells = lng_gc_ri_flag.size() - gc_active_cells;
        
        amrex::Print() << "[LNG DEBUG] Phase 5: gravity_current"
                       << "  h_max=" << h_max_val << " m"
                       << "  u_max=" << u_max_val << " m/s"
                       << "  v_max=" << lng_gc_v.max(0) << " m/s\n"
                       << "[LNG DEBUG] Phase 5:"
                       << "  g_prime=" << g_prime << " m/s^2"
                       << "  Ri_max=" << ri_max_val
                       << "  Ri_min=" << ri_min_val
                       << "  gc_active_cells=" << gc_active_cells
                       << "  mixed_cells=" << mixed_cells << "\n";
    }
}

#endif /* ERF_USE_LNG */
