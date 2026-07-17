/**
 * @file ERF_LNGGravityCurrent.cpp
 * @brief Implementation of shallow-water gravity current PDEs for LNG Phase 5
 *
 * Fixes applied:
 *
 * Fix 1 -- CFL-based velocity cap (replaces hard-coded U_MAX=50 m/s).
 *
 * Fix 2 -- Pool depletion coupling: zero gc state when pool_mask==0 AND h < H_MIN.
 *
 * Fix 3 -- Atmospheric dilution decay: exponential decay of h away from source.
 *
 * Fix 5 -- Pool confinement: zero gc state in ALL off-pool cells after Euler update.
 *
 * Fix 6 -- Source accumulation cap (THIS FIX):
 *   The depth equation h += F_evap*dt/rho_vapor has no sink in pool cells,
 *   causing h to grow to ~131 m over 7200 steps (= 0.064/1.76 * 0.5 * 7200).
 *   gc_h represents the DENSE VAPOR CLOUD HEIGHT above the pool, which is
 *   physically bounded by the pool depth. After the Euler update, cap h at
 *   pool_depth in all cells where pool_mask==1.
 */

#include "ERF_LNGGravityCurrent.H"

#ifdef ERF_USE_LNG

#include <AMReX_MFIter.H>
#include <AMReX_Reduce.H>
#include <AMReX_Print.H>
#include <cmath>

void advance_gravity_current(amrex::MultiFab&       lng_gc_h,
                              amrex::MultiFab&       lng_gc_u,
                              amrex::MultiFab&       lng_gc_v,
                              amrex::MultiFab&       lng_gc_ri_flag,
                              const amrex::MultiFab& lng_evap_flux,
                              const amrex::MultiFab& lng_ustar,
                              const amrex::MultiFab& lng_pool_mask,
                              const amrex::MultiFab& lng_pool_depth,
                              const amrex::Geometry& geom_lng,
                              amrex::Real            rho_vapor,
                              amrex::Real            rho_air,
                              amrex::Real            Cd,
                              amrex::Real            dt,
                              bool                   lng_debug)
{
    amrex::Real g_prime     = compute_reduced_gravity(rho_vapor, rho_air);
    const auto& dx          = geom_lng.CellSize();
    amrex::Real dx_inv      = 1.0 / dx[0];
    amrex::Real dt_over_rho = dt / rho_vapor;

    // Fix 1: CFL-based velocity cap
    const amrex::Real u_max_cfl = LNGGravityConst::CFL_SAFETY
                                  * amrex::min(dx[0], dx[1]) / dt;

    // Fix 3: atmospheric dilution decay factor
    const amrex::Real decay_factor = 1.0 - dt / LNGGravityConst::H_DECAY_TIMESCALE;

    for (amrex::MFIter mfi(lng_gc_h, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();

        auto h_arr    = lng_gc_h.array(mfi);
        auto u_arr    = lng_gc_u.array(mfi);
        auto v_arr    = lng_gc_v.array(mfi);
        auto ri_flag  = lng_gc_ri_flag.array(mfi);
        auto evap     = lng_evap_flux.const_array(mfi);
        auto ustar    = lng_ustar.const_array(mfi);
        auto pmask    = lng_pool_mask.const_array(mfi);
        auto pdepth   = lng_pool_depth.const_array(mfi);  // Fix 6

        amrex::FArrayBox h_new(bx, 1, amrex::The_Async_Arena());
        amrex::FArrayBox u_new(bx, 1, amrex::The_Async_Arena());
        amrex::FArrayBox v_new(bx, 1, amrex::The_Async_Arena());
        auto h_new_arr = h_new.array();
        auto u_new_arr = u_new.array();
        auto v_new_arr = v_new.array();

        // Step 1: Euler update
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real h_old  = h_arr(i, j, k);
            amrex::Real u_old  = u_arr(i, j, k);
            amrex::Real v_old  = v_arr(i, j, k);
            amrex::Real hu_old = h_old * u_old;
            amrex::Real hv_old = h_old * v_old;

            amrex::Real h_new_val = amrex::max(h_old + evap(i,j,k) * dt_over_rho, 0.0);

            // Fix 2: zero off-pool cells with negligible depth
            if (pmask(i,j,k) < 0.5 && h_new_val < LNGGravityConst::H_MIN) {
                h_new_arr(i,j,k) = 0.0;
                u_new_arr(i,j,k) = 0.0;
                v_new_arr(i,j,k) = 0.0;
                return;
            }

            // Fix 3: dilution decay away from active source
            if (evap(i,j,k) < 1.0e-10 && h_new_val > 0.0) {
                h_new_val = amrex::max(h_new_val * decay_factor, 0.0);
            }

            // Fix 6: cap gc_h at pool_depth in pool cells
            // gc_h is the dense vapor cloud height, bounded by the liquid depth.
            if (pmask(i,j,k) >= 0.5) {
                h_new_val = amrex::min(h_new_val, amrex::max(pdepth(i,j,k), 0.0));
            }

            // Pressure gradient (central differences)
            amrex::Real dhx = 0.0, dhy = 0.0;
            if (i > bx.loVect()[0] && i < bx.hiVect()[0])
                dhx = (h_arr(i+1,j,k) - h_arr(i-1,j,k)) * 0.5 * dx_inv;
            else if (i < bx.hiVect()[0])
                dhx = (h_arr(i+1,j,k) - h_arr(i,j,k)) * dx_inv;
            else if (i > bx.loVect()[0])
                dhx = (h_arr(i,j,k) - h_arr(i-1,j,k)) * dx_inv;

            if (j > bx.loVect()[1] && j < bx.hiVect()[1])
                dhy = (h_arr(i,j+1,k) - h_arr(i,j-1,k)) * 0.5 * dx_inv;
            else if (j < bx.hiVect()[1])
                dhy = (h_arr(i,j+1,k) - h_arr(i,j,k)) * dx_inv;
            else if (j > bx.loVect()[1])
                dhy = (h_arr(i,j,k) - h_arr(i,j-1,k)) * dx_inv;

            amrex::Real h_avg      = 0.5 * (h_old + h_new_val);
            amrex::Real hu_new_val = hu_old + (-g_prime*h_avg*dhx - Cd*u_old*std::abs(u_old)) * dt;
            amrex::Real hv_new_val = hv_old + (-g_prime*h_avg*dhy - Cd*v_old*std::abs(v_old)) * dt;

            amrex::Real u_new_val = 0.0, v_new_val = 0.0;
            if (h_new_val > LNGGravityConst::H_MIN) {
                u_new_val = hu_new_val / h_new_val;
                v_new_val = hv_new_val / h_new_val;
            } else {
                h_new_val = 0.0;
            }

            // Fix 1: CFL cap
            u_new_val = amrex::max(amrex::min(u_new_val,  u_max_cfl), -u_max_cfl);
            v_new_val = amrex::max(amrex::min(v_new_val,  u_max_cfl), -u_max_cfl);

            h_new_arr(i,j,k) = h_new_val;
            u_new_arr(i,j,k) = u_new_val;
            v_new_arr(i,j,k) = v_new_val;
        });

        // Step 2: write back + Richardson flag
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real h_val = h_new_arr(i,j,k);
            h_arr(i,j,k)     = h_val;
            u_arr(i,j,k)     = u_new_arr(i,j,k);
            v_arr(i,j,k)     = v_new_arr(i,j,k);
            amrex::Real Ri   = compute_richardson(g_prime, h_val, ustar(i,j,k));
            ri_flag(i,j,k)   = is_gravity_current_active(Ri) ? 0.0 : 1.0;
        });

        // Fix 5: pool confinement — zero ALL off-pool cells
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (pmask(i,j,k) < 0.5) {
                h_arr(i,j,k)   = 0.0;
                u_arr(i,j,k)   = 0.0;
                v_arr(i,j,k)   = 0.0;
                ri_flag(i,j,k) = 0.0;
            }
        });
    }

    lng_gc_h.FillBoundary(geom_lng.periodicity());
    lng_gc_u.FillBoundary(geom_lng.periodicity());
    lng_gc_v.FillBoundary(geom_lng.periodicity());
    lng_gc_ri_flag.FillBoundary(geom_lng.periodicity());

    if (lng_debug) {
        amrex::Real h_max_val = lng_gc_h.max(0);
        amrex::Real u_max_val = lng_gc_u.max(0);
        amrex::Real v_max_val = lng_gc_v.max(0);

        amrex::Long total_cells = static_cast<amrex::Long>(
            lng_gc_ri_flag.boxArray().numPts());
        amrex::Real ri_flag_sum = lng_gc_ri_flag.sum(0);
        amrex::Long mixed_cells = static_cast<amrex::Long>(ri_flag_sum);
        amrex::Long gc_active   = total_cells - mixed_cells;

        amrex::MultiFab ri_scratch(lng_gc_h.boxArray(),
                                   lng_gc_h.DistributionMap(), 1, 0);
        for (amrex::MFIter mfi(ri_scratch, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto ri_arr = ri_scratch.array(mfi);
            auto h_a    = lng_gc_h.const_array(mfi);
            auto us_arr = lng_ustar.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                ri_arr(i,j,k) = compute_richardson(g_prime, h_a(i,j,k), us_arr(i,j,k));
            });
        }
        amrex::Real ri_max_val = ri_scratch.max(0);
        amrex::Real ri_min_val = ri_scratch.min(0);

        amrex::Print() << "[LNG DEBUG] Phase 5: gravity_current"
                       << "  h_max=" << h_max_val << " m"
                       << "  u_max=" << u_max_val << " m/s"
                       << "  v_max=" << v_max_val << " m/s\n"
                       << "[LNG DEBUG] Phase 5:"
                       << "  g_prime=" << g_prime << " m/s^2"
                       << "  Ri_max=" << ri_max_val
                       << "  Ri_min=" << ri_min_val
                       << "  gc_active_cells=" << gc_active
                       << "  mixed_cells=" << mixed_cells
                       << "  u_max_cfl=" << u_max_cfl << " m/s\n";
    }
}

#endif /* ERF_USE_LNG */
