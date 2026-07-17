/**
 * @file ERF_LNGGravityCurrent.cpp
 * @brief Implementation of shallow-water gravity current PDEs for LNG Phase 5
 *
 * Four fixes applied vs the original implementation:
 *
 * Fix 1 -- CFL-based velocity cap (replaces hard-coded U_MAX=50 m/s):
 *   U_MAX = CFL_SAFETY * min(dx,dy) / dt
 *   For a 3000 m domain on 128 cells (grid_ratio=4, n_cell=32):
 *     dx = 3000/128 = 23.4375 m
 *     U_MAX = 0.9 * 23.4375 / 0.5 = 42.2 m/s  (CFL = 0.9, stable)
 *   The previous 50 m/s cap allowed CFL > 1 -> velocities grew unboundedly.
 *
 * Fix 2 -- Pool depletion coupling (new lng_pool_mask argument):
 *   When pool_mask==0 AND gc_h < H_MIN, zero gc_h/gc_u/gc_v.
 *   Stops gravity current spreading after pool evaporates.
 *
 * Fix 3 -- Atmospheric dilution decay:
 *   Away from active evaporation source (evap==0), apply exponential decay
 *   to h with timescale H_DECAY_TIMESCALE=600 s.
 *
 * Fix 4 -- Spreading threshold (H_SPREAD_MIN = 1 cm):
 *   Skip the pressure gradient in off-pool cells where h_old < H_SPREAD_MIN.
 *   Without this, negligible-depth cells (h > H_MIN = 0.1 mm but < 1 cm)
 *   still computed the pressure gradient and accumulated depth from adjacent
 *   pool cells, causing gc_h_max to grow to 152 m at t=3600 s even with
 *   an active 200 kg/s spill maintaining the pool.
 *   Physical interpretation: a 1 cm vapor layer is too thin to sustain
 *   gravity-current-driven spreading away from the pool source region.
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

    // Fix 1: CFL-based velocity cap computed at runtime.
    const amrex::Real u_max_cfl = LNGGravityConst::CFL_SAFETY
                                  * amrex::min(dx[0], dx[1]) / dt;

    // Fix 3: decay factor for atmospheric dilution away from source
    const amrex::Real decay_factor = 1.0 - dt / LNGGravityConst::H_DECAY_TIMESCALE;

    // -------------------------------------------------------------------------
    // Physics update -- one MFIter tile at a time
    // -------------------------------------------------------------------------
    for (amrex::MFIter mfi(lng_gc_h, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();

        auto h_arr   = lng_gc_h.array(mfi);
        auto u_arr   = lng_gc_u.array(mfi);
        auto v_arr   = lng_gc_v.array(mfi);
        auto ri_flag = lng_gc_ri_flag.array(mfi);
        auto evap    = lng_evap_flux.const_array(mfi);
        auto ustar   = lng_ustar.const_array(mfi);
        auto pmask   = lng_pool_mask.const_array(mfi);

        // Temporary FABs for the updated state (GPU-safe arena)
        amrex::FArrayBox h_new(bx, 1, amrex::The_Async_Arena());
        amrex::FArrayBox u_new(bx, 1, amrex::The_Async_Arena());
        amrex::FArrayBox v_new(bx, 1, amrex::The_Async_Arena());
        auto h_new_arr = h_new.array();
        auto u_new_arr = u_new.array();
        auto v_new_arr = v_new.array();

        // Step 1: explicit Euler update of depth and momentum
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real h_old  = h_arr(i, j, k);
            amrex::Real u_old  = u_arr(i, j, k);
            amrex::Real v_old  = v_arr(i, j, k);
            amrex::Real hu_old = h_old * u_old;
            amrex::Real hv_old = h_old * v_old;

            // Depth update from evaporation source
            amrex::Real h_new_val = amrex::max(h_old + evap(i,j,k) * dt_over_rho, 0.0);

            // Fix 2: pool depletion coupling.
            // If pool has evaporated (mask==0) and depth is negligible, zero out.
            if (pmask(i,j,k) < 0.5 && h_new_val < LNGGravityConst::H_MIN) {
                h_new_arr(i,j,k) = 0.0;
                u_new_arr(i,j,k) = 0.0;
                v_new_arr(i,j,k) = 0.0;
                return;
            }

            // Fix 3: atmospheric dilution decay away from active source
            if (evap(i,j,k) < 1.0e-10 && h_new_val > 0.0) {
                h_new_val *= decay_factor;
                h_new_val  = amrex::max(h_new_val, 0.0);
            }

            // Fix 4: spreading threshold.
            // Off-pool cells with h < H_SPREAD_MIN do not participate in the
            // shallow-water pressure gradient. Without this, thin off-pool cells
            // accumulate depth from adjacent pool cells via pressure spreading,
            // causing gc_h to grow to unphysical values (152 m at t=3600 s).
            if (pmask(i,j,k) < 0.5 && h_old < LNGGravityConst::H_SPREAD_MIN) {
                // Apply decay only; no pressure gradient, no momentum transfer
                h_new_arr(i,j,k) = h_new_val;
                u_new_arr(i,j,k) = 0.0;
                v_new_arr(i,j,k) = 0.0;
                return;
            }

            // Pressure gradient dh/dx, dh/dy -- central differences
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

            // Pressure gradient + drag: semi-implicit depth average
            amrex::Real h_avg      = 0.5 * (h_old + h_new_val);
            amrex::Real hu_new_val = hu_old + (-g_prime*h_avg*dhx - Cd*u_old*std::abs(u_old)) * dt;
            amrex::Real hv_new_val = hv_old + (-g_prime*h_avg*dhy - Cd*v_old*std::abs(v_old)) * dt;

            // Recover velocities; zero out if depth too small
            amrex::Real u_new_val = 0.0, v_new_val = 0.0;
            if (h_new_val > LNGGravityConst::H_MIN) {
                u_new_val = hu_new_val / h_new_val;
                v_new_val = hv_new_val / h_new_val;
            } else {
                h_new_val = 0.0;
            }

            // Fix 1: CFL-based cap (replaces hard-coded 50 m/s)
            u_new_val = amrex::max(amrex::min(u_new_val,  u_max_cfl), -u_max_cfl);
            v_new_val = amrex::max(amrex::min(v_new_val,  u_max_cfl), -u_max_cfl);

            h_new_arr(i,j,k) = h_new_val;
            u_new_arr(i,j,k) = u_new_val;
            v_new_arr(i,j,k) = v_new_val;
        });

        // Step 2: write back updated state and compute Richardson flag
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real h_val = h_new_arr(i,j,k);
            h_arr(i,j,k)     = h_val;
            u_arr(i,j,k)     = u_new_arr(i,j,k);
            v_arr(i,j,k)     = v_new_arr(i,j,k);
            amrex::Real Ri   = compute_richardson(g_prime, h_val, ustar(i,j,k));
            ri_flag(i,j,k)   = is_gravity_current_active(Ri) ? 0.0 : 1.0;
        });
    }

    // -------------------------------------------------------------------------
    // FillBoundary -- REQUIRED for >1 MPI rank (Rule B4)
    // -------------------------------------------------------------------------
    lng_gc_h.FillBoundary(geom_lng.periodicity());
    lng_gc_u.FillBoundary(geom_lng.periodicity());
    lng_gc_v.FillBoundary(geom_lng.periodicity());
    lng_gc_ri_flag.FillBoundary(geom_lng.periodicity());

    // -------------------------------------------------------------------------
    // Debug diagnostics -- ALL MPI-collective MultiFab operations (Rule B6)
    // -------------------------------------------------------------------------
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
            auto h_arr  = lng_gc_h.const_array(mfi);
            auto us_arr = lng_ustar.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                ri_arr(i,j,k) = compute_richardson(g_prime, h_arr(i,j,k), us_arr(i,j,k));
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
