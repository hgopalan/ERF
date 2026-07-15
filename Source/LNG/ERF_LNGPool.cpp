/**
 * @file ERF_LNGPool.cpp
 * @brief Implementation of LNG pool dynamics and geometry functions
 * @note Phase 2: Basic pool evolution (no gravity current spreading yet)
 * @ref ERF_LNGPool.H — function declarations and theory
 */

#include "ERF_LNGPool.H"
#include <AMReX_MFIter.H>
#include <cmath>

void update_pool_mask(amrex::MultiFab& lng_pool_mask,
                      const amrex::MultiFab& lng_pool_depth,
                      amrex::Real depth_threshold)
{
    for (amrex::MFIter mfi(lng_pool_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& bx   = mfi.tilebox();
        const auto depth_arr = lng_pool_depth[mfi].array();
        auto       mask_arr  = lng_pool_mask[mfi].array();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            mask_arr(i, j, k) = (depth_arr(i, j, k) > depth_threshold) ? 1.0 : 0.0;
        });
    }

    lng_pool_mask.FillBoundary();
}

void apply_spill_source(amrex::MultiFab& lng_pool_depth,
                        const amrex::Geometry& geom_lng,
                        amrex::Real spill_rate_kg_s,
                        amrex::Real rho_LNG,
                        amrex::Real pool_area_m2,
                        amrex::Real cx, amrex::Real cy,
                        amrex::Real dt)
{
    if (spill_rate_kg_s <= 0.0 || dt <= 0.0) return;

    const auto& dx = geom_lng.CellSize();
    amrex::Real pool_radius  = std::sqrt(pool_area_m2 / M_PI);
    amrex::Real dh_per_cell  = (spill_rate_kg_s * dt) / (rho_LNG * pool_area_m2);

    for (amrex::MFIter mfi(lng_pool_depth, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& bx   = mfi.tilebox();
        auto       depth_arr = lng_pool_depth[mfi].array();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            // Compute cell center coordinates
            amrex::Real x_cell = geom_lng.ProbLo(0) + (i + 0.5) * dx[0];
            amrex::Real y_cell = geom_lng.ProbLo(1) + (j + 0.5) * dx[1];

            // Distance from pool center
            amrex::Real dx_pool = x_cell - cx;
            amrex::Real dy_pool = y_cell - cy;
            amrex::Real r_pool  = std::sqrt(dx_pool*dx_pool + dy_pool*dy_pool);

            // Add source if inside pool radius
            if (r_pool <= pool_radius) {
                depth_arr(i, j, k) += dh_per_cell;
            }
        });
    }

    lng_pool_depth.FillBoundary();
}

void deplete_pool_from_evaporation(amrex::MultiFab& lng_pool_depth,
                                   const amrex::MultiFab& lng_evap_flux,
                                   amrex::Real rho_LNG,
                                   amrex::Real dt,
                                   bool lng_debug)
{
    if (dt <= 0.0) return;

    for (amrex::MFIter mfi(lng_pool_depth, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& bx    = mfi.tilebox();
        auto        depth_arr = lng_pool_depth[mfi].array();
        const auto  evap_arr  = lng_evap_flux[mfi].array();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real F_evap  = evap_arr(i, j, k);
            amrex::Real dh_evap = (F_evap / rho_LNG) * dt;
            depth_arr(i, j, k)  = amrex::max(0.0, depth_arr(i, j, k) - dh_evap);
        });
    }

    lng_pool_depth.FillBoundary();

    if (lng_debug) {
        amrex::Real depth_max = lng_pool_depth.max(0);
        amrex::Real depth_min = lng_pool_depth.min(0);
        amrex::Print() << "[LNG DEBUG] Phase 2: pool_depth_max=" << depth_max << " m"
                       << "  pool_depth_min=" << depth_min << " m\n";
    }
}

amrex::Real compute_pool_mass(const amrex::MultiFab& lng_pool_depth,
                               const amrex::Geometry& geom_lng,
                               amrex::Real rho_LNG)
{
    const auto& dx         = geom_lng.CellSize();
    amrex::Real cell_area  = dx[0] * dx[1];
    amrex::Real depth_sum  = lng_pool_depth.sum(0);
    return depth_sum * rho_LNG * cell_area;
}

amrex::Real compute_pool_area(const amrex::MultiFab& lng_pool_mask,
                               const amrex::Geometry& geom_lng)
{
    const auto& dx        = geom_lng.CellSize();
    amrex::Real cell_area = dx[0] * dx[1];
    amrex::Real mask_sum  = lng_pool_mask.sum(0);
    return mask_sum * cell_area;
}