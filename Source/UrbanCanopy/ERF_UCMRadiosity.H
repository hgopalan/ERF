/**
 * @file ERF_UCMRadiosity.H
 * @brief Header-only SW multi-bounce radiosity solver (Phase 5.1b)
 *
 * Computes infinite-bounce SW reflection between wall_A, wall_B, and road
 * via a 2×2 Cramer solve. Roof unshaded (F_roof_sky=1); LW multi-bounce
 * deferred to Phase 5.1c.
 *
 * Physics (Cramer's rule):
 *   Given SW0_wall, SW0_road, α_w, α_r, F_ww, F_wr, F_rw:
 *   Solve M · [B_w, B_r]ᵀ = rhs for absorbed radiances B_w, B_r.
 *   Return effective SW incident:
 *     SW_wall_eff = SW0_wall + F_ww·B_w + F_wr·B_r
 *     SW_road_eff = SW0_road + 2·F_rw·B_w
 *
 * Sanity:
 *   1. α=0 → B=0 → SW_eff = SW0 (bit-identical to single-bounce)
 *   2. F=0 (isolated) → SW_eff = SW0
 *   3. SW_eff ≥ SW0 always (enhancement)
 *   4. det > 0 for physical inputs
 */

#ifndef ERF_UCM_RADIOSITY_H
#define ERF_UCM_RADIOSITY_H

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

/**
 * @brief 2×2 Cramer solver for SW multi-bounce between walls and road
 *
 * @param SW0_wall    Single-bounce SW incident on walls [W/m²]
 * @param SW0_road    Single-bounce SW incident on road [W/m²]
 * @param alpha_w     Wall SW albedo [-]
 * @param alpha_r     Road SW albedo [-]
 * @param F_ww        Wall-to-wall view factor [-]
 * @param F_wr        Wall-to-road view factor [-]
 * @param F_rw        Road-to-wall view factor [-]
 * @param SW_wall_eff Output: effective SW incident on walls (preserves SEB contract) [W/m²]
 * @param SW_road_eff Output: effective SW incident on road [W/m²]
 *
 * GPU-safe: AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE with value captures only.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void sw_radiosity_multi_bounce(amrex::Real SW0_wall, amrex::Real SW0_road,
                               amrex::Real alpha_w, amrex::Real alpha_r,
                               amrex::Real F_ww, amrex::Real F_wr, amrex::Real F_rw,
                               amrex::Real& SW_wall_eff, amrex::Real& SW_road_eff)
{
    // Determinant of system matrix M
    const amrex::Real det = (1.0 - alpha_w * F_ww) + 2.0 * alpha_w * alpha_r * F_wr * F_rw;
    
    // Guard against singular/degenerate system
    const amrex::Real det_safe = amrex::max(det, amrex::Real(1.0e-6));
    
    // Absorbed radiances via Cramer's rule
    const amrex::Real B_w = (alpha_w * SW0_wall + alpha_w * F_wr * alpha_r * SW0_road) / det_safe;
    const amrex::Real B_r = (2.0 * alpha_r * F_rw * alpha_w * SW0_wall
                             + alpha_r * SW0_road * (1.0 - alpha_w * F_ww)) / det_safe;
    
    // Effective SW incident (includes multi-bounce enhancement)
    SW_wall_eff = SW0_wall + F_ww * B_w + F_wr * B_r;
    SW_road_eff = SW0_road + 2.0 * F_rw * B_w;
}

#endif // ERF_UCM_RADIOSITY_H
