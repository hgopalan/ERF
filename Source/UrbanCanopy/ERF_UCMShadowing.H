/**
 * @file ERF_UCMShadowing.H
 * @brief Sky-view-factor (SVF) computation for canyon shadowing
 *
 * Phase 2.4: Compute per-cell sky view factors (SVF) for walls and roads
 * to model shadowing from building geometry. SVF reduces the shortwave
 * radiation absorbed on shaded facets in proportion to the canyon aspect ratio.
 *
 * Physics (Kusaka et al. 2001, Canyon Urban Heat Island Model):
 *   - aspect = H_bldg / max(W_road, 1.0e-6)
 *   - SVF_road(i,j) = sqrt(aspect^2 + 1) - aspect               [Kusaka eq. 24]
 *   - SVF_wall(i,j) = 0.5 * (aspect + 1 - sqrt(aspect^2 + 1)) / aspect   [Kusaka eq. 25]
 *   - SVF_roof(i,j) = 1.0  (always unshaded from above)
 *
 * shadow_frac = 1 - SVF is then used to reduce SW_absorbed on wall and road:
 *   - SW_absorbed_road = (1 - albedo_road) * SW_down * SVF_road
 *   - SW_absorbed_wall = (1 - albedo_wall) * SW_down * SVF_wall
 *   - SW_absorbed_roof = (1 - albedo_roof) * SW_down * 1.0  [unchanged]
 *
 * Reference:
 *   Kusaka, H., Kondo, H., Kikegawa, Y., & Kimura, F. (2001).
 *   A simple single-layer urban canopy model for atmospheric models:
 *   Comparison with multi-layer and slab models. Boundary-Layer Meteorology, 101, 329-358.
 *
 * Fire-pattern compliance:
 *   - setVal(0.0, ncomp, ngrow) at top of compute function
 *   - OVERWRITE (=) inside kernel, not accumulate (+=)
 *   - [=] AMREX_GPU_DEVICE kernel with is_urban guard
 *   - IOProcessor() used only for Print(), no collectives inside kernels
 */

#ifndef ERF_UCM_SHADOWING_H
#define ERF_UCM_SHADOWING_H

#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>

/**
 * @brief Compute per-cell sky view factors (SVF) for wall and road
 *
 * Implements canyon shadowing model based on Kusaka et al. (2001) equations 24-25.
 * Uses canyon aspect ratio to determine what fraction of sky each facet sees.
 * Higher SVF = less shadowing = more direct SW absorption.
 * Lower SVF = more shadowing = less direct SW absorption.
 *
 * Fire-pattern implementation:
 *   1. setVal(0.0, ncomp, ngrow) on all three SVF fields
 *   2. Iterate MFIter over H_bldg
 *   3. Extract arrays by value ([=] AMREX_GPU_DEVICE)
 *   4. Guard with is_urban(i,j,0) == 0 check; return early if not urban
 *   5. Compute aspect = H_bldg / max(W_road, 1e-6)
 *   6. Compute SVF_road, SVF_wall using Kusaka formulas
 *   7. Set SVF_roof = 1.0 (always unshaded)
 *   8. OVERWRITE arrays with computed values
 *
 * @param SVF_wall_out Output: wall sky view factor [-]. Phase 2.4 allocated MultiFab.
 * @param SVF_road_out Output: road sky view factor [-]. Phase 2.4 allocated MultiFab.
 * @param SVF_roof_out Output: roof sky view factor [-]. Phase 2.4 allocated MultiFab (always 1.0).
 * @param H_bldg Input: building height [m]. Phase 2.2+ per-cell heterogeneous.
 * @param W_road Input: road width [m]. Phase 2.2+ per-cell heterogeneous.
 * @param is_urban Input: urban mask [0 or 1]. Phase 1.2+ per-cell heterogeneous.
 * @param lev Current AMR level (typically 0 for UCM).
 * @param debug If true, print min/max diagnostics to stdout (gated by IOProcessor).
 */
inline void compute_sky_view_factors(
    amrex::MultiFab& SVF_wall_out,
    amrex::MultiFab& SVF_road_out,
    amrex::MultiFab& SVF_roof_out,
    const amrex::MultiFab& H_bldg,
    const amrex::MultiFab& W_road,
    const amrex::iMultiFab& is_urban,
    int lev = 0,
    bool debug = false)
{
    using namespace amrex;

    // Fire-pattern: zero-initialize output fields (setVal with ngrow)
    // This ensures any leftover state from a previous stage does not leak through.
    SVF_wall_out.setVal(0.0, 0, 1, SVF_wall_out.nGrowVect());
    SVF_road_out.setVal(0.0, 0, 1, SVF_road_out.nGrowVect());
    SVF_roof_out.setVal(0.0, 0, 1, SVF_roof_out.nGrowVect());

    // Compute SVF_wall, SVF_road per cell
    for (MFIter mfi(H_bldg, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const& h_a = H_bldg.const_array(mfi);
        auto const& w_a = W_road.const_array(mfi);
        auto const& is_urb_a = is_urban.const_array(mfi);

        auto svf_wall_a = SVF_wall_out.array(mfi);
        auto svf_road_a = SVF_road_out.array(mfi);
        auto svf_roof_a = SVF_roof_out.array(mfi);

        const Box& bx = mfi.tilebox();

        // Fire-pattern GPU kernel: [=] capture, AMREX_GPU_DEVICE, early return for non-urban
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // Fire-pattern guard: skip non-urban cells early
            if (is_urb_a(i, j, 0) == 0) {
                svf_wall_a(i, j, 0) = 0.0;
                svf_road_a(i, j, 0) = 0.0;
                svf_roof_a(i, j, 0) = 0.0;
                return;
            }

            // Kusaka 2001 canyon aspect ratio
            const Real H = h_a(i, j, 0);
            const Real W = amrex::max(w_a(i, j, 0), 1.0e-6);
            const Real aspect = H / W;

            // Kusaka eq. 24: SVF for road surface (canyon floor)
            // SVF_road = sqrt(aspect^2 + 1) - aspect
            const Real sqrt_term = std::sqrt(aspect * aspect + 1.0);
            const Real svf_rd = sqrt_term - aspect;

            // Kusaka eq. 25: SVF for wall surface (canyon sidewall)
            // SVF_wall = 0.5 * (aspect + 1 - sqrt(aspect^2 + 1)) / aspect
            // = 0.5 * (aspect + 1 - sqrt_term) / aspect
            const Real svf_wl = (aspect > 1.0e-8)
                              ? (0.5 * (aspect + 1.0 - sqrt_term) / aspect)
                              : 0.5;

            // SVF_roof = 1.0 always (unshaded from above)
            const Real svf_rf = 1.0;

            // Fire-pattern: OVERWRITE with computed values (not accumulate with +=)
            svf_wall_a(i, j, 0) = svf_wl;
            svf_road_a(i, j, 0) = svf_rd;
            svf_roof_a(i, j, 0) = svf_rf;
        });
    }

    // Debug trace: optional per-cell diagnostics
    if (debug) {
        // Collectives on ALL ranks (must be outside IOProcessor guard)
        Real svf_wall_min = SVF_wall_out.min(0, 0);
        Real svf_wall_max = SVF_wall_out.max(0, 0);
        Real svf_road_min = SVF_road_out.min(0, 0);
        Real svf_road_max = SVF_road_out.max(0, 0);

        if (ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][2.4][compute_sky_view_factors] SVF computed on level "
                          << lev << ":\n"
                          << "  SVF_wall range: [" << svf_wall_min << ", " << svf_wall_max << "]\n"
                          << "  SVF_road range: [" << svf_road_min << ", " << svf_road_max << "]\n";
        }
    }
}

#endif // ERF_UCM_SHADOWING_H
