/**
 * @file ERF_UCMAtmAggregation.H
 * @brief Morphology aggregation from UCM 2D slab to ATM 3D grid (Phase 2.5)
 *
 * Implements scale-aware source aggregation by computing per-ATM-cell morphology
 * statistics (urban fraction, building height moments, area densities) from the
 * fine UCM 2D slab to the coarser ATM 3D grid.
 *
 * **Physics (per ATM cell covering grid_ratio x grid_ratio UCM cells):**
 *
 * \f[
 *   f_{\text{urb}} = \frac{\sum_{i,j} \text{is_urban}(i,j)}{\text{n_cells}}
 * \f]
 *
 * \f[
 *   H_{\text{mean}} = \frac{\sum_{i,j} H_{\text{bldg}}(i,j) \cdot \text{is_urban}(i,j)}{\sum_{i,j} \text{is_urban}(i,j)}
 * \f]
 *
 * \f[
 *   H_{\text{std}} = \sqrt{\frac{\sum_{i,j} (H_{\text{bldg}}(i,j) - H_{\text{mean}})^2 \cdot \text{is_urban}(i,j)}{\sum_{i,j} \text{is_urban}(i,j)}}
 * \f]
 *
 * \f[
 *   \lambda_p = \frac{\sum_{i,j} \text{plan\_area\_frac}(i,j) \cdot \text{is_urban}(i,j)}{\sum_{i,j} \text{is_urban}(i,j)}
 * \f]
 *
 * \f[
 *   \lambda_f = \frac{\sum_{i,j} 2 H_{\text{bldg}}(i,j) W_{\text{road}}(i,j) \cdot \text{is_urban}(i,j)}{W_{\text{road}} W_{\text{roof}} \cdot \text{n_cells}}
 * \f]
 *
 * where n_cells = grid_ratio² for horizontal coarsening at factor grid_ratio.
 *
 * **Design (Fire pattern compliance Phase 2.5):**
 * - Collective min/max computed globally, printed on IO rank only (no deadlock).
 * - GPU kernels are stateless: `[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept`
 * - All outputs are set to 0.0 at entry with full ghost cells (persistent source).
 * - Aggregates written with `=` (not `+=`) inside kernel to avoid RK-stage drift.
 * - Early is_urban guard prevents OOB access on non-urban cells.
 *
 * **References:**
 * - Source/Dust/ coarsening pattern (averaging with masks)
 * - Source/Fire/ERF_FireAtmCoupling.H (exponential injection layout)
 * - WRF Single-Layer Urban Canopy Model (Chen et al., 2011)
 */

#ifndef ERF_UCM_ATM_AGGREGATION_H
#define ERF_UCM_ATM_AGGREGATION_H

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>
#include <cmath>

/**
 * @brief Compute per-ATM-cell morphology aggregates from UCM 2D slab.
 *
 * For each ATM cell (covering grid_ratio² UCM cells), aggregates:
 * - Urban fraction (count of urban cells / total)
 * - Mean building height (area-weighted average over urban cells)
 * - Std dev of building height (variance over urban cells)
 * - Mean plan-area density (building footprint ratio)
 * - Mean frontal-area density (wall area ratio)
 *
 * Aggregates are stored on the ATM 3D grid (one value per ATM cell, constant in z).
 *
 * **Debug trace format (IO rank only):**
 * ```
 * [UCM][2.5][aggregate_ucm_morphology_to_atm]
 *   grid_ratio={ratio}
 *   f_urb: min={min} max={max}
 *   H_bldg_mean: min={min} max={max} [m]
 *   H_bldg_std: min={min} max={max} [m]
 *   lambda_p: min={min} max={max}
 *   lambda_f: min={min} max={max}
 * ```
 *
 * @param[out] f_urb_atm             Urban fraction per ATM cell [0,1]
 * @param[out] H_bldg_mean_atm       Mean building height per ATM cell [m]
 * @param[out] H_bldg_std_atm        Std dev of building height per ATM cell [m]
 * @param[out] lambda_p_atm          Mean plan-area density per ATM cell [-]
 * @param[out] lambda_f_atm          Mean frontal-area density per ATM cell [-]
 * @param[in]  H_bldg_ucm            Building height on UCM grid [m]
 * @param[in]  W_road_ucm            Road width on UCM grid [m]
 * @param[in]  plan_area_frac_ucm    Plan-area density on UCM grid (building footprint fraction)
 * @param[in]  is_urban_ucm          Urban mask on UCM grid (1 = urban, 0 = non-urban)
 * @param[in]  geom_ucm              UCM geometry
 * @param[in]  geom_atm              ATM geometry
 * @param[in]  grid_ratio            Refinement factor (UCM vs ATM); typically 2 or 1
 * @param[in]  ucm_debug             Print debug info (default false)
 * @param[in]  lev                   AMR level (default 0)
 *
 * @note All outputs are on ATM 3D grid, one value per ATM cell across all z.
 * @note Urban fraction can be < 1 when mixed urban/non-urban exists in ATM cell.
 * @note When f_urb = 0 (no urban cells), mean/std are set to 0.0 (safe division).
 * @note GPU-safe and respects is_urban mask (no access to non-urban cells).
 */
void aggregate_ucm_morphology_to_atm(
    amrex::MultiFab&           f_urb_atm,
    amrex::MultiFab&           H_bldg_mean_atm,
    amrex::MultiFab&           H_bldg_std_atm,
    amrex::MultiFab&           lambda_p_atm,
    amrex::MultiFab&           lambda_f_atm,
    const amrex::MultiFab&     H_bldg_ucm,
    const amrex::MultiFab&     W_road_ucm,
    const amrex::MultiFab&     plan_area_frac_ucm,
    const amrex::iMultiFab&    is_urban_ucm,
    const amrex::Geometry&     geom_ucm,
    const amrex::Geometry&     geom_atm,
    int                        grid_ratio,
    int                        lev,
    bool                       ucm_debug)
{
    using namespace amrex;

    // Zero all outputs with ghost cells (persistent source pattern)
    f_urb_atm.setVal(0.0, 0, 1, f_urb_atm.nGrow());
    H_bldg_mean_atm.setVal(0.0, 0, 1, H_bldg_mean_atm.nGrow());
    H_bldg_std_atm.setVal(0.0, 0, 1, H_bldg_std_atm.nGrow());
    lambda_p_atm.setVal(0.0, 0, 1, lambda_p_atm.nGrow());
    lambda_f_atm.setVal(0.0, 0, 1, lambda_f_atm.nGrow());

    // Get ATM domain to know the k level range
    const int klo_atm = geom_atm.Domain().smallEnd(2);

    // Compute aggregates for each ATM cell
    for (MFIter mfi(f_urb_atm, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        // ATM cell box (to loop over i,j)
        const Box& atm_box = mfi.tilebox();

        auto f_urb_a         = f_urb_atm.array(mfi);
        auto H_mean_a        = H_bldg_mean_atm.array(mfi);
        auto H_std_a         = H_bldg_std_atm.array(mfi);
        auto lambda_p_a      = lambda_p_atm.array(mfi);
        auto lambda_f_a      = lambda_f_atm.array(mfi);

        auto const H_bldg_a       = H_bldg_ucm.const_array(mfi);
        auto const W_road_a       = W_road_ucm.const_array(mfi);
        auto const plan_area_a    = plan_area_frac_ucm.const_array(mfi);
        auto const is_urban_a     = is_urban_ucm.const_array(mfi);

        // Loop over ATM cells (i, j) at k=klo_atm
        ParallelFor(atm_box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // Aggregate over grid_ratio × grid_ratio UCM cells
            int n_cells = grid_ratio * grid_ratio;
            int n_urban = 0;
            Real sum_H = 0.0;
            Real sum_H_sq = 0.0;  // for std dev computation
            Real sum_lambda_p = 0.0;
            Real sum_lambda_f = 0.0;
            Real mean_W_road = 0.0;

            // First pass: count urban cells and compute sums
            for (int jj = 0; jj < grid_ratio; ++jj) {
                for (int ii = 0; ii < grid_ratio; ++ii) {
                    int i_ucm = i * grid_ratio + ii;
                    int j_ucm = j * grid_ratio + jj;

                    if (is_urban_a(i_ucm, j_ucm, 0) == 0) continue;

                    n_urban++;
                    Real H = H_bldg_a(i_ucm, j_ucm, 0);
                    Real W_road = W_road_a(i_ucm, j_ucm, 0);
                    Real plan_area = plan_area_a(i_ucm, j_ucm, 0);

                    sum_H += H;
                    sum_lambda_p += plan_area;
                    mean_W_road += W_road;  // for frontal area computation
                }}

            // Urban fraction: n_urban / n_cells
            f_urb_a(i, j, k) = static_cast<Real>(n_urban) / static_cast<Real>(n_cells);

            // If no urban cells, set aggregates to 0.0 and return
            if (n_urban == 0) {
                H_mean_a(i, j, k) = 0.0;
                H_std_a(i, j, k) = 0.0;
                lambda_p_a(i, j, k) = 0.0;
                lambda_f_a(i, j, k) = 0.0;
                return;
            }

            // Mean building height and mean plan-area density
            Real H_mean = sum_H / static_cast<Real>(n_urban);
            Real lambda_p_mean = sum_lambda_p / static_cast<Real>(n_urban);
            mean_W_road /= static_cast<Real>(n_urban);

            H_mean_a(i, j, k) = H_mean;
            lambda_p_a(i, j, k) = lambda_p_mean;

            // Second pass: compute std dev and frontal area density
            for (int jj = 0; jj < grid_ratio; ++jj) {
                for (int ii = 0; ii < grid_ratio; ++ii) {
                    int i_ucm = i * grid_ratio + ii;
                    int j_ucm = j * grid_ratio + jj;

                    if (is_urban_a(i_ucm, j_ucm, 0) == 0) continue;

                    Real H = H_bldg_a(i_ucm, j_ucm, 0);
                    Real W_road = W_road_a(i_ucm, j_ucm, 0);

                    // Accumulate squared deviations
                    Real diff = H - H_mean;
                    sum_H_sq += diff * diff;

                    // Frontal area: 2 * H * W_road per cell, normalized
                    sum_lambda_f += 2.0 * H * W_road;
                }}

            // Std dev of building height
            Real variance = sum_H_sq / static_cast<Real>(n_urban);
            H_std_a(i, j, k) = std::sqrt(std::max(0.0, variance));

            // Mean frontal-area density (wall area ratio)
            // lambda_f = sum(2*H*W_road) / (W_road * W_roof * n_cells)
            // Approximate: W_roof = W_road (canyon geometry), so denominator = W_road^2 * n_cells
            Real W_road_safe = std::max(mean_W_road, 1.0e-6);
            lambda_f_a(i, j, k) = sum_lambda_f / (W_road_safe * W_road_safe * static_cast<Real>(n_cells));
        });
    }

    // Compute global min/max for diagnostics (collectives, not inside IOProcessor)
    Real f_urb_min = f_urb_atm.min(0, 0);
    Real f_urb_max = f_urb_atm.max(0, 0);
    Real H_mean_min = H_bldg_mean_atm.min(0, 0);
    Real H_mean_max = H_bldg_mean_atm.max(0, 0);
    Real H_std_min = H_bldg_std_atm.min(0, 0);
    Real H_std_max = H_bldg_std_atm.max(0, 0);
    Real lambda_p_min = lambda_p_atm.min(0, 0);
    Real lambda_p_max = lambda_p_atm.max(0, 0);
    Real lambda_f_min = lambda_f_atm.min(0, 0);
    Real lambda_f_max = lambda_f_atm.max(0, 0);

    if (ucm_debug && ParallelDescriptor::IOProcessor()) {
        Print() << "[UCM][2.5][aggregate_ucm_morphology_to_atm]\n"
                << "  grid_ratio=" << grid_ratio << "\n"
                << "  f_urb: min=" << f_urb_min << " max=" << f_urb_max << "\n"
                << "  H_bldg_mean: min=" << H_mean_min << " max=" << H_mean_max << " [m]\n"
                << "  H_bldg_std: min=" << H_std_min << " max=" << H_std_max << " [m]\n"
                << "  lambda_p: min=" << lambda_p_min << " max=" << lambda_p_max << "\n"
                << "  lambda_f: min=" << lambda_f_min << " max=" << lambda_f_max << "\n";
    }
}

#endif  // ERF_UCM_ATM_AGGREGATION_H
