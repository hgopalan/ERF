#ifndef ERF_FIREBREAK_H
#define ERF_FIREBREAK_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <string>
#include <vector>
#include <cmath>

/**
 * @file ERF_FireBreak.H
 * @brief Firebreak barriers for fire simulation
 *
 * Ported from wildfire_levelset/src/barrier_polygons.H
 *
 * Implements rectangular and circular non-burnable zones. Their cells are
 * stamped into the level-set phi as FIREBREAK_PHI_SENTINEL at initialization,
 * which only starts them unburned (both propagation paths wear the value
 * away); with erf.fire.firebreak.use_mask = true (the default) the
 * non-burnable mask, built from the same shapes by mark_firebreak_cells(),
 * holds them for the whole run.
 *
 * Reference:
 *   Finney, M.A. (1998). FARSITE: Fire Area Simulator. RMRS-RP-4.
 */

/**
 * @brief Sentinel value for firebreak cells in fire_phi.
 *
 * Positive and larger than farsite_phi_threshold (default 0.1), so a stamped
 * cell starts unburned on both paths. Nothing compares phi against it: the
 * level-set no-disc start (the domain diagonal) exceeds it on a domain whose
 * diagonal passes 1000 km, where barrier cells are a basin of lower but still
 * positive phi, held by the mask.
 */
constexpr amrex::Real FIREBREAK_PHI_SENTINEL = 1.0e6;

/**
 * @brief A firebreak is a rectangular or circular non-burnable zone.
 *
 * Cells within the barrier have phi set to FIREBREAK_PHI_SENTINEL at
 * initialization; with firebreak.use_mask (the default) the non-burnable
 * mask keeps them unburned for the duration of the simulation.
 *
 * Reference:
 *   Finney, M.A. (1998). FARSITE: Fire Area Simulator. RMRS-RP-4.
 */
struct FireBreak {
    std::string type   = "rect";  ///< "rect" or "circle"
    amrex::Real x_lo   = 0.0;     ///< rectangle: west edge [m]
    amrex::Real y_lo   = 0.0;     ///< rectangle: south edge [m]
    amrex::Real x_hi   = 0.0;     ///< rectangle: east edge [m]
    amrex::Real y_hi   = 0.0;     ///< rectangle: north edge [m]
    amrex::Real cx     = 0.0;     ///< circle: centre x [m]
    amrex::Real cy     = 0.0;     ///< circle: centre y [m]
    amrex::Real radius = 0.0;     ///< circle: radius [m]
};

/**
 * @brief Set every fire-grid cell whose centre lies in a barrier to `value`.
 *
 * The one cell test of the firebreaks, shared by apply_firebreaks() (the
 * level-set sentinel) and the non-burnable mask (FireLayer::
 * build_nonburnable_mask), so the mask marks exactly the cells the sentinel
 * marks without reading them back from the level set: an unburned level set
 * is the domain diagonal from a front, which on a domain wider than the
 * sentinel's half would read as a firebreak. Cells outside all barriers are
 * not modified; ghost cells are left to the caller's fill.
 *
 * Cell centre: x = prob_lo[0] + (i + 0.5) * dx[0],
 *              y = prob_lo[1] + (j + 0.5) * dx[1]
 *
 * Rectangle check: x_lo <= x <= x_hi && y_lo <= y <= y_hi
 * Circle check: sqrt((x-cx)^2 + (y-cy)^2) <= radius
 *
 * @param[inout] mf            Field to mark (component 0)
 * @param[in]    breaks        Vector of firebreak barriers
 * @param[in]    geom          Fire grid geometry
 * @param[in]    value         Value written in barrier cells
 */
inline void mark_firebreak_cells(amrex::MultiFab& mf,
                                 const std::vector<FireBreak>& breaks,
                                 const amrex::Geometry& geom,
                                 amrex::Real value)
{
    if (breaks.empty()) { return; }

    // Copy firebreaks to device vectors for GPU access
    int n_breaks = static_cast<int>(breaks.size());

    // Allocate device vectors for barrier parameters
    // Note: use char instead of std::string — PODVector requires trivially copyable types,
    // and only the first character ('r' or 'c') is needed in the GPU kernel.
    amrex::Gpu::DeviceVector<char>        d_types(n_breaks);
    amrex::Gpu::DeviceVector<amrex::Real> d_x_lo(n_breaks);
    amrex::Gpu::DeviceVector<amrex::Real> d_y_lo(n_breaks);
    amrex::Gpu::DeviceVector<amrex::Real> d_x_hi(n_breaks);
    amrex::Gpu::DeviceVector<amrex::Real> d_y_hi(n_breaks);
    amrex::Gpu::DeviceVector<amrex::Real> d_cx(n_breaks);
    amrex::Gpu::DeviceVector<amrex::Real> d_cy(n_breaks);
    amrex::Gpu::DeviceVector<amrex::Real> d_radius(n_breaks);

    // Host copies for MPI transfer
    std::vector<char>        h_types(n_breaks);
    std::vector<amrex::Real> h_x_lo(n_breaks);
    std::vector<amrex::Real> h_y_lo(n_breaks);
    std::vector<amrex::Real> h_x_hi(n_breaks);
    std::vector<amrex::Real> h_y_hi(n_breaks);
    std::vector<amrex::Real> h_cx(n_breaks);
    std::vector<amrex::Real> h_cy(n_breaks);
    std::vector<amrex::Real> h_radius(n_breaks);

    // Copy data to host vectors
    for (int nb = 0; nb < n_breaks; ++nb) {
        h_types[nb]  = breaks[nb].type[0];  // store first char only: 'r' or 'c'
        h_x_lo[nb]   = breaks[nb].x_lo;
        h_y_lo[nb]   = breaks[nb].y_lo;
        h_x_hi[nb]   = breaks[nb].x_hi;
        h_y_hi[nb]   = breaks[nb].y_hi;
        h_cx[nb]     = breaks[nb].cx;
        h_cy[nb]     = breaks[nb].cy;
        h_radius[nb] = breaks[nb].radius;
    }

    // Copy to device
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     h_types.begin(), h_types.end(),
                     d_types.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     h_x_lo.begin(), h_x_lo.end(), d_x_lo.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     h_y_lo.begin(), h_y_lo.end(), d_y_lo.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     h_x_hi.begin(), h_x_hi.end(), d_x_hi.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     h_y_hi.begin(), h_y_hi.end(), d_y_hi.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     h_cx.begin(), h_cx.end(), d_cx.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     h_cy.begin(), h_cy.end(), d_cy.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                     h_radius.begin(), h_radius.end(), d_radius.begin());

    const auto prob_lo = geom.ProbLoArray();
    const auto dx      = geom.CellSizeArray();

    const auto* d_types_p  = d_types.data();
    const auto* d_x_lo_p   = d_x_lo.data();
    const auto* d_y_lo_p   = d_y_lo.data();
    const auto* d_x_hi_p   = d_x_hi.data();
    const auto* d_y_hi_p   = d_y_hi.data();
    const auto* d_cx_p     = d_cx.data();
    const auto* d_cy_p     = d_cy.data();
    const auto* d_radius_p = d_radius.data();

    // ParallelFor over fire grid
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto phi_arr = mf.array(mfi);  // Array4 by value — required for GPU lambda capture

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
            amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];

            // Check all barriers
            for (int nb = 0; nb < n_breaks; ++nb) {
                bool in_barrier = false;

                // Check barrier type using first char: 'r' = rect, 'c' = circle
                if (d_types_p[nb] == 'r') {
                    in_barrier = (x >= d_x_lo_p[nb] && x <= d_x_hi_p[nb] &&
                                  y >= d_y_lo_p[nb] && y <= d_y_hi_p[nb]);
                } else if (d_types_p[nb] == 'c') {
                    amrex::Real dx_c   = x - d_cx_p[nb];
                    amrex::Real dy_c   = y - d_cy_p[nb];
                    amrex::Real dist_sq = dx_c * dx_c + dy_c * dy_c;
                    in_barrier = (dist_sq <= d_radius_p[nb] * d_radius_p[nb]);
                }

                // Mark the cell if it is in the barrier
                if (in_barrier) {
                    phi_arr(i, j, k) = value;
                }
            }
        });
    }
    // The ParallelFor above is asynchronous while the Gpu::DeviceVectors free their
    // device allocations at scope exit; let the kernels finish reading them first.
    amrex::Gpu::streamSynchronize();
}

/**
 * @brief Apply firebreak barriers to fire_phi: phi = FIREBREAK_PHI_SENTINEL in
 * every cell whose centre falls within a barrier (rectangle:
 * x_lo <= x <= x_hi && y_lo <= y <= y_hi; circle: distance to the centre
 * <= radius, at the cell centre x = prob_lo + (i + 0.5) dx).
 *
 * Called once in FireLayer::initialize(), after the ignition disc and before
 * a t = 0 polygon, polyline or scheduled ignition, which min-merges its
 * distance over the barrier and so removes the sentinel there.
 */
inline void apply_firebreaks(amrex::MultiFab& fire_phi,
                             const std::vector<FireBreak>& breaks,
                             const amrex::Geometry& geom)
{
    mark_firebreak_cells(fire_phi, breaks, geom, FIREBREAK_PHI_SENTINEL);
}

#endif  // ERF_FIREBREAK_H