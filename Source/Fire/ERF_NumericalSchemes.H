#ifndef ERF_NUMERICAL_SCHEMES_H
#define ERF_NUMERICAL_SCHEMES_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <cmath>

/**
 * @file ERF_NumericalSchemes.H
 * @brief Numerical schemes for level-set advection: WENO3 and Godunov norm gradient
 *
 * Ported from wildfire_levelset/src/numerical_schemes.H
 */

namespace fire_levelset {

/**
 * @brief WENO-3 reconstruction on grid point (i,j) from left stencil (i-2,i-1,i,i+1)
 *
 * Reconstructs the third-order accurate flux on the right side of cell (i-1),
 * which is the left side of cell i.
 *
 * Note: v0 (value at i-2) is accepted for interface consistency but is not used
 * in the 2-stencil WENO3 formulation.
 *
 * @param v0  Function value at i-2 (unused in current formulation)
 * @param v1  Function value at i-1
 * @param v2  Function value at i
 * @param v3  Function value at i+1
 * @return Reconstructed value on the right of i-1 (left of i)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real weno3_reconstruct(amrex::Real /*v0*/, amrex::Real v1,
                               amrex::Real v2, amrex::Real v3) noexcept
{
    constexpr amrex::Real eps = 1.0e-6;

    // Smoothness indicators
    amrex::Real IS0 = (v2 - v1) * (v2 - v1);
    amrex::Real IS1 = (v3 - v2) * (v3 - v2);

    // Weights
    amrex::Real w0 = 1.0 / (3.0 * (eps + IS0) * (eps + IS0));
    amrex::Real w1 = 2.0 / (3.0 * (eps + IS1) * (eps + IS1));
    amrex::Real w_sum = w0 + w1;

    // Normalize weights
    w0 /= w_sum;
    w1 /= w_sum;

    // ENO stencils
    amrex::Real u0 = (v1 + v2) / 2.0;
    amrex::Real u1 = (v2 + v3) / 2.0;

    // Weighted combination
    return w0 * u0 + w1 * u1;
}

/**
 * @brief Level-set stencil that does not cross a non-burnable wall
 *
 * With `on` set, a stencil point inside the mask returns the value of the
 * centre cell instead of its own: the level set is extrapolated into the
 * wall with zero normal gradient, so a masked cell's untouched (and, next to
 * a burned cell, very different) value never enters a derivative. Without it
 * the one-sided difference toward the wall dominates the Godunov norm, the
 * wall-adjacent cells burn down far faster than the spread rate, and the
 * flank along the wall runs about 20% fast. Off (`on = false`) the stencil is
 * the plain neighbour value, so every existing result is unchanged.
 */
struct WallStencil {
    amrex::Array4<const amrex::Real> mask;
    bool on = false;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool masked(int i, int j, int k) const noexcept {
        return on && mask(i, j, k) > amrex::Real(0.5);
    }

    /// phi at (i+di, j+dj, k) as seen from open cell (i, j, k)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real at(const amrex::Array4<const amrex::Real>& phi,
                   int i, int j, int k, int di, int dj) const noexcept {
        if (on && mask(i + di, j + dj, k) > amrex::Real(0.5)) { return phi(i, j, k); }
        return phi(i + di, j + dj, k);
    }
};

/// Wall stencil for one fab: active only when a mask is given and the option is on.
inline WallStencil make_wall_stencil(const amrex::MultiFab* mask, bool on,
                                     const amrex::MFIter& mfi)
{
    WallStencil w;
    w.on = on && (mask != nullptr);
    if (w.on) { w.mask = mask->const_array(mfi); }
    return w;
}

/**
 * @brief Godunov norm of the gradient, with optional terrain arc-length correction
 *
 * Computes |∇φ| using Godunov's scheme on a staggered grid.
 * With slope_x and slope_y parameters, applies terrain arc-length correction:
 *   |∇φ|_corrected = sqrt(φ_x^2 + φ_y^2 + (φ_x * slope_x + φ_y * slope_y)^2)
 *
 * @param[in] phi         Level-set field (assumed with sufficient ghosts for stencil)
 * @param[in] dx, dy      Cell sizes [m]
 * @param[in] i, j, k     Cell indices
 * @param[in] slope_x     Terrain slope dz/dx (default 0)
 * @param[in] slope_y     Terrain slope dz/dy (default 0)
 * @return Godunov norm of the gradient |∇φ|
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real godunov_norm_grad_phi(const amrex::Array4<const amrex::Real>& phi,
                                   amrex::Real dx, amrex::Real dy,
                                   int i, int j, int k,
                                   amrex::Real slope_x = 0.0,
                                   amrex::Real slope_y = 0.0,
                                   const WallStencil& wall = WallStencil{}) noexcept
{
    // One-sided derivatives (Godunov scheme). Through a wall stencil a masked
    // neighbour carries the centre value, so no difference crosses a wall.
    amrex::Real dxm = (phi(i, j, k) - wall.at(phi, i, j, k, -1, 0)) / dx;
    amrex::Real dxp = (wall.at(phi, i, j, k, 1, 0) - phi(i, j, k)) / dx;
    amrex::Real dym = (phi(i, j, k) - wall.at(phi, i, j, k, 0, -1)) / dy;
    amrex::Real dyp = (wall.at(phi, i, j, k, 0, 1) - phi(i, j, k)) / dy;

    // Take max of upwind/downwind (Godunov)
    amrex::Real phi_x = (dxm * dxm > dxp * dxp) ? dxm : dxp;
    amrex::Real phi_y = (dym * dym > dyp * dyp) ? dym : dyp;

    // Godunov norm in map coordinates
    amrex::Real grad_sq = phi_x * phi_x + phi_y * phi_y;

    // Terrain projection. The front travels at R along the terrain surface, so
    // its map-view speed is R cos(theta) with theta the slope angle along the
    // spread direction n = grad(phi)/|grad(phi)|:
    //
    //   |grad(phi)|_surface = |grad(phi)| / sqrt(1 + (s . n)^2)
    //                       = |grad(phi)|^2 / sqrt(|grad(phi)|^2 + (s . grad(phi))^2)
    //
    // which reduces exactly to the component-wise arc-length spacing
    // dx sqrt(1 + s_x^2) in one dimension, and to |grad(phi)| on flat terrain.
    const amrex::Real s_dot_grad = phi_x * slope_x + phi_y * slope_y;
    const amrex::Real denom = std::sqrt(amrex::max(grad_sq + s_dot_grad * s_dot_grad,
                                                   amrex::Real(1.0e-14)));

    return grad_sq / denom;
}

} // namespace fire_levelset

#endif