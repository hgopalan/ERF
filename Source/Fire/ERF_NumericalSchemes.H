#ifndef ERF_NUMERICAL_SCHEMES_H
#define ERF_NUMERICAL_SCHEMES_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <cmath>

/**
 * @file ERF_NumericalSchemes.H
 * @brief Numerical schemes for level-set advection: the Godunov norm of the
 *        gradient, from first-order or HJ-WENO5-Z one-sided derivatives
 *
 * Ported from wildfire_levelset/src/numerical_schemes.H
 */

namespace fire_levelset {


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


/// Gradient schemes for the level-set one-sided derivatives
inline constexpr int LEVELSET_GRAD_UPWIND1      = 0;   ///< first-order one-sided differences everywhere
inline constexpr int LEVELSET_GRAD_WENO5Z       = 1;   ///< HJ-WENO5-Z everywhere
inline constexpr int LEVELSET_GRAD_WENO5Z_FRONT = 2;   ///< HJ-WENO5-Z within `band` of the front, first order elsewhere

/**
 * @brief Which one-sided derivatives feed the Godunov Hamiltonian, and the
 *        artificial viscosity near the front
 *
 * The WRF-Fire / CFBM default (Munoz-Esparza et al. 2018; Jimenez y Munoz
 * et al. 2026) is the hybrid: fifth-order WENO where the level set is
 * within a few cells of the front, where its value matters, and the cheap
 * first-order difference elsewhere. `band` is the half-width of that
 * region in the units of phi (metres on the signed-distance path); a
 * non-positive band with the hybrid scheme means WENO everywhere.
 *
 * WRF-Fire also carries two artificial viscosities, one within a few cells
 * of the front (fire_viscosity_bg, fire_viscosity_ngp) and one elsewhere
 * (fire_viscosity), blended over a fraction of the advection band
 * (fire_viscosity_band). `eps_visc_front` is the near-front value; the
 * coefficient is eps_visc_front for |phi| <= visc_d0, the background value
 * for |phi| >= visc_d1, and linear in between. A negative eps_visc_front
 * keeps the single background value everywhere.
 */
struct LevelSetGradient {
    int         scheme         = LEVELSET_GRAD_UPWIND1;
    amrex::Real band           = -1.0;
    amrex::Real eps_visc_front = -1.0;
    amrex::Real visc_d0        = 0.0;
    amrex::Real visc_d1        = 0.0;

    /// Artificial viscosity at a cell with level-set value phi, given the
    /// background coefficient eps_bg
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real eps_at(amrex::Real phi, amrex::Real eps_bg) const noexcept {
        if (eps_visc_front < amrex::Real(0.0)) { return eps_bg; }
        const amrex::Real a = std::abs(phi);
        if (a <= visc_d0) { return eps_visc_front; }
        if (a >= visc_d1 || visc_d1 <= visc_d0) { return eps_bg; }
        const amrex::Real w = (a - visc_d0) / (visc_d1 - visc_d0);
        return eps_visc_front + w * (eps_bg - eps_visc_front);
    }
};

/**
 * @brief HJ-WENO5-Z one-sided derivative from five consecutive divided differences
 *
 * Jiang and Peng (2000) Hamilton-Jacobi WENO: v1..v5 are the first divided
 * differences on the side being reconstructed, ordered away from the
 * cell. For the left derivative at i they are D^-_{i-2}, D^-_{i-1}, D^-_i,
 * D^-_{i+1}, D^-_{i+2} with D^-_m = (phi_m - phi_{m-1})/dx; for the right
 * derivative the same five taken as D^+_{i+2}, ..., D^+_{i-2}. The weights
 * are the Borges et al. (2008) Z form, which keeps fifth order at smooth
 * extrema; the smoothness floor is scaled by the differences themselves
 * as in Jiang and Peng so that it is independent of the units of phi.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real hj_weno5z(amrex::Real v1, amrex::Real v2, amrex::Real v3,
                      amrex::Real v4, amrex::Real v5) noexcept
{
    const amrex::Real is0 = (13.0/12.0) * (v1 - 2.0*v2 + v3) * (v1 - 2.0*v2 + v3)
                          + 0.25 * (v1 - 4.0*v2 + 3.0*v3) * (v1 - 4.0*v2 + 3.0*v3);
    const amrex::Real is1 = (13.0/12.0) * (v2 - 2.0*v3 + v4) * (v2 - 2.0*v3 + v4)
                          + 0.25 * (v2 - v4) * (v2 - v4);
    const amrex::Real is2 = (13.0/12.0) * (v3 - 2.0*v4 + v5) * (v3 - 2.0*v4 + v5)
                          + 0.25 * (3.0*v3 - 4.0*v4 + v5) * (3.0*v3 - 4.0*v4 + v5);
    amrex::Real vmax2 = v1*v1;
    vmax2 = amrex::max(vmax2, v2*v2); vmax2 = amrex::max(vmax2, v3*v3);
    vmax2 = amrex::max(vmax2, v4*v4); vmax2 = amrex::max(vmax2, v5*v5);
    const amrex::Real eps = 1.0e-6 * vmax2 + 1.0e-99;
    const amrex::Real tau = std::abs(is0 - is2);
    amrex::Real a0 = 0.1 * (1.0 + tau / (is0 + eps));
    amrex::Real a1 = 0.6 * (1.0 + tau / (is1 + eps));
    amrex::Real a2 = 0.3 * (1.0 + tau / (is2 + eps));
    const amrex::Real asum = a0 + a1 + a2;
    a0 /= asum; a1 /= asum; a2 /= asum;
    const amrex::Real q0 = v1/3.0 - 7.0*v2/6.0 + 11.0*v3/6.0;
    const amrex::Real q1 = -v2/6.0 + 5.0*v3/6.0 + v4/3.0;
    const amrex::Real q2 = v3/3.0 + 5.0*v4/6.0 - v5/6.0;
    return a0*q0 + a1*q1 + a2*q2;
}

/**
 * @brief One-sided derivatives of phi at (i, j, k) along one axis
 *
 * First order, or HJ-WENO5-Z when `grad` asks for it (everywhere, or only
 * within `grad.band` of the front) and no point of the seven-cell stencil
 * is masked: next to a wall the stencil
 * falls back to the first-order difference through the wall stencil, so a
 * masked cell's value never enters a derivative.
 *
 * @param[in]  di, dj  Unit step along the axis (1,0) or (0,1)
 * @param[out] dm, dp  Left and right derivatives
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void one_sided_derivatives(const amrex::Array4<const amrex::Real>& phi,
                           amrex::Real h, int i, int j, int k, int di, int dj,
                           const LevelSetGradient& grad, const WallStencil& wall,
                           amrex::Real& dm, amrex::Real& dp) noexcept
{
    dm = (phi(i, j, k) - wall.at(phi, i, j, k, -di, -dj)) / h;
    dp = (wall.at(phi, i, j, k, di, dj) - phi(i, j, k)) / h;
    if (grad.scheme == LEVELSET_GRAD_UPWIND1) { return; }
    if (grad.scheme == LEVELSET_GRAD_WENO5Z_FRONT && grad.band > amrex::Real(0.0)
        && std::abs(phi(i, j, k)) > grad.band) { return; }

    bool clear = true;
    if (wall.on) {
        for (int m = -3; m <= 3; ++m) {
            if (m != 0 && wall.masked(i + m*di, j + m*dj, k)) { clear = false; }
        }
    }
    if (!clear) { return; }

    // phi at i-3 .. i+3 along the axis
    amrex::Real f[7];
    for (int m = -3; m <= 3; ++m) { f[m + 3] = phi(i + m*di, j + m*dj, k); }
    // Divided differences D_m = (f_{m+1} - f_m)/h for m = -3 .. 2
    amrex::Real d[6];
    for (int m = 0; m < 6; ++m) { d[m] = (f[m + 1] - f[m]) / h; }
    // Left: D^-_{i-2} .. D^-_{i+2} = d[0] .. d[4]; right: D^+_{i+2} .. D^+_{i-2} = d[5] .. d[1]
    dm = hj_weno5z(d[0], d[1], d[2], d[3], d[4]);
    dp = hj_weno5z(d[5], d[4], d[3], d[2], d[1]);
}

/**
 * @brief Godunov norm of the gradient, with optional terrain arc-length correction
 *
 * Computes |∇φ| with the first-order Godunov upwind scheme for an expanding
 * front (R >= 0).
 * With slope_x and slope_y parameters, applies terrain arc-length correction:
 *   |∇φ|_corrected = sqrt(φ_x^2 + φ_y^2 + (φ_x * slope_x + φ_y * slope_y)^2)
 *
 * @param[in] phi         Level-set field (assumed with sufficient ghosts for stencil)
 * @param[in] dx, dy      Cell sizes [m]
 * @param[in] i, j, k     Cell indices
 * @param[in] slope_x     Terrain slope dz/dx (default 0)
 * @param[in] slope_y     Terrain slope dz/dy (default 0)
 * @param[in] wall        Wall stencil (masked neighbours take the centre value)
 * @param[in] grad        Scheme and front band for the one-sided derivatives
 * @return Godunov norm of the gradient |∇φ|
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real godunov_norm_grad_phi(const amrex::Array4<const amrex::Real>& phi,
                                   amrex::Real dx, amrex::Real dy,
                                   int i, int j, int k,
                                   amrex::Real slope_x = 0.0,
                                   amrex::Real slope_y = 0.0,
                                   const WallStencil& wall = WallStencil{},
                                   const LevelSetGradient& grad = LevelSetGradient{}) noexcept
{
    // One-sided derivatives, first order or HJ-WENO5-Z. Through a wall
    // stencil a masked neighbour carries the centre value, so no difference
    // crosses a wall.
    amrex::Real dxm, dxp, dym, dyp;
    one_sided_derivatives(phi, dx, i, j, k, 1, 0, grad, wall, dxm, dxp);
    one_sided_derivatives(phi, dy, i, j, k, 0, 1, grad, wall, dym, dyp);

    // Godunov (Osher-Sethian) choice for an expanding front, phi_t + R|grad
    // phi| = 0 with R >= 0: the backward difference counts only where it is
    // positive and the forward difference only where it is negative, so the
    // update always reads the burned side. Taking whichever one-sided
    // difference is larger in magnitude, regardless of sign, is not the
    // Godunov scheme: wherever the profile ahead of the front is convex the
    // downwind slope wins, the scheme becomes anti-dissipative, and the front
    // runs ahead of R (the same defect the reinitialisation had; see
    // ERF_Reinitialize.H).
    const amrex::Real ax = amrex::max(dxm, amrex::Real(0.0));
    const amrex::Real bx = amrex::min(dxp, amrex::Real(0.0));
    const amrex::Real ay = amrex::max(dym, amrex::Real(0.0));
    const amrex::Real by = amrex::min(dyp, amrex::Real(0.0));
    amrex::Real phi_x = (ax * ax > bx * bx) ? ax : bx;
    amrex::Real phi_y = (ay * ay > by * by) ? ay : by;

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