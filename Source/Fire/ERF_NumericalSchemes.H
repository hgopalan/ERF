#ifndef ERF_NUMERICAL_SCHEMES_H
#define ERF_NUMERICAL_SCHEMES_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

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
 * (fire_viscosity), blended linearly over fire_viscosity_band times the
 * near-front width (module_fr_fire_core.F: threshold_AV =
 * fire_viscosity_ngp dx, so 0.5 x 2 = 1 cell with WRF's defaults; ERF's
 * visc_transition_cells defaults to 2). `eps_visc_front` is the near-front
 * value; the coefficient is eps_visc_front for |phi| <= visc_d0, the background value
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
        // ramp weight from a floored width, computed before the early return:
        // with the defaults visc_d0 = visc_d1 = 0 the unselected (a-0)/0 of
        // every cell would be speculated under the fpe traps
        const amrex::Real w = (a - visc_d0) / amrex::max(visc_d1 - visc_d0, amrex::Real(1.0e-30));
        if (a >= visc_d1 || visc_d1 <= visc_d0) { return eps_bg; }
        return eps_visc_front + w * (eps_bg - eps_visc_front);
    }
};

/// Real-axis stability limit of every three-stage, third-order Runge-Kutta
/// scheme (the level-set SSP-RK3 and Wicker-Skamarock share the polynomial
/// 1 + z + z^2/2 + z^3/6 on a linear problem): stable for -2.5127 <= z <= 0.
inline constexpr amrex::Real LEVELSET_RK3_REAL_LIMIT = amrex::Real(2.5127453266183286);

/// Gain of the HJ-WENO5 one-sided derivative on the two-cell wave
/// (-1)^i, in units of 1/dx: 64/60 with its linear weights, which WENO-Z
/// keeps on that wave (its three smoothness indicators are equal there).
/// The first-order difference gives 2.
inline constexpr amrex::Real LEVELSET_WENO5_TWO_CELL_GAIN = amrex::Real(16.0 / 15.0);

/// A substep whose largest |G| (levelset_substep_gain) exceeds 1 by more
/// than this is unstable; the margin only absorbs round-off at the limit.
inline constexpr double LEVELSET_GAIN_TOL = 1.0e-9;

/**
 * @brief Largest amplification factor of one level-set substep
 *
 * A substep advances phi_t = -R (|grad phi| - eps lap phi) with three-stage
 * RK3 at dt = cfl h / R_max, h = min(dx, dy) (FireLayer::advance). Frozen
 * at the fastest cells (R = R_max) about a front of unit normal n
 * (n_x, n_y >= 0; the other quadrants are mirror images), the upwind
 * one-sided differences D^-_x, D^-_y and the five-point Laplacian act on a
 * wave exp(i (theta_x i + theta_y j)) as multipliers, and the substep
 * multiplies it by G(z) = 1 + z + z^2/2 + z^3/6 with
 *
 *   z = cfl h [ -(n_x D^-_x + n_y D^-_y) + eps (L_x + L_y) ],
 *   L_x = (2 cos theta_x - 2)/dx^2,
 *
 * D^-_x = (1 - e^{-i theta_x})/dx for first-order differences and, for
 * HJ-WENO5 with its linear weights (2, -13, 47, 27, -3)/60 on the five
 * divided differences, the matching five-point multiplier. This returns
 * max |G| over theta_x in [0, pi] (65 points), theta_y in [-pi, pi] (129)
 * and 17 directions of n from the x axis to the y axis. Host only, in
 * double whatever the build precision; about 1.4e5 evaluations.
 *
 * For first-order differences the largest |G| is always the two-cell wave
 * theta = (pi, pi), whose z is real (see levelset_viscous_stability). For
 * WENO5 the two-cell wave binds only up to cfl of about 0.55; above that a
 * complex mode of about 2.4 cells (theta near 2.58) on a front near the
 * diagonal is tighter.
 */
inline double
levelset_substep_gain (double cfl, double eps, double dx, double dy, bool weno)
{
    using C = std::complex<double>;
    constexpr int    NT = 64;
    constexpr int    NA = 16;
    constexpr double pi = 3.14159265358979323846;
    const double h = std::min(dx, dy);
    // multiplier of the backward one-sided difference along one axis, times the cell size
    auto dminus = [weno] (double t) {
        const C e1 = std::exp(C(0.0, t)) - 1.0;
        if (!weno) { return C(1.0, 0.0) - std::exp(C(0.0, -t)); }
        // v_k = D^+ phi_{i-4+k}, k = 1..5, with the linear weights
        constexpr double w[5] = {2.0/60.0, -13.0/60.0, 47.0/60.0, 27.0/60.0, -3.0/60.0};
        C s(0.0, 0.0);
        for (int k = 1; k <= 5; ++k) { s += w[k-1] * std::exp(C(0.0, (k - 4) * t)) * e1; }
        return s;
    };
    std::vector<C> Dx(NT + 1), Dy(2 * NT + 1);
    std::vector<double> Lx(NT + 1), Ly(2 * NT + 1);
    for (int a = 0; a <= NT; ++a) {
        const double t = pi * a / NT;
        Dx[a] = dminus(t) / dx;
        Lx[a] = (2.0 * std::cos(t) - 2.0) / (dx * dx);
    }
    for (int b = 0; b <= 2 * NT; ++b) {
        const double t = -pi + pi * b / NT;
        Dy[b] = dminus(t) / dy;
        Ly[b] = (2.0 * std::cos(t) - 2.0) / (dy * dy);
    }
    double g2 = 0.0;
    for (int m = 0; m <= NA; ++m) {
        const double ang = 0.5 * pi * m / NA;
        const double nx = std::cos(ang), ny = std::sin(ang);
        for (int a = 0; a <= NT; ++a) {
            for (int b = 0; b <= 2 * NT; ++b) {
                const C z = cfl * h * (-(nx * Dx[a] + ny * Dy[b]) + eps * (Lx[a] + Ly[b]));
                const C G = 1.0 + z * (1.0 + z * (0.5 + z / 6.0));
                g2 = std::max(g2, std::norm(G));
            }
        }
    }
    return std::sqrt(g2);
}

/// Stability of one level-set substep at a given cfl and artificial
/// viscosity (see levelset_viscous_stability).
struct LevelSetViscousStability {
    amrex::Real gain;      ///< largest |G| per substep over wave numbers and front directions (> 1: unstable)
    amrex::Real eps_max;   ///< largest stable eps [m] at this cfl (< 0: no eps is stable at this cfl)
    amrex::Real cfl_max;   ///< largest stable cfl at this eps
    amrex::Real eps_axis;  ///< closed form: eps [m] at which the two-cell wave on a front facing the coarser axis reaches the limit
};

/**
 * @brief Linear stability of one level-set substep with artificial viscosity
 *
 * The substep is stable when levelset_substep_gain() <= 1 + LEVELSET_GAIN_TOL. For
 * first-order differences the binding mode is the two-cell wave
 * (-1)^(i+j), whose z is real:
 *
 *   z = -cfl h [ a (n_x/dx + n_y/dy) + 4 eps (1/dx^2 + 1/dy^2) ],
 *
 * a = 2 (16/15 for HJ-WENO5), and it must stay within
 * LEVELSET_RK3_REAL_LIMIT. n_x/dx + n_y/dy runs from 1/max(dx, dy), a front
 * facing the coarser axis (eps_axis), to sqrt(1/dx^2 + 1/dy^2), the normal
 * along (1/dx, 1/dy). The scan and this closed form agree to 1e-4 in eps/dx
 * for first-order differences at every cfl; for WENO5 they agree up to cfl
 * of about 0.55 and the scan is tighter above (cfl 0.8, square cells:
 * eps/h <= 0.173 against 0.204 from the two-cell wave). With cfl = 0.4 on
 * square cells eps/h may reach 0.432 with first-order differences and
 * 0.597 with WENO5. Without viscosity the substep needs cfl <= 0.888
 * (first order) and 1.015 (WENO5) on square cells.
 *
 * eps_max and cfl_max (bisection on the scan, 50 steps each) are computed
 * only when `limits` is set.
 *
 * @param[in] cfl  erf.fire.levelset.cfl (> 0)
 * @param[in] eps  Artificial viscosity [m]
 * @param[in] dx, dy  Fire cell size [m]
 * @param[in] weno  HJ-WENO5 derivatives (true) or first-order (false)
 * @param[in] limits  Also find eps_max and cfl_max
 */
inline LevelSetViscousStability
levelset_viscous_stability (amrex::Real cfl, amrex::Real eps, amrex::Real dx, amrex::Real dy, bool weno,
                            bool limits = true)
{
    constexpr double tol = LEVELSET_GAIN_TOL;
    const double a  = weno ? double(LEVELSET_WENO5_TWO_CELL_GAIN) : 2.0;
    const double ddx = dx, ddy = dy;
    const double h  = std::min(ddx, ddy);
    const double S  = 1.0 / (ddx * ddx) + 1.0 / (ddy * ddy);
    const double lim = LEVELSET_RK3_REAL_LIMIT;
    LevelSetViscousStability r;
    r.gain     = amrex::Real(levelset_substep_gain(cfl, eps, ddx, ddy, weno));
    r.eps_axis = amrex::Real((lim / (double(cfl) * h) - a / std::max(ddx, ddy)) / (4.0 * S));
    r.eps_max  = amrex::Real(-1.0);
    r.cfl_max  = amrex::Real(-1.0);
    if (!limits) { return r; }
    // the two-cell wave on an axis-facing front is one of the scanned modes,
    // so the scan is unstable just above eps_axis and above cfl_axis
    if (levelset_substep_gain(cfl, 0.0, ddx, ddy, weno) <= 1.0 + tol) {
        double lo = 0.0, hi = 1.01 * double(r.eps_axis) + 1.0e-12 * h;
        for (int it = 0; it < 50; ++it) {
            const double m = 0.5 * (lo + hi);
            if (levelset_substep_gain(cfl, m, ddx, ddy, weno) > 1.0 + tol) { hi = m; } else { lo = m; }
        }
        r.eps_max = amrex::Real(lo);
    }
    const double e = std::max(double(eps), 0.0);
    double lo = 0.0, hi = 1.01 * lim / (h * (a / std::max(ddx, ddy) + 4.0 * e * S));
    for (int it = 0; it < 50; ++it) {
        const double m = 0.5 * (lo + hi);
        if (levelset_substep_gain(m, e, ddx, ddy, weno) > 1.0 + tol) { hi = m; } else { lo = m; }
    }
    r.cfl_max = amrex::Real(lo);
    return r;
}

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
    // The additive floor keeps a constant stencil (every v zero, so every
    // smoothness indicator zero) from dividing 0 by 0; it must be a normal
    // number of the build precision, and 1e-99 underflows to zero in single.
    constexpr amrex::Real tiny = (sizeof(amrex::Real) == 8) ? amrex::Real(1.0e-99)
                                                            : amrex::Real(1.0e-30);
    const amrex::Real eps = 1.0e-6 * vmax2 + tiny;
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
 * @brief Godunov norm of the gradient, with optional terrain projection
 *
 * Computes |∇φ| with the Godunov (Osher-Sethian) selection of the one-sided
 * derivatives for an expanding front (R >= 0); the derivatives are first
 * order or HJ-WENO5-Z depending on the scheme.
 * With slope_x and slope_y parameters the map-view norm is reduced to the
 * along-surface one (the rate R is a surface rate, the front moves R cos θ
 * in map view):
 *   |∇φ|_terrain = |∇φ|² / sqrt(|∇φ|² + (φ_x slope_x + φ_y slope_y)²)
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