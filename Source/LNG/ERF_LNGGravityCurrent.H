/**
 * @file ERF_LNGGravityCurrent.H
 * @brief Depth-averaged shallow-water gravity current PDEs for LNG dense vapor.
 * @note Phase 5
 *
 * Governing equations on the 2D slab (depth h [m], depth-averaged velocity u,v [m/s]):
 *   dh/dt   + d(hu)/dx + d(hv)/dy = F_evap/rho_vapor            [mass]
 *   d(hu)/dt + d(hu^2)/dx = -g'*h*dh/dx - Cd*u*|u|             [x-momentum]
 *   d(hv)/dt + d(hv^2)/dy = -g'*h*dh/dy - Cd*v*|v|             [y-momentum]
 *
 * where:
 *   g' = g*(rho_vapor - rho_air)/rho_air  [reduced gravity, m/s^2]
 *   Cd  = surface drag coefficient [-]
 *
 * Richardson transition criterion:
 *   Ri = g'*h / u*^2
 *   Ri > Ri_crit (~0.25) -> gravity current regime (2D model active)
 *   Ri < Ri_crit         -> well-mixed regime (3D ERF dispersion dominates)
 *
 * References:
 *   Webber, D.M. & Brighton, P.W.M. (1987) UKAEA SRD R390.
 *   Didden, N. & Maxworthy, T. (1982) J. Fluid Mech. 121:43.
 *   Benjamin, T.B. (1968) J. Fluid Mech. 31:209.
 *   DEGADIS (Spicer & Havens 1986): dense gas dispersion model.
 */

#ifndef ERF_LNG_GRAVITY_CURRENT_H
#define ERF_LNG_GRAVITY_CURRENT_H

#ifdef ERF_USE_LNG

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>

namespace LNGGravityConst {
    /// Gravitational acceleration [m/s^2]
    constexpr amrex::Real GRAV_ACCEL  = 9.81;

    /// Surface drag coefficient [-] -- Webber & Brighton (1987)
    constexpr amrex::Real CD_DEFAULT  = 2.0e-3;

    /// Richardson transition threshold [-]
    constexpr amrex::Real RI_CRIT     = 0.25;

    /// Minimum cloud depth before zeroing velocity [m]
    /// Cells with h < H_MIN are treated as dry and zeroed out.
    constexpr amrex::Real H_MIN       = 1.0e-4;

    /// Minimum cloud depth for pressure gradient propagation [m]
    /// Cells with h < H_SPREAD_MIN outside the active pool do NOT participate
    /// in the shallow-water pressure gradient. This prevents negligible-depth
    /// off-pool cells from accumulating vapor via pressure spreading, which
    /// caused gc_h_max to grow to 152 m at t=3600 s.
    /// H_SPREAD_MIN = 0.01 m (1 cm) = 100 * H_MIN.
    constexpr amrex::Real H_SPREAD_MIN = 1.0e-2;

    /// CFL safety factor for velocity cap [-]
    /// U_MAX is computed as CFL_SAFETY * min(dx,dy) / dt at runtime.
    /// A fixed 50 m/s cap was physically wrong for a 23 m cell at dt=0.5 s
    /// (CFL = 50*0.5/23 = 1.09 > 1 -- unstable). Using CFL_SAFETY=0.9 gives
    /// U_MAX = 0.9 * 23.4375 / 0.5 = 42.2 m/s, safely below CFL=1.
    constexpr amrex::Real CFL_SAFETY  = 0.9;

    /// Minimum friction velocity for Richardson number [m/s]
    constexpr amrex::Real USTAR_MIN   = 1.0e-8;

    /// Exponential decay timescale for gravity current depth [s]
    /// When there is no local evaporation source, atmospheric dilution
    /// dissipates the dense vapor layer. Typical LNG dense gas persistence ~600 s.
    constexpr amrex::Real H_DECAY_TIMESCALE = 600.0;
}

/**
 * @brief Compute reduced gravity g' = g*(rho_vapor - rho_air)/rho_air [m/s^2]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_reduced_gravity(amrex::Real rho_vapor, amrex::Real rho_air) noexcept
{
    if (rho_air < 1.0e-10) return 0.0;
    return LNGGravityConst::GRAV_ACCEL * (rho_vapor - rho_air) / rho_air;
}

/**
 * @brief Compute Richardson number Ri = g'*h / ustar^2 [-]
 *
 * Returns Ri_crit+1 (gravity current active) when ustar is too small.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_richardson(amrex::Real g_prime, amrex::Real h,
                                amrex::Real ustar) noexcept
{
    amrex::Real u2 = amrex::max(ustar * ustar,
                                 LNGGravityConst::USTAR_MIN * LNGGravityConst::USTAR_MIN);
    if (h < 1.0e-10 || g_prime < 1.0e-10)
        return LNGGravityConst::RI_CRIT + 1.0;  // Conservative: treat as active
    return amrex::max(0.0, g_prime * h / u2);
}

/**
 * @brief Returns true when gravity current regime is active (Ri > Ri_crit).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool is_gravity_current_active(amrex::Real Ri) noexcept
{
    return (Ri > LNGGravityConst::RI_CRIT);
}

/**
 * @brief Advance shallow-water gravity current one timestep.
 *
 * Fix 1 -- CFL-based velocity cap: U_MAX = CFL_SAFETY * min(dx,dy) / dt.
 *
 * Fix 2 -- Pool depletion coupling: zero gc_h/u/v when pool_mask==0 AND
 * h < H_MIN to stop spreading after pool evaporates.
 *
 * Fix 3 -- Atmospheric dilution decay: exponential decay of h away from
 * evaporation source with timescale H_DECAY_TIMESCALE=600 s.
 *
 * Fix 4 -- Spreading threshold: skip pressure gradient in off-pool cells
 * where h < H_SPREAD_MIN (1 cm). This prevents negligible-depth cells from
 * accumulating vapor via the shallow-water pressure gradient and growing to
 * unphysical depths (observed: 152 m at t=3600 s with active spill).
 *
 * @param[in,out] lng_gc_h       Cloud depth [m]
 * @param[in,out] lng_gc_u       Depth-averaged u-velocity [m/s]
 * @param[in,out] lng_gc_v       Depth-averaged v-velocity [m/s]
 * @param[out]    lng_gc_ri_flag Transition flag [0=GC active, 1=mixed regime]
 * @param[in]     lng_evap_flux  Source from evaporation [kg/m^2/s]
 * @param[in]     lng_ustar      Friction velocity [m/s]
 * @param[in]     lng_pool_mask  Pool mask [0/1] -- pool depletion coupling
 * @param[in]     geom_lng       LNG geometry (provides dx for CFL cap)
 * @param[in]     rho_vapor      LNG vapor density [kg/m^3]
 * @param[in]     rho_air        Air density [kg/m^3]
 * @param[in]     Cd             Surface drag coefficient [-]
 * @param[in]     dt             Timestep [s]
 * @param[in]     lng_debug      Debug output flag
 */
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
                              bool                   lng_debug);

#endif /* ERF_USE_LNG */
#endif /* ERF_LNG_GRAVITY_CURRENT_H */
