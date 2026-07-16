/**
 * @file ERF_LNGGravityCurrent.H
 * @brief Depth-averaged shallow-water gravity current PDEs for LNG dense vapor.
 * @note Phase 5
 * 
 * Governing equations on the 2D slab (depth h [m], depth-averaged velocity u,v [m/s]):
 *   ∂h/∂t   + ∂(hu)/∂x + ∂(hv)/∂y = F_evap/ρ_vapor            [mass]
 *   ∂(hu)/∂t + ∂(hu²)/∂x = -g'·h·∂h/∂x - Cd·u·|u|             [x-momentum]
 *   ∂(hv)/∂t + ∂(hv²)/∂y = -g'·h·∂h/∂y - Cd·v·|v|             [y-momentum]
 * 
 * where:
 *   g' = g*(ρ_vapor - ρ_air)/ρ_air  [reduced gravity, m/s^2]
 *   Cd  = surface drag coefficient [-]
 * 
 * Richardson transition criterion:
 *   Ri = g'*h / u*²
 *   Ri > Ri_crit (~0.25) → gravity current regime (2D model active)
 *   Ri < Ri_crit         → well-mixed regime (3D ERF dispersion dominates)
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
    constexpr amrex::Real GRAV_ACCEL      = 9.81;
    
    /// Surface drag coefficient [-]
    /// From Webber & Brighton (1987)
    constexpr amrex::Real CD_DEFAULT      = 2.0e-3;
    
    /// Richardson transition threshold [-]
    /// When Ri exceeds this, transition from gravity current to mixed regime
    constexpr amrex::Real RI_CRIT         = 0.25;
    
    /// Minimum cloud depth before zeroing velocity [m]
    /// Prevents numerical issues from vanishingly small depths
    constexpr amrex::Real H_MIN           = 1.0e-4;
    
    /// Hard cap on gravity current speed [m/s]
    /// Prevents runaway velocities from numerical instability
    constexpr amrex::Real U_MAX           = 50.0;
    
    /// Minimum friction velocity for Richardson number [m/s]
    /// Prevents division by zero
    constexpr amrex::Real USTAR_MIN       = 1.0e-8;
}

/**
 * @brief Compute reduced gravity g' = g*(rho_vapor - rho_air)/rho_air [m/s^2]
 * 
 * @param[in] rho_vapor  LNG vapor density [kg/m^3]
 * @param[in] rho_air    Air density [kg/m^3]
 * @return g' [m/s^2]
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
 * Returns Ri_crit+1 (gravity current active) when ustar is too small to avoid division by zero.
 * 
 * @param[in] g_prime  Reduced gravity [m/s^2]
 * @param[in] h        Cloud depth [m]
 * @param[in] ustar    Friction velocity [m/s]
 * @return Richardson number [-]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_richardson(amrex::Real g_prime, amrex::Real h,
                                amrex::Real ustar) noexcept
{
    amrex::Real u2 = amrex::max(ustar*ustar, 
                                 LNGGravityConst::USTAR_MIN * LNGGravityConst::USTAR_MIN);
    if (h < 1.0e-10 || g_prime < 1.0e-10) {
        return LNGGravityConst::RI_CRIT + 1.0;  // Treat as active (conservative)
    }
    return amrex::max(0.0, g_prime * h / u2);
}

/**
 * @brief Returns true when gravity current regime is active (Ri > Ri_crit).
 * 
 * @param[in] Ri  Richardson number [-]
 * @return true if Ri > Ri_crit (gravity current active)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool is_gravity_current_active(amrex::Real Ri) noexcept
{
    return (Ri > LNGGravityConst::RI_CRIT);
}

/**
 * @brief Advance shallow-water gravity current one timestep.
 * 
 * Updates lng_gc_h, lng_gc_u, lng_gc_v using explicit first-order scheme.
 * Only active where is_gravity_current_active(Ri) == true.
 * Cells where Ri < Ri_crit are flagged in lng_gc_ri_flag (0=GC active, 1=mixed).
 * 
 * @param[in,out] lng_gc_h       Cloud depth [m]
 * @param[in,out] lng_gc_u       Depth-averaged u-velocity [m/s]
 * @param[in,out] lng_gc_v       Depth-averaged v-velocity [m/s]
 * @param[out]    lng_gc_ri_flag Transition flag [0=GC active, 1=mixed regime]
 * @param[in]     lng_evap_flux  Source from evaporation [kg/m^2/s]
 * @param[in]     lng_ustar      Friction velocity [m/s]
 * @param[in]     geom_lng       LNG geometry
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
                              const amrex::Geometry& geom_lng,
                              amrex::Real            rho_vapor,
                              amrex::Real            rho_air,
                              amrex::Real            Cd,
                              amrex::Real            dt,
                              bool                   lng_debug);

#endif /* ERF_USE_LNG */

#endif /* ERF_LNG_GRAVITY_CURRENT_H */
