/**
 * @file ERF_LNGEvaporation.H
 * @brief LNG pool evaporation physics kernels
 * @details
 * Implements Stefan-diffusion / Chilton-Colburn mass transfer evaporation model
 * for boiling LNG pools. The evaporation flux is computed from the mass transfer
 * coefficient derived from boundary layer similarity:
 * 
 *   F_evap = k_mass * rho_vapor * (Y_sat - Y_inf)
 * 
 * where k_mass is the mass transfer coefficient [m/s] computed from:
 *   k_mass = u* * kappa / (Sc^(2/3) * ln(z_ref/z0))
 * 
 * For a boiling pool (Y_sat = 1.0, Y_inf = 0.0):
 *   F_evap = u* * kappa / (Sc^(2/3) * ln(z_ref/z0)) * rho_vapor
 * 
 * Model references:
 * - Brighton, P.W.M. (1990). Evaporation from a plane liquid surface into a 
 *   turbulent boundary layer. J. Fluid Mech., 159, 323-345.
 * - Webber, D.M., & Brighton, P.W.M. (1987). An integral model of spreading 
 *   vaporizing pools. UKAEA SRD R390.
 * - Koopman, R.P. et al. (1982). Burro series data report. LLNL UCID-19075.
 * - NFPA 59A (2023). Standard for the Production, Storage, and Handling of LNG.
 * 
 * @note Phase 2: Implements evaporation kernel only. Phase 3+ will add gravity current
 *       spreading, and Phase 4+ will replace placeholder ATM fields with live extraction.
 * 
 * @see Source/LNG/ERF_LNGPool.H — Pool dynamics and mass tracking
 * @see Source/LNG/ERF_LNGLayer.cpp — Integration into advance() sequence
 */

#ifndef ERF_LNG_EVAPORATION_H
#define ERF_LNG_EVAPORATION_H

#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <cmath>

/**
 * @namespace LNGEvapConst
 * @brief Physical constants for LNG evaporation model
 */
namespace LNGEvapConst {
    /// von Kármán constant (boundary layer similarity)
    constexpr amrex::Real VON_KARMAN    = 0.4;
    
    /// Schmidt number for CH4 in air (Sc = nu / D_CH4_air)
    constexpr amrex::Real SC_CH4_AIR    = 0.9;
    
    /// Hard cap on evaporation flux [kg/m^2/s]
    /// Physically unrealistic values prevented; typical max ~0.5 kg/m^2/s
    constexpr amrex::Real FLUX_MAX_EVAP = 1.0;
    
    /// Minimum friction velocity before flux clamped to zero [m/s]
    /// Below this, boundary layer is too weak for evaporation
    constexpr amrex::Real USTAR_MIN     = 1.0e-4;
    
    /// Default aerodynamic roughness if z0 <= 0 [m]
    /// Brighton (1990): smooth liquid surfaces ~0.01 m
    constexpr amrex::Real Z0_DEFAULT    = 0.01;
};

/**
 * @brief Compute mass transfer coefficient [m/s] from Chilton-Colburn analogy
 * @param[in] ustar   Friction velocity [m/s]
 * @param[in] z_ref   Reference height [m]
 * @param[in] z0      Aerodynamic roughness [m]
 * @return k_mass [m/s]
 * 
 * Formula (Brighton 1990):
 *   k_mass = u* * kappa / (Sc^(2/3) * ln(z_ref/z0))
 * 
 * If z0 <= 0, uses Z0_DEFAULT. If result becomes negative or inf, returns 0.
 * 
 * @note AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE for maximum performance on GPU.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_mass_transfer_coeff(amrex::Real ustar, amrex::Real z_ref,
                                         amrex::Real z0) noexcept
{
    using namespace LNGEvapConst;
    
    if (z0 <= 0.0) z0 = Z0_DEFAULT;
    if (ustar < USTAR_MIN) return 0.0;
    if (z_ref <= z0) return 0.0;
    
    amrex::Real ln_ratio = std::log(z_ref / z0);
    if (ln_ratio <= 0.0) return 0.0;
    
    // Sc^(2/3)
    amrex::Real sc_power = std::pow(SC_CH4_AIR, 2.0/3.0);
    
    // k_mass = u* * kappa / (Sc^(2/3) * ln(z_ref/z0))
    amrex::Real k_mass = (ustar * VON_KARMAN) / (sc_power * ln_ratio);
    
    return (k_mass > 0.0) ? k_mass : 0.0;
}

/**
 * @brief Compute evaporation flux [kg/m^2/s] for boiling LNG pool
 * @param[in] ustar       Friction velocity [m/s]
 * @param[in] z_ref       Reference height [m]
 * @param[in] z0          Aerodynamic roughness [m]
 * @param[in] rho_vapor   Vapor density at saturation [kg/m^3]
 * @return F_evap [kg/m^2/s], clamped to [0, FLUX_MAX_EVAP]
 * 
 * For boiling pool: Y_sat = 1.0 (saturation at interface), Y_inf = 0.0 (far-field)
 * 
 *   F_evap = k_mass * rho_vapor * (1.0 - 0.0) = k_mass * rho_vapor
 * 
 * Result is clamped to [0, FLUX_MAX_EVAP] for physical sanity.
 * 
 * @note If u* < USTAR_MIN, returns 0 immediately (no wind, no evaporation).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_evap_flux_boiling(amrex::Real ustar, amrex::Real z_ref,
                                       amrex::Real z0,
                                       amrex::Real rho_vapor) noexcept
{
    using namespace LNGEvapConst;
    
    if (ustar < USTAR_MIN) return 0.0;
    
    amrex::Real k_mass = compute_mass_transfer_coeff(ustar, z_ref, z0);
    amrex::Real F_evap = k_mass * rho_vapor;
    
    // Clamp to hard maximum
    if (F_evap > FLUX_MAX_EVAP) F_evap = FLUX_MAX_EVAP;
    
    return (F_evap < 0.0) ? 0.0 : F_evap;
}

/**
 * @brief Compute latent heat flux [W/m^2] from evaporation
 * @param[in] F_evap  Evaporation flux [kg/m^2/s]
 * @param[in] Hv      Latent heat of vaporization [J/kg]
 * @return Q_latent [W/m^2] = F_evap * Hv
 * 
 * Heat extraction from surface is positive (cooling).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_latent_heat_flux(amrex::Real F_evap, amrex::Real Hv) noexcept
{
    return F_evap * Hv;
}

/**
 * @brief Fill lng_evap_flux and lng_latent_flux MultiFabs over 2D LNG grid
 * @param[in,out] lng_evap_flux    Evaporation flux [kg/m^2/s]
 * @param[in,out] lng_latent_flux  Latent heat flux [W/m^2]
 * @param[in]     lng_pool_mask    Pool mask [0/1]
 * @param[in]     lng_ustar        Friction velocity [m/s]
 * @param[in]     z_ref            Reference height [m]
 * @param[in]     z0               Aerodynamic roughness [m]
 * @param[in]     rho_vapor        Vapor density [kg/m^3]
 * @param[in]     Hv               Latent heat [J/kg]
 * @param[in]     lng_debug        If true, print summary statistics
 * 
 * @details
 * Iterates over LNG 2D grid using MFIter with TilingIfNotGPU.
 * For each cell:
 *   - If pool_mask > 0.5: compute evap/latent flux and store
 *   - Otherwise: set flux to 0
 * 
 * After kernel, if lng_debug=true, prints:
 *   [LNG DEBUG] Phase 2: evap step  evap_flux_max=<val> kg/m^2/s  evap_flux_sum=<val> kg/m^2/s  latent_flux_max=<val> W/m^2  active_cells=<N>
 * 
 * @note Called from LNGLayer::advance() Step E.
 */
void compute_lng_evap_flux(amrex::MultiFab& lng_evap_flux,
                            amrex::MultiFab& lng_latent_flux,
                            const amrex::MultiFab& lng_pool_mask,
                            const amrex::MultiFab& lng_ustar,
                            amrex::Real z_ref,
                            amrex::Real z0,
                            amrex::Real rho_vapor,
                            amrex::Real Hv,
                            bool lng_debug);

#endif /* ERF_LNG_EVAPORATION_H */
