/**
 * \file ERF_FireWindExtract.H
 *
 * \brief Wind extraction from MOST layer for fire module.
 *
 * Extracts 10-m reference wind speed from the Monin-Obukhov Similarity Theory (MOST)
 * surface layer, applies Wind Adjustment Factor (WAF), and optionally applies
 * FARSITE terrain-based wind corrections.
 *
 * References:
 * - Dyer, A.J. (1974), A review of flux-profile relationships, Boundary-Layer
 *   Meteorology, 7, 363-372.
 * - Andrews, P.L., & Rothermel, R.C. (1982), Fire behavior associated with the
 *   witch and fawn complex fire, USDA Forest Service General Technical Report INT-133.
 * - FARSITE wind model (Finney, M.A., 1998, FARSITE: Fire Area Simulator).
 * - BehavePlus reference (Andrews et al., 2008).
 *
 * Ported from: hgopalan/wildfire_levelset/src/andrews_model.H,
 *              hgopalan/wildfire_levelset/src/farsite_wind_model.H,
 *              hgopalan/wildfire_levelset/src/effective_wind_speed.H
 */

#ifndef ERF_FIRE_WIND_EXTRACT_H
#define ERF_FIRE_WIND_EXTRACT_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array.H>
#include <cmath>

#include "ERF_FireGrid.H"

// ============================================================================
// Monin-Obukhov Similarity Theory (MOST) Functions
// ============================================================================

/**
 * \brief Dyer (1974) stability correction for momentum (psi_m).
 *
 * Implements the Businger-Dyer formulation used in ERF's MOST layer.
 *
 * \param zeta  Dimensionless height z/L where L is Obukhov length
 * \return      psi_m(zeta) for use in MOST wind profile
 *
 * Formula:
 * - Unstable (zeta < 0): Businger-Dyer with x = (1-16*zeta)^0.25
 *   psi_m = 2*ln(0.5*(1+x)) + ln(0.5*(1+x^2)) - 2*atan(x) + pi/2
 * - Stable (zeta > 0): -5*zeta
 * - Neutral (zeta == 0): 0
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real psi_m_dyer(amrex::Real zeta) noexcept
{
    constexpr amrex::Real pi = 3.14159265358979323846;
    constexpr amrex::Real tolerance = 1.0e-12;

    if (std::abs(zeta) < tolerance) {
        return 0.0;
    }

    if (zeta < 0.0) {
        // Unstable conditions (Businger-Dyer)
        amrex::Real x = std::pow(1.0 - 16.0 * zeta, 0.25);
        amrex::Real psi_1 = 2.0 * std::log(0.5 * (1.0 + x));
        amrex::Real psi_2 = std::log(0.5 * (1.0 + x * x));
        amrex::Real psi_3 = 2.0 * std::atan(x) - pi / 2.0;
        return psi_1 + psi_2 + psi_3;
    } else {
        // Stable conditions
        return -5.0 * zeta;
    }
}

/**
 * \brief Reconstruct wind speed at target height using MOST.
 *
 * Computes wind speed at any height above the surface using the log profile
 * with stability corrections.
 *
 * \param ustar      Friction velocity [m/s]
 * \param z0         Surface roughness length [m]
 * \param olen       Obukhov length [m] (dimensionless if |olen| << 1e-6: neutral)
 * \param z_target   Target height [m]
 * \return           Wind speed at z_target [m/s]
 *
 * Formula:
 *   U(z_target) = (ustar / kappa) * [ln(z_target/z0) - psi_m(z_target/olen) + psi_m(z0/olen)]
 *
 * Notes:
 * - Enforces minimum z0 = 1e-4 m (smooth surface)
 * - Treats |olen| < 1e-6 as neutral stratification
 * - Assumes z_target >> z0 (applies in wind farm flow)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real most_wind_at_height(amrex::Real ustar, amrex::Real z0,
                                 amrex::Real olen, amrex::Real z_target) noexcept
{
    constexpr amrex::Real kappa = 0.4;  // von Karman constant
    constexpr amrex::Real z0_min = 1.0e-4;

    // Enforce minimum roughness
    z0 = std::max(z0, z0_min);

    // Treat very small |olen| as neutral (psi_m = 0)
    amrex::Real psi_target = 0.0;
    amrex::Real psi_z0 = 0.0;
    if (std::abs(olen) > 1.0e-6) {
        psi_target = psi_m_dyer(z_target / olen);
        psi_z0 = psi_m_dyer(z0 / olen);
    }

    // MOST wind profile
    amrex::Real U = (ustar / kappa) * (std::log(z_target / z0) - psi_target + psi_z0);
    return std::max(U, 0.0);  // Ensure non-negative
}

// ============================================================================
// Main Wind Extraction Function
// ============================================================================

/**
 * \brief Extract 10-m reference wind from MOST layer and populate fire grid.
 *
 * For each fire cell, maps to the corresponding atmospheric column,
 * extracts MOST parameters (u*, z0, L), reconstructs wind at z_ref,
 * and populates fire_wind_ref with 10-m wind components.
 *
 * \param fire_wind_ref  Output: C*nx × C*ny × 1, 2 comp (u,v) at z_ref
 * \param ustar_mf       Input: u_star from SurfaceLayer, atmospheric grid
 * \param z0_mf          Input: z0 from SurfaceLayer, atmospheric grid
 * \param olen_mf        Input: olen from SurfaceLayer, atmospheric grid
 * \param uavg_mf        Input: u MAC-average, atmospheric grid (wind direction)
 * \param vavg_mf        Input: v MAC-average, atmospheric grid (wind direction)
 * \param fg             FireGrid structure
 * \param z_target       Target height for wind extraction [m] (typically 6.1 m)
 *
 * Algorithm per fire cell (i_f, j_f):
 * 1. Map to atmospheric column: (i_a, j_a) = (i_f/C, j_f/C)
 * 2. Read u_star(i_a, j_a, 0), z0(i_a, j_a, 0), olen(i_a, j_a, 0)
 * 3. Compute wind speed: U_ref = most_wind_at_height(u_star, z0, olen, z_target)
 * 4. Read wind direction from uavg/vavg: theta = atan2(v, u)
 * 5. Set: fire_wind_ref(i_f, j_f, comp=0) = U_ref * cos(theta)
 *         fire_wind_ref(i_f, j_f, comp=1) = U_ref * sin(theta)
 * All reads are purely local — no FillBoundary or MPI required.
 */
void fill_fire_wind_from_most(
    amrex::MultiFab& fire_wind_ref,
    const amrex::MultiFab& ustar_mf,
    const amrex::MultiFab& z0_mf,
    const amrex::MultiFab& olen_mf,
    const amrex::MultiFab& uavg_mf,
    const amrex::MultiFab& vavg_mf,
    const FireGrid& fg,
    amrex::Real z_target);

// ============================================================================
// Wind Adjustment Factor (WAF) Functions
// ============================================================================

/**
 * \brief Unsheltered WAF using Andrews/Albini & Baughman (1979) formula.
 *
 * Logarithmic function of flame length / fuel bed depth ratio.
 *
 * \param delta_ft  Fuel bed depth [feet]
 * \return          WAF (open-canopy factor)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_waf_unsheltered(amrex::Real delta_ft) noexcept
{
    // Avoid division by zero or log of negative
    if (delta_ft < 0.1) return 1.0;
    return 1.83 / std::log((20.0 + 0.36 * delta_ft) / (0.13 * delta_ft));
}

/**
 * \brief Sheltered WAF using FARSITE formula.
 *
 * Accounts for canopy shelter with crown fraction and height.
 *
 * \param f_c          Crown fraction (0 to 1)
 * \param h_c_ft       Crown base height [feet]
 * \param delta_ft     Fuel bed depth [feet]
 * \return             WAF (sheltered factor)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_waf_sheltered(amrex::Real f_c, amrex::Real h_c_ft,
                                   amrex::Real delta_ft) noexcept
{
    if (delta_ft < 0.1 || f_c < 0.01) return 1.0;
    amrex::Real denom = std::sqrt(f_c * h_c_ft) * std::log((20.0 + 0.36 * h_c_ft) / (0.13 * h_c_ft));
    if (denom < 1.0e-12) return 1.0;
    return 0.555 / denom;
}

/**
 * \brief BehavePlus linear WAF formula.
 *
 * Simpler linear approximation to WAF based on fuel depth.
 *
 * \param delta_ft  Fuel bed depth [feet]
 * \return          WAF (BehavePlus factor)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_waf_behaviorplus(amrex::Real delta_ft) noexcept
{
    amrex::Real h_in = delta_ft / 12.0;  // Convert feet to inches
    return 0.36 + 0.004 * h_in;
}

/**
 * \brief Canopy exponential reduction using Beer-Lambert law.
 *
 * Applies exponential decay to WAF under canopy.
 *
 * \param waf_open  Open-area WAF
 * \param f_c       Crown fraction (0 to 1)
 * \param alpha_c   Canopy extinction coefficient (default ~0.1)
 * \return          Sheltered WAF
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_waf_canopy_exp(amrex::Real waf_open, amrex::Real f_c,
                                    amrex::Real alpha_c = 0.1) noexcept
{
    return waf_open * std::exp(-alpha_c * f_c);
}

/**
 * \brief Dispatch function to compute WAF for a single fuel cell.
 *
 * \param f_c              Crown fraction (0 to 1)
 * \param h_c_ft           Crown base height [feet]
 * \param delta_ft         Fuel bed depth [feet]
 * \param shelter_threshold Crown fraction threshold for sheltering (default 0.2)
 * \param waf_formula_id   0 = "andrews", 1 = "behaviorplus"
 * \param alpha_c          Canopy extinction coefficient
 * \return                 WAF factor to apply to wind speed
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_waf_per_cell(amrex::Real f_c, amrex::Real h_c_ft,
                                  amrex::Real delta_ft, amrex::Real shelter_threshold,
                                  int waf_formula_id, amrex::Real alpha_c = 0.1) noexcept
{
    amrex::Real waf = 1.0;

    // Select base formula
    if (waf_formula_id == 0) {
        // Andrews formula
        waf = compute_waf_unsheltered(delta_ft);
    } else if (waf_formula_id == 1) {
        // BehavePlus formula
        waf = compute_waf_behaviorplus(delta_ft);
    }

    // Apply canopy reduction if sheltered
    if (f_c > shelter_threshold) {
        waf = compute_waf_canopy_exp(waf, f_c, alpha_c);
    }

    return waf;
}

/**
 * \brief Compute Maximum Effective Wind Speed (MEWS) cap in ft/min.
 *
 * Used to limit fire spread rate under extreme wind conditions.
 *
 * \param I_R            Reaction intensity [BTU/ft^2/min]
 * \param C              Wind speed coefficient from Rothermel
 * \param B              Wind speed exponent from Rothermel
 * \param beta_ratio_E   (beta/beta_opt)^E term from Rothermel
 * \return               Maximum wind speed [ft/min] for MEWS cap
 *
 * Formula: phi_w_max = 0.9 * I_R; inverted to find U_max
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real compute_max_ews_ftmin(amrex::Real I_R, amrex::Real C, amrex::Real B,
                                   amrex::Real beta_ratio_E) noexcept
{
    // Prevent division by zero
    if (std::abs(C) < 1.0e-12 || std::abs(beta_ratio_E) < 1.0e-12 || I_R < 1.0e-12) {
        return 1.0e6;  // No cap
    }

    // phi_w_max = 0.9 * I_R
    amrex::Real phi_w_max = 0.9 * I_R;

    // Rothermel: phi_w = C * U^B * beta_ratio_E
    // Solve: U_max = (phi_w_max / (C * beta_ratio_E))^(1/B)
    if (std::abs(B) < 1.0e-12) {
        return 1.0e6;  // No cap if B ≈ 0
    }
    amrex::Real exponent = 1.0 / B;
    amrex::Real argument = phi_w_max / (C * beta_ratio_E);
    if (argument < 0.0) return 1.0e6;

    return std::pow(argument, exponent);
}

// ============================================================================
// FARSITE Terrain Wind Corrections
// ============================================================================

/**
 * \brief Compute terrain curvature for FARSITE wind model.
 *
 * Curvature captures convex (ridges) and concave (valleys) terrain.
 * Used to determine wind speed and direction modifications.
 *
 * \param curvature    Output: 1 component, fire grid resolution
 * \param fire_slopes  Input: 2 components (dz/dx, dz/dy), fire grid resolution
 * \param geom_fire    Fire grid Geometry (for spacing)
 *
 * Algorithm: Compute principal curvatures from terrain slopes via finite differences.
 */
void compute_terrain_curvature(
    amrex::MultiFab& curvature,
    const amrex::MultiFab& fire_slopes,
    const amrex::Geometry& geom_fire);

/**
 * \brief Apply FARSITE terrain-based wind corrections.
 *
 * Modifies wind vector for terrain features:
 * - Ridge speed-up (k_ridge > 1): Wind is accelerated on ridges
 * - Lee-side shelter (k_shelter < 1): Wind is reduced on lee side
 * - Valley channeling (k_valley variable): Wind is channeled in valleys
 * - Directional deflection: Wind direction shifts near terrain features
 *
 * \param fire_wind      In/out: u,v components after extraction
 * \param fire_slopes    Input: dz/dx, dz/dy (2 components)
 * \param curvature      Input: Terrain curvature (1 component)
 * \param k_ridge        Ridge enhancement factor (typically 1.5)
 * \param k_shelter      Lee-side shelter factor (typically 0.6)
 * \param k_valley       Valley channeling factor (typically 0.8)
 * \param k_deflect      Directional deflection factor (typically 0.3)
 * \param min_curv       Minimum curvature threshold for terrain effects (default 0.001)
 */
void apply_farsite_terrain_wind(
    amrex::MultiFab& fire_wind,
    const amrex::MultiFab& fire_slopes,
    const amrex::MultiFab& curvature,
    amrex::Real k_ridge, amrex::Real k_shelter,
    amrex::Real k_valley, amrex::Real k_deflect,
    amrex::Real min_curv = 0.001);

#endif // ERF_FIRE_WIND_EXTRACT_H
