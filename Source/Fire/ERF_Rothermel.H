/**
 * \file ERF_Rothermel.H
 *
 * \brief Rothermel fire spread model implementation.
 *
 * Implements the semi-empirical Rothermel (1972) model for fire rate of spread,
 * including:
 * - Anderson FBFM13 fuel database (13 standard fuel models)
 * - Rothermel ROS computation with wind and slope effects
 * - Maximum Effective Wind Speed (MEWS) cap
 *
 * References:
 * - Rothermel, R.C. (1972), A mathematical model for predicting fire spread in
 *   wildland fuels. USDA Forest Service Research Paper INT-115.
 * - Anderson, H.E. (1982), Aids to determining fuel models for estimating fire behavior.
 *   USDA Forest Service General Technical Report INT-122.
 *
 * Ported from: hgopalan/wildfire_levelset/src/compute_rothermel_R.H,
 *              hgopalan/wildfire_levelset/src/rothermel_model.H,
 *              hgopalan/wildfire_levelset/src/fuel_database.H
 */

#ifndef ERF_FIRE_ROTHERMEL_H
#define ERF_FIRE_ROTHERMEL_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Array.H>
#include <cmath>

// ============================================================================
// Fuel Model Database (Anderson FBFM13)
// ============================================================================

/**
 * \struct FuelModelParams
 *
 * Holds fuel model parameters for a single Anderson FBFM13 model.
 * All values are as specified in Anderson (1982).
 */
struct FuelModelParams
{
    amrex::Real w_d1 = 0.0;          ///< 1-hour dead fuel load [lb/ac]
    amrex::Real w_d10 = 0.0;         ///< 10-hour dead fuel load [lb/ac]
    amrex::Real w_d100 = 0.0;        ///< 100-hour dead fuel load [lb/ac]
    amrex::Real w_lh = 0.0;          ///< Live herbaceous fuel load [lb/ac]
    amrex::Real w_lw = 0.0;          ///< Live woody fuel load [lb/ac]
    amrex::Real sigma_d1 = 0.0;      ///< 1-hour dead surface-to-volume ratio [1/ft]
    amrex::Real sigma_lh = 0.0;      ///< Live herbaceous S/V ratio [1/ft]
    amrex::Real sigma_lw = 0.0;      ///< Live woody S/V ratio [1/ft]
    amrex::Real delta = 0.0;         ///< Fuel bed depth [ft]
    amrex::Real Mx = 0.0;            ///< Extinction moisture content [%] (dry)
    amrex::Real heat_content = 0.0;  ///< Heat content [BTU/lb]
    amrex::Real rho_p = 0.0;         ///< Particle density [lb/ft^3]
};

/**
 * \brief Retrieve Anderson FBFM13 fuel model parameters.
 *
 * Returns pre-computed fuel parameters for one of the 13 standard models.
 *
 * \param fuel_model_id  Fuel model ID (1-13)
 * \return               FuelModelParams struct with all fuel parameters
 *
 * Notes:
 * - Model 1: Short grass
 * - Model 5: Brush (GR1, typical wildfire fuel)
 * - Model 8: Closed timber litter
 * - See Anderson (1982) for descriptions of all 13 models
 */
FuelModelParams get_anderson_fuel_params(int fuel_model_id);

// ============================================================================
// Rothermel Model Precomputed Data
// ============================================================================

/**
 * \struct RothermelComputed
 *
 * Pre-computed intermediate values in the Rothermel model.
 * Computed once at initialization based on fuel parameters and moisture.
 */
struct RothermelComputed
{
    amrex::Real R0;              ///< No-wind, no-slope ROS [m/s]
    amrex::Real C;               ///< Wind coefficient (Rothermel eq. 52)
    amrex::Real B;               ///< Wind exponent (Rothermel eq. 53)
    amrex::Real beta_ratio_E;    ///< (beta / beta_opt)^E term
    amrex::Real beta;            ///< Actual packing ratio
    amrex::Real phi_s_const;     ///< Constant part of slope factor
    amrex::Real waf;             ///< Wind Adjustment Factor
    amrex::Real U_max_ftmin;     ///< Maximum wind speed [ft/min] (MEWS cap)
    amrex::Real wind_conv;       ///< Conversion factor m/s to ft/min (1.94384)
    amrex::Real ros_conv;        ///< Conversion factor ft/min to m/s (0.00508)
    amrex::Real I_R;             ///< Reaction intensity [BTU/ft^2/min]
};

/**
 * \brief Compute Rothermel model parameters from fuel and moisture.
 *
 * This function computes all intermediate values needed for per-cell ROS
 * calculation. Called once at initialization for a given fuel type and
 * moisture distribution.
 *
 * \param fp           Fuel model parameters
 * \param M_1hr        1-hour fuel moisture [fraction]
 * \param M_10hr       10-hour fuel moisture [fraction]
 * \param M_100hr      100-hour fuel moisture [fraction]
 * \return             RothermelComputed struct
 *
 * Computation steps:
 * 1. Compute moisture of extinction adjustment
 * 2. Compute net fuel load weighted by moisture
 * 3. Compute packing ratio and ratio to optimum (beta/beta_opt)
 * 4. Compute reaction intensity I_R
 * 5. Compute wind and slope coefficients C, B, E
 * 6. Compute MEWS cap U_max_ftmin
 */
RothermelComputed compute_rothermel_params(const FuelModelParams& fp,
                                           amrex::Real M_1hr = 0.08,
                                           amrex::Real M_10hr = 0.08,
                                           amrex::Real M_100hr = 0.10);

// ============================================================================
// Per-Cell ROS Kernel
// ============================================================================

/**
 * \brief Compute fire rate of spread for a single cell (GPU kernel).
 *
 * Implements the complete Rothermel ROS formula with wind and slope effects
 * vectorially combined.
 *
 * \param ux_eff       Effective midflame wind u-component [m/s]
 * \param uy_eff       Effective midflame wind v-component [m/s]
 * \param sx           Terrain slope component dz/dx [-]
 * \param sy           Terrain slope component dz/dy [-]
 * \param rc           Pre-computed Rothermel parameters
 * \return             Rate of spread [m/s]
 *
 * Algorithm:
 * 1. Compute wind factor: phi_w = C * (U_eff)^B * (beta_ratio_E)
 *    where U_eff is capped at U_max_ftmin for MEWS limit
 * 2. Compute slope factor: phi_s = slope_mod * tan(slope) where
 *    slope = atan(sqrt(sx^2 + sy^2))
 * 3. Compute ROS: R = R0 * (1 + phi_w) * (1 + phi_s) [in ft/min]
 * 4. Convert to m/s and return
 *
 * Notes:
 * - Wind and slope effects are multiplicative (Rothermel 1972)
 * - MEWS cap applied to U_eff before wind factor computation
 * - Slope factor scaled using slope-parallel distance
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real rothermel_ros_cell(
    amrex::Real ux_eff, amrex::Real uy_eff,
    amrex::Real sx, amrex::Real sy,
    const RothermelComputed& rc) noexcept;

// ============================================================================
// MultiFab Kernel
// ============================================================================

/**
 * \brief Compute ROS field on fire grid.
 *
 * Iterates over fire grid and calls rothermel_ros_cell for each location,
 * storing results in fire_ros MultiFab.
 *
 * \param fire_ros     Output: Rate of spread [m/s], 1 component
 * \param fire_wind    Input: Effective midflame wind (u,v), 2 components
 * \param fire_slopes  Input: Terrain slopes (dz/dx, dz/dy), 2 components
 * \param rc           Pre-computed Rothermel parameters (same for all cells)
 *
 * Algorithm: For each cell, read effective wind and slopes, call
 * rothermel_ros_cell to compute ROS, store in fire_ros.
 */
void compute_ros_field(
    amrex::MultiFab& fire_ros,
    const amrex::MultiFab& fire_wind,
    const amrex::MultiFab& fire_slopes,
    const RothermelComputed& rc);

#endif // ERF_FIRE_ROTHERMEL_H
