#ifndef ERF_ROTHERMEL_H
#define ERF_ROTHERMEL_H

#include <vector>
#include <limits>
#include <AMReX_MultiFab.H>
#include <cmath>

/**
 * @file ERF_Rothermel.H
 * @brief Rothermel fire spread model implementation
 *
 * Implements the Rothermel (1972) fire spread model for rate-of-spread (ROS)
 * computation as a function of fuel type, moisture, wind, and slope.
 *
 * Key references:
 * - Rothermel, R.C. (1972): "A Mathematical Model for Predicting Fire Spread
 *   in Wildland Fuels", USDA Forest Service Research Paper INT-115
 * - Anderson, H.E. (1982): "Aids to determining fuel models for estimating fire
 *   behavior", General Technical Report INT-122, USDA Forest Service
 * - Albini, F.A. (1976): "Estimating wildfire behavior and effects",
 *   General Technical Report INT-30, USDA Forest Service
 *
 * Ported from: wildfire_levelset/src/rothermel_model.H and
 *              wildfire_levelset/src/compute_rothermel_R.H
 */

/**
 * @struct FuelModelParams
 * @brief Anderson FBFM13 fuel model parameters
 *
 * Stores parameters for one of the 13 standard fuel models used by
 * the fire behavior community (Anderson 1982).
 */
struct FuelModelParams
{
    // Fuel load [lb/ft² or kg/m², converted as needed]
    amrex::Real w_d1;       ///< 1-hour dead fuel load
    amrex::Real w_d10;      ///< 10-hour dead fuel load
    amrex::Real w_d100;     ///< 100-hour dead fuel load
    amrex::Real w_lh;       ///< Live herbaceous fuel load
    amrex::Real w_lw;       ///< Live woody fuel load

    // Surface area to volume ratio [1/ft or 1/m]
    amrex::Real sigma_d1;   ///< 1-hour SAV
    amrex::Real sigma_lh;   ///< Herbaceous SAV
    amrex::Real sigma_lw;   ///< Woody SAV

    // Fuel bed depth [ft or m]
    amrex::Real delta;

    // Fuel moisture of extinction [fraction]
    amrex::Real Mx;

    // Heat content [BTU/lb or J/kg]
    amrex::Real heat_content;

    // Fuel particle density [lb/ft³ or kg/m³]
    amrex::Real rho_p;
};

/**
 * @brief Get Anderson FBFM13 fuel model parameters
 *
 * @param[in] model_id Anderson fuel model ID (1–13)
 * @return FuelModelParams for the specified model
 *
 * All 13 Anderson fuel models are hard-coded with standard values.
 */
// Anderson FBFM13 fuel model database
// Reference: Anderson, H.E. (1982) "Aids to determining fuel models..."
// All values are standard published values for these fuel models

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FuelModelParams get_anderson_fuel_params(int model_id)
{
    FuelModelParams fp;

    // All fuel loads in lb/ft² (Anderson 1982, Table 1: t/ac x 2000/43560)
    // SAV in 1/ft: sigma_d1 is the 1-hour class value of Anderson's Table 1
    // for every model. The single-class Rothermel path uses it as the fuel-bed
    // SAV; the BEHAVE path derives the characteristic SAV from the classes
    // itself (10-hr = 109, 100-hr = 30 ft⁻¹ there). Albini's (1976) tabulated
    // characteristic SAVs (FM4 1739, FM5 1683, FM7 1552) are NOT used.
    // The live load of FM2, 4, 5, 7 and 10 is carried in the herbaceous class
    // (w_lh), with live SAV 1500 1/ft, so the BEHAVE transfer window acts on
    // it; BehavePlus treats the 13 models as static (no transfer) and files
    // the live load of FM4, 5, 7 and 10 as woody: a deliberate departure of
    // this code (the single-class Rothermel path sums the two live classes).
    // Heat content in BTU/lb
    // Particle density in lb/ft³
    // Depth in ft
    // Reference: Anderson (1982) GTR INT-122, Table 1 (also Albini 1976, Table 7)

    switch (model_id) {
    case 1:  // Short grass (1 ft)
        fp.w_d1 = 0.034;   fp.w_d10 = 0.0;    fp.w_d100 = 0.0;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 3500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 1.0;    fp.Mx = 0.12;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 2:  // Timber (Grass and Understory): 2.00 / 1.00 / 0.50 t/ac dead, 0.50 live
        fp.w_d1 = 0.092;   fp.w_d10 = 0.046;  fp.w_d100 = 0.023;
        fp.w_lh = 0.023;   fp.w_lw = 0.0;
        fp.sigma_d1 = 3000.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 1.0;    fp.Mx = 0.15;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 3:  // Tall grass (2.5 ft)
        fp.w_d1 = 0.138;   fp.w_d10 = 0.0;    fp.w_d100 = 0.0;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 1500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 2.5;    fp.Mx = 0.25;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 4:  // Chaparral (6 ft): 1-h SAV 2000 (1739 is Albini's characteristic value)
        fp.w_d1 = 0.230;   fp.w_d10 = 0.184;  fp.w_d100 = 0.092;
        fp.w_lh = 0.230;   fp.w_lw = 0.0;
        fp.sigma_d1 = 2000.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 6.0;    fp.Mx = 0.20;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 5:  // Brush (2 ft): 1-h SAV 2000 (1683 is Albini's characteristic value)
        fp.w_d1 = 0.046;   fp.w_d10 = 0.023;  fp.w_d100 = 0.0;
        fp.w_lh = 0.092;   fp.w_lw = 0.0;
        fp.sigma_d1 = 2000.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 2.0;    fp.Mx = 0.20;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 6:  // Dormant Brush, Hardwood Slash: 1.50 / 2.50 / 2.00 t/ac
        fp.w_d1 = 0.069;   fp.w_d10 = 0.115;  fp.w_d100 = 0.092;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 1750.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 2.5;    fp.Mx = 0.25;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 7:  // Southern Rough: 1.13 / 1.87 / 1.50 t/ac dead, ONE live class of 0.37 t/ac
        fp.w_d1 = 0.052;   fp.w_d10 = 0.086;  fp.w_d100 = 0.069;
        fp.w_lh = 0.017;   fp.w_lw = 0.0;
        fp.sigma_d1 = 1750.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 2.5;    fp.Mx = 0.40;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 8:  // Closed Timber Litter
        fp.w_d1 = 0.069;   fp.w_d10 = 0.046;  fp.w_d100 = 0.115;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 2000.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 0.2;    fp.Mx = 0.30;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 9:  // Hardwood Litter
        fp.w_d1 = 0.134;   fp.w_d10 = 0.019;  fp.w_d100 = 0.007;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 2500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 0.2;    fp.Mx = 0.25;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 10:  // Timber (Litter and Understory)
        fp.w_d1 = 0.138;   fp.w_d10 = 0.092;  fp.w_d100 = 0.230;
        fp.w_lh = 0.092;   fp.w_lw = 0.0;
        fp.sigma_d1 = 2000.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 1.0;    fp.Mx = 0.25;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 11:  // Light Logging Slash
        fp.w_d1 = 0.069;   fp.w_d10 = 0.207;  fp.w_d100 = 0.253;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 1500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 1.0;    fp.Mx = 0.15;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 12:  // Medium Logging Slash
        fp.w_d1 = 0.184;   fp.w_d10 = 0.644;  fp.w_d100 = 0.759;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 1500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 2.3;    fp.Mx = 0.20;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 13:  // Heavy Logging Slash
        fp.w_d1 = 0.322;   fp.w_d10 = 1.058;  fp.w_d100 = 1.288;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 1500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 3.0;    fp.Mx = 0.25;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    default:
        // GPU-safe fallback: an unknown id silently uses model 1. amrex::Print()
        // and the recursive call this replaced are both illegal on device, and
        // this function is called from inside ParallelFor kernels.
        fp.w_d1 = 0.034;   fp.w_d10 = 0.0;    fp.w_d100 = 0.0;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 3500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 1.0;    fp.Mx = 0.12;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;
    }

    return fp;
}

/// Which wind limit erf.fire.use_wind_limit applies, erf.fire.wind_limit
namespace fire_wind_limit {
    constexpr int rothermel  = 0;   ///< "rothermel": U_max = 0.9 I_R (Rothermel 1972 eq. 87; Andrews 2018)
    constexpr int fuel_class = 1;   ///< "fuel_class": 300 ft/min for sigma > 1000 1/ft, 500 ft/min otherwise (this code's rule before 2026-10)
}

/**
 * @brief Rothermel's (1972) wind limit [ft/min]: U_max = 0.9 I_R
 *
 * Eq. 87 of INT-115 (the "wind limit" of Andrews 2018, applied in BehavePlus
 * and FARSITE): the effective wind speed in ft/min is limited to 0.9 times the
 * reaction intensity in BTU/ft2/min. For short grass at 8 % moisture I_R is
 * about 765 BTU/ft2/min, so the cap is 690 ft/min (3.5 m/s midflame), and it
 * rises with drier or heavier fuel.
 *
 * @param[in] I_R  Reaction intensity [BTU/ft2/min]
 * @return U_max [ft/min]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rothermel_wind_limit_ftmin(amrex::Real I_R) noexcept
{
    return amrex::Real(0.9) * amrex::max(I_R, amrex::Real(0.0));
}

/**
 * @brief Midflame wind cap of the Rothermel wind factor [ft/min]
 *
 * With erf.fire.use_wind_limit = true (default) and erf.fire.wind_limit =
 * "rothermel" (default) this is Rothermel's own U_max = 0.9 I_R; with
 * wind_limit = "fuel_class" it is the absolute rule this code used until
 * October 2026, 300 ft/min for fine fuels (sigma > 1000 1/ft) and 500 ft/min
 * otherwise, which capped every grass fire above a 4.2 m/s reference wind.
 * With use_wind_limit = false there is no cap: the largest Real, which the
 * kernels' min() never selects.
 *
 * @param[in] sigma           Characteristic surface-area-to-volume ratio [1/ft]
 * @param[in] I_R             Reaction intensity [BTU/ft2/min]
 * @param[in] use_wind_limit  erf.fire.use_wind_limit
 * @param[in] wind_limit_mode fire_wind_limit::rothermel or fire_wind_limit::fuel_class
 * @return U_max [ft/min]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rothermel_wind_cap_ftmin(amrex::Real sigma, amrex::Real I_R, bool use_wind_limit,
                                     int wind_limit_mode = fire_wind_limit::rothermel) noexcept
{
    if (!use_wind_limit) { return std::numeric_limits<amrex::Real>::max(); }
    if (wind_limit_mode == fire_wind_limit::fuel_class) {
        return (sigma > amrex::Real(1000.0)) ? amrex::Real(300.0) : amrex::Real(500.0);
    }
    return rothermel_wind_limit_ftmin(I_R);
}

/**
 * @struct RothermelComputed
 * @brief Precomputed Rothermel coefficients (fuel + moisture dependent)
 *
 * Stores intermediate results computed once per timestep to avoid
 * redundant calculations in the per-cell GPU kernel.
 */
struct RothermelComputed
{
    amrex::Real R0;              ///< No-wind, no-slope ROS [m/s]
    amrex::Real C, B;            ///< Wind factor φ_w = C * U^B * (β/β_op)^-E
    amrex::Real beta_ratio_E;    ///< (β/β_op)^-E packing ratio factor (Rothermel Eq. 47)
    amrex::Real beta;            ///< Packing ratio (stored for diagnostics; the kernels read beta_ratio_E and phi_s_const)
    amrex::Real phi_s_const;     ///< Base slope factor (before terrain effects)
    amrex::Real U_max_ftmin;     ///< MEWS wind speed cap [ft/min]
    amrex::Real wind_conv;       ///< m/s to ft/min conversion factor
    amrex::Real ros_conv;        ///< ft/min to m/s conversion factor
    amrex::Real I_R;             ///< Reaction intensity [BTU/ft²/min]
};

/**
 * @brief Rothermel fuel/moisture coefficients, evaluable per cell on the device
 *
 * The body of compute_rothermel_params(): every quantity that depends on the
 * fuel and the dead moistures but not on wind or slope. Host callers build one
 * state per step through compute_rothermel_params(); the kernels build one per
 * cell from the cell's own moistures with erf.fire.rothermel_cell_moisture
 * (RothermelCellInputs), the way BEHAVE does.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RothermelComputed rothermel_coefficients(const FuelModelParams& fp,
                                         amrex::Real moisture_1hr,
                                         amrex::Real moisture_10hr,
                                         amrex::Real moisture_100hr,
                                         bool use_wind_limit = true,
                                         int wind_limit_mode = fire_wind_limit::rothermel,
                                         bool use_rothermel_a_formula = false,
                                         bool wrf_bmst_compat = false) noexcept
{
    using amrex::Real;
    RothermelComputed rc;

    // Unit conversions
    const Real FT_MIN_TO_M_S = 0.00508;

    // Standard mineral contents (Rothermel 1972)
    const Real S_T = 0.0555;   // Total mineral content
    const Real S_e = 0.010;    // Effective mineral content

    // ===================================================================
    // 1. Total fuel load and weighted moisture (single-class path)
    // ===================================================================
    Real w_0 = fp.w_d1 + fp.w_d10 + fp.w_d100 + fp.w_lh + fp.w_lw;  // total oven-dry load [lb/ft²]

    // Compute weighted dead fuel moisture
    Real w_d = fp.w_d1 + fp.w_d10 + fp.w_d100;
    Real M_f = 0.0;  // weighted fuel moisture fraction
    // floored dead load before the branch (a live-only custom fuel has
    // w_d = 0 and the unselected 0/0 would be speculated under the traps)
    const Real inv_w_d = 1.0 / amrex::max(w_d, Real(1.0e-6));
    if (w_d > 1.0e-6) {
        Real r_d1 = fp.w_d1 * inv_w_d;
        Real r_d10 = fp.w_d10 * inv_w_d;
        Real r_d100 = fp.w_d100 * inv_w_d;
        M_f = r_d1 * moisture_1hr + r_d10 * moisture_10hr + r_d100 * moisture_100hr;
    }

    // WRF-Fire comparison (erf.fire.wrf_bmst_compat): WRF-Fire takes its fuel
    // load as wet mass and removes the water before Rothermel,
    // fuelloadm = (1-bmst)*fgip, bmst = fmc_g/(1+fmc_g)
    // (module_fr_fire_phys.F). The deflation is applied to w_0 here, before
    // w_n, rho_b, beta, I_R, xi and R0 are formed, as WRF-Fire uses the
    // deflated load for all of them. M_f is the dead-load-weighted moisture.
    if (wrf_bmst_compat) {
        const Real bmst = M_f / (1.0 + M_f);
        w_0 *= (1.0 - bmst);
    }

    // ===================================================================
    // 2. Net fuel load (Eq. 24)
    // ===================================================================
    Real w_n = w_0 * (1.0 - S_T);

    // ===================================================================
    // 3. Bulk density and packing ratio (Eq. 24)
    // ===================================================================
    Real rho_b = w_0 / fp.delta;  // lb/ft³
    Real beta = rho_b / fp.rho_p;

    // ===================================================================
    // 4-7. Reaction velocity components (Eqs. 36-38)
    // ===================================================================
    // Use sigma_d1 as characteristic SAV
    Real sigma = fp.sigma_d1;
    sigma = amrex::max(static_cast<amrex::Real>(sigma), static_cast<amrex::Real>(100.0));  // Minimum SAV guard

    Real beta_op = 3.348 * std::pow(sigma, -0.8189);           // Eq. 37: optimum packing ratio
    Real sigma_1p5 = std::pow(sigma, 1.5);
    Real Gamma_max = sigma_1p5 / (495.0 + 0.0594 * sigma_1p5); // Eq. 36: maximum reaction velocity
    // A coefficient (Eq. 39): Albini's (1976) reformulation (ERF's original
    // form) or WRF-Fire's original Rothermel (1972) form, selected by
    // erf.fire.reaction_velocity_formula. The WRF form has a pole at
    // sigma ~= 67 ft^-1 = (7.27/4.774)^10, below which A goes negative; the
    // minimum SAV guard above keeps sigma >= 100 here, safely above it.
    Real A = use_rothermel_a_formula
        ? 1.0 / (4.774 * std::pow(sigma, 0.1) - 7.27)
        : 133.0 * std::pow(sigma, -0.7913);
    Real beta_ratio = beta / beta_op;
    Real Gamma_prime = Gamma_max * std::pow(beta_ratio, A) * std::exp(A * (1.0 - beta_ratio)); // Eq. 38

    // ===================================================================
    // 8-10. Moisture damping (Eq. 29) and mineral damping (Eq. 30)
    // ===================================================================
    Real rm = amrex::min(static_cast<amrex::Real>(M_f / fp.Mx), static_cast<amrex::Real>(1.0));
    Real eta_M = amrex::max(0.0, 1.0 - 2.59*rm + 5.11*rm*rm - 3.52*rm*rm*rm);  // Eq. 29
    Real eta_s = 0.174 * std::pow(S_e, -0.19);                                  // Eq. 30

    // ===================================================================
    // 11. Reaction intensity (Eq. 27)
    // ===================================================================
    Real I_R = Gamma_prime * w_n * fp.heat_content * eta_M * eta_s;
    I_R = amrex::max(static_cast<amrex::Real>(I_R), static_cast<amrex::Real>(0.01));

    // ===================================================================
    // 12-15. Propagating flux ratio, heating number, heat of preignition, and R0 (Eqs. 1, 12, 14, 42)
    // ===================================================================
    Real xi = std::exp((0.792 + 0.681 * std::sqrt(sigma)) * (beta + 0.1))
              / (192.0 + 0.2595 * sigma);                                      // Eq. 42
    Real eps_h = std::exp(-138.0 / sigma);                                     // Eq. 14
    Real Q_ig = 250.0 + 1116.0 * M_f;                                          // Eq. 12: heat of preignition
    Real R0_ft_min = (I_R * xi) / (rho_b * eps_h * Q_ig);                      // Eq. 1: no-wind ROS [ft/min]

    // ===================================================================
    // 16-18. Wind factor coefficients (Eqs. 47-49)
    // ===================================================================
    Real C = 7.47 * std::exp(-0.133 * std::pow(sigma, 0.55));           // Eq. 47 (Rothermel 1972, raw sigma path)
    Real B = 0.02526 * std::pow(sigma, 0.54);                           // Eq. 48
    Real E = 0.715 * std::exp(-3.59e-4 * sigma);                        // Eq. 49

    // ===================================================================
    // 19. Packing ratio wind factor
    // ===================================================================
    Real beta_ratio_E = std::pow(beta_ratio, -E);

    // ===================================================================
    // 20. Slope factor coefficient (Eq. 51)
    // ===================================================================
    Real phi_s_const = 5.275 * std::pow(beta, -0.3);

    // ===================================================================
    // 21. Wind limit
    // ===================================================================
    // Rothermel (1972) Eq. 87 caps the WIND SPEED at U_max [ft/min] = 0.9 I_R
    // [BTU/ft²/min] (the "wind limit" of Andrews 2018), the default
    // (erf.fire.wind_limit = "rothermel"). Until October 2026 this code used
    // its own fuel-class rule instead, an absolute cap on the midflame wind
    // (300 ft/min for sigma > 1000 ft⁻¹, 500 ft/min otherwise), which capped
    // short grass above a 4.2 m/s reference wind at 2.3x below Rothermel's
    // limit; it is kept as erf.fire.wind_limit = "fuel_class".
    // erf.fire.use_wind_limit = false removes the cap (rothermel_wind_cap_ftmin).
    Real U_max_ftmin = rothermel_wind_cap_ftmin(sigma, I_R, use_wind_limit, wind_limit_mode);

    // ===================================================================
    // Store results in RothermelComputed
    // ===================================================================
    rc.R0           = R0_ft_min * FT_MIN_TO_M_S;  // Convert ft/min → m/s
    rc.C            = C;
    rc.B            = B;
    rc.beta_ratio_E = beta_ratio_E;
    rc.beta         = beta;
    rc.phi_s_const  = phi_s_const;
    rc.U_max_ftmin  = U_max_ftmin;
    rc.wind_conv    = 196.85;    // m/s → ft/min
    rc.ros_conv     = 1.0;       // 1, not a unit factor: rc.R0 is already in m/s
    rc.I_R          = I_R;

    return rc;
}


/**
 * @brief Compute Rothermel fuel/moisture coefficients
 *
 * Precomputes all Rothermel intermediate quantities that depend only on
 * fuel type and moisture content (not on wind/slope).
 *
 * @param[in] fp FuelModelParams for selected fuel model
 * @param[in] use_wind_limit  Apply the wind limit (erf.fire.use_wind_limit)
 * @param[in] wind_limit_mode Which limit (erf.fire.wind_limit): fire_wind_limit::rothermel
 *                            (U_max = 0.9 I_R, default) or ::fuel_class (300/500 ft/min)
 * @param[in] use_rothermel_a_formula  Select the reaction-velocity exponent A:
 *                        false (default) Albini's (1976, GTR-INT-30) form,
 *                        A = 133*sigma^-0.7913 (ERF's original behavior);
 *                        true Rothermel's (1972) Eq. 39, used by WRF-Fire,
 *                        A = 1/(4.774*sigma^0.1-7.27)
 *                        (erf.fire.reaction_velocity_formula = "rothermel").
 *                        The WRF form has a pole at sigma ~= 67 ft^-1; this
 *                        function's minimum-SAV guard (sigma >= 100) keeps
 *                        every call safely above it.
 * @param[in] wrf_bmst_compat  Deflate the total dry fuel load w_0 by
 *                        (1-bmst), bmst = M_f/(1+M_f), before it feeds
 *                        w_n/rho_b/beta/I_R/xi/R0, as WRF-Fire's
 *                        fuelloadm = (1-bmst)*fgip (module_fr_fire_phys.F),
 *                        which treats its (Anderson-derived) fgip as a wet
 *                        mass and strips adsorbed water before Rothermel.
 *                        Default false: ERF's w_d1/w_d10/... are already
 *                        oven-dry, so this is a WRF-compatibility knob, not
 *                        a physical correction (erf.fire.wrf_bmst_compat).
 * @return RothermelComputed with precomputed coefficients
 *
 * Implements the full Rothermel model:
 * - Reaction intensity I_R from fuel load, SAV, moisture, heat content
 * - Propagating flux ratio Ξ and damping coefficients
 * - No-wind ROS R0 from I_R and fuel characteristics
 * - Wind and slope factor coefficients C, B, E
 * - Wind limit U_max_ftmin (none when use_wind_limit is false)
 */
RothermelComputed compute_rothermel_params(const FuelModelParams& fp,
                                           amrex::Real moisture_1hr,
                                           amrex::Real moisture_10hr,
                                           amrex::Real moisture_100hr,
                                           bool use_wind_limit = true,
                                           int wind_limit_mode = fire_wind_limit::rothermel,
                                           bool use_rothermel_a_formula = false,
                                           bool wrf_bmst_compat = false);

/**
 * @brief Compute rate-of-spread for a single fire cell (GPU kernel)
 *
 * Computes ROS = R0 * (1 + φ_w + φ_s) where:
 * - φ_w is wind factor (uses effective midflame wind after WAF and terrain)
 * - φ_s is the slope factor of the slope magnitude (Rothermel Eq. 51)
 * - Includes the wind-speed cap on the effective wind speed
 *
 * @param[in] ux_eff, uy_eff   Effective midflame wind [m/s] after WAF and terrain
 * @param[in] sx, sy           Terrain slopes [dz/dx, dz/dy]
 * @param[in] rc               RothermelComputed coefficients
 * @param[in] cos_theta_wind   Cosine between the wind and the spread direction,
 *                             applied to phi_w after it is computed from
 *                             |(ux_eff, uy_eff)|; 1 (no-op) for the isotropic
 *                             call sites. Lets a caller (see
 *                             ERF_DirectionalRos.H's DIRECTIONAL_WIND_COUPLING_ADVECTIVE)
 *                             exponentiate phi_w from the raw wind speed and
 *                             scale by this cosine afterward, rather than
 *                             passing an already-projected wind in ux_eff/uy_eff.
 * @param[in] cos_theta_slope  Cosine between the slope gradient and the spread
 *                             direction, applied to phi_s the same way; 1
 *                             (no-op) for the isotropic call sites.
 * @return ROS [m/s]
 *
 * Wind factor (Rothermel Eq. 47):
 *   φ_w = C * U_eff^B * (β/β_op)^-E   where U_eff is the effective wind speed [ft/min]
 *   times cos_theta_wind (1 for the isotropic call sites)
 *
 * Slope factor (Rothermel Eq. 51):
 *   φ_s = 5.275 β^-0.3 tan²(θ) with tan²θ = |∇z|² capped at 32, times
 *   cos_theta_slope (1 for the isotropic call sites); never negative. Which
 *   way the slope faces relative to the spread direction enters only through
 *   the directional callers (ERF_DirectionalRos.H), not here.
 *
 * Wind-speed cap:
 *   - U_eff above U_max_ftmin is replaced by U_max_ftmin in the wind factor
 *   - U_max_ftmin is Rothermel's U_max = 0.9 I_R (Eq. 87; Andrews 2018 "wind
 *     limit") with erf.fire.wind_limit = "rothermel" (default), or this code's
 *     earlier fuel-class rule (300 ft/min for sigma > 1000 1/ft, 500 ft/min
 *     otherwise) with "fuel_class"; see rothermel_wind_cap_ftmin
 *   - With erf.fire.use_wind_limit = false U_max_ftmin is the largest Real
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rothermel_ros_cell(
    amrex::Real ux_eff, amrex::Real uy_eff,
    amrex::Real sx, amrex::Real sy,
    const RothermelComputed& rc,
    amrex::Real cos_theta_wind = amrex::Real(1.0),
    amrex::Real cos_theta_slope = amrex::Real(1.0)) noexcept
{
    // Effective wind speed [m/s]
    amrex::Real U_eff = std::sqrt(ux_eff*ux_eff + uy_eff*uy_eff);

    // Wind factor: φ_w = C * U_eff^B * (β/β_opt)^E
    // Convert to ft/min for Rothermel calculation
    amrex::Real U_eff_ftmin = U_eff * rc.wind_conv;

    // Apply MEWS cap
    U_eff_ftmin = amrex::min(U_eff_ftmin, rc.U_max_ftmin);

    // Slope magnitude and squared tangent (for Rothermel Eq. 51)
    amrex::Real tan_slope_sq = sx*sx + sy*sy;

    // Cap at tan²(80°) ≈ 32.16; steeper than any realistic or surveyed terrain.
    // Prevents corrupted boundary slopes from producing unphysical ROS.
    tan_slope_sq = amrex::min(tan_slope_sq, amrex::Real(32.0));

    // Slope factor (Rothermel Eq. 51): φ_s = 5.275 * β^(-0.3) * tan²(φ)
    amrex::Real phi_s = rc.phi_s_const * tan_slope_sq;

    // Wind factor
    amrex::Real phi_w = rc.C * std::pow(U_eff_ftmin, rc.B) * rc.beta_ratio_E;

    // The advective wind coupling (erf.fire.directional_wind_coupling =
    // "advective") couples direction in by scaling the already-exponentiated
    // wind/slope excess by the cosine to the spread direction, rather than
    // projecting (ux_eff,uy_eff) before raising it to the B power -- matching
    // WRF-Fire's fire_ros (module_fr_fire_phys.F). cos_theta_* = 1 (the
    // default) reproduces the omnidirectional rate exactly, unchanged from
    // before this parameter existed; a caller doing directional spread may
    // pass the raw wind speed as (ux_eff,uy_eff) and the true cosine here
    // instead, which keeps R linear in cos(theta) and so a convex (stadium)
    // support function instead of cos^B(theta) -- see ERF_DirectionalRos.H's
    // file comment.
    phi_w *= amrex::max(cos_theta_wind, amrex::Real(0.0));
    phi_s *= amrex::max(cos_theta_slope, amrex::Real(0.0));

    // Rate of spread: R = R0 * (1 + φ_w + φ_s)
    amrex::Real ROS = rc.R0 * (1.0 + phi_w + phi_s);

    return amrex::max(static_cast<amrex::Real>(ROS), static_cast<amrex::Real>(0.0));
}

/**
 * @brief Compute rate-of-spread field on fire grid
 *
 * Applies rothermel_ros_cell() to all cells on the fire grid.
 *
 * @param[out] fire_ros        Output ROS [m/s]
 * @param[in]  fire_wind       Effective midflame wind after WAF + terrain [m/s]
 * @param[in]  fire_slopes     Terrain slopes [dz/dx, dz/dy]
 * @param[in]  rc              RothermelComputed coefficients
 */
void compute_ros_field(
    amrex::MultiFab& fire_ros,
    const amrex::MultiFab& fire_wind,
    const amrex::MultiFab& fire_slopes,
    const RothermelComputed& rc);

/// Components of the packed per-cell coefficient field (pack_rothermel)
inline constexpr int ROTHERMEL_RC_NCOMP = 10;

/// Write a coefficient set into component 0.. of a cell of a packed field
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void pack_rothermel(const RothermelComputed& rc, amrex::Array4<amrex::Real> const& a, int i, int j) noexcept
{
    a(i, j, 0, 0) = rc.R0;          a(i, j, 0, 1) = rc.C;           a(i, j, 0, 2) = rc.B;
    a(i, j, 0, 3) = rc.beta_ratio_E; a(i, j, 0, 4) = rc.beta;       a(i, j, 0, 5) = rc.phi_s_const;
    a(i, j, 0, 6) = rc.U_max_ftmin; a(i, j, 0, 7) = rc.wind_conv;   a(i, j, 0, 8) = rc.ros_conv;
    a(i, j, 0, 9) = rc.I_R;
}

/// Read a cell's coefficient set back from a packed field
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
RothermelComputed unpack_rothermel(amrex::Array4<const amrex::Real> const& a, int i, int j) noexcept
{
    RothermelComputed rc;
    rc.R0 = a(i, j, 0, 0);           rc.C = a(i, j, 0, 1);         rc.B = a(i, j, 0, 2);
    rc.beta_ratio_E = a(i, j, 0, 3); rc.beta = a(i, j, 0, 4);      rc.phi_s_const = a(i, j, 0, 5);
    rc.U_max_ftmin = a(i, j, 0, 6);  rc.wind_conv = a(i, j, 0, 7); rc.ros_conv = a(i, j, 0, 8);
    rc.I_R = a(i, j, 0, 9);
    return rc;
}

/**
 * @struct RothermelCellInputs
 * @brief The per-cell Rothermel coefficients (erf.fire.rothermel_cell_moisture
 *        with erf.fire.moisture_dynamic): a packed field of ROTHERMEL_RC_NCOMP
 *        components built once per fire step from each cell's own dead
 *        moistures (build_cell_rothermel_coefficients), which the kernels
 *        read instead of the uniform set or the per-fuel table. Inactive
 *        (rc_cell null) the kernels take the coefficients they are handed.
 */
struct RothermelCellInputs
{
    const amrex::MultiFab* rc_cell = nullptr;   ///< packed coefficients per cell
    bool active() const { return rc_cell != nullptr; }
};

/**
 * @brief Build the packed per-cell coefficient field from the moisture field
 *
 * Each cell's coefficients come from its own dead moistures (components 0-2
 * of fuel_mc, clamped to [0.01, 0.40] as the domain-mean rebuild clamps them)
 * and its fuel: the code of fuel_model through the slot table when fuel_model
 * is given (erf.fire.rothermel_per_fuel on a spatial fuel map), the uniform
 * fuel otherwise; a code outside the table keeps the uniform fuel, as the
 * per-fuel table path does. A cell whose fuel has no load (a non-burnable
 * code) gets the zeroed set and cannot spread. Once per step, not per kernel call: the
 * directional stages read the field at every substep.
 *
 * @param[out] rc_cell    ROTHERMEL_RC_NCOMP-component field on the fire grid
 * @param[in]  fuel_mc    Moisture field (components 0-2 the dead classes)
 * @param[in]  fuel_model Per-cell fuel code, or nullptr for the uniform fuel
 * @param[in]  fp_tbl     Device fuel properties by slot (with fuel_model)
 * @param[in]  fp_tbl_size Entries in fp_tbl
 * @param[in]  fuel_set   FUEL_SET_ANDERSON13 or FUEL_SET_SCOTT_BURGAN40
 * @param[in]  M_live     Live moisture for the Scott-Burgan transfer
 * @param[in]  fp_uniform The uniform fuel
 * @param[in]  use_wind_limit, wind_limit_mode  erf.fire.use_wind_limit, wind_limit
 * @param[in]  use_rothermel_a_formula, wrf_bmst_compat  erf.fire.reaction_velocity_formula
 *                        = "rothermel" and erf.fire.wrf_bmst_compat, as for
 *                        compute_rothermel_params(), so the per-cell path
 *                        (rothermel_cell_moisture) honours them too
 */
void build_cell_rothermel_coefficients(amrex::MultiFab& rc_cell,
                                       const amrex::MultiFab& fuel_mc,
                                       const amrex::MultiFab* fuel_model,
                                       const FuelModelParams* fp_tbl,
                                       int fp_tbl_size,
                                       int fuel_set,
                                       amrex::Real M_live,
                                       const FuelModelParams& fp_uniform,
                                       bool use_wind_limit,
                                       int wind_limit_mode,
                                       bool use_rothermel_a_formula = false,
                                       bool wrf_bmst_compat = false);

/// Entries in the per-fuel Rothermel table, one per fuel table slot.
inline constexpr int ROTHERMEL_TABLE_SIZE = 70;   ///< FUEL_SLOT_COUNT: 0 non-burnable, 1-13 Anderson, 14-53 Scott-Burgan, 54-69 deck-defined
///< ERF_FuelModels.H static_asserts this against FUEL_SLOT_COUNT; this header cannot include it (it is included by it).

/**
 * @brief Build a per-fuel-code lookup table of Rothermel coefficients
 *
 * Mirrors build_fuel_balbi_table(): indexed by fuel table slot, with index 0
 * holding a zeroed entry (R0 = 0, no spread), 1-13 the Anderson models, 14-53
 * the Scott-Burgan models and 54-69 the deck-defined ones, at the given dead
 * fuel moistures.
 *
 * @param[in] moisture_1hr   1-hour dead fuel moisture [fraction]
 * @param[in] moisture_10hr  10-hour dead fuel moisture [fraction]
 * @param[in] moisture_100hr 100-hour dead fuel moisture [fraction]
 * @param[in] use_wind_limit Apply the wind limit (erf.fire.use_wind_limit)
 * @param[in] fp_tbl         Fuel properties of every slot
 *                           (build_fuel_params_slot_table), which is how the
 *                           deck-defined models of ERF_CustomFuel.H reach the
 *                           coefficients; nullptr takes the published sets only
 * @param[in] wind_limit_mode Which wind limit (erf.fire.wind_limit), fire_wind_limit::*
 * @param[in] use_rothermel_a_formula Select the reaction-velocity A formula;
 *                        see compute_rothermel_params()
 *                        (erf.fire.reaction_velocity_formula = "rothermel")
 * @param[in] wrf_bmst_compat Deflate w_0 by (1-bmst) before Rothermel; see
 *                        compute_rothermel_params() (erf.fire.wrf_bmst_compat)
 * @return host-side vector of size ROTHERMEL_TABLE_SIZE
 */
std::vector<RothermelComputed> build_fuel_rothermel_table(
    amrex::Real moisture_1hr,
    amrex::Real moisture_10hr,
    amrex::Real moisture_100hr,
    int fuel_set = 0,
    amrex::Real moisture_live = -1.0,
    bool use_wind_limit = true,
    const FuelModelParams* fp_tbl = nullptr,
    int wind_limit_mode = fire_wind_limit::rothermel,
    bool use_rothermel_a_formula = false,
    bool wrf_bmst_compat = false);

/**
 * @brief Compute the rate-of-spread field with per-cell Rothermel coefficients
 *
 * Cells whose fuel code lies inside the table use that code's coefficients;
 * any other cell, and every cell when no table is given, uses rc_default.
 * With fuel_model or table null this is identical to the uniform overload.
 *
 * @param[out] fire_ros     Output ROS [m/s]
 * @param[in]  fire_wind    Effective midflame wind after WAF + terrain [m/s]
 * @param[in]  fire_slopes  Terrain slopes [dz/dx, dz/dy]
 * @param[in]  rc_default   Coefficients for cells outside the table
 * @param[in]  fuel_model   Per-cell fuel code MultiFab, or nullptr
 * @param[in]  table        Device pointer to the per-fuel table, or nullptr
 * @param[in]  table_size   Entries in table
 * @param[in]  fuel_set     Fuel model set the codes belong to (ERF_FuelModels.H); with
 *                          the Anderson set a code outside 0-13 falls to rc_default as before
 * @param[in]  cell_mc      With fuel_mc set, rebuild the coefficients per cell from the
 *                          cell's own dead moistures (and its own fuel on a map) instead
 *                          of reading rc_default / the table
 */
void compute_ros_field(
    amrex::MultiFab& fire_ros,
    const amrex::MultiFab& fire_wind,
    const amrex::MultiFab& fire_slopes,
    const RothermelComputed& rc_default,
    const amrex::MultiFab* fuel_model,
    const RothermelComputed* table,
    int table_size,
    int fuel_set = 0,
    const RothermelCellInputs* cell_mc = nullptr);

#endif
