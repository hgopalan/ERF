/**
 * \file ERF_Rothermel.cpp
 *
 * \brief Implementation of Rothermel fire spread model and fuel database.
 */

#include "ERF_Rothermel.H"
#include <AMReX_ParallelFor.H>
#include <cmath>

// ============================================================================
// Anderson FBFM13 Fuel Database
// ============================================================================

/**
 * \brief Anderson (1982) FBFM13 fuel model database.
 *
 * All 13 standard fuel models with parameters from Anderson, H.E. (1982),
 * "Aids to determining fuel models for estimating fire behavior",
 * USDA Forest Service General Technical Report INT-122.
 *
 * Models are numbered 1-13. Model 5 (GR1) is commonly used for grassland.
 */
FuelModelParams get_anderson_fuel_params(int fuel_model_id)
{
    FuelModelParams fp;

    switch (fuel_model_id) {
        case 1:  // Short grass
            fp.w_d1 = 0.74;
            fp.w_d10 = 0.0;
            fp.w_d100 = 0.0;
            fp.w_lh = 0.0;
            fp.w_lw = 0.0;
            fp.sigma_d1 = 12000.0;
            fp.sigma_lh = 0.0;
            fp.sigma_lw = 0.0;
            fp.delta = 1.0 / 12.0;  // 1 inch = 1/12 ft
            fp.Mx = 12.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        case 2:  // Timber litter
            fp.w_d1 = 2.0;
            fp.w_d10 = 1.0;
            fp.w_d100 = 0.5;
            fp.w_lh = 0.0;
            fp.w_lw = 0.0;
            fp.sigma_d1 = 9000.0;
            fp.sigma_lh = 0.0;
            fp.sigma_lw = 0.0;
            fp.delta = 2.0 / 12.0;
            fp.Mx = 15.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        case 3:  // Tall grass
            fp.w_d1 = 3.01;
            fp.w_d10 = 0.0;
            fp.w_d100 = 0.0;
            fp.w_lh = 0.0;
            fp.w_lw = 0.0;
            fp.sigma_d1 = 1500.0;
            fp.sigma_lh = 0.0;
            fp.sigma_lw = 0.0;
            fp.delta = 2.5 / 12.0;
            fp.Mx = 25.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        case 4:  // Chaparral
            fp.w_d1 = 5.01;
            fp.w_d10 = 4.51;
            fp.w_d100 = 3.56;
            fp.w_lh = 0.0;
            fp.w_lw = 7.52;
            fp.sigma_d1 = 1739.0;
            fp.sigma_lh = 0.0;
            fp.sigma_lw = 109.0;
            fp.delta = 6.0 / 12.0;
            fp.Mx = 20.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        case 5:  // Brush (GR1 - commonly used grassland model)
            fp.w_d1 = 1.0;
            fp.w_d10 = 0.5;
            fp.w_d100 = 0.0;
            fp.w_lh = 1.5;
            fp.w_lw = 0.5;
            fp.sigma_d1 = 2000.0;
            fp.sigma_lh = 1500.0;
            fp.sigma_lw = 1500.0;
            fp.delta = 2.0 / 12.0;
            fp.Mx = 20.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        case 6:  // Dormant brush
            fp.w_d1 = 1.5;
            fp.w_d10 = 2.0;
            fp.w_d100 = 1.25;
            fp.w_lh = 0.0;
            fp.w_lw = 2.0;
            fp.sigma_d1 = 1739.0;
            fp.sigma_lh = 0.0;
            fp.sigma_lw = 1500.0;
            fp.delta = 2.5 / 12.0;
            fp.Mx = 25.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        case 7:  // Southern rough
            fp.w_d1 = 1.13;
            fp.w_d10 = 1.87;
            fp.w_d100 = 1.5;
            fp.w_lh = 0.37;
            fp.w_lw = 0.37;
            fp.sigma_d1 = 1836.0;
            fp.sigma_lh = 1500.0;
            fp.sigma_lw = 1500.0;
            fp.delta = 2.5 / 12.0;
            fp.Mx = 20.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        case 8:  // Closed timber litter
            fp.w_d1 = 1.5;
            fp.w_d10 = 1.25;
            fp.w_d100 = 1.25;
            fp.w_lh = 0.0;
            fp.w_lw = 0.0;
            fp.sigma_d1 = 2000.0;
            fp.sigma_lh = 0.0;
            fp.sigma_lw = 0.0;
            fp.delta = 0.2;  // 2.4 inches
            fp.Mx = 30.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        case 9:  // Hardwood litter
            fp.w_d1 = 2.92;
            fp.w_d10 = 0.41;
            fp.w_d100 = 0.15;
            fp.w_lh = 0.0;
            fp.w_lw = 0.0;
            fp.sigma_d1 = 2500.0;
            fp.sigma_lh = 0.0;
            fp.sigma_lw = 0.0;
            fp.delta = 0.2;
            fp.Mx = 25.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        case 10:  // Timber blowdown
            fp.w_d1 = 3.01;
            fp.w_d10 = 16.48;
            fp.w_d100 = 2.41;
            fp.w_lh = 0.0;
            fp.w_lw = 0.0;
            fp.sigma_d1 = 1731.0;
            fp.sigma_lh = 0.0;
            fp.sigma_lw = 0.0;
            fp.delta = 1.0;  // 12 inches
            fp.Mx = 25.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        case 11:  // Light logging slash
            fp.w_d1 = 1.41;
            fp.w_d10 = 7.89;
            fp.w_d100 = 1.59;
            fp.w_lh = 0.0;
            fp.w_lw = 0.0;
            fp.sigma_d1 = 1564.0;
            fp.sigma_lh = 0.0;
            fp.sigma_lw = 0.0;
            fp.delta = 1.0 / 3.0;
            fp.Mx = 15.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        case 12:  // Medium logging slash
            fp.w_d1 = 1.92;
            fp.w_d10 = 12.83;
            fp.w_d100 = 7.21;
            fp.w_lh = 0.0;
            fp.w_lw = 0.0;
            fp.sigma_d1 = 1562.0;
            fp.sigma_lh = 0.0;
            fp.sigma_lw = 0.0;
            fp.delta = 0.4;
            fp.Mx = 20.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        case 13:  // Heavy logging slash
            fp.w_d1 = 1.41;
            fp.w_d10 = 21.09;
            fp.w_d100 = 26.34;
            fp.w_lh = 0.0;
            fp.w_lw = 0.0;
            fp.sigma_d1 = 1488.0;
            fp.sigma_lh = 0.0;
            fp.sigma_lw = 0.0;
            fp.delta = 0.5;
            fp.Mx = 25.0;
            fp.heat_content = 8000.0;
            fp.rho_p = 32.0;
            break;

        default:
            amrex::Abort("Unknown fuel model ID: " + std::to_string(fuel_model_id));
    }

    return fp;
}

// ============================================================================
// Rothermel Parameter Computation
// ============================================================================

RothermelComputed compute_rothermel_params(const FuelModelParams& fp,
                                           amrex::Real M_1hr,
                                           amrex::Real M_10hr,
                                           amrex::Real M_100hr)
{
    RothermelComputed rc;

    // Constants
    constexpr amrex::Real h = 8000.0;          // Heat yield [BTU/lb]
    constexpr amrex::Real rho_b_opt = 0.0315;  // Optimum bulk density [lb/ft^3]
    constexpr amrex::Real w_0_max = 0.33;      // Maximum loading [lb/ft^3]

    // Compute composite surface-to-volume ratio
    amrex::Real sigma = (fp.sigma_d1 * (fp.w_d1 + fp.w_d10 + fp.w_d100) +
                         fp.sigma_lh * fp.w_lh + fp.sigma_lw * fp.w_lw) /
                        (fp.w_d1 + fp.w_d10 + fp.w_d100 + fp.w_lh + fp.w_lw + 1.0e-12);

    // Compute bulk density
    amrex::Real w_0 = fp.w_d1 + fp.w_d10 + fp.w_d100 + fp.w_lh + fp.w_lw;
    amrex::Real rho_b = w_0 / fp.delta;

    // Packing ratio
    rc.beta = rho_b / fp.rho_p;
    amrex::Real beta_ratio = rc.beta / rho_b_opt;

    // Compute E exponent (Rothermel eq. 52)
    constexpr amrex::Real E = -0.01094;
    amrex::Real beta_ratio_E = std::pow(beta_ratio, E);
    rc.beta_ratio_E = beta_ratio_E;

    // Compute moisture of extinction adjustment (Rothermel eq. 27)
    amrex::Real M_e = fp.Mx * (1.0 - 2.59 * rc.beta + 5.11 * rc.beta * rc.beta -
                               3.52 * rc.beta * rc.beta * rc.beta);

    // Compute weighted moisture (Rothermel eq. 28)
    amrex::Real w_dead = fp.w_d1 + fp.w_d10 + fp.w_d100;
    amrex::Real w_live = fp.w_lh + fp.w_lw;
    amrex::Real M_x_dead = 0.0;
    if (w_dead > 1.0e-6) {
        M_x_dead = (fp.w_d1 * M_1hr + fp.w_d10 * M_10hr + fp.w_d100 * M_100hr) / w_dead;
    }

    // Dead fuel moisture ratio
    amrex::Real M_x_ratio = M_x_dead / M_e;
    M_x_ratio = std::max(0.0, std::min(M_x_ratio, 1.0));  // Clamp to [0,1]

    // Compute moisture damping factor
    constexpr amrex::Real mu_d_coeff = 1.0 - 2.59;
    amrex::Real mu_d = std::max(0.0, 1.0 - 1.5 * M_x_ratio + 0.85 * M_x_ratio * M_x_ratio);

    // Compute reaction intensity I_R (Rothermel eq. 40)
    rc.I_R = fp.heat_content * mu_d * 0.1 * sigma * w_0;  // Scaled for units

    // Compute no-wind, no-slope ROS R0 (Rothermel eq. 51)
    // R0 [ft/min] = (sigma^1.5) * (w_0 + 0.0555) * (192 + 0.2566*sigma) * I_R / (hc * rho_b)
    constexpr amrex::Real hc = 32.0;
    amrex::Real R0_ftmin = std::pow(sigma, 1.5) * (w_0 + 0.0555) * (192.0 + 0.2566 * sigma) *
                           rc.I_R / (hc * rho_b + 1.0e-12);
    rc.R0 = R0_ftmin * 0.00508;  // Convert ft/min to m/s

    // Compute wind coefficient C and exponent B (Rothermel eq. 52-53)
    rc.C = 7.5 * std::pow(rc.beta, -0.1) * std::exp(0.1 * sigma);
    rc.B = 0.02526 * std::pow(sigma, 0.54) * std::exp(-0.04 * sigma);

    // Compute slope factor (Rothermel eq. 68)
    constexpr amrex::Real s_phi = 5.275 * std::pow(rc.beta, -0.3);
    rc.phi_s_const = s_phi;

    // Unit conversions
    rc.wind_conv = 1.94384;  // m/s to ft/min
    rc.ros_conv = 0.00508;   // ft/min to m/s

    // MEWS cap (0.9 * I_R)
    constexpr amrex::Real mews_factor = 0.9;
    amrex::Real phi_w_max = mews_factor * rc.I_R;
    if (rc.C > 1.0e-12 && rc.B > 1.0e-12) {
        rc.U_max_ftmin = std::pow(phi_w_max / (rc.C * rc.beta_ratio_E), 1.0 / rc.B);
    } else {
        rc.U_max_ftmin = 1.0e6;  // No cap
    }

    return rc;
}

// ============================================================================
// Per-Cell ROS Kernel
// ============================================================================

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real rothermel_ros_cell(
    amrex::Real ux_eff, amrex::Real uy_eff,
    amrex::Real sx, amrex::Real sy,
    const RothermelComputed& rc) noexcept
{
    // Compute effective wind speed magnitude
    amrex::Real U_eff_ms = std::sqrt(ux_eff * ux_eff + uy_eff * uy_eff);
    amrex::Real U_eff_ftmin = U_eff_ms * rc.wind_conv;

    // Apply MEWS cap
    U_eff_ftmin = std::min(U_eff_ftmin, rc.U_max_ftmin);

    // Compute wind factor phi_w
    amrex::Real phi_w = 0.0;
    if (U_eff_ftmin > 1.0e-6) {
        phi_w = rc.C * std::pow(U_eff_ftmin, rc.B) * rc.beta_ratio_E;
    }

    // Compute slope magnitude and tangent
    amrex::Real slope_mag = std::sqrt(sx * sx + sy * sy);
    amrex::Real tan_slope = std::tan(std::atan(slope_mag));

    // Compute slope factor phi_s
    amrex::Real phi_s = rc.phi_s_const * tan_slope;

    // Compute ROS with wind and slope effects (Rothermel eq. 19)
    // R = R0 * (1 + phi_w) * (1 + phi_s)
    amrex::Real R_ftmin = rc.R0 / rc.ros_conv * (1.0 + phi_w) * (1.0 + phi_s);

    // Convert to m/s
    amrex::Real R_ms = R_ftmin * rc.ros_conv;

    return std::max(0.0, R_ms);
}

// ============================================================================
// MultiFab Kernel
// ============================================================================

void compute_ros_field(
    amrex::MultiFab& fire_ros,
    const amrex::MultiFab& fire_wind,
    const amrex::MultiFab& fire_slopes,
    const RothermelComputed& rc)
{
    using amrex::ParallelFor;

    for (amrex::MFIter mfi(fire_ros); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.tilebox();
        auto ros_arr = fire_ros.array(mfi);
        auto wind_arr = fire_wind.const_array(mfi);
        auto slope_arr = fire_slopes.const_array(mfi);

        ParallelFor(box, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            int i = iv[0];
            int j = iv[1];
            int k = iv[2];

            amrex::Real ux = wind_arr(i, j, k, 0);
            amrex::Real uy = wind_arr(i, j, k, 1);
            amrex::Real sx = slope_arr(i, j, k, 0);
            amrex::Real sy = slope_arr(i, j, k, 1);

            ros_arr(i, j, k) = rothermel_ros_cell(ux, uy, sx, sy, rc);
        });
    }
}
