/**
 * \file ERF_FireWindExtract.cpp
 *
 * \brief Implementation of wind extraction and WAF/terrain corrections.
 */

#include "ERF_FireWindExtract.H"
#include <AMReX_ParallelDescriptor.H>
#include <cmath>

void fill_fire_wind_from_most(
    amrex::MultiFab& fire_wind_ref,
    const amrex::MultiFab& ustar_mf,
    const amrex::MultiFab& z0_mf,
    const amrex::MultiFab& olen_mf,
    const amrex::MultiFab& uavg_mf,
    const amrex::MultiFab& vavg_mf,
    const FireGrid& fg,
    amrex::Real z_target)
{
    using amrex::ParallelFor;

    int C = fg.C;

    // Iterate over fire grid boxes
    for (amrex::MFIter mfi(fire_wind_ref); mfi.isValid(); ++mfi) {
        const amrex::Box& fire_box = mfi.tilebox();
        auto fire_u = fire_wind_ref.array(mfi, 0);  // u component
        auto fire_v = fire_wind_ref.array(mfi, 1);  // v component

        // Source arrays (on atmospheric grid, level 0)
        auto ustar_arr = ustar_mf.const_array(mfi);
        auto z0_arr = z0_mf.const_array(mfi);
        auto olen_arr = olen_mf.const_array(mfi);
        auto uavg_arr = uavg_mf.const_array(mfi, 0);
        auto vavg_arr = vavg_mf.const_array(mfi, 1);

        // GPU kernel: fill fire wind
        ParallelFor(fire_box, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            int i_f = iv[0];
            int j_f = iv[1];
            int k_f = iv[2];

            // Map fire cell to atmospheric column
            int i_a = i_f / C;
            int j_a = j_f / C;
            int k_a = 0;  // Always level 0

            // Read MOST parameters from atmospheric column
            amrex::Real ustar = ustar_arr(i_a, j_a, k_a);
            amrex::Real z0 = z0_arr(i_a, j_a, k_a);
            amrex::Real olen = olen_arr(i_a, j_a, k_a);

            // Read wind direction from MAC-averaged components
            amrex::Real u_avg = uavg_arr(i_a, j_a, k_a);
            amrex::Real v_avg = vavg_arr(i_a, j_a, k_a);

            // Compute wind speed at z_target using MOST
            amrex::Real U_ref = most_wind_at_height(ustar, z0, olen, z_target);

            // Extract wind direction and normalize
            amrex::Real wind_mag = std::sqrt(u_avg * u_avg + v_avg * v_avg);
            amrex::Real cos_dir = 0.0;
            amrex::Real sin_dir = 0.0;
            if (wind_mag > 1.0e-6) {
                cos_dir = u_avg / wind_mag;
                sin_dir = v_avg / wind_mag;
            }

            // Populate fire wind components
            fire_u(i_f, j_f, k_f) = U_ref * cos_dir;
            fire_v(i_f, j_f, k_f) = U_ref * sin_dir;
        });
    }
}

void compute_terrain_curvature(
    amrex::MultiFab& curvature,
    const amrex::MultiFab& fire_slopes,
    const amrex::Geometry& geom_fire)
{
    using amrex::ParallelFor;

    amrex::Real dx = geom_fire.CellSize(0);
    amrex::Real dy = geom_fire.CellSize(1);
    amrex::Real dxinv = 1.0 / dx;
    amrex::Real dyinv = 1.0 / dy;

    // Iterate over curvature boxes
    for (amrex::MFIter mfi(curvature); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.tilebox();
        auto curv_arr = curvature.array(mfi);
        auto slope_arr = fire_slopes.const_array(mfi);

        // Compute curvature from slopes using finite differences
        // Curvature approximation: sum of second derivatives in x and y
        ParallelFor(box, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            int i = iv[0];
            int j = iv[1];
            int k = iv[2];

            // Avoid boundaries (use 0 curvature at boundaries)
            if (i == 0 || j == 0 || i >= curvature.nComp() || j >= curvature.nComp()) {
                curv_arr(i, j, k) = 0.0;
                return;
            }

            // Get slopes: comp 0 = dz/dx, comp 1 = dz/dy
            // Compute d2z/dx2 and d2z/dy2
            amrex::Real dzx_c = slope_arr(i, j, k, 0);
            amrex::Real dzx_l = slope_arr(i-1, j, k, 0);
            amrex::Real dzx_r = slope_arr(i+1, j, k, 0);
            amrex::Real d2z_dx2 = (dzx_r - 2.0*dzx_c + dzx_l) * dxinv * dxinv;

            amrex::Real dzy_c = slope_arr(i, j, k, 1);
            amrex::Real dzy_b = slope_arr(i, j-1, k, 1);
            amrex::Real dzy_t = slope_arr(i, j+1, k, 1);
            amrex::Real d2z_dy2 = (dzy_t - 2.0*dzy_c + dzy_b) * dyinv * dyinv;

            // Total curvature (mean curvature approximation)
            curv_arr(i, j, k) = d2z_dx2 + d2z_dy2;
        });
    }
}

void apply_farsite_terrain_wind(
    amrex::MultiFab& fire_wind,
    const amrex::MultiFab& fire_slopes,
    const amrex::MultiFab& curvature,
    amrex::Real k_ridge, amrex::Real k_shelter,
    amrex::Real k_valley, amrex::Real k_deflect,
    amrex::Real min_curv)
{
    using amrex::ParallelFor;

    // Iterate over fire wind boxes
    for (amrex::MFIter mfi(fire_wind); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.tilebox();
        auto wind_arr = fire_wind.array(mfi);
        auto slope_arr = fire_slopes.const_array(mfi);
        auto curv_arr = curvature.const_array(mfi);

        // Apply terrain wind corrections
        ParallelFor(box, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            int i = iv[0];
            int j = iv[1];
            int k = iv[2];

            // Read slopes and curvature
            amrex::Real sx = slope_arr(i, j, k, 0);  // dz/dx
            amrex::Real sy = slope_arr(i, j, k, 1);  // dz/dy
            amrex::Real curv = curv_arr(i, j, k, 0);

            // Compute slope magnitude and aspect
            amrex::Real slope_mag = std::sqrt(sx*sx + sy*sy);
            amrex::Real aspect = std::atan2(sy, sx);  // Wind direction angle

            // Read current wind
            amrex::Real u = wind_arr(i, j, k, 0);
            amrex::Real v = wind_arr(i, j, k, 1);
            amrex::Real wind_mag = std::sqrt(u*u + v*v);
            amrex::Real wind_dir = std::atan2(v, u);

            amrex::Real wind_mod = 1.0;  // Wind modification factor
            amrex::Real dir_deflect = 0.0;  // Direction deflection [radians]

            // Ridge/valley effects (positive curvature = convex/ridge, negative = concave/valley)
            if (std::abs(curv) > min_curv) {
                if (curv > 0.0) {
                    // Ridge: wind speed-up
                    wind_mod *= k_ridge;
                } else {
                    // Valley: wind channeling (could enhance or reduce depending on aspect)
                    wind_mod *= k_valley;
                }
            }

            // Lee-side shelter effect (wind component perpendicular to slope)
            // If wind is blowing toward slope (upslope), shelter on lee side reduces flow
            amrex::Real wind_perp = std::sin(wind_dir - aspect) * wind_mag;
            if (wind_perp < 0.0) {  // Wind blowing into hill
                wind_mod *= k_shelter;
            }

            // Directional deflection: turn wind along contours
            // deflect_amount = k_deflect * sin(aspect - wind_dir) * tan(slope)
            // cap deflection at ±45 degrees
            if (slope_mag > 1.0e-6) {
                amrex::Real tan_slope = std::tan(std::atan(slope_mag));
                dir_deflect = k_deflect * std::sin(aspect - wind_dir) * tan_slope;
                // Cap deflection at ±45° (π/4 radians)
                constexpr amrex::Real max_deflect = 0.785398163;  // π/4
                dir_deflect = std::max(-max_deflect, std::min(dir_deflect, max_deflect));
            }

            // Apply modifications
            amrex::Real new_wind_mag = wind_mod * wind_mag;
            amrex::Real new_wind_dir = wind_dir + dir_deflect;

            wind_arr(i, j, k, 0) = new_wind_mag * std::cos(new_wind_dir);
            wind_arr(i, j, k, 1) = new_wind_mag * std::sin(new_wind_dir);
        });
    }
}
