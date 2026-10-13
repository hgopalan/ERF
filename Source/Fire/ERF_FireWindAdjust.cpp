#include <ERF_FireWindAdjust.H>

using namespace amrex;

void compute_terrain_curvature(
    MultiFab& curvature,
    const MultiFab& fire_slopes,
    const Geometry& geom_fire)
{
    // Curvature = d²z/dx² + d²z/dy²
    // For now, implement as a stub that computes from slope field
    // using finite differences

    Real dx = geom_fire.CellSize(0);
    Real dy = geom_fire.CellSize(1);

    for (MFIter mfi(curvature, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> curv = curvature.array(mfi);
        Array4<const Real> slopes = fire_slopes.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            int i = iv[0];
            int j = iv[1];
            int k = 0;

            // dz/dx at (i±1, j)
            Real dzdx_ip = slopes(i+1, j, k, 0);  // dz/dx component
            Real dzdx_im = slopes(i-1, j, k, 0);
            Real d2zdx2 = (dzdx_ip - dzdx_im) / (2.0 * dx);

            // dz/dy at (i, j±1)
            Real dzdy_jp = slopes(i, j+1, k, 1);  // dz/dy component
            Real dzdy_jm = slopes(i, j-1, k, 1);
            Real d2zdy2 = (dzdy_jp - dzdy_jm) / (2.0 * dy);

            curv(i, j, k) = d2zdx2 + d2zdy2;
        });
    }
}

void apply_farsite_terrain_wind(
    MultiFab& fire_wind_eff,
    const MultiFab& fire_slopes,
    const MultiFab& fire_curvature,
    Real k_ridge, Real k_shelter,
    Real k_valley, Real k_deflect)
{
    // Apply FARSITE terrain wind corrections per Finney (1998)
    // Ridge speed-up, lee sheltering, valley channeling, directional deflection

    for (MFIter mfi(fire_wind_eff, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> wind = fire_wind_eff.array(mfi);
        Array4<const Real> slopes = fire_slopes.array(mfi);
        Array4<const Real> curv = fire_curvature.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            int i = iv[0];
            int j = iv[1];
            int k = 0;

            // Read terrain properties
            Real sx = slopes(i, j, k, 0);  // dz/dx
            Real sy = slopes(i, j, k, 1);  // dz/dy
            Real curv_val = curv(i, j, k);
            Real ux = wind(i, j, k, 0);
            Real uy = wind(i, j, k, 1);

            // Compute magnitudes
            Real slope_mag = std::sqrt(sx*sx + sy*sy);
            Real wind_mag = std::sqrt(ux*ux + uy*uy);

            // Compute wind-upslope alignment: cos(angle between wind and slope gradient)
            // cos_upslope > 0 means wind pointing upslope (against gravity, climbing)
            // (divide by the floored product before the test: the unselected
            // 0/0 of a flat or calm cell would be speculated under the traps)
            const Real sw = slope_mag * wind_mag;
            const Real cos_raw = (sx*ux + sy*uy) / amrex::max(sw, Real(1.0e-10));
            Real cos_upslope = (sw > 1.0e-10) ? cos_raw : Real(0.0);

            // Classify terrain position and compute speed factor.
            // Slope-based classification uses wind-slope alignment (cos_upslope)
            // to identify windward (upslope) and lee (downslope) faces.
            // Curvature is used only for valley detection (concave terrain).
            // NOTE: uniform planar slopes have zero curvature, so curvature
            // must NOT be required for ridge/shelter classification.
            Real factor = 1.0;

            // Windward slope: wind climbing the slope (upslope face / ridge)
            if (slope_mag > 0.05 && cos_upslope > 0.0) {
                factor = k_ridge;
            }
            // Shelter: lee-side slope (wind blowing downslope)
            else if (slope_mag > 0.05 && cos_upslope < -0.5) {
                factor = k_shelter;
            }
            // Valley: concave terrain (negative curvature)
            else if (curv_val < -0.01) {
                factor = k_valley + (1.0_rt - k_valley) * (1.0_rt - amrex::min(slope_mag, 1.0_rt));
            }
            // Flat/other: no correction
            else {
                factor = 1.0;
            }

            // Clamp factor to reasonable range
            factor = amrex::max(0.1_rt, amrex::min(factor, 3.0_rt));

            // Apply speed scaling: scale wind magnitude
            ux *= factor;
            uy *= factor;
            Real wind_mag_new = wind_mag * factor;

            // Apply valley wind deflection toward slope aspect
            // Only when in valley (curv < -0.01) and on sloped terrain (slope_mag > 0.05)
            if (curv_val < -0.01 && slope_mag > 0.05) {
                // Compute z-component of slope × wind cross product. ux, uy
                // already carry the speed factor, so the sine of the wind-slope
                // angle is the cross product over |slope| |U_new|: one factor,
                // not two (the earlier denominator had it twice, which divided
                // the sine by the factor and over-deflected valley winds).
                Real sin_cross = sx * uy - sy * ux;

                // Compute deflection angle
                Real denom = amrex::max(static_cast<amrex::Real>(slope_mag * wind_mag_new), static_cast<amrex::Real>(1.0e-10));
                Real sin_cross_norm = amrex::max(-1.0_rt, amrex::min(1.0_rt, sin_cross / denom));
                Real deflect_angle = k_deflect * std::asin(sin_cross_norm);

                // Apply 2D rotation matrix to deflect wind direction
                Real cos_angle = std::cos(deflect_angle);
                Real sin_angle = std::sin(deflect_angle);
                Real ux_new = cos_angle * ux - sin_angle * uy;
                Real uy_new = sin_angle * ux + cos_angle * uy;
                ux = ux_new;
                uy = uy_new;
            }

            // Write back
            wind(i, j, k, 0) = ux;
            wind(i, j, k, 1) = uy;
        });
    }
}
