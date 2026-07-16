/**
 * @file ERF_LNGWindExtract.cpp
 * @brief Implementation of wind and surface field extraction from ERF 3D solver to LNG 2D grid.
 *
 * Extracts atmospheric wind at reference height and surface fields
 * from the 3D atmospheric solver onto the LNG 2D grid each timestep.
 * The wind interpolation algorithm copies fill_fire_wind_from_interpolation
 * from Source/Fire/ERF_FireWindExtract.cpp, with LNGGrid substituted for FireGrid.
 * 
 * References:
 *   ERF_DustWindExtract.cpp (Phase 9 analog)
 *   ERF_FireWindExtract.cpp (original algorithm)
 */

#include <AMReX_MFIter.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H>
#include <ERF_LNGGrid.H>
#include <iomanip>

#ifdef ERF_USE_LNG

using namespace amrex;

void fill_lng_wind_from_interpolation(
    MultiFab&       lng_wind_ref,
    const MultiFab& xvel_mf,
    const MultiFab& yvel_mf,
    const MultiFab& z_phys_cc_mf,
    const LNGGrid&  lg,
    Real            zref,
    int             nz,
    bool            lng_debug)
{
    // Direct vertical interpolation from atmospheric grid to LNG grid
    int C = lg.grid_ratio;

    for (MFIter mfi(lng_wind_ref, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> lng_wind = lng_wind_ref.array(mfi);
        Array4<const Real> xvel = xvel_mf.array(mfi);
        Array4<const Real> yvel = yvel_mf.array(mfi);
        Array4<const Real> z_phys_cc = z_phys_cc_mf.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            int i_l = iv[0];  // LNG grid index
            int j_l = iv[1];

            // Map to atmospheric column
            int i_a = i_l / C;
            int j_a = j_l / C;

            // Get surface height
            Real z_surf = z_phys_cc(i_a, j_a, 0);

            // Compute target height
            Real z_target = z_surf + zref;

            // Find vertical level bracket.
            // Initialize k_lo to the top interval (nz-2) so that if z_target
            // is above all levels, the topmost wind values are used.
            int k_lo = nz - 2;
            for (int k = 0; k < nz - 1; ++k) {
                if (z_phys_cc(i_a, j_a, k) <= z_target && 
                    z_target < z_phys_cc(i_a, j_a, k + 1)) {
                    k_lo = k;
                    break;
                }
            }

            // Compute interpolation weight
            Real z_lo = z_phys_cc(i_a, j_a, k_lo);
            Real z_hi = z_phys_cc(i_a, j_a, k_lo + 1);
            Real alpha = 0.0;
            if (z_hi > z_lo) {
                alpha = (z_target - z_lo) / (z_hi - z_lo);
                alpha = amrex::max(0.0, amrex::min(1.0, alpha));
            }

            int k_hi = k_lo + 1;

            // Average u/v from faces to cell centers
            Real u_cc_lo = 0.5 * (xvel(i_a, j_a, k_lo) + xvel(i_a + 1, j_a, k_lo));
            Real v_cc_lo = 0.5 * (yvel(i_a, j_a, k_lo) + yvel(i_a, j_a + 1, k_lo));

            Real u_cc_hi = 0.5 * (xvel(i_a, j_a, k_hi) + xvel(i_a + 1, j_a, k_hi));
            Real v_cc_hi = 0.5 * (yvel(i_a, j_a, k_hi) + yvel(i_a, j_a + 1, k_hi));

            // Interpolate to target height
            lng_wind(i_l, j_l, 0, 0) = u_cc_lo + alpha * (u_cc_hi - u_cc_lo);
            lng_wind(i_l, j_l, 0, 1) = v_cc_lo + alpha * (v_cc_hi - v_cc_lo);
        });
    }

    // Debug output if enabled
    if (lng_debug) {
        Real u_max = lng_wind_ref.max(0);
        Real v_max = lng_wind_ref.max(1);
        amrex::Print() << "[LNG DEBUG] Phase 4: wind extracted  u_max=" << std::scientific 
                       << std::setprecision(3) << u_max << " v_max=" << v_max 
                       << " m/s at zref=" << zref << " m\n";
    }
}

void fill_lng_ustar_from_surface_layer(
    MultiFab&       lng_ustar,
    const MultiFab& ustar_atm,
    const LNGGrid&  lg,
    bool            lng_debug)
{
    const int C = lg.grid_ratio;
    for (MFIter mfi(lng_ustar, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto lu = lng_ustar.array(mfi);
        auto ua = ustar_atm.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            lu(i,j,k) = ua(i/C, j/C, 0);
        });
    }

    // Debug output if enabled
    if (lng_debug) {
        Real ustar_max = lng_ustar.max(0);
        Real ustar_min = lng_ustar.min(0);
        amrex::Print() << "[LNG DEBUG] Phase 4: u* extracted  ustar_max=" << std::scientific 
                       << std::setprecision(3) << ustar_max << " ustar_min=" << ustar_min 
                       << " m/s\n";
    }
}

void fill_lng_scalar_from_atm(
    MultiFab&       lng_field,
    const MultiFab& atm_field,
    const LNGGrid&  lg,
    bool            lng_debug,
    const std::string& field_name)
{
    const int C = lg.grid_ratio;
    for (MFIter mfi(lng_field, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto lf = lng_field.array(mfi);
        auto af = atm_field.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            lf(i,j,k) = af(i/C, j/C, 0);
        });
    }

    // Debug output if enabled
    if (lng_debug) {
        Real field_max = lng_field.max(0);
        Real field_min = lng_field.min(0);
        if (field_name == "T_sfc") {
            amrex::Print() << "[LNG DEBUG] Phase 4: T_sfc extracted  T_max=" << std::scientific 
                           << std::setprecision(3) << field_max << " T_min=" << field_min 
                           << " K\n";
        } else if (field_name == "PBLH") {
            amrex::Print() << "[LNG DEBUG] Phase 4: PBLH extracted  PBLH_max=" << std::scientific 
                           << std::setprecision(3) << field_max << " PBLH_min=" << field_min 
                           << " m\n";
        }
    }
}

#endif // ERF_USE_LNG
