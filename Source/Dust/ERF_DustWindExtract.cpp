/**
 * @file ERF_DustWindExtract.cpp
 * @brief Implementation of wind and surface field extraction from ERF 3D solver to 2D dust grid.
 *
 * Extracts atmospheric wind at reference height and surface fields
 * from the 3D atmospheric solver onto the 2D dust grid each timestep.
 * The wind interpolation follows column_wind_at_height in
 * Source/Fire/ERF_FireWindExtract.cpp (bisection bracket, clamped to the lowest
 * and highest cell centres), with DustGrid substituted for FireGrid.
 */

#include <ERF_DustWindExtract.H>

#ifdef ERF_USE_DUST

#include <AMReX_MFIter.H>

using namespace amrex;

void fill_dust_wind_from_interpolation(
    MultiFab&       dust_wind_ref,
    const MultiFab& xvel_mf,
    const MultiFab& yvel_mf,
    const MultiFab& z_phys_cc_mf,
    const DustGrid& dg,
    Real            zref,
    int             nz)
{
    // Direct vertical interpolation from atmospheric grid to dust grid
    int C = dg.grid_ratio;

    for (MFIter mfi(dust_wind_ref, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> dust_wind = dust_wind_ref.array(mfi);
        Array4<const Real> xvel = xvel_mf.array(mfi);
        Array4<const Real> yvel = yvel_mf.array(mfi);
        Array4<const Real> z_phys_cc = z_phys_cc_mf.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            int i_d = iv[0];  // dust grid index
            int j_d = iv[1];

            // Map to atmospheric column
            int i_a = i_d / C;
            int j_a = j_d / C;

            // Surface height: half a cell below the first cell centre (the fire grid
            // uses the terrain surface; using the first centre put the sample half
            // a cell too high and made the wind depend on the vertical resolution).
            Real z_surf = z_phys_cc(i_a, j_a, 0) - 0.5 * (z_phys_cc(i_a, j_a, 1) - z_phys_cc(i_a, j_a, 0));

            // Compute target height
            Real z_target = z_surf + zref;

            // Bracket the target height by bisection on the column's cell-centre
            // heights, clamped to the lowest and highest cell centres. The linear
            // scan this replaces (copied from the fire before its own fix) left
            // k_lo at nz-2 whenever z_target was below the first cell centre, so
            // with zref < dz/2 (or any terrain or stretching that raises the
            // first centre above zref) every dust cell got the wind of the
            // second-highest cell in the domain.
            int k_lo;
            if (z_target <= z_phys_cc(i_a, j_a, 0)) {
                k_lo = 0;
            } else if (z_target >= z_phys_cc(i_a, j_a, nz - 1)) {
                k_lo = nz - 2;
            } else {
                int lo = 0;
                int hi = nz - 1;
                while (hi - lo > 1) {
                    const int mid = (lo + hi) / 2;
                    if (z_phys_cc(i_a, j_a, mid) <= z_target) { lo = mid; } else { hi = mid; }
                }
                k_lo = lo;
            }
            k_lo = amrex::max(0, amrex::min(k_lo, nz - 2));

            // Compute interpolation weight
            Real z_lo = z_phys_cc(i_a, j_a, k_lo);
            Real z_hi = z_phys_cc(i_a, j_a, k_lo + 1);
            Real alpha = 0.0;
            if (z_hi > z_lo) {
                alpha = (z_target - z_lo) / (z_hi - z_lo);
                alpha = amrex::max(Real(0.0), amrex::min(Real(1.0), alpha));
            }

            int k_hi = k_lo + 1;

            // Average u/v from faces to cell centers
            Real u_cc_lo = 0.5 * (xvel(i_a, j_a, k_lo) + xvel(i_a + 1, j_a, k_lo));
            Real v_cc_lo = 0.5 * (yvel(i_a, j_a, k_lo) + yvel(i_a, j_a + 1, k_lo));

            Real u_cc_hi = 0.5 * (xvel(i_a, j_a, k_hi) + xvel(i_a + 1, j_a, k_hi));
            Real v_cc_hi = 0.5 * (yvel(i_a, j_a, k_hi) + yvel(i_a, j_a + 1, k_hi));

            // Interpolate to target height
            dust_wind(i_d, j_d, 0, 0) = u_cc_lo + alpha * (u_cc_hi - u_cc_lo);
            dust_wind(i_d, j_d, 0, 1) = v_cc_lo + alpha * (v_cc_hi - v_cc_lo);
        });
    }
}

void fill_dust_ustar_from_surface_layer(
    MultiFab&       dust_ustar_in,
    const MultiFab& ustar_atm,
    const DustGrid& dg)
{
    const int C = dg.grid_ratio;
    for (MFIter mfi(dust_ustar_in, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto du = dust_ustar_in.array(mfi);
        auto ua = ustar_atm.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            du(i,j,k) = ua(i/C, j/C, 0);
        });
    }
}

void fill_dust_scalar_from_atm(
    MultiFab&       dust_field,
    const MultiFab& atm_field,
    const DustGrid& dg)
{
    const int C = dg.grid_ratio;
    for (MFIter mfi(dust_field, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto df = dust_field.array(mfi);
        auto af = atm_field.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            df(i,j,k) = af(i/C, j/C, 0);
        });
    }
}

void scale_dust_ustar_by_wind_ratio(
    MultiFab&       dust_ustar_in,
    const MultiFab& wind_corrected,
    const MultiFab& wind_raw)
{
    // u* is linear in the wind speed in a neutral log law, so the terrain
    // correction factor of the wind is the factor of u*. Re-deriving u* from the
    // corrected wind with a log law on z0_dust (the option before October 2026,
    // erf.dust.terrain_ustar = loglaw) replaced the surface layer's stability-
    // corrected u* on erf.most.z0 by a neutral one on erf.dust.z0_dust, 0.70x
    // on flat ground where the factor is 1.
    for (MFIter mfi(dust_ustar_in, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto ust = dust_ustar_in.array(mfi);
        auto wc  = wind_corrected.const_array(mfi);
        auto wr  = wind_raw.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real sc = std::sqrt(wc(i,j,k,0)*wc(i,j,k,0) + wc(i,j,k,1)*wc(i,j,k,1));
            const Real sr = std::sqrt(wr(i,j,k,0)*wr(i,j,k,0) + wr(i,j,k,1)*wr(i,j,k,1));
            // calm raw wind: no factor to apply (floored inside the select, the
            // unselected x/0 is speculated under the fpe traps)
            const Real ratio = (sr > Real(0.0)) ? sc / sr : Real(1.0);   // the branch guards the division
            ust(i,j,k) *= ratio;
        });
    }
}

void compute_dust_ustar_from_wind(
    MultiFab&       dust_ustar_in,
    const MultiFab& dust_wind_ref,
    Real            z_ref,
    Real            z0)
{
    // Compute friction velocity from wind speed using simple log-profile:
    // u* = κ * U / ln(z_ref/z0)
    // where κ = 0.4 (von Karman constant)
    // Reference: Businger et al. (1971), and MOST in ERF_MOSTStress.H

    constexpr Real kappa = 0.4;  // von Karman constant
    Real ln_ratio = std::log(z_ref / z0);
    if (ln_ratio <= 0.0) ln_ratio = 1.0;  // Safety guard

    for (MFIter mfi(dust_ustar_in, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto ust = dust_ustar_in.array(mfi);
        auto wind = dust_wind_ref.const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // Extract (u, v) components
            Real uu = wind(i, j, k, 0);
            Real vv = wind(i, j, k, 1);

            // Compute wind speed magnitude
            Real wind_mag = std::sqrt(uu*uu + vv*vv);

            // Compute u* = κ * U / ln(z_ref/z0)
            ust(i, j, k) = kappa * wind_mag / ln_ratio;
        });
    }
}

#endif // ERF_USE_DUST
