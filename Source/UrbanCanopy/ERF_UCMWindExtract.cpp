/**
 * @file ERF_UCMWindExtract.cpp
 * @brief Implementation of wind and scalar extraction functions
 *
 * Extracts atmospheric forcings (u*, wind, T, q) from ERF lowest level
 * to UCM 2D grid. Handles terrain-aware height interpolation and radiation
 * placeholder fill.
 *
 * References:
 *  - WRF phys/module_sf_urban.F (Kusaka 2001)
 *  - Chen et al. (2011)
 */

#include <UrbanCanopy/ERF_UCMWindExtract.H>
#include <UrbanCanopy/ERF_UCMGrid.H>
#include <UrbanCanopy/ERF_UCMParams.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>

// ============================================================================
// UCMForcing methods
// ============================================================================

bool UCMForcing::all_allocated() const
{
    return (u_star != nullptr && wind_ref != nullptr && T_atm_ref != nullptr &&
            SW_down != nullptr && LW_down != nullptr);
    // q_atm_ref is optional (null-safe)
}

void UCMForcing::clear()
{
    u_star.reset();
    wind_ref.reset();
    T_atm_ref.reset();
    q_atm_ref.reset();
    SW_down.reset();
    LW_down.reset();
}

// ============================================================================
// Extraction function implementations
// ============================================================================

void fill_ucm_ustar_from_surface_layer(amrex::MultiFab& ucm_ustar,
                                       const amrex::MultiFab& atm_u_star,
                                       const UCMGrid& /*ucm_grid*/,
                                       int /*lev*/)
{
    // Simple copy from ATM to UCM (grid_ratio=1 in Phase 1.3)
    amrex::MultiFab::Copy(ucm_ustar, atm_u_star, 0, 0, ucm_ustar.nComp(), ucm_ustar.nGrowVect());
}

// ============================================================================

void fill_ucm_wind_from_interpolation(amrex::MultiFab& ucm_wind_ref,
                                      const amrex::MultiFab& xvel,
                                      const amrex::MultiFab& yvel,
                                      const amrex::MultiFab& z_phys_cc,
                                      const amrex::MultiFab& H_bldg,
                                      amrex::Real zref,
                                      const UCMGrid& /*ucm_grid*/,
                                      int /*nz_atm*/,
                                      int /*lev*/)
{
    // Extract wind at lowest ATM level and interpolate to z_target via log-law
    const int klo = 0;              // Lowest ATM level
    const amrex::Real z0 = 0.1;     // Roughness length [m]

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ucm_wind_ref, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto wind_arr = ucm_wind_ref.array(mfi);
        auto const u_arr = xvel.const_array(mfi);
        auto const v_arr = yvel.const_array(mfi);
        auto const z_arr = z_phys_cc.const_array(mfi);
        auto const h_arr = H_bldg.const_array(mfi);

        const amrex::Box& bx = mfi.tilebox();

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) noexcept
            {
                // Target height
                amrex::Real z_sfc    = z_arr(i, j, klo);
                amrex::Real H_b      = h_arr(i, j, 0);
                amrex::Real z_target = z_sfc + H_b + zref;

                // Wind at ATM lowest level
                amrex::Real u_atm    = u_arr(i, j, klo);
                amrex::Real v_atm    = v_arr(i, j, klo);
                amrex::Real wspd_atm = std::sqrt(u_atm*u_atm + v_atm*v_atm);

                // Log-law interpolation
                amrex::Real ratio = 1.0;
                if (wspd_atm > 1.0e-6) {
                    amrex::Real ln_target = std::log((z_target + z0) / z0);
                    amrex::Real ln_atm    = std::log((z_sfc    + z0) / z0);
                    if (ln_atm > 1.0e-6) {
                        ratio = ln_target / ln_atm;
                    }
                }

                wind_arr(i, j, 0, 0) = u_atm * ratio;
                wind_arr(i, j, 0, 1) = v_atm * ratio;
            });
    }
}

// ============================================================================

void fill_ucm_scalar_from_atm(amrex::MultiFab& ucm_scalar,
                              const amrex::MultiFab& atm_scalar,
                              const UCMGrid& /*ucm_grid*/,
                              const amrex::Geometry& /*geom_atm*/,
                              int comp,
                              int /*lev*/)
{
    // Extract scalar from ATM lowest level to UCM grid
    const int klo = 0;

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ucm_scalar, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto scalar_arr = ucm_scalar.array(mfi);
        auto const atm_arr = atm_scalar.const_array(mfi);

        const amrex::Box& bx = mfi.tilebox();

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int /*k*/) noexcept
            {
                scalar_arr(i, j, 0) = atm_arr(i, j, klo, comp);
            });
    }
}
