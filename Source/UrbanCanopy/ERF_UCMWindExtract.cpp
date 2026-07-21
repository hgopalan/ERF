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
                                       const UCMGrid& ucm_grid,
                                       int lev)
{
    // Bilinear interpolation / box averaging from ATM to UCM cells
    // For Phase 1.1/1.2 (grid_ratio = 1), this is a direct copy with
    // potential for ghost cell handling.

    int grid_ratio = ucm_grid.grid_ratio;
    bool ucm_debug = false;  // Default; will be read from params in caller context

    // Copy loop: iterate UCM cells and average/interpolate ATM values
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ucm_ustar, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto& ucm_ustar_box = ucm_ustar[mfi];
        auto& atm_u_star_box = atm_u_star[mfi];

        amrex::Box const& bx = mfi.tilebox();
        int ncomp = ucm_ustar_box.nComp();

        amrex::ParallelFor(bx, ncomp,
            [=] AMREX_GPU_DEVICE(amrex::Box const& tbx, int n) {
                auto ucm_arr = ucm_ustar_box.array();
                auto atm_arr = atm_u_star_box.array();

                amrex::ParallelForRNG(tbx,
                    [=] AMREX_GPU_DEVICE(amrex::Box const& b, int /*dummy*/) {
                        amrex::LoopOnCpu(b, [=] (amrex::IntVect const& iv) {
                            int i = iv[0];
                            int j = iv[1];
                            int k = iv[2];

                            // Simple averaging from ATM to UCM
                            // Phase 1.1: grid_ratio=1, so direct copy
                            if (grid_ratio == 1) {
                                ucm_arr(i, j, k, n) = atm_arr(i, j, k, n);
                            } else {
                                // Phase 3.1+: average over grid_ratio x grid_ratio ATM cells
                                amrex::Real sum = 0.0;
                                int cnt = 0;
                                for (int ii = 0; ii < grid_ratio; ++ii) {
                                    for (int jj = 0; jj < grid_ratio; ++jj) {
                                        int ia = i*grid_ratio + ii;
                                        int ja = j*grid_ratio + jj;
                                        sum += atm_arr(ia, ja, k, n);
                                        cnt++;
                                    }
                                }
                                ucm_arr(i, j, k, n) = sum / cnt;
                            }
                        });
                    }, n);
            });
    }

    // Debug trace
    if (ucm_debug) {
        amrex::Real ustar_min = ucm_ustar.min(0);
        amrex::Real ustar_max = ucm_ustar.max(0);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.3][fill_ucm_ustar_from_surface_layer] "
                          << "u_star min=" << ustar_min << " max=" << ustar_max << "\n";
        }
    }
}

// ============================================================================

void fill_ucm_wind_from_interpolation(amrex::MultiFab& ucm_wind_ref,
                                      const amrex::MultiFab& xvel,
                                      const amrex::MultiFab& yvel,
                                      const amrex::MultiFab& z_phys_cc,
                                      const amrex::MultiFab& H_bldg,
                                      amrex::Real zref,
                                      const UCMGrid& ucm_grid,
                                      int nz_atm,
                                      int lev)
{
    // Extract wind at lowest ATM level and interpolate to z_target via log-law

    int grid_ratio = ucm_grid.grid_ratio;
    bool ucm_debug = false;  // Default; read from context in real implementation
    int klo = 0;  // Lowest ATM level (Phase 1: non-terrain assumed)

    amrex::Real z0 = 0.1;  // Roughness length [m] (hardcoded Phase 1.3; CSV Phase 2.1)

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ucm_wind_ref, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto& wind_box = ucm_wind_ref[mfi];
        auto& xvel_box = xvel[mfi];
        auto& yvel_box = yvel[mfi];
        auto& z_phys_box = z_phys_cc[mfi];
        auto& H_bldg_box = H_bldg[mfi];

        amrex::Box const& bx = mfi.tilebox();

        // Wind has 2 components (u, v)
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(amrex::Box const& tbx) {
                auto wind_arr = wind_box.array();
                auto u_arr = xvel_box.array();
                auto v_arr = yvel_box.array();
                auto z_arr = z_phys_box.array();
                auto h_arr = H_bldg_box.array();

                amrex::ParallelForRNG(tbx,
                    [=] AMREX_GPU_DEVICE(amrex::Box const& b, int /*dummy*/) {
                        amrex::LoopOnCpu(b, [=] (amrex::IntVect const& iv) {
                            int i = iv[0];
                            int j = iv[1];
                            int k_ucm = iv[2];  // UCM level (always 0 for 2D)

                            // Map UCM cell (i, j) to ATM cell (ia, ja)
                            int ia = i * grid_ratio;
                            int ja = j * grid_ratio;

                            // Target height (terrain-aware)
                            amrex::Real z_sfc = z_arr(ia, ja, klo);  // Surface height
                            amrex::Real H_b = h_arr(i, j, k_ucm);     // Building height
                            amrex::Real z_target = z_sfc + H_b + zref;

                            // ATM lowest level height
                            amrex::Real z_atm = z_arr(ia, ja, klo);

                            // Wind at ATM lowest level
                            amrex::Real u_atm = u_arr(ia, ja, klo);
                            amrex::Real v_atm = v_arr(ia, ja, klo);
                            amrex::Real wspd_atm = std::sqrt(u_atm*u_atm + v_atm*v_atm);

                            // Interpolate via log-law (neutral assumption; Phase 1.4: add stability)
                            // U(z) = (u*/κ) * ln((z + z0)/z0)
                            // At z_atm: U_atm = (u*/κ) * ln((z_atm + z0)/z0)
                            // At z_target: U_target = (u*/κ) * ln((z_target + z0)/z0)
                            // Ratio: U_target/U_atm = ln((z_target+z0)/z0) / ln((z_atm+z0)/z0)

                            amrex::Real const kappa = 0.41;  // von Karman constant
                            amrex::Real ratio = 1.0;  // Default (if z_atm near z_target)

                            if (wspd_atm > 1.0e-6) {  // Avoid division by near-zero
                                amrex::Real ln_target = std::log((z_target + z0) / z0);
                                amrex::Real ln_atm = std::log((z_atm + z0) / z0);
                                if (ln_atm > 1.0e-6) {
                                    ratio = ln_target / ln_atm;
                                }
                            }

                            wind_arr(i, j, k_ucm, 0) = u_atm * ratio;  // u-component
                            wind_arr(i, j, k_ucm, 1) = v_atm * ratio;  // v-component
                        });
                    });
            });
    }

    // Debug trace
    if (ucm_debug) {
        amrex::Real wind_max = ucm_wind_ref.max(0);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.3][fill_ucm_wind_from_interpolation] "
                          << "z_target at (0,0)=" << (z_phys_cc[0][0][0] + H_bldg[0][0][0] + zref)
                          << " m; wind max=" << wind_max << " m/s\n";
        }
    }
}

// ============================================================================

void fill_ucm_scalar_from_atm(amrex::MultiFab& ucm_scalar,
                              const amrex::MultiFab& atm_scalar,
                              const UCMGrid& ucm_grid,
                              const amrex::Geometry& /*geom_atm*/,
                              int comp,
                              int lev)
{
    // Extract scalar from ATM lowest level to UCM grid

    int grid_ratio = ucm_grid.grid_ratio;
    bool ucm_debug = false;  // Default
    int klo = 0;  // Lowest ATM level

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(ucm_scalar, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto& scalar_box = ucm_scalar[mfi];
        auto& atm_box = atm_scalar[mfi];

        amrex::Box const& bx = mfi.tilebox();

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(amrex::Box const& tbx) {
                auto scalar_arr = scalar_box.array();
                auto atm_arr = atm_box.array();

                amrex::ParallelForRNG(tbx,
                    [=] AMREX_GPU_DEVICE(amrex::Box const& b, int /*dummy*/) {
                        amrex::LoopOnCpu(b, [=] (amrex::IntVect const& iv) {
                            int i = iv[0];
                            int j = iv[1];
                            int k = iv[2];

                            int ia = i * grid_ratio;
                            int ja = j * grid_ratio;

                            // Simple averaging from ATM to UCM (grid_ratio=1: copy)
                            if (grid_ratio == 1) {
                                scalar_arr(i, j, k) = atm_arr(ia, ja, klo, comp);
                            } else {
                                amrex::Real sum = 0.0;
                                int cnt = 0;
                                for (int ii = 0; ii < grid_ratio; ++ii) {
                                    for (int jj = 0; jj < grid_ratio; ++jj) {
                                        int iia = ia + ii;
                                        int jja = ja + jj;
                                        sum += atm_arr(iia, jja, klo, comp);
                                        cnt++;
                                    }
                                }
                                scalar_arr(i, j, k) = sum / cnt;
                            }
                        });
                    });
            });
    }

    // Debug trace
    if (ucm_debug) {
        amrex::Real scalar_min = ucm_scalar.min(0);
        amrex::Real scalar_max = ucm_scalar.max(0);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.3][fill_ucm_scalar_from_atm] "
                          << "scalar (comp=" << comp << ") min=" << scalar_min
                          << " max=" << scalar_max << "\n";
        }
    }
}
