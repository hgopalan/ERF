/**
 * @file ERF_UCMLayer.cpp
 * @brief Implementation of UCMLayer physics driver for facet SEB and conduction
 *
 * Implements the per-timestep SLUCM integration:
 * 1. Extract forcing (u*, wind, T, q, SW/LW)
 * 2. Solve facet SEB via Newton iteration
 * 3. Advance slab conduction
 * 4. Compute and store fluxes
 *
 * References:
 *  - Kusaka et al. (2001)
 *  - Chen et al. (2011)
 *  - WRF phys/module_sf_urban.F
 */

#include <UrbanCanopy/ERF_UCMLayer.H>
#include <UrbanCanopy/ERF_UCMSlabConduction.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelFor.H>
#include <cmath>

// ============================================================================
// Constructors
// ============================================================================

UCMLayer::UCMLayer(const UCMParams& params, int lev)
    : m_params(params), m_lev(lev), m_warn_radiation_placeholder_printed(false)
{
    // Phase 1.1: enforce anchor_level == 0
    if (lev != params.anchor_level) {
        std::string msg = std::string("[UCM] UCMLayer constructed at level ")
                        + std::to_string(lev) + " but params.anchor_level = "
                        + std::to_string(params.anchor_level)
                        + ". Phase 1.3 supports only anchor_level=0.";
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(false, msg.c_str());
    }
}

// ============================================================================
// Main advance function
// ============================================================================

void UCMLayer::advance(UCMFields& fields,
                       UCMForcing& forcing,
                       const UCMGrid& ucm_grid,
                       const amrex::MultiFab& atm_u_star,
                       const amrex::MultiFab& atm_t_star,
                       const amrex::MultiFab& atm_q_star,
                       const amrex::MultiFab& xvel,
                       const amrex::MultiFab& yvel,
                       const amrex::MultiFab& z_phys_cc,
                       const amrex::MultiFab& T_atm_lowest,
                       const amrex::MultiFab& q_atm_lowest,
                       const amrex::Geometry& geom_atm,
                       amrex::Real time,
                       amrex::Real dt,
                       int nz_atm,
                       int lev)
{
    // ========================================================================
    // Step 1: Allocate and extract forcing (u*, wind, T, q, SW/LW)
    // ========================================================================

    // Allocate forcing container if not already done
    if (!forcing.all_allocated()) {
        forcing.u_star = std::make_unique<amrex::MultiFab>(
            ucm_grid.ba, ucm_grid.dm, 1, amrex::IntVect(1, 1, 0));
        forcing.wind_ref = std::make_unique<amrex::MultiFab>(
            ucm_grid.ba, ucm_grid.dm, 2, amrex::IntVect(1, 1, 0));
        forcing.T_atm_ref = std::make_unique<amrex::MultiFab>(
            ucm_grid.ba, ucm_grid.dm, 1, amrex::IntVect(1, 1, 0));
        forcing.q_atm_ref = std::make_unique<amrex::MultiFab>(
            ucm_grid.ba, ucm_grid.dm, 1, amrex::IntVect(1, 1, 0));
        forcing.SW_down = std::make_unique<amrex::MultiFab>(
            ucm_grid.ba, ucm_grid.dm, 1, amrex::IntVect(1, 1, 0));
        forcing.LW_down = std::make_unique<amrex::MultiFab>(
            ucm_grid.ba, ucm_grid.dm, 1, amrex::IntVect(1, 1, 0));
    }

    // Extract u* from SurfaceLayer
    fill_ucm_ustar_from_surface_layer(*forcing.u_star, atm_u_star, ucm_grid, lev);

    // Extract wind via log-law interpolation to z_target
    fill_ucm_wind_from_interpolation(*forcing.wind_ref, xvel, yvel, z_phys_cc,
                                     *fields.H_bldg, m_params.zref, ucm_grid, nz_atm, lev);

    // Extract temperature from ATM lowest level
    fill_ucm_scalar_from_atm(*forcing.T_atm_ref, T_atm_lowest, ucm_grid, geom_atm, 0, lev);

    // Extract water vapor (if available; null-safe)
    if (q_atm_lowest.boxArray().size() > 0) {
        fill_ucm_scalar_from_atm(*forcing.q_atm_ref, q_atm_lowest, ucm_grid, geom_atm, 0, lev);
    }

    // ========================================================================
    // Step 2: Fill radiation (analytical placeholder, Phase 1.3)
    // ========================================================================
    // TODO(UCM Phase 4.2): replace with radiation scheme extraction

    forcing.SW_down->setVal(0.0);
    forcing.LW_down->setVal(350.0);

    // Analytical diurnal cycle for SW
    // phase = 2π * elapsed_time / 86400 - π/2
    // SW = 800 * max(0, cos(phase)) [W/m²]

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*forcing.SW_down, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto& sw_box = (*forcing.SW_down)[mfi];
        amrex::Box const& bx = mfi.tilebox();

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(amrex::Box const& tbx) {
                auto sw_arr = sw_box.array();
                amrex::Real phase = 2.0*M_PI*time/86400.0 - 0.5*M_PI;
                amrex::Real sw_val = 800.0 * std::max(0.0, std::cos(phase));

                amrex::ParallelForRNG(tbx,
                    [=] AMREX_GPU_DEVICE(amrex::Box const& b, int /*dummy*/) {
                        amrex::LoopOnCpu(b, [=] (amrex::IntVect const& iv) {
                            sw_arr(iv, 0) = sw_val;
                        });
                    });
            });
    }

    // One-time warning about radiation placeholder
    if (!m_warn_radiation_placeholder_printed) {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.3][WARNING] Radiation (SW/LW) filled analytically. "
                          << "Phase 4.2 will replace with radiation solver extraction.\n";
        }
        m_warn_radiation_placeholder_printed = true;
    }

    // ========================================================================
    // Step 3: Solve facet SEB and advance slab conduction
    // ========================================================================

    // For each cell (i, j):
    //   Check is_urban mask
    //   Solve roof/wall/road skin T via Newton on SEB
    //   Advance slab conduction for each facet
    //   Compute sensible/latent fluxes
    //   Update canyon-air temperature

    amrex::Real rho_cp_const = 1200.0;  // ρ*cp for air [J/m³/K] (simplified, hardcoded)
    amrex::Real Ch_const = 0.004;       // Drag coefficient (simplified, hardcoded)

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*fields.T_skin_roof, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        // Box domain (only UCM cells exist; k=0 for 2D)
        amrex::Box const& bx = mfi.tilebox();

        // Get MultiFab arrays for all fields
        auto T_roof = (*fields.T_skin_roof)[mfi].array();
        auto T_wall = (*fields.T_skin_wall)[mfi].array();
        auto T_road = (*fields.T_skin_road)[mfi].array();
        auto T_canyon = (*fields.T_canyon_air)[mfi].array();
        auto H_sens = (*fields.H_sensible)[mfi].array();
        auto LE_lat = (*fields.LE_latent)[mfi].array();
        auto is_u = (*fields.is_urban)[mfi].array();

        auto H_bldg = (*fields.H_bldg)[mfi].array();
        auto W_road = (*fields.W_road)[mfi].array();
        auto W_roof = (*fields.W_roof)[mfi].array();
        auto albedo_r = (*fields.albedo_roof)[mfi].array();
        auto albedo_w = (*fields.albedo_wall)[mfi].array();
        auto albedo_rd = (*fields.albedo_road)[mfi].array();
        auto eps_r = (*fields.emissivity_roof)[mfi].array();
        auto eps_w = (*fields.emissivity_wall)[mfi].array();
        auto eps_rd = (*fields.emissivity_road)[mfi].array();

        auto u_star_box = (*forcing.u_star)[mfi].array();
        auto wind_box = (*forcing.wind_ref)[mfi].array();
        auto T_atm_ref = (*forcing.T_atm_ref)[mfi].array();
        auto SW_box = (*forcing.SW_down)[mfi].array();
        auto LW_box = (*forcing.LW_down)[mfi].array();

        // Loop over cells
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(amrex::Box const& tbx) {
                amrex::LoopOnCpu(tbx, [=] (amrex::IntVect const& iv) {
                    int i = iv[0];
                    int j = iv[1];
                    int k = iv[2];  // Always 0 for 2D UCM

                    // Skip if not urban
                    if (is_u(i, j, k) == 0) return;

                    // Extract forcing for this cell
                    amrex::Real u_s = u_star_box(i, j, k);
                    amrex::Real u_wind = wind_box(i, j, k, 0);
                    amrex::Real v_wind = wind_box(i, j, k, 1);
                    amrex::Real wspd = std::sqrt(u_wind*u_wind + v_wind*v_wind);
                    amrex::Real T_atm = T_atm_ref(i, j, k);
                    amrex::Real SW = SW_box(i, j, k);
                    amrex::Real LW = LW_box(i, j, k);

                    // Physical constants
                    amrex::Real sigma = 5.67e-8;  // Stefan-Boltzmann [W/m²/K⁴]
                    amrex::Real dz_slab = m_params.slab_L / m_params.slab_N_layers;

                    // ============================================================
                    // Simplified SEB solve: Newton iteration on skin T
                    // ============================================================
                    // For each facet (roof, wall, road):
                    //   Rn - H - G = 0 (LE=0 in Phase 1.3)
                    //   Newton: iterate on T_skin until |residual| < tol

                    // Roof (Phase 1.3 simplified: uses T_atm as reference)
                    {
                        amrex::Real T_skin = T_roof(i, j, k);  // Initial guess
                        amrex::Real T_ref = T_atm;  // Roof sees atm directly
                        amrex::Real H_b = H_bldg(i, j, k);

                        for (int iter = 0; iter < m_params.newton_max_iter; ++iter) {
                            // Radiative net flux
                            amrex::Real Rn = (1.0 - albedo_r(i, j, k))*SW 
                                           + eps_r(i, j, k)*(LW - sigma*T_skin*T_skin*T_skin*T_skin);

                            // Sensible heat flux (assuming wspd > 0)
                            amrex::Real H = rho_cp_const * Ch_const * std::max(wspd, 0.1) 
                                          * (T_skin - T_ref);

                            // Ground heat flux (simplified: top layer BC)
                            amrex::Real G = m_params.k_therm_uniform * (T_skin - T_roof(i, j, k))
                                          / (0.5 * dz_slab);

                            // Residual
                            amrex::Real residual = Rn - H - G;

                            // Check convergence
                            if (std::abs(residual) < m_params.newton_tol_K * 10.0) break;

                            // Newton update (simplified Jacobian, dR/dT ~ -4*ε*σ*T³ - H_coeff - k/dz)
                            amrex::Real dRdT = -4.0*eps_r(i, j, k)*sigma*T_skin*T_skin*T_skin
                                            - rho_cp_const*Ch_const*std::max(wspd, 0.1)
                                            - m_params.k_therm_uniform / (0.5 * dz_slab);
                            if (std::abs(dRdT) > 1.0e-6) {
                                T_skin -= residual / dRdT;
                            }

                            // Bounds check
                            T_skin = std::max(250.0, std::min(330.0, T_skin));
                        }

                        T_roof(i, j, k) = T_skin;
                    }

                    // Wall (similar to roof, but uses canyon air reference)
                    {
                        amrex::Real T_skin = T_wall(i, j, k);
                        amrex::Real T_ref = T_canyon(i, j, k);  // Canyon air

                        for (int iter = 0; iter < m_params.newton_max_iter; ++iter) {
                            amrex::Real Rn = (1.0 - albedo_w(i, j, k))*SW 
                                           + eps_w(i, j, k)*(LW - sigma*T_skin*T_skin*T_skin*T_skin);
                            amrex::Real H = rho_cp_const * Ch_const * std::max(wspd, 0.1) 
                                          * (T_skin - T_ref);
                            amrex::Real G = m_params.k_therm_uniform * (T_skin - T_wall(i, j, k))
                                          / (0.5 * dz_slab);
                            amrex::Real residual = Rn - H - G;

                            if (std::abs(residual) < m_params.newton_tol_K * 10.0) break;

                            amrex::Real dRdT = -4.0*eps_w(i, j, k)*sigma*T_skin*T_skin*T_skin
                                            - rho_cp_const*Ch_const*std::max(wspd, 0.1)
                                            - m_params.k_therm_uniform / (0.5 * dz_slab);
                            if (std::abs(dRdT) > 1.0e-6) {
                                T_skin -= residual / dRdT;
                            }

                            T_skin = std::max(250.0, std::min(330.0, T_skin));
                        }

                        T_wall(i, j, k) = T_skin;
                    }

                    // Road (similar)
                    {
                        amrex::Real T_skin = T_road(i, j, k);
                        amrex::Real T_ref = T_canyon(i, j, k);

                        for (int iter = 0; iter < m_params.newton_max_iter; ++iter) {
                            amrex::Real Rn = (1.0 - albedo_rd(i, j, k))*SW 
                                           + eps_rd(i, j, k)*(LW - sigma*T_skin*T_skin*T_skin*T_skin);
                            amrex::Real H = rho_cp_const * Ch_const * std::max(wspd, 0.1) 
                                          * (T_skin - T_ref);
                            amrex::Real G = m_params.k_therm_uniform * (T_skin - T_road(i, j, k))
                                          / (0.5 * dz_slab);
                            amrex::Real residual = Rn - H - G;

                            if (std::abs(residual) < m_params.newton_tol_K * 10.0) break;

                            amrex::Real dRdT = -4.0*eps_rd(i, j, k)*sigma*T_skin*T_skin*T_skin
                                            - rho_cp_const*Ch_const*std::max(wspd, 0.1)
                                            - m_params.k_therm_uniform / (0.5 * dz_slab);
                            if (std::abs(dRdT) > 1.0e-6) {
                                T_skin -= residual / dRdT;
                            }

                            T_skin = std::max(250.0, std::min(330.0, T_skin));
                        }

                        T_road(i, j, k) = T_skin;
                    }

                    // ============================================================
                    // Step 4: Advance slab conduction (implicit Euler TDMA)
                    // ============================================================
                    // For each facet, solve diffusion equation over dt
                    // Top BC: Q_top from SEB residual (approximately 0 after Newton solve)
                    // Bottom BC: T_deep = slab_T_deep

                    amrex::Real T_roof_layers[50];
                    amrex::Real T_wall_layers[50];
                    amrex::Real T_road_layers[50];

                    // Initialize with current skin T in first layer
                    // (Simplified: assume all layers equal to skin initially)
                    for (int il = 0; il < m_params.slab_N_layers; ++il) {
                        T_roof_layers[il] = T_roof(i, j, k);
                        T_wall_layers[il] = T_wall(i, j, k);
                        T_road_layers[il] = T_road(i, j, k);
                    }

                    // Top heat flux ~ 0 after Newton convergence (SEB balanced)
                    amrex::Real Q_top = 0.0;
                    amrex::Real T_deep = m_params.slab_T_deep;

                    advance_slab_conduction_column(T_roof_layers, Q_top, T_deep,
                        m_params.k_therm_uniform, m_params.rho_cp_uniform,
                        dz_slab, dt, m_params.slab_N_layers);
                    advance_slab_conduction_column(T_wall_layers, Q_top, T_deep,
                        m_params.k_therm_uniform, m_params.rho_cp_uniform,
                        dz_slab, dt, m_params.slab_N_layers);
                    advance_slab_conduction_column(T_road_layers, Q_top, T_deep,
                        m_params.k_therm_uniform, m_params.rho_cp_uniform,
                        dz_slab, dt, m_params.slab_N_layers);

                    // Update skin T (first layer)
                    T_roof(i, j, k) = T_roof_layers[0];
                    T_wall(i, j, k) = T_wall_layers[0];
                    T_road(i, j, k) = T_road_layers[0];

                    // ============================================================
                    // Step 5: Compute sensible and latent heat fluxes
                    // ============================================================

                    amrex::Real H_roof = rho_cp_const * Ch_const * std::max(wspd, 0.1)
                                       * (T_roof(i, j, k) - T_atm);
                    amrex::Real H_wall = rho_cp_const * Ch_const * std::max(wspd, 0.1)
                                       * (T_wall(i, j, k) - T_canyon(i, j, k));
                    amrex::Real H_road = rho_cp_const * Ch_const * std::max(wspd, 0.1)
                                       * (T_road(i, j, k) - T_canyon(i, j, k));

                    // Area-weighted average (simplified; Phase 1.3 homogeneous)
                    amrex::Real W_tot = W_roof(i, j, k) + 2.0*H_bldg(i, j, k) + W_road(i, j, k);
                    H_sens(i, j, k) = (H_roof*W_roof(i, j, k) + H_wall*2.0*H_bldg(i, j, k) 
                                    + H_road*W_road(i, j, k)) / W_tot;
                    LE_lat(i, j, k) = 0.0;  // Phase 1.3: LE=0

                    // ============================================================
                    // Step 6: Update canyon-air temperature (simplified, Kusaka eq. 21)
                    // ============================================================
                    // T_canyon = (H_road/(..) + 2H/W*H_wall/(..) + T_atm) / (...)

                    amrex::Real denom = 1.0 + 1.0/Ch_const + 2.0*H_bldg(i, j, k)/W_road(i, j, k) * 1.0/Ch_const;
                    if (denom > 1.0e-6) {
                        T_canyon(i, j, k) = (H_road/(rho_cp_const*Ch_const) 
                                           + 2.0*H_bldg(i, j, k)/W_road(i, j, k) * H_wall/(rho_cp_const*Ch_const)
                                           + T_atm) / denom;
                    } else {
                        T_canyon(i, j, k) = T_atm;
                    }

                    // Bounds check
                    T_canyon(i, j, k) = std::max(250.0, std::min(330.0, T_canyon(i, j, k)));
                });
            });
    }

    // ========================================================================
    // Step 7: Debug trace (Phase 1.3 mandatory)
    // ========================================================================

    if (m_params.ucm_debug) {
        amrex::Real T_roof_min = fields.T_skin_roof->min(0);
        amrex::Real T_roof_max = fields.T_skin_roof->max(0);
        amrex::Real T_wall_min = fields.T_skin_wall->min(0);
        amrex::Real T_wall_max = fields.T_skin_wall->max(0);
        amrex::Real T_road_min = fields.T_skin_road->min(0);
        amrex::Real T_road_max = fields.T_skin_road->max(0);
        amrex::Real T_canyon_min = fields.T_canyon_air->min(0);
        amrex::Real T_canyon_max = fields.T_canyon_air->max(0);

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "[UCM][1.3][UCMLayer::advance] "
                          << "dt=" << dt << "s, sim_time=" << time << "s; "
                          << "T_roof=[" << T_roof_min << "," << T_roof_max << "], "
                          << "T_wall=[" << T_wall_min << "," << T_wall_max << "], "
                          << "T_road=[" << T_road_min << "," << T_road_max << "], "
                          << "T_canyon=[" << T_canyon_min << "," << T_canyon_max << "] K\n";
        }
    }
}
