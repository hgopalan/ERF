#ifndef ERF_UCM_SEB_SOLVER_4VAR_H
#define ERF_UCM_SEB_SOLVER_4VAR_H

#include <AMReX_REAL.H>
#include <cmath>

/**
 * @file ERF_UCMSEBSolver4Var.H
 * @brief GPU-safe header-only 4-facet Surface Energy Balance Newton solver (Phase 6.2b)
 *
 * Phase 6.2b: Crown as separate 4th SEB facet (roof, wall, road, crown).
 * Solves coupled system using Gauss-Seidel iteration (no LU decomposition).
 * Contract #31: Crown-facet feedback uses semi-implicit lagged T_crown in the 3
 * existing facet rows. Within a Newton call, roof/wall/road read T_crown from
 * previous outer iteration (or T_canyon_init at n=0). Crown row uses current
 * iterates for T_roof, and lagged for T_wall_prev, T_road_prev.
 *
 * SIGN CONVENTION (consistent with WRF/module_sf_urban.F):
 *   Positive H = surface → atmosphere (surface LOSING heat, cooling)
 *   Negative H = atmosphere → surface (surface GAINING heat, warming)
 *
 * Crown physics (Contract #32):
 *   - SW absorption via Q_tree_SW_abs (reused from Phase 6.2a)
 *   - 2-sided LW emission: upward σ*T_crown^4, downward to canyon
 *   - Canyon-LW absorption: receives LW_down from atmosphere
 *   - 2-sided sensible flux: H_crown_up (to atmosphere), H_crown_down (to canyon)
 *   - NO conduction (no slab). NO LE (transpiration deferred to Phase 6.3).
 *
 * Semi-implicit coupling (Contract #31):
 *   Roof/wall/road LW_in gains: +eps_facet * σ * T_crown_prev^4 * crown_view_factor
 *   T_crown_prev is from previous Newton iteration (or T_canyon_init at n=0).
 *   This keeps the system weakly coupled, solvable with Gauss-Seidel.
 *
 * System architecture:
 *   Row 0 (Roof):   F_roof  = Rn_roof  - H_roof  - G_roof  = 0
 *   Row 1 (Wall):   F_wall  = Rn_wall  - H_wall  = 0  (no G; semi-implicit LW from crown)
 *   Row 2 (Road):   F_road  = Rn_road  - H_road  - G_road  = 0  (semi-implicit LW from crown)
 *   Row 3 (Crown):  F_crown = Q_SW_crown - 2*H_crown - sigma*eps*T_crown^4 + sigma*eps*LW_down = 0
 *
 * Gauss-Seidel iteration:
 *   1. Compute roof residual and Jacobian; Newton step for T_roof
 *   2. Compute wall residual using updated T_roof; Newton step for T_wall
 *   3. Compute road residual using updated T_roof; Newton step for T_road
 *   4. Compute crown residual using updated T_roof, T_wall, T_road; Newton step for T_crown
 *   5. Repeat until convergence
 */

/**
 * @brief Phase 6.2b: 4-facet SEB solver with crown as separate prognostic variable
 *
 * Phase 6.2b hotfix1: Fixed bugs B, C, D, E, F
 *   - Bug B: Convergence now tracks dT_max from all 4 rows (not frozen at 0.0)
 *   - Bug C: Wall row now includes conduction term
 *   - Bug D: Crown sensible split into upward (to atmosphere) and downward (to canyon)
 *   - Bug E: Crown row now absorbs canyon LW upward from roof, wall, road
 *   - Bug F: Proper Gauss-Seidel — LW feedback recomputed at start of each outer iter
 *
 * Inputs:
 *   T_roof_in, T_wall_in, T_road_in, T_crown_in : current temperature estimates [K]
 *   T_wall_prev, T_road_prev : lagged wall and road for canyon LW (frozen per outer iter)
 *   T1_roof, T1_wall, T1_road : subsurface boundary temps [K]
 *   T_canyon : canyon air temperature [K]
 *   T_atm_ref : reference atmosphere temperature [K] (for crown upward sensible)
 *   SW_down : incident shortwave [W/m²]
 *   LW_down : incident longwave from atmosphere [W/m²]
 *   Q_tree_SW_abs : absorbed shortwave by crown [W/m²] (from Phase 6.2a)
 *   alpha_roof, alpha_wall, alpha_road : shortwave albedo [-]
 *   eps_roof, eps_wall, eps_road, eps_leaf : longwave emissivity [-]
 *   k_th_roof, k_th_wall, k_th_road : thermal conductivity [W/m/K]
 *   dz_roof, dz_wall, dz_road : slab thickness [m]
 *   Ch_roof, Ch_wall, Ch_road, Ch_leaf : drag coefficient [-]
 *   U_ref : reference wind speed [m/s]
 *   rho_cp : air density * specific heat [J/m³/K]
 *   crown_view_factor : crown→canyon geometric view factor [-]
 *   crown_area_frac : crown area fraction [-] for effective area weighting
 *   max_iter : max Newton iterations
 *   tol_K : convergence tolerance [K]
 *
 * Outputs (via references):
 *   T_roof, T_wall, T_road, T_crown : final temperatures [K] (clamped)
 *   H_roof, H_wall, H_road : sensible heat fluxes [W/m²]
 *   H_crown_up : sensible heat from crown to atmosphere [W/m²]
 *   H_crown_down : sensible heat from crown to canyon [W/m²]
 *   n_iter_out : number of iterations completed
 *   residual_out : final residual (max dT)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void solve_facet_seb_4var_with_diag(
    // Input: current temperature estimates
    amrex::Real T_roof_in,
    amrex::Real T_wall_in,
    amrex::Real T_road_in,
    amrex::Real T_crown_in,
    // Input: lagged values for Gauss-Seidel (frozen per outer Newton iteration)
    amrex::Real T_wall_prev,
    amrex::Real T_road_prev,
    // Input: subsurface / canyon boundary temperatures
    amrex::Real T1_roof,
    amrex::Real T1_wall,
    amrex::Real T1_road,
    amrex::Real T_canyon,
    amrex::Real T_atm_ref,      // Phase 6.2b hotfix1: for crown upward H
    // Input: radiation
    amrex::Real SW_down,
    amrex::Real LW_down,
    amrex::Real Q_tree_SW_abs,
    // Input: albedo and emissivity
    amrex::Real alpha_roof,
    amrex::Real alpha_wall,
    amrex::Real alpha_road,
    amrex::Real eps_roof,
    amrex::Real eps_wall,
    amrex::Real eps_road,
    amrex::Real eps_leaf,     // Phase 6.2b: crown emissivity
    // Input: thermal properties
    amrex::Real k_th_roof,
    amrex::Real k_th_wall,
    amrex::Real k_th_road,
    amrex::Real dz_roof,
    amrex::Real dz_wall,
    amrex::Real dz_road,
    // Input: aerodynamic
    amrex::Real Ch_roof,
    amrex::Real Ch_wall,
    amrex::Real Ch_road,
    amrex::Real Ch_leaf,      // Phase 6.2b: crown drag coefficient
    amrex::Real U_ref,
    amrex::Real rho_cp,
    // Input: crown coupling
    amrex::Real crown_view_factor,  // Phase 6.2b: crown→canyon VF
    amrex::Real crown_area_frac,    // Phase 6.2b hotfix1: crown area fraction
    // Input: solver control
    int         max_iter,
    amrex::Real tol_K,
    // Output: final temperatures
    amrex::Real& T_roof_out,
    amrex::Real& T_wall_out,
    amrex::Real& T_road_out,
    amrex::Real& T_crown_out,
    // Output: sensible heat fluxes
    amrex::Real& H_roof_out,
    amrex::Real& H_wall_out,
    amrex::Real& H_road_out,
    amrex::Real& H_crown_up_out,    // Phase 6.2b hotfix1: crown → atmosphere
    amrex::Real& H_crown_down_out,  // Phase 6.2b hotfix1: crown → canyon
    // Output: diagnostics
    int&         n_iter_out,
    amrex::Real& residual_out)
{
    constexpr amrex::Real sigma_sb       = 5.670374419e-8;
    constexpr amrex::Real T_min_K        = 260.0;
    constexpr amrex::Real T_max_K        = 380.0;
    constexpr amrex::Real U_min          = 0.01;
    constexpr amrex::Real jacobian_min   = 1.e-10;
    constexpr amrex::Real conduction_min = 1.e-6;
    constexpr amrex::Real T_ref_min      = 200.0;
    constexpr amrex::Real T_ref_max      = 400.0;
    constexpr amrex::Real max_step_K     = 20.0;

    // Clamp and bound all input temperatures
    amrex::Real U      = amrex::max(std::abs(U_ref), U_min);
    amrex::Real T_can  = amrex::max(amrex::min(T_canyon, T_ref_max), T_ref_min);
    amrex::Real T_atm  = amrex::max(amrex::min(T_atm_ref, T_ref_max), T_ref_min);
    amrex::Real T1r    = amrex::max(amrex::min(T1_roof, T_ref_max), T_ref_min);
    amrex::Real T1w    = amrex::max(amrex::min(T1_wall, T_ref_max), T_ref_min);
    amrex::Real T1d    = amrex::max(amrex::min(T1_road, T_ref_max), T_ref_min);

    // Compute conduction coefficients (G = 2*k/dz if dz >= threshold)
    amrex::Real G_coeff_roof = 0.0;
    if (dz_roof >= conduction_min) {
        G_coeff_roof = 2.0 * k_th_roof / dz_roof;
    }
    // Bug C fix: Wall NOW has conduction
    amrex::Real G_coeff_wall = 0.0;
    if (dz_wall >= conduction_min) {
        G_coeff_wall = 2.0 * k_th_wall / dz_wall;
    }
    amrex::Real G_coeff_road = 0.0;
    if (dz_road >= conduction_min) {
        G_coeff_road = 2.0 * k_th_road / dz_road;
    }

    // Compute sensible heat coefficients
    amrex::Real H_coeff_roof = rho_cp * Ch_roof * U;
    amrex::Real H_coeff_wall = rho_cp * Ch_wall * U;
    amrex::Real H_coeff_road = rho_cp * Ch_road * U;
    amrex::Real H_coeff_crown = rho_cp * Ch_leaf * U;

    // Shortwave absorption
    amrex::Real SW_abs_roof  = (1.0 - alpha_roof) * amrex::max(SW_down, amrex::Real(0.0));
    amrex::Real SW_abs_wall  = (1.0 - alpha_wall) * amrex::max(SW_down, amrex::Real(0.0));
    amrex::Real SW_abs_road  = (1.0 - alpha_road) * amrex::max(SW_down, amrex::Real(0.0));

    // Initialize working temperatures with clamping
    amrex::Real T_roof  = amrex::max(amrex::min(T_roof_in, T_max_K), T_min_K);
    amrex::Real T_wall  = amrex::max(amrex::min(T_wall_in, T_max_K), T_min_K);
    amrex::Real T_road  = amrex::max(amrex::min(T_road_in, T_max_K), T_min_K);
    amrex::Real T_crown = amrex::max(amrex::min(T_crown_in, T_max_K), T_min_K);

    // Clamp lagged values for use in crown row (Bug E)
    amrex::Real T_wall_lag = amrex::max(amrex::min(T_wall_prev, T_max_K), T_min_K);
    amrex::Real T_road_lag = amrex::max(amrex::min(T_road_prev, T_max_K), T_min_K);

    // Gauss-Seidel Newton iteration (Bug F: LW feedback recomputed each outer iter)
    int n_iter = 0;
    amrex::Real residual_max = 1.0e10;
    for (; n_iter < max_iter; ++n_iter) {

        // Bug F fix: Recompute lagged crown-LW-feedback at START of each outer iteration
        // using LATEST T_crown from previous outer iteration (proper Gauss-Seidel)
        amrex::Real T4_crown  = T_crown * T_crown * T_crown * T_crown;
        amrex::Real LW_crown_rad = sigma_sb * T4_crown;
        amrex::Real LW_fb_roof = eps_roof * LW_crown_rad * crown_view_factor;
        amrex::Real LW_fb_wall = eps_wall * LW_crown_rad * crown_view_factor;
        amrex::Real LW_fb_road = eps_road * LW_crown_rad * crown_view_factor;

        amrex::Real dT_max = 0.0;

        // ========== Row 0: Roof ==========
        {
            amrex::Real T3 = T_roof * T_roof * T_roof;
            amrex::Real T4 = T3 * T_roof;
            amrex::Real LW_abs = eps_roof * (LW_down + LW_fb_roof - sigma_sb * T4);
            amrex::Real Rn     = SW_abs_roof + LW_abs;
            amrex::Real H      = H_coeff_roof * (T_roof - T_can);
            amrex::Real G      = G_coeff_roof * (T_roof - T1r);
            amrex::Real F      = Rn - H - G;

            amrex::Real Fp     = -4.0 * eps_roof * sigma_sb * T3 - H_coeff_roof - G_coeff_roof;
            if (std::abs(Fp) > jacobian_min) {
                amrex::Real step = -F / Fp;
                if (step >  max_step_K) step =  max_step_K;
                if (step < -max_step_K) step = -max_step_K;
                amrex::Real T_new = T_roof + step;
                T_roof = amrex::max(amrex::min(T_new, T_max_K), T_min_K);
                dT_max = amrex::max(dT_max, std::abs(step));
            }
        }

        // ========== Row 1: Wall (Bug C: NOW includes conduction) ==========
        {
            amrex::Real T3 = T_wall * T_wall * T_wall;
            amrex::Real T4 = T3 * T_wall;
            amrex::Real LW_abs = eps_wall * (LW_down + LW_fb_wall - sigma_sb * T4);
            amrex::Real Rn     = SW_abs_wall + LW_abs;
            amrex::Real H      = H_coeff_wall * (T_wall - T_can);
            amrex::Real G      = G_coeff_wall * (T_wall - T1w);  // NEW: wall conduction
            amrex::Real F      = Rn - H - G;

            amrex::Real Fp     = -4.0 * eps_wall * sigma_sb * T3 - H_coeff_wall - G_coeff_wall;
            if (std::abs(Fp) > jacobian_min) {
                amrex::Real step = -F / Fp;
                if (step >  max_step_K) step =  max_step_K;
                if (step < -max_step_K) step = -max_step_K;
                amrex::Real T_new = T_wall + step;
                T_wall = amrex::max(amrex::min(T_new, T_max_K), T_min_K);
                dT_max = amrex::max(dT_max, std::abs(step));
            }
        }

        // ========== Row 2: Road ==========
        {
            amrex::Real T3 = T_road * T_road * T_road;
            amrex::Real T4 = T3 * T_road;
            amrex::Real LW_abs = eps_road * (LW_down + LW_fb_road - sigma_sb * T4);
            amrex::Real Rn     = SW_abs_road + LW_abs;
            amrex::Real H      = H_coeff_road * (T_road - T_can);
            amrex::Real G      = G_coeff_road * (T_road - T1d);
            amrex::Real F      = Rn - H - G;

            amrex::Real Fp     = -4.0 * eps_road * sigma_sb * T3 - H_coeff_road - G_coeff_road;
            if (std::abs(Fp) > jacobian_min) {
                amrex::Real step = -F / Fp;
                if (step >  max_step_K) step =  max_step_K;
                if (step < -max_step_K) step = -max_step_K;
                amrex::Real T_new = T_road + step;
                T_road = amrex::max(amrex::min(T_new, T_max_K), T_min_K);
                dT_max = amrex::max(dT_max, std::abs(step));
            }
        }

        // ========== Row 3: Crown (Bug D+E fixes) ==========
        {
            amrex::Real T3 = T_crown * T_crown * T_crown;
            amrex::Real T4 = T3 * T_crown;

            // Bug E fix: canyon LW upward using CURRENT T_roof and LAGGED wall/road
            // Crown receives upward LW from roof (already updated in Gauss-Seidel order),
            // and lagged wall/road from previous outer iteration (semi-implicit)
            amrex::Real T4_roof_cur  = T_roof * T_roof * T_roof * T_roof;
            amrex::Real T4_wall_lag  = T_wall_lag * T_wall_lag * T_wall_lag * T_wall_lag;
            amrex::Real T4_road_lag  = T_road_lag * T_road_lag * T_road_lag * T_road_lag;
            amrex::Real LW_up_canyon = sigma_sb * (
                  eps_roof * T4_roof_cur
                + eps_wall * T4_wall_lag
                + eps_road * T4_road_lag) / 3.0;

            amrex::Real SW_abs   = Q_tree_SW_abs;
            amrex::Real LW_in    = eps_leaf * crown_area_frac *
                                  (LW_down + (1.0 - crown_view_factor) * LW_up_canyon);
            amrex::Real LW_out   = 2.0 * eps_leaf * crown_area_frac * sigma_sb * T4;

            // Bug D fix: split H_crown into upward (to atmosphere) and downward (to canyon)
            amrex::Real H_up     = H_coeff_crown * crown_area_frac * (T_crown - T_atm);
            amrex::Real H_down   = H_coeff_crown * crown_area_frac * (T_crown - T_can);

            amrex::Real F  = SW_abs + LW_in - LW_out - H_up - H_down;
            // Jacobian: diagonal only; off-diagonal coupling is 1-outer-iter lagged per Contract #31
            amrex::Real Fp = -8.0 * eps_leaf * crown_area_frac * sigma_sb * T3
                           - 2.0 * H_coeff_crown * crown_area_frac;

            if (std::abs(Fp) > jacobian_min) {
                amrex::Real step = -F / Fp;
                if (step >  max_step_K) step =  max_step_K;
                if (step < -max_step_K) step = -max_step_K;
                amrex::Real T_new = T_crown + step;
                T_crown = amrex::max(amrex::min(T_new, T_max_K), T_min_K);
                dT_max = amrex::max(dT_max, std::abs(step));
            }
        }

        // Bug B fix: track dT_max from all rows and break on convergence
        residual_max = dT_max;
        if (dT_max < tol_K) {
            ++n_iter;
            break;
        }
    }

    // Final outputs
    T_roof_out  = amrex::max(amrex::min(T_roof, T_max_K), T_min_K);
    T_wall_out  = amrex::max(amrex::min(T_wall, T_max_K), T_min_K);
    T_road_out  = amrex::max(amrex::min(T_road, T_max_K), T_min_K);
    T_crown_out = amrex::max(amrex::min(T_crown, T_max_K), T_min_K);

    // Compute final fluxes
    H_roof_out = H_coeff_roof * (T_roof_out - T_can);
    H_wall_out = H_coeff_wall * (T_wall_out - T_can);
    H_road_out = H_coeff_road * (T_road_out - T_can);
    // Bug D fix: split crown sensible into upward and downward
    H_crown_up_out   = H_coeff_crown * crown_area_frac * (T_crown_out - T_atm);
    H_crown_down_out = H_coeff_crown * crown_area_frac * (T_crown_out - T_can);

    // Diagnostics
    n_iter_out   = n_iter;
    residual_out = residual_max;
}

#endif // ERF_UCM_SEB_SOLVER_4VAR_H
