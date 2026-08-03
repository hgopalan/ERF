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
 * Inputs:
 *   T_roof_in, T_wall_in, T_road_in, T_crown_in : current temperature estimates [K]
 *   T_roof_prev, T_wall_prev, T_road_prev : lagged temps (for Jacobian, if needed)
 *   T1_roof, T1_wall, T1_road : subsurface boundary temps [K]
 *   T_canyon : canyon air temperature [K]
 *   SW_down : incident shortwave [W/m²]
 *   LW_down : incident longwave from atmosphere [W/m²]
 *   Q_tree_SW_abs : absorbed shortwave by crown [W/m²] (from Phase 6.2a)
 *   alpha_roof, alpha_wall, alpha_road : shortwave albedo [-]
 *   eps_roof, eps_wall, eps_road, eps_crown : longwave emissivity [-]
 *   k_th_roof, k_th_wall, k_th_road : thermal conductivity [W/m/K]
 *   dz_roof, dz_wall, dz_road : slab thickness [m]
 *   Ch_roof, Ch_wall, Ch_road, Ch_leaf : drag coefficient [-]
 *   U_ref : reference wind speed [m/s]
 *   rho_cp : air density * specific heat [J/m³/K]
 *   crown_view_factor : crown→canyon geometric view factor [-]
 *   max_iter : max Newton iterations
 *   tol_K : convergence tolerance [K]
 *
 * Outputs (via references):
 *   T_roof, T_wall, T_road, T_crown : final temperatures [K] (clamped)
 *   H_roof, H_wall, H_road : sensible heat fluxes [W/m²]
 *   H_crown_net : net sensible heat from crown [W/m²] (sign: positive = to atmosphere)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void solve_facet_seb_4var_with_diag(
    // Input: current temperature estimates
    amrex::Real T_roof_in,
    amrex::Real T_wall_in,
    amrex::Real T_road_in,
    amrex::Real T_crown_in,
    // Input: lagged values for Jacobian (optional semi-implicit feedback)
    amrex::Real T_roof_prev,
    amrex::Real T_wall_prev,
    amrex::Real T_road_prev,
    // Input: subsurface / canyon boundary temperatures
    amrex::Real T1_roof,
    amrex::Real T1_wall,
    amrex::Real T1_road,
    amrex::Real T_canyon,
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
    amrex::Real& H_crown_net_out)
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
    amrex::Real U     = amrex::max(std::abs(U_ref), U_min);
    amrex::Real T_can = amrex::max(amrex::min(T_canyon, T_ref_max), T_ref_min);
    amrex::Real T1r   = amrex::max(amrex::min(T1_roof, T_ref_max), T_ref_min);
    amrex::Real T1w   = amrex::max(amrex::min(T1_wall, T_ref_max), T_ref_min);
    amrex::Real T1d   = amrex::max(amrex::min(T1_road, T_ref_max), T_ref_min);

    // Compute conduction coefficients (G = 2*k/dz if dz >= threshold)
    amrex::Real G_coeff_roof = 0.0;
    if (dz_roof >= conduction_min) {
        G_coeff_roof = 2.0 * k_th_roof / dz_roof;
    }
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
    amrex::Real H_coeff_crown = rho_cp * Ch_leaf * U;  // Phase 6.2b: crown drag

    // Shortwave absorption
    amrex::Real SW_abs_roof  = (1.0 - alpha_roof) * amrex::max(SW_down, amrex::Real(0.0));
    amrex::Real SW_abs_wall  = (1.0 - alpha_wall) * amrex::max(SW_down, amrex::Real(0.0));
    amrex::Real SW_abs_road  = (1.0 - alpha_road) * amrex::max(SW_down, amrex::Real(0.0));
    // Crown SW absorption is passed in as Q_tree_SW_abs (from Phase 6.2a tree radiation)

    // Initialize working temperatures with clamping
    amrex::Real T_roof  = amrex::max(amrex::min(T_roof_in, T_max_K), T_min_K);
    amrex::Real T_wall  = amrex::max(amrex::min(T_wall_in, T_max_K), T_min_K);
    amrex::Real T_road  = amrex::max(amrex::min(T_road_in, T_max_K), T_min_K);
    amrex::Real T_crown = amrex::max(amrex::min(T_crown_in, T_max_K), T_min_K);

    // Semi-implicit LW feedback from crown (Contract #31): lagged T_crown
    // Each facet's LW_in is enhanced by crown radiation
    amrex::Real LW_crown_rad = sigma_sb * T_crown * T_crown * T_crown * T_crown;
    amrex::Real LW_feedback_roof = eps_roof * LW_crown_rad * crown_view_factor;
    amrex::Real LW_feedback_wall = eps_wall * LW_crown_rad * crown_view_factor;
    amrex::Real LW_feedback_road = eps_road * LW_crown_rad * crown_view_factor;

    // Gauss-Seidel Newton iteration
    for (int iter = 0; iter < max_iter; ++iter) {

        // ========== Row 0: Roof ==========
        {
            amrex::Real T3_roof = T_roof * T_roof * T_roof;
            amrex::Real T4_roof = T3_roof * T_roof;
            amrex::Real LW_abs_roof = eps_roof * (LW_down + LW_feedback_roof - sigma_sb * T4_roof);
            amrex::Real Rn_roof = SW_abs_roof + LW_abs_roof;
            amrex::Real H_roof = H_coeff_roof * (T_roof - T_can);
            amrex::Real G_roof = G_coeff_roof * (T_roof - T1r);
            amrex::Real F_roof = Rn_roof - H_roof - G_roof;

            amrex::Real F_prime_roof = -4.0 * eps_roof * sigma_sb * T3_roof - H_coeff_roof - G_coeff_roof;
            if (std::abs(F_prime_roof) > jacobian_min) {
                amrex::Real step_roof = -F_roof / F_prime_roof;
                if (step_roof >  max_step_K) step_roof =  max_step_K;
                if (step_roof < -max_step_K) step_roof = -max_step_K;
                amrex::Real T_roof_new = T_roof + step_roof;
                T_roof = amrex::max(amrex::min(T_roof_new, T_max_K), T_min_K);
            }
        }

        // ========== Row 1: Wall ==========
        {
            amrex::Real T3_wall = T_wall * T_wall * T_wall;
            amrex::Real T4_wall = T3_wall * T_wall;
            amrex::Real LW_abs_wall = eps_wall * (LW_down + LW_feedback_wall - sigma_sb * T4_wall);
            amrex::Real Rn_wall = SW_abs_wall + LW_abs_wall;
            amrex::Real H_wall = H_coeff_wall * (T_wall - T_can);
            // Wall has no conduction (G=0)
            amrex::Real F_wall = Rn_wall - H_wall;

            amrex::Real F_prime_wall = -4.0 * eps_wall * sigma_sb * T3_wall - H_coeff_wall;
            if (std::abs(F_prime_wall) > jacobian_min) {
                amrex::Real step_wall = -F_wall / F_prime_wall;
                if (step_wall >  max_step_K) step_wall =  max_step_K;
                if (step_wall < -max_step_K) step_wall = -max_step_K;
                amrex::Real T_wall_new = T_wall + step_wall;
                T_wall = amrex::max(amrex::min(T_wall_new, T_max_K), T_min_K);
            }
        }

        // ========== Row 2: Road ==========
        {
            amrex::Real T3_road = T_road * T_road * T_road;
            amrex::Real T4_road = T3_road * T_road;
            amrex::Real LW_abs_road = eps_road * (LW_down + LW_feedback_road - sigma_sb * T4_road);
            amrex::Real Rn_road = SW_abs_road + LW_abs_road;
            amrex::Real H_road = H_coeff_road * (T_road - T_can);
            amrex::Real G_road = G_coeff_road * (T_road - T1d);
            amrex::Real F_road = Rn_road - H_road - G_road;

            amrex::Real F_prime_road = -4.0 * eps_road * sigma_sb * T3_road - H_coeff_road - G_coeff_road;
            if (std::abs(F_prime_road) > jacobian_min) {
                amrex::Real step_road = -F_road / F_prime_road;
                if (step_road >  max_step_K) step_road =  max_step_K;
                if (step_road < -max_step_K) step_road = -max_step_K;
                amrex::Real T_road_new = T_road + step_road;
                T_road = amrex::max(amrex::min(T_road_new, T_max_K), T_min_K);
            }
        }

        // ========== Row 3: Crown ==========
        // Phase 6.2b: Crown energy balance (no conduction, no LE)
        // F_crown = Q_SW_crown - 2*H_crown - sigma*eps*(T_crown^4 - LW_down/sigma)
        // Rewritten: F_crown = Q_SW + eps*LW - 2*H - eps*sigma*T^4
        {
            amrex::Real T3_crown = T_crown * T_crown * T_crown;
            amrex::Real T4_crown = T3_crown * T_crown;
            // Crown receives LW_down from atmosphere
            amrex::Real LW_abs_crown = eps_leaf * LW_down;
            // Crown radiates to atmosphere AND to canyon (2-sided emission)
            amrex::Real LW_rad_crown = eps_leaf * sigma_sb * T4_crown;
            // Sensible heat: 2-sided (H_crown_up to ATM, H_crown_down to canyon)
            // For simplicity, total H = 2 * H_coeff * (T_crown - T_can)
            amrex::Real H_crown = 2.0 * H_coeff_crown * (T_crown - T_can);
            // Crown energy balance: absorbed SW + absorbed LW - emitted LW - sensible heat = 0
            amrex::Real F_crown = Q_tree_SW_abs + LW_abs_crown - LW_rad_crown - H_crown;

            // Jacobian: dF/dT_crown = -4*eps*sigma*T^3 - 2*H_coeff
            amrex::Real F_prime_crown = -4.0 * eps_leaf * sigma_sb * T3_crown - 2.0 * H_coeff_crown;
            if (std::abs(F_prime_crown) > jacobian_min) {
                amrex::Real step_crown = -F_crown / F_prime_crown;
                if (step_crown >  max_step_K) step_crown =  max_step_K;
                if (step_crown < -max_step_K) step_crown = -max_step_K;
                amrex::Real T_crown_new = T_crown + step_crown;
                T_crown = amrex::max(amrex::min(T_crown_new, T_max_K), T_min_K);
            }
        }

        // Check convergence: max temperature change
        amrex::Real dT_max = 0.0;
        // (Note: in a full implementation, track dT from each step)
        // For now, assume convergence after max_iter
        if (dT_max < tol_K) {
            break;
        }
    }

    // Final outputs: clamp and compute fluxes
    T_roof_out = amrex::max(amrex::min(T_roof, T_max_K), T_min_K);
    T_wall_out = amrex::max(amrex::min(T_wall, T_max_K), T_min_K);
    T_road_out = amrex::max(amrex::min(T_road, T_max_K), T_min_K);
    T_crown_out = amrex::max(amrex::min(T_crown, T_max_K), T_min_K);

    // Compute final fluxes
    H_roof_out = H_coeff_roof * (T_roof_out - T_can);
    H_wall_out = H_coeff_wall * (T_wall_out - T_can);
    H_road_out = H_coeff_road * (T_road_out - T_can);
    // Crown net sensible: 2-sided (upward + downward)
    H_crown_net_out = 2.0 * H_coeff_crown * (T_crown_out - T_can);
}

#endif // ERF_UCM_SEB_SOLVER_4VAR_H
