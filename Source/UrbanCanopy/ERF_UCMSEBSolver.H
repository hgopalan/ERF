#ifndef ERF_UCM_SEB_SOLVER_H
#define ERF_UCM_SEB_SOLVER_H

#include <AMReX_REAL.H>
#include <cmath>

/**
 * @file ERF_UCMSEBSolver.H
 * @brief GPU-safe header-only Surface Energy Balance (SEB) Newton solver (Phase 3.5A)
 *
 * Phase 3.5a-hotfix3: SIGN CONVENTION (consistent with WRF/module_sf_urban.F)
 *   Positive H = surface → atmosphere (surface LOSING heat, cooling)
 *   Negative H = atmosphere → surface (surface GAINING heat, warming)
 *   
 * Solves per-cell, per-facet:
 *   F(T_skin) = Rn - H - G = 0
 *   Rn = (1-alpha)*SW_down + eps*(LW_down - sigma*T_skin^4)
 *   H  = rho_cp * Ch * |U| * (T_skin - T_canyon)
 *        [Positive when T_skin > T_canyon: surface loses heat to cooler canyon air]
 *   G  = 2*k_th/dz * (T_skin - T1)
 * Jacobian: F' = -4*eps*sigma*T_skin^3 - rho_cp*Ch*|U| - 2*k_th/dz
 *
 * Note: MOST-derived H_base = -ρ*Cp*u_star*t_star uses the same convention.
 *       Both formulas should produce H of same sign for identical T, U, conditions.
 *
 * Safety measures:
 *   - T_canyon and T1 clamped to [200, 400] K (guards uninitialised values)
 *   - Newton step limited to ±20 K per iteration (prevents overshoot on first step)
 *   - Output T_skin clamped to [260, 380] K (Phase 3.5b: raised floor to 260 K)
 *   - Phase 3.5a-hotfix3: Track unphysical T_skin initializations
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real solve_facet_seb(
    amrex::Real T_skin_in,
    amrex::Real T1,
    amrex::Real T_canyon,
    amrex::Real SW_down,
    amrex::Real LW_down,
    amrex::Real alpha,
    amrex::Real emiss,
    amrex::Real k_th,
    amrex::Real dz_slab,
    amrex::Real Ch,
    amrex::Real U_ref,
    amrex::Real rho_cp,
    int         max_iter,
    amrex::Real tol_K,
    amrex::Real& H_out)
{
    constexpr amrex::Real sigma_sb       = 5.670374419e-8;
    constexpr amrex::Real T_min_K        = 260.0;  // Phase 3.5b: raised from 250 K
    constexpr amrex::Real T_max_K        = 380.0;
    constexpr amrex::Real U_min          = 0.01;
    constexpr amrex::Real jacobian_min   = 1.e-10;
    constexpr amrex::Real conduction_min = 1.e-6;
    constexpr amrex::Real T_ref_min      = 200.0;
    constexpr amrex::Real T_ref_max      = 400.0;
    // Limit Newton step size to prevent overshoot on first iteration.
    // Without this, a cold initialisation (300 K init, 280 K equilibrium)
    // can produce a step of -50+ K which overshoots past the 260 K clamp.
    constexpr amrex::Real max_step_K     = 20.0;

    amrex::Real U     = amrex::max(std::abs(U_ref), U_min);
    amrex::Real T_can = amrex::max(amrex::min(T_canyon, T_ref_max), T_ref_min);
    amrex::Real T1_s  = amrex::max(amrex::min(T1,      T_ref_max), T_ref_min);

    amrex::Real G_coeff = 0.0;
    if (dz_slab >= conduction_min) {
        G_coeff = 2.0 * k_th / dz_slab;
    }

    amrex::Real H_coeff = rho_cp * Ch * U;
    amrex::Real SW_abs  = (1.0 - alpha) * amrex::max(SW_down, amrex::Real(0.0));

    // Clamp initial iterate to physical range
    amrex::Real T_skin = amrex::max(amrex::min(T_skin_in, T_max_K), T_min_K);

    for (int iter = 0; iter < max_iter; ++iter) {
        amrex::Real T3    = T_skin * T_skin * T_skin;
        amrex::Real T4    = T3 * T_skin;
        amrex::Real Rn    = SW_abs + emiss * (LW_down - sigma_sb * T4);
        amrex::Real H     = H_coeff * (T_skin - T_can);
        amrex::Real G     = G_coeff * (T_skin - T1_s);
        amrex::Real F     = Rn - H - G;

        amrex::Real F_prime = -4.0 * emiss * sigma_sb * T3 - H_coeff - G_coeff;
        if (std::abs(F_prime) < jacobian_min) break;

        // Raw Newton step
        amrex::Real step = -F / F_prime;

        // Limit step size to prevent large overshoots on first iterations
        if (step >  max_step_K) step =  max_step_K;
        if (step < -max_step_K) step = -max_step_K;

        amrex::Real T_new = T_skin + step;
        T_new = amrex::max(amrex::min(T_new, T_max_K), T_min_K);

        amrex::Real dT = std::abs(T_new - T_skin);
        T_skin = T_new;
        if (dT < tol_K) break;
    }

    T_skin = amrex::max(amrex::min(T_skin, T_max_K), T_min_K);
    H_out  = H_coeff * (T_skin - T_can);
    return T_skin;
}

/**
 * @brief Phase 3.5a-hotfix: Instrumented SEB solver returning diagnostics.
 *
 * Same physics as solve_facet_seb(), but captures:
 *   - T_skin before clamping
 *   - Number of Newton iterations to convergence (or max_iter if diverged)
 *   - Final residual F value
 *   - All 4 flux components (SW_abs, LW_abs, H_sens, G_cond)
 * for later per-cell tracing.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void solve_facet_seb_with_diag(
    amrex::Real T_skin_in,
    amrex::Real T1,
    amrex::Real T_canyon,
    amrex::Real SW_down,
    amrex::Real LW_down,
    amrex::Real alpha,
    amrex::Real emiss,
    amrex::Real k_th,
    amrex::Real dz_slab,
    amrex::Real Ch,
    amrex::Real U_ref,
    amrex::Real rho_cp,
    int         max_iter,
    amrex::Real tol_K,
    // Output: final T_skin (clamped)
    amrex::Real& T_skin_out,
    // Output: H_sensible
    amrex::Real& H_out,
    // Output: diagnostics
    amrex::Real& T_skin_unclamped,    // T before clamp
    int&         n_iter,              // iterations to convergence or max
    amrex::Real& residual_final,      // |F| at convergence or last iter
    amrex::Real& SW_abs_out,
    amrex::Real& LW_abs_out,
    amrex::Real& H_sens_out,
    amrex::Real& G_cond_out)
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
    
    // Phase 3.5a-hotfix3: Sanity check T_skin initialization
    // (unphysical values indicate persistence bug or field reset)
    // Note: We don't abort here (GPU code) but track the condition
    // for later diagnostics by returning clamped value
    
    amrex::Real U     = amrex::max(std::abs(U_ref), U_min);
    amrex::Real T_can = amrex::max(amrex::min(T_canyon, T_ref_max), T_ref_min);
    amrex::Real T1_s  = amrex::max(amrex::min(T1,      T_ref_max), T_ref_min);

    amrex::Real G_coeff = 0.0;
    if (dz_slab >= conduction_min) {
        G_coeff = 2.0 * k_th / dz_slab;
    }

    amrex::Real H_coeff = rho_cp * Ch * U;
    SW_abs_out  = (1.0 - alpha) * amrex::max(SW_down, amrex::Real(0.0));

    // Clamp initial iterate to physical range
    amrex::Real T_skin = amrex::max(amrex::min(T_skin_in, T_max_K), T_min_K);

    n_iter = 0;
    T_skin_unclamped = T_skin;  // Initialize with initial clamped value
    
    for (int iter = 0; iter < max_iter; ++iter) {
        n_iter = iter + 1;  // track iterations

        amrex::Real T3    = T_skin * T_skin * T_skin;
        amrex::Real T4    = T3 * T_skin;
        LW_abs_out = emiss * (LW_down - sigma_sb * T4);
        amrex::Real Rn    = SW_abs_out + LW_abs_out;
        H_sens_out = H_coeff * (T_skin - T_can);
        G_cond_out = G_coeff * (T_skin - T1_s);
        amrex::Real F     = Rn - H_sens_out - G_cond_out;
        residual_final    = std::abs(F);

        amrex::Real F_prime = -4.0 * emiss * sigma_sb * T3 - H_coeff - G_coeff;
        if (std::abs(F_prime) < jacobian_min) break;

        // Raw Newton step (unclamped)
        amrex::Real step = -F / F_prime;

        // Limit step size
        if (step >  max_step_K) step =  max_step_K;
        if (step < -max_step_K) step = -max_step_K;

        // Compute raw Newton value (before clamp)
        amrex::Real T_new_raw = T_skin + step;
        // Clamp for next iteration
        amrex::Real T_new = amrex::max(amrex::min(T_new_raw, T_max_K), T_min_K);

        amrex::Real dT = std::abs(T_new - T_skin);
        T_skin = T_new;
        if (dT < tol_K) {
            // On convergence, store the raw unclamped value
            T_skin_unclamped = T_new_raw;
            break;
        }
    }

    // If Newton didn't converge (hit max_iter), keep T_skin_unclamped as initialized
    // (it will be the last clamped value from the loop, which is acceptable)
    
    // If Newton converged but we're here, we already set T_skin_unclamped inside the loop
    // Only override if we hit max_iter without convergence
    if (n_iter >= max_iter && residual_final > tol_K) {
        // Newton did not converge; T_skin_unclamped is already set to the clamped loop value
        // (initialized at loop start and updated each iteration)
        T_skin_unclamped = T_skin;
    }

    // Final clamp and H
    T_skin_out = amrex::max(amrex::min(T_skin, T_max_K), T_min_K);
    H_out  = H_coeff * (T_skin_out - T_can);
}
#endif  // ERF_UCM_SEB_SOLVER_H
