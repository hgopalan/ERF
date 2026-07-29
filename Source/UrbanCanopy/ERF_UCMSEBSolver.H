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
 * Phase 5.3-hotfix2: Optional LE_in argument added to solve_facet_seb_with_diag
 *   to include latent heat cooling from green roof / permeable pavement.
 *   Residual becomes:  F = Rn - H - G - LE = 0
 *   LE is treated as a constant during the Newton iteration (semi-implicit lag
 *   using T_skin from previous timestep), so the Jacobian F' is unchanged.
 *
 * Solves per-cell, per-facet:
 *   F(T_skin) = Rn - H - G - LE = 0
 *   Rn = (1-alpha)*SW_down + eps*(LW_down - sigma*T_skin^4)
 *   H  = rho_cp * Ch * |U| * (T_skin - T_canyon)
 *   G  = 2*k_th/dz * (T_skin - T1)
 *   LE = external latent-heat flux [W/m^2] (0 for standard facets)
 * Jacobian: F' = -4*eps*sigma*T_skin^3 - rho_cp*Ch*|U| - 2*k_th/dz
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
    constexpr amrex::Real T_min_K        = 260.0;
    constexpr amrex::Real T_max_K        = 380.0;
    constexpr amrex::Real U_min          = 0.01;
    constexpr amrex::Real jacobian_min   = 1.e-10;
    constexpr amrex::Real conduction_min = 1.e-6;
    constexpr amrex::Real T_ref_min      = 200.0;
    constexpr amrex::Real T_ref_max      = 400.0;
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

        amrex::Real step = -F / F_prime;

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
 * @brief Phase 3.5a-hotfix / 5.3-hotfix2: Instrumented SEB solver with LE_in.
 *
 * LE_in [W/m^2] is subtracted from the residual: F = Rn - H - G - LE.
 * Pass 0.0 for standard facets (walls, non-green roofs, non-permeable roads).
 * Pass fields.LE_green_roof_diag(i,j,0)   for green-roof cells.
 * Pass fields.LE_permeable_road_diag(i,j,0) for permeable-road cells.
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
    amrex::Real LE_in,                // Phase 5.3-hotfix2: latent heat [W/m^2]
    // Output: final T_skin (clamped)
    amrex::Real& T_skin_out,
    // Output: H_sensible
    amrex::Real& H_out,
    // Output: diagnostics
    amrex::Real& T_skin_unclamped,
    int&         n_iter,
    amrex::Real& residual_final,
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

    amrex::Real U     = amrex::max(std::abs(U_ref), U_min);
    amrex::Real T_can = amrex::max(amrex::min(T_canyon, T_ref_max), T_ref_min);
    amrex::Real T1_s  = amrex::max(amrex::min(T1,      T_ref_max), T_ref_min);

    amrex::Real G_coeff = 0.0;
    if (dz_slab >= conduction_min) {
        G_coeff = 2.0 * k_th / dz_slab;
    }

    amrex::Real H_coeff = rho_cp * Ch * U;
    SW_abs_out  = (1.0 - alpha) * amrex::max(SW_down, amrex::Real(0.0));

    // Phase 5.3-hotfix2: LE is a constant during Newton (lagged T_skin).
    // Guard against nonfinite / negative (condensation) values.
    amrex::Real LE_use = (amrex::Math::isfinite(LE_in)) ? LE_in : amrex::Real(0.0);

    amrex::Real T_skin = amrex::max(amrex::min(T_skin_in, T_max_K), T_min_K);

    n_iter = 0;
    T_skin_unclamped = T_skin;

    for (int iter = 0; iter < max_iter; ++iter) {
        n_iter = iter + 1;

        amrex::Real T3    = T_skin * T_skin * T_skin;
        amrex::Real T4    = T3 * T_skin;
        LW_abs_out = emiss * (LW_down - sigma_sb * T4);
        amrex::Real Rn    = SW_abs_out + LW_abs_out;
        H_sens_out = H_coeff * (T_skin - T_can);
        G_cond_out = G_coeff * (T_skin - T1_s);
        // Phase 5.3-hotfix2: subtract LE from residual (surface loses heat)
        amrex::Real F     = Rn - H_sens_out - G_cond_out - LE_use;
        residual_final    = std::abs(F);

        amrex::Real F_prime = -4.0 * emiss * sigma_sb * T3 - H_coeff - G_coeff;
        if (std::abs(F_prime) < jacobian_min) break;

        amrex::Real step = -F / F_prime;

        if (step >  max_step_K) step =  max_step_K;
        if (step < -max_step_K) step = -max_step_K;

        amrex::Real T_new_raw = T_skin + step;
        amrex::Real T_new = amrex::max(amrex::min(T_new_raw, T_max_K), T_min_K);

        amrex::Real dT = std::abs(T_new - T_skin);
        T_skin = T_new;
        if (dT < tol_K) {
            T_skin_unclamped = T_new_raw;
            break;
        }
    }

    if (n_iter >= max_iter && residual_final > tol_K) {
        T_skin_unclamped = T_skin;
    }

    T_skin_out = amrex::max(amrex::min(T_skin, T_max_K), T_min_K);
    H_out  = H_coeff * (T_skin_out - T_can);
}
#endif  // ERF_UCM_SEB_SOLVER_H