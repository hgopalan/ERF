#ifndef ERF_UCM_SEB_SOLVER_H
#define ERF_UCM_SEB_SOLVER_H

#include <AMReX_REAL.H>
#include <cmath>

/**
 * @file ERF_UCMSEBSolver.H
 * @brief GPU-safe header-only Surface Energy Balance (SEB) Newton solver (Phase 3.5A)
 *
 * Solves per-cell, per-facet:
 *   F(T_skin) = Rn - H - G = 0
 *   Rn = (1-alpha)*SW_down + eps*(LW_down - sigma*T_skin^4)
 *   H  = rho_cp * Ch * |U| * (T_skin - T_canyon)
 *   G  = 2*k_th/dz * (T_skin - T1)
 * Jacobian: F' = -4*eps*sigma*T_skin^3 - rho_cp*Ch*|U| - 2*k_th/dz
 *
 * Safety: T_canyon and T1 are clamped to [200, 400] K before entering the
 * Newton loop. This prevents divergence on the first timestep if the
 * canyon-air budget has not yet been initialised to a physical value.
 *
 * References:
 *  - Kusaka et al. (2001) J. Meteor. Soc. Japan
 *  - Chen et al. (2011)
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
    constexpr amrex::Real T_min_K        = 250.0;
    constexpr amrex::Real T_max_K        = 380.0;
    constexpr amrex::Real U_min          = 0.01;
    constexpr amrex::Real jacobian_min   = 1.e-10;
    constexpr amrex::Real conduction_min = 1.e-6;
    // Safety bounds on reference temperatures.
    // Prevents Newton divergence if canyon-air or slab T1 are uninitialised.
    constexpr amrex::Real T_ref_min      = 200.0;
    constexpr amrex::Real T_ref_max      = 400.0;

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

        amrex::Real T_new = T_skin - F / F_prime;
        // Bound iterate to prevent runaway before convergence check
        T_new = amrex::max(amrex::min(T_new, T_max_K), T_min_K);

        amrex::Real dT = std::abs(T_new - T_skin);
        T_skin = T_new;
        if (dT < tol_K) break;
    }

    T_skin = amrex::max(amrex::min(T_skin, T_max_K), T_min_K);
    H_out  = H_coeff * (T_skin - T_can);
    return T_skin;
}

#endif  // ERF_UCM_SEB_SOLVER_H