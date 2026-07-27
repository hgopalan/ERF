/**
 * @file ERF_UCMSEBSolver.H
 * @brief GPU-safe header-only Surface Energy Balance (SEB) Newton solver
 *
 * Implements per-cell, per-facet Newton-iteration SEB solver that computes
 * equilibrium skin temperature and sensible heat flux based on energy balance.
 *
 * **Governing Equation (facet SEB):**
 * \f[
 *   F(T_{skin}) = R_n(T_{skin}) - H(T_{skin}) - G(T_{skin}) = 0
 * \f]
 *
 * where:
 *   - Rn = (1-α)·SW_abs + ε·(LW_down - σ·T_skin⁴)  [W/m²]   (net radiation)
 *   - H  = ρ·Cp·Ch·|U|·(T_skin - T_canyon)           [W/m²]   (sensible heat)
 *   - G  = 2·k_th·(T_skin - T1) / dz                [W/m²]   (conduction to slab)
 *
 * **Newton Update:**
 * \f[
 *   T_{skin}^{n+1} = T_{skin}^n - \frac{F(T_{skin}^n)}{F'(T_{skin}^n)}
 * \f]
 *
 * Jacobian:
 * \f[
 *   F'(T_{skin}) = -4·ε·σ·T_{skin}^3 - ρ·Cp·Ch·|U| - 2·k_th/dz
 * \f]
 *
 * **Physical Constants:**
 *   - Stefan-Boltzmann: σ = 5.670374419e-8 W/m²/K⁴
 *   - Default Ch (heat exchange coeff): 0.01 (Kusaka 2001 Table 1)
 *
 * **GPU Safety:**
 * All functions marked `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE`.
 * No allocations, no host-only I/O, thread-safe.
 *
 * References:
 *  - Kusaka et al. (2001). "A Simple Single-Layer Urban Canopy Model..." J. Meteor. Soc. Japan.
 *  - Chen et al. (2011). "Coupling an advanced land surface-hydrology model..."
 *  - Martilli et al. (2002). "A parametrization of drag properties for unresolved buildings"
 */

#ifndef ERF_UCM_SEB_SOLVER_H
#define ERF_UCM_SEB_SOLVER_H

#include <AMReX_REAL.H>
#include <cmath>

/**
 * @brief Per-cell, per-facet Surface Energy Balance solver (Newton iteration).
 *
 * Solves the implicit facet SEB:
 *   F(T_skin) = Rn(T_skin) - H(T_skin) - G(T_skin) = 0
 *
 * where:
 *   Rn = (1-alpha)*SW_down + eps*(LW_down - sigma*T_skin^4)   [W/m^2]
 *   H  = rho*Cp*Ch*|U|*(T_skin - T_canyon)                  [W/m^2]
 *   G  = 2*k_th*(T_skin - T1) / dz                          [W/m^2]
 *        (T1 = first interior slab layer temperature)
 *
 * Newton update:
 *   T_skin_new = T_skin - F / F'
 *
 * F'(T_skin) = -4*eps*sigma*T_skin^3 - rho*Cp*Ch*|U| - 2*k_th/dz
 *
 * @param[in]  T_skin_in    Previous skin temperature [K]
 * @param[in]  T1           First interior slab layer temperature [K]
 * @param[in]  T_canyon     Canyon-air (reference) temperature [K]
 * @param[in]  SW_down      Downwelling shortwave [W/m^2]
 * @param[in]  LW_down      Downwelling longwave [W/m^2]
 * @param[in]  alpha        Facet albedo [-]
 * @param[in]  emiss        Facet emissivity [-]
 * @param[in]  k_th         Thermal conductivity [W/m/K]
 * @param[in]  dz_slab      Half-thickness of top slab layer [m]
 * @param[in]  Ch           Heat exchange coefficient [-]
 * @param[in]  U_ref        Reference wind speed [m/s]
 * @param[in]  rho_cp       rho * Cp [J/m^3/K]
 * @param[in]  max_iter     Maximum Newton iterations (default 20)
 * @param[in]  tol_K        Convergence tolerance [K] (default 0.01)
 * @param[out] H_out        Converged sensible heat flux [W/m^2]
 * @return                  Converged skin temperature [K]
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
    // Physical constants
    constexpr amrex::Real sigma_sb = 5.670374419e-8;  // Stefan-Boltzmann [W/m^2/K^4]
    constexpr amrex::Real T_min_K = 250.0;            // Minimum temperature bound [K]
    constexpr amrex::Real T_max_K = 380.0;            // Maximum temperature bound [K]
    constexpr amrex::Real U_min = 0.01;               // Minimum wind speed guard [m/s]
    constexpr amrex::Real jacobian_min = 1.e-10;      // Singular Jacobian guard
    constexpr amrex::Real conduction_min = 1.e-6;     // Minimum dz_slab guard [m]

    // Guard: prevent degenerate heat exchange due to zero wind
    // Use std::abs for HOST+DEVICE compatibility (amrex::abs is device-only)
    amrex::Real U = amrex::max(std::abs(U_ref), U_min);

    // Guard: if dz_slab too small, effectively no conduction term
    amrex::Real G_coeff = 0.0;
    if (dz_slab >= conduction_min) {
        G_coeff = 2.0 * k_th / dz_slab;  // [W/m^2/K]
    }

    // Coefficients (constant over Newton iterations)
    amrex::Real H_coeff = rho_cp * Ch * U;  // [J/m^3/K * m/s] = [W/m^2/K]
    amrex::Real SW_abs = (1.0 - alpha) * SW_down;  // [W/m^2]

    amrex::Real T_skin = T_skin_in;  // Current iterate

    // Newton iteration loop
    for (int iter = 0; iter < max_iter; ++iter) {
        // Evaluate residual F(T_skin) = Rn - H - G
        amrex::Real T_skin_4 = T_skin * T_skin * T_skin * T_skin;
        amrex::Real Rn = SW_abs + emiss * (LW_down - sigma_sb * T_skin_4);
        amrex::Real H = H_coeff * (T_skin - T_canyon);
        amrex::Real G = G_coeff * (T_skin - T1);
        amrex::Real F = Rn - H - G;

        // Evaluate Jacobian F'(T_skin)
        amrex::Real dRn_dT = -4.0 * emiss * sigma_sb * T_skin * T_skin * T_skin;
        amrex::Real dH_dT = H_coeff;
        amrex::Real dG_dT = G_coeff;
        amrex::Real F_prime = dRn_dT - dH_dT - dG_dT;

        // Guard against singular Jacobian
        if (std::abs(F_prime) < jacobian_min) {
            break;  // Cannot continue iteration
        }

        // Newton step
        amrex::Real T_skin_new = T_skin - F / F_prime;

        // Check convergence: |ΔT| < tol_K
        amrex::Real dT = std::abs(T_skin_new - T_skin);
        T_skin = T_skin_new;

        if (dT < tol_K) {
            break;  // Converged
        }
    }

    // Clamp output to physical bounds
    T_skin = amrex::max(amrex::min(T_skin, T_max_K), T_min_K);

    // Compute final sensible heat flux H_out at converged T_skin
    H_out = H_coeff * (T_skin - T_canyon);

    return T_skin;
}

#endif  // ERF_UCM_SEB_SOLVER_H