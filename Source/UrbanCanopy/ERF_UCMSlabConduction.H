/**
 * @file ERF_UCMSlabConduction.H
 * @brief GPU-safe header-only TDMA implicit-Euler slab conduction solver
 *
 * Implements the governing equation for vertical heat conduction through building
 * materials (roof, wall, road) using implicit Euler timestepping and Tridiagonal
 * Matrix Algorithm (TDMA / Thomas algorithm).
 *
 * **Governing Equation (Kusaka 2001, Chen 2011):**
 * \f[
 *   \rho c_p \frac{\partial T}{\partial t} = \frac{\partial}{\partial z}\left(k \frac{\partial T}{\partial z}\right)
 * \f]
 *
 * **Discretization (implicit Euler, uniform dz):**
 * \f[
 *   \rho c_p \frac{T_i^{n+1} - T_i^n}{\Delta t} = k \frac{T_{i+1}^{n+1} - 2T_i^{n+1} + T_{i-1}^{n+1}}{dz^2}
 * \f]
 *
 * Rearranged into standard tridiagonal all-plus form:
 * \f[
 *   a_i T_{i-1}^{n+1} + b_i T_i^{n+1} + c_i T_{i+1}^{n+1} = d_i
 * \f]
 *
 * where (all-plus convention):
 * - \f$ a_i = c_i = -\frac{k \Delta t}{\rho c_p dz^2} = -Fo \f$ (interior; NEGATIVE)
 * - \f$ b_i = 1 + 2\frac{k \Delta t}{\rho c_p dz^2} = 1 + 2Fo \f$ (interior)
 * - \f$ d_i = T_i^n \f$ (interior)
 * - Top BC (i=0): \f$ k \frac{\partial T}{\partial z}\big|_{\text{top}} = Q_{\text{top}} \f$
 * - Bottom BC (i=N-1): \f$ T = T_{\text{deep}} \f$
 *
 * **Sign convention note (Phase 3.5c-hotfix2):**
 * The all-plus TDMA convention requires the superdiagonal (c) and subdiagonal (a)
 * coefficients from a Laplacian discretization to be NEGATIVE, because the
 * physical equation produces `-k*(T_{i+1} - 2*T_i + T_{i-1})/dz^2` = `+2k/dz^2 * T_i - k/dz^2 * (T_{i+1} + T_{i-1})`.
 * Under-storing them as positive causes standard back-substitution
 * `T_i = gamma_i - alpha_i * T_{i+1}` to drift T_i downward each step even at
 * steady state (approximately -2*Fo*T_reference per step). Bug fixed.
 *
 * **Thomas Algorithm (TDMA):**
 * Forward elimination, back substitution. Stable when coefficients satisfy
 * diagonal dominance condition. See Numerical Recipes Sec. 2.4.
 *
 * **GPU Safety:**
 * All functions marked `AMREX_GPU_HOST_DEVICE`. No host-only I/O, no memory
 * allocation. Thread-safe: each thread solves one independent column.
 * Array sizes fixed at compile-time or caller-provided stack arrays.
 *
 * References:
 *  - Kusaka et al. (2001). "A Simple Single-Layer Urban Canopy Model..." J. Meteor. Soc. Japan.
 *  - Chen et al. (2011). "Coupling an advanced land surface-hydrology model..."
 *  - WRF module_sf_urban.F, multi_layer subroutine (https://github.com/wrf-model/WRF)
 *  - Numerical Recipes, 2nd ed., Ch. 2.4 (Tridiagonal systems)
 */

#ifndef ERF_UCM_SLAB_CONDUCTION_H
#define ERF_UCM_SLAB_CONDUCTION_H

#include <AMReX_REAL.H>
#include <AMReX_Print.H>
#include <cmath>

/**
 * @brief Solve one column of implicit-Euler slab conduction using TDMA
 *
 * Solves the tridiagonal system arising from vertical heat diffusion through
 * a single slab (roof, wall, or road) over one timestep using the Thomas
 * algorithm. Implicit Euler ensures numerical stability for large Courant numbers.
 *
 * **Top Boundary Condition:** Neumann with specified heat flux
 * \f[ k \frac{\partial T}{\partial z}\big|_{\text{top}} = Q_{\text{top}} \f]
 *
 * Discretized as (forward difference into domain):
 * \f[ T_1 - T_0 = \frac{Q_{\text{top}} \cdot dz}{k} \f]
 *
 * **Bottom Boundary Condition:** Dirichlet
 * \f[ T_{\text{bottom}} = T_{\text{deep}} \f]
 *
 * **Stability:** Implicit Euler is unconditionally stable; no Courant restriction.
 * However, very large \f$ \Delta t \f$ may reduce accuracy. Use \f$ Fo = \frac{k \Delta t}{\rho c_p dz^2} \sim 0.1 \f$
 * for accurate diffusion timescale representation.
 *
 * @param[in,out] T        Array of N_layers layer temperatures [K].
 *                         On entry: temperatures at previous timestep.
 *                         On exit: temperatures at current timestep (implicit).
 * @param[in]     Q_top    Heat flux at top boundary (>0 = into slab) [W/m²]
 * @param[in]     T_deep   Temperature at bottom boundary [K]
 * @param[in]     k_therm  Uniform thermal conductivity [W/m/K]
 * @param[in]     rho_cp   Uniform volumetric heat capacity [J/m³/K]
 * @param[in]     dz       Layer thickness (uniform, L / N_layers) [m]
 * @param[in]     dt       Timestep [s]
 * @param[in]     N_layers Number of layers in vertical discretization [-]
 *
 * **Example Usage (host):**
 * ```cpp
 * Real T[4] = {300.0, 295.0, 290.0, 288.0};  // Initial temps
 * Real Q_top = 200.0;                         // W/m² downward into slab
 * Real T_deep = 287.0;                        // K (deep soil temp)
 * Real k = 1.5, rho_cp = 2.0e6;              // W/m/K, J/m³/K
 * Real dz = 0.3 / 4;                         // m (L/N)
 * Real dt = 600.0;                           // 10 min
 * advance_slab_conduction_column(T, Q_top, T_deep, k, rho_cp, dz, dt, 4);
 * // T now contains updated layer temperatures
 * ```
 *
 * **GPU Thread Safety:**
 * Each thread operates on independent T[] array (per cell, per facet).
 * No shared memory, no atomics. Non-reentrant in host code (fine for GPU kernels).
 *
 * **Cost:** O(N_layers) forward elimination + O(N_layers) back substitution.
 * Typical 4-layer slab: ~32 floating-point ops. Negligible GPU cost.
 */
AMREX_GPU_HOST_DEVICE
inline void advance_slab_conduction_column(amrex::Real* T,
                                           amrex::Real Q_top,
                                           amrex::Real T_deep,
                                           amrex::Real k_therm,
                                           amrex::Real rho_cp,
                                           amrex::Real dz,
                                           amrex::Real dt,
                                           int N_layers)
{
    // Fourier number (dimensionless diffusion parameter)
    amrex::Real Fo = k_therm * dt / (rho_cp * dz * dz);

    // ------------------------------------------------------------------------
    // Tridiagonal coefficients — ALL-PLUS TDMA convention
    //   a * T_{i-1} + b * T_i + c * T_{i+1} = d
    // Physical Laplacian -k*(T_{i+1} - 2*T_i + T_{i-1})/dz² produces
    // NEGATIVE off-diagonal terms in this convention.
    // Phase 3.5c-hotfix2: fixed sign of a and c (were incorrectly stored as +Fo).
    // ------------------------------------------------------------------------
    amrex::Real a_coeff = -Fo;         // subdiagonal (NEGATIVE)
    amrex::Real b_coeff = 1.0 + 2.0*Fo; // diagonal (positive)
    amrex::Real c_coeff = -Fo;         // superdiagonal (NEGATIVE)

    // Stack work arrays (typical N=4; hard cap at 50 for safety)
    amrex::Real alpha[50];  // Forward elimination coefficients
    amrex::Real gamma[50];  // Forward elimination RHS
    amrex::Real T_old[50];  // Copy of input

    if (N_layers > 50) {
        return;  // Safety check (caller must respect this cap)
    }

    for (int i = 0; i < N_layers; ++i) {
        T_old[i] = T[i];
    }

    // ------------------------------------------------------------------------
    // Top boundary condition: Neumann flux BC
    //   k * dT/dz|_top = Q_top  (positive Q_top = heat INTO slab from top)
    //
    // Discretization (control-volume around cell 0 of thickness dz):
    //   rho_cp * dz * (T_0^{n+1} - T_0^n) / dt = Q_top - k*(T_0^{n+1} - T_1^{n+1})/dz
    //
    // Rearranged into all-plus form:
    //   (1 + Fo) * T_0^{n+1}  +  (-Fo) * T_1^{n+1}  =  T_0^n + Q_top*dt/(rho_cp*dz)
    //
    // Phase 3.5c-hotfix2: c_top sign now matches all-plus TDMA convention
    // (was +Fo, correct value is -Fo). This eliminated the spurious ~0.09 K
    // per-step cooling drift observed even at steady-state initial conditions.
    // ------------------------------------------------------------------------
    amrex::Real b_top = 1.0 + Fo;
    amrex::Real c_top = -Fo;                                    // NEGATIVE (all-plus convention)
    amrex::Real d_top = T_old[0] + Q_top * dt / (rho_cp * dz);

    // Forward elimination, row 0
    alpha[0] = c_top / b_top;
    gamma[0] = d_top / b_top;

    // ------------------------------------------------------------------------
    // Interior rows i = 1 to N-2 (standard Thomas forward sweep)
    //   b_mod  = b - a * alpha_{i-1}
    //   gamma_i = (d_i - a * gamma_{i-1}) / b_mod
    //   alpha_i = c / b_mod
    // ------------------------------------------------------------------------
    for (int i = 1; i < N_layers - 1; ++i) {
        amrex::Real b_mod = b_coeff - a_coeff * alpha[i-1];
        gamma[i] = (T_old[i] - a_coeff * gamma[i-1]) / b_mod;
        alpha[i] = c_coeff / b_mod;
    }

    // ------------------------------------------------------------------------
    // Bottom boundary condition: Dirichlet
    //   T_{N-1} = T_deep (imposed in back-substitution).
    //
    // The last interior row (i = N-2) is finalized in the loop above.
    // Phase 3.5c-hotfix2: removed erroneous re-computation of gamma[N-2] that
    // was reading and overwriting the same slot with incorrect index math.
    // Standard Thomas correctly handles the top-most interior row via loop.
    // ------------------------------------------------------------------------

    // ------------------------------------------------------------------------
    // Back substitution
    //   T_i = gamma_i - alpha_i * T_{i+1}   for i = N-2 down to 0
    // With the correct all-plus signs on a and c, this reduces to the
    // identity mapping when Q_top = 0 and all T_i = T_deep (validated).
    // ------------------------------------------------------------------------
    T[N_layers - 1] = T_deep;  // Enforce Dirichlet at bottom

    for (int i = N_layers - 2; i >= 0; --i) {
        T[i] = gamma[i] - alpha[i] * T[i + 1];
    }
}

/**
 * @brief Multifab-based wrapper for slab conduction over a box
 *
 * For each cell in the box, advances the slab temperature array one timestep.
 * Thread-safe GPU kernel.
 *
 * @param[in,out] T_slab      Temperature MultiFab (N_layers components)
 * @param[in]     H_flux      Sensible heat flux MultiFab (scalar input BC).
 *                            Uses Newton H convention: positive = surface loses
 *                            heat to atm. The wrapper negates internally to
 *                            match slab Q_top convention (positive = into slab).
 * @param[in]     k_therm     Thermal conductivity MultiFab
 * @param[in]     rho_cp      Volumetric heat capacity MultiFab
 * @param[in]     is_urban    Urban mask iMultiFab
 * @param[in]     dt          Timestep [s]
 * @param[in]     params      UCM parameters (includes slab_L, slab_N_layers, slab_T_deep)
 * @param[in]     lev         AMR level
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void advance_slab_conduction_mfi(
    amrex::Array4<amrex::Real> const& T_slab_arr,
    amrex::Array4<amrex::Real> const& H_flux_arr,
    amrex::Array4<amrex::Real> const& k_therm_arr,
    amrex::Array4<amrex::Real> const& rho_cp_arr,
    amrex::Array4<int> const& is_urban_arr,
    amrex::Box const& bx,
    amrex::Real dt,
    int N_layers,
    amrex::Real slab_L,
    amrex::Real T_deep)
{
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        if (is_urban_arr(i,j,0) == 0) return;

        // Stack array for single-column temps (max 50 layers per header)
        amrex::Real T_col[50];

        if (N_layers > 50) {
            return;  // Safety check
        }

        // Load current temps from MultiFab
        for (int n = 0; n < N_layers; ++n) {
            T_col[n] = T_slab_arr(i, j, 0, n);
        }

        // Compute layer thickness
        amrex::Real dz = slab_L / static_cast<amrex::Real>(N_layers);

        // Phase 3.5c-hotfix: Sign convention reconciliation.
        // Newton H convention: positive = surface loses heat to atm (cooling).
        // Slab top BC convention: positive = heat flows INTO slab from top.
        // These are OPPOSITE conventions: when H < 0 (atm heats surface),
        // heat enters the slab from above, i.e., Q_top > 0. Hence negate.
        amrex::Real Q_top = -H_flux_arr(i, j, 0);
        amrex::Real kcond = k_therm_arr(i, j, 0);
        amrex::Real rhcp  = rho_cp_arr(i, j, 0);

        advance_slab_conduction_column(T_col, Q_top, T_deep, kcond, rhcp, dz, dt, N_layers);

        // Store updated temps back to MultiFab
        for (int n = 0; n < N_layers; ++n) {
            T_slab_arr(i, j, 0, n) = T_col[n];
        }
    });
}

#endif  // ERF_UCM_SLAB_CONDUCTION_H