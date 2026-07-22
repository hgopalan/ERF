/**
 * @file ERF_UCMAtmCoupling.H
 * @brief Atmospheric coupling for UCM (coarsen and inject fluxes)
 *
 * Implements the one-way coupling pipeline between the UCM (on a refined 2D slab)
 * and the ATM (on a coarser 3D domain). Two main functions:
 *
 * 1. **coarsen_ucm_flux_to_atm**: Coarsen UCM fluxes from refined UCM grid to ATM grid.
 *    Uses `amrex::average_down` when grid_ratio > 1; `amrex::MultiFab::Copy` when ratio = 1.
 *
 * 2. **apply_ucm_tendency_to_cc_source**: Apply exponential-decay vertical injection
 *    (after Mandel 2011 / WRF-SFIRE fire_tendency pattern) to cc_source at every k.
 *
 * **Key Design Rules (Phase 1.4):**
 * - All ParallelFor kernels use: `[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { ... }`
 * - Array4 access via `mf.array(mfi)` / `mf.const_array(mfi)` (by value), never `mf[mfi].array()` with `auto&`
 * - MultiFab copy via `amrex::MultiFab::Copy`, never `amrex::Copy`
 * - Every kernel guarded by `if (is_urban(...)==0) return;`
 * - Debug traces with `[UCM][1.4][function_name]` prefix on IO rank only
 * - Terrain-following coords: `z_phys_cc(i,j,k) - z_phys_cc(i,j,klo)` for height-above-surface
 *
 * References:
 *  - WRF Single-Layer Urban Canopy Model (Chen et al., 2011)
 *  - Mandel et al. (2011) "Coupled atmosphere-wildland fire modeling"
 *  - WRF Module: module_sf_urban.F / module_fire_tendency
 *  - Source/Dust/ERF_DustAtmCoupling.H (coarsen pattern)
 *  - Source/Fire/ERF_FireAtmCoupling.H (exponential injection pattern)
 */

#ifndef ERF_UCM_ATM_COUPLING_H
#define ERF_UCM_ATM_COUPLING_H

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Geometry.H>

/**
 * Refine an ATM-grid MultiFab onto the UCM 2D slab grid via piecewise-constant
 * injection. Handles both 3D ATM MFs (uses k = klo_atm plane) and 2D ATM slab MFs.
 */
void refine_atm_to_ucm(amrex::MultiFab&       Q_ucm_out,
                       const amrex::MultiFab& Q_atm_in,
                       int                    grid_ratio,
                       int                    klo_atm);

/**
 * @brief Coarsen UCM fluxes to ATM grid.
 *
 * When grid_ratio == 1, performs a direct copy. When grid_ratio > 1, uses
 * `amrex::average_down` with refinement vector (grid_ratio, grid_ratio, 1) to
 * spatially average the finer UCM flux to the coarser ATM grid (matching Dust pattern).
 *
 * **Debug trace format (IO rank only):**
 * ```
 * [UCM][1.4][coarsen_ucm_flux_to_atm]
 *   grid_ratio={ratio}
 *   before: Q_ucm min={min_ucm} max={max_ucm}
 *   after:  Q_atm min={min_atm} max={max_atm}
 * ```
 *
 * @param[out] Q_atm_out      Output coarsened flux on ATM grid (ncomp=1)
 * @param[in]  Q_ucm          Input flux on UCM grid (ncomp=1): H_sensible or LE_latent [W/m²]
 * @param[in]  geom_ucm       UCM geometry
 * @param[in]  geom_atm       ATM geometry
 * @param[in]  grid_ratio     Refinement factor (UCM vs ATM); typically 2 or 1
 * @param[in]  lev            AMR level (default 0)
 *
 * @note All ParallelFor kernels respect is_urban mask and GPU-safe (noexcept).
 * @note Ghost cells are properly handled by average_down.
 */
void coarsen_ucm_flux_to_atm(
    amrex::MultiFab&       Q_atm_out,
    const amrex::MultiFab& Q_ucm,
    const amrex::Geometry& geom_ucm,
    const amrex::Geometry& geom_atm,
    int                    grid_ratio,
    int                    lev = 0);

/**
 * @brief Apply exponential-decay vertical tendency to cc_source.
 *
 * Implements the WRF-SFIRE exponential injection pattern (Mandel 2011):
 * Sensible heat flux from UCM surface is vertically distributed with exponential decay
 * using depth-of-atmosphere and characteristic decay length scale.
 *
 * **Physics (per column at each k):**
 * \f[
 *   z_{\text{sfc}} = z_{\text{phys\_cc}}(i,j,k_{\text{lo}})
 * \f]
 * \f[
 *   z_k = z_{\text{phys\_cc}}(i,j,k) - z_{\text{sfc}}
 * \f]
 * \f[
 *   h(z_k) = \frac{Q_{\text{sfc}}}{C_{p,d}} \exp\left(-\frac{z_k}{\alpha_{\text{ucm}}}\right) \quad [\text{K/s}]
 * \f]
 * \f[
 *   \theta_{\text{tend}}(k) = -\rho(k) \frac{h(z_{k+1}) - h(z_k)}{\Delta z(k)} \quad [\text{K/s}]
 * \f]
 * \f[
 *   \text{cc\_source}(i,j,k,\text{RhoTheta\_comp}) += \text{feedback} \cdot \theta_{\text{tend}}(k)
 * \f]
 *
 * Similarly for latent heat (LE_atm → RhoQ1_comp) when has_moisture=true.
 *
 * **Debug trace format (IO rank only, once per call):**
 * ```
 * [UCM][1.4][apply_ucm_tendency_to_cc_source]
 *   atm_feedback={feedback} (one-way injection={is_active})
 *   k=0: rho={rho_0} dz={dz_0}
 *   alpha_ucm={alpha} [m]
 *   H_atm min={min_H} max={max_H} [W/m²]
 *   RhoTheta_tend min={min_tend} max={max_tend}
 *   expected surface tendency magnitude: rho*Q/Cp*(1-exp(-dz/alpha))/dz ≈ {expected} [K/s]
 * ```
 *
 * If atm_feedback == 0.0, function prints a one-time warning and returns without modification.
 *
 * @param[in,out] cc_source              Cell-centered source term MultiFab (modified)
 * @param[in]     H_atm                  Sensible heat flux on ATM grid [W/m²]
 * @param[in]     LE_atm                 Latent heat flux on ATM grid [W/m²]; may be nullptr
 * @param[in]     z_phys_cc              Physical z at cell centers on ATM grid [m]
 * @param[in]     S_old                  State (for density at each k); component 0 = RhoTheta
 * @param[in]     geom_atm               ATM geometry (to get dz)
 * @param[in]     is_urban_atm           is_urban mask on ATM grid (coarsened from UCM)
 * @param[in]     alpha_ucm              Exponential e-folding depth [m]; typical 10-20 m
 * @param[in]     feedback               Coupling strength [0.0, 1.0]; 0=off, 1=full
 * @param[in]     has_moisture           Whether moisture component exists
 * @param[in]     ucm_debug              Print per-step debug info (default false)
 * @param[in]     lev                    AMR level (default 0)
 *
 * @note Internally, uses `Cp_d = 1004.64 J/kg/K` for sensible heat; `L_V = 2.501e6 J/kg` for latent.
 * @note Handles terrain via z_phys_cc; flat terrain is special case where z_phys_cc(i,j,k) = k * dz.
 * @note All ParallelFor kernels respect is_urban mask and use correct GPU-safe signature.
 * @note LE_atm pointer may be nullptr; function must handle gracefully.
 */
void apply_ucm_tendency_to_cc_source(
    amrex::MultiFab&       cc_source,
    const amrex::MultiFab& H_atm,
    const amrex::MultiFab* LE_atm,
    const amrex::MultiFab& z_phys_cc,
    const amrex::MultiFab& S_old,
    const amrex::Geometry& geom_atm,
    const amrex::iMultiFab& is_urban_atm,
    amrex::Real            alpha_ucm,
    amrex::Real            feedback,
    bool                   has_moisture,
    bool                   ucm_debug = false,
    int                    lev = 0);

#endif  // ERF_UCM_ATM_COUPLING_H
