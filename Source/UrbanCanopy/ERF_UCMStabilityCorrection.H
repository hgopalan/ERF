/**
 * @file ERF_UCMStabilityCorrection.H
 * @brief Stability-aware correction to canyon-atmosphere heat exchange coefficient (Phase 3.4)
 *
 * Implements the Obukhov-length-corrected heat exchange coefficient as follows:
 *
 * ```
 * Ch_corrected(i,j) = Ch_base * Phi_h_correction(zeta(i,j))
 * ```
 *
 * where:
 * - `zeta(i,j) = zref / L(i,j)` (dimensionless stability parameter, from Obukhov length L)
 * - `Phi_h_correction` = heat stability function (Businger-Dyer formulation)
 * - `Ch_base` = baseline heat exchange coefficient (computed from u*, theta*, before stability correction)
 *
 * **Businger-Dyer stability functions (Dyer 1974, Businger et al. 1971):**
 *
 * For stable conditions (zeta > 0):
 *   Phi_h(zeta) = 1 + beta_h * zeta
 *   (Ch corrected = Ch_base / (1 + beta_h * zeta) for stability reduction in stable ABL)
 *
 * For unstable conditions (zeta < 0):
 *   Phi_h(zeta) = (1 - gamma_h * zeta)^(-0.5)  [note: exponent is -0.5 for heat]
 *   (Ch corrected = Ch_base * (1 - gamma_h * zeta)^(-0.5) for enhanced exchange in unstable ABL)
 *
 * **Clipping logic (Phase 3.4):**
 * - Stable: zeta is clipped to [0, zeta_max_stable] to prevent Ch → 0
 * - Unstable: zeta is clipped to [zeta_min_unstable, 0] to prevent Ch blow-up
 * - Default clipping: zeta_max_stable = 2.0, zeta_min_unstable = -5.0
 *
 * **References:**
 * - Businger et al. (1971): "Flux-profile relationships in the atmospheric surface layer"
 * - Dyer (1974): "A review of flux-profile relationships" Boundary-Layer Meteorology 7(3)
 * - WRF module_sf_urban.F: SLUCM implementation in compressible WRF core
 * - MRF/YSU PBL scheme: MRF_ComputeDiffusivityMRF.cpp, ERF_ComputeDiffusivityYSU.cpp
 *
 * **Integration points (Phase 3.4+):**
 * 1. Called from `UCMLayer::compute_facet_heat_fluxes()` if `use_stability_correction = true`
 * 2. Receives Obukhov length `olen` from `SurfLayer->get_olen(lev)` (populated by MRF/YSU/MYNN2.5)
 * 3. Corrects Ch before SEB solution for canyon walls/roof/road
 */

#ifndef ERF_UCM_STABILITY_CORRECTION_H
#define ERF_UCM_STABILITY_CORRECTION_H

#include <AMReX_REAL.H>
<<<<<<< HEAD
#include <AMReX_GpuQualifiers.H>
=======
#include <AMReX_GpuQualifiers.H> 
>>>>>>> ce9809f8 (Fix)
#include <cmath>

/**
 * @struct StabilityFunctions
 * @brief Businger-Dyer stability correction functions (Phi_h formulation)
 *
 * Computes the dimensionless heat transfer function Phi_h(zeta) and its
 * inverse (needed to correct exchange coefficient Ch).
 *
 * **Terminology (from Dyer 1974, Businger et al. 1971):**
 * - zeta = z / L (dimensionless stability parameter)
 *   - zeta > 0: stable (nocturnal, strong inversion)
 *   - zeta < 0: unstable (daytime, buoyancy-driven)
 *   - zeta ≈ 0: neutral (mixed neutral/weak stability)
 * - Phi_h(zeta) = heat transfer similarity function
 * - Ch_corrected = Ch_base / Phi_h(zeta) is the corrected exchange coefficient
 *
 * **Constants (Businger-Dyer Dyer 1974):**
 * - beta_h = 5.0 (stable heat transfer coefficient)
 * - gamma_h = 16.0 (unstable heat transfer coefficient)
 */
struct StabilityFunctions
{
    static constexpr amrex::Real beta_h = 5.0;   ///< Stable Phi_h coefficient (Dyer 1974)
    static constexpr amrex::Real gamma_h = 16.0; ///< Unstable Phi_h coefficient (Dyer 1974)

    /**
     * @brief Compute Phi_h(zeta) = z/L-dependent heat transfer function (Businger-Dyer)
     *
     * **Stable (zeta > 0):**
     *   Phi_h = 1 + beta_h * zeta
     *   (Ch_corrected = Ch_base / (1 + beta_h * zeta) → reduced in stable conditions)
     *
     * **Unstable (zeta < 0):**
     *   Phi_h = (1 - gamma_h * zeta)^(-0.5)
     *   (Ch_corrected = Ch_base / (1 - gamma_h * zeta)^(-0.5) → enhanced in unstable)
     *
     * @param zeta Dimensionless stability parameter z/L (pre-clipped by caller)
     * @return Phi_h(zeta) stability correction factor (dimensionless)
     */
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::Real
    phi_h (amrex::Real zeta)
    {
        if (zeta >= 0.0) {
            // Stable branch: Phi_h = 1 + beta_h * zeta
            return 1.0 + beta_h * zeta;
        } else {
            // Unstable branch: Phi_h = (1 - gamma_h * zeta)^(-0.5)
            amrex::Real arg = 1.0 - gamma_h * zeta;
            return 1.0 / std::sqrt(arg);  // Note: exponent is -0.5, so we take 1/sqrt
        }
    }

    /**
     * @brief Compute the inverse of Phi_h(zeta) for use in Ch correction
     *
     * Returns 1 / Phi_h(zeta), which is the stability correction multiplier applied to Ch:
     *
     *   Ch_corrected = Ch_base * (1 / Phi_h(zeta))
     *
     * **Stable (zeta >= 0):**
     *   1 / Phi_h = 1 / (1 + beta_h * zeta)
     *
     * **Unstable (zeta < 0):**
     *   1 / Phi_h = sqrt(1 - gamma_h * zeta)
     *
     * @param zeta Dimensionless stability parameter z/L (pre-clipped by caller)
     * @return 1 / Phi_h(zeta) stability correction multiplier for Ch (dimensionless)
     */
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    static amrex::Real
    phi_h_inverse (amrex::Real zeta)
    {
        if (zeta >= 0.0) {
            // Stable branch: 1 / Phi_h = 1 / (1 + beta_h * zeta)
            return 1.0 / (1.0 + beta_h * zeta);
        } else {
            // Unstable branch: 1 / Phi_h = sqrt(1 - gamma_h * zeta)
            amrex::Real arg = 1.0 - gamma_h * zeta;
            return std::sqrt(arg);
        }
    }
};

/**
 * @brief Compute stability-corrected heat exchange coefficient
 *
 * Applies Obukhov-length-based stability correction to the baseline heat
 * exchange coefficient Ch using Businger-Dyer similarity functions.
 *
 * **Usage pattern:**
 * ```cpp
 * amrex::Real Ch_base = ... ; // baseline from u*, theta* similarity
 * amrex::Real olen = ... ;    // Obukhov length (from SurfLayer)
 * amrex::Real zref = 2.0;     // reference height above roof [m]
 * amrex::Real zeta_max_stable = 2.0;
 * amrex::Real zeta_min_unstable = -5.0;
 * amrex::Real Ch_corrected = compute_ch_stability_correction(
 *     Ch_base, olen, zref, zeta_max_stable, zeta_min_unstable
 * );
 * ```
 *
 * @param[in] Ch_base           Baseline heat exchange coefficient (dimensionless)
 * @param[in] olen              Obukhov length [m] (from SurfLayer->get_olen)
 * @param[in] zref              Reference height above roof [m]
 * @param[in] zeta_max_stable   Upper clip for zeta in stable branch (default 2.0) [-]
 * @param[in] zeta_min_unstable Lower clip for zeta in unstable branch (default -5.0) [-]
 * @return Ch_corrected         Stability-corrected heat exchange coefficient (dimensionless)
 *
 * **Physical constraints enforced:**
 * - If olen ≤ 0 (unstable but L not computed), returns Ch_base unchanged
 * - zeta = zref / olen; clipped to [zeta_min_unstable, zeta_max_stable]
 * - Ch_corrected = Ch_base / Phi_h(zeta_clipped)
 * - Never returns negative or zero Ch
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
compute_ch_stability_correction (amrex::Real Ch_base,
                                  amrex::Real olen,
                                  amrex::Real zref,
                                  amrex::Real zeta_max_stable,
                                  amrex::Real zeta_min_unstable)
{
    // Guard: if olen is not physical (≤ 0, or huge), return baseline
    // Convention: olen > 0 in stable conditions, olen < 0 in unstable
    if (olen <= 0.0) {
        return Ch_base;  // Obukhov length not available; use baseline
    }

    // Compute dimensionless stability parameter
    amrex::Real zeta = zref / olen;

    // Clip zeta to physically reasonable range using ternary operators
    // (compatible with both host and device code)
    if (zeta > 0.0) {
        // Stable: clip to [0, zeta_max_stable]
        zeta = (zeta > zeta_max_stable) ? zeta_max_stable : zeta;
    } else {
        // Unstable: clip to [zeta_min_unstable, 0]
        zeta = (zeta < zeta_min_unstable) ? zeta_min_unstable : zeta;
    }

    // Compute stability correction multiplier: 1 / Phi_h(zeta)
    amrex::Real correction_factor = StabilityFunctions::phi_h_inverse(zeta);

    // Return corrected Ch
    return Ch_base * correction_factor;
}

#endif // ERF_UCM_STABILITY_CORRECTION_H
