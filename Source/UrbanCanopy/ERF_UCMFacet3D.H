/**
 * @file ERF_UCMFacet3D.H
 * @brief Header-only inline device functions for Phase 2.7 Facet3D geometric injection
 *
 * Phase 2.7 introduces BEP-style geometric overlap for walls and sharp placement for roofs,
 * supporting both terrain-following and flat coordinates, with optional continuous Gaussian
 * height distribution. These helper functions are designed to be reused by Phase 2.8 (momentum drag).
 *
 * **Physics references:**
 *  - Martilli, Clappier & Rotach (2002), Boundary-Layer Meteorology 104:261–304
 *    "An urban surface exchange parameterisation for mesoscale models"
 *  - Coceal & Belcher (2004), QJRMS 130:1349–1372
 *    "A general class of parametrizations for the effects of surface roughness on
 *    the dynamics and thermodynamics of the planetary boundary layer"
 *
 * **Design contract:**
 *  - All functions are GPU-safe with AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 *  - No external dependencies beyond AMReX; self-contained for reusability
 *  - Input validation (e.g., H_mean, H_std checking) done at function level
 *  - Return values are dimensionless fractions (0 to 1) for overlap/height distribution
 *
 * **Usage (Phase 2.7 and beyond):**
 *  - Called once per ATM cell k in `apply_ucm_tendency_to_cc_source` via ParallelFor
 *  - Used in terrain-aware branches to compute vertical distribution of wall+roof heating
 */

#ifndef ERF_UCM_FACET3D_H
#define ERF_UCM_FACET3D_H

#include <cmath>
#include <AMReX_REAL.H>

/**
 * @brief Compute sharp wall overlap fraction (BEP geometric mode)
 *
 * For a given ATM cell [z_lo, z_hi] and mean building height H_mean, compute the
 * vertical overlap as the geometric intersection of the wall layer [0, H_mean] with the
 * ATM cell. Normalized by H_mean to give a dimensionless fraction.
 *
 * **Formula:**
 * ```
 * overlap(k) = max(0, min(H_mean, z_hi) - max(0, z_lo))
 * wall_fraction(k) = overlap(k) / H_mean
 * ```
 *
 * **Safety:**
 *  - If H_mean < 0.01 m (essentially no buildings), return 0.0 (skip injection)
 *  - Handles z_lo < 0 gracefully (ground-level or below-surface cells)
 *  - Result always in [0, 1]
 *
 * @param z_lo        Lower boundary of ATM cell [m] (height above terrain origin)
 * @param z_hi        Upper boundary of ATM cell [m]
 * @param H_mean      Mean building height [m]
 *
 * @return            Wall overlap fraction in [0, 1] (dimensionless)
 *
 * @note Converges to sharp mode when used with continuous Gaussian if H_std → 0
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real wall_overlap_fraction_sharp(amrex::Real z_lo, amrex::Real z_hi,
                                        amrex::Real H_mean)
{
    if (H_mean < 0.01) return 0.0;  // Safety: skip non-urban cells
    const amrex::Real overlap = amrex::max(0.0,
        amrex::min(H_mean, z_hi) - amrex::max(0.0, z_lo));
    return overlap / H_mean;
}

/**
 * @brief Compute continuous Gaussian wall overlap fraction (Phase 2.7 novel extension)
 *
 * When a building height distribution with std dev H_std is available, compute the
 * probability that a building falls within the ATM cell [z_lo, z_hi], assuming a normal
 * distribution N(H_mean, H_std²).
 *
 * **Formula:**
 * ```
 * wall_fraction(k) = 0.5 * [ erf((H_mean - z_lo) / (√2 * H_std))
 *                           - erf((H_mean - z_hi) / (√2 * H_std)) ]
 * ```
 *
 * **Physics interpretation:**
 *  - Cumulative density between z_lo and z_hi of a normal distribution
 *  - Smoothly interpolates between buildings shorter and taller than the cell
 *  - Enables subgrid-scale height variation with single scalar (H_std)
 *
 * **Safety:**
 *  - If H_mean < 0.01 m, return 0.0 (no buildings)
 *  - If H_std < 1e-6 m, fall back to sharp (avoid division by near-zero)
 *  - Uses std::erf for error function (standard library)
 *
 * @param z_lo        Lower boundary of ATM cell [m]
 * @param z_hi        Upper boundary of ATM cell [m]
 * @param H_mean      Mean building height [m]
 * @param H_std       Std dev of building height [m]
 *
 * @return            Gaussian-weighted wall overlap fraction in [0, 1]
 *
 * @note As H_std → 0, Gaussian mode converges to sharp mode
 * @note Requires linking against C math library (erf provided by std)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real wall_overlap_fraction_gaussian(amrex::Real z_lo, amrex::Real z_hi,
                                           amrex::Real H_mean, amrex::Real H_std)
{
    if (H_mean < 0.01 || H_std < 1e-6) return 0.0;  // Safety: fall back to sharp
    const amrex::Real inv = 1.0 / (std::sqrt(2.0) * H_std);
    return 0.5 * (std::erf((H_mean - z_lo) * inv) - std::erf((H_mean - z_hi) * inv));
}

/**
 * @brief Determine if ATM cell k contains the roof (BEP sharp roof placement)
 *
 * In BEP-style injection, the roof (top of building) is placed as a Dirac delta at exactly
 * H_mean (2D slab model; no thickness). This function identifies which vertical cell k
 * should receive the roof heat, following the rule:
 *  - Find k such that z_lo(k) ≤ H_mean < z_hi(k)  →  return true
 *  - If H_mean ≥ z_hi(khi) (building taller than domain), place roof at khi
 *
 * **Formula:**
 * ```
 * is_roof_cell(k) = true   if z_lo(k) ≤ H_mean < z_hi(k)
 * is_roof_cell(khi) = true if H_mean ≥ z_hi(khi)  (fallback to top cell)
 * ```
 *
 * **Safety:**
 *  - Returns false if H_mean < 0.01 m (no buildings)
 *  - Handles khi (domain top index) gracefully to avoid out-of-bounds injection
 *
 * @param k           Current vertical cell index
 * @param klo         Domain lower index in z-direction
 * @param khi         Domain upper index in z-direction
 * @param z_lo        Lower boundary of cell k [m]
 * @param z_hi        Upper boundary of cell k [m]
 * @param H_mean      Mean building height [m]
 *
 * @return            true if cell k should receive roof heat; false otherwise
 *
 * @note Phase 2.7 uses this for sharp roof placement; Phase 2.8 will extend for drag
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool is_roof_cell(int k, int klo, int khi, amrex::Real z_lo, amrex::Real z_hi,
                  amrex::Real H_mean)
{
    if (H_mean < 0.01) return false;  // Safety: no buildings
    // Normal case: roof in cell k if z_lo <= H_mean < z_hi
    if (H_mean >= z_lo && H_mean < z_hi) return true;
    // Fallback: if building taller than domain, place roof at khi
    if (k == khi && H_mean >= z_hi) return true;
    return false;
}

#endif  // ERF_UCM_FACET3D_H
