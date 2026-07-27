/**
 * @file ERF_UCMStabilityCorrection.cpp
 * @brief Implementation of stability-aware canyon-atmosphere exchange (Phase 3.4)
 *
 * This file provides any non-inline utility functions and documentation for the
 * stability correction system. The core computation functions are header-only
 * in ERF_UCMStabilityCorrection.H for GPU efficiency.
 *
 * **Phase 3.4 Overview:**
 * Upgrades the canyon-atmosphere heat exchange coefficient from a fixed bulk
 * coefficient to one corrected for local atmospheric stability using the Obukhov
 * length L available from SurfaceLayer (populated by MRF/YSU/MYNN2.5 PBL schemes).
 *
 * **Key equations:**
 * - Ch_corrected(i,j) = Ch_base(i,j) * (1 / Phi_h(zeta(i,j)))
 * - zeta(i,j) = zref / L(i,j)
 * - Phi_h follows Businger-Dyer stability functions (same as MRF/YSU)
 *
 * **Integration points:**
 * 1. When `erf.ucm.use_stability_correction = true` in input file
 * 2. Called during facet SEB solution (wall/roof/road heat balance)
 * 3. Receives olen from SurfLayer->get_olen(lev) populated by MRF/YSU
 *
 * **Parameters (read from input via ParmParse "erf.ucm" namespace):**
 * - `erf.ucm.use_stability_correction` (bool, default false): Enable Obukhov correction
 * - `erf.ucm.zeta_max_stable` (real, default 2.0): Clip zeta in stable branch
 * - `erf.ucm.zeta_min_unstable` (real, default -5.0): Clip zeta in unstable branch
 *
 * **References:**
 * - Businger et al. (1971): "Flux-profile relationships in the atmospheric surface layer"
 *   Journal of Atmospheric Sciences, 28(2), 181-189
 * - Dyer, A.J. (1974): "A review of flux-profile relationships"
 *   Boundary-Layer Meteorology, 7(3), 363-372
 * - Paulson, C.A. (1970): "The mathematical representation of wind speed and
 *   temperature profiles in the unstable atmospheric boundary layer"
 *   Journal of Applied Meteorology, 9(6), 857-861
 * - WRF Single-Layer UCM: Chen et al. (2011), module_sf_urban.F
 */

#include <UrbanCanopy/ERF_UCMStabilityCorrection.H>

// Namespace and any utility instantiations can go here if needed in future phases
// Currently, all functionality is header-only for GPU efficiency
