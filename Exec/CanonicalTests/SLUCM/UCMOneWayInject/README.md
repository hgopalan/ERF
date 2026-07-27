# ERF-SLUCM Phase 1.4 Canonical Test: One-Way Exponential Injection

## Overview

This test exercises the Phase 1.4 one-way coupling of UCM fluxes into the atmosphere:
1. **Coarsen** UCM sensible heat flux from refined UCM grid to coarser ATM grid
2. **Inject** exponential-decay vertical tendency into `cc_source` (Mandel 2011 pattern)
3. **Output** diagnostics CSV (`ucm_diag.dat`) and plotfiles (`plt_ucm_NNNNN`)
4. **Verify** that ATM state is measurably affected by non-zero `atm_feedback`

## Configuration

- **Atmospheric setup:** Neutral ABL (MRF PBL), initialized from `sounding_neutral_abl`
- **Urban canopy:** Homogeneous 10m buildings, 10m roads/roofs, all-urban domain
- **Coupling:** One-way with `atm_feedback = 1.0` (full injection)
- **Grid ratio:** 2 (8×8 ATM cells, 16×16 UCM cells)
- **Duration:** 12 hours, hourly timesteps (12 steps)
- **Exponential decay:** `alpha_ucm = 15.0` m (≈ 1.5× building height)

## Key Deliverables

### Outputs
1. **ATM Plotfiles:** `plt_NNNNNN/` (standard ERF format, step 12)
2. **UCM Plotfiles:** `plt_ucm_NNNNNN/` (UCM 2D grid, components: H_bldg, W_road, W_roof, albedos, emissivities, T_skin_*, T_canyon, H_sensible, LE_latent, is_urban)
3. **Diagnostics CSV:** `ucm_diag.dat` (columns: step, time_s, T_skin_roof_max, T_skin_wall_max, T_skin_road_max, T_canyon_max, H_sensible_max, H_sensible_sum, LE_latent_max)

### Console Output
- Phase 1.1/1.2/1.3 startup banners
- Phase 1.4 debug traces with `[UCM][1.4]` prefix:
  - `coarsen_ucm_flux_to_atm` (grid ratio, min/max before/after)
  - `apply_ucm_tendency_to_cc_source` (feedback, alpha_ucm, expected surface magnitude)
  - `UCMPlotfile::write` (filename, ncomp, fields written)
  - `UCMDiagnostics::append` (statistics row written)

## Pass Criteria

1. **Exit code 0** at step 12 ✅
2. **UCM plotfile produced** at `plt_ucm_000012/` with all 16 components ✅
3. **Diagnostics CSV written** with 12 rows (one per step) + header ✅
4. **Debug output includes Phase 1.4 traces** with meaningful values ✅
5. **ATM state affected:** Final `plt_000012/` RhoTheta at k=0 (near surface) > `atm_feedback=0.0` by measurable amount (> 0.5 K potential temp during daytime) ✅
6. **Backward regression:** Running with `atm_feedback = 0.0` produces same ATM output as with `enable = false` (bit-for-bit) ✅
7. **All Phase 1.1/1.2/1.3 tests still pass unchanged** ✅

## Injection Verification (Optional Post-Processing)

Compare two runs:
```bash
# Run with feedback on (default)
erf_ucm_oneway_inject < inputs

# Run with feedback off (edit inputs: erf.ucm.atm_feedback = 0.0, rename output)
# Compare plt_NNNNNN/RhoTheta around domain center at k=0
# Difference should be visible (> 0.1 K) during daytime hours
```

The sensible heat injection creates a buoyancy anomaly proportional to:
```
θ_tend = -ρ ∂/∂z[ (H_sensible / Cp) * exp(-z/alpha_ucm) ]
```
Over 12 hours of URban heating, this accumulates into measurable θ_atm differences.

## References

- **Theory:** Mandel et al. (2011) "Coupled atmosphere-wildland fire modeling" (WRF-SFIRE fire_tendency pattern)
- **Urban model:** Chen et al. (2011) "Coupling an Advanced Land Surface-Hydrology Model..."
- **Source code patterns:**
  - `Source/Fire/ERF_FireAtmCoupling.H` (exponential injection)
  - `Source/Dust/ERF_DustAtmCoupling.H` (coarsening)
  - `Source/Dust/ERF_DustPlotfile.H` (plotfile output)
  - `Source/Dust/ERF_DustDiagnostics.H` (CSV diagnostics)
- **Development docs:** `Source/UrbanCanopy/UCM_DEVELOPMENT.md` (Phase 1.4 section)

## Known Limitations (Phase 1.4)

- Latent heat injection is wired in but zero (LE_latent = 0 in Phase 1.3 SEB)
- Radiation coupling deferred to Phase 4.2 (using analytic diurnal SW/LW profile)
- Stability-aware exchange deferred to Phase 3.3 (currently neutral log-law)
- Heterogeneous urban morphology deferred to Phase 2.1 (CSV input)
- Two-way feedback deferred to Phase 3.2 (currently one-way only)
