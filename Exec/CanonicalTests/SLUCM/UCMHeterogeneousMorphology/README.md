# UCMHeterogeneousMorphology — Phase 2.2 Test Case

## Objective

Test Phase 2.2 per-cell morphology wiring: verify that z0 and displacement height
are computed per-cell from building height using the Phase 2.2 algorithm,
and that heterogeneous wind profiles result from the heterogeneous morphology.

## Setup

- **Domain:** 16×16 UCM grid with 16×16 ATM cells (grid_ratio=1)
- **Morphology:** Two-region domain:
  - Region 1 (i < 8): H_bldg = 5.0 m, mat_id = 1 (concrete, k=1.5)
  - Region 2 (i ≥ 8): H_bldg = 25.0 m, mat_id = 2 (insulated, k=0.3)
- **Phase 2.2 Parameters:** z0_over_H = 0.1, d_over_H = 0.7
- **Expected z0 range:** 0.5 m (5×0.1) to 2.5 m (25×0.1)
- **Expected d_disp range:** 3.5 m (5×0.7) to 17.5 m (25×0.7)
- **Timesteps:** 100 (CFL=0.5)

## Pass Criteria

1. **Exit code 0** on all ranks
2. **[UCM][2.2][BANNER] prints:**
   ```
   H_bldg      min=5 max=25 m
   albedo_roof min=0.2 max=0.2
   k_therm_roof min=0.3 max=1.5 W/m/K
   z0          min=0.5 max=2.5 m
   d_disp      min=3.5 max=17.5 m
   ```
   If any range collapses to a single value, the CSV wiring has regressed.
3. **plt_ucm_00100** plotfile exists (final time step)
4. **ucm_diag.dat** has exactly 100 data rows (one per timestep)

## Description

This test exercises the Phase 2.2 heterogeneous morphology pathway:
- Two different building heights (5 m and 25 m) trigger different z0 and d values
- The z0 and d_disp MultiFabs are populated via `fill_ucm_z0_and_disp`
- Wind extraction uses per-cell z0/d in the log-law interpolation
- The resulting wind profiles differ between the two regions due to different aerodynamic properties

This verifies that the per-cell wiring works end-to-end without introducing hidden regressions to the homogeneous fallback path.
