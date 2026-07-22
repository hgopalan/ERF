# UCMHeterogeneousMaterials — Phase 2.2 Test Case

## Objective

Test Phase 2.2 per-cell material property wiring: verify that albedo and thermal properties
are read from the material registry on a per-cell basis, creating a spatial checkerboard pattern
of material properties while morphology remains uniform.

## Setup

- **Domain:** 16×16 UCM grid with 16×16 ATM cells (grid_ratio=1)
- **Morphology:** Uniform across entire domain: H_bldg = 15.0 m everywhere
- **Materials:** Checkerboard pattern on roofs (walls/roads also checkerboard for consistency):
  - Cells where (i+j) % 2 == 0: mat_id = 1 (cool roof: albedo=0.6, k=0.2 W/m/K)
  - Cells where (i+j) % 2 == 1: mat_id = 2 (dark roof: albedo=0.1, k=2.0 W/m/K)
- **Phase 2.2 Parameters:** z0_over_H = 0.1, d_over_H = 0.7 (computed as 1.5m and 10.5m respectively)
- **Timesteps:** 100 (CFL=0.5)

## Pass Criteria

1. **Exit code 0** on all ranks
2. **[UCM][2.2][BANNER] prints:**
   ```
   H_bldg      min=15 max=15 m
   albedo_roof min=0.1 max=0.6
   k_therm_roof min=0.2 max=2.0 W/m/K
   z0          min=1.5 max=1.5 m
   d_disp      min=10.5 max=10.5 m
   ```
   If albedo_roof collapses to a single value or z0/d_disp show unexpected variation, the material or morphology wiring has regressed.
3. **plt_ucm_00100** plotfile exists (final time step)
4. **ucm_diag.dat** has exactly 100 data rows (one per timestep)
5. **Bonus:** In final plt_ucm_00100, T_skin_roof field displays a visible checkerboard pattern with cool-roof cells cooler than dark-roof cells (visible in paraview)

## Description

This test exercises the Phase 2.2 heterogeneous material pathway:
- Morphology is uniform (all cells 15m) to isolate material effects
- Material properties (albedo, k_therm, etc.) are read per-cell from registry
- The resulting energy balance and skin temperature differ between the two materials
- Checkerboard pattern should create a visually distinctive T_skin_roof field if SEB is energy-conserving

This verifies that the material database lookup and per-cell wiring work end-to-end,
complementing the morphology test which focused on z0/d heterogeneity.
