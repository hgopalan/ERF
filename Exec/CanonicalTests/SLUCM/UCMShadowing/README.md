# UCMShadowing Canonical Test

## Purpose

Canonical test for **ERF-SLUCM Phase 2.4**: Canyon shadowing with sky-view-factor (SVF) computation.

Tests that:
1. Per-cell SVF (sky-view-factor) is computed from canyon aspect ratio (Kusaka 2001)
2. SVF_wall and SVF_road satisfy Kusaka equations 24-25
3. SVF_roof = 1.0 always (unshaded from above)
4. 0 ≤ SVF_wall, SVF_road ≤ 1 (physical bounds maintained)
5. SVF values vary spatially with heterogeneous H_bldg and W_road
6. Plotfile correctly includes SVF_wall, SVF_road, SVF_roof fields
7. Bit-for-bit atmospheric regression vs Phase 2.3 (no injection changes)

## Configuration

**Inputs:** `inputs` (Phase 2.4)
- ATM domain: 8×8×32 (small for quick testing)
- UCM domain: 8×8×1 (grid_ratio=1, same as ATM)
- Total timesteps: 1 (verification run only)
- `atm_feedback=1.0` (bidirectional coupling enabled)
- `plan_area_frac_uniform=0.5` (building coverage)

**CSV files:** Heterogeneous canyon geometry
- `building_layout.csv`: 64 rows (8×8 UCM cells)
- Heterogeneous heights: alternating 10m and 15m buildings
- Constant road width: 10m (tests SVF variation with aspect ratio)
- Same material for all cells (mat_id=1) for simplicity

**Supporting files:**
- `sounding_neutral_abl` — neutral ABL atmospheric sounding
- `materials.csv` — material properties (single material for this test)

## Physics (Kusaka 2001 canyon shadowing)

For each urban cell (i, j):
- **Canyon aspect ratio:** `aspect = H_bldg / max(W_road, 1.0e-6)`
- **Road SVF (Kusaka eq. 24):** `SVF_road = sqrt(aspect^2 + 1) - aspect`
- **Wall SVF (Kusaka eq. 25):** `SVF_wall = 0.5 * (aspect + 1 - sqrt(aspect^2 + 1)) / aspect`
- **Roof SVF:** `SVF_roof = 1.0` (always)

Bounds check: For valid canyons (0 < aspect < ∞):
- `0 < SVF_road < 1` (road sees reduced sky)
- `0 < SVF_wall < 1` (wall sees reduced sky)
- `SVF_roof = 1` (roof always sees full sky)

## Pass Criteria

### 1. Exit Code
```
✓ Exit code 0 (normal completion, no error/abort)
```

### 2. Compilation
Verify:
- ERF_UCMShadowing.H compiles without errors
- compute_sky_view_factors() function callable
- ERF_UCMFields includes SVF_wall, SVF_road, SVF_roof

### 3. SVF Initialization (from plotfile)
In `plt_ucm_000001/` check:
- `SVF_wall` component exists (index 16)
- `SVF_road` component exists (index 17)
- `SVF_roof` component exists (index 18)

Values should vary spatially (8×8 cells with 2 building heights):
- Cells with H=10m, W=10: `aspect=1.0`
  - Expected SVF_road ≈ sqrt(2) - 1 ≈ 0.414
  - Expected SVF_wall ≈ 0.5 * (1 + 1 - sqrt(2)) / 1 ≈ 0.293
- Cells with H=15m, W=10: `aspect=1.5`
  - Expected SVF_road ≈ sqrt(3.25) - 1.5 ≈ 0.303
  - Expected SVF_wall ≈ 0.5 * (1.5 + 1 - sqrt(3.25)) / 1.5 ≈ 0.405

### 4. Debug Output (ucm_debug=true)
Verify in stdout:
```
[UCM][2.4][compute_sky_view_factors] SVF computed on level 0:
  SVF_wall range: [0.29, 0.41]
  SVF_road range: [0.30, 0.41]
```
(Exact ranges depend on spatial heterogeneity)

### 5. Physical Bounds Check
Verify all plotfile values satisfy:
```
0 ≤ SVF_wall ≤ 1
0 ≤ SVF_road ≤ 1
SVF_roof = 1.0 (everywhere)
```

### 6. Regression vs Phase 2.3
Compare ATM final state (should be identical):
```
diff plt00001/Rho       vs phase2.3/Rho       → 0 differences
diff plt00001/RhoTheta  vs phase2.3/RhoTheta  → 0 differences
diff plt00001/U         vs phase2.3/U         → 0 differences
```
Rationale: SVF computation is pre-SEB (not yet used in SW absorption); ATM forcing unchanged.

### 7. Check Script
```python
import amrex
import numpy as np

# Load SVF fields from plotfile
data = amrex.PlotFile('plt_ucm_000001')
svf_wall = data.get('SVF_wall')
svf_road = data.get('SVF_road')
svf_roof = data.get('SVF_roof')

# Check bounds
assert np.all((svf_wall >= 0) & (svf_wall <= 1)), "SVF_wall out of bounds"
assert np.all((svf_road >= 0) & (svf_road <= 1)), "SVF_road out of bounds"
assert np.allclose(svf_roof, 1.0), "SVF_roof != 1.0"

# Check variation (heterogeneous input)
assert svf_wall.std() > 0.01, "SVF_wall not varying (should be heterogeneous)"
assert svf_road.std() > 0.01, "SVF_road not varying (should be heterogeneous)"

print("✓ All SVF tests passed!")
print(f"  SVF_wall: min={svf_wall.min():.4f}, max={svf_wall.max():.4f}")
print(f"  SVF_road: min={svf_road.min():.4f}, max={svf_road.max():.4f}")
print(f"  SVF_roof: all={svf_roof[0, 0]:.4f}")
```

## Known Issues

None identified in Phase 2.4.

## Related Tests

- **UCMFacetSplit** (Phase 2.3): Baseline facet-split heat; UCMShadowing should be identical in ATM output (SVF not yet affects physics).
- **UCMAnthroHeat** (Phase 2.3): Anthropogenic heat test; can be combined with UCMShadowing in Phase 2.5 when SW absorption uses SVF.

## Future Work (Phase 2.5+)

- **SW absorption kernel:** Modify SEB to use SVF: `SW_abs_road = (1 - albedo) * SW_down * SVF_road`
- **Plotfile diagnostics:** Add `shadow_frac` = 1 - SVF to visualize shadowing effect
- **Facet3D integration:** Phase 2.7 will use per-facet SVF from ray-tracing; compare with Kusaka model

## References

- Kusaka et al. (2001): "A Simple Single-Layer Urban Canopy Model for Atmospheric Models: Comparison with Multi-Layer and Slab Models." *Boundary-Layer Meteorology*, 101:329-358.
  - Equations 24-25: Canyon SVF formulas
- `Source/UrbanCanopy/ERF_UCMShadowing.H` — SVF computation implementation
- `Source/UrbanCanopy/ERF_UCMLayer.cpp` — Integration into advance()
- `UCM_DEVELOPMENT.md` — Phase 2.4 specification
