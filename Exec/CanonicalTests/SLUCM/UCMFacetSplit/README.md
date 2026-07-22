# UCMFacetSplit Canonical Test

## Purpose

Canonical test for **ERF-SLUCM Phase 2.3**: Facet-split sensible heat flux decomposition.

Tests that:
1. Single `H_sensible` is split into three facet fluxes: `H_road`, `H_wall`, `H_roof`
2. Each facet flux is computed using area fractions and weighted by MOST stress velocity
3. Anthropogenic heat (AH) is initialized but zero for this baseline test
4. Sum of facet fluxes `H_road + H_wall + H_roof + AH` equals lumped `H_sensible` (diagnostic invariant)
5. Bit-for-bit atmospheric regression vs Phase 2.2 (injection kernel unchanged)
6. Phase 2.3 BANNER includes extended diagnostics for new fields

## Configuration

**Inputs:** `inputs` (Phase 2.3)
- ATM domain: 16×16×64
- UCM domain: 16×16×1 (grid_ratio=1, same as ATM)
- Total timesteps: 100 (short run)
- `atm_feedback=1.0` (bidirectional coupling enabled)
- `AH_uniform_Wm2=0.0` (no anthropogenic heat)
- `plan_area_frac_uniform=0.5` (building coverage)

**CSV files:** Inherited from `UCMHeterogeneousMaterials`
- `building_hetero_mat.csv`: 256 rows (16×16 UCM cells)
- Heterogeneous height: 5m and 25m buildings
- Two material types (roof_mat_id, wall_mat_id, road_mat_id ∈ {1,2})
- `ah_profile_id=0` for all urban cells (uses uniform AH)

## Pass Criteria

### 1. Exit Code
```
✓ Exit code 0 (normal completion, no error/abort)
```

### 2. BANNER Output
Verify in stdout during first `UCMLayer::advance`:
```
[UCM][2.3][BANNER]
  plan_area_frac min=0.5 max=0.5
  H_road min=-50.0 max=50.0 W/m^2        (example range)
  H_wall min=-60.0 max=60.0 W/m^2        (example range)
  H_roof min=-40.0 max=40.0 W/m^2        (example range)
  AH min=0.0 max=0.0 W/m^2
```

**Interpretation:** Facet flux ranges are distinct and non-zero (split is working), AH is uniformly zero.

### 3. Flux Sum Invariant
In `ucm_diag.dat` (last row after 100 steps):
```
H_road_max + H_wall_max + H_roof_max ≈ H_sensible_max
```
(Within round-off error; both are domain maxima computed separately.)

**Check script:**
```python
import pandas as pd
df = pd.read_csv('ucm_diag.dat')
last_row = df.iloc[-1]
computed_sum = last_row['H_road_max'] + last_row['H_wall_max'] + last_row['H_roof_max']
sensible_max = last_row['H_sensible_max']
relative_error = abs(computed_sum - sensible_max) / abs(sensible_max)
print(f"Facet sum: {computed_sum:.2f}, H_sensible: {sensible_max:.2f}, RelErr: {relative_error:.2e}")
assert relative_error < 1e-5, "Sum invariant violated!"
```

### 4. ATM Regression (Optional but Strongly Recommended)
Compare final plotfiles vs Phase 2.2 `UCMHeterogeneousMaterials`:
```
diff plt00002/Rho       vs phase2.2/Rho       → 0 differences
diff plt00002/RhoTheta  vs phase2.2/RhoTheta  → 0 differences (bit-for-bit)
diff plt00002/U         vs phase2.2/U         → 0 differences
diff plt00002/V         vs phase2.2/V         → 0 differences
diff plt00002/W         vs phase2.2/W         → 0 differences
```
**Rationale:** Since injection wiring is unchanged and sum equals lumped flux, ATM solution is identical.

### 5. Debug Output
When `ucm_debug=true`, verify Phase 2.3 messages appear in stdout:
```
[UCM][2.3][compute_anthropogenic_heat] time=0.0s AH min=0.0 max=0.0 W/m^2 diurnal_factor=0.0
[UCM][2.3][ATM_COUPLING] injection uses lumped H_sensible = H_road + H_wall + H_roof + AH. Facet3D is Phase 2.7.
```

## Known Issues

None identified in Phase 2.3.

## Related Tests

- **UCMHeterogeneousMaterials** (Phase 2.2): Baseline without facet split; **UCMFacetSplit** should be identical in ATM output.
- **UCMAnthroHeat** (Phase 2.3): Same setup but with AH≠0; should show higher roof flux and ATM temperature.

## References

- `UCM_DEVELOPMENT.md` — Phase 2.3 specification
- `Source/UrbanCanopy/ERF_UCMLayer.cpp` — Facet-split SEB kernel
- `Source/UrbanCanopy/ERF_UCMAllocate.cpp` — compute_anthropogenic_heat helper
