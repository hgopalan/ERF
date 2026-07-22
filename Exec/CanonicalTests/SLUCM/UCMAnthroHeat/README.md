# UCMAnthroHeat Canonical Test

## Purpose

Canonical test for **ERF-SLUCM Phase 2.3**: Anthropogenic heat injection via rooftop.

Tests that:
1. Anthropogenic heat (AH) parameter `AH_uniform_Wm2` is correctly applied to all urban roof facets
2. AH is applied uniformly to all cells (profile_id=0)
3. AH adds to roof flux: `H_roof_final = H_roof_sensible + AH`
4. Measurable atmospheric warming via extra heat injection into the boundary layer
5. Diurnal profile is *disabled* for this test (profile_id=0 → uniform)

## Configuration

**Inputs:** `inputs` (Phase 2.3 with AH enabled)
- ATM domain: 16×16×64
- UCM domain: 16×16×1 (grid_ratio=1, same as ATM)
- Total timesteps: 100 (short run)
- `atm_feedback=1.0` (bidirectional coupling enabled)
- **`AH_uniform_Wm2=30.0`** ← ACTIVE ANTHROPOGENIC HEAT
- `AH_profile_type_default=0` (uniform profile in time)
- `plan_area_frac_uniform=0.5` (building coverage)

**CSV files:** Inherited from `UCMHeterogeneousMaterials` with modification
- `building_hetero_mat.csv`: 256 rows (16×16 UCM cells)
- Every urban cell has `ah_profile_id=0` (uses uniform `AH_uniform_Wm2`)
- Heterogeneous height: 5m and 25m buildings
- Two material types

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
  H_road min=-50.0 max=50.0 W/m^2        (similar to UCMFacetSplit)
  H_wall min=-60.0 max=60.0 W/m^2        (similar to UCMFacetSplit)
  H_roof min=-40.0 max=70.0 W/m^2        (HIGHER than UCMFacetSplit due to +30 W/m^2)
  AH min=30.0 max=30.0 W/m^2             (ACTIVATED, uniform)
```

**Interpretation:** AH is uniformly 30 W/m², roof flux is boosted accordingly.

### 3. Roof Flux Boost
Compare `ucm_diag.dat` last row vs `UCMFacetSplit`:
```
H_roof_max(AnthroHeat) - H_roof_max(FacetSplit) ≈ 30.0 W/m²
```
(Within round-off; the extra AH should appear in roof flux, not road/wall.)

**Check script:**
```python
import pandas as pd
df_anthro = pd.read_csv('ucm_diag.dat')  # UCMAnthroHeat output
df_facet  = pd.read_csv('../UCMFacetSplit/ucm_diag.dat')  # UCMFacetSplit output
last_anthro = df_anthro.iloc[-1]
last_facet = df_facet.iloc[-1]
roof_diff = last_anthro['H_roof_max'] - last_facet['H_roof_max']
print(f"H_roof boost: {roof_diff:.2f} W/m² (expected ~30)")
assert 29.0 <= roof_diff <= 31.0, f"Roof boost out of range: {roof_diff}"
```

### 4. Atmospheric Warming
Compare final ATM plotfile vs `UCMFacetSplit`:
```
RhoTheta_max(AnthroHeat) > RhoTheta_max(FacetSplit)
```
**Expected:** Final potential temperature in boundary layer is measurably higher (~0.1–0.5 K increase depending on domain size and coupling strength).

**Check script:**
```bash
# Both require VisIt/AMReX analysis tools or custom postprocessing
cd AnthroHeat_build && yt analyze plt00002/RhoTheta > rhotheta_anthro.txt
cd ../FacetSplit_build && yt analyze plt00002/RhoTheta > rhotheta_facet.txt
# Compare max values
```

### 5. Debug Output
When `ucm_debug=true`, verify Phase 2.3 messages in stdout:
```
[UCM][2.3][compute_anthropogenic_heat] time=0.0s AH min=30.0 max=30.0 W/m^2 diurnal_factor=0.0
[UCM][2.3][ATM_COUPLING] injection uses lumped H_sensible = H_road + H_wall + H_roof + AH. Facet3D is Phase 2.7.
```

## Known Issues

None identified in Phase 2.3.

## Related Tests

- **UCMFacetSplit** (Phase 2.3): Baseline without AH; `AH_uniform_Wm2=0.0`. Use for comparison.
- **UCMHeterogeneousMaterials** (Phase 2.2): Phase 2.2 baseline; no facet split, no AH.

## References

- `UCM_DEVELOPMENT.md` — Phase 2.3 specification
- `Source/UrbanCanopy/ERF_UCMLayer.cpp` — Facet-split SEB kernel with AH injection
- `Source/UrbanCanopy/ERF_UCMAllocate.cpp` — compute_anthropogenic_heat helper

## Notes for Future Work

**Phase 6.2** will move AH to a building-energy model (BEM) coupled solver, replacing the simple uniform/diurnal profiles. Placeholder for future BEM profiling.
