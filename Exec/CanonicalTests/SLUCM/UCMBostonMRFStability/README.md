# UCM Boston MRF Stability Regression Test (Phase 3.3)

## Purpose

Phase 3.3 audit to verify that MRF + UCM two-way heat coupling remains **stable** over extended simulations (7200 time steps = 1 hour). This test is part of the MRF re-audit + PBLH consumer guard initiative before Phase 3.4 (stability-aware canyon exchange).

## Key Findings

### 1. Static PBLH Audit
- **Result:** ✅ PASS — Zero calls to `SurfaceLayer::get_pblh()` in UCM code
- **Design Contract #4:** UCM consumes only `u_star`, `t_star`, `q_star` (no PBLH dependency)

### 2. MRF cc_src Ownership Conflict
- **Finding:** MRF counter-gradient heat flux is handled through implicit diffusion operator, not written directly into `cc_src[RhoTheta_comp]`
- **UCM Ownership:** UCM owns `cc_src[RhoTheta_comp]` and zeros it at entry, then `+=` only
- **Status:** ✅ PASS — No ownership conflict detected

### 3. Two-Way Heat Stability
- **Simulation Length:** 7200 steps (1 hour)
- **Initial State:** Boston urban layout with diurnal cycle (SW analytical)
- **Validation Metrics:**
  1. ✅ Theta bounded [294–310] K at k=10 (no blow-up)
  2. ✅ UHI signal maintained at k=10 (ΔT > 0.02 K, center − edge)
  3. ✅ Wind reduction still > 10% (drag not broken by heat feedback)
  4. ✅ All fields finite (no NaN/Inf)

## Inputs Specification

**`inputs_mrf_stability`**

Based on `UCMBostonTwoWayHeat/inputs_twoway_heat` with modifications:

```
max_step = 7200                    (extended from 3600 for 1-hour stability run)
erf.ucm.atm_feedback_heat = 1.0    (same as two-way heat)
erf.ucm.atm_feedback_momentum = 1.0 (drag active)
erf.ucm.ucm_debug = 1               (extra diagnostics for MRF conflict check)
```

All other parameters inherited from `UCMBostonTwoWayHeat`:
- Boston morphology (building_layout.csv, materials.csv)
- MRF PBL with counter-gradient enabled
- Inflow sounding (sounding_boston, inflow_boston.txt)
- Analytical radiation (SW diurnal, LW constant)

## Validation Script

**`check_mrf_stability.py`**

Automated Python validation that:
1. Loads the final plotfile (`plt_NNNNN`)
2. Extracts 3D fields: `theta`, `x_velocity`, `y_velocity`
3. Checks 5 metrics:
   - **Theta bounded:** min ≥ 294 K, max ≤ 310 K at k=10
   - **UHI signal:** ΔT(center − edge) > 0.02 K at k=10 (~210 m AGL)
   - **Wind reduction:** > 10% at k=1 (~30 m AGL)
   - **Finite values:** no NaN/Inf in any field
   - **MRF conflict:** greps log for `[UCM][3.3][mrf-conflict-check]` and reports max

**Usage:**
```bash
cd UCMBostonMRFStability
./check_mrf_stability.py
```

Exit code: 0 on PASS (all 4 metrics), 1 on FAIL.

## Files

| File | Type | Purpose |
|------|------|---------|
| `inputs_mrf_stability` | Inputs | ParmParse configuration for 7200-step MRF + UCM run |
| `check_mrf_stability.py` | Script | Validation and regression checking |
| `building_layout.csv` | Symlink | Urban morphology (from UCMBostonTwoWayHeat) |
| `materials.csv` | Symlink | Material properties (from UCMBostonTwoWayHeat) |
| `inflow_boston.txt` | Symlink | Inflow sounding (from UCMBostonTwoWayHeat) |
| `sounding_boston` | Symlink | Initial condition sounding (from UCMBostonTwoWayHeat) |
| `README.md` | Doc | This file |

## Known Limitations

1. **Analytical Radiation:** SW and LW are not coupled to radiation solver
   - SW: diurnal cycle with peak ~800 W/m²
   - LW: constant 350 W/m²
   - Phase 4.2 will integrate real radiation solver

2. **No Moisture Feedback:** `atm_feedback_moisture = 0.0`
   - Reserved for Phase 3.3+ enhancements
   - LE fluxes computed but not fed back to atmosphere

3. **Single-Level URB:** `anchor_level = 0` only
   - Multi-level UCM (anchor_level > 0) deferred to Phase 3.7

4. **Homogeneous Forcing:**
   - Anthropogenic heat: `AH_uniform_Wm2 = 30.0` (uniform over domain)
   - Surface temperature: `test_surf_temp_K = 300.0` (uniform)
   - Heterogeneous forcing planned for Phase 5.1+

## References

- **Phase 3.3 Problem Statement:** MRF Re-audit + PBLH Consumer Guard
- **Phase 3.2 Predecessor:** `UCMBostonTwoWayHeat` (two-way heat coupling validation)
- **Phase 3.4 Successor:** Stability-aware canyon exchange
- **Design Contracts:** See `Source/UrbanCanopy/ERF_UCM.H`

## Regression Harness Integration

This test is automatically discovered by `Exec/CanonicalTests/SLUCM/run_all_regressions.sh` via the `inputs*` pattern matching. No manual registration needed.

**Expected Baseline:**
- Bit-identical to baseline once stable after 7200 steps
- All metrics from `check_mrf_stability.py` should PASS consistently

## Contact / Questions

See `Source/UrbanCanopy/UCM_DEVELOPMENT.md` for implementation notes and phase roadmap.
