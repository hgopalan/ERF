# UCMBostonTwoWayHeat — Phase 3.2 Two-Way Heat Coupling Test

## Overview

This canonical test validates that the two-way ATM→UCM data plumbing (T_air + wind) enables UCM heat feedback to modify atmospheric theta (θ) near the surface, producing an Urban Heat Island (UHI) signature in the lowest model level (k=0).

## Key Differences from UCMBoston (Phase 2.11 baseline)

| Setting | UCMBoston (one-way) | UCMBostonTwoWayHeat (two-way) |
|---------|-------------------|------------------------------|
| `erf.ucm.atm_feedback_heat` | 0.0 (OFF) | 1.0 (ON) |
| `erf.ucm.atm_feedback_momentum` | 1.0 (ON) | 1.0 (ON) |
| Expected θ change at k=0 over urban cells | No UHI at surface | +0.01–0.05 K UHI signal |

## Physics

The UCM computes sensible heat fluxes (H_road, H_wall, H_roof) from urban surfaces using locally extracted ATM fields:
- **T_atm** (temperature at lowest ATM level, refined from coarse grid to UCM 2D slab)
- **U_atm, V_atm** (horizontal wind components, refined similarly)

When `atm_feedback_heat = 1.0`, these fluxes are injected into the ATM as source terms to `RhoTheta`, producing a local warming of ~0.01–0.05 K over the downtown urban core within 1 hour of simulation.

## Test Structure

```
UCMBostonTwoWayHeat/
├── inputs_twoway_heat              # Main inputs file (feedback_heat=1.0)
├── materials.csv                    # Material properties (inherited from UCMBoston)
├── building_layout.csv              # Boston building morphology (inherited)
├── inflow_boston.txt                # Inflow sounding (inherited)
├── sounding_boston                  # Initial vertical profile (inherited)
├── check_twoway_heat.py             # Validation script
└── README.md                        # This file
```

## Validation Criteria

The `check_twoway_heat.py` script verifies three key metrics:

| Metric | Threshold | Physical Meaning |
|--------|-----------|------------------|
| UHI signal (θ_urban - θ_rural at k=0) | > 0.01 K | Heat feedback is active |
| Rural contamination (std of non-urban θ) | < 0.005 K | Spurious heating is minimal |
| cc_source[RhoTheta] max | > 0 | Injection is numerically non-zero |

## Running the Test

From the `Exec/CanonicalTests/SLUCM/` directory:

```bash
# Run with the regression harness
./run_all_regressions.sh UCMBostonTwoWayHeat

# Or manually
cd UCMBostonTwoWayHeat
erf_ucm_< ... > 2>&1 | tee log.txt
python3 check_twoway_heat.py
```

## Expected Output

On successful test (PASS):
- UHI signal: +0.03–0.05 K (typical range)
- Rural std: 0.001–0.003 K (minimal contamination)
- Verbose debug output from `[UCM][3.2][*]` instrumentation lines

On failed test (FAIL):
- UHI signal < 0.01 K → heat feedback not working or too weak
- Rural std > 0.005 K → spurious heating of non-urban cells (load imbalance or boundary issue)
- Any NaN detected in fields → numerical stability issue

## Regression Notes

This test is **NOT** bit-identical to UCMBoston (one-way) because heat feedback changes ATM state. The baseline for Phase 3.2 is the first run of this test after merging.

Prior to merging Phase 3.2, confirm that:
1. `./run_all_regressions.sh UCMBoston` still PASS (bit-identical baseline check at `feedback_heat=0`)
2. `./run_all_regressions.sh UCMBostonTwoWayHeat` PASS (new two-way heat PASS threshold)

## Debug Instrumentation

With `erf.ucm.ucm_debug = 1` (default in inputs_twoway_heat), the simulation prints:
- `[UCM][3.2][twoway-heat-injection]` — heat source stats per RK stage
- `[UCM][3.2][SEB-inputs]` — ATM fields consumed by SEB
- `[UCM][3.2][pre-injection-check]` — ATM field sanity checks
- `[UCM][3.2][rk-stage-inject]` — RK-stage source term evolution

These traces confirm end-to-end plumbing of T_air and wind, and validate that cc_source[RhoTheta] receives non-zero injection.

## Known Limitations

- **No radiation coupling** — fluxes use test surface temperature (300 K), not computed from diurnal cycle
- **No moisture feedback** — `atm_feedback_moisture = 0.0` (reserved for Phase 3.3)
- **Homogeneous UCM forcing** — all test_surf_temp and AH are uniform (heterogeneous forcing is Phase 5.1+)
- **Single-level UCM** — anchor_level = 0 only (multi-level is Phase 3.6+)

## References

- **Problem Statement:** Phase 3.2 specification in PR #XXX
- **Development Log:** `Source/UrbanCanopy/UCM_DEVELOPMENT.md` (Phase 3.2 section)
- **Phase 2.11 baseline:** UCMBoston one-way test (PR #226)
- **WRF SLUCM:** Chen et al. (2011), adapted for compressible AMReX dycore
