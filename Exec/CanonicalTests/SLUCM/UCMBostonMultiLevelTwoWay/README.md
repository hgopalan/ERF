# UCMBostonMultiLevelTwoWay — Phase 3.10 Multi-Level Two-Way Heat Coupling (Phase 3 Finale)

## Purpose

This is the **Phase 3 finale**: verify that SLUCM runs correctly with:
1. **Multi-level refinement** (`anchor_level = 1`, same as Phase 3.6)
2. **Two-way heat coupling** (`atm_feedback_heat = 1.0`, same as Phase 3.2)

Combined, this tests that the UCM operates on a refined AMR level and feeds back atmospheric heating to modify θ, producing an Urban Heat Island (UHI) signal aloft.

## Configuration

Identical to UCMBostonMultiLevel except for heat feedback:

- Base grid: 20×20×64 (from UCMBoston baseline)
- One refinement level over urban core (5–15 km × 5–15 km, ref_ratio=2)
- UCM anchor_level = 1 (operates on coarse level 1, receives ATM fields from refined level 1)
- **atm_feedback_heat = 1.0** (new; Phase 3.6 had this OFF)
- atm_feedback_momentum = 1.0 (inherited from baseline)
- Duration: 600 steps (~14 minutes simulated) — plumbing test
- Plotfile prefix: `plt_multilevel_twoway_`

## Key Differences from Earlier Phases

| Feature | Phase 3.5c (Single-Level One-Way) | Phase 3.6 (Multi-Level One-Way) | Phase 3.10 (Multi-Level Two-Way) |
|---------|-------------------------------------|----------------------------------|--------------------------------------|
| Grid   | Base only                           | Base + refined level 1            | Base + refined level 1               |
| Heat feedback | OFF (atm_feedback_heat=0.0) | OFF (atm_feedback_heat=0.0) | **ON (atm_feedback_heat=1.0)** |
| Momentum drag | ON | ON | ON |
| UHI signature expected | At k=0, single level | At k=0–1, two levels | **At k=0–1, two levels, with warming** |

## Data Files

All data files are symlinked from `../UCMBoston/`:

```bash
for f in materials.csv building_layout.csv inflow_boston.txt sounding_boston; do
    ln -sf ../UCMBoston/$f .
done
```

These are the same as Phase 3.6 (and inherited from Phase 2.11 baseline).

## Validation Metrics (`check_multilevel_twoway.py`)

The validation script asserts:

1. **All fields finite** on both level 0 and level 1 (no NaN/Inf in θ, u, v)
2. **θ bounded in [280, 320] K** on both levels
3. **UHI signal on level 1 at k=0**: `mean(θ_urban_core) − mean(θ_edge) > 0.01 K`
   - Heat feedback should produce a positive warming over the downtown core
4. **Rural contamination on level 1**: std of θ over non-urban cells at k=0 `< 0.01 K`
   - Spurious heating away from the urban core should be minimal
5. **Wind reduction on level 1 at k=1 > 10%** relative to inflow
   - Momentum drag from tall buildings persists on the refined level

Exit code 0 = PASS (all metrics met), 1 = FAIL.

## Running the Test

From `Exec/CanonicalTests/SLUCM/`:

```bash
# Run with regression harness
./run_all_regressions.sh UCMBostonMultiLevelTwoWay

# Or manually
cd UCMBostonMultiLevelTwoWay
../../../Build/erf_slucm inputs_multilevel_twoway > run.log 2>&1
python3 check_multilevel_twoway.py
```

Typical runtime: ~5–10 minutes on a laptop.

## Expected Results

On PASS:
- All fields finite (no NaN/Inf)
- UHI warming ~0.01–0.05 K at k=0 over urban cells
- Rural region std < 0.01 K (minimal contamination)
- Wind reduction ~15–25% at k=1 (momentum drag intact)
- Verbose debug output from `[UCM][3.10][*]` instrumentation lines in run.log

On FAIL:
- UHI signal < 0.01 K → heat feedback not working or too weak
- Rural std > 0.01 K → spurious heating away from urban core
- Wind reduction < 10% → momentum drag degraded (suggests coupling issue)
- Any NaN detected → numerical stability problem

## Prerequisites

All earlier Phase 3 work must merge first:
- **Phase 3.5c** (single-level one-way baseline) — physics end-to-end validated
- **Phase 3.6** (multi-level one-way plumbing) — refinement level handling verified
- **Phase 3.2** (single-level two-way heat) — heat injection plumbing verified

This test **combines** the two-way heat injection of 3.2 with the multi-level plumbing of 3.6.

## Regression Notes

This test is **NOT** bit-identical to UCMBostonMultiLevel (Phase 3.6) because heat feedback modifies ATM state. The baseline for Phase 3.10 is the first run after merging.

Expected first-run metrics from problem statement:
- UHI > 0.01 K (achieved with 1 hour of simulation)
- Rural contamination std < 0.01 K (diffusion limited on refined level)
- Wind reduction > 10% (momentum drag persists)

## Design Contracts (Preserved from Phase 3 Work)

All nine contracts from Phase 3 remain in effect:
1. No hardcoded `int lev = 0` — level-aware indexing throughout
2. No PBLH dependency — MRF PBL operates regardless of urban presence
3. is_urban mask exclusivity — built in, not assumed
4. Terrain-following coordinates — flat domain, no terrain effects
5. MPI safety — domain decomposition tested elsewhere
6. No new `.cpp` files in `Source/` — all changes in UCM/ and refinement
7. RK-stage safety for `cc_source[RhoTheta]` — injection synchronized with RK stages
8. feedback_heat/momentum separation — decoupled switches (heat ON, momentum ON)
9. Convention B aggregation — 2D UCM aggregates to 3D ATM cell volumes

## References

- **Problem Statement:** Phase 3.10 specification (this PR)
- **Development Log:** `Source/UrbanCanopy/UCM_DEVELOPMENT.md` (Phase 3.10 section)
- **Phase 3.6 Multi-Level:** UCMBostonMultiLevel (PR #XXX)
- **Phase 3.2 Two-Way Heat:** UCMBostonTwoWayHeat (PR #XXX)
- **Phase 3.5c Single-Level One-Way:** UCMBoston one-way baseline
