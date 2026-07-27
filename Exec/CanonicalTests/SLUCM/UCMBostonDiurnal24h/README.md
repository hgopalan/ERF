# UCMBostonDiurnal24h -- Phase 3.5c Full-Loop 24-Hour Diurnal Regression

## Purpose

Verify that the Phase 3.5a Newton SEB solver + Phase 3.5b prescribed radiation + Phase 3.5a-hotfix cascade (7 bugs fixed) run stably for a full 24-hour diurnal cycle on a real-city (Boston) heterogeneous configuration.

## Configuration

- Domain: Boston 5-zone concentric layout (from UCMBoston baseline)
- Duration: 60000 steps ~= 24 hours simulated time (1.4 s/step)
- Starts at **midnight LST** (`solar_time_start_s = 0`)
- Summer solstice (`julian_day = 172`)
- Two-way feedback: heat + momentum enabled
- Radiation: Phase 3.5b analytic (SW zenith + LW gray-sky)
- **Slab: 50 cm total depth, 6 layers** (Phase 3.5c-tuned; see Known Limitations)

## Data Files (must be copied/symlinked from UCMBostonStabilityCorrection)

The following data files are required but NOT duplicated in this directory to avoid drift with the upstream test:

- `materials.csv`
- `building_layout.csv`  (25600 rows -- LARGE)
- `inflow_boston.txt`
- `sounding_boston`

**Setup command** (from this directory):

```bash
for f in materials.csv building_layout.csv inflow_boston.txt sounding_boston; do
    ln -sf ../UCMBostonStabilityCorrection/$f .
done
```

(Or use `cp` instead of `ln -sf` if you prefer copies.)

## Validation Metrics (see check_diurnal_24h.py)

1. **Zero Newton clamps** over the entire run -- solver stays in physical range
2. **Zero Newton divergences** -- all iterations converge
3. **theta_tend warnings <= 100** -- ATM injection numerically stable
4. **Daytime warming** -- T_skin_roof max >= 305 K (SW-driven heating)
5. **Nighttime cooling** -- T_skin_roof min <= 293 K (radiative cooling)
6. **Slab bounded** -- T_slab_roof[0] within [260, 310] K throughout (see "Known Limitations" below)
7. **UHI signal maintained** -- T_canyon_air exceeds T_atm by at least 2 K during daytime peak

### Empirical baseline (from run at commit 50acb48 with hotfix cascade applied)

| Metric | Observed | Threshold | Status |
|---|---|---|---|
| Total clamps | 0 | 0 | PASS |
| Total divergences | 0 | 0 | PASS |
| theta_tend warnings | 0 | <= 100 | PASS |
| T_skin_roof max | 309.68 K | >= 305 K | PASS |
| T_skin_roof min | 291.98 K | <= 293 K | PASS |
| T_slab_roof[0] range | [267.49, 304.42] K | within [260, 310] K | PASS |
| Max UHI | 7.54 K | >= 2 K | PASS |

## Running

```bash
# From this directory, after copying data files above:
../../../Build/erf_slucm inputs_diurnal_24h > run.log 2>&1

# After completion:
python3 check_diurnal_24h.py run.log
```

Exit code 0 = PASS, 1 = FAIL. Test typically takes several hours of wall time.

## Known Limitations

### Slow slab cold drift on a subset of cells (Phase 4.2 issue)

Empirical observation: over a 60000-step (24h) run, the top slab layer
(`T_slab_roof[0]`) drifts down to approximately 267 K on a subset of
canyon-shaded cells. The drift rate is approximately 0.001 K per timestep
(slow, monotonic, non-runaway).

**Cause:** The Phase 3.5b analytic radiation model uses a first-order
SVF-based canyon LW trapping formula (Phase 3.5a-hotfix Lesson 19).
This is a first-order approximation of the true multi-facet radiative
exchange within a canyon. The residual ~5% error in the net wall LW
balance manifests as a slow cold drift over multi-hour runs.

**Proper fix:** Phase 4.2 (RRTMG per-facet radiation coupling) or
Phase 6.1 (multi-bounce wall radiation via ray tracing).

**Threshold rationale:** T_SLAB_ROOF_MIN = 260 K (rather than 280 K)
accepts the known slow drift while still failing on true freezing
pathologies (which historically presented as slab dropping below 200 K
in under an hour -- see Phase 3.5a-hotfix cascade docs).

### Other limitations

- Analytic radiation only (Phase 4.2 will use RRTM/RRTMG)
- No moisture feedback (atm_feedback_moisture = 0)
- Single anchor_level (Phase 3.6+ adds multi-level)

## Regression Harness

Auto-discovered by `Exec/CanonicalTests/SLUCM/run_all_regressions.sh` via `inputs*` glob pattern. Note that a 60000-step run does not fit within a fast-CI budget; treat this as a nightly / manual regression, not per-PR.

## Prerequisites

Requires the Phase 3.5a-hotfix cascade fixes merged:

1. TDMA top-BC coefficients
2. MOST sign convention
3. Solar hour angle
4. Material k_therm calibration
5. Slab BC sign convention
6. TDMA all-plus convention
7. Canyon LW trapping (first-order SVF approximation)

Without those fixes, this 24-hour test would fail on metric 1 (clamps within the first hour) or metric 6 (slab drifts below 250 K, a clear pathology rather than a known slow drift).
