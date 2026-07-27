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

### Empirical baseline (60000-step run with slab_L=0.5, slab_N_layers=6)

| Metric | Observed | Threshold | Status |
|---|---|---|---|
| Total clamps | 0 | 0 | PASS |
| Total divergences | 0 | 0 | PASS |
| theta_tend warnings | 0 | <= 100 | PASS |
| T_skin_roof max | 310.39 K | >= 305 K | PASS |
| T_skin_roof min | 291.98 K | <= 293 K | PASS |
| T_slab_roof[0] range | [269.36, 304.74] K | within [260, 310] K | PASS |
| Max UHI | 7.56 K | >= 2 K | PASS |
| Wall time | 890 s | -- | 96x real-time |

### Comparison: shallow vs. deep slab (empirical)

| Metric | slab_L=0.3, N=4 (default) | slab_L=0.5, N=6 (this test) |
|---|---|---|
| T_skin_roof max | 309.68 K | 310.39 K |
| T_slab_roof[0] min | 267.49 K | 269.36 K |
| Max UHI | 7.54 K | 7.56 K |
| Wall time | 898 s | 890 s |

Deeper slab yields ~2 K less cold drift on shaded cells at no computational cost.

## Running

```bash
# From this directory, after copying data files above:
../../../Build/erf_slucm inputs_diurnal_24h > run.log 2>&1

# After completion:
python3 check_diurnal_24h.py run.log
```

Exit code 0 = PASS, 1 = FAIL. Test typically takes ~15 min wall time.

## Known Limitations

### Slow slab cold drift on shaded cells (Phase 4.2 issue)

Even with the deeper slab (50 cm, 6 layers), the top slab layer on shaded
wall/road cells drifts down to ~269 K over a full 24h run. The drift rate
is approximately 0.001 K per timestep (slow, monotonic, non-runaway).

**Cause:** The Phase 3.5b analytic radiation model uses a first-order
SVF-based canyon LW trapping formula (Phase 3.5a-hotfix Lesson 19).
This is a first-order approximation of the true multi-facet radiative
exchange within a canyon. The residual error in the net wall LW
balance manifests as a slow cold drift.

**Proper fix:** Phase 4.2 (RRTMG per-facet radiation coupling) or
Phase 6.1 (multi-bounce wall radiation via ray tracing).

**Threshold rationale:** T_SLAB_ROOF_MIN = 260 K accepts the known
slow drift while still failing on true freezing pathologies (which
historically presented as slab dropping below 200 K in under an hour --
see Phase 3.5a-hotfix cascade docs).

### Other limitations

- Analytic radiation only (Phase 4.2 will use RRTM/RRTMG)
- No moisture feedback (atm_feedback_moisture = 0)
- Single anchor_level (Phase 3.6+ adds multi-level)
- Deep BC still influenced by residual diurnal wave (~5% amplitude at 50 cm)

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
