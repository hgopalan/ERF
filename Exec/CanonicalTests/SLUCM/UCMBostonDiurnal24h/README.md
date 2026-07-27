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
4. **Diurnal warming present** -- T_skin_roof max exceeds 305 K during daytime
5. **Diurnal cooling present** -- T_skin_roof min drops below 290 K at night
6. **Slab stays bounded** -- T_slab_roof[0] stays within [280, 305] K throughout
7. **UHI signal maintained** -- T_canyon_air exceeds T_atm by at least 2 K during daytime peak

## Running

```bash
# From this directory, after copying data files above:
../../../Build/erf_slucm inputs_diurnal_24h > run.log 2>&1

# After completion:
python3 check_diurnal_24h.py run.log
```

Exit code 0 = PASS, 1 = FAIL. Test typically takes several hours of wall time.

## Known Limitations

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
7. Canyon LW trapping

Without those fixes, this 24-hour test would fail on metric 1 (clamps within the first hour) or metric 6 (slab drift).
