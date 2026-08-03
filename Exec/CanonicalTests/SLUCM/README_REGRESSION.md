# SLUCM Canonical Regression Harness

## Purpose

Runs every SLUCM canonical test, invokes each canonical's Python check
script, and reports a pass/fail summary. Intended for pre-merge validation
when merging `ERF-SLUCM` back into `development`.

## Quick start

Build the SLUCM executables first (any one of the canonicals will build them):

    cd Exec/CanonicalTests/SLUCM/UCMBoston
    make -j                          # or cmake build

Then run the full suite from the SLUCM directory:

    cd Exec/CanonicalTests/SLUCM
    ./run_all_regressions.sh

## Subsetting

Run one or a few canonicals:

    ./run_all_regressions.sh UCMBoston
    ./run_all_regressions.sh UCMBoston UCMSalamancaMadrid

## Fast smoke test

Override `max_step` for quick iteration during development:

    MAX_STEPS=10 ./run_all_regressions.sh

## CI Mode (Phase 3.9)

Run in GitHub Actions–friendly mode with JSON output and annotations:

    ./run_all_regressions.sh --ci-mode

**CI Mode features:**
- Emits GitHub Actions workflow commands (`::error::`, `::notice::`, etc.)
- Writes `regression_summary.json` with machine-readable results
- Exit code: 0 (all pass), 1 (any fail), 2 (setup error)
- Suitable for automated dashboards and CI/CD pipelines

**Example JSON output:**
```json
{
  "timestamp": "2026-07-27T23:31:56Z",
  "harness_dir": "/path/to/SLUCM",
  "results_dir": "/path/to/_regression_results_20260727_233156",
  "ci_mode": 1,
  "passed_count": 3,
  "failed_count": 0,
  "skipped_count": 1,
  "passed": ["UCMBoston", "UCMSalamanca", "UCMKanda"],
  "failed": [],
  "skipped": ["UCMOsaka (no dir)"]
}
```

## Executable location

The script auto-discovers executables in each canonical directory. If you
build in a separate directory, point to it:

    ERF_BUILD_DIR=/path/to/build ./run_all_regressions.sh

## Interpreting output

- **PASS** — case ran to completion and check script returned 0
- **FAIL (run)** — executable crashed; see `_regression_results_*/CANONICAL_run.log`
- **FAIL (check)** — check script returned non-zero; see `_regression_results_*/CANONICAL_check.log`
- **SKIP** — canonical directory missing, no executable, or no inputs file

## Check scripts

Each canonical has (or should have) a `check_*.py` script in its directory.
Phase 3.1c ensures every canonical has at least a minimal check that verifies:
- All fields finite (no NaN/Inf)
- theta in [280, 320] K, wind mag < 30 m/s
- Solver produced non-trivial output (theta spread > 0.001 K)

Physics-specific checks (e.g. Boston UHI, drag validation) live in their
canonical's original check script and are called by the harness verbatim.

## Adding a new canonical

1. Create `Exec/CanonicalTests/SLUCM/UCMYourCanonical/`
2. Add `inputs` (or `inputs_singlelevel`) file
3. Add `check_your_canonical.py` (see any existing check_*.py for pattern)
4. Add executable-building `GNUmakefile` or CMake integration
5. `./run_all_regressions.sh UCMYourCanonical` — should auto-discover

## Merge-to-development checklist

Before opening a PR from `ERF-SLUCM` to `development`:

    cd Exec/CanonicalTests/SLUCM
    ./run_all_regressions.sh 2>&1 | tee /tmp/slucm_regression.log

All canonicals must PASS. Attach `/tmp/slucm_regression.log` to the merge PR body.

## GitHub Actions Integration (Phase 3.9)

The `.github/workflows/slucm_regression.yml` workflow automatically:
1. Builds unit tests with `-DERF_ENABLE_UCM=ON -DERF_ENABLE_UNIT_TESTS=ON`
2. Runs the unit test binary and gates the PR on failure
3. Builds the full `erf_exec` with MPI support
4. Runs canonical regressions with `--ci-mode`
5. Uploads `regression_summary.json` as a workflow artifact
6. Reports results to PR via annotations and step summary

The canonical regression job is informational only (`continue-on-error: true`),
while the unit test job gates the PR (must pass to merge).

## Phase 6.2a: Tree Radiation (Beer-Lambert)

New canonical: **UCMTreeRadUnit** (under development)

Tests Beer-Lambert SW attenuation through tree crown. Includes three variants:
- `inputs_off`: tree_rad_mode="off" (bit-identity test; output must match pre-Phase-6.2a)
- `inputs_on`: tree_rad_mode="beer_lambert" with standard tree layout
- `inputs_on_dense`: tree_rad_mode="beer_lambert" with dense tree layout

Verification:
- Q_tree_SW_abs field must be zero when tree_rad_mode="off"
- Q_tree_SW_abs must be nonzero when tree_rad_mode="beer_lambert" during daytime
- No Newton SEB dimensionality change; stays 3-variable (roof/wall/road)
- Attenuation formula: tau_tree = exp(-k_ext * LAD_bulk * L_path)
- Diagnostic only in Phase 6.2a (not prognostic)
