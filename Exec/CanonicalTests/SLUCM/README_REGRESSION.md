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

## Future CI integration

A future `.github/workflows/slucm_regression.yml` can invoke this harness
after building. The harness exit code (0=pass, 1=fail, 2=setup-error) is
CI-friendly. Do not add the workflow in Phase 3.1c — that is Phase 3.9.
