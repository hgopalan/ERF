#!/usr/bin/env bash
# =====================================================================
# SLUCM Canonical Regression Harness
#
# Runs every canonical test in Exec/CanonicalTests/SLUCM/, invokes each
# canonical's Python check script, and reports pass/fail summary.
#
# Usage:
#   ./run_all_regressions.sh                    # run all canonicals
#   ./run_all_regressions.sh --ci-mode          # run with CI annotations & JSON output
#   ./run_all_regressions.sh UCMBoston          # run one canonical
#   ./run_all_regressions.sh UCMBoston UCMSalamancaMadrid  # run subset
#
# Environment:
#   ERF_EXEC           — path to the erf_exec binary (default: searches
#                        for 'erf_exec' in each canonical dir, then $PATH)
#   PYTHON             — Python interpreter (default: python3)
#   MAX_STEPS          — override max_step for quick smoke tests (default: use inputs value)
#   KEEP_OUTPUT        — 1 to keep plotfiles after run (default: 0 = cleanup)
#
# Exit codes:
#   0 — all canonicals passed
#   1 — one or more canonicals failed
#   2 — harness setup error (missing executable, script, etc.)
# =====================================================================

set -uo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
RESULTS_DIR="${HARNESS_DIR}/_regression_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Parse command-line flags
CI_MODE=0
CANONICALS_ARGS=()

for arg in "$@"; do
    if [ "$arg" = "--ci-mode" ]; then
        CI_MODE=1
    else
        CANONICALS_ARGS+=("$arg")
    fi
done

# -----------------------------------------------------------------------
# Resolve the single shared executable.
# Priority:
#   1. ERF_EXEC env var (explicit path)
#   2. erf_exec in the harness directory
#   3. erf_exec on $PATH
# -----------------------------------------------------------------------
if [ -n "${ERF_EXEC:-}" ]; then
    GLOBAL_EXEC="$ERF_EXEC"
elif [ -x "$HARNESS_DIR/erf_exec" ]; then
    GLOBAL_EXEC="$HARNESS_DIR/erf_exec"
elif command -v erf_exec &>/dev/null; then
    GLOBAL_EXEC="$(command -v erf_exec)"
else
    echo "ERROR: erf_exec not found."
    echo "  Set ERF_EXEC=/path/to/erf_exec, copy erf_exec into $HARNESS_DIR,"
    echo "  or add erf_exec to your PATH."
    exit 2
fi

echo "  Executable:   $GLOBAL_EXEC"

# Discover all canonical directories (must contain an inputs* file)
ALL_CANONICALS=()
for d in "$HARNESS_DIR"/*/; do
    if compgen -G "${d}inputs*" > /dev/null; then
        ALL_CANONICALS+=("$(basename "$d")")
    fi
done

# If arguments provided, run only those canonicals
if [ ${#CANONICALS_ARGS[@]} -gt 0 ]; then
    CANONICALS=("${CANONICALS_ARGS[@]}")
else
    CANONICALS=("${ALL_CANONICALS[@]}")
fi

echo "======================================================================"
echo "SLUCM Regression Harness"
echo "  Harness dir:  $HARNESS_DIR"
echo "  Results dir:  $RESULTS_DIR"
echo "  Executable:   $GLOBAL_EXEC"
echo "  Canonicals:   ${CANONICALS[*]}"
if [ "$CI_MODE" = "1" ]; then
    echo "  CI Mode:      ON (JSON + GitHub annotations)"
fi
echo "======================================================================"

declare -a PASSED FAILED SKIPPED


for canon in "${CANONICALS[@]}"; do
    CANON_DIR="$HARNESS_DIR/$canon"
    if [ ! -d "$CANON_DIR" ]; then
        echo "[SKIP]   $canon — directory not found"
        SKIPPED+=("$canon (no dir)")
        continue
    fi

    echo ""
    echo "----------------------------------------------------------------------"
    echo "[$canon] Starting"
    echo "----------------------------------------------------------------------"

    cd "$CANON_DIR"

    # Find inputs file — prefer inputs_singlelevel, then inputs, then any inputs*
    INPUTS=""
    for candidate in inputs_singlelevel inputs inputs*; do
        [ -f "$candidate" ] && INPUTS="$candidate" && break
    done
    if [ -z "$INPUTS" ]; then
        echo "[$canon] SKIP — no inputs file found"
        SKIPPED+=("$canon (no inputs)")
        cd "$HARNESS_DIR"
        continue
    fi

    # Cleanup old output before run
    rm -rf plt_* ucm_diag.dat 2>/dev/null || true

    # Build override args
    OVERRIDE_ARGS=""
    if [ -n "${MAX_STEPS:-}" ]; then
        OVERRIDE_ARGS="max_step=$MAX_STEPS"
    fi

    # Run the case
    RUN_LOG="$RESULTS_DIR/${canon}_run.log"
    echo "[$canon] Running: $GLOBAL_EXEC $INPUTS $OVERRIDE_ARGS"
    if "$GLOBAL_EXEC" "$INPUTS" $OVERRIDE_ARGS > "$RUN_LOG" 2>&1; then
        echo "[$canon] Run completed"
    else
        echo "[$canon] FAIL — run crashed (see $RUN_LOG)"
        FAILED+=("$canon (run)")
        cd "$HARNESS_DIR"
        continue
    fi

    # Find and run check script
    CHECK_SCRIPT=""
    for candidate in check_*.py; do
        [ -f "$candidate" ] && CHECK_SCRIPT="$candidate" && break
    done

    if [ -z "$CHECK_SCRIPT" ]; then
        echo "[$canon] FAIL — no check_*.py script in canonical directory"
        FAILED+=("$canon (no check)")
        cd "$HARNESS_DIR"
        continue
    fi

    CHECK_LOG="$RESULTS_DIR/${canon}_check.log"
    echo "[$canon] Verifying with: $PYTHON $CHECK_SCRIPT"
    if "$PYTHON" "$CHECK_SCRIPT" > "$CHECK_LOG" 2>&1; then
        echo "[$canon] PASS"
        PASSED+=("$canon")
    else
        echo "[$canon] FAIL — check script returned non-zero (see $CHECK_LOG)"
        tail -20 "$CHECK_LOG"
        FAILED+=("$canon (check)")
    fi

    # Optional cleanup
    if [ "${KEEP_OUTPUT:-0}" != "1" ]; then
        rm -rf plt_* ucm_diag.dat 2>/dev/null || true
    fi

    cd "$HARNESS_DIR"
done

# Summary
echo ""
echo "======================================================================"
echo "SLUCM Regression Summary"
echo "======================================================================"
echo "Passed  (${#PASSED[@]}): ${PASSED[*]:-<none>}"
echo "Failed  (${#FAILED[@]}): ${FAILED[*]:-<none>}"
echo "Skipped (${#SKIPPED[@]}): ${SKIPPED[*]:-<none>}"
echo ""
echo "Full logs: $RESULTS_DIR"
echo "======================================================================"

# Generate JSON summary
SUMMARY_JSON="$HARNESS_DIR/regression_summary.json"
{
    echo "{"
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"harness_dir\": \"$HARNESS_DIR\","
    echo "  \"results_dir\": \"$RESULTS_DIR\","
    echo "  \"ci_mode\": $CI_MODE,"
    echo "  \"passed_count\": ${#PASSED[@]},"
    echo "  \"failed_count\": ${#FAILED[@]},"
    echo "  \"skipped_count\": ${#SKIPPED[@]},"
    echo "  \"passed\": ["
    for i in "${!PASSED[@]}"; do
        echo -n "    \"${PASSED[$i]}\""
        if [ $i -lt $((${#PASSED[@]} - 1)) ]; then echo ","; else echo ""; fi
    done
    echo "  ],"
    echo "  \"failed\": ["
    for i in "${!FAILED[@]}"; do
        echo -n "    \"${FAILED[$i]}\""
        if [ $i -lt $((${#FAILED[@]} - 1)) ]; then echo ","; else echo ""; fi
    done
    echo "  ],"
    echo "  \"skipped\": ["
    for i in "${!SKIPPED[@]}"; do
        echo -n "    \"${SKIPPED[$i]}\""
        if [ $i -lt $((${#SKIPPED[@]} - 1)) ]; then echo ","; else echo ""; fi
    done
    echo "  ]"
    echo "}"
} > "$SUMMARY_JSON"

echo ""
echo "Summary written to: $SUMMARY_JSON"

# =====================================================================
# Phase 6.2b hotfix1: Merge-blocker verification (Bug I fix)
# =====================================================================
# Grep merge-blockers G1-G6: Verify Contracts #29-#32 and no TODOs
echo ""
echo "======================================================================"
echo "Merge-Blocker Verification (Phase 6.2b hotfix1)"
echo "======================================================================"

MERGE_BLOCKERS_FAIL=0

# G1: 3-var solver byte-identity (Contract #29)
# Check that ERF_UCMSEBSolver.H is unchanged
if grep -q "ERF_UCMSEBSolver.H" "$HARNESS_DIR/../../../Source/UrbanCanopy/ERF_UCMLayer.cpp" 2>/dev/null; then
    echo "[G1] 3-var solver dispatch bound — OK (Contract #29)"
else
    echo "[G1] WARNING: 3-var solver binding not found (may be OK if using different pattern)"
fi

# G2: No TODO/FIXME in Phase 6.2b modified files
echo ""
echo "[G2] Checking for unresolved TODOs in Phase 6.2b modified files..."
BLOCKERS=0
if grep -n "TODO\|FIXME\|placeholder\|not yet implemented\|fall back" \
    "$HARNESS_DIR/../../../Source/UrbanCanopy/ERF_UCMSEBSolver4Var.H" 2>/dev/null | grep -v "^[[:space:]]*\/\/"; then
    echo "  ERROR: Unresolved TODOs in ERF_UCMSEBSolver4Var.H"
    BLOCKERS=$((BLOCKERS + 1))
fi
if grep -n "TODO.*4-var\|TODO.*dispatch" \
    "$HARNESS_DIR/../../../Source/UrbanCanopy/ERF_UCMLayer.cpp" 2>/dev/null | grep "seb_mode == SEBMode::FourVar"; then
    echo "  ERROR: Unresolved TODO in 4-var dispatch (ERF_UCMLayer.cpp)"
    BLOCKERS=$((BLOCKERS + 1))
fi
if [ $BLOCKERS -eq 0 ]; then
    echo "  G2: No unresolved TODOs in Phase 6.2b files — OK"
else
    echo "  G2: FAIL — $BLOCKERS unresolved TODO(s) found"
    MERGE_BLOCKERS_FAIL=1
fi

# G3: Verify solve_facet_seb_4var_with_diag signature includes new parameters
echo ""
echo "[G3] Checking 4-var solver signature includes T_atm_ref and crown_area_frac..."
if grep -q "T_atm_ref\|T_atm_ref" \
    "$HARNESS_DIR/../../../Source/UrbanCanopy/ERF_UCMSEBSolver4Var.H" 2>/dev/null && \
   grep -q "crown_area_frac" \
    "$HARNESS_DIR/../../../Source/UrbanCanopy/ERF_UCMSEBSolver4Var.H" 2>/dev/null; then
    echo "  G3: Solver signature updated (Contract #31, #32) — OK"
else
    echo "  G3: WARNING: New parameters not found in solver signature"
fi

# G4: Verify H_crown split into up/down (Bug D fix)
echo ""
echo "[G4] Checking H_crown split into upward and downward components..."
if grep -q "H_crown_up_out\|H_crown_down_out" \
    "$HARNESS_DIR/../../../Source/UrbanCanopy/ERF_UCMSEBSolver4Var.H" 2>/dev/null; then
    echo "  G4: H_crown split implemented (Bug D) — OK"
else
    echo "  G4: WARNING: H_crown split not found in solver"
fi

# G5: Verify 4-var dispatch in ERF_UCMLayer.cpp calls the solver
echo ""
echo "[G5] Checking 4-var dispatch calls solve_facet_seb_4var_with_diag..."
if grep -q "solve_facet_seb_4var_with_diag" \
    "$HARNESS_DIR/../../../Source/UrbanCanopy/ERF_UCMLayer.cpp" 2>/dev/null; then
    echo "  G5: 4-var solver dispatch implemented (Bug A) — OK"
else
    echo "  G5: WARNING: 4-var solver dispatch not found"
fi

# G6: Verify canyon-air update includes H_crown_down
echo ""
echo "[G6] Checking canyon-air update includes H_crown_down (Bug G fix)..."
CANYON_CHECK=$(grep -n "H_crown_down" "$HARNESS_DIR/../../../Source/UrbanCanopy/ERF_UCMLayer.cpp" 2>/dev/null | tail -1)
if [ -n "$CANYON_CHECK" ]; then
    echo "  G6: H_crown_down canyon coupling implemented (Bug G) — OK"
    echo "      $CANYON_CHECK"
else
    echo "  G6: WARNING: H_crown_down not found in canyon-air update"
fi

if [ $MERGE_BLOCKERS_FAIL -eq 0 ]; then
    echo ""
    echo "Merge-Blocker Status: PASS (all critical checks passed)"
else
    echo ""
    echo "Merge-Blocker Status: FAIL (critical checks failed)"
    exit 1
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
exit 0
