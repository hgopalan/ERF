#!/usr/bin/env python3
"""
Phase 3.6 Multi-Level One-Way Regression Check

Parses run.log from a completed UCMBostonMultiLevel run and validates
that the SLUCM stack runs correctly with anchor_level = 1 (one AMR 
refinement level above the base).

This is a plumbing verification; no physics changes are introduced.

Usage:
    python3 check_multilevel.py [path_to_run.log]

Default: ./run.log

Exit codes:
    0 = PASS (all metrics met)
    1 = FAIL (one or more metrics violated)
"""

import re
import sys
from pathlib import Path


def parse_log(log_path):
    """Extract validation metrics from run.log."""
    text = Path(log_path).read_text()

    # Metric 1: Check for anchor_level=1
    anchor_level_re = re.compile(r"anchor_level\s*=\s*1", re.IGNORECASE)
    has_anchor_level_1 = bool(anchor_level_re.search(text))

    # Metric 2: Check for assertion failures or aborts
    assertion_re = re.compile(r"\bAssertion\b", re.IGNORECASE)
    abort_re = re.compile(r"\babort\b", re.IGNORECASE)
    has_assertion = bool(assertion_re.search(text))
    has_abort = bool(abort_re.search(text))

    # Metric 3: Check for NaN or Inf in temperature fields
    nan_re = re.compile(r"\bnan\b", re.IGNORECASE)
    inf_re = re.compile(r"\binf\b", re.IGNORECASE)
    has_nan = bool(nan_re.search(text))
    has_inf = bool(inf_re.search(text))

    # Metric 4: Check "Clamped to T_skin_min" lines for zero clamps
    clamp_re = re.compile(
        r"Clamped to T_skin_min=\d+K:\s+roof=(\d+)\s+wall=(\d+)\s+road=(\d+)"
    )
    total_clamps = 0
    for m in clamp_re.finditer(text):
        total_clamps += int(m.group(1)) + int(m.group(2)) + int(m.group(3))

    # Metric 5: Check for T_skin_roof= line (SEB solver was called)
    t_skin_roof_re = re.compile(r"T_skin_roof\s*=")
    has_t_skin_roof = bool(t_skin_roof_re.search(text))

    return {
        "has_anchor_level_1": has_anchor_level_1,
        "has_assertion": has_assertion,
        "has_abort": has_abort,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "total_clamps": total_clamps,
        "has_t_skin_roof": has_t_skin_roof,
    }


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "run.log"
    if not Path(log_path).exists():
        print(f"FAIL: log file not found: {log_path}")
        return 1

    m = parse_log(log_path)
    print("Phase 3.6 Multi-Level One-Way Regression -- parsed metrics:")
    print(f"  anchor_level=1 found:       {m['has_anchor_level_1']}")
    print(f"  No Assertion found:         {not m['has_assertion']}")
    print(f"  No abort found:             {not m['has_abort']}")
    print(f"  No nan found:               {not m['has_nan']}")
    print(f"  No inf found:               {not m['has_inf']}")
    print(f"  Total T_skin clamps:        {m['total_clamps']}")
    print(f"  T_skin_roof line found:     {m['has_t_skin_roof']}")
    print()

    failures = []
    if not m["has_anchor_level_1"]:
        failures.append("anchor_level=1 not found in log")
    if m["has_assertion"]:
        failures.append("Assertion found in log")
    if m["has_abort"]:
        failures.append("abort found in log")
    if m["has_nan"]:
        failures.append("nan found in temperature fields")
    if m["has_inf"]:
        failures.append("inf found in temperature fields")
    if m["total_clamps"] > 0:
        failures.append(f"total_clamps={m['total_clamps']} > 0 (expected zero)")
    if not m["has_t_skin_roof"]:
        failures.append("T_skin_roof line not found (SEB solver not called)")

    if failures:
        print("FAIL -- validation errors:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASS -- all metrics met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
