#!/usr/bin/env python3
"""
Phase 5.3: Cool Roof Unit Test Verification Script

Compares high-albedo vs low-albedo roof skin temperatures to verify
that higher albedo reduces surface temperature (passive cooling effect).

Parse logs for T_skin_roof min/max values and assert:
  - high_albedo case: T_skin_roof_max < low_albedo case: T_skin_roof_max

Exit codes:
  0: Test passed (high albedo cooler than low albedo)
  1: Test failed (assertion error or physical inconsistency)
  2: Missing log file
"""

import argparse
import re
import sys
from pathlib import Path


def parse_log(path):
    """Return a dict of per-step time series parsed from an ERF run log."""
    steps = []
    times = []
    t_roof_min = []
    t_roof_max = []

    # per-step scratch (reset when a STEP_START is seen)
    scratch = {}

    # Regex patterns matching the debug output produced by ERF_UCMLayer.cpp
    RE_STEP_START = re.compile(
        r"\[Level 0 step (\d+)\] ADVANCE from elapsed time = ([\d.eE+-]+)"
    )
    RE_TROOF = re.compile(
        r"T_skin_roof=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
    )

    def flush():
        if not scratch:
            return
        steps.append(scratch.get("step"))
        times.append(scratch.get("time"))
        t_roof_min.append(scratch.get("troof_min"))
        t_roof_max.append(scratch.get("troof_max"))

    with open(path) as f:
        for line in f:
            m = RE_STEP_START.search(line)
            if m:
                flush()
                scratch = {"step": int(m.group(1)), "time": float(m.group(2))}
                continue

            m = RE_TROOF.search(line)
            if m:
                scratch["troof_min"] = float(m.group(1))
                scratch["troof_max"] = float(m.group(2))
                continue

        flush()

    return {
        "path": str(path),
        "step": steps,
        "time_s": times,
        "t_roof_min": t_roof_min,
        "t_roof_max": t_roof_max,
    }


def safe_last(seq):
    for v in reversed(seq):
        if v is not None:
            return v
    return None


def check_run(data, label, target_step):
    """Return (passed, messages) for a parsed run."""
    msgs = []
    ok = True

    n = len(data["step"])
    msgs.append(f"[{label}] parsed {n} timesteps from {data['path']}")

    if n == 0:
        msgs.append(f"[{label}] FAIL: no timesteps parsed - check log path/format")
        return False, msgs

    last_step = safe_last(data["step"])
    if last_step is None:
        msgs.append(f"[{label}] FAIL: no step number found")
        ok = False
    elif target_step is not None and last_step < target_step:
        msgs.append(
            f"[{label}] FAIL: reached step {last_step} but expected {target_step} "
            f"(likely hung or crashed)"
        )
        ok = False
    else:
        msgs.append(f"[{label}] OK: reached step {last_step}")

    # T_skin_roof checks
    troof = [t for t in data["t_roof_max"] if t is not None]
    if troof:
        msgs.append(
            f"[{label}] T_skin_roof: min={min(troof):.2f} K max={max(troof):.2f} K"
        )
    else:
        msgs.append(f"[{label}] WARN: no T_skin_roof data found")
        ok = False

    return ok, msgs


def compare_runs(high_albedo, low_albedo):
    """Physical consistency: high-albedo roof should be cooler."""
    msgs = []
    ok = True

    # Get final T_skin_roof_max values
    high_t_max = safe_last([t for t in high_albedo["t_roof_max"] if t is not None])
    low_t_max = safe_last([t for t in low_albedo["t_roof_max"] if t is not None])

    if high_t_max is None or low_t_max is None:
        msgs.append("[compare] SKIP: missing T_skin_roof data in one or both runs")
        return True, msgs

    delta = low_t_max - high_t_max
    msgs.append(
        f"[compare] T_skin_roof_max HIGH_ALBEDO={high_t_max:.2f} K, "
        f"LOW_ALBEDO={low_t_max:.2f} K, delta={delta:+.2f} K"
    )

    if delta < 0.1:
        msgs.append(
            f"[compare] FAIL: high-albedo roof NOT cooler than low-albedo "
            f"(delta={delta:+.2f} K, should be > 0.1 K)"
        )
        ok = False
    else:
        msgs.append(
            f"[compare] OK: high-albedo roof cooler than low-albedo (delta={delta:+.2f} K)"
        )

    return ok, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--high", required=True, help="Path to high-albedo run log"
    )
    ap.add_argument(
        "--low", required=True, help="Path to low-albedo run log"
    )
    ap.add_argument(
        "--max-step", type=int, default=2,
        help="Expected final step (fail if not reached). Default: 2"
    )
    args = ap.parse_args()

    high_path = Path(args.high)
    low_path = Path(args.low)
    for p in (high_path, low_path):
        if not p.exists():
            print(f"ERROR: log file not found: {p}", file=sys.stderr)
            sys.exit(2)

    high = parse_log(high_path)
    low = parse_log(low_path)

    all_ok = True
    all_msgs = []

    ok, msgs = check_run(high, "high_albedo", target_step=args.max_step)
    all_ok = all_ok and ok
    all_msgs += msgs
    all_msgs.append("")

    ok, msgs = check_run(low, "low_albedo", target_step=args.max_step)
    all_ok = all_ok and ok
    all_msgs += msgs
    all_msgs.append("")

    ok, msgs = compare_runs(high, low)
    all_ok = all_ok and ok
    all_msgs += msgs
    all_msgs.append("")

    for m in all_msgs:
        print(m)

    if all_ok:
        print("PASS: Phase 5.3 Cool Roof unit test.")
        sys.exit(0)
    else:
        print("FAIL: one or more checks did not pass. See messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
