#!/usr/bin/env python3
"""
Phase 5.3: Permeable Road Unit Test Verification Script

Compares permeable_road_mode=off vs permeable_road_mode=simple to verify
that permeable roads produce latent heat flux (LE_perm > 0).

Parse logs for [UCM][5.3][permeable-road] LE_perm=[min, max] and assert:
  - off case: LE_perm not found or all zeros
  - simple case: LE_perm_max > 1.0 W/m^2

Exit codes:
  0: Test passed (LE_perm > 0 in simple case, 0 in off case)
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
    le_perm_min = []
    le_perm_max = []

    # per-step scratch (reset when a STEP_START is seen)
    scratch = {}

    # Regex patterns matching the debug output produced by ERF_UCMLayer.cpp
    RE_STEP_START = re.compile(
        r"\[Level 0 step (\d+)\] ADVANCE from elapsed time = ([\d.eE+-]+)"
    )
    RE_LE_PERM = re.compile(
        r"\[UCM\]\[5\.3\]\[permeable-road\].*LE_perm=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
    )

    def flush():
        if not scratch:
            return
        steps.append(scratch.get("step"))
        times.append(scratch.get("time"))
        le_perm_min.append(scratch.get("le_perm_min"))
        le_perm_max.append(scratch.get("le_perm_max"))

    with open(path) as f:
        for line in f:
            m = RE_STEP_START.search(line)
            if m:
                flush()
                scratch = {"step": int(m.group(1)), "time": float(m.group(2))}
                continue

            m = RE_LE_PERM.search(line)
            if m:
                scratch["le_perm_min"] = float(m.group(1))
                scratch["le_perm_max"] = float(m.group(2))
                continue

        flush()

    return {
        "path": str(path),
        "step": steps,
        "time_s": times,
        "le_perm_min": le_perm_min,
        "le_perm_max": le_perm_max,
    }


def safe_last(seq):
    for v in reversed(seq):
        if v is not None:
            return v
    return None


def check_run(data, label, expect_le_perm, target_step):
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

    # LE_perm checks
    le_present = [le for le in data["le_perm_max"] if le is not None]
    if expect_le_perm:
        if len(le_present) == 0:
            msgs.append(f"[{label}] FAIL: permeable_road_mode=simple but no LE_perm lines found")
            ok = False
        else:
            le_max_ever = max(le_present)
            le_min_ever = min(le_present)
            msgs.append(
                f"[{label}] LE_perm: min={le_min_ever:.2f} max={le_max_ever:.2f} W/m^2"
            )
            if le_max_ever < 1.0:
                msgs.append(
                    f"[{label}] FAIL: permeable road never activated "
                    f"(max LE_perm={le_max_ever:.4f} < 1.0 W/m^2)"
                )
                ok = False
            else:
                msgs.append(f"[{label}] OK: permeable road produced latent heat (max={le_max_ever:.2f} W/m^2)")
    else:
        if len(le_present) > 0:
            msgs.append(
                f"[{label}] FAIL: permeable_road_mode=off but {len(le_present)} LE_perm lines found "
                f"(should be zero)"
            )
            ok = False
        else:
            msgs.append(f"[{label}] OK: no LE_perm lines (permeable_road_mode=off)")

    return ok, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--off", required=True, help="Path to permeable_road_mode=off run log"
    )
    ap.add_argument(
        "--simple", required=True, help="Path to permeable_road_mode=simple run log"
    )
    ap.add_argument(
        "--max-step", type=int, default=2,
        help="Expected final step (fail if not reached). Default: 2"
    )
    args = ap.parse_args()

    off_path = Path(args.off)
    simple_path = Path(args.simple)
    for p in (off_path, simple_path):
        if not p.exists():
            print(f"ERROR: log file not found: {p}", file=sys.stderr)
            sys.exit(2)

    off = parse_log(off_path)
    simple = parse_log(simple_path)

    all_ok = True
    all_msgs = []

    ok, msgs = check_run(off, "permeable_road_off", expect_le_perm=False, target_step=args.max_step)
    all_ok = all_ok and ok
    all_msgs += msgs
    all_msgs.append("")

    ok, msgs = check_run(simple, "permeable_road_simple", expect_le_perm=True, target_step=args.max_step)
    all_ok = all_ok and ok
    all_msgs += msgs
    all_msgs.append("")

    for m in all_msgs:
        print(m)

    if all_ok:
        print("PASS: Phase 5.3 Permeable Road unit test.")
        sys.exit(0)
    else:
        print("FAIL: one or more checks did not pass. See messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
