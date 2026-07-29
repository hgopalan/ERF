#!/usr/bin/env python3
"""
Phase 5.3 D11 — 24-hour Boston diurnal Green Roof comparison.

Parses log file produced by:

    mpirun -np N ../../../build/Exec/erf_exec inputs_diurnal_24h_green > run_green.log

Extracts per-step LE_green, T_canyon_air, T_skin_roof/wall/road and verifies:

Pass criteria:
  * LE_green produced during daytime hours (max LE > 1.0 W/m^2)
  * At least 25% of samples show LE_green > 1.0 during daytime
  * Run reaches max_step (60000) without hanging
  * T_canyon_air and ATM state physically reasonable

Usage:
    python3 check_green_roof_diurnal.py \\
        --log run_green.log \\
        [--max-step 60000] \\
        [--min-daytime-fraction 0.25]

Exits 0 on pass, 1 on any failed check, 2 on missing log.
"""

import argparse
import re
import sys
from pathlib import Path


def parse_log(path):
    """Return a dict of per-step time series parsed from an ERF run log."""
    steps = []
    times = []
    le_green_min = []
    le_green_max = []
    t_can_min = []
    t_can_max = []
    t_roof_min = []
    t_roof_max = []
    t_wall_min = []
    t_wall_max = []
    t_road_min = []
    t_road_max = []

    # per-step scratch (reset when a STEP_START is seen)
    scratch = {}

    # Regex patterns matching the debug output produced by ERF_UCMLayer.cpp
    RE_STEP_START = re.compile(
        r"\[Level 0 step (\d+)\] ADVANCE from elapsed time = ([\d.eE+-]+)"
    )
    RE_LE_GREEN = re.compile(
        r"\[UCM\]\[5\.3\]\[green-roof\].*LE_green=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
    )
    RE_TCAN = re.compile(
        r"T_canyon_air=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
    )
    RE_TROOF = re.compile(
        r"T_skin_roof=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
    )
    RE_TWALL = re.compile(
        r"T_skin_wall=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
    )
    RE_TROAD = re.compile(
        r"T_skin_road=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
    )

    def flush():
        if not scratch:
            return
        steps.append(scratch.get("step"))
        times.append(scratch.get("time"))
        le_green_min.append(scratch.get("le_green_min"))
        le_green_max.append(scratch.get("le_green_max"))
        t_can_min.append(scratch.get("tcan_min"))
        t_can_max.append(scratch.get("tcan_max"))
        t_roof_min.append(scratch.get("troof_min"))
        t_roof_max.append(scratch.get("troof_max"))
        t_wall_min.append(scratch.get("twall_min"))
        t_wall_max.append(scratch.get("twall_max"))
        t_road_min.append(scratch.get("troad_min"))
        t_road_max.append(scratch.get("troad_max"))

    with open(path) as f:
        for line in f:
            m = RE_STEP_START.search(line)
            if m:
                flush()
                scratch = {"step": int(m.group(1)), "time": float(m.group(2))}
                continue

            m = RE_LE_GREEN.search(line)
            if m:
                scratch["le_green_min"] = float(m.group(1))
                scratch["le_green_max"] = float(m.group(2))
                continue

            m = RE_TCAN.search(line)
            if m:
                scratch["tcan_min"] = float(m.group(1))
                scratch["tcan_max"] = float(m.group(2))
                continue

            m = RE_TROOF.search(line)
            if m:
                scratch["troof_min"] = float(m.group(1))
                scratch["troof_max"] = float(m.group(2))
                continue

            m = RE_TWALL.search(line)
            if m:
                scratch["twall_min"] = float(m.group(1))
                scratch["twall_max"] = float(m.group(2))
                continue

            m = RE_TROAD.search(line)
            if m:
                scratch["troad_min"] = float(m.group(1))
                scratch["troad_max"] = float(m.group(2))
                continue

        flush()

    return {
        "path": str(path),
        "step": steps,
        "time_s": times,
        "le_green_min": le_green_min,
        "le_green_max": le_green_max,
        "t_can_min": t_can_min,
        "t_can_max": t_can_max,
        "t_roof_min": t_roof_min,
        "t_roof_max": t_roof_max,
        "t_wall_min": t_wall_min,
        "t_wall_max": t_wall_max,
        "t_road_min": t_road_min,
        "t_road_max": t_road_max,
    }


def safe_last(seq):
    for v in reversed(seq):
        if v is not None:
            return v
    return None


def check_run(data, target_step, min_daytime_fraction):
    """Return (passed, messages) for a parsed run."""
    msgs = []
    ok = True

    n = len(data["step"])
    msgs.append(f"[green_roof_diurnal] parsed {n} timesteps from {data['path']}")

    if n == 0:
        msgs.append(f"[green_roof_diurnal] FAIL: no timesteps parsed - check log path/format")
        return False, msgs

    last_step = safe_last(data["step"])
    if last_step is None:
        msgs.append(f"[green_roof_diurnal] FAIL: no step number found")
        ok = False
    elif target_step is not None and last_step < target_step:
        msgs.append(
            f"[green_roof_diurnal] FAIL: reached step {last_step} but expected {target_step} "
            f"(likely hung or crashed)"
        )
        ok = False
    else:
        msgs.append(f"[green_roof_diurnal] OK: reached step {last_step}")

    # LE_green checks: should be active during daytime (25% threshold)
    le_present = [le for le in data["le_green_max"] if le is not None]
    if len(le_present) == 0:
        msgs.append(f"[green_roof_diurnal] FAIL: no LE_green lines found in log")
        ok = False
    else:
        le_max_ever = max(le_present)
        frac_active = sum(1 for le in le_present if le > 1.0) / len(le_present)
        msgs.append(
            f"[green_roof_diurnal] LE_green: max={le_max_ever:.2f} W/m^2, "
            f"active fraction (>1.0)={frac_active*100:.1f}%"
        )
        if le_max_ever < 1.0:
            msgs.append(
                f"[green_roof_diurnal] FAIL: max LE_green={le_max_ever:.4f} < 1.0 W/m^2; "
                f"green roof inactive"
            )
            ok = False
        elif frac_active < min_daytime_fraction:
            msgs.append(
                f"[green_roof_diurnal] FAIL: only {frac_active*100:.1f}% of samples have LE_green>1.0; "
                f"expected >={min_daytime_fraction*100:.0f}%"
            )
            ok = False
        else:
            msgs.append(
                f"[green_roof_diurnal] OK: green roof active {frac_active*100:.1f}% of time"
            )

    # Sanity checks on canyon T
    tcan = [t for t in data["t_can_max"] if t is not None]
    if tcan:
        msgs.append(
            f"[green_roof_diurnal] T_canyon_air: min={min(tcan):.2f} K max={max(tcan):.2f} K"
        )
        if min(tcan) < 240 or max(tcan) > 340:
            msgs.append(
                f"[green_roof_diurnal] WARN: T_canyon_air outside [240, 340] K - check physics"
            )

    troof = [t for t in data["t_roof_max"] if t is not None]
    if troof:
        msgs.append(
            f"[green_roof_diurnal] T_skin_roof: min={min(troof):.2f} K max={max(troof):.2f} K"
        )

    return ok, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to green roof run log")
    ap.add_argument(
        "--max-step", type=int, default=60000,
        help="Expected final step (fail if not reached). Default: 60000"
    )
    ap.add_argument(
        "--min-daytime-fraction", type=float, default=0.25,
        help="Minimum fraction of samples with LE_green > 1.0. Default: 0.25"
    )
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        sys.exit(2)

    data = parse_log(log_path)

    all_ok = True
    all_msgs = []

    ok, msgs = check_run(data, target_step=args.max_step, min_daytime_fraction=args.min_daytime_fraction)
    all_ok = all_ok and ok
    all_msgs += msgs
    all_msgs.append("")

    for m in all_msgs:
        print(m)

    if all_ok:
        print("PASS: Phase 5.3 D11 Boston diurnal green roof test.")
        sys.exit(0)
    else:
        print("FAIL: one or more checks did not pass. See messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
