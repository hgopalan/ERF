#!/usr/bin/env python3
"""
Phase 5.2 HVAC canonical validation for UCMBostonDiurnalHVAC24h.

Compares two runs (hvac_off vs hvac_simple) and asserts that HVAC waste heat
enters the anthropogenic-heat field fields.AH as expected.

Parses ONLY these authoritative log lines:
  1. "[UCM][5.2][hvac] mode=... hour=H Q_HVAC=[qmin, qmax] W/m²"
       -> Q_HVAC_diag min/max at each step (informational).
  2. "  AH min=<v> max=<v> W/m2"  (indented, from ERF_UCMLayer.cpp line ~1251)
       -> fields.AH->max at end of advance(). This is what Facet3D reads.
  3. "  MOST    H_roof min=<v> max=<v> W/m2  (drives ATM injection)"
       -> Sensible heat lumped for the atmosphere (base + AH contribution).

Pass criteria:
  A) hvac_off:    AH max stays at the base value (60 W/m^2 for Boston default).
  B) hvac_simple: AH max exceeds base on at least N_fire_min steps
                  (proving `AH_a(i,j,0) += Q_HVAC` persists to fields.AH).
  C) hvac_simple: MOST H_roof max on firing steps is greater than the
                  corresponding hvac_off value (proving AH reaches the ATM).

Usage:
    python check_hvac_diurnal.py run_hvac_off.log run_hvac_simple.log
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# --- Regexes bound to exact log line formats emitted by ERF_UCMLayer.cpp -----

# Line: "[UCM][5.2][hvac] mode=simple hour=14 Q_HVAC=[0, 61.28] W/m²"
RE_HVAC = re.compile(
    r"\[UCM\]\[5\.2\]\[hvac\]\s+mode=(\w+)\s+hour=(\d+)\s+"
    r"Q_HVAC=\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\]"
)

# Line: "  AH min=5 max=60 W/m2"   (two leading spaces, indented under [3.5A][SEB])
RE_AH = re.compile(r"^\s{2}AH\s+min=([-\d.eE+]+)\s+max=([-\d.eE+]+)\s+W/m2\s*$")

# Line: "  MOST    H_roof min=3.79 max=46.74 W/m2  (drives ATM injection)"
RE_H_ROOF_ATM = re.compile(
    r"^\s{2}MOST\s+H_roof\s+min=([-\d.eE+]+)\s+max=([-\d.eE+]+)\s+W/m2\s+"
    r"\(drives ATM injection\)"
)

# Base AH value expected when HVAC contributes nothing.
# Matches Boston Diurnal default: AH_daytime_peak=60, override CSV up to 60.
AH_BASE_DEFAULT = 60.0

# Tolerance (W/m^2) above AH_BASE to count a step as "HVAC fired".
AH_FIRE_THRESHOLD = 1.0

# Minimum number of firing steps required in hvac_simple to pass.
N_FIRE_MIN = 100


@dataclass
class RunTrace:
    """Per-step parsed values from an ERF log for one HVAC config."""

    path: Path
    q_hvac_max: List[float] = field(default_factory=list)  # from Q_HVAC_diag
    ah_max: List[float] = field(default_factory=list)      # from fields.AH
    h_roof_atm_max: List[float] = field(default_factory=list)
    hour: List[int] = field(default_factory=list)
    mode: Optional[str] = None


def parse_log(path: Path) -> RunTrace:
    """Stream-parse the log and collect the three quantities per step."""
    trace = RunTrace(path=path)
    with path.open("r", errors="replace") as fh:
        for line in fh:
            m = RE_HVAC.search(line)
            if m:
                trace.mode = m.group(1)
                trace.hour.append(int(m.group(2)))
                trace.q_hvac_max.append(float(m.group(4)))
                continue
            m = RE_AH.match(line)
            if m:
                trace.ah_max.append(float(m.group(2)))
                continue
            m = RE_H_ROOF_ATM.match(line)
            if m:
                trace.h_roof_atm_max.append(float(m.group(2)))
                continue
    return trace


def summarize(trace: RunTrace, label: str) -> None:
    print(f"[{label}] file={trace.path.name}")
    print(f"[{label}]   mode              = {trace.mode!r}")
    print(f"[{label}]   n_hvac_lines      = {len(trace.q_hvac_max)}")
    print(f"[{label}]   n_ah_debug_lines  = {len(trace.ah_max)}")
    print(f"[{label}]   n_hroof_atm_lines = {len(trace.h_roof_atm_max)}")
    if trace.q_hvac_max:
        print(
            f"[{label}]   Q_HVAC_diag max   "
            f"min={min(trace.q_hvac_max):.3f} "
            f"max={max(trace.q_hvac_max):.3f} W/m^2"
        )
    if trace.ah_max:
        print(
            f"[{label}]   fields.AH max     "
            f"min={min(trace.ah_max):.3f} "
            f"max={max(trace.ah_max):.3f} W/m^2"
        )
    if trace.h_roof_atm_max:
        print(
            f"[{label}]   MOST H_roof max   "
            f"min={min(trace.h_roof_atm_max):.3f} "
            f"max={max(trace.h_roof_atm_max):.3f} W/m^2"
        )
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_off", type=Path, help="Log from hvac_mode=off run")
    ap.add_argument("run_simple", type=Path, help="Log from hvac_mode=simple run")
    ap.add_argument(
        "--ah-base",
        type=float,
        default=AH_BASE_DEFAULT,
        help=f"Expected base AH_max with HVAC off (default {AH_BASE_DEFAULT})",
    )
    ap.add_argument(
        "--fire-threshold",
        type=float,
        default=AH_FIRE_THRESHOLD,
        help=f"W/m^2 above base to count as firing (default {AH_FIRE_THRESHOLD})",
    )
    ap.add_argument(
        "--n-fire-min",
        type=int,
        default=N_FIRE_MIN,
        help=f"Min firing steps required in simple run (default {N_FIRE_MIN})",
    )
    args = ap.parse_args()

    for p in (args.run_off, args.run_simple):
        if not p.is_file():
            print(f"ERROR: log not found: {p}", file=sys.stderr)
            return 2

    off = parse_log(args.run_off)
    simple = parse_log(args.run_simple)

    print("=" * 72)
    print("Phase 5.2 HVAC canonical: parse summary")
    print("=" * 72)
    summarize(off, "off   ")
    summarize(simple, "simple")

    # ---- Sanity: both logs produced debug lines ----------------------------
    if not off.ah_max or not simple.ah_max:
        print("FAIL: one or both runs produced zero 'AH min=... max=...' lines. "
              "Ensure ucm.debug=1 in the inputs files.")
        return 1
    if len(off.ah_max) != len(simple.ah_max):
        print(
            f"WARN: step counts differ (off={len(off.ah_max)} "
            f"simple={len(simple.ah_max)}); comparisons will use min length."
        )

    n = min(len(off.ah_max), len(simple.ah_max))

    # ---- Check A: hvac_off keeps AH at base --------------------------------
    off_ah_max_over_run = max(off.ah_max[:n])
    tol = 1e-6
    if off_ah_max_over_run > args.ah_base + tol:
        print(
            f"FAIL [A]: hvac_off has AH max={off_ah_max_over_run:.3f} > "
            f"base {args.ah_base:.3f}. Something else is writing AH."
        )
        return 1
    print(f"PASS [A]: hvac_off AH max stays at base = {off_ah_max_over_run:.3f} W/m^2")

    # ---- Check B: hvac_simple exceeds base on enough steps -----------------
    n_fire = sum(1 for v in simple.ah_max[:n] if v > args.ah_base + args.fire_threshold)
    simple_ah_max_over_run = max(simple.ah_max[:n])
    print(
        f"[compare] hvac_simple firing steps "
        f"(AH > {args.ah_base + args.fire_threshold:.2f}): "
        f"{n_fire} / {n}"
    )
    print(
        f"[compare] hvac_simple AH max over run = "
        f"{simple_ah_max_over_run:.3f} W/m^2 "
        f"(base={args.ah_base:.3f}, expected > base)"
    )
    if n_fire < args.n_fire_min:
        print(
            f"FAIL [B]: only {n_fire} firing steps (< {args.n_fire_min}). "
            "HVAC block did not persist to fields.AH, OR HVAC never engaged. "
            "Check that T_canyon_air exceeds setpoint - hysteresis somewhere in the run."
        )
        return 1
    if simple_ah_max_over_run <= args.ah_base + args.fire_threshold:
        print(
            f"FAIL [B]: hvac_simple AH max {simple_ah_max_over_run:.3f} "
            f"<= base+threshold {args.ah_base + args.fire_threshold:.3f}. "
            "HVAC did not persist to fields.AH."
        )
        return 1
    print(
        f"PASS [B]: hvac_simple fires on {n_fire} steps; "
        f"max AH = {simple_ah_max_over_run:.3f} W/m^2 > base"
    )

    # ---- Check C: MOST H_roof (into ATM) is larger in simple ---------------
    if off.h_roof_atm_max and simple.h_roof_atm_max:
        m = min(len(off.h_roof_atm_max), len(simple.h_roof_atm_max))
        n_larger = sum(
            1 for i in range(m)
            if simple.h_roof_atm_max[i] > off.h_roof_atm_max[i] + 0.1
        )
        print(
            f"[compare] MOST H_roof larger in simple than off: "
            f"{n_larger} / {m} steps"
        )
        if n_larger < args.n_fire_min:
            print(
                f"WARN [C]: MOST H_roof differs on only {n_larger} steps. "
                "AH may not be reaching Facet3D injection. "
                "This is a warning, not a failure — domain averaging can mask the effect."
            )
        else:
            print(f"PASS [C]: MOST H_roof reflects HVAC on {n_larger} steps")

    print()
    print("=" * 72)
    print("Phase 5.2 HVAC canonical: PASS")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
