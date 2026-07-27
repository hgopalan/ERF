#!/usr/bin/env python3
"""
Phase 3.5c Diurnal 24-Hour Regression Check

Parses run.log from a completed UCMBostonDiurnal24h run and validates
that the full-loop physics remains stable over a 24-hour diurnal cycle.

Usage:
    python3 check_diurnal_24h.py [path_to_run.log]

Default: ./run.log

Exit codes:
    0 = PASS (all metrics met)
    1 = FAIL (one or more metrics violated)
"""

import re
import sys
from pathlib import Path

# Thresholds (documented in README.md)
CLAMP_TOTAL_MAX = 0            # Zero clamps allowed over entire run
DIVERGE_TOTAL_MAX = 0          # Zero divergences allowed
THETA_WARN_TOTAL_MAX = 100     # Small tolerance for transient injection warnings
T_SKIN_ROOF_DAYTIME_MIN = 305  # K, must exceed at some point (daytime warming)
T_SKIN_ROOF_NIGHTTIME_MAX = 290  # K, must drop below at some point (nighttime cooling)
T_SLAB_ROOF_MIN = 280          # K, must stay above
T_SLAB_ROOF_MAX = 305          # K, must stay below
UHI_DAYTIME_MIN = 2.0          # K, T_canyon - T_atm must exceed this at daytime peak


def parse_log(log_path):
    """Extract all validation metrics from run.log."""
    text = Path(log_path).read_text()

    # Metrics 1 & 2: clamps and divergences
    clamp_re = re.compile(
        r"Clamped to T_skin_min=\d+K:\s+roof=(\d+)\s+wall=(\d+)\s+road=(\d+)"
    )
    diverge_re = re.compile(
        r"Newton diverged \(hit max_iter\):\s+roof=(\d+)\s+wall=(\d+)\s+road=(\d+)"
    )
    total_clamps = 0
    total_diverges = 0
    for m in clamp_re.finditer(text):
        total_clamps += int(m.group(1)) + int(m.group(2)) + int(m.group(3))
    for m in diverge_re.finditer(text):
        total_diverges += int(m.group(1)) + int(m.group(2)) + int(m.group(3))

    # Metric 3: theta_tend warnings
    theta_warn_re = re.compile(r"theta_tend.*exceeded")
    total_theta_warns = len(theta_warn_re.findall(text))

    # Metric 4 & 5: T_skin_roof extremes
    t_skin_roof_re = re.compile(
        r"T_skin_roof=\[([\d.]+),([\d.]+)\]\s+K"
    )
    t_roof_mins = []
    t_roof_maxs = []
    for m in t_skin_roof_re.finditer(text):
        t_roof_mins.append(float(m.group(1)))
        t_roof_maxs.append(float(m.group(2)))
    t_roof_max_ever = max(t_roof_maxs) if t_roof_maxs else 0.0
    t_roof_min_ever = min(t_roof_mins) if t_roof_mins else 999.0

    # Metric 6: T_slab_roof[0] extremes
    t_slab_re = re.compile(
        r"T_slab_roof\[0\]=\[([\d.]+),([\d.]+)\]\s+K"
    )
    t_slab_mins = []
    t_slab_maxs = []
    for m in t_slab_re.finditer(text):
        t_slab_mins.append(float(m.group(1)))
        t_slab_maxs.append(float(m.group(2)))
    t_slab_min_ever = min(t_slab_mins) if t_slab_mins else 999.0
    t_slab_max_ever = max(t_slab_maxs) if t_slab_maxs else 0.0

    # Metric 7: UHI signal
    canyon_re = re.compile(r"T_canyon_air=\[[\d.]+,([\d.]+)\]\s+K")
    tatm_re = re.compile(r"T_atm_ucm min=[\d.]+\s+max=([\d.]+)")
    canyon_maxes = [float(m.group(1)) for m in canyon_re.finditer(text)]
    tatm_maxes = [float(m.group(1)) for m in tatm_re.finditer(text)]
    max_uhi = 0.0
    for c, a in zip(canyon_maxes, tatm_maxes):
        max_uhi = max(max_uhi, c - a)

    return {
        "total_clamps": total_clamps,
        "total_diverges": total_diverges,
        "total_theta_warns": total_theta_warns,
        "t_roof_max_ever": t_roof_max_ever,
        "t_roof_min_ever": t_roof_min_ever,
        "t_slab_min_ever": t_slab_min_ever,
        "t_slab_max_ever": t_slab_max_ever,
        "max_uhi": max_uhi,
    }


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "run.log"
    if not Path(log_path).exists():
        print(f"FAIL: log file not found: {log_path}")
        return 1

    m = parse_log(log_path)
    print("Phase 3.5c Diurnal 24h Regression -- parsed metrics:")
    print(f"  Total clamps:           {m['total_clamps']}")
    print(f"  Total diverges:         {m['total_diverges']}")
    print(f"  Total theta warnings:   {m['total_theta_warns']}")
    print(f"  T_skin_roof max ever:   {m['t_roof_max_ever']:.2f} K")
    print(f"  T_skin_roof min ever:   {m['t_roof_min_ever']:.2f} K")
    print(f"  T_slab_roof[0] range:   [{m['t_slab_min_ever']:.2f}, {m['t_slab_max_ever']:.2f}] K")
    print(f"  Max UHI (canyon-atm):   {m['max_uhi']:.2f} K")
    print()

    failures = []
    if m["total_clamps"] > CLAMP_TOTAL_MAX:
        failures.append(f"total_clamps={m['total_clamps']} > {CLAMP_TOTAL_MAX}")
    if m["total_diverges"] > DIVERGE_TOTAL_MAX:
        failures.append(f"total_diverges={m['total_diverges']} > {DIVERGE_TOTAL_MAX}")
    if m["total_theta_warns"] > THETA_WARN_TOTAL_MAX:
        failures.append(f"total_theta_warns={m['total_theta_warns']} > {THETA_WARN_TOTAL_MAX}")
    if m["t_roof_max_ever"] < T_SKIN_ROOF_DAYTIME_MIN:
        failures.append(
            f"t_roof_max_ever={m['t_roof_max_ever']:.2f} < "
            f"{T_SKIN_ROOF_DAYTIME_MIN} (no daytime warming)"
        )
    if m["t_roof_min_ever"] > T_SKIN_ROOF_NIGHTTIME_MAX:
        failures.append(
            f"t_roof_min_ever={m['t_roof_min_ever']:.2f} > "
            f"{T_SKIN_ROOF_NIGHTTIME_MAX} (no nighttime cooling)"
        )
    if m["t_slab_min_ever"] < T_SLAB_ROOF_MIN:
        failures.append(
            f"t_slab_min_ever={m['t_slab_min_ever']:.2f} < "
            f"{T_SLAB_ROOF_MIN} (slab froze)"
        )
    if m["t_slab_max_ever"] > T_SLAB_ROOF_MAX:
        failures.append(
            f"t_slab_max_ever={m['t_slab_max_ever']:.2f} > "
            f"{T_SLAB_ROOF_MAX} (slab overheated)"
        )
    if m["max_uhi"] < UHI_DAYTIME_MIN:
        failures.append(
            f"max_uhi={m['max_uhi']:.2f} < {UHI_DAYTIME_MIN} (no UHI signal)"
        )

    if failures:
        print("FAIL -- validation errors:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASS -- all metrics met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
