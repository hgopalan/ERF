#!/usr/bin/env python3
"""
check_lw_radiosity.py — Phase 5.1c LW Multi-Bounce Radiosity Unit Test Checker

Parses existing run_multi_lagged.log and run_single.log (or user-supplied paths) and
verifies:
  1. Mode strings match input ("multi-lagged" or "single")
  2. Multi-lagged banner reports T_sky in physically reasonable range (200-320 K for Boston overnight)
  3. Multi-lagged banner reports emissivities matching input (eps_wall=0.90, eps_road=0.94)

Usage:
  ./check_lw_radiosity.py                                 # defaults: run_multi_lagged.log, run_single.log
  ./check_lw_radiosity.py my_multi_lagged.log my_single.log   # custom paths

Exit 0 if all assertions pass; 1 otherwise.
"""

import sys
import re
import os


def parse_lw_radiosity_log(log_file):
    """Parse an existing ERF log file for LW radiosity diagnostics."""
    if not os.path.exists(log_file):
        print(f"ERROR: log file '{log_file}' not found")
        return None

    with open(log_file) as f:
        output = f.read()

    parsed = {
        "lw_radiosity_mode": None,
        "eps_wall": None,
        "eps_road": None,
        "T_sky_min": None,
        "T_sky_max": None,
    }

    # [UCM][5.1c][lw-radiosity] mode=... eps_wall=... eps_road=... [T_sky=[..., ...]]
    lw_radiosity_pattern = (
        r"\[UCM\]\[5\.1c\]\[lw-radiosity\]\s+mode=(\w+-?\w*)"
        r".*?eps_wall=([\d.eE+-]+).*?eps_road=([\d.eE+-]+)"
    )
    m = re.search(lw_radiosity_pattern, output)
    if m:
        parsed["lw_radiosity_mode"] = m.group(1)
        parsed["eps_wall"] = float(m.group(2))
        parsed["eps_road"] = float(m.group(3))

    # T_sky=[min, max] (only present in multi-lagged mode)
    tsky_pattern = r"T_sky=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
    m = re.search(tsky_pattern, output)
    if m:
        parsed["T_sky_min"] = float(m.group(1))
        parsed["T_sky_max"] = float(m.group(2))

    return parsed


def check_lw_radiosity(multi_lagged_log, single_log):
    """Run all checks and report results."""
    print("[check_lw_radiosity] Starting Phase 5.1c LW Radiosity Unit Test validation")
    print()

    # Parse logs
    multi_lagged_data = parse_lw_radiosity_log(multi_lagged_log)
    single_data = parse_lw_radiosity_log(single_log)

    passed = 0
    failed = 0

    # Check 1: Mode strings
    print("Check 1: Mode strings in banners")
    if multi_lagged_data and multi_lagged_data["lw_radiosity_mode"] == "multi-lagged":
        print("  ✓ multi_lagged log contains mode='multi-lagged'")
        passed += 1
    else:
        print(f"  ✗ multi_lagged log mode mismatch: got {multi_lagged_data.get('lw_radiosity_mode') if multi_lagged_data else 'PARSE FAILED'}")
        failed += 1

    if single_data and single_data["lw_radiosity_mode"] == "single":
        print("  ✓ single log contains mode='single'")
        passed += 1
    else:
        print(f"  ✗ single log mode mismatch: got {single_data.get('lw_radiosity_mode') if single_data else 'PARSE FAILED'}")
        failed += 1

    print()

    # Check 2: Emissivities
    print("Check 2: Emissivities in multi-lagged banner")
    eps_wall_expected = 0.90
    eps_road_expected = 0.94
    if multi_lagged_data:
        if multi_lagged_data["eps_wall"] is not None:
            if abs(multi_lagged_data["eps_wall"] - eps_wall_expected) < 0.001:
                print(f"  ✓ eps_wall={multi_lagged_data['eps_wall']:.6f} ≈ {eps_wall_expected}")
                passed += 1
            else:
                print(f"  ✗ eps_wall={multi_lagged_data['eps_wall']:.6f} != {eps_wall_expected}")
                failed += 1
        if multi_lagged_data["eps_road"] is not None:
            if abs(multi_lagged_data["eps_road"] - eps_road_expected) < 0.001:
                print(f"  ✓ eps_road={multi_lagged_data['eps_road']:.6f} ≈ {eps_road_expected}")
                passed += 1
            else:
                print(f"  ✗ eps_road={multi_lagged_data['eps_road']:.6f} != {eps_road_expected}")
                failed += 1

    print()

    # Check 3: T_sky range (multi-lagged only)
    print("Check 3: T_sky physical range (multi-lagged only)")
    T_sky_min_phys = 200.0  # Reasonable lower bound for night
    T_sky_max_phys = 320.0  # Reasonable upper bound for night
    if multi_lagged_data and multi_lagged_data["T_sky_min"] is not None:
        if T_sky_min_phys <= multi_lagged_data["T_sky_min"] <= T_sky_max_phys:
            print(f"  ✓ T_sky_min={multi_lagged_data['T_sky_min']:.2f} K in range [{T_sky_min_phys}, {T_sky_max_phys}] K")
            passed += 1
        else:
            print(f"  ✗ T_sky_min={multi_lagged_data['T_sky_min']:.2f} K outside range")
            failed += 1
    if multi_lagged_data and multi_lagged_data["T_sky_max"] is not None:
        if T_sky_min_phys <= multi_lagged_data["T_sky_max"] <= T_sky_max_phys:
            print(f"  ✓ T_sky_max={multi_lagged_data['T_sky_max']:.2f} K in range [{T_sky_min_phys}, {T_sky_max_phys}] K")
            passed += 1
        else:
            print(f"  ✗ T_sky_max={multi_lagged_data['T_sky_max']:.2f} K outside range")
            failed += 1

    # Single mode should NOT have T_sky in banner
    if single_data and single_data["T_sky_min"] is None:
        print(f"  ✓ single log does not report T_sky (as expected)")
        passed += 1
    elif single_data and single_data["T_sky_min"] is not None:
        print(f"  ✗ single log unexpectedly reports T_sky={single_data['T_sky_min']:.2f}")
        failed += 1

    print()
    print(f"Summary: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        multi_lagged_log = "run_multi_lagged.log"
        single_log = "run_single.log"
    elif len(sys.argv) == 3:
        multi_lagged_log = sys.argv[1]
        single_log = sys.argv[2]
    else:
        print(f"Usage: {sys.argv[0]} [multi_lagged_log single_log]")
        sys.exit(1)

    exit_code = check_lw_radiosity(multi_lagged_log, single_log)
    sys.exit(exit_code)
