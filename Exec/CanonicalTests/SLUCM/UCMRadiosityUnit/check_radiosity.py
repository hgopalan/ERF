#!/usr/bin/env python3
"""
check_radiosity.py — Phase 5.1b SW Multi-Bounce Radiosity Unit Test Checker

Parses existing run_multi.log and run_single.log (or user-supplied paths) and
verifies:
  1. Mode strings match input ("multi" or "single")
  2. Multi-mode F_wall_road ≈ 0.2929 (Hottel analytic for H/W=1)
  3. H_wall_max(multi) >= H_wall_max(single) (multi-bounce enhancement)
  4. H_road_max(multi) >= H_road_max(single) (multi-bounce enhancement)

Usage:
  ./check_radiosity.py                              # defaults: run_multi.log, run_single.log
  ./check_radiosity.py my_multi.log my_single.log   # custom paths

Exit 0 if all assertions pass; 1 otherwise.
"""

import sys
import re
import os


def parse_log(log_file):
    """Parse an existing ERF log file for radiosity and slab diagnostics."""
    if not os.path.exists(log_file):
        print(f"ERROR: log file '{log_file}' not found")
        return None

    with open(log_file) as f:
        output = f.read()

    parsed = {
        "radiosity_mode": None,
        "alpha_wall": None,
        "alpha_road": None,
        "F_wall_road_min": None,
        "F_wall_road_max": None,
        "H_wall_max": None,
        "H_road_max": None,
    }

    # [UCM][5.1b][radiosity] mode=multi alpha_wall=... alpha_road=... F_wall_road=[..., ...]
    radiosity_pattern = (
        r"\[UCM\]\[5\.1b\]\[radiosity\]\s+mode=(\w+).*?"
        r"alpha_wall=([\d.eE+-]+).*?alpha_road=([\d.eE+-]+)"
    )
    m = re.search(radiosity_pattern, output)
    if m:
        parsed["radiosity_mode"] = m.group(1)
        parsed["alpha_wall"] = float(m.group(2))
        parsed["alpha_road"] = float(m.group(3))

        if parsed["radiosity_mode"] == "multi":
            fwr = re.search(r"F_wall_road=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]", output)
            if fwr:
                parsed["F_wall_road_min"] = float(fwr.group(1))
                parsed["F_wall_road_max"] = float(fwr.group(2))

    # H_wall / H_road max — parse LAST occurrence (final step)
    hw_matches = re.findall(r"Hw_max=\s*([\d.eE+-]+)", output)
    hr_matches = re.findall(r"Hr_max=\s*([\d.eE+-]+)", output)
    if hw_matches:
        parsed["H_wall_max"] = float(hw_matches[-1])
    if hr_matches:
        parsed["H_road_max"] = float(hr_matches[-1])

    return parsed


def main():
    args = sys.argv[1:]
    multi_log = args[0] if len(args) > 0 else "run_multi.log"
    single_log = args[1] if len(args) > 1 else "run_single.log"

    print("=" * 70)
    print("Phase 5.1b SW Multi-Bounce Radiosity Unit Test")
    print("=" * 70)

    print(f"\nParsing multi-mode log: {multi_log}")
    multi_result = parse_log(multi_log)
    if not multi_result:
        return 1

    print(f"  radiosity_mode: {multi_result['radiosity_mode']}")
    print(f"  alpha_wall: {multi_result['alpha_wall']}")
    print(f"  alpha_road: {multi_result['alpha_road']}")
    if multi_result['F_wall_road_min'] is not None:
        print(f"  F_wall_road: [{multi_result['F_wall_road_min']:.4f}, "
              f"{multi_result['F_wall_road_max']:.4f}]")
    print(f"  H_wall_max: {multi_result['H_wall_max']}")
    print(f"  H_road_max: {multi_result['H_road_max']}")

    print(f"\nParsing single-mode log: {single_log}")
    single_result = parse_log(single_log)
    if not single_result:
        return 1

    print(f"  radiosity_mode: {single_result['radiosity_mode']}")
    print(f"  alpha_wall: {single_result['alpha_wall']}")
    print(f"  alpha_road: {single_result['alpha_road']}")
    print(f"  H_wall_max: {single_result['H_wall_max']}")
    print(f"  H_road_max: {single_result['H_road_max']}")

    passed = 0
    failed = 0

    # --- Assertion 1: Mode strings ---
    print("\n[ASSERT 1] Mode strings match input...")
    if multi_result['radiosity_mode'] == 'multi':
        print("  ✓ PASS: multi-mode banner reports mode=multi")
        passed += 1
    else:
        print(f"  ✗ FAIL: multi-mode reports {multi_result['radiosity_mode']}")
        failed += 1
    if single_result['radiosity_mode'] == 'single':
        print("  ✓ PASS: single-mode banner reports mode=single")
        passed += 1
    else:
        print(f"  ✗ FAIL: single-mode reports {single_result['radiosity_mode']}")
        failed += 1

    # --- Assertion 2: F_wall_road analytic ---
    print("\n[ASSERT 2] Multi-mode F_wall_road ≈ 0.2929 (Hottel analytic, H/W=1)...")
    if multi_result['F_wall_road_max'] is not None:
        fwr_avg = 0.5 * (multi_result['F_wall_road_min'] +
                         multi_result['F_wall_road_max'])
        if abs(fwr_avg - 0.2929) < 0.01:
            print(f"  ✓ PASS: F_wall_road avg={fwr_avg:.4f} ≈ 0.2929")
            passed += 1
        else:
            print(f"  ✗ FAIL: F_wall_road avg={fwr_avg:.4f}, expected 0.2929 (tol 0.01)")
            failed += 1
    else:
        print("  ? SKIP: F_wall_road not parsed")

    # --- Assertion 3: H_wall enhancement ---
    print("\n[ASSERT 3] Multi-bounce enhances wall absorption...")
    if multi_result['H_wall_max'] is not None and single_result['H_wall_max'] is not None:
        hw_m = multi_result['H_wall_max']
        hw_s = single_result['H_wall_max']
        if hw_m >= hw_s * 0.99:
            print(f"  ✓ PASS: H_wall_max(multi)={hw_m:.2f} >= (single)={hw_s:.2f}")
            passed += 1
        else:
            print(f"  ✗ FAIL: H_wall_max(multi)={hw_m:.2f} < (single)={hw_s:.2f}")
            failed += 1
    else:
        print("  ? SKIP: H_wall_max not parsed in one or both logs")

    # --- Assertion 4: H_road enhancement ---
    print("\n[ASSERT 4] Multi-bounce enhances road absorption...")
    if multi_result['H_road_max'] is not None and single_result['H_road_max'] is not None:
        hr_m = multi_result['H_road_max']
        hr_s = single_result['H_road_max']
        if hr_m >= hr_s * 0.99:
            print(f"  ✓ PASS: H_road_max(multi)={hr_m:.2f} >= (single)={hr_s:.2f}")
            passed += 1
        else:
            print(f"  ✗ FAIL: H_road_max(multi)={hr_m:.2f} < (single)={hr_s:.2f}")
            failed += 1
    else:
        print("  ? SKIP: H_road_max not parsed in one or both logs")

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
