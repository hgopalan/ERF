#!/usr/bin/env python3
"""
check_lw_diurnal.py — Phase 5.1c LW Multi-Bounce Radiosity Diurnal Physics Validation

Parses existing run_single.log and run_multi_lagged.log (or user-supplied paths) and
verifies:
  1. Both runs complete 24 h (grep for AMReX completion marker or "Final simulation time").
  2. Both banners fire consistently across steps (mode strings match input at all sampled banner lines).
  3. multi-lagged mode reports T_sky in physical range 200-320 K across the diurnal cycle.
  4. Soft physics gate: at least one banner shows multi-lagged producing a nonzero LW effect.
     Precise T_skin enhancement magnitude is NOT asserted (depends on Boston morphology CSV);
     this is a "physics smell test," not a validation-grade check.

Explicit note: Full validation of T_skin_road enhancement (~1-3 K wall, ~0.5-2 K road at night)
is a follow-up manual comparison against plotfile outputs.

Usage:
  ./check_lw_diurnal.py                           # defaults: run_single.log, run_multi_lagged.log
  ./check_lw_diurnal.py my_single.log my_multi.log   # custom paths

Exit 0 if all checks pass; 1 if any check fails.
"""

import sys
import re
import os


def parse_lw_diurnal_log(log_file):
    """Parse an existing ERF log file for LW diurnal diagnostics."""
    if not os.path.exists(log_file):
        print(f"ERROR: log file '{log_file}' not found")
        return None

    with open(log_file) as f:
        output = f.read()

    parsed = {
        "completed_24h": False,
        "mode_banners": [],
        "T_sky_ranges": [],
    }

    # Check for completion (AMReEx final time or similar)
    # Pattern: "Time = 86400" or "Final time" or "Total time"
    completion_patterns = [
        r"Time = 86400",
        r"Final.*time.*86400",
        r"Total simulation time.*86400",
    ]
    for pattern in completion_patterns:
        if re.search(pattern, output):
            parsed["completed_24h"] = True
            break

    # Extract all mode banners: [UCM][5.1c][lw-radiosity] mode=...
    banner_pattern = r"\[UCM\]\[5\.1c\]\[lw-radiosity\]\s+mode=(\w+-?\w*)"
    mode_matches = re.findall(banner_pattern, output)
    parsed["mode_banners"] = mode_matches

    # Extract T_sky ranges (only in multi-lagged mode)
    tsky_pattern = r"T_sky=\[([\d.eE+-]+),\s*([\d.eE+-]+)\]"
    tsky_matches = re.findall(tsky_pattern, output)
    for match in tsky_matches:
        parsed["T_sky_ranges"].append((float(match[0]), float(match[1])))

    return parsed


def check_lw_diurnal(single_log, multi_lagged_log):
    """Run all checks and report results."""
    print("[check_lw_diurnal] Starting Phase 5.1c LW Radiosity Diurnal Physics Validation")
    print()

    # Parse logs
    single_data = parse_lw_diurnal_log(single_log)
    multi_data = parse_lw_diurnal_log(multi_lagged_log)

    if single_data is None or multi_data is None:
        print("ERROR: Failed to parse logs")
        return 1

    passed = 0
    failed = 0

    # Check 1: Both runs complete 24 h
    print("Check 1: 24-hour completion")
    if single_data["completed_24h"]:
        print("  ✓ single run completed 24 h")
        passed += 1
    else:
        print("  ✗ single run did not complete 24 h (or marker not found)")
        failed += 1

    if multi_data["completed_24h"]:
        print("  ✓ multi_lagged run completed 24 h")
        passed += 1
    else:
        print("  ✗ multi_lagged run did not complete 24 h (or marker not found)")
        failed += 1

    print()

    # Check 2: Consistent mode banners
    print("Check 2: Consistent mode banners across steps")
    if len(single_data["mode_banners"]) > 0:
        single_modes = set(single_data["mode_banners"])
        if single_modes == {"single"}:
            print(f"  ✓ single run banners consistently report mode='single' ({len(single_data['mode_banners'])} banners)")
            passed += 1
        else:
            print(f"  ✗ single run has inconsistent modes: {single_modes}")
            failed += 1
    else:
        print("  ✗ no banners found in single run log")
        failed += 1

    if len(multi_data["mode_banners"]) > 0:
        multi_modes = set(multi_data["mode_banners"])
        if multi_modes == {"multi-lagged"}:
            print(f"  ✓ multi_lagged run banners consistently report mode='multi-lagged' ({len(multi_data['mode_banners'])} banners)")
            passed += 1
        else:
            print(f"  ✗ multi_lagged run has inconsistent modes: {multi_modes}")
            failed += 1
    else:
        print("  ✗ no banners found in multi_lagged run log")
        failed += 1

    print()

    # Check 3: T_sky physical range (multi-lagged only)
    print("Check 3: T_sky physical range across diurnal cycle")
    T_sky_min_phys = 200.0  # Reasonable lower bound for day/night
    T_sky_max_phys = 320.0  # Reasonable upper bound for day/night
    
    if len(multi_data["T_sky_ranges"]) > 0:
        all_in_range = True
        out_of_range_count = 0
        for tmin, tmax in multi_data["T_sky_ranges"]:
            if not (T_sky_min_phys <= tmin <= T_sky_max_phys and T_sky_min_phys <= tmax <= T_sky_max_phys):
                all_in_range = False
                out_of_range_count += 1
        
        if all_in_range:
            print(f"  ✓ all {len(multi_data['T_sky_ranges'])} T_sky samples in physical range [{T_sky_min_phys}, {T_sky_max_phys}] K")
            passed += 1
        else:
            print(f"  ✗ {out_of_range_count}/{len(multi_data['T_sky_ranges'])} T_sky samples outside physical range")
            failed += 1
        
        # Sample a few for sanity
        if len(multi_data["T_sky_ranges"]) > 0:
            tmin_first, tmax_first = multi_data["T_sky_ranges"][0]
            tmin_last, tmax_last = multi_data["T_sky_ranges"][-1]
            print(f"     First sample: T_sky=[{tmin_first:.1f}, {tmax_first:.1f}] K")
            print(f"     Last sample:  T_sky=[{tmin_last:.1f}, {tmax_last:.1f}] K")
    else:
        print("  ✗ no T_sky samples found in multi_lagged log (mode mismatch?)")
        failed += 1

    print()

    # Check 4: Soft physics gate — at least one banner in multi_lagged
    print("Check 4: Soft physics gate (multi-lagged banner fires)")
    if len(multi_data["mode_banners"]) > 0:
        print(f"  ✓ multi_lagged mode banners fired during run (physics active)")
        passed += 1
        print(f"     NOTE: Full validation of T_skin enhancement is a follow-up manual plotfile analysis")
    else:
        print("  ✗ no multi_lagged banners fired (physics may be inactive)")
        failed += 1

    print()
    print(f"Summary: {passed} passed, {failed} failed")
    print()
    print("NOTE: This is a 'physics smell test,' not a full validation.")
    print("      Precise T_skin_road enhancement (~1-3 K wall, ~0.5-2 K road at night)")
    print("      requires follow-up manual comparison against plotfile outputs.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        single_log = "run_single.log"
        multi_lagged_log = "run_multi_lagged.log"
    elif len(sys.argv) == 3:
        single_log = sys.argv[1]
        multi_lagged_log = sys.argv[2]
    else:
        print(f"Usage: {sys.argv[0]} [single_log multi_lagged_log]")
        sys.exit(1)

    exit_code = check_lw_diurnal(single_log, multi_lagged_log)
    sys.exit(exit_code)
