#!/usr/bin/env python3
"""
check_view_factors.py — Regression check for Phase 5.1a view-factor computation

For H_bldg = W_road = 20 m (r=1, d=sqrt(2)):
  F_wall_sky   ≈ 0.2929
  F_wall_wall  ≈ 0.4142
  F_wall_road  ≈ 0.2929
  F_road_sky   ≈ 0.4142
  F_road_wall  ≈ 0.2929
"""

import sys
import re
import math

TOL = 1e-3

EXPECTED = {
    'F_wall_sky':   0.5 * (1 + 1 - math.sqrt(2)),
    'F_wall_wall':  math.sqrt(2) - 1,
    'F_wall_road':  0.5 * (1 + 1 - math.sqrt(2)),
    'F_road_sky':   math.sqrt(2) - 1,
    'F_road_wall':  0.5 * (1 - (math.sqrt(2) - 1)),
}


def parse(log_file):
    ranges = {}
    with open(log_file) as f:
        for line in f:
            if '[UCM][5.1a][BANNER]' in line:
                continue
            m = re.search(
                r'(F_wall_sky|F_wall_wall|F_wall_road|F_road_sky|F_road_wall)'
                r'\s+min=([-\d.eE+]+)\s+max=([-\d.eE+]+)',
                line,
            )
            if m:
                ranges[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return ranges


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else 'run.log'
    ranges = parse(log)

    print('\n' + '=' * 70)
    print('Phase 5.1a View-Factor Unit Test')
    print('=' * 70)

    passed = 0
    failed = 0

    for name, (lo, hi) in ranges.items():
        expected = EXPECTED[name]
        if abs(lo - expected) < TOL and abs(hi - expected) < TOL:
            print(f'  ✓ PASS: {name} = [{lo:.4f}, {hi:.4f}], expected {expected:.4f}')
            passed += 1
        else:
            print(f'  ✗ FAIL: {name} = [{lo:.4f}, {hi:.4f}], expected {expected:.4f}')
            failed += 1

    # Closure identity: F_wall_sky + F_wall_wall + F_wall_road == 1
    if all(k in ranges for k in ('F_wall_sky', 'F_wall_wall', 'F_wall_road')):
        total = ranges['F_wall_sky'][0] + ranges['F_wall_wall'][0] + ranges['F_wall_road'][0]
        if abs(total - 1.0) < TOL:
            print(f'  ✓ PASS: wall closure: sum = {total:.6f}')
            passed += 1
        else:
            print(f'  ✗ FAIL: wall closure violated: sum = {total:.6f}')
            failed += 1

    # Closure identity: F_road_sky + 2 * F_road_wall == 1
    if 'F_road_sky' in ranges and 'F_road_wall' in ranges:
        total = ranges['F_road_sky'][0] + 2 * ranges['F_road_wall'][0]
        if abs(total - 1.0) < TOL:
            print(f'  ✓ PASS: road closure: sum = {total:.6f}')
            passed += 1
        else:
            print(f'  ✗ FAIL: road closure violated: sum = {total:.6f}')
            failed += 1

    print('=' * 70)
    print(f'Results: {passed} passed, {failed} failed')
    print('=' * 70)
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
