#!/usr/bin/env python3
"""
check_view_factors_heterogeneous.py — Phase 5.1a regression on heterogeneous grid

Complements the uniform UCMViewFactorsUnit canonical. On a per-cell heterogeneous
grid, we can't assert against a single scalar; instead we verify:
  1. Field ranges are non-degenerate (min < max) — proves per-cell computation.
  2. All values in [0, 1] — physical bounds.
  3. Symmetry identities across fields: F_wall_sky range == F_wall_road range,
     F_road_sky range == F_wall_wall range.
  4. Analytic-formula endpoint match: reverse-solve r from min/max of F_wall_sky,
     verify F_wall_wall / F_road_sky ranges are consistent with the same r.
"""

import sys
import re
import math


def parse(log_file):
    ranges = {}
    with open(log_file) as f:
        for line in f:
            m = re.search(
                r'(F_wall_sky|F_wall_wall|F_wall_road|F_road_sky|F_road_wall|F_roof_sky)'
                r'\s+min=([-\d.eE+]+)\s+max=([-\d.eE+]+)',
                line,
            )
            if m:
                ranges[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return ranges


def solve_aspect_ratio_from_wall_sky(F):
    """Reverse-solve r from F_wall_sky = 0.5 * (1 + r - sqrt(1 + r^2))"""
    # F = 0.5*(1 + r - sqrt(1+r^2))
    # 2F - 1 - r = -sqrt(1+r^2)
    # Solve: r^2 - 2(1-2F)r - (1-2F)^2 ... use numerical solve
    from scipy.optimize import brentq
    def eq(r): return 0.5 * (1 + r - math.sqrt(1 + r * r)) - F
    return brentq(eq, 0.001, 100.0)


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else 'run.log'
    ranges = parse(log)

    print('\n' + '=' * 70)
    print('Phase 5.1a View-Factor Heterogeneous Regression')
    print('=' * 70)

    passed = 0
    failed = 0

    # --- Test 1: Non-degenerate ranges (per-cell heterogeneity is real) ---
    print('\n[Test 1] Field ranges non-degenerate (min < max, proves per-cell)')
    for name, (lo, hi) in ranges.items():
        if name == 'F_roof_sky':
            # Roof always = 1
            if lo == hi == 1.0:
                print(f'  ✓ PASS: {name} = [{lo}, {hi}] (uniform 1.0 by design)')
                passed += 1
            else:
                print(f'  ✗ FAIL: {name} = [{lo}, {hi}], expected [1.0, 1.0]')
                failed += 1
        else:
            if hi - lo > 0.01:
                print(f'  ✓ PASS: {name} = [{lo:.4f}, {hi:.4f}], span = {hi-lo:.4f}')
                passed += 1
            else:
                print(f'  ✗ FAIL: {name} = [{lo:.4f}, {hi:.4f}], span too small ({hi-lo:.4f})')
                failed += 1

    # --- Test 2: Physical bounds [0, 1] ---
    print('\n[Test 2] All values in [0, 1] (physical bounds)')
    all_ok = True
    for name, (lo, hi) in ranges.items():
        if lo < -1e-6 or hi > 1.0 + 1e-6:
            print(f'  ✗ FAIL: {name} = [{lo}, {hi}] violates [0, 1]')
            all_ok = False
    if all_ok:
        print('  ✓ PASS: all fields in [0, 1]')
        passed += 1
    else:
        failed += 1

    # --- Test 3: Symmetry F_wall_sky == F_wall_road, F_road_sky == F_wall_wall ---
    print('\n[Test 3] Analytic symmetries')
    ws = ranges.get('F_wall_sky')
    wr = ranges.get('F_wall_road')
    rs = ranges.get('F_road_sky')
    ww = ranges.get('F_wall_wall')
    if ws and wr:
        if abs(ws[0] - wr[0]) < 1e-4 and abs(ws[1] - wr[1]) < 1e-4:
            print(f'  ✓ PASS: F_wall_sky range == F_wall_road range')
            passed += 1
        else:
            print(f'  ✗ FAIL: F_wall_sky {ws} != F_wall_road {wr}')
            failed += 1
    if rs and ww:
        if abs(rs[0] - ww[0]) < 1e-4 and abs(rs[1] - ww[1]) < 1e-4:
            print(f'  ✓ PASS: F_road_sky range == F_wall_wall range')
            passed += 1
        else:
            print(f'  ✗ FAIL: F_road_sky {rs} != F_wall_wall {ww}')
            failed += 1

    # --- Test 4: Aspect-ratio consistency ---
    # From F_wall_sky min/max, solve for r range, then verify F_wall_wall lies within
    # the corresponding predicted range.
    print('\n[Test 4] Cross-field aspect-ratio consistency')
    if ws and ww:
        try:
            # F_wall_sky is DECREASING in r → min corresponds to max r
            r_max = solve_aspect_ratio_from_wall_sky(ws[0])
            r_min = solve_aspect_ratio_from_wall_sky(ws[1])
            # F_wall_wall = sqrt(1+r^2) - r; INCREASING in r
            ww_predicted_min = math.sqrt(1 + r_min**2) - r_min
            ww_predicted_max = math.sqrt(1 + r_max**2) - r_max
            if abs(ww[0] - ww_predicted_min) < 0.01 and abs(ww[1] - ww_predicted_max) < 0.01:
                print(f'  ✓ PASS: F_wall_wall range consistent with r ∈ [{r_min:.2f}, {r_max:.2f}]')
                print(f'         predicted [{ww_predicted_min:.4f}, {ww_predicted_max:.4f}],'
                      f' observed {ww}')
                passed += 1
            else:
                print(f'  ✗ FAIL: F_wall_wall inconsistent')
                print(f'         r ∈ [{r_min:.2f}, {r_max:.2f}] → predicted'
                      f' [{ww_predicted_min:.4f}, {ww_predicted_max:.4f}], observed {ww}')
                failed += 1
        except ImportError:
            print('  ? SKIP: scipy not available (needed for reverse-solve)')

    print('=' * 70)
    print(f'Results: {passed} passed, {failed} failed')
    print('=' * 70)
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
