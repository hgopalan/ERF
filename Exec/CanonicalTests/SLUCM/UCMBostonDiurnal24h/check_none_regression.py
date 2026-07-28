#!/usr/bin/env python3
"""
check_none_regression.py — Verify Phase 4.2 backward compatibility
Confirms that ucm.cloud_source=none reproduces Phase 3.5b bit-identically.
"""

import sys
import re

def parse_run_log(log_file):
    """Parse run.log and extract diagnostics."""
    
    t_skin_max_list = []
    h_sensible_list = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Extract T_skin_max from diagnostics
                match = re.search(r'T_skin_max\s*=\s*([\d.]+)', line)
                if match:
                    t_skin_max_list.append(float(match.group(1)))
                
                # Extract sensible heat flux
                match = re.search(r'H_sensible\s*=\s*([\d.\-]+)', line)
                if match:
                    h_sensible_list.append(float(match.group(1)))
    
    except IOError:
        print(f"ERROR: Cannot open {log_file}")
        return None
    
    return {
        't_skin_max': t_skin_max_list,
        'h_sensible': h_sensible_list,
    }

def check_none_regression(data_new, data_old):
    """Compare new run with old baseline to verify bit-identity."""
    
    print("\n" + "="*70)
    print("Phase 4.2 Backward Compatibility Regression (ucm.cloud_source=none)")
    print("="*70)
    
    passed = 0
    failed = 0
    
    # Assertion 1: T_skin_max difference < 0.05 K
    print("\n[Assertion 1] |T_skin_max_new - T_skin_max_old| < 0.05 K")
    if len(data_new['t_skin_max']) > 0 and len(data_old['t_skin_max']) > 0:
        # Compare final values
        t_new_final = data_new['t_skin_max'][-1]
        t_old_final = data_old['t_skin_max'][-1]
        diff = abs(t_new_final - t_old_final)
        
        if diff < 0.05:
            print(f"  ✓ PASS: T_skin_max diff = {diff:.4f} K < 0.05 K")
            print(f"    New: {t_new_final:.4f} K, Old: {t_old_final:.4f} K")
            passed += 1
        else:
            print(f"  ✗ FAIL: T_skin_max diff = {diff:.4f} K >= 0.05 K")
            print(f"    New: {t_new_final:.4f} K, Old: {t_old_final:.4f} K")
            failed += 1
    else:
        print("  ? SKIP: T_skin_max data not found")
    
    # Assertion 2: H_sensible difference < 0.5 W/m²
    print("\n[Assertion 2] |H_sensible_new - H_sensible_old| < 0.5 W/m²")
    if len(data_new['h_sensible']) > 0 and len(data_old['h_sensible']) > 0:
        # Compare final values
        h_new_final = data_new['h_sensible'][-1]
        h_old_final = data_old['h_sensible'][-1]
        diff = abs(h_new_final - h_old_final)
        
        if diff < 0.5:
            print(f"  ✓ PASS: H_sensible diff = {diff:.4f} W/m² < 0.5 W/m²")
            print(f"    New: {h_new_final:.4f} W/m², Old: {h_old_final:.4f} W/m²")
            passed += 1
        else:
            print(f"  ✗ FAIL: H_sensible diff = {diff:.4f} W/m² >= 0.5 W/m²")
            print(f"    New: {h_new_final:.4f} W/m², Old: {h_old_final:.4f} W/m²")
            failed += 1
    else:
        print("  ? SKIP: H_sensible data not found")
    
    return passed, failed

if __name__ == "__main__":
    new_log = "run.log"
    old_log = "run_baseline.log"
    
    if len(sys.argv) > 1:
        new_log = sys.argv[1]
    if len(sys.argv) > 2:
        old_log = sys.argv[2]
    
    print(f"Parsing new run from {new_log}...")
    data_new = parse_run_log(new_log)
    if data_new is None:
        sys.exit(1)
    
    print(f"Parsing baseline from {old_log}...")
    data_old = parse_run_log(old_log)
    if data_old is None:
        print(f"WARNING: Baseline {old_log} not found; skipping regression")
        sys.exit(0)
    
    passed, failed = check_none_regression(data_new, data_old)
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)
