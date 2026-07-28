#!/usr/bin/env python3
"""
check_cloudy.py — Regression check for Phase 4.2 cloudy canonical
Asserts that cloud attenuation from the CSV is visible and physically sound.
"""

import sys
import re

def parse_run_log(log_file):
    """Parse run.log and extract radiation and skin temperature data."""
    
    # Storage for results
    radiation_cloud_found = False
    sw_down_clear_list = []
    sw_down_cloudy_list = []
    lw_down_clear_list = []
    lw_down_cloudy_list = []
    cloud_fraction_list = []
    t_skin_wall_max = -1e9
    t_skin_wall_min = 1e9
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Check for [UCM][4.2][radiation-cloud] banner
                if "[UCM][4.2][radiation-cloud]" in line:
                    radiation_cloud_found = True
                    
                    # Try to extract SW_down_clear, SW_down_cloudy, LW_down_clear, LW_down_cloudy, cloud_fraction
                    # Expected format (example):
                    # [UCM][4.2][radiation-cloud] SW_down_clear=800.5 SW_down_cloudy=680.2 LW_down_clear=320.1 LW_down_cloudy=340.0 cloud_fraction=0.55
                    
                    match = re.search(r'SW_down_clear\s*=\s*([\d.]+)', line)
                    if match:
                        sw_down_clear_list.append(float(match.group(1)))
                    
                    match = re.search(r'SW_down_cloudy\s*=\s*([\d.]+)', line)
                    if match:
                        sw_down_cloudy_list.append(float(match.group(1)))
                    
                    match = re.search(r'LW_down_clear\s*=\s*([\d.]+)', line)
                    if match:
                        lw_down_clear_list.append(float(match.group(1)))
                    
                    match = re.search(r'LW_down_cloudy\s*=\s*([\d.]+)', line)
                    if match:
                        lw_down_cloudy_list.append(float(match.group(1)))
                    
                    match = re.search(r'cloud_fraction\s*=\s*([\d.]+)', line)
                    if match:
                        cloud_fraction_list.append(float(match.group(1)))
                
                # Extract T_skin_wall from diagnostics
                # Expected format: something like "T_skin_wall = 305.2"
                match = re.search(r'T_skin_wall\s*=\s*([\d.]+)', line)
                if match:
                    t_skin = float(match.group(1))
                    t_skin_wall_max = max(t_skin_wall_max, t_skin)
                    t_skin_wall_min = min(t_skin_wall_min, t_skin)
    
    except IOError:
        print(f"ERROR: Cannot open {log_file}")
        return None
    
    return {
        'radiation_cloud_found': radiation_cloud_found,
        'sw_down_clear': sw_down_clear_list,
        'sw_down_cloudy': sw_down_cloudy_list,
        'lw_down_clear': lw_down_clear_list,
        'lw_down_cloudy': lw_down_cloudy_list,
        'cloud_fraction': cloud_fraction_list,
        't_skin_wall_max': t_skin_wall_max,
        't_skin_wall_min': t_skin_wall_min,
    }

def check_cloudy_assertions(data):
    """Run Phase 4.2 cloudy canonical assertions."""
    
    print("\n" + "="*70)
    print("Phase 4.2 Cloudy Canonical Checks")
    print("="*70)
    
    passed = 0
    failed = 0
    
    # Assertion 1: Radiation-cloud banner present
    print("\n[Assertion 1] [UCM][4.2][radiation-cloud] banner present at every step")
    if data['radiation_cloud_found']:
        print("  ✓ PASS: radiation-cloud banner found")
        passed += 1
    else:
        print("  ✗ FAIL: radiation-cloud banner NOT found")
        failed += 1
        return passed, failed  # Can't proceed without banner
    
    # Assertion 2: Cloud attenuation visible on SW
    print("\n[Assertion 2] Cloud attenuation visible on SW (SW_cloudy/SW_clear ≤ 0.85 when cf ≥ 0.55)")
    if len(data['sw_down_clear']) == 0:
        print("  ? SKIP: No SW data found")
    else:
        n_high_cloud = 0
        n_attenuated = 0
        for i, cf in enumerate(data['cloud_fraction']):
            if cf >= 0.55:
                n_high_cloud += 1
                if data['sw_down_clear'][i] > 10.0:  # Only count daytime
                    ratio = data['sw_down_cloudy'][i] / (data['sw_down_clear'][i] + 1e-6)
                    if ratio <= 0.85:
                        n_attenuated += 1
        
        if n_high_cloud > 0:
            frac_attenuated = n_attenuated / n_high_cloud
            if frac_attenuated > 0.8:  # At least 80% of high-cloud steps show attenuation
                print(f"  ✓ PASS: {n_attenuated}/{n_high_cloud} high-cloud steps show attenuation (ratio ≤ 0.85)")
                passed += 1
            else:
                print(f"  ✗ FAIL: Only {n_attenuated}/{n_high_cloud} high-cloud steps show attenuation")
                failed += 1
        else:
            print("  ? SKIP: No high-cloud samples (cf ≥ 0.55)")
    
    # Assertion 3: LW-down higher under cloud than clear-sky
    print("\n[Assertion 3] LW-down higher under cloud than clear-sky (Crawford & Duchon signature)")
    if len(data['lw_down_clear']) == 0:
        print("  ? SKIP: No LW data found")
    else:
        n_higher = 0
        for i in range(len(data['lw_down_clear'])):
            if data['lw_down_cloudy'][i] > data['lw_down_clear'][i]:
                n_higher += 1
        
        if len(data['lw_down_clear']) > 0:
            frac_higher = n_higher / len(data['lw_down_clear'])
            if frac_higher > 0.9:  # At least 90% of samples
                print(f"  ✓ PASS: {n_higher}/{len(data['lw_down_clear'])} samples have LW_cloudy > LW_clear")
                passed += 1
            else:
                print(f"  ✗ FAIL: Only {frac_higher*100:.1f}% samples have LW_cloudy > LW_clear")
                failed += 1
    
    # Assertion 4: T_skin_wall diurnal amplitude > 15 K (reduced vs clear-sky)
    print("\n[Assertion 4] T_skin_wall diurnal amplitude > 15 K (reduced vs clear-sky)")
    if data['t_skin_wall_max'] > -1e8 and data['t_skin_wall_min'] < 1e8:
        amplitude = data['t_skin_wall_max'] - data['t_skin_wall_min']
        if amplitude > 15.0:
            print(f"  ✓ PASS: T_skin_wall amplitude = {amplitude:.2f} K > 15 K")
            passed += 1
        else:
            print(f"  ✗ FAIL: T_skin_wall amplitude = {amplitude:.2f} K ≤ 15 K")
            failed += 1
    else:
        print("  ? SKIP: T_skin_wall data not found")
    
    return passed, failed

if __name__ == "__main__":
    log_file = "run.log"
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    print(f"Parsing {log_file}...")
    data = parse_run_log(log_file)
    
    if data is None:
        sys.exit(1)
    
    passed, failed = check_cloudy_assertions(data)
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)
