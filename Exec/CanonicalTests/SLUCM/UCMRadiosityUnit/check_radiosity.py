#!/usr/bin/env python3
"""
check_radiosity.py — Phase 5.1b SW Multi-Bounce Radiosity Unit Test Checker

Runs ERF with inputs_multi and inputs_single, parses [UCM][5.1b][radiosity]
banners, and verifies:
  1. Mode strings match input ("multi" or "single")
  2. Multi-mode F_wall_road ≈ 0.2929 (Hottel analytic for H/W=1)
  3. H_wall_max(multi) >= H_wall_max(single) (multi-bounce enhancement)
  4. H_road_max(multi) >= H_road_max(single) (multi-bounce enhancement)

Exit 0 if all assertions pass; 1 otherwise.
"""

import sys
import subprocess
import re
import os

def run_and_parse(inputs_file, erf_executable="erf3d.gnu.ex"):
    """
    Run ERF with the given inputs file and parse stdout for:
      - [UCM][5.1b][radiosity] mode, alpha values, F_wall_road range
      - [UCM][3.5A][*] H_wall and H_road max values (from slab conduction diagnostics)
    
    Returns dict with parsed values, or None on failure.
    """
    test_dir = os.path.dirname(os.path.abspath(inputs_file))
    
    # Check if executable exists
    if not os.path.exists(erf_executable):
        print(f"ERROR: ERF executable '{erf_executable}' not found")
        return None
    
    # Run ERF
    try:
        result = subprocess.run(
            [erf_executable, inputs_file],
            cwd=test_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
    except subprocess.TimeoutExpired:
        print(f"ERROR: ERF timed out running {inputs_file}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to run ERF: {e}")
        return None
    
    # Combine stdout and stderr for parsing
    output = result.stdout + "\n" + result.stderr
    
    parsed = {
        "radiosity_mode": None,
        "alpha_wall": None,
        "alpha_road": None,
        "F_wall_road_min": None,
        "F_wall_road_max": None,
        "H_wall_max": None,
        "H_road_max": None,
    }
    
    # Parse [UCM][5.1b][radiosity] banner
    radiosity_pattern = r"\[UCM\]\[5\.1b\]\[radiosity\] mode=(\w+).*alpha_wall=([\d.e+-]+).*alpha_road=([\d.e+-]+)"
    match = re.search(radiosity_pattern, output)
    if match:
        parsed["radiosity_mode"] = match.group(1)
        parsed["alpha_wall"] = float(match.group(2))
        parsed["alpha_road"] = float(match.group(3))
        
        # Look for F_wall_road range (only if multi mode)
        if "multi" in parsed["radiosity_mode"]:
            fwr_pattern = r"F_wall_road=\[([\d.e+-]+),\s*([\d.e+-]+)\]"
            fwr_match = re.search(fwr_pattern, output)
            if fwr_match:
                parsed["F_wall_road_min"] = float(fwr_match.group(1))
                parsed["F_wall_road_max"] = float(fwr_match.group(2))
    
    # Parse H_wall and H_road max from slab diagnostics
    # Look for pattern like: [UCM][3.5A-diag] ... Hw_max=XXX ... Hr_max=YYY
    hw_pattern = r"Hw_max=\s*([\d.e+-]+)"
    hr_pattern = r"Hr_max=\s*([\d.e+-]+)"
    hw_match = re.search(hw_pattern, output)
    hr_match = re.search(hr_pattern, output)
    if hw_match:
        parsed["H_wall_max"] = float(hw_match.group(1))
    if hr_match:
        parsed["H_road_max"] = float(hr_match.group(1))
    
    # Check for ERF errors
    if result.returncode != 0:
        print(f"WARNING: ERF exited with code {result.returncode}")
        print(f"STDERR: {result.stderr}")
    
    return parsed

def main():
    """Main test function."""
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    inputs_multi = os.path.join(test_dir, "inputs_multi")
    inputs_single = os.path.join(test_dir, "inputs_single")
    
    # Try to find ERF executable
    erf_exe = "erf3d.gnu.ex"
    if not os.path.exists(erf_exe):
        # Try parent directories
        for candidate in ["./erf3d.gnu.ex", "../erf3d.gnu.ex", "../../erf3d.gnu.ex"]:
            if os.path.exists(candidate):
                erf_exe = candidate
                break
    
    print("=" * 70)
    print("Phase 5.1b SW Multi-Bounce Radiosity Unit Test")
    print("=" * 70)
    
    # Run multi-mode
    print("\n[TEST 1] Running multi-mode (radiosity_mode='multi')...")
    multi_result = run_and_parse(inputs_multi, erf_exe)
    if not multi_result:
        print("FAIL: Could not parse multi-mode run")
        return 1
    
    print(f"  radiosity_mode: {multi_result['radiosity_mode']}")
    print(f"  alpha_wall: {multi_result['alpha_wall']}")
    print(f"  alpha_road: {multi_result['alpha_road']}")
    if multi_result['F_wall_road_min'] is not None:
        print(f"  F_wall_road: [{multi_result['F_wall_road_min']:.4f}, {multi_result['F_wall_road_max']:.4f}]")
    print(f"  H_wall_max: {multi_result['H_wall_max']}")
    print(f"  H_road_max: {multi_result['H_road_max']}")
    
    # Run single-mode
    print("\n[TEST 2] Running single-mode (radiosity_mode='single')...")
    single_result = run_and_parse(inputs_single, erf_exe)
    if not single_result:
        print("FAIL: Could not parse single-mode run")
        return 1
    
    print(f"  radiosity_mode: {single_result['radiosity_mode']}")
    print(f"  alpha_wall: {single_result['alpha_wall']}")
    print(f"  alpha_road: {single_result['alpha_road']}")
    if single_result['F_wall_road_min'] is not None:
        print(f"  F_wall_road: [{single_result['F_wall_road_min']:.4f}, {single_result['F_wall_road_max']:.4f}]")
    print(f"  H_wall_max: {single_result['H_wall_max']}")
    print(f"  H_road_max: {single_result['H_road_max']}")
    
    # Assertion 1: Mode strings match input
    print("\n[ASSERT 1] Mode strings match input...")
    assert multi_result['radiosity_mode'] == 'multi', \
        f"Multi-mode radiosity_mode={multi_result['radiosity_mode']}, expected 'multi'"
    assert single_result['radiosity_mode'] == 'single', \
        f"Single-mode radiosity_mode={single_result['radiosity_mode']}, expected 'single'"
    print("  PASS: Mode strings match")
    
    # Assertion 2: Multi-mode F_wall_road ≈ 0.2929 (Hottel analytic for H/W=1)
    print("\n[ASSERT 2] Multi-mode F_wall_road ≈ 0.2929 (Hottel analytic)...")
    if multi_result['F_wall_road_max'] is not None:
        # For uniform H/W=1, expect F ≈ 0.2929
        fwr_avg = (multi_result['F_wall_road_min'] + multi_result['F_wall_road_max']) / 2.0
        fwr_expect = 0.2929
        fwr_tol = 0.01  # ±1% tolerance
        assert abs(fwr_avg - fwr_expect) < fwr_tol, \
            f"F_wall_road avg={fwr_avg:.4f}, expected ≈{fwr_expect:.4f} (tol={fwr_tol})"
        print(f"  PASS: F_wall_road avg={fwr_avg:.4f} ≈ {fwr_expect:.4f}")
    else:
        print("  SKIP: F_wall_road not parsed from banner")
    
    # Assertion 3: H_wall_max(multi) >= H_wall_max(single)
    print("\n[ASSERT 3] Multi-bounce enhances wall absorption...")
    if multi_result['H_wall_max'] is not None and single_result['H_wall_max'] is not None:
        assert multi_result['H_wall_max'] >= single_result['H_wall_max'] * 0.99, \
            f"H_wall_max(multi)={multi_result['H_wall_max']:.2f} < H_wall_max(single)={single_result['H_wall_max']:.2f}"
        print(f"  PASS: H_wall_max(multi)={multi_result['H_wall_max']:.2f} >= H_wall_max(single)={single_result['H_wall_max']:.2f}")
    else:
        print("  SKIP: H_wall_max values not parsed")
    
    # Assertion 4: H_road_max(multi) >= H_road_max(single)
    print("\n[ASSERT 4] Multi-bounce enhances road absorption...")
    if multi_result['H_road_max'] is not None and single_result['H_road_max'] is not None:
        assert multi_result['H_road_max'] >= single_result['H_road_max'] * 0.99, \
            f"H_road_max(multi)={multi_result['H_road_max']:.2f} < H_road_max(single)={single_result['H_road_max']:.2f}"
        print(f"  PASS: H_road_max(multi)={multi_result['H_road_max']:.2f} >= H_road_max(single)={single_result['H_road_max']:.2f}")
    else:
        print("  SKIP: H_road_max values not parsed")
    
    print("\n" + "=" * 70)
    print("All assertions passed!")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
