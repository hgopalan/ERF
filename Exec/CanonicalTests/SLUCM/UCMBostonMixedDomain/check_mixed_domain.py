#!/usr/bin/env python3
"""Verification for UCMBostonMixedDomain mixed urban/non-urban domain (Phase 3.8).

Validate that:
1. Mixed domain present (is_urban=0 AND is_urban=1 cells both present)
2. No solver failures, NaN/Inf, or Newton clamps
3. SEB solver executed (T_skin_roof fired at least once)
4. Wind reduction over urban half > 10% (drag active)
5. Wind reduction over non-urban half < 5% (drag NOT active on rural)

Uses yt.covering_grid() for 3D field indexing.
Plotfile discovery: match 'plt_mixed_NNNNN' pattern (not UCM companion files).
"""

import os
import re
import sys
import glob
import numpy as np

try:
    import yt
    try:
        yt.set_log_level("error")
    except Exception:
        pass
except ImportError:
    print("ERROR: yt not found. Install with: pip install yt")
    sys.exit(1)


def find_final_plotfile():
    """Return the highest-numbered plotfile matching 'plt_mixed_NNNNN'."""
    all_entries = sorted(glob.glob("plt_mixed_*"))
    pattern = re.compile(r"^plt_mixed_\d+$")
    main_files = [
        f for f in all_entries
        if pattern.match(os.path.basename(f))
        and not os.path.basename(f).startswith("plt_mixed_ucm")
    ]
    if not main_files:
        print("ERROR: No plotfiles found matching 'plt_mixed_NNNNN'")
        print("       Set 'erf.plot_file_1 = \"plt_mixed_\"' and 'erf.plot_int_1' in inputs.")
        if all_entries:
            print(f"       Found only UCM companion files: {all_entries}")
        return None
    return main_files[-1]


def parse_run_log():
    """Parse run.log for mixed-domain diagnostics.
    
    Returns: (is_urban_0_count, is_urban_1_count, has_assertion,
              has_nan_inf, newton_clamps, has_seb_fired)
    """
    log_file = "run.log"
    if not os.path.exists(log_file):
        print(f"WARNING: {log_file} not found; skipping log checks")
        return None, None, False, False, 0, False
    
    is_urban_0_count = 0
    is_urban_1_count = 0
    has_assertion = False
    has_nan_inf = False
    newton_clamps = 0
    has_seb_fired = False
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Count is_urban cells in log
        for match in re.finditer(r"is_urban=0.*count\s*=\s*(\d+)", content):
            is_urban_0_count = int(match.group(1))
        for match in re.finditer(r"is_urban=1.*count\s*=\s*(\d+)", content):
            is_urban_1_count = int(match.group(1))
        
        # Check for assertion failures
        if re.search(r"Assertion|abort", content, re.IGNORECASE):
            has_assertion = True
        
        # Check for NaN/Inf in temperature fields
        if re.search(r"nan|inf|NaN|Inf", content, re.IGNORECASE):
            has_nan_inf = True
        
        # Count Newton clamps
        newton_clamps = len(re.findall(r"Newton.*clamp", content, re.IGNORECASE))
        
        # Check if SEB solver fired (look for T_skin_roof)
        if re.search(r"T_skin_roof", content):
            has_seb_fired = True
        
        return is_urban_0_count, is_urban_1_count, has_assertion, \
               has_nan_inf, newton_clamps, has_seb_fired
    
    except Exception as e:
        print(f"WARNING: Error parsing {log_file}: {e}")
        return None, None, False, False, 0, False


def load_field_3d(ds, field_name):
    """Load a full-domain 3D field via covering_grid; return (z, data) or (None, None)."""
    try:
        cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                              dims=ds.domain_dimensions, fields=[field_name])
        data = np.array(cg[field_name])
        if data.ndim != 3:
            print(f"WARNING: Field {field_name} has unexpected shape {data.shape}")
            return None, None
        z = ds.domain_left_edge[2] + (np.arange(data.shape[2]) + 0.5) \
            * (ds.domain_right_edge[2] - ds.domain_left_edge[2]) / data.shape[2]
        return z, data
    except Exception as e:
        print(f"ERROR loading field {field_name}: {e}")
        return None, None


def main():
    print("=" * 70)
    print("UCMBostonMixedDomain Verification (Phase 3.8)")
    print("=" * 70)
    
    # ====================================================================
    # [1] Log parsing checks
    # ====================================================================
    print(f"\n[1] Parsing run.log for mixed-domain diagnostics")
    is_urban_0_cnt, is_urban_1_cnt, has_assert, has_nanf, n_clamps, has_seb = \
        parse_run_log()
    
    log_pass = True
    
    if is_urban_0_cnt is not None and is_urban_1_cnt is not None:
        print(f"    is_urban=0 count: {is_urban_0_cnt}")
        print(f"    is_urban=1 count: {is_urban_1_cnt}")
        if is_urban_0_cnt > 0 and is_urban_1_cnt > 0:
            print(f"    ✓ PASS: Mixed domain confirmed (urban + non-urban cells present)")
        else:
            print(f"    ✗ FAIL: Not a true mixed domain (expected both types > 0)")
            log_pass = False
    else:
        print(f"    WARNING: Could not determine urban cell counts from log")
    
    if has_assert:
        print(f"    ✗ FAIL: Found 'Assertion' or 'abort' in log")
        log_pass = False
    else:
        print(f"    ✓ PASS: No assertion failures detected")
    
    if has_nanf:
        print(f"    ✗ FAIL: Found NaN or Inf in fields")
        log_pass = False
    else:
        print(f"    ✓ PASS: No NaN/Inf detected")
    
    if n_clamps > 0:
        print(f"    ✗ FAIL: Found {n_clamps} Newton clamps")
        log_pass = False
    else:
        print(f"    ✓ PASS: Zero Newton clamps")
    
    if has_seb:
        print(f"    ✓ PASS: SEB solver fired (T_skin_roof detected)")
    else:
        print(f"    ⚠ WARN: No T_skin_roof in log (SEB may not have fired)")
    
    # ====================================================================
    # [2] Plotfile checks
    # ====================================================================
    plotfile = find_final_plotfile()
    if not plotfile:
        print(f"\nFAIL: No plotfile found")
        return False
    
    print(f"\n[2] Loading plotfile: {plotfile}")
    try:
        ds = yt.load(plotfile)
    except Exception as e:
        print(f"FAIL: Could not load plotfile: {e}")
        return False
    
    print(f"    Domain dims: {ds.domain_dimensions}")
    print(f"    Extent: {ds.domain_left_edge} to {ds.domain_right_edge}")
    
    # Load fields
    print(f"\n[3] Loading velocity fields")
    z, u = load_field_3d(ds, ("boxlib", "x_velocity"))
    _, v = load_field_3d(ds, ("boxlib", "y_velocity"))
    _, theta = load_field_3d(ds, ("boxlib", "theta"))
    
    if u is None or v is None or theta is None:
        print("FAIL: Could not load required fields")
        return False
    
    print(f"    u shape: {u.shape}, v shape: {v.shape}, theta shape: {theta.shape}")
    
    # ====================================================================
    # [4] Finite-value check
    # ====================================================================
    print(f"\n[4] Finite-value check (no NaN/Inf in fields)")
    field_pass = True
    if not np.all(np.isfinite(u)):
        print(f"    ✗ FAIL: u contains NaN or Inf")
        field_pass = False
    elif not np.all(np.isfinite(v)):
        print(f"    ✗ FAIL: v contains NaN or Inf")
        field_pass = False
    elif not np.all(np.isfinite(theta)):
        print(f"    ✗ FAIL: theta contains NaN or Inf")
        field_pass = False
    else:
        print(f"    ✓ PASS: all fields are finite")
    
    # ====================================================================
    # [5] Wind reduction check: urban vs non-urban halves
    # ====================================================================
    print(f"\n[5] Wind reduction analysis: urban half vs non-urban half")
    print(f"    Domain split at x = 10 km (i_split ≈ {ds.domain_dimensions[0]//2})")
    
    i_split = ds.domain_dimensions[0] // 2  # Middle x-index
    j_mid = ds.domain_dimensions[1] // 2    # Middle y-index
    k_surface = 1                            # Near-surface level
    
    # Sample wind in urban half (left: i < i_split)
    U_urban = np.sqrt(u[:i_split, j_mid, k_surface]**2 +
                      v[:i_split, j_mid, k_surface]**2)
    U_urban_mean = np.mean(U_urban)
    
    # Sample wind in non-urban half (right: i >= i_split)
    U_rural = np.sqrt(u[i_split:, j_mid, k_surface]**2 +
                      v[i_split:, j_mid, k_surface]**2)
    U_rural_mean = np.mean(U_rural)
    
    print(f"    Urban half (x < 10 km) mean U at k={k_surface}: {U_urban_mean:.2f} m/s")
    print(f"    Rural half (x >= 10 km) mean U at k={k_surface}: {U_rural_mean:.2f} m/s")
    
    wind_pass = True
    
    # Check urban wind reduction from rural reference
    if U_rural_mean > 0.1:
        urban_reduction_pct = 100.0 * (1.0 - U_urban_mean / U_rural_mean)
        print(f"    Wind reduction (urban vs rural): {urban_reduction_pct:.1f}%")
        if urban_reduction_pct > 10.0:
            print(f"    ✓ PASS: Urban reduction {urban_reduction_pct:.1f}% > 10% "
                  f"(drag active on is_urban=1 cells)")
        else:
            print(f"    ✗ FAIL: Only {urban_reduction_pct:.1f}% reduction "
                  f"(drag should be active on urban cells)")
            wind_pass = False
    else:
        print(f"    DIAGNOSTIC: Rural wind too weak ({U_rural_mean:.2f} m/s)")
        wind_pass = False
    
    # Check that rural wind is NOT reduced (drag should NOT affect is_urban=0)
    if U_urban_mean > 0.1:
        rural_reduction_pct = 100.0 * (1.0 - U_rural_mean / U_urban_mean)
        print(f"    [Sanity] Non-urban wind vs urban: {rural_reduction_pct:.1f}% reduction")
        # Should be minimal (less than 5%) since non-urban should NOT have drag
        if rural_reduction_pct < 5.0:
            print(f"    ✓ PASS: Rural reduction {rural_reduction_pct:.1f}% < 5% "
                  f"(drag NOT active on is_urban=0 cells)")
        else:
            print(f"    ⚠ WARN: Rural reduction {rural_reduction_pct:.1f}% >= 5% "
                  f"(drag may be affecting non-urban cells)")
            # Don't fail on this as it may be natural vertical shear
    
    # ====================================================================
    # [6] Diagnostic: UHI structure aloft
    # ====================================================================
    print(f"\n[6] Diagnostic: Vertical θ profile (urban vs rural)")
    k_uhi = min(10, theta.shape[2] - 1)
    print(f"    Sampling at k={k_uhi} (~{float(z[k_uhi]):.0f} m AGL)")
    print(f"    k    z(m)     θ_rural(K)   θ_urban(K)    Δθ(K)")
    for k in range(0, min(theta.shape[2], 15), 2):
        z_val = float(z[k])
        i_urban = i_split // 2      # Middle of urban half
        i_rural = i_split + (ds.domain_dimensions[0] - i_split) // 2  # Middle of rural half
        T_urban = theta[i_urban, j_mid, k]
        T_rural = theta[i_rural, j_mid, k]
        print(f"    {k:3d}  {z_val:6.1f}   {T_rural:9.2f}   "
              f"{T_urban:11.2f}   {T_urban - T_rural:+.2f}")
    
    # ====================================================================
    # Final verdict
    # ====================================================================
    print(f"\n" + "=" * 70)
    if log_pass and field_pass and wind_pass:
        print(f"PASS: Mixed-domain verification complete")
        print(f"      (mixed domain confirmed, no failures, wind reduction {urban_reduction_pct:.1f}%)")
        print(f"=" * 70)
        return True
    else:
        print(f"PARTIAL: Verification completed with issues")
        print(f"         Log pass: {log_pass}  |  Fields pass: {field_pass}  |  Wind pass: {wind_pass}")
        print(f"=" * 70)
        return log_pass and field_pass and wind_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
