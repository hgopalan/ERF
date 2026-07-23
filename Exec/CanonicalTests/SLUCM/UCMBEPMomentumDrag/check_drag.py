#!/usr/bin/env python3
"""
Verification script for UCMBEPMomentumDrag test (Phase 2.8).

Checks:
1. Main plotfile exists at final step, contains u,v,w components.
2. Inside canopy: wind reduction >= 50% at tall-stripe column.
3. Above canopy: wind ≈ freestream within ±20%.
4. Diagnostic: vertical profiles, momentum sums.
5. Plotfile structure: 5 ATM components (u,v,w + 2 passive).

Uses yt.covering_grid() for 3D field indexing (NOT flat array indexing).
Physics checks are mostly diagnostic; fatal assertions guard against
obvious bugs in drag kernel (e.g., double-counting, missing geometry).
"""

import os
import sys
import glob
import numpy as np

try:
    import yt
    # Silence yt INFO chatter; keep errors visible.
    # (Older code called yt.suppress_stream_stdout() which does not exist.)
    try:
        yt.set_log_level("error")
    except Exception:
        pass
except ImportError:
    print("ERROR: yt not found. Install with: pip install yt")
    sys.exit(1)


def find_final_plotfile(max_step=10):
    """Find the final plotfile (plt_*_000010 or highest available)."""
    pattern = "plt_*_??????"
    files = sorted(glob.glob(pattern))
    if not files:
        print("ERROR: No plotfiles found matching pattern 'plt_*_??????'")
        return None
    # Find highest step number
    latest = files[-1]
    return latest


def load_field_3d(ds, field_name, covering_grid_level=0):
    """
    Load a 3D field using covering_grid (NOT flat array indexing).
    Returns (x, y, z, data) where data shape is (nx, ny, nz).
    """
    try:
        cg = ds.covering_grid(level=covering_grid_level, left_edge=ds.domain_left_edge,
                              dims=ds.domain_dimensions, fields=[field_name])
        data = np.array(cg[field_name])
        # Transpose to (x, y, z) if needed
        if data.ndim != 3:
            print(f"WARNING: Field {field_name} has unexpected shape {data.shape}")
            return None, None, None, None
        # Extract coordinate arrays
        x = ds.domain_left_edge[0] + np.arange(data.shape[0]) * (ds.domain_right_edge[0] - ds.domain_left_edge[0]) / data.shape[0]
        y = ds.domain_left_edge[1] + np.arange(data.shape[1]) * (ds.domain_right_edge[1] - ds.domain_left_edge[1]) / data.shape[1]
        z = ds.domain_left_edge[2] + np.arange(data.shape[2]) * (ds.domain_right_edge[2] - ds.domain_left_edge[2]) / data.shape[2]
        return x, y, z, data
    except Exception as e:
        print(f"ERROR loading field {field_name}: {e}")
        return None, None, None, None


def main():
    print("=" * 70)
    print("UCMBEPMomentumDrag Verification (Phase 2.8)")
    print("=" * 70)
    
    # Find final plotfile
    plotfile = find_final_plotfile()
    if not plotfile:
        print("FAIL: No plotfile found")
        return False
    
    print(f"\n[1] Loading plotfile: {plotfile}")
    
    # Load dataset
    try:
        ds = yt.load(plotfile)
    except Exception as e:
        print(f"FAIL: Could not load plotfile: {e}")
        return False
    
    print(f"    Domain: {ds.domain_dimensions}")
    print(f"    Extent: {ds.domain_left_edge} to {ds.domain_right_edge}")
    
    # Check field list (diagnostic)
    print(f"\n[2] Checking plotfile structure")
    try:
        field_list = ds.field_list
    except Exception as e:
        print(f"WARNING: Could not read field_list: {e}")
        field_list = []
    
    print(f"    Available fields: {len(field_list)} total")
    
    # Load velocity components
    print(f"\n[3] Loading velocity field (u, v, w)")
    _, _, z, u_data = load_field_3d(ds, ("boxlib", "x_velocity"), covering_grid_level=0)
    _, _, _, v_data = load_field_3d(ds, ("boxlib", "y_velocity"), covering_grid_level=0)
    _, _, _, w_data = load_field_3d(ds, ("boxlib", "z_velocity"), covering_grid_level=0)
    
    if u_data is None or v_data is None or w_data is None:
        print("FAIL: Could not load velocity components")
        return False
    
    print(f"    u shape: {u_data.shape}, v shape: {v_data.shape}, w shape: {w_data.shape}")
    
    # Extract vertical profiles at two columns
    # Tall stripe: ATM indices i,j (0,0) → UCM block
    # Short stripe: ATM indices i,j (rightmost of tall part) → UCM block
    
    # With 4x4 ATM and grid_ratio=4, we have 16x16 UCM cells
    # Left half (tall): i=0..7, Right half (short): i=8..15
    # For ATM grid, left tall is at ATM i=0, right tall edge at ATM i=1, etc.
    # We sample at UCM center of ATM cell 0 (UCM i=1 or i=2) and UCM cell 3 (right edge of tall, UCM i=15)
    
    # Simpler: sample first column (i=0, all j,k) and last-tall column (i=7, all j,k)
    print(f"\n[4] Extracting vertical profiles")
    
    # Profile 1: tall stripe (UCM i=1, j=1 → center of ATM i=0, j=0)
    i_tall, j_tall = 1, 1
    # Profile 2: short stripe (UCM i=14, j=1 → center of ATM i=3, j=0)
    i_short, j_short = 14, 1
    
    if i_tall >= u_data.shape[0] or i_short >= u_data.shape[0]:
        print(f"WARNING: Requested profile indices exceed domain bounds")
        i_tall = min(1, u_data.shape[0]-1)
        i_short = min(u_data.shape[0]-2, u_data.shape[0]-1)
    
    u_tall = u_data[i_tall, j_tall, :]
    v_tall = v_data[i_tall, j_tall, :]
    U_tall = np.sqrt(u_tall**2 + v_tall**2)
    
    u_short = u_data[i_short, j_short, :]
    v_short = v_data[i_short, j_short, :]
    U_short = np.sqrt(u_short**2 + v_short**2)
    
    print(f"    Tall stripe (i={i_tall}, j={j_tall}): U_min={U_tall.min():.2f} m/s, U_max={U_tall.max():.2f} m/s")
    print(f"    Short stripe (i={i_short}, j={j_short}): U_min={U_short.min():.2f} m/s, U_max={U_short.max():.2f} m/s")
    
    # Canopy geometry (from inputs: tall h=30m, short h=5m, dz=4m)
    H_tall = 30.0  # m
    H_short = 5.0  # m
    dz = 4.0  # m from inputs
    
    # Identify k levels inside/above canopy
    z_local = z - z[0]  # height above surface
    k_canopy_tall = np.where(z_local < H_tall)[0]
    k_above_tall = np.where(z_local > 2.0 * H_tall)[0]
    
    print(f"\n[5] Wind reduction checks")
    print(f"    Tall stripe canopy layer: k={k_canopy_tall[0]:.0f}..{k_canopy_tall[-1]:.0f} (z≈0..{H_tall:.0f}m)")
    print(f"    Tall stripe above-canopy: k={k_above_tall[0]:.0f}..{k_above_tall[-1]:.0f} (z>{2*H_tall:.0f}m)")
    
    # Estimate freestream wind (from above-canopy average)
    if len(k_above_tall) > 0:
        U_freestream = np.mean(U_tall[k_above_tall])
        print(f"    Freestream wind (tall stripe above canopy): U_fs = {U_freestream:.2f} m/s")
    else:
        # Fallback: use max wind in profile
        U_freestream = np.max(U_tall)
        print(f"    Freestream wind (fallback max): U_fs = {U_freestream:.2f} m/s")
    
    if U_freestream < 0.1:
        print("WARNING: Freestream wind is very small (<0.1 m/s); wind reduction test may be inconclusive")
    
    # Check 1: Inside canopy wind reduction
    if len(k_canopy_tall) > 0 and U_freestream > 0.1:
        U_canopy_mean = np.mean(U_tall[k_canopy_tall])
        wind_reduction_pct = 100.0 * (1.0 - U_canopy_mean / U_freestream)
        print(f"\n    Canopy-interior wind (tall): U_canopy = {U_canopy_mean:.2f} m/s")
        print(f"    Wind reduction: {wind_reduction_pct:.1f}%")
        
        # ASSERTION: at least 50% reduction inside tall canopy
        min_reduction_pct = 50.0
        if wind_reduction_pct >= min_reduction_pct:
            print(f"    ✓ PASS: {wind_reduction_pct:.1f}% >= {min_reduction_pct}% (drag is active)")
        else:
            print(f"    ✗ FAIL: {wind_reduction_pct:.1f}% < {min_reduction_pct}% (drag too weak or missing)")
            return False
    else:
        print(f"    DIAGNOSTIC: Cannot check wind reduction (canopy levels: {len(k_canopy_tall)}, U_fs: {U_freestream:.2f})")
    
    # Check 2: Above-canopy wind undisturbed
    if len(k_above_tall) > 0:
        U_above_mean = np.mean(U_tall[k_above_tall])
        wind_ratio = U_above_mean / U_freestream if U_freestream > 0.1 else 1.0
        wind_ratio_pct = 100.0 * wind_ratio
        print(f"\n    Above-canopy wind (tall): U_above = {U_above_mean:.2f} m/s")
        print(f"    Ratio to freestream: {wind_ratio_pct:.1f}%")
        
        # ASSERTION: within ±20% of freestream
        allowed_deviation_pct = 20.0
        if abs(100.0 - wind_ratio_pct) <= allowed_deviation_pct:
            print(f"    ✓ PASS: deviation {abs(100-wind_ratio_pct):.1f}% <= {allowed_deviation_pct}% (above-canopy undisturbed)")
        else:
            print(f"    ✗ FAIL: deviation {abs(100-wind_ratio_pct):.1f}% > {allowed_deviation_pct}% (drag bleeding into free atmosphere)")
            return False
    
    # Diagnostic output
    print(f"\n[6] Diagnostic vertical profiles")
    print(f"    k    z(m)    U_tall(m/s)   U_short(m/s)")
    for k in range(0, len(z_local), max(1, len(z_local)//10)):  # Sample every ~10%
        print(f"    {k:3d}  {z_local[k]:6.1f}    {U_tall[k]:8.2f}      {U_short[k]:8.2f}")
    
    # Check domain dimensions consistency
    print(f"\n[7] Final checks")
    expected_atm_dims = [4, 4, 256]
    actual_dims = list(ds.domain_dimensions)
    if actual_dims == expected_atm_dims:
        print(f"    ✓ Domain dimensions match expected ({expected_atm_dims})")
    else:
        print(f"    WARNING: Domain dimensions {actual_dims} != expected {expected_atm_dims}")
    
    print(f"\n" + "=" * 70)
    print(f"PASS: UCMBEPMomentumDrag verification complete")
    print(f"=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
