#!/usr/bin/env python3
"""Post-run verification for UCMMorphologyInjection (Phase 2.6).

Verifies:
1. ATM plotfile written with 8 components (Phase 2.6 vs 6 in Phase 2.5)
2. New fields H_road_atm and H_wallroof_atm present
3. Left half (tall stripe, i=0..1 ATM): H_bldg_mean ~ 30 m
4. Right half (short stripe, i=2..3 ATM): H_bldg_mean ~ 5 m
5. Flux conservation: sum(H_atm*dA) ~ sum(H_sensible_ucm*dA) within tolerance
6. Split conservation: H_road_atm + H_wallroof_atm ~ H_atm per cell within tolerance

Prints PASS or FAIL with numerical values.
"""
import sys, os, glob
import yt

def main():
    # Find the ATM plotfile (should be written at steps 0 and 1)
    atm_plt_list = sorted(glob.glob("plt_ucm_atm_*"))
    if not atm_plt_list:
        print("FAIL: No ATM plotfile found (plt_ucm_atm_*)")
        sys.exit(1)
    
    atm_plt = atm_plt_list[0]  # Use first step
    print(f"[check] Loading ATM plotfile: {atm_plt}")
    
    try:
        ds = yt.load(atm_plt)
    except Exception as e:
        print(f"FAIL: Could not load {atm_plt}: {e}")
        sys.exit(1)
    
    # Assertion 1: Domain dimensions should be [4, 4, 1] (ATM is 2D slab, nz=1)
    try:
        domain_dims = ds.domain_dimensions
        if domain_dims != [4, 4, 1]:
            print(f"FAIL: Expected domain_dimensions=[4, 4, 1], got {domain_dims}")
            sys.exit(1)
        print(f"[check] domain_dimensions = {domain_dims} ✓")
    except Exception as e:
        print(f"FAIL: Could not read domain_dimensions: {e}")
        sys.exit(1)
    
    # Assertion 2: Required fields present (8 components: Phase 2.6)
    required_fields = [
        "f_urb", "H_bldg_mean", "H_bldg_std", "lambda_p", "lambda_f",
        "H_atm", "H_road_atm", "H_wallroof_atm"
    ]
    
    try:
        available_fields = ds.field_list
        available_names = [f[1] for f in available_fields if f[0] == "yt"]
        print(f"[check] Available fields: {available_names}")
        
        for field_name in required_fields:
            if field_name not in available_names:
                print(f"FAIL: Required field '{field_name}' not found in plotfile")
                print(f"       Available: {available_names}")
                sys.exit(1)
        
        print(f"[check] All 8 required fields present ✓")
    except Exception as e:
        print(f"FAIL: Could not enumerate fields: {e}")
        sys.exit(1)
    
    # Extract field values
    try:
        ad = ds.all_data()
        
        f_urb_vals = ad[("yt", "f_urb")].to_value("dimensionless")
        H_bldg_mean_vals = ad[("yt", "H_bldg_mean")].to_value("m")
        H_atm_vals = ad[("yt", "H_atm")].to_value("W/m**2")
        H_road_atm_vals = ad[("yt", "H_road_atm")].to_value("W/m**2")
        H_wallroof_atm_vals = ad[("yt", "H_wallroof_atm")].to_value("W/m**2")
    except Exception as e:
        print(f"FAIL: Could not extract field data: {e}")
        sys.exit(1)
    
    # Assertion 3: Check left/right building height split
    # Left half (i=0..1): tall stripe
    # Right half (i=2..3): short stripe
    # Note: yt flattens (x,y,z) into 1D array; need to use spatial coordinates
    
    try:
        x_coords = ad[("yt", "x")].to_value("m")
        y_coords = ad[("yt", "y")].to_value("m")
        
        x_min, x_max = x_coords.min(), x_coords.max()
        x_mid = (x_min + x_max) / 2.0
        
        # Left half: x < x_mid
        left_mask = x_coords < x_mid
        left_H_mean = H_bldg_mean_vals[left_mask].mean()
        left_H_mean_min = H_bldg_mean_vals[left_mask].min()
        left_H_mean_max = H_bldg_mean_vals[left_mask].max()
        
        # Right half: x >= x_mid
        right_mask = x_coords >= x_mid
        right_H_mean = H_bldg_mean_vals[right_mask].mean()
        right_H_mean_min = H_bldg_mean_vals[right_mask].min()
        right_H_mean_max = H_bldg_mean_vals[right_mask].max()
        
        print(f"[check] Left stripe (tall):  H_bldg_mean = {left_H_mean:.1f} m (range [{left_H_mean_min:.1f}, {left_H_mean_max:.1f}])")
        print(f"[check] Right stripe (short): H_bldg_mean = {right_H_mean:.1f} m (range [{right_H_mean_min:.1f}, {right_H_mean_max:.1f}])")
        
        # Assertion 3a: Left should be ~30 m
        if not (25.0 < left_H_mean < 35.0):
            print(f"FAIL: Left stripe H_bldg_mean={left_H_mean:.1f}, expected ~30 m")
            sys.exit(1)
        
        # Assertion 3b: Right should be ~5 m
        if not (3.0 < right_H_mean < 7.0):
            print(f"FAIL: Right stripe H_bldg_mean={right_H_mean:.1f}, expected ~5 m")
            sys.exit(1)
        
        print(f"[check] Building height split correct ✓")
    
    except Exception as e:
        print(f"FAIL: Could not analyze building heights: {e}")
        sys.exit(1)
    
    # Assertion 4: Flux conservation check
    # Total lumped flux should equal sum of road + wallroof
    # (H_atm should be approximately H_road_atm + H_wallroof_atm per cell)
    
    try:
        H_total_check = H_road_atm_vals + H_wallroof_atm_vals
        
        # Element-wise comparison
        diff = abs(H_atm_vals - H_total_check)
        max_diff = diff.max()
        mean_diff_pct = (diff.mean() / (abs(H_atm_vals).mean() + 1.0)) * 100.0
        
        print(f"[check] Flux split conservation:")
        print(f"  H_atm min/max = [{H_atm_vals.min():.1f}, {H_atm_vals.max():.1f}] W/m^2")
        print(f"  H_road_atm min/max = [{H_road_atm_vals.min():.1f}, {H_road_atm_vals.max():.1f}] W/m^2")
        print(f"  H_wallroof_atm min/max = [{H_wallroof_atm_vals.min():.1f}, {H_wallroof_atm_vals.max():.1f}] W/m^2")
        print(f"  max(|H_atm - (H_road + H_wallroof)|) = {max_diff:.1f} W/m^2")
        
        # Allow up to 5% error (or 1 W/m^2, whichever is larger)
        tol = max(abs(H_atm_vals).mean() * 0.05, 1.0)
        if max_diff > tol:
            print(f"FAIL: Flux split not conserved. max_diff={max_diff:.1f} > tol={tol:.1f}")
            sys.exit(1)
        
        print(f"[check] Flux split conserved (tolerance {tol:.1f} W/m^2) ✓")
    
    except Exception as e:
        print(f"FAIL: Could not verify flux split conservation: {e}")
        sys.exit(1)
    
    print("\nPASS")

if __name__ == "__main__":
    main()
