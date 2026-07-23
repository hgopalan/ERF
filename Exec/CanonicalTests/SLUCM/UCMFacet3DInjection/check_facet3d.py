#!/usr/bin/env python3
"""Post-run verification for UCMFacet3DInjection (Phase 2.7).

Verifies:
1. ATM plotfile written with 9 components (Phase 2.7: split wall+roof from Phase 2.6)
2. Fields H_wall_atm and H_roof_atm present (NO H_wallroof_atm)
3. Left half (tall stripe, i=0..1 ATM): H_bldg_mean ~ 30 m
4. Right half (short stripe, i=2..3 ATM): H_bldg_mean ~ 5 m
5. Flux conservation: H_atm ≈ H_road_atm*(1-lambda_p) + H_wall_atm*lambda_f + H_roof_atm*lambda_p within 5%
6. (Optional) Vertical extent: wall injection extends higher over tall buildings

Prints PASS or FAIL with numerical values.

References:
- Martilli, Clappier & Rotach (2002): BEP geometric overlap
- Phase 2.6 check_injection.py: template for verification structure
"""
import sys, os, glob
import numpy as np

try:
    import yt
    yt.set_log_level(50)  # Suppress yt verbosity
except ImportError:
    print("FAIL: yt module not found. Install with: pip install yt")
    sys.exit(1)

def main():
    # Find the ATM plotfile (should be written at steps 0 and 1)
    atm_plt_list = sorted(glob.glob("plt_ucm_atm_*"))
    if not atm_plt_list:
        print("FAIL: No ATM plotfile found (plt_ucm_atm_*)")
        sys.exit(1)
    
    atm_plt = atm_plt_list[0]  # Use first step
    print(f"[check_facet3d] Loading ATM plotfile: {atm_plt}")
    
    try:
        ds = yt.load(atm_plt)
        ds.index  # Force indexing
    except Exception as e:
        print(f"FAIL: Could not load {atm_plt}: {e}")
        sys.exit(1)
    
    # Assertion 1: Domain dimensions should be [4, 4, 1] (ATM is 2D slab, nz=1)
    try:
        domain_dims = list(ds.domain_dimensions)
        if domain_dims != [4, 4, 1]:
            print(f"FAIL: Expected domain_dimensions=[4, 4, 1], got {domain_dims}")
            sys.exit(1)
        print(f"[check_facet3d] domain_dimensions = {domain_dims} ✓")
    except Exception as e:
        print(f"FAIL: Could not read domain_dimensions: {e}")
        sys.exit(1)
    
    # Assertion 2: Required fields present (9 components for Phase 2.7)
    required_fields = [
        "f_urb", "H_bldg_mean", "H_bldg_std", "lambda_p", "lambda_f",
        "H_atm", "H_road_atm", "H_wall_atm", "H_roof_atm"
    ]
    
    # Phase 2.7: H_wallroof_atm should NOT be present (split into H_wall_atm and H_roof_atm)
    forbidden_fields = ["H_wallroof_atm"]
    
    try:
        ds.index  # Ensure index is populated
        available_fields = ds.field_list
        available_names = [f[1] for f in available_fields if f[0] == "boxlib"]
        print(f"[check_facet3d] Available fields: {available_names}")
        print(f"[check_facet3d] Total components: {len(available_names)}")
        
        for field_name in required_fields:
            if field_name not in available_names:
                print(f"FAIL: Required field '{field_name}' not found in plotfile")
                print(f"       Available: {available_names}")
                sys.exit(1)
        
        for forbidden in forbidden_fields:
            if forbidden in available_names:
                print(f"FAIL: Old Phase 2.6 field '{forbidden}' found (should be split into H_wall_atm and H_roof_atm)")
                sys.exit(1)
        
        if len(available_names) != 9:
            print(f"WARNING: Expected 9 fields, got {len(available_names)}. Continuing anyway...")
        
        print(f"[check_facet3d] All 9 required fields present ✓")
    except Exception as e:
        print(f"FAIL: Could not enumerate fields: {e}")
        sys.exit(1)
    
    # Extract field values using numpy arrays (avoid .to_value on dimensionless)
    try:
        ad = ds.all_data()
        
        f_urb_vals = np.array(ad[("boxlib", "f_urb")])
        H_bldg_mean_vals = np.array(ad[("boxlib", "H_bldg_mean")])
        H_bldg_std_vals = np.array(ad[("boxlib", "H_bldg_std")])
        lambda_p_vals = np.array(ad[("boxlib", "lambda_p")])
        lambda_f_vals = np.array(ad[("boxlib", "lambda_f")])
        H_atm_vals = np.array(ad[("boxlib", "H_atm")])
        H_road_atm_vals = np.array(ad[("boxlib", "H_road_atm")])
        H_wall_atm_vals = np.array(ad[("boxlib", "H_wall_atm")])
        H_roof_atm_vals = np.array(ad[("boxlib", "H_roof_atm")])
    except Exception as e:
        print(f"FAIL: Could not extract field data: {e}")
        sys.exit(1)
    
    # Assertion 3: Check left/right building height split
    # Left half (i=0..1): tall stripe (should be ~30 m)
    # Right half (i=2..3): short stripe (should be ~5 m)
    try:
        # Assuming ATM grid is 4x4x1 in (i,j,k)
        # We need to flatten and check left vs right halves
        left_half_heights = H_bldg_mean_vals[:2, :, :]
        right_half_heights = H_bldg_mean_vals[2:4, :, :]
        
        left_mean = np.mean(left_half_heights)
        right_mean = np.mean(right_half_heights)
        
        print(f"[check_facet3d] Left stripe (i=0..1):  H_bldg_mean ≈ {left_mean:.2f} m (expect ~30)")
        print(f"[check_facet3d] Right stripe (i=2..3): H_bldg_mean ≈ {right_mean:.2f} m (expect ~5)")
        
        if left_mean < 25 or left_mean > 35:
            print(f"FAIL: Left stripe height {left_mean:.2f} m outside expected range [25, 35]")
            sys.exit(1)
        
        if right_mean < 2 or right_mean > 8:
            print(f"FAIL: Right stripe height {right_mean:.2f} m outside expected range [2, 8]")
            sys.exit(1)
        
        print(f"[check_facet3d] Height split verification ✓")
    except Exception as e:
        print(f"FAIL: Could not verify height split: {e}")
        sys.exit(1)
    
    # Assertion 4: Flux conservation: H_atm ≈ H_road_atm*(1-lambda_p) + H_wall_atm*lambda_f + H_roof_atm*lambda_p
    try:
        # Compute reconstructed flux (BEP conservation rule for Phase 2.7)
        H_reconstructed = (H_road_atm_vals * (1.0 - lambda_p_vals) +
                          H_wall_atm_vals * lambda_f_vals +
                          H_roof_atm_vals * lambda_p_vals)
        
        # Compare with lumped H_atm
        # Allow 5% relative error tolerance
        tolerance = 0.05
        
        # Avoid division by zero: only check cells where H_atm is significant
        significant = np.abs(H_atm_vals) > 1e-6
        
        if np.any(significant):
            relative_error = np.abs(H_reconstructed[significant] - H_atm_vals[significant]) / np.abs(H_atm_vals[significant])
            max_error = np.max(relative_error)
            mean_error = np.mean(relative_error)
            
            print(f"[check_facet3d] Flux conservation check:")
            print(f"  Mean relative error: {mean_error*100:.2f}%")
            print(f"  Max relative error:  {max_error*100:.2f}%")
            
            if max_error > tolerance:
                print(f"FAIL: Flux conservation error {max_error*100:.2f}% exceeds tolerance {tolerance*100:.1f}%")
                # Print some debug info
                for idx in np.argwhere(relative_error > tolerance)[:3]:  # Show first 3 violations
                    i, j, k = idx
                    print(f"  Cell ({i},{j},{k}): H_atm={H_atm_vals[i,j,k]:.2f}, H_recon={H_reconstructed[i,j,k]:.2f}")
                sys.exit(1)
            
            print(f"[check_facet3d] Flux conservation ✓ (error within {tolerance*100:.1f}%)")
        else:
            print(f"[check_facet3d] No significant fluxes to check, skipping conservation test")
    except Exception as e:
        print(f"FAIL: Could not verify flux conservation: {e}")
        sys.exit(1)
    
    # Assertion 5: Phase 2.7 specific - no H_wallroof_atm should exist
    try:
        for field_name in forbidden_fields:
            if field_name in available_names:
                print(f"FAIL: Old Phase 2.6 field '{field_name}' still present")
                sys.exit(1)
        print(f"[check_facet3d] Phase 2.7 field split verified (H_wallroof_atm absent) ✓")
    except Exception as e:
        print(f"FAIL: Could not verify Phase 2.7 structure: {e}")
        sys.exit(1)
    
    print("\nPASS: All Phase 2.7 checks passed ✓")
    sys.exit(0)

if __name__ == "__main__":
    main()
