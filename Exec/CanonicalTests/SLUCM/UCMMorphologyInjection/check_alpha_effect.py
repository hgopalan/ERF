#!/usr/bin/env python3
"""Alpha effect verification for UCMMorphologyInjection (Phase 2.6).

Extracts RhoTheta profiles above tall (left, i=0) vs short (right, i=3) ATM cells
at the final step and compares vertical penetration.

Expected: Tall building column should have measurable heat perturbation at higher
altitudes (e.g., z > 15 m) compared to short building column.

This script is optional for initial validation. If yt/analysis is problematic,
just print the arrays and eyeball the difference.
"""
import sys, os, glob
import yt
import numpy as np

def main():
    # Find main ATM plotfile at final step (should be step 1 for max_step=2)
    atm_plt_list = sorted(glob.glob("plt_0000*"))
    if not atm_plt_list:
        print("[alpha_effect] No main ATM plotfile found (plt_0000*)")
        print("[alpha_effect] (This is expected if only UCM diagnostic output was written)")
        return
    
    atm_plt = atm_plt_list[-1]  # Use last step
    print(f"[alpha_effect] Loading ATM plotfile: {atm_plt}")
    
    try:
        ds = yt.load(atm_plt)
    except Exception as e:
        print(f"[alpha_effect] Could not load {atm_plt}: {e}")
        print("[alpha_effect] (Continuing; this step is optional)")
        return
    
    try:
        # Get domain extent and cell count
        domain_dims = ds.domain_dimensions
        nx, ny, nz = domain_dims
        
        print(f"[alpha_effect] Main ATM grid: {nx} x {ny} x {nz}")
        
        # Extract RhoTheta field
        ad = ds.all_data()
        rho_theta = ad[("yt", "RhoTheta")].to_value("kg*K/m**3")
        z_phys = ad[("yt", "z")].to_value("m")
        x_coords = ad[("yt", "x")].to_value("m")
        
        # Find domain extent
        x_min, x_max = x_coords.min(), x_coords.max()
        z_min, z_max = z_phys.min(), z_phys.max()
        
        print(f"[alpha_effect] Domain: x ∈ [{x_min:.1f}, {x_max:.1f}] m, z ∈ [{z_min:.1f}, {z_max:.1f}] m")
        
        # Identify "left" cell (x near x_min): tall buildings
        # Identify "right" cell (x near x_max): short buildings
        x_mid = (x_min + x_max) / 2.0
        
        # Extract vertical columns
        left_mask = (x_coords < x_mid)
        right_mask = (x_coords >= x_mid)
        
        left_rho_theta = rho_theta[left_mask]
        left_z = z_phys[left_mask]
        
        right_rho_theta = rho_theta[right_mask]
        right_z = z_phys[right_mask]
        
        # Sort by z and compute column averages
        left_sorted_idx = np.argsort(left_z)
        left_z_sorted = left_z[left_sorted_idx]
        left_rho_theta_sorted = left_rho_theta[left_sorted_idx]
        
        right_sorted_idx = np.argsort(right_z)
        right_z_sorted = right_z[right_sorted_idx]
        right_rho_theta_sorted = right_rho_theta[right_sorted_idx]
        
        # Print some diagnostics
        print(f"\n[alpha_effect] Tall stripe (left, x < {x_mid:.1f}):")
        print(f"  RhoTheta range: [{left_rho_theta.min():.1f}, {left_rho_theta.max():.1f}] kg*K/m^3")
        print(f"  z range: [{left_z.min():.1f}, {left_z.max():.1f}] m")
        
        print(f"\n[alpha_effect] Short stripe (right, x >= {x_mid:.1f}):")
        print(f"  RhoTheta range: [{right_rho_theta.min():.1f}, {right_rho_theta.max():.1f}] kg*K/m^3")
        print(f"  z range: [{right_z.min():.1f}, {right_z.max():.1f}] m")
        
        # Rough check: find cells above z=15m and see if tall stripe is warmer
        z_threshold = 15.0
        left_above_threshold = left_rho_theta[left_z > z_threshold]
        right_above_threshold = right_rho_theta[right_z > z_threshold]
        
        if len(left_above_threshold) > 0 and len(right_above_threshold) > 0:
            left_mean_high = left_above_threshold.mean()
            right_mean_high = right_above_threshold.mean()
            print(f"\n[alpha_effect] Above z={z_threshold}m:")
            print(f"  Tall stripe RhoTheta mean: {left_mean_high:.1f} kg*K/m^3")
            print(f"  Short stripe RhoTheta mean: {right_mean_high:.1f} kg*K/m^3")
            
            # Reference: should be similar to background, but tall might be slightly warmer due to injection
            print(f"  Difference (tall - short): {left_mean_high - right_mean_high:.2f} kg*K/m^3")
            print(f"[alpha_effect] (Note: difference should be ≥0 if tall stripe injects deeper)")
        
        print("\n[alpha_effect] Optional check complete")
    
    except Exception as e:
        print(f"[alpha_effect] Error during analysis: {e}")
        print("[alpha_effect] (Continuing; this step is optional)")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
