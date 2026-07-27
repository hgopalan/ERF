#!/usr/bin/env python3
"""Generate UCMBoston canonical layout (Phase 3.7: physical-coordinate format).

Boston-stylized concentric layout (Chebyshev/L-infinity distance from center):
  d = max(|i - 39.5|, |j - 39.5|)

Ring definitions (in 80×80 UCM grid):
  d = 0..7    (inner)   Downtown core        (Financial District style)     λ_p=0.55, H=100 m
  d = 8..15             Dense mid-rise       (Back Bay / Beacon Hill style)  λ_p=0.50, H=40 m
  d = 16..24            Residential dense    (South End / Cambridge style)  λ_p=0.35, H=15 m
  d = 25..32            Residential sparse   (Somerville / Brookline style)λ_p=0.20, H=8 m
  d = 33..39  (outer)   Suburban / rural     (Newton / outer metro style)  λ_p=0.05, H=5 m

Y-direction uniform (periodic BC).

Rationale: Boston UHI validation requires concentric rings mirroring the
actual urban morphology: high-rise downtown core, mid-rise mid-zone, dense
and sparse residential rings, and outer suburban transition. This is a
synthetic layout suitable for Phase 2.11 baseline validation.

Phase 3.7: Wrapper around gen_building_layout.py with boston_5zone preset.
"""

import os
import sys
import subprocess

# Domain: 20 km × 20 km, 80×80 UCM cells
PROB_LO = [0.0, 0.0]
PROB_HI = [20000.0, 20000.0]
N_UCM = [80, 80]
LAYOUT = "boston_5zone"
OUTPUT = "building_layout.csv"

# Materials CSV (unchanged, still generated separately)
def write_materials():
    """Write materials.csv (unchanged from Phase 2.x)."""
    import csv
    materials = [
        dict(
            mat_id=1,
            name="glass_steel_downtown",
            albedo=0.30,
            emissivity=0.85,
            k_therm_W_per_mK=50.0,
            rho_cp_J_per_m3K=3.5e6,
            thickness_m=0.10,
            description="Curtain-wall high-rise (downtown core)",
        ),
        dict(
            mat_id=2,
            name="brick_concrete",
            albedo=0.20,
            emissivity=0.90,
            k_therm_W_per_mK=1.1,
            rho_cp_J_per_m3K=1.8e6,
            thickness_m=0.30,
            description="Standard mid-rise and dense residential",
        ),
        dict(
            mat_id=3,
            name="wood_vinyl_residential",
            albedo=0.25,
            emissivity=0.92,
            k_therm_W_per_mK=0.15,
            rho_cp_J_per_m3K=1.2e6,
            thickness_m=0.15,
            description="Suburban residential wood/vinyl siding",
        ),
        dict(
            mat_id=4,
            name="grassland_rural",
            albedo=0.20,
            emissivity=0.95,
            k_therm_W_per_mK=0.3,
            rho_cp_J_per_m3K=1.4e6,
            thickness_m=0.10,
            description="Rural surrogate (reserved for future is_urban=0 use)",
        ),
    ]
    
    header = ['mat_id', 'name', 'albedo', 'emissivity',
              'k_therm_W_per_mK', 'rho_cp_J_per_m3K', 'thickness_m', 'description']
    
    with open("materials.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for m in materials:
            w.writerow(m)
    
    print(f"[gen_boston] wrote {len(materials)} materials to materials.csv")


def main():
    """Generate building_layout.csv using the new gen_building_layout.py tool."""
    # Find gen_building_layout.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gen_tool = os.path.join(os.path.dirname(script_dir), "scripts", "gen_building_layout.py")
    
    if not os.path.exists(gen_tool):
        print(f"[ERROR] gen_building_layout.py not found at {gen_tool}", file=sys.stderr)
        sys.exit(1)
    
    # Call gen_building_layout.py with boston_5zone preset
    cmd = [
        sys.executable, gen_tool,
        "--prob-lo", str(PROB_LO[0]), str(PROB_LO[1]),
        "--prob-hi", str(PROB_HI[0]), str(PROB_HI[1]),
        "--n-ucm", str(N_UCM[0]), str(N_UCM[1]),
        "--output", OUTPUT,
        "--layout", LAYOUT,
    ]
    
    print(f"[gen_boston] Calling: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"[ERROR] gen_building_layout.py failed with return code {result.returncode}",
              file=sys.stderr)
        sys.exit(1)
    
    # Generate materials.csv
    write_materials()


if __name__ == "__main__":
    main()

