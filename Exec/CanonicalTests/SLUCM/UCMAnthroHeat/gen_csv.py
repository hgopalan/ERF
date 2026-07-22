#!/usr/bin/env python3
"""Generate building_layout.csv and materials.csv for UCMHeterogeneousMaterials test.

This script regenerates the CSV files using the ERF-SLUCM Phase 2.9 toolchain.
Run this script instead of hand-editing the CSV files.

Usage:
  python3 gen_csv.py
"""
import sys
import os

# Add tools directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir = os.path.join(script_dir, "..", "tools")
tools_dir = os.path.normpath(tools_dir)
sys.path.insert(0, tools_dir)

from ucm_csv import write_layout, write_materials
from ucm_generators import uniform_urban, checkerboard_materials

# Configuration for this test
NX_UCM = 16
NY_UCM = 16

print(f"[UCMHeterogeneousMaterials] Regenerating CSV files "
      f"({NX_UCM}x{NY_UCM} checkerboard with contrast materials)...")

# Generate checkerboard pattern with alternating materials
base = uniform_urban(H_bldg=10.0, plan_frac=0.5)
cell_fn = checkerboard_materials(base, mat_a=1, mat_b=2)

write_layout("building_layout.csv", NX_UCM, NY_UCM, cell_fn)

# Write materials with high albedo contrast for testing
write_materials(
    "materials.csv",
    [
        dict(
            mat_id=1,
            name="White_Roof",
            albedo=0.6,
            emissivity=0.9,
            k_therm_W_per_mK=0.2,
            rho_cp_J_per_m3K=5.0e5,
            thickness_m=0.3,
            description="High-albedo insulating roof",
        ),
        dict(
            mat_id=2,
            name="Dark_Asphalt",
            albedo=0.1,
            emissivity=0.9,
            k_therm_W_per_mK=2.0,
            rho_cp_J_per_m3K=3.0e6,
            thickness_m=0.3,
            description="Low-albedo dense pavement",
        ),
    ],
)

print("[UCMHeterogeneousMaterials] Done.")
