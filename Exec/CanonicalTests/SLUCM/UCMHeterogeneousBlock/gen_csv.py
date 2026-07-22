#!/usr/bin/env python3
"""Generate building_layout.csv and materials.csv for UCMHeterogeneousBlock test.

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

print(f"[UCMHeterogeneousBlock] Regenerating CSV files "
      f"({NX_UCM}x{NY_UCM} checkerboard)...")

# Generate checkerboard pattern with alternating materials
base = uniform_urban(H_bldg=10.0, plan_frac=0.5)
cell_fn = checkerboard_materials(base, mat_a=1, mat_b=2)

write_layout("building_layout.csv", NX_UCM, NY_UCM, cell_fn)

# Write materials
write_materials(
    "materials.csv",
    [
        dict(
            mat_id=1,
            name="Material_A",
            albedo=0.20,
            emissivity=0.90,
            k_therm_W_per_mK=1.5,
            rho_cp_J_per_m3K=2.0e6,
            thickness_m=0.3,
            description="First urban material",
        ),
        dict(
            mat_id=2,
            name="Material_B",
            albedo=0.25,
            emissivity=0.92,
            k_therm_W_per_mK=1.6,
            rho_cp_J_per_m3K=2.1e6,
            thickness_m=0.35,
            description="Second urban material for heterogeneous testing",
        ),
    ],
)

print("[UCMHeterogeneousBlock] Done.")
