#!/usr/bin/env python3
"""Generate a uniform urban grid for testing synthetic patterns.

This is a simple example that generates a homogeneous urban domain.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ucm_csv import write_layout, write_materials
from ucm_generators import uniform_urban

NX_ATM, NY_ATM, GRID_RATIO = 8, 8, 2
NX_UCM, NY_UCM = NX_ATM * GRID_RATIO, NY_ATM * GRID_RATIO

print(f"[gen_uniform] generating {NX_UCM}x{NY_UCM} grid...")
write_layout("building_layout.csv", NX_UCM, NY_UCM, uniform_urban())
write_materials("materials.csv", [
    dict(mat_id=1, name="concrete", albedo=0.20, emissivity=0.90,
         k_therm_W_per_mK=1.5, rho_cp_J_per_m3K=2.0e6,
         thickness_m=0.3, description="generic urban"),
])
print("[gen_uniform] done.")
