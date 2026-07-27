#!/usr/bin/env python3
"""Generate a checkerboard material pattern for testing heterogeneity.

This example demonstrates using the checkerboard_materials generator to
create alternating material properties in a grid pattern.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ucm_csv import write_layout, write_materials
from ucm_generators import uniform_urban, checkerboard_materials

NX_ATM, NY_ATM, GRID_RATIO = 8, 8, 2
NX_UCM, NY_UCM = NX_ATM * GRID_RATIO, NY_ATM * GRID_RATIO

print(f"[gen_checkerboard] generating {NX_UCM}x{NY_UCM} checkerboard grid...")
base = uniform_urban()
cell_fn = checkerboard_materials(base, mat_a=1, mat_b=2)
write_layout("building_layout.csv", NX_UCM, NY_UCM, cell_fn)

write_materials("materials.csv", [
    dict(mat_id=1, name="concrete", albedo=0.20, emissivity=0.90,
         k_therm_W_per_mK=1.5, rho_cp_J_per_m3K=2.0e6,
         thickness_m=0.3, description="darker material"),
    dict(mat_id=2, name="light_concrete", albedo=0.35, emissivity=0.88,
         k_therm_W_per_mK=1.2, rho_cp_J_per_m3K=1.8e6,
         thickness_m=0.25, description="lighter material"),
])
print("[gen_checkerboard] done.")
