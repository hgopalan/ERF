#!/usr/bin/env python3
"""Generate a domain with a non-urban patch in the center.

This example demonstrates using the with_nonurban_box generator to punch
out a non-urban rectangle (e.g., park, water body) into an urban grid.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ucm_csv import write_layout, write_materials
from ucm_generators import uniform_urban, with_nonurban_box

NX_ATM, NY_ATM, GRID_RATIO = 8, 8, 2
NX_UCM, NY_UCM = NX_ATM * GRID_RATIO, NY_ATM * GRID_RATIO

print(f"[gen_nonurban_patch] generating {NX_UCM}x{NY_UCM} grid with "
      f"non-urban patch...")

base = uniform_urban(H_bldg=10.0, plan_frac=0.5)
cell_fn = with_nonurban_box(base, i0=6, i1=10, j0=6, j1=10)

write_layout("building_layout.csv", NX_UCM, NY_UCM, cell_fn)
write_materials("materials.csv", [
    dict(mat_id=1, name="concrete", albedo=0.20, emissivity=0.90,
         k_therm_W_per_mK=1.5, rho_cp_J_per_m3K=2.0e6,
         thickness_m=0.3, description="generic urban"),
])
print("[gen_nonurban_patch] done.")
