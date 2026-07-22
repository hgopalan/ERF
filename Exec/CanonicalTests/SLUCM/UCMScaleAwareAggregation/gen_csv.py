#!/usr/bin/env python3
"""Generate CSV for UCMScaleAwareAggregation test.

Domain: ATM 4x4, grid_ratio=4 -> UCM 16x16.
Pattern: diagonal urban wedge. Cells with (i+j) < 12 are urban.

At grid_ratio=4, this produces ATM cells with f_urb ranging from 0
(both indices high) to 1 (both indices low), plus partially-filled
cells in between. This exercises the urban-fraction-weighted coarsening.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))
from ucm_csv import write_layout, write_materials

NX_ATM, NY_ATM, GRID_RATIO = 4, 4, 4
NX_UCM, NY_UCM = NX_ATM * GRID_RATIO, NY_ATM * GRID_RATIO   # 16 x 16

def cell_fn(i, j):
    is_urban = 1 if (i + j) < 12 else 0
    if is_urban:
        return dict(i=i, j=j, bldg_id=1, height_m=10.0, plan_area_frac=0.5,
                    W_road_m=10.0, W_roof_m=10.0,
                    roof_mat_id=1, wall_mat_id=1, road_mat_id=1,
                    orientation_deg=0.0, ah_profile_id=0, is_urban=1)
    else:
        return dict(i=i, j=j, bldg_id=1, height_m=0.0, plan_area_frac=0.0,
                    W_road_m=0.0, W_roof_m=0.0,
                    roof_mat_id=0, wall_mat_id=0, road_mat_id=0,
                    orientation_deg=0.0, ah_profile_id=0, is_urban=0)

write_layout("building_layout.csv", NX_UCM, NY_UCM, cell_fn)
write_materials("materials.csv", [
    dict(mat_id=1, name="concrete", albedo=0.20, emissivity=0.90,
         k_therm_W_per_mK=1.5, rho_cp_J_per_m3K=2.0e6,
         thickness_m=0.30, description="generic concrete"),
])
