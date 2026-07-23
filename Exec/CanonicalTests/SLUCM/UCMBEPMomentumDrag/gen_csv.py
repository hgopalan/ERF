#!/usr/bin/env python3
"""Generate CSV for UCMBEPMomentumDrag test (Phase 2.8).

Domain: ATM 4x4, grid_ratio=4 -> UCM 16x16.
Pattern: two vertical stripes.
  - Left half (i=0..7): tall dense buildings (h=30m, plan_area=0.6)
  - Right half (i=8..15): short sparse buildings (h=5m, plan_area=0.2)

Phase 2.8 physics: BEP momentum drag with per-cell wall overlap geometry.
Tall buildings should show 50%+ wind reduction inside canopy (30 m tall);
short buildings show less reduction. Tests Martilli 2002 drag formulation.

Uses the shared ucm_csv toolchain (Exec/CanonicalTests/SLUCM/tools/ucm_csv.py)
so the CSV schema matches what the UCM CSV reader expects.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))
from ucm_csv import write_layout, write_materials

NX_ATM, NY_ATM, GRID_RATIO = 4, 4, 4
NX_UCM, NY_UCM = NX_ATM * GRID_RATIO, NY_ATM * GRID_RATIO   # 16 x 16

def cell_fn(i, j):
    # Left half: tall dense stripe
    if i < NX_UCM // 2:
        return dict(i=i, j=j, bldg_id=1, height_m=30.0, plan_area_frac=0.6,
                    W_road_m=5.0, W_roof_m=5.0,
                    roof_mat_id=1, wall_mat_id=1, road_mat_id=1,
                    orientation_deg=0.0, ah_profile_id=0, is_urban=1)
    # Right half: short sparse stripe
    else:
        return dict(i=i, j=j, bldg_id=1, height_m=5.0, plan_area_frac=0.2,
                    W_road_m=5.0, W_roof_m=5.0,
                    roof_mat_id=1, wall_mat_id=1, road_mat_id=1,
                    orientation_deg=0.0, ah_profile_id=0, is_urban=1)

write_layout("building_layout.csv", NX_UCM, NY_UCM, cell_fn)
write_materials("materials.csv", [
    dict(mat_id=1, name="concrete", albedo=0.20, emissivity=0.90,
         k_therm_W_per_mK=1.5, rho_cp_J_per_m3K=2.0e6,
         thickness_m=0.30, description="generic concrete"),
])
