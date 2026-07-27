#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))
from ucm_csv import write_layout, write_materials

NX_ATM, NY_ATM, GRID_RATIO = 10, 10, 4
NX_UCM, NY_UCM = NX_ATM * GRID_RATIO, NY_ATM * GRID_RATIO


def lambda_for_i(i):
    if i <= 9:
        return 0.11
    if i <= 19:
        return 0.25
    if i <= 29:
        return 0.33
    return 0.44


def cell_fn(i, j):
    return dict(
        i=i,
        j=j,
        bldg_id=1,
        height_m=15.0,
        plan_area_frac=lambda_for_i(i),
        W_road_m=8.0,
        W_roof_m=12.0,
        roof_mat_id=1,
        wall_mat_id=1,
        road_mat_id=1,
        orientation_deg=0.0,
        ah_profile_id=0,
        AH_Wm2=0.0,
        is_urban=1,
    )


write_layout("building_layout.csv", NX_UCM, NY_UCM, cell_fn)
write_materials(
    "materials.csv",
    [
        dict(
            mat_id=1,
            name="kanda_concrete",
            albedo=0.20,
            emissivity=0.90,
            k_therm_W_per_mK=1.1,
            rho_cp_J_per_m3K=1.8e6,
            thickness_m=0.30,
            description="Neutral momentum-only canonical material",
        )
    ],
)
