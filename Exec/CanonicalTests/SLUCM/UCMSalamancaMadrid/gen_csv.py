#!/usr/bin/env python3
"""Generate Salamanca Madrid canonical layout with urban-rural contrast.

Layout in x (i = 0..39 UCM cells, 5 km domain):
  i =  0..7   (0.0 – 1.0 km)  : Rural upwind reference (is_urban=0)
  i =  8..15  (1.0 – 2.0 km)  : Suburban transition   (lambda_p=0.25, H=5 m)
  i = 16..23  (2.0 – 3.0 km)  : Madrid urban core     (lambda_p=0.55, H=20 m)
  i = 24..31  (3.0 – 4.0 km)  : Suburban transition   (lambda_p=0.25, H=5 m)
  i = 32..39  (4.0 – 5.0 km)  : Rural downwind         (is_urban=0)

Y-direction uniform (periodic BC).

Rationale: Salamanca 2011 UHI validation requires an upwind rural reference
so T_urban - T_rural can be measured. A uniform-urban domain has no rural
surrogate. The urban core (i=16..23) represents ~1 km of dense Madrid centro;
the transitions are residential rings; the endcaps are the Meseta plateau.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))
from ucm_csv import write_layout, write_materials

NX_ATM, NY_ATM, GRID_RATIO = 10, 10, 4
NX_UCM, NY_UCM = NX_ATM * GRID_RATIO, NY_ATM * GRID_RATIO


def morphology_for_i(i):
    """Return (is_urban, plan_area_frac, height_m, mat_id, AH_Wm2) for column i."""
    if i <= 7:
        # Rural upwind reference (Meseta grassland surrogate).
        return dict(is_urban=0, plan_area_frac=0.0, height_m=0.0,
                    mat_id=2, AH_Wm2=0.0)
    if i <= 15:
        # Suburban ring, residential.
        return dict(is_urban=1, plan_area_frac=0.25, height_m=5.0,
                    mat_id=1, AH_Wm2=20.0)
    if i <= 23:
        # Madrid urban core (Salamanca 2011 target region).
        return dict(is_urban=1, plan_area_frac=0.55, height_m=20.0,
                    mat_id=1, AH_Wm2=40.0)
    if i <= 31:
        # Suburban ring, residential (downwind).
        return dict(is_urban=1, plan_area_frac=0.25, height_m=5.0,
                    mat_id=1, AH_Wm2=20.0)
    # Rural downwind (i = 32..39).
    return dict(is_urban=0, plan_area_frac=0.0, height_m=0.0,
                mat_id=2, AH_Wm2=0.0)


def cell_fn(i, j):
    m = morphology_for_i(i)
    # Road/roof widths scale with height (rough W_road ~ H, W_roof ~ 0.6*H
    # for a typical H/W ~ 1 canyon).
    W_road = m["height_m"] if m["is_urban"] else 0.0
    W_roof = 0.6 * m["height_m"] if m["is_urban"] else 0.0
    return dict(
        i=i,
        j=j,
        bldg_id=1 if m["is_urban"] else 0,
        height_m=m["height_m"],
        plan_area_frac=m["plan_area_frac"],
        W_road_m=W_road,
        W_roof_m=W_roof,
        roof_mat_id=m["mat_id"],
        wall_mat_id=m["mat_id"],
        road_mat_id=m["mat_id"],
        orientation_deg=0.0,
        ah_profile_id=0,
        AH_Wm2=m["AH_Wm2"],
        is_urban=m["is_urban"],
    )


write_layout("building_layout.csv", NX_UCM, NY_UCM, cell_fn)
write_materials(
    "materials.csv",
    [
        dict(
            mat_id=1,
            name="madrid_brick_concrete",
            albedo=0.20,
            emissivity=0.90,
            k_therm_W_per_mK=1.1,
            rho_cp_J_per_m3K=1.8e6,
            thickness_m=0.30,
            description="Salamanca 2011 Table 1 representative values",
        ),
        dict(
            mat_id=2,
            name="meseta_grassland",
            albedo=0.20,
            emissivity=0.95,
            k_therm_W_per_mK=0.3,
            rho_cp_J_per_m3K=1.4e6,
            thickness_m=0.10,
            description="Rural surrogate: grassland surface for is_urban=0 cells",
        ),
    ],
)