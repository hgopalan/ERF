#!/usr/bin/env python3
"""Generate UCMBoston canonical layout with 5 concentric rings and urban-rural contrast.

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
synthetic layout suitable for Phase 2.11 baseline validation; Phase 2.9's
gen_real_boston_full.py can regenerate with real WUDAPT/OSM data for
manual QA in Phase 4+.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))
from ucm_csv import write_layout, write_materials

NX_ATM, NY_ATM, GRID_RATIO = 20, 20, 4
NX_UCM, NY_UCM = NX_ATM * GRID_RATIO, NY_ATM * GRID_RATIO


def chebyshev_distance(i, j, center_i=39.5, center_j=39.5):
    """Compute L-infinity (Chebyshev) distance from center."""
    return max(abs(i - center_i), abs(j - center_j))


def morphology_for_ring(d):
    """Return (is_urban, plan_area_frac, height_m, wall_mat_id, AH_Wm2) for ring d."""
    if d <= 7:
        # Downtown core (Financial District style)
        return dict(is_urban=1, plan_area_frac=0.55, height_m=100.0,
                    wall_mat_id=1, AH_Wm2=60.0)
    if d <= 15:
        # Dense mid-rise (Back Bay / Beacon Hill style)
        return dict(is_urban=1, plan_area_frac=0.50, height_m=40.0,
                    wall_mat_id=2, AH_Wm2=45.0)
    if d <= 24:
        # Residential dense (South End / Cambridge style)
        return dict(is_urban=1, plan_area_frac=0.35, height_m=15.0,
                    wall_mat_id=2, AH_Wm2=30.0)
    if d <= 32:
        # Residential sparse (Somerville / Brookline style)
        return dict(is_urban=1, plan_area_frac=0.20, height_m=8.0,
                    wall_mat_id=3, AH_Wm2=15.0)
    # Suburban / rural (Newton / outer metro style, d = 33..39)
    return dict(is_urban=1, plan_area_frac=0.05, height_m=5.0,
                wall_mat_id=3, AH_Wm2=5.0)


def cell_fn(i, j):
    d = chebyshev_distance(i, j)
    m = morphology_for_ring(d)

    # Road/roof widths scale with height
    W_road = m["height_m"] if m["is_urban"] else 0.0
    W_roof = 0.6 * m["height_m"] if m["is_urban"] else 0.0

    # Material assignments: downtown uses glass/steel (mat_id=1),
    # dense and residential dense use brick/concrete (mat_id=2),
    # suburban uses wood/vinyl (mat_id=3).
    # Road material: brick/concrete (mat_id=2) for all except suburban (mat_id=3).
    road_mat_id = 3 if d >= 33 else 2

    return dict(
        i=i,
        j=j,
        bldg_id=1 if m["is_urban"] else 0,
        height_m=m["height_m"],
        plan_area_frac=m["plan_area_frac"],
        W_road_m=W_road,
        W_roof_m=W_roof,
        roof_mat_id=m["wall_mat_id"],
        wall_mat_id=m["wall_mat_id"],
        road_mat_id=road_mat_id,
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
    ],
)
