#!/usr/bin/env python3
"""Generate building_layout.csv with physical coordinates (x_m, y_m) from domain parameters.

Phase 3.7: Replaces grid-index based CSV with physical-coordinate format for
scalability and grid-independence. Supports multiple layout presets.

Usage:
    python3 gen_building_layout.py \\
        --prob-lo 0 0 \\
        --prob-hi 20000 20000 \\
        --n-ucm 80 80 \\
        --output building_layout.csv \\
        --layout boston_5zone

Supported layouts:
    - homogeneous: Flat uniform city (H=15m, W=10m for all cells)
    - boston_5zone: Concentric zones mimicking real Boston (downtown core to suburban)
    - simple_cbd: Single tall center block (H=50m) surrounded by low-rise (H=10m)
"""

import argparse
import csv
import sys
import math


def homogeneous_cell_fn(i, j, x_m, y_m, prob_lo, prob_hi, n_ucm, **kwargs):
    """Homogeneous morphology: uniform city everywhere.
    
    Parameters:
        H_bldg: building height [m], default 15.0
        W_road: road width [m], default 10.0
        W_roof: roof width [m], default 6.0
        mat_id: material ID, default 2 (brick_concrete)
        f_urb: urban fraction, default 1.0
    """
    H = kwargs.get('H_bldg', 15.0)
    W_road = kwargs.get('W_road', 10.0)
    W_roof = kwargs.get('W_roof', 6.0)
    mat_id = kwargs.get('mat_id', 2)
    f_urb = kwargs.get('f_urb', 1.0)
    
    return {
        'x_m': x_m,
        'y_m': y_m,
        'bldg_id': 1,
        'height_m': H,
        'plan_area_frac': f_urb,
        'W_road_m': W_road,
        'W_roof_m': W_roof,
        'roof_mat_id': mat_id,
        'wall_mat_id': mat_id,
        'road_mat_id': mat_id,
        'orientation_deg': 0.0,
        'ah_profile_id': 0,
        'AH_Wm2': 30.0,
        'is_urban': 1,
    }


def boston_5zone_cell_fn(i, j, x_m, y_m, prob_lo, prob_hi, n_ucm, **kwargs):
    """Boston-stylized concentric layout based on Chebyshev distance.
    
    Rings (in 80×80 UCM grid, center at 39.5, 39.5):
        d = 0..7    (inner):   Downtown core              H=100m, λ_p=0.55
        d = 8..15:             Dense mid-rise             H=40m,  λ_p=0.50
        d = 16..24:            Residential dense          H=15m,  λ_p=0.35
        d = 25..32:            Residential sparse         H=8m,   λ_p=0.20
        d = 33..39  (outer):   Suburban / rural           H=5m,   λ_p=0.05
    """
    # Compute Chebyshev distance from center (UCM grid coordinates)
    nx_ucm, ny_ucm = n_ucm[0], n_ucm[1]
    center_i, center_j = (nx_ucm - 1) / 2.0, (ny_ucm - 1) / 2.0
    d = max(abs(i - center_i), abs(j - center_j))
    
    # Determine morphology based on ring
    if d <= 7:
        # Downtown core (Financial District style)
        morphology = {
            'is_urban': 1,
            'plan_area_frac': 0.55,
            'height_m': 100.0,
            'wall_mat_id': 1,  # glass_steel
            'AH_Wm2': 60.0,
        }
    elif d <= 15:
        # Dense mid-rise (Back Bay / Beacon Hill style)
        morphology = {
            'is_urban': 1,
            'plan_area_frac': 0.50,
            'height_m': 40.0,
            'wall_mat_id': 2,  # brick_concrete
            'AH_Wm2': 45.0,
        }
    elif d <= 24:
        # Residential dense (South End / Cambridge style)
        morphology = {
            'is_urban': 1,
            'plan_area_frac': 0.35,
            'height_m': 15.0,
            'wall_mat_id': 2,  # brick_concrete
            'AH_Wm2': 30.0,
        }
    elif d <= 32:
        # Residential sparse (Somerville / Brookline style)
        morphology = {
            'is_urban': 1,
            'plan_area_frac': 0.20,
            'height_m': 8.0,
            'wall_mat_id': 3,  # wood_vinyl
            'AH_Wm2': 15.0,
        }
    else:
        # Suburban / rural (Newton / outer metro style)
        morphology = {
            'is_urban': 1,
            'plan_area_frac': 0.05,
            'height_m': 5.0,
            'wall_mat_id': 3,  # wood_vinyl
            'AH_Wm2': 5.0,
        }
    
    # Road and roof widths scale with height
    W_road = morphology['height_m'] if morphology['is_urban'] else 0.0
    W_roof = 0.6 * morphology['height_m'] if morphology['is_urban'] else 0.0
    
    # Road material: brick_concrete (2) for most, wood_vinyl (3) for suburban
    road_mat_id = 3 if d >= 33 else 2
    
    return {
        'x_m': x_m,
        'y_m': y_m,
        'bldg_id': 1 if morphology['is_urban'] else 0,
        'height_m': morphology['height_m'],
        'plan_area_frac': morphology['plan_area_frac'],
        'W_road_m': W_road,
        'W_roof_m': W_roof,
        'roof_mat_id': morphology['wall_mat_id'],
        'wall_mat_id': morphology['wall_mat_id'],
        'road_mat_id': road_mat_id,
        'orientation_deg': 0.0,
        'ah_profile_id': 0,
        'AH_Wm2': morphology['AH_Wm2'],
        'is_urban': morphology['is_urban'],
    }


def simple_cbd_cell_fn(i, j, x_m, y_m, prob_lo, prob_hi, n_ucm, **kwargs):
    """Simple CBD: single tall center block surrounded by low-rise.
    
    - Center 5×5 cells: H=50m (high-rise CBD)
    - Outer ring: H=10m (residential)
    """
    nx_ucm, ny_ucm = n_ucm[0], n_ucm[1]
    center_i, center_j = (nx_ucm - 1) / 2.0, (ny_ucm - 1) / 2.0
    
    # Distance from center (Chebyshev)
    d = max(abs(i - center_i), abs(j - center_j))
    
    # Inner CBD: 5×5 block (d ≤ 2)
    if d <= 2:
        height_m = 50.0
        plan_area_frac = 0.6
        mat_id = 1  # glass_steel
        AH_Wm2 = 50.0
    else:
        # Residential surrounding
        height_m = 10.0
        plan_area_frac = 0.3
        mat_id = 2  # brick_concrete
        AH_Wm2 = 20.0
    
    W_road = height_m
    W_roof = 0.6 * height_m
    
    return {
        'x_m': x_m,
        'y_m': y_m,
        'bldg_id': 1,
        'height_m': height_m,
        'plan_area_frac': plan_area_frac,
        'W_road_m': W_road,
        'W_roof_m': W_roof,
        'roof_mat_id': mat_id,
        'wall_mat_id': mat_id,
        'road_mat_id': mat_id,
        'orientation_deg': 0.0,
        'ah_profile_id': 0,
        'AH_Wm2': AH_Wm2,
        'is_urban': 1,
    }


def write_building_layout(output_path, prob_lo, prob_hi, n_ucm, cell_fn, **kwargs):
    """Write building_layout.csv with physical coordinates.
    
    Args:
        output_path: Output CSV file path
        prob_lo: (x_min, y_min) domain lower corner [m]
        prob_hi: (x_max, y_max) domain upper corner [m]
        n_ucm: (nx, ny) UCM grid size
        cell_fn: Function(i, j, x_m, y_m, ...) → row dict
        **kwargs: Additional arguments to pass to cell_fn
    """
    nx_ucm, ny_ucm = n_ucm[0], n_ucm[1]
    dx_ucm = (prob_hi[0] - prob_lo[0]) / nx_ucm
    dy_ucm = (prob_hi[1] - prob_lo[1]) / ny_ucm
    
    header = [
        'x_m', 'y_m', 'bldg_id', 'height_m', 'plan_area_frac',
        'W_road_m', 'W_roof_m',
        'roof_mat_id', 'wall_mat_id', 'road_mat_id',
        'orientation_deg', 'ah_profile_id',
        'AH_Wm2',
        'is_urban',
    ]
    
    n = 0
    with open(output_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        
        for j in range(ny_ucm):
            for i in range(nx_ucm):
                # Compute cell center in physical coordinates
                x_m = prob_lo[0] + (i + 0.5) * dx_ucm
                y_m = prob_lo[1] + (j + 0.5) * dy_ucm
                
                # Call layout-specific function
                row = cell_fn(i, j, x_m, y_m, prob_lo, prob_hi, n_ucm, **kwargs)
                
                # Validate row
                _validate_row(i, j, row)
                
                # Write row
                w.writerow(row)
                n += 1
    
    print(f"[gen_building_layout] wrote {n} rows to {output_path} "
          f"(expected {nx_ucm * ny_ucm}) — OK")


def _validate_row(i, j, row):
    """Validate a single building_layout row."""
    if row['is_urban'] not in (0, 1):
        raise ValueError(f"({i},{j}): is_urban must be 0 or 1, "
                         f"got {row['is_urban']}")
    if row['is_urban'] == 1:
        for key in ('roof_mat_id', 'wall_mat_id', 'road_mat_id'):
            if int(row[key]) < 1:
                raise ValueError(f"({i},{j}): urban cell needs {key} >= 1, "
                                 f"got {row[key]}")
    if not (0.0 <= float(row['plan_area_frac']) <= 1.0):
        raise ValueError(f"({i},{j}): plan_area_frac must be in [0,1], "
                         f"got {row['plan_area_frac']}")
    if float(row['height_m']) < 0.0:
        raise ValueError(f"({i},{j}): height_m must be >= 0")
    if float(row['AH_Wm2']) < 0.0:
        raise ValueError(f"({i},{j}): AH_Wm2 must be >= 0, "
                         f"got {row['AH_Wm2']}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate building_layout.csv with physical coordinates (Phase 3.7)')
    
    parser.add_argument('--prob-lo', type=float, nargs=2, required=True,
                        metavar=('X', 'Y'),
                        help='Domain lower corner in x,y [m]')
    parser.add_argument('--prob-hi', type=float, nargs=2, required=True,
                        metavar=('X', 'Y'),
                        help='Domain upper corner in x,y [m]')
    parser.add_argument('--n-ucm', type=int, nargs=2, required=True,
                        metavar=('NX', 'NY'),
                        help='Number of UCM cells in x,y')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--layout', type=str, default='homogeneous',
                        choices=['homogeneous', 'boston_5zone', 'simple_cbd'],
                        help='Layout preset (default: homogeneous)')
    
    # Optional parameters for specific layouts
    parser.add_argument('--H-bldg', type=float, default=15.0,
                        help='Building height for homogeneous layout [m]')
    parser.add_argument('--W-road', type=float, default=10.0,
                        help='Road width for homogeneous layout [m]')
    parser.add_argument('--W-roof', type=float, default=6.0,
                        help='Roof width for homogeneous layout [m]')
    parser.add_argument('--mat-id', type=int, default=2,
                        help='Material ID for homogeneous layout')
    parser.add_argument('--f-urb', type=float, default=1.0,
                        help='Urban fraction for homogeneous layout')
    
    args = parser.parse_args()
    
    # Select layout function
    layout_functions = {
        'homogeneous': homogeneous_cell_fn,
        'boston_5zone': boston_5zone_cell_fn,
        'simple_cbd': simple_cbd_cell_fn,
    }
    
    cell_fn = layout_functions[args.layout]
    
    # Prepare kwargs for cell function
    cell_kwargs = {}
    if args.layout == 'homogeneous':
        cell_kwargs = {
            'H_bldg': args.H_bldg,
            'W_road': args.W_road,
            'W_roof': args.W_roof,
            'mat_id': args.mat_id,
            'f_urb': args.f_urb,
        }
    
    # Generate CSV
    try:
        write_building_layout(
            args.output,
            tuple(args.prob_lo),
            tuple(args.prob_hi),
            tuple(args.n_ucm),
            cell_fn,
            **cell_kwargs
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
