#!/usr/bin/env python3
"""Generator for mixed urban/non-urban domain building_layout_mixed.csv (Phase 3.8).

Domain: 20 km × 20 km, 80 × 80 UCM cells (dx_ucm = 250 m).
Layout:
  - Left half (x_m < 10000 m): urban (is_urban=1, uniform Boston-5-zone properties)
  - Right half (x_m >= 10000 m): non-urban (is_urban=0, grassland_rural material)

Output CSV schema (Phase 3.7 physical-coordinate):
  x_m,y_m,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,roof_mat_id,
  wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,AH_Wm2,is_urban
"""

import sys

def generate_mixed_layout():
    """Generate building_layout_mixed.csv for Phase 3.8 test."""
    
    # Domain and grid parameters
    domain_size_m = 20000.0  # 20 km × 20 km
    grid_size = 80           # 80 × 80 UCM cells
    dx_ucm = domain_size_m / grid_size  # 250 m
    
    # Split point (middle of domain)
    split_x_m = domain_size_m / 2.0  # 10000 m
    
    # Urban half parameters (left: x < 10000 m)
    urban_height_m = 15.0
    urban_plan_area_frac = 0.05
    urban_w_road_m = 10.0
    urban_w_roof_m = 10.0
    urban_roof_mat_id = 2       # brick_concrete (from materials.csv)
    urban_wall_mat_id = 2       # brick_concrete
    urban_road_mat_id = 2       # brick_concrete
    urban_orientation_deg = 0.0
    urban_ah_profile_id = 0
    urban_ah_wm2 = 30.0
    urban_is_urban = 1
    
    # Non-urban half parameters (right: x >= 10000 m)
    rural_height_m = 0.0
    rural_plan_area_frac = 0.0
    rural_w_road_m = 0.0
    rural_w_roof_m = 0.0
    rural_roof_mat_id = 1       # grassland_rural (from materials.csv)
    rural_wall_mat_id = 1       # grassland_rural
    rural_road_mat_id = 1       # grassland_rural
    rural_orientation_deg = 0.0
    rural_ah_profile_id = 0
    rural_ah_wm2 = 0.0
    rural_is_urban = 0
    
    # Generate CSV header
    lines = []
    lines.append("x_m,y_m,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,"
                 "roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,"
                 "ah_profile_id,AH_Wm2,is_urban")
    
    urban_count = 0
    rural_count = 0
    bldg_id = 1
    
    # Generate grid
    for j in range(grid_size):
        for i in range(grid_size):
            # Cell center coordinates (physical space, meters)
            x_m = (i + 0.5) * dx_ucm
            y_m = (j + 0.5) * dx_ucm
            
            # Determine if urban or non-urban
            if x_m < split_x_m:
                # Urban half
                height_m = urban_height_m
                plan_area_frac = urban_plan_area_frac
                w_road_m = urban_w_road_m
                w_roof_m = urban_w_roof_m
                roof_mat_id = urban_roof_mat_id
                wall_mat_id = urban_wall_mat_id
                road_mat_id = urban_road_mat_id
                orientation_deg = urban_orientation_deg
                ah_profile_id = urban_ah_profile_id
                ah_wm2 = urban_ah_wm2
                is_urban = urban_is_urban
                urban_count += 1
            else:
                # Non-urban half
                height_m = rural_height_m
                plan_area_frac = rural_plan_area_frac
                w_road_m = rural_w_road_m
                w_roof_m = rural_w_roof_m
                roof_mat_id = rural_roof_mat_id
                wall_mat_id = rural_wall_mat_id
                road_mat_id = rural_road_mat_id
                orientation_deg = rural_orientation_deg
                ah_profile_id = rural_ah_profile_id
                ah_wm2 = rural_ah_wm2
                is_urban = rural_is_urban
                rural_count += 1
            
            # Format row (match Phase 3.7 physical-coordinate schema exactly)
            row = (
                f"{x_m:.1f},"
                f"{y_m:.1f},"
                f"{bldg_id},"
                f"{height_m:.1f},"
                f"{plan_area_frac:.2f},"
                f"{w_road_m:.1f},"
                f"{w_roof_m:.1f},"
                f"{roof_mat_id},"
                f"{wall_mat_id},"
                f"{road_mat_id},"
                f"{orientation_deg:.1f},"
                f"{ah_profile_id},"
                f"{ah_wm2:.1f},"
                f"{is_urban}"
            )
            lines.append(row)
            bldg_id += 1
    
    # Write CSV
    csv_filename = "building_layout_mixed.csv"
    try:
        with open(csv_filename, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"Generated 6400 rows: {urban_count} urban, {rural_count} non-urban.")
        return 0
    except Exception as e:
        print(f"ERROR writing {csv_filename}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(generate_mixed_layout())
