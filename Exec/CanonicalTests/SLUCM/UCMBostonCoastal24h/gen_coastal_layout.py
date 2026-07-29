#!/usr/bin/env python3
"""Generator for coastal sea-breeze canonical building_layout_coastal.csv (Phase 5.7).

Domain: 20 km × 20 km, 80 × 80 UCM cells (dx_ucm = 250 m).
Layout:
  - Sea (x=[0, 5000] m): is_urban=0, water material (high thermal inertia)
  - Coast transition (x=[5000, 6000] m): jagged checkerboard of sea/urban (Phase 5.6 blending)
  - Urban Boston (x=[6000, 14000] m): is_urban=1, urban buildings with Phase 5.1–5.5 physics
  - Rural-urban transition (x=[14000, 15000] m): jagged checkerboard of urban/rural
  - Rural inland (x=[15000, 20000] m): is_urban=0, grassland

Output CSV schema (Phase 3.7 physical-coordinate):
  x_m,y_m,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,roof_mat_id,
  wall_mat_id,road_mat_id,orientation_deg,ah_profile_id,AH_Wm2,is_urban
"""

import sys

def generate_coastal_layout():
    """Generate building_layout_coastal.csv for Phase 5.7 coastal canonical."""
    
    # Domain and grid parameters
    domain_size_m = 20000.0  # 20 km × 20 km
    grid_size = 80           # 80 × 80 UCM cells
    dx_ucm = domain_size_m / grid_size  # 250 m
    
    # Region boundaries (in meters)
    sea_west = 0.0
    sea_east = 5000.0
    coast_trans_west = 5000.0
    coast_trans_east = 6000.0
    urban_west = 6000.0
    urban_east = 14000.0
    rural_trans_west = 14000.0
    rural_trans_east = 15000.0
    rural_east = 20000.0
    
    # Sea parameters (water: high thermal inertia, minimal buildings)
    sea_height_m = 0.0
    sea_plan_area_frac = 0.0
    sea_w_road_m = 0.0
    sea_w_roof_m = 0.0
    sea_roof_mat_id = 5       # sea_water material (mat_id=5 in materials.csv)
    sea_wall_mat_id = 5
    sea_road_mat_id = 5
    sea_is_urban = 0
    
    # Urban parameters (Boston, similar to UCMBostonDiurnal24h)
    urban_height_m = 15.0
    urban_plan_area_frac = 0.05
    urban_w_road_m = 10.0
    urban_w_roof_m = 10.0
    urban_roof_mat_id = 2     # brick_concrete
    urban_wall_mat_id = 2
    urban_road_mat_id = 2
    urban_is_urban = 1
    
    # Rural parameters (grassland)
    rural_height_m = 0.0
    rural_plan_area_frac = 0.0
    rural_w_road_m = 0.0
    rural_w_roof_m = 0.0
    rural_roof_mat_id = 4     # grassland_rural (or surrogate if renamed)
    rural_wall_mat_id = 4
    rural_road_mat_id = 4
    rural_is_urban = 0
    
    # Generate CSV header
    lines = []
    lines.append("x_m,y_m,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,"
                 "roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,"
                 "ah_profile_id,AH_Wm2,is_urban")
    
    sea_count = 0
    coast_count = 0
    urban_count = 0
    rural_trans_count = 0
    rural_count = 0
    bldg_id = 1
    
    # Generate grid
    for j in range(grid_size):
        for i in range(grid_size):
            # Cell center coordinates (physical space, meters)
            x_m = (i + 0.5) * dx_ucm
            y_m = (j + 0.5) * dx_ucm
            
            # Determine region and properties
            if x_m < sea_east:
                # Pure sea region
                height_m = sea_height_m
                plan_area_frac = sea_plan_area_frac
                w_road_m = sea_w_road_m
                w_roof_m = sea_w_roof_m
                roof_mat_id = sea_roof_mat_id
                wall_mat_id = sea_wall_mat_id
                road_mat_id = sea_road_mat_id
                is_urban = sea_is_urban
                sea_count += 1
                
            elif x_m < coast_trans_east:
                # Coast transition band: jagged checkerboard (Phase 5.6 recipe)
                # Use (i+j)%2 to create checkerboard pattern within the band
                # This creates genuine fractional f_urb at coastal ATM cells
                use_urban = (i + j) % 2
                
                if use_urban:
                    # Urban tile within transition
                    height_m = urban_height_m
                    plan_area_frac = urban_plan_area_frac
                    w_road_m = urban_w_road_m
                    w_roof_m = urban_w_roof_m
                    roof_mat_id = urban_roof_mat_id
                    wall_mat_id = urban_wall_mat_id
                    road_mat_id = urban_road_mat_id
                    is_urban = 1
                else:
                    # Sea tile within transition
                    height_m = sea_height_m
                    plan_area_frac = sea_plan_area_frac
                    w_road_m = sea_w_road_m
                    w_roof_m = sea_w_roof_m
                    roof_mat_id = sea_roof_mat_id
                    wall_mat_id = sea_wall_mat_id
                    road_mat_id = sea_road_mat_id
                    is_urban = 0
                coast_count += 1
                
            elif x_m < urban_east:
                # Pure urban Boston
                height_m = urban_height_m
                plan_area_frac = urban_plan_area_frac
                w_road_m = urban_w_road_m
                w_roof_m = urban_w_roof_m
                roof_mat_id = urban_roof_mat_id
                wall_mat_id = urban_wall_mat_id
                road_mat_id = urban_road_mat_id
                is_urban = 1
                urban_count += 1
                
            elif x_m < rural_trans_east:
                # Rural-urban transition band: jagged checkerboard
                use_urban = (i + j) % 2
                
                if use_urban:
                    # Urban tile within transition
                    height_m = urban_height_m
                    plan_area_frac = urban_plan_area_frac
                    w_road_m = urban_w_road_m
                    w_roof_m = urban_w_roof_m
                    roof_mat_id = urban_roof_mat_id
                    wall_mat_id = urban_wall_mat_id
                    road_mat_id = urban_road_mat_id
                    is_urban = 1
                else:
                    # Rural tile within transition
                    height_m = rural_height_m
                    plan_area_frac = rural_plan_area_frac
                    w_road_m = rural_w_road_m
                    w_roof_m = rural_w_roof_m
                    roof_mat_id = rural_roof_mat_id
                    wall_mat_id = rural_wall_mat_id
                    road_mat_id = rural_road_mat_id
                    is_urban = 0
                rural_trans_count += 1
                
            else:
                # Pure rural inland
                height_m = rural_height_m
                plan_area_frac = rural_plan_area_frac
                w_road_m = rural_w_road_m
                w_roof_m = rural_w_roof_m
                roof_mat_id = rural_roof_mat_id
                wall_mat_id = rural_wall_mat_id
                road_mat_id = rural_road_mat_id
                is_urban = 0
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
                f"0.0,"  # orientation_deg
                f"0,"    # ah_profile_id
                f"30.0," # AH_Wm2 (anthropogenic heat, 30 W/m² is typical urban)
                f"{is_urban}"
            )
            lines.append(row)
            bldg_id += 1
    
    # Write CSV
    csv_filename = "building_layout_coastal.csv"
    try:
        with open(csv_filename, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"Generated 6400 rows:")
        print(f"  Sea:              {sea_count}")
        print(f"  Coast transition: {coast_count}")
        print(f"  Urban:            {urban_count}")
        print(f"  Rural transition: {rural_trans_count}")
        print(f"  Rural:            {rural_count}")
        return 0
    except Exception as e:
        print(f"ERROR writing {csv_filename}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(generate_coastal_layout())
