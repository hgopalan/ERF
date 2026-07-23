#!/usr/bin/env python3
"""
Generate building layout CSV for Phase 2.7 canonical test.

Creates two vertical stripes:
  - Left (i=0..7): tall dense buildings
  - Right (i=8..15): short sparse buildings

This tests the BEP-style geometric overlap and sharp vs Gaussian modes.
"""

import csv

def generate_building_layout():
    """Generate Phase 2.7 test layout: two height classes."""
    
    with open("building_layout.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "i_ucm", "j_ucm", "height_m", "plan_area_frac", "is_urban"
        ])
        writer.writeheader()
        
        # Left stripe: tall dense buildings
        for i in range(0, 8):
            for j in range(0, 16):
                writer.writerow({
                    "i_ucm": i,
                    "j_ucm": j,
                    "height_m": 30.0,      # 30 m tall
                    "plan_area_frac": 0.6, # 60% dense
                    "is_urban": 1
                })
        
        # Right stripe: short sparse buildings
        for i in range(8, 16):
            for j in range(0, 16):
                writer.writerow({
                    "i_ucm": i,
                    "j_ucm": j,
                    "height_m": 5.0,       # 5 m tall
                    "plan_area_frac": 0.2, # 20% sparse
                    "is_urban": 1
                })

def generate_materials():
    """Generate materials library (single generic material)."""
    
    with open("materials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "material_id", "name", "wall_emissivity", "wall_albedo",
            "roof_emissivity", "roof_albedo"
        ])
        writer.writeheader()
        
        # Generic material (indices don't matter for this simple test)
        writer.writerow({
            "material_id": 1,
            "name": "concrete",
            "wall_emissivity": 0.90,
            "wall_albedo": 0.20,
            "roof_emissivity": 0.90,
            "roof_albedo": 0.20
        })

if __name__ == "__main__":
    generate_building_layout()
    generate_materials()
    print("Generated building_layout.csv and materials.csv")
