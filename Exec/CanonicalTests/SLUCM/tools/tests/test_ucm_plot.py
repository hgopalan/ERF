"""Minimal smoke tests for ucm_plot."""

import csv

import pytest

_ = pytest.importorskip("matplotlib")

from ucm_plot import plot_all


def test_plot_all_smoke(tmp_path):
    layout_path = tmp_path / "building_layout.csv"
    materials_path = tmp_path / "materials.csv"
    output_dir = tmp_path / "plots"

    with open(layout_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "i", "j", "bldg_id", "height_m", "plan_area_frac", "W_road_m",
            "W_roof_m", "roof_mat_id", "wall_mat_id", "road_mat_id",
            "orientation_deg", "ah_profile_id", "AH_Wm2", "is_urban",
        ])
        writer.writerow([0, 0, 1, 10.0, 0.4, 12.0, 8.0, 1, 1, 1, 0.0, 0, 25.0, 1])
        writer.writerow([1, 0, 1, 0.0, 0.0, 20.0, 0.0, 0, 0, 0, 0.0, 0, 0.0, 0])

    with open(materials_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "mat_id", "name", "albedo", "emissivity", "k_therm_W_per_mK",
            "rho_cp_J_per_m3K", "thickness_m", "description",
        ])
        writer.writerow([1, "concrete", 0.2, 0.9, 1.5, 2.0e6, 0.3, "generic"])

    outputs = plot_all(str(layout_path), str(materials_path), str(output_dir))

    assert len(outputs) == 6
    assert (output_dir / "urban_mask.png").exists()
