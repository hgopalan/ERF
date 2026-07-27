"""test_ucm_csv.py — pytest tests for ucm_csv module.

Tests the synthetic path. Runs in CI without any network or GIS deps.
"""
import pytest
import csv
import tempfile
import os
from ucm_csv import write_layout, write_materials, BUILDING_HEADER
from ucm_generators import uniform_urban


def test_write_layout_produces_correct_row_count():
    """write_layout produces exactly nx*ny data rows + 1 header."""
    nx_ucm, ny_ucm = 4, 5
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        write_layout(path, nx_ucm, ny_ucm, uniform_urban())

        with open(path, "r") as f:
            reader = csv.reader(f)
            lines = list(reader)

        assert len(lines) == nx_ucm * ny_ucm + 1, \
            f"Expected {nx_ucm * ny_ucm + 1} lines, got {len(lines)}"
        assert lines[0] == BUILDING_HEADER


def test_write_layout_roundtrip():
    """Read back layout and verify written values."""
    nx_ucm, ny_ucm = 3, 3
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        write_layout(path, nx_ucm, ny_ucm, uniform_urban(H_bldg=15.0,
                                                          plan_frac=0.6))

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == nx_ucm * ny_ucm
        for row in rows:
            assert float(row["height_m"]) == 15.0
            assert float(row["plan_area_frac"]) == 0.6
            assert int(row["is_urban"]) == 1


def test_write_layout_nonurban_zero_mat_accepted():
    """Non-urban row with mat_id = 0 accepted."""
    def cell_fn(i, j):
        if i == 0 and j == 0:
            return dict(i=0, j=0, bldg_id=1, height_m=0.0,
                        plan_area_frac=0.0, W_road_m=0.0, W_roof_m=0.0,
                        roof_mat_id=0, wall_mat_id=0, road_mat_id=0,
                        orientation_deg=0.0, ah_profile_id=0, is_urban=0)
        return uniform_urban()(i, j)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        write_layout(path, 2, 2, cell_fn)

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            row_0_0 = next(r for r in rows if r["i"] == "0" and r["j"] == "0")
            assert int(row_0_0["roof_mat_id"]) == 0


def test_write_layout_urban_zero_mat_raises():
    """Urban row with mat_id = 0 raises ValueError."""
    def bad_cell_fn(i, j):
        return dict(i=i, j=j, bldg_id=1, height_m=10.0,
                    plan_area_frac=0.5, W_road_m=10.0, W_roof_m=10.0,
                    roof_mat_id=0, wall_mat_id=1, road_mat_id=1,
                    orientation_deg=0.0, ah_profile_id=0, is_urban=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        with pytest.raises(ValueError, match="urban cell needs"):
            write_layout(path, 2, 2, bad_cell_fn)


def test_write_layout_plan_frac_outside_range_raises():
    """plan_area_frac outside [0, 1] raises ValueError."""
    def bad_cell_fn(i, j):
        return dict(i=i, j=j, bldg_id=1, height_m=10.0,
                    plan_area_frac=1.5, W_road_m=10.0, W_roof_m=10.0,
                    roof_mat_id=1, wall_mat_id=1, road_mat_id=1,
                    orientation_deg=0.0, ah_profile_id=0, is_urban=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        with pytest.raises(ValueError, match="plan_area_frac must be"):
            write_layout(path, 2, 2, bad_cell_fn)


def test_write_materials_rejects_duplicate_mat_id():
    """write_materials rejects duplicate mat_id."""
    materials = [
        dict(mat_id=1, name="mat1", albedo=0.2, emissivity=0.9,
             k_therm_W_per_mK=1.5, rho_cp_J_per_m3K=2.0e6,
             thickness_m=0.3, description="first"),
        dict(mat_id=1, name="mat1_dup", albedo=0.3, emissivity=0.85,
             k_therm_W_per_mK=1.6, rho_cp_J_per_m3K=2.1e6,
             thickness_m=0.31, description="duplicate"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        with pytest.raises(ValueError, match="duplicate"):
            write_materials(path, materials)


def test_write_materials_rejects_albedo_out_of_range():
    """write_materials rejects albedo > 1."""
    materials = [
        dict(mat_id=1, name="bad_mat", albedo=1.5, emissivity=0.9,
             k_therm_W_per_mK=1.5, rho_cp_J_per_m3K=2.0e6,
             thickness_m=0.3, description="bad"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        with pytest.raises(ValueError, match="albedo out of"):
            write_materials(path, materials)


def test_write_materials_rejects_thickness_zero():
    """write_materials rejects thickness_m <= 0."""
    materials = [
        dict(mat_id=1, name="bad_mat", albedo=0.2, emissivity=0.9,
             k_therm_W_per_mK=1.5, rho_cp_J_per_m3K=2.0e6,
             thickness_m=0.0, description="bad"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        with pytest.raises(ValueError, match="thickness_m must be"):
            write_materials(path, materials)


def test_write_materials_success():
    """write_materials successfully writes valid materials."""
    materials = [
        dict(mat_id=1, name="concrete", albedo=0.2, emissivity=0.9,
             k_therm_W_per_mK=1.5, rho_cp_J_per_m3K=2.0e6,
             thickness_m=0.3, description="generic"),
        dict(mat_id=2, name="asphalt", albedo=0.08, emissivity=0.95,
             k_therm_W_per_mK=0.7, rho_cp_J_per_m3K=1.4e6,
             thickness_m=0.05, description="road"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        write_materials(path, materials)

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert int(rows[0]["mat_id"]) == 1
        assert int(rows[1]["mat_id"]) == 2


def test_write_layout_AH_Wm2_roundtrip():
    """Round-trip test with AH_Wm2 = 42.0 (Phase 2.9 new column)."""
    def cell_fn(i, j):
        row = uniform_urban()(i, j)
        row["AH_Wm2"] = 42.0  # Override with non-zero value
        return row

    nx_ucm, ny_ucm = 2, 2
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        write_layout(path, nx_ucm, ny_ucm, cell_fn)

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == nx_ucm * ny_ucm
        for row in rows:
            assert float(row["AH_Wm2"]) == 42.0
            assert int(row["is_urban"]) == 1


def test_write_layout_AH_Wm2_default_backward_compat():
    """Backward compatibility: if cell_fn omits AH_Wm2, defaults to 0.0."""
    # This generator does NOT provide AH_Wm2 (simulating old code)
    # but write_layout should add it via setdefault
    def old_style_cell_fn(i, j):
        # Deliberately omit AH_Wm2 to test backward compat
        return dict(i=i, j=j, bldg_id=1, height_m=10.0,
                    plan_area_frac=0.5, W_road_m=10.0, W_roof_m=10.0,
                    roof_mat_id=1, wall_mat_id=1, road_mat_id=1,
                    orientation_deg=0.0, ah_profile_id=0, is_urban=1)

    nx_ucm, ny_ucm = 2, 2
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        write_layout(path, nx_ucm, ny_ucm, old_style_cell_fn)

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == nx_ucm * ny_ucm
        for row in rows:
            assert float(row["AH_Wm2"]) == 0.0  # Should default to 0.0


def test_write_layout_AH_Wm2_negative_raises():
    """Negative AH_Wm2 raises ValueError."""
    def bad_cell_fn(i, j):
        row = uniform_urban()(i, j)
        row["AH_Wm2"] = -5.0  # Invalid: negative
        return row

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        with pytest.raises(ValueError, match="AH_Wm2 must be"):
            write_layout(path, 2, 2, bad_cell_fn)
