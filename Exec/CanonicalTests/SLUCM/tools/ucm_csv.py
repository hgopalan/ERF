"""ucm_csv.py — canonical CSV writers for ERF-SLUCM.

Convention (locked in Phase 2.3, R14):
  * i,j are UCM cell indices in [0, nx_ucm) x [0, ny_ucm).
  * building_layout.csv has exactly nx_ucm * ny_ucm data rows, one per cell.
  * Non-urban cells (is_urban=0) may have mat_id = 0 sentinels.
  * Urban cells (is_urban=1) must have mat_id >= 1.
"""
import csv
from typing import Callable, Iterable, Mapping

BUILDING_HEADER = [
    "i", "j", "bldg_id", "height_m", "plan_area_frac",
    "W_road_m", "W_roof_m",
    "roof_mat_id", "wall_mat_id", "road_mat_id",
    "orientation_deg", "ah_profile_id", "is_urban",
]
MATERIAL_HEADER = [
    "mat_id", "name", "albedo", "emissivity",
    "k_therm_W_per_mK", "rho_cp_J_per_m3K", "thickness_m", "description",
]


def write_layout(path: str, nx_ucm: int, ny_ucm: int,
                 cell_fn: Callable[[int, int], Mapping]) -> None:
    """Write building_layout.csv from a per-cell generator function.

    Args:
        path: Path where CSV will be written.
        nx_ucm: Number of cells in x-direction.
        ny_ucm: Number of cells in y-direction.
        cell_fn: Function that takes (i, j) and returns a dict with row data.
    """
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BUILDING_HEADER)
        for j in range(ny_ucm):
            for i in range(nx_ucm):
                row = dict(cell_fn(i, j))
                row.setdefault("i", i)
                row.setdefault("j", j)
                _validate_row(i, j, row)
                w.writerow([row[k] for k in BUILDING_HEADER])
                n += 1
    print(f"[ucm_csv] wrote {n} rows to {path} "
          f"(expected {nx_ucm * ny_ucm}) — OK")


def write_materials(path: str, materials: Iterable[Mapping]) -> None:
    """Write materials.csv from an iterable of material dicts.

    Args:
        path: Path where CSV will be written.
        materials: Iterable of dicts, each with material properties.
    """
    seen = set()
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(MATERIAL_HEADER)
        for m in materials:
            _validate_material(m, seen)
            w.writerow([m[k] for k in MATERIAL_HEADER])
            seen.add(m["mat_id"])
            n += 1
    print(f"[ucm_csv] wrote {n} materials to {path}")


def _validate_row(i, j, row):
    """Validate a single building_layout row."""
    if row["i"] != i or row["j"] != j:
        raise ValueError(f"cell_fn returned wrong (i,j)=({row['i']},{row['j']}); "
                         f"expected ({i},{j})")
    if row["is_urban"] not in (0, 1):
        raise ValueError(f"({i},{j}): is_urban must be 0 or 1, "
                         f"got {row['is_urban']}")
    if row["is_urban"] == 1:
        for key in ("roof_mat_id", "wall_mat_id", "road_mat_id"):
            if int(row[key]) < 1:
                raise ValueError(f"({i},{j}): urban cell needs {key} >= 1, "
                                 f"got {row[key]}")
    if not (0.0 <= float(row["plan_area_frac"]) <= 1.0):
        raise ValueError(f"({i},{j}): plan_area_frac must be in [0,1], "
                         f"got {row['plan_area_frac']}")
    if float(row["height_m"]) < 0.0:
        raise ValueError(f"({i},{j}): height_m must be >= 0")


def _validate_material(m, seen):
    """Validate a single material dict."""
    if int(m["mat_id"]) < 1:
        raise ValueError(f"mat_id must be >= 1, got {m['mat_id']}")
    if m["mat_id"] in seen:
        raise ValueError(f"duplicate mat_id={m['mat_id']}")
    for key in ("albedo", "emissivity"):
        if not (0.0 <= float(m[key]) <= 1.0):
            raise ValueError(f"mat_id={m['mat_id']}: {key} out of [0,1]")
    for key in ("k_therm_W_per_mK", "rho_cp_J_per_m3K", "thickness_m"):
        if float(m[key]) <= 0.0:
            raise ValueError(f"mat_id={m['mat_id']}: {key} must be > 0")
