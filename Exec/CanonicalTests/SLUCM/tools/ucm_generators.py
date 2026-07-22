"""ucm_generators.py — synthetic UCM CSV generators.

Provides ready-to-use cell_fn factories for common urban patterns.
"""
from typing import Callable, Mapping


def uniform_urban(H_bldg=10.0, plan_frac=0.5, W_road=10.0, W_roof=10.0,
                  roof_mat_id=1, wall_mat_id=1, road_mat_id=1,
                  ah_profile_id=0) -> Callable[[int, int], Mapping]:
    """Every cell is urban with the same properties.

    Args:
        H_bldg: Building height in meters.
        plan_frac: Plan area fraction (0-1).
        W_road: Road width in meters.
        W_roof: Roof width in meters.
        roof_mat_id: Material ID for roof.
        wall_mat_id: Material ID for walls.
        road_mat_id: Material ID for roads.
        ah_profile_id: Anthropogenic heat profile ID.

    Returns:
        Function that takes (i, j) and returns row dict.
    """
    def fn(i, j):
        return dict(i=i, j=j, bldg_id=1, height_m=H_bldg,
                    plan_area_frac=plan_frac,
                    W_road_m=W_road, W_roof_m=W_roof,
                    roof_mat_id=roof_mat_id, wall_mat_id=wall_mat_id,
                    road_mat_id=road_mat_id, orientation_deg=0.0,
                    ah_profile_id=ah_profile_id, is_urban=1)
    return fn


def checkerboard_materials(base=None, mat_a=1,
                           mat_b=2) -> Callable[[int, int], Mapping]:
    """Alternate roof/wall mat_id in a checkerboard on top of base.

    Args:
        base: Base generator function (default: uniform_urban()).
        mat_a: Material ID for cells where (i+j) % 2 == 0.
        mat_b: Material ID for cells where (i+j) % 2 == 1.

    Returns:
        Function that takes (i, j) and returns row dict.
    """
    base = base or uniform_urban()
    def fn(i, j):
        row = dict(base(i, j))
        m = mat_a if (i + j) % 2 == 0 else mat_b
        row["roof_mat_id"] = m
        row["wall_mat_id"] = m
        return row
    return fn


def with_nonurban_box(base, i0, i1, j0, j1) -> Callable[[int, int], Mapping]:
    """Punch a non-urban rectangle [i0,i1) x [j0,j1) into base.

    Args:
        base: Base generator function.
        i0, i1: Row range (inclusive min, exclusive max).
        j0, j1: Column range (inclusive min, exclusive max).

    Returns:
        Function that takes (i, j) and returns row dict.
    """
    def fn(i, j):
        if i0 <= i < i1 and j0 <= j < j1:
            return dict(i=i, j=j, bldg_id=1, height_m=0.0,
                        plan_area_frac=0.0, W_road_m=0.0, W_roof_m=0.0,
                        roof_mat_id=0, wall_mat_id=0, road_mat_id=0,
                        orientation_deg=0.0, ah_profile_id=0, is_urban=0)
        return base(i, j)
    return fn


def two_halves_heights(H_short=5.0, H_tall=25.0, split_axis="i",
                       split_frac=0.5, roof_mat_short=1,
                       roof_mat_tall=2) -> Callable[[int, int], Callable]:
    """Left/bottom half short, right/top half tall. For scale-aware tests.

    Args:
        H_short: Building height for short half in meters.
        H_tall: Building height for tall half in meters.
        split_axis: "i" for vertical split, "j" for horizontal split.
        split_frac: Fraction along split_axis where transition occurs.
        roof_mat_short: Material ID for short buildings.
        roof_mat_tall: Material ID for tall buildings.

    Returns:
        Factory function that takes (nx_ucm, ny_ucm) and returns cell_fn.
    """
    def factory(nx_ucm, ny_ucm):
        split_thresh = int((nx_ucm if split_axis == "i" else ny_ucm)
                           * split_frac)
        def inner(i, j):
            val = i if split_axis == "i" else j
            is_tall = val >= split_thresh
            H = H_tall if is_tall else H_short
            m = roof_mat_tall if is_tall else roof_mat_short
            return dict(i=i, j=j, bldg_id=1, height_m=H,
                        plan_area_frac=0.5, W_road_m=10.0, W_roof_m=10.0,
                        roof_mat_id=m, wall_mat_id=m, road_mat_id=1,
                        orientation_deg=0.0, ah_profile_id=0, is_urban=1)
        return inner
    return factory
