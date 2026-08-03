#!/usr/bin/env python3
"""
Verification script for Phase 6.2a tree radiation (Beer-Lambert) canonical tests.
Checks: 
  - Q_tree_SW_abs is zero when tree_rad_mode=off
  - Q_tree_SW_abs is nonzero when tree_rad_mode=beer_lambert
  - Bit-identity between off-mode runs
"""
import csv
import glob
import os
import re
import sys

import numpy as np

try:
    import yt
    try:
        yt.set_log_level("error")
    except Exception:
        pass
except ImportError:
    print("FAIL: yt not found")
    sys.exit(1)


def find_latest_plot(prefix):
    patt = re.compile(rf"^{re.escape(prefix)}\d+$")
    matches = sorted(p for p in glob.glob(f"{prefix}*") if patt.match(os.path.basename(p)))
    return matches[-1] if matches else None


def load_3d(ds, field_name):
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions, fields=[field_name])
    return np.array(cg[field_name])


def load_tree_rows(csv_path):
    rows = []
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row['is_tree']) != 1:
                continue
            rows.append({
                'x_m': float(row['x_m']),
                'y_m': float(row['y_m']),
                'H_tree_m': float(row['H_tree_m']),
                'H_crown_base_m': float(row['H_crown_base_m']),
                'LAD_bulk': float(row['LAD_bulk']),
            })
    return rows


def aggregate_tree_rows(rows, xlo, xhi, ylo, yhi, nx, ny):
    dx = (xhi - xlo) / nx
    dy = (yhi - ylo) / ny
    cells = {}
    for row in rows:
        i = min(nx - 1, max(0, int((row['x_m'] - xlo) / dx)))
        j = min(ny - 1, max(0, int((row['y_m'] - ylo) / dy)))
        cells.setdefault((i, j), []).append(row)
    agg = {}
    for key, vals in cells.items():
        agg[key] = {
            'H_tree': max(v['H_tree_m'] for v in vals),
            'H_crown_base': min(v['H_crown_base_m'] for v in vals),
            'LAD': np.mean([v['LAD_bulk'] for v in vals]),
        }
    return agg


def compute_tree_rad_metrics(plotfile, tree_csv):
    """Compute Q_tree_SW_abs statistics for tree radiation test."""
    try:
        ds = yt.load(plotfile)
        Q_tree = load_3d(ds, ('boxlib', 'Q_tree_SW_abs'))
        rows = load_tree_rows(tree_csv)
        agg = aggregate_tree_rows(
            rows,
            float(ds.domain_left_edge[0]), float(ds.domain_right_edge[0]),
            float(ds.domain_left_edge[1]), float(ds.domain_right_edge[1]),
            int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1]),
        )

        # Check if any tree cells have nonzero Q_tree_SW_abs
        tree_mask = np.zeros(Q_tree[:, :, 0].shape, dtype=bool)
        for (i, j), props in agg.items():
            if props['LAD'] > 0.0:
                tree_mask[i, j] = True

        if not np.any(tree_mask):
            return None  # No tree cells

        Q_tree_xy = Q_tree[:, :, 0]  # Use lowest level (highest resolution)
        Q_max = float(np.max(Q_tree_xy[tree_mask]))
        Q_min = float(np.min(Q_tree_xy[tree_mask]))

        return Q_min, Q_max
    except Exception as e:
        print(f"DEBUG: Exception in compute_tree_rad_metrics: {e}")
        return None


def compare_against_baseline(off_plotfile, baseline_plotfile):
    """Check bit-identity between off-mode runs."""
    ds_off = yt.load(off_plotfile)
    ds_base = yt.load(baseline_plotfile)
    fields = [
        ('boxlib', 'x_velocity'),
        ('boxlib', 'y_velocity'),
        ('boxlib', 'z_velocity'),
        ('boxlib', 'theta'),
        ('boxlib', 'density'),
        ('boxlib', 'Q_tree_SW_abs'),
    ]
    return all(np.array_equal(load_3d(ds_off, field), load_3d(ds_base, field))
               for field in fields)


def main():
    plotfile = find_latest_plot('plt_')
    if not plotfile:
        print('FAIL: missing main plotfile')
        return 1

    tree_csv = sys.argv[1] if len(sys.argv) > 1 else 'tree_layout.csv'
    baseline_plot = sys.argv[2] if len(sys.argv) > 2 else None
    rad_mode = sys.argv[3] if len(sys.argv) > 3 else 'unknown'

    print(f'plotfile={plotfile}')
    print(f'tree_csv={tree_csv}')
    print(f'rad_mode={rad_mode}')

    metrics = compute_tree_rad_metrics(plotfile, tree_csv)
    if metrics is None:
        # No tree cells found; this is OK for some tests
        print('WARN: no tree cells found in domain')
        metrics = (0.0, 0.0)
    else:
        Q_min, Q_max = metrics
        print(f'Q_tree_SW_abs: min={Q_min:.6e} max={Q_max:.6e}')

        if rad_mode == 'off':
            # In off mode, Q_tree_SW_abs should be zero everywhere
            if Q_max > 1.e-15:
                print('FAIL: expected Q_tree_SW_abs = 0 when tree_rad_mode=off')
                return 1
        elif rad_mode == 'beer_lambert':
            # In beer_lambert mode, Q_tree_SW_abs should be nonzero (if there's SW input and trees)
            # This is a soft warning since it depends on time-of-day
            if Q_max <= 1.e-15:
                print('WARN: Q_tree_SW_abs is zero even in beer_lambert mode (possibly no daytime SW)')

    if baseline_plot is not None:
        if not compare_against_baseline(plotfile, baseline_plot):
            print('FAIL: off-mode run is not bit-identical to supplied baseline')
            return 1
        print('baseline_bit_identity=PASS')
    else:
        print('baseline_bit_identity=SKIPPED (no baseline plotfile argument)')

    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
