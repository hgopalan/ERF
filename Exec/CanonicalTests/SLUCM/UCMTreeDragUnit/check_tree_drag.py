#!/usr/bin/env python3
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
                'crown_area_frac': float(row['crown_area_frac']),
                'Cd_leaf': float(row['Cd_leaf'])
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
        total_crown = sum(v['crown_area_frac'] for v in vals)
        if total_crown <= 0.0:
            total_crown = float(len(vals))
        agg[key] = {
            'H_tree': max(v['H_tree_m'] for v in vals),
            'H_crown_base': min(v['H_crown_base_m'] for v in vals),
            'LAD': sum(v['LAD_bulk'] * v['crown_area_frac'] for v in vals) / total_crown,
            'crown_area_frac': sum(v['crown_area_frac'] for v in vals) / len(vals),
            'Cd_leaf': sum(v['Cd_leaf'] * v['crown_area_frac'] for v in vals) / total_crown,
        }
    return agg


def compute_tree_drag_metrics(plotfile, tree_csv):
    ds = yt.load(plotfile)
    u = load_3d(ds, ('boxlib', 'x_velocity'))
    v = load_3d(ds, ('boxlib', 'y_velocity'))
    rho = load_3d(ds, ('boxlib', 'density'))
    rows = load_tree_rows(tree_csv)
    agg = aggregate_tree_rows(
        rows,
        float(ds.domain_left_edge[0]), float(ds.domain_right_edge[0]),
        float(ds.domain_left_edge[1]), float(ds.domain_right_edge[1]),
        int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1]),
    )
    z0 = float(ds.domain_left_edge[2])
    dz = (float(ds.domain_right_edge[2]) - float(ds.domain_left_edge[2])) / int(ds.domain_dimensions[2])
    zc = z0 + (np.arange(int(ds.domain_dimensions[2])) + 0.5) * dz

    xdrag = []
    for (i, j), props in agg.items():
        mask = (zc >= props['H_crown_base']) & (zc <= props['H_tree'])
        if not np.any(mask):
            continue
        Uh = np.sqrt(u[i, j, mask] ** 2 + v[i, j, mask] ** 2)
        coeff = -0.5 * rho[i, j, mask] * props['LAD'] * props['crown_area_frac'] * props['Cd_leaf']
        xdrag.extend((coeff * Uh * u[i, j, mask]).tolist())

    if not xdrag:
        return None
    return float(np.min(xdrag)), float(np.max(xdrag))


def compare_against_baseline(off_plotfile, baseline_plotfile):
    ds_off = yt.load(off_plotfile)
    ds_base = yt.load(baseline_plotfile)
    fields = [
        ('boxlib', 'x_velocity'),
        ('boxlib', 'y_velocity'),
        ('boxlib', 'z_velocity'),
        ('boxlib', 'theta'),
        ('boxlib', 'density'),
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

    metrics = compute_tree_drag_metrics(plotfile, tree_csv)
    if metrics is None:
        print('FAIL: no tree drag-active cells found')
        return 1

    xmin, xmax = metrics
    print(f'plotfile={plotfile}')
    print(f'tree_xmom_src_estimate: min={xmin:.6e} max={xmax:.6e}')
    if xmin >= 0.0:
        print('FAIL: expected negative tree drag in x momentum')
        return 1

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
