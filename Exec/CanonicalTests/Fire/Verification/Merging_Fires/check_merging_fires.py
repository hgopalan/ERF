#!/usr/bin/env python3
"""Merging_Fires: two fires growing into each other, against geometry.

    python3 check_merging_fires.py two_discs

Two discs of radius r_ig a distance d apart, spreading at a constant rate R: at
time t the burned region is the union of two discs of radius r = r_ig + R t, so
the arrival time is (min distance to the two centres - r_ig) / R and, once
they overlap (r > d/2), the area is 2 pi r^2 minus the lens
2 r^2 acos(d / 2r) - (d/2) sqrt(4 r^2 - d^2). The neck where they meet is a
pair of inward cusps that the front must fill without rounding ahead of them.

The schedule may stamp the ignitions at the first step rather than at t = 0;
the offset t0 is taken from the arrival time inside the discs and reported.
The checks: the burned area at every plotfile to half a cell width times the
perimeter, the arrival time over the grid, and along the line joining the
centres, where the two fronts meet at the midpoint at t0 + (d/2 - r_ig) / R.
"""
import glob, math, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt")

# The default level set lags an exact front by a fraction of a cell: its artificial
# viscosity slows a curved front by R eps kappa, a lag of about eps ln(r1/r0) as a
# circle grows from r0 to r1 (eps = erf.fire.levelset.eps_visc_front near the front),
# and a thin ignition is stamped to within a fraction of a cell. The tolerances
# allow half a cell-crossing time on average and one cell at the 95th percentile.
TOL_MEAN = 0.5    # mean |arrival error|, in cell-crossing times h/R
TOL_P95  = 1.0    # 95th percentile of |arrival error|, in cell-crossing times
TOL_AREA = 0.5    # |area error| in cell widths times the perimeter
results = []

def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {name:28s} {'PASS' if ok else 'FAIL'}  {detail}")

def plotfiles(v):
    return sorted(glob.glob(f"plt_fire_{v}_?????")) or sorted(glob.glob("plt_fire_?????"))

def load(pf):
    ds = yt.load(pf)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    f = {n: np.asarray(g[("boxlib", n)])[:, :, 0] for n in ("fire_phi", "fire_arrival_time", "fire_ros")}
    lo = ds.domain_left_edge.d; dx = (ds.domain_right_edge.d - lo) / ds.domain_dimensions
    x = lo[0] + (np.arange(f["fire_phi"].shape[0]) + 0.5) * dx[0]
    y = lo[1] + (np.arange(f["fire_phi"].shape[1]) + 0.5) * dx[1]
    X, Y = np.meshgrid(x, y, indexing="ij")
    return float(ds.current_time), f, dx[0], X, Y, ds.domain_right_edge.d

def arrival(label, at, T, sel, crossing, tol_mean=TOL_MEAN, tol_p95=TOL_P95):
    sel = sel & (at >= 0.0)
    if sel.sum() < 20:
        check(label, False, f"only {int(sel.sum())} cells to compare"); return
    e = (at[sel] - T[sel]) / crossing
    m, p = np.abs(e).mean(), np.percentile(np.abs(e), 95)
    check(label, m <= tol_mean and p <= tol_p95,
          f"{int(sel.sum())} cells: error mean {e.mean():+.3f}, mean |e| {m:.3f}, 95th pct {p:.3f} "
          f"cell crossings ({crossing:.2f} s)")

def burned_area(phi, h):
    return np.clip(0.5 - phi / h, 0.0, 1.0).sum() * h * h

def finish(case):
    n_fail = results.count(False)
    print(f"{case}: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)

R, R_IG, X1, X2, YC = 1.0, 6.0, 160.0, 240.0, 100.0

def union_area(r, d):
    if r <= d / 2:
        return 2 * math.pi * r * r
    return 2 * math.pi * r * r - (2 * r * r * math.acos(d / (2 * r)) - 0.5 * d * math.sqrt(4 * r * r - d * d))

def main():
    d = X2 - X1
    for v in (sys.argv[1:] or ["two_discs"]):
        print(f"{v}:")
        pfs = plotfiles(v)
        if not pfs:
            check("plotfiles", False, "none"); continue
        t, f, h, X, Y, hi = load(pfs[-1])
        at = f["fire_arrival_time"]
        dmin = np.minimum(np.hypot(X - X1, Y - YC), np.hypot(X - X2, Y - YC))
        core = (dmin < R_IG - h) & (at >= 0)
        t0 = float(np.median(at[core])) if core.any() else 0.0
        print(f"  ignition stamped at t0 = {t0:.2f} s")
        for pf in pfs[1:]:
            tt, ff, h, X, Y, hi = load(pf)
            r = R_IG + R * max(tt - t0, 0.0)
            A_ex = union_area(r, d)
            dr = 1e-3
            perim = (union_area(r + dr, d) - union_area(r - dr, d)) / (2 * dr)
            A_num = burned_area(ff["fire_phi"], h)
            check(f"t = {tt:4.0f} s area", abs(A_num - A_ex) <= TOL_AREA * h * perim,
                  f"{A_num:.1f} m2 vs {A_ex:.1f} m2 ({(A_num - A_ex) / (h * perim):+.3f} cell widths of perimeter)")
        T = t0 + (dmin - R_IG) / R
        sel = (dmin > R_IG + 2 * h) & (T < t - 2 * h / R) & (X > 3 * h) & (X < hi[0] - 3 * h) \
              & (Y > 3 * h) & (Y < hi[1] - 3 * h)
        arrival("arrival over the grid", at, T, sel, h / R)
        neck = sel & (np.abs(X - 0.5 * (X1 + X2)) < 10.0)
        arrival("neck between the fires", at, T, neck, h / R, tol_mean=0.5, tol_p95=1.5)
    finish("Merging_Fires")

if __name__ == "__main__":
    main()
