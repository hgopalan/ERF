#!/usr/bin/env python3
"""Polygon_Growth: fires started from a square and a cross, against geometry.

    python3 check_polygon_growth.py square cross

With a constant rate R the burned region at time t is every point within r = R t
of the ignition polygon, so outside the polygon T = (distance to it) / R. Its
area is exact: for the convex square Steiner's formula A0 + P0 r + pi r^2, and
for the cross, whose eight convex corners each add a quarter disc and whose
four inner (reflex) corners each lose the r x r square the two edge strips
share, A0 + P0 r + (2 pi - 4) r^2 while r is under half the arm length. The
inner corners must stay sharp: an entropy-violating or overly diffusive scheme
rounds them and burns more there.

The checks, at every plotfile: the burned area (sub-cell, from the signed
distance) against the formula to half a cell width times the perimeter; the
arrival time over the whole grid outside the polygon; and, for the cross, along
the diagonals out of the four inner corners, where T = d / (sqrt(2) R).
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

R, C = 1.0, 200.0

def polygon(v):
    pts = [tuple(map(float, l.split())) for l in open(f"{v}.csv") if l.strip() and l[0] not in "#!"]
    return pts

def dist_to_boundary(X, Y, pts):
    best = np.full(X.shape, np.inf)
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
        ux, uy = bx - ax, by - ay
        s = np.clip(((X - ax) * ux + (Y - ay) * uy) / (ux * ux + uy * uy), 0.0, 1.0)
        best = np.minimum(best, np.hypot(X - ax - s * ux, Y - ay - s * uy))
    return best

def inside(X, Y, pts):
    ins = np.zeros(X.shape, dtype=bool)
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
        crosses = ((ay > Y) != (by > Y)) & (X < ax + (Y - ay) * (bx - ax) / np.where(by == ay, 1e-30, by - ay))
        ins ^= crosses
    return ins

def main():
    for v in (sys.argv[1:] or ["square", "cross"]):
        print(f"{v}:")
        pts = polygon(v)
        A0 = 0.5 * abs(sum(ax * by - bx * ay for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1])))
        P0 = sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]))
        k2 = math.pi if v == "square" else 2 * math.pi - 4.0
        pfs = plotfiles(v)
        if not pfs:
            check("plotfiles", False, "none"); continue
        for pf in pfs[1:]:
            t, f, h, X, Y, hi = load(pf)
            r = R * t
            A_ex = A0 + P0 * r + k2 * r * r
            A_num = burned_area(f["fire_phi"], h)
            perim = P0 + 2 * k2 * r
            check(f"t = {t:4.0f} s area", abs(A_num - A_ex) <= TOL_AREA * h * perim,
                  f"{A_num:.1f} m2 vs {A_ex:.1f} m2 ({(A_num - A_ex) / (h * perim):+.3f} cell widths of perimeter)")
        at = f["fire_arrival_time"]
        d = dist_to_boundary(X, Y, pts)
        out = ~inside(X, Y, pts) & (d > 2 * h) & (d < r - 2 * h)
        arrival("arrival outside the polygon", at, d / R, out, h / R)
        if v == "cross":
            diag = np.zeros(X.shape, dtype=bool)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    ux, uy = X - (C + sx * 20.0), Y - (C + sy * 20.0)
                    diag |= (np.abs(ux * sx - uy * sy) < 1.5 * h) & (ux * sx > 2 * h) & (uy * sy > 2 * h)
            arrival("inner-corner diagonals", at, d / R, out & diag, h / R, tol_mean=0.5, tol_p95=1.5)
    finish("Polygon_Growth")

if __name__ == "__main__":
    main()
