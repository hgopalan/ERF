#!/usr/bin/env python3
"""Junction_Fire: two fire lines meeting at an angle, against geometry.

    python3 check_junction_fire.py v30 v60 v90

With a constant rate R every point burns when the front, which is the set of
points within w + R t of the ignition polyline, reaches it:
T = (distance to the V - w) / R. Inside the wedge the two inner fronts meet on
the bisector, and that meeting point advances at R / sin(theta/2), faster the
narrower the angle: the geometric "jump" of a junction fire. Viegas et al.
(2012, Int. J. Wildland Fire 21, 843-856) measured junction fires without wind
or slope in the laboratory and found the meeting point running faster still,
from the convective interaction of the flames, which a one-way model with a
prescribed rate leaves out by design.

The checks: the arrival time everywhere the nearest point of the V lies inside
the domain, three cells from the boundary, and separately along the bisector
inside the wedge; and the meeting point's speed from a straight-line fit of the
bisector arrival times, against R / sin(theta/2).
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

R, W, AX, AY, ARM = 1.0, 4.0, 40.0, 100.0, 180.0

def dist_to_polyline(X, Y, pts):
    best = np.full(X.shape, np.inf); fx = np.zeros(X.shape); fy = np.zeros(X.shape)
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        ux, uy = bx - ax, by - ay
        s = np.clip(((X - ax) * ux + (Y - ay) * uy) / (ux * ux + uy * uy), 0.0, 1.0)
        px, py = ax + s * ux, ay + s * uy
        d = np.hypot(X - px, Y - py)
        m = d < best
        best, fx, fy = np.where(m, d, best), np.where(m, px, fx), np.where(m, py, fy)
    return best, fx, fy

def main():
    for v in (sys.argv[1:] or ["v30", "v60", "v90"]):
        ang = float(v[1:]); half = math.radians(ang / 2.0)
        print(f"{v}: {ang:g} degrees, meeting point at R/sin(theta/2) = {R / math.sin(half):.3f} m/s")
        pfs = plotfiles(v)
        if not pfs:
            check("plotfiles", False, "none"); continue
        t, f, h, X, Y, hi = load(pfs[-1])
        at = f["fire_arrival_time"]
        pts = [(AX + ARM * math.cos(half), AY + ARM * math.sin(half)), (AX, AY),
               (AX + ARM * math.cos(half), AY - ARM * math.sin(half))]
        d, fx, fy = dist_to_polyline(X, Y, pts)
        T = (d - W) / R
        inside = lambda a, b, m: (a > m) & (a < hi[0] - m) & (b > m) & (b < hi[1] - m)
        sel = inside(X, Y, 3 * h) & inside(fx, fy, 3 * h) & (d > W + 2 * h) & (T < t - 2 * h / R)
        arrival("whole field", at, T, sel, h / R)
        wedge = sel & (np.abs(Y - AY) < 1.5 * h) & (X > AX + W / math.sin(half) + 2 * h) \
                & (fx < pts[0][0] - 3 * h)
        arrival("bisector inside the wedge", at, T, wedge, h / R, tol_mean=0.5, tol_p95=1.5)
        xs, ts = X[wedge & (at >= 0)], at[wedge & (at >= 0)]
        if xs.size > 5:
            speed = 1.0 / np.polyfit(xs, ts, 1)[0]
            exact = R / math.sin(half)
            check("meeting-point speed", abs(speed / exact - 1.0) < 0.02,
                  f"{speed:.3f} m/s from {xs.size} bisector cells vs R/sin(theta/2) = {exact:.3f} m/s "
                  f"({(speed / exact - 1) * 100:+.2f} %)")
    finish("Junction_Fire")

if __name__ == "__main__":
    main()
