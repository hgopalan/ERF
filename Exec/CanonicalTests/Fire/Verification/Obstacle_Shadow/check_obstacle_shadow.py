#!/usr/bin/env python3
"""Obstacle_Shadow: arrival times around a non-burnable disc against geometry.

    python3 check_obstacle_shadow.py disc30

With a constant rate R the arrival time is the shortest path from the ignition
disc to each point that avoids the obstacle, divided by R (Huygens). A point
the ignition centre s can see is reached in a straight line; behind the disc the
shortest path is the taut string: tangent from s to the disc, the arc of the
disc, tangent to the point p,

    L = sqrt(|s-o|^2 - a^2) + sqrt(|p-o|^2 - a^2) + a (gamma - acos(a/|s-o|) - acos(a/|p-o|)),

with o the disc centre, a its radius and gamma the angle between s-o and p-o;
p is in the shadow exactly when that angle term is positive. T = (L - r_ig)/R.
The mask is whole fire cells, those whose centre lies inside the disc, and the
rate is zero across each of them, so the obstacle the front actually meets is
the union of those cells: about half a cell wider than the nominal disc. Since
dL/da is the wrap angle, a path wrapping the disc by an angle w is longer by
w h/2, up to three quarters of a cell crossing here. The shadow is compared
with the taut string around a disc of radius a + h/2, and the lag against the
nominal disc is reported.

The checks: every obstacle cell stays unburned, and the arrival times in the lit
region and in the shadow agree with T, three cells away from the obstacle's edge
and the domain boundary.
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

R, SX, SY, R_IG = 1.0, 60.0, 100.0, 6.0
OX, OY, A = 160.0, 100.0, 30.0

def taut_string(X, Y, a):
    ds, dp = math.hypot(SX - OX, SY - OY), np.hypot(X - OX, Y - OY)
    cosg = ((SX - OX) * (X - OX) + (SY - OY) * (Y - OY)) / (ds * np.maximum(dp, 1e-9))
    gamma = np.arccos(np.clip(cosg, -1.0, 1.0))
    wrap = gamma - math.acos(a / ds) - np.arccos(np.clip(a / np.maximum(dp, a), -1.0, 1.0))
    shadow = (dp > a) & (wrap > 0.0)
    L = np.where(shadow, math.sqrt(ds * ds - a * a) + np.sqrt(np.maximum(dp * dp - a * a, 0.0)) + a * wrap,
                 np.hypot(X - SX, Y - SY))
    return (L - R_IG) / R, shadow

def main():
    for v in (sys.argv[1:] or ["disc30", "disc30_wallx"]):
        print(f"{v}:")
        pfs = plotfiles(v)
        if not pfs:
            check("plotfiles", False, "none"); continue
        t, f, h, X, Y, hi = load(pfs[-1])
        at, phi = f["fire_arrival_time"], f["fire_phi"]
        dp = np.hypot(X - OX, Y - OY)
        T_nominal, shadow = taut_string(X, Y, A)
        T, shadow_eff = taut_string(X, Y, A + 0.5 * h)
        obstacle = dp <= A
        check("obstacle unburned", np.all(at[obstacle] < 0.0) and np.all(phi[obstacle] >= 0.0),
              f"{int(obstacle.sum())} obstacle cells, {int((at[obstacle] >= 0).sum())} with an arrival time")
        away = (dp > A + 3 * h) & (np.hypot(X - SX, Y - SY) > R_IG + 3 * h) \
             & (X > 3 * h) & (X < hi[0] - 3 * h) & (Y > 3 * h) & (Y < hi[1] - 3 * h) & (T < t - 2 * h / R)
        arrival("lit region", at, T, away & ~shadow, h / R)
        # the front that wraps a masked obstacle lags a little further (0.44 cell crossings
        # with the default stencil, 0.52 with wall_extrapolate), so the shadow gets
        # three quarters of a cell on average and one and a half at the 95th percentile
        arrival("shadow, disc of a + h/2", at, T, away & shadow_eff, h / R, tol_mean=0.75, tol_p95=1.5)
        m = away & shadow_eff & (at >= 0)
        if m.sum() > 20:
            print(f"  against the nominal disc of radius {A:g} m the shadow is "
                  f"{((at[m] - T_nominal[m]) * R / h).mean():+.3f} cell crossings late on average")
    finish("Obstacle_Shadow")

if __name__ == "__main__":
    main()
