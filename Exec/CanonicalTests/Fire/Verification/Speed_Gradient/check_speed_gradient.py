#!/usr/bin/env python3
"""Speed_Gradient: a point fire where the rate of spread varies linearly, against the closed form.

    python3 check_speed_gradient.py gradient

For a speed that varies linearly, R(p) = R0 + g . (p - s), the eikonal equation
|grad T| = 1/R has the closed-form travel time from a point source s (the
constant-velocity-gradient result of seismology: rays are circular arcs and the
fronts are circles whose centres move up the gradient),

    T(p) = (1/|g|) acosh(1 + |g|^2 |p - s|^2 / (2 R(s) R(p))).

Along the gradient it reduces to (1/|g|) ln(R(p)/R(s)), and for |g| -> 0 to |p - s|/R0.
The ignition is a disc of radius r_ig rather than a point; subtracting
r_ig / R(s) is exact to O(|g| r_ig^2 / R), 0.06 s here. The checks: the rate
field itself; the arrival time over the grid and separately up and down the
gradient; and the front's extent at the last plotfile towards +x and -x
against the positions where T equals the time.
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

def finish(case):
    n_fail = results.count(False)
    print(f"{case}: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)

R0, GX, SX, SY, R_IG = 1.0, 0.004, 200.0, 200.0, 4.0

def main():
    for v in (sys.argv[1:] or ["gradient"]):
        print(f"{v}: R = {R0} + {GX} (x - {SX}) m/s")
        pfs = plotfiles(v)
        if not pfs:
            check("plotfiles", False, "none"); continue
        t, f, h, X, Y, hi = load(pfs[-1])
        at, ros, phi = f["fire_arrival_time"], f["fire_ros"], f["fire_phi"]
        Rp = R0 + GX * (X - SX)
        check("rate field", np.allclose(ros, np.maximum(Rp, 0.0), rtol=0, atol=1e-10),
              f"max |fire_ros - R(x)| = {np.abs(ros - np.maximum(Rp, 0.0)).max():.2e} m/s")
        dist2 = (X - SX) ** 2 + (Y - SY) ** 2
        T = np.arccosh(1.0 + GX * GX * dist2 / (2.0 * R0 * Rp)) / GX - R_IG / R0
        edge = (X > 3 * h) & (X < hi[0] - 3 * h) & (Y > 3 * h) & (Y < hi[1] - 3 * h)
        sel = edge & (np.sqrt(dist2) > R_IG + 3 * h) & (T < t - 2 * h / np.maximum(Rp, 1e-3)) & (Rp > 0.25)
        crossing = h / R0
        arrival("whole field", at, T, sel, crossing)
        arrival("up the gradient (x > s)", at, T, sel & (X > SX + 10.0), crossing)
        arrival("down the gradient (x < s)", at, T, sel & (X < SX - 10.0), crossing, tol_mean=0.5, tol_p95=1.5)
        row = np.argmin(np.abs(Y[0, :] - SY))
        for sgn, name in ((1, "east"), (-1, "west")):
            xs = X[:, row]; line = phi[:, row]
            m = (sgn * (xs - SX) > 0)
            xb = xs[m][np.argmin(np.abs(line[m]))]
            # exact extent: solve T(x) = t along the row
            xx = np.linspace(SX, SX + sgn * 199.0, 20001)
            rr = R0 + GX * (xx - SX)
            TT = np.where(rr > 0, np.log(np.maximum(rr, 1e-12) / R0) / GX, np.inf) * sgn - R_IG / R0
            TT = np.abs(np.log(np.maximum(rr, 1e-12) / R0) / GX) - R_IG / R0
            xe = xx[np.argmin(np.abs(TT - t))]
            check(f"front extent {name}", abs(xb - xe) <= 2 * h,
                  f"front at x = {xb:.1f} m vs {xe:.1f} m from T = ln(R/R0)/g ({(xb - xe) / h:+.2f} cells)")
    finish("Speed_Gradient")

if __name__ == "__main__":
    main()
