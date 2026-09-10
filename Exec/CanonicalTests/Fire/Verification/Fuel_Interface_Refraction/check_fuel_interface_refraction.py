#!/usr/bin/env python3
"""Fuel_Interface_Refraction: a straight front crossing a fuel boundary, against Snell's law.

    python3 check_fuel_interface_refraction.py fast_to_slow slow_to_fast

A plane front spreading at R1 with its normal at theta1 to the boundary normal
reaches the boundary x = 200 m along a line, so its arrival there grows with y
at the trace slowness sin(theta1) / R1. On the far side the fastest path from
each boundary point spreads at R2, and Huygens' principle makes the transmitted
front plane again, at the angle theta2 with sin(theta2) / R2 = sin(theta1) / R1,
Snell's law of refraction. The arrival time there is exactly

    T2 = T1(200, y_q) + (x - 200) / (R2 cos theta2),  y_q = y - (x - 200) tan theta2,

with T1 = (n1 . (p - p0) - w) / R1 the plane front of the ignition line.

With a constant rate on each side the arrival time is the minimum over the points
of the ignition line of the travel time from them, so a point is reached exactly
as from an infinite line as long as its own optimal source point lies on the
stretch of the line that is inside the domain and in the first fuel; only such
points are compared, ten metres clear of either end. Where the 40 degree line
runs on into the second fuel near the bottom of the domain, that stretch ignites
the second fuel directly, and the second-fuel arrival is the earlier of the
refracted front and the direct spread from it.

The checks: the arrival time in each fuel (in the first fuel away from the
boundary, where a front refracted back from a faster second fuel could otherwise
arrive first), and a least-squares plane through the second fuel's arrival
times, whose gradient must have magnitude 1/R2 and point along theta2.
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

X0, Y0, W, XI = 60.0, 200.0, 3.0, 200.0
CASES = {"fast_to_slow": (40.0, 1.0, 0.5), "slow_to_fast": (25.0, 0.5, 1.0)}

def main():
    for v in (sys.argv[1:] or list(CASES)):
        ang, r1, r2 = CASES[v]
        th1 = math.radians(ang); th2 = math.asin(r2 / r1 * math.sin(th1))
        print(f"{v}: R1 = {r1} m/s, R2 = {r2} m/s, theta1 = {ang} deg, Snell theta2 = {math.degrees(th2):.3f} deg")
        pfs = plotfiles(v)
        if not pfs:
            check("plotfiles", False, "none"); continue
        t, f, h, X, Y, hi = load(pfs[-1])
        at, ros = f["fire_arrival_time"], f["fire_ros"]
        n1x, n1y = math.cos(th1), math.sin(th1)
        check("rate field", np.allclose(ros[X < XI], r1) and np.allclose(ros[X > XI], r2),
              f"fuel 1 in [{ros[X < XI].min():.4g}, {ros[X < XI].max():.4g}], fuel 2 in [{ros[X > XI].min():.4g}, {ros[X > XI].max():.4g}] m/s")
        s1 = n1x * (X - X0) + n1y * (Y - Y0)
        T1 = (s1 - W) / r1
        yq = Y - (X - XI) * math.tan(th2)
        T2 = (n1x * (XI - X0) + n1y * (yq - Y0) - W) / r1 + (X - XI) / (r2 * math.cos(th2))
        # the ignition line inside the domain, split into its stretch in the first fuel
        # (an interval of u = (p - p0) . (-sin theta1, cos theta1)) and any stretch in the second
        tx, ty = -math.sin(th1), math.cos(th1)
        uu = np.linspace(-800.0, 800.0, 160001)
        lx, ly = X0 + uu * tx, Y0 + uu * ty
        indom = (lx > 0.0) & (lx < hi[0]) & (ly > 0.0) & (ly < hi[1])
        in1 = indom & (lx < XI)
        u_lo, u_hi = uu[in1].min(), uu[in1].max()
        in2 = indom & (lx >= XI)
        def on_first_stretch(src_u):
            return (src_u > u_lo + 10.0) & (src_u < u_hi - 10.0)
        edge = (X > 3 * h) & (X < hi[0] - 3 * h) & (Y > 3 * h) & (Y < hi[1] - 3 * h)
        u1 = (X - X0) * tx + (Y - Y0) * ty
        sel1 = edge & (X < XI - 20.0) & (s1 > W + 2 * h) & (T1 < t - 2 * h / r1) & on_first_stretch(u1)
        arrival("first fuel", at, T1, sel1, h / r1)
        uq = (XI - X0) * tx + (yq - Y0) * ty
        if in2.any():
            ua, ub = uu[in2].min(), uu[in2].max()
            ax_, ay_, bx_, by_ = X0 + ua * tx, Y0 + ua * ty, X0 + ub * tx, Y0 + ub * ty
            sx, sy = bx_ - ax_, by_ - ay_
            k = np.clip(((X - ax_) * sx + (Y - ay_) * sy) / (sx * sx + sy * sy), 0.0, 1.0)
            T_direct = (np.hypot(X - ax_ - k * sx, Y - ay_ - k * sy) - W) / r2
            T2 = np.minimum(T2, T_direct)
            print(f"  the line enters the second fuel for {ub - ua:.0f} m inside the domain")
        sel2 = edge & (X > XI + 3 * h) & (yq > 3 * h) & (yq < hi[1] - 3 * h) & (T2 < t - 2 * h / r2) \
               & on_first_stretch(uq)
        arrival("second fuel", at, T2, sel2, h / r2)
        m = sel2 & (at >= 0)
        if m.sum() > 50:
            A = np.column_stack([np.ones(m.sum()), X[m], Y[m]])
            c0, cx, cy = np.linalg.lstsq(A, at[m], rcond=None)[0]
            slow, angle = math.hypot(cx, cy), math.degrees(math.atan2(cy, cx))
            check("transmitted front", abs(slow * r2 - 1.0) < 0.01 and abs(angle - math.degrees(th2)) < 0.5,
                  f"|grad T| = {slow:.5f} s/m vs 1/R2 = {1 / r2:.5f}; normal at {angle:.3f} deg vs Snell {math.degrees(th2):.3f} deg")
    finish("Fuel_Interface_Refraction")

if __name__ == "__main__":
    main()
