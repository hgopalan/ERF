#!/usr/bin/env python3
"""FarsiteFrontUpdate: head, back and flank rates of the FARSITE front updates.

    python3 check_farsitefrontupdate.py [front_cell legacy]

A 50 m radius disc centred at (500, 960) m burns short grass (Anderson fuel
model 1 at 5.5 %) on flat ground in a uniform 3 m/s westerly for 10 minutes on
the FARSITE path. Its normal speed is the head rate R times the Richards (1990)
coefficients, a cos + b |sin| ahead of the wind and b |sin| - c cos behind, with
a = 1, c = 0.2 and b = 1.2 / (2 L/W) from Anderson's length-to-width ratio of the
midflame wind in mph. That is the support function of the rectangle
[-c R, a R] x [-b R, b R], so the exact burned region is the disc swept by it:
the head runs at a R, the back at c R and the flanks at b R, and the region stays
convex.

Each plotfile is measured from fire_arrival_time (burned where >= 0) on the 10 m
fire cells: head and back are the downwind and upwind edges of the burned cells
in the rows within one cell of the centre line, the half-width is half the
burned extent across the wind, and the convexity is the burned area over the
area of its convex hull. The rates are straight-line fits over the plotfiles
after step 0.

front_cell  erf.fire.farsite.front_update = "front_cell": head within 3 % of a R,
            back and flanks within 10 % of c R and b R, area at least 0.95 of
            its hull at every plotfile.
legacy      the update before 2026-09, which advanced two rows per cell of
            travel. Reported, not checked, with its ratios to front_cell.

R is the largest fire_ros in the plotfiles and the midflame wind the mean
fire_wind_eff_u, which must be uniform and steady to 1 %.
"""
import glob, math, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt")

U_WIND = 3.0
XC, YC = 500.0, 960.0
CHECKED = ("front_cell",)
TOL_HEAD = 0.03
TOL_SIDE = 0.10
MIN_CONVEX = 0.95
results = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    results.append(bool(ok))


def richards(u_mid):
    """Head, flank and back coefficients of the FARSITE path (ERF_FarsiteEllipse.H)."""
    mph = u_mid * 2.237
    lw = 1.0 if mph < 1.0 else min(max(0.936 * math.exp(0.2566 * mph) - 0.397 * math.sqrt(mph), 1.0), 8.0)
    return 1.0, 1.2 / (2.0 * lw), 0.2, lw


def hull_area(pts):
    """Area of the convex hull of 2-D points (Andrew's monotone chain)."""
    pts = np.unique(pts, axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h
    lower, upper = half(pts), half(pts[::-1])
    poly = np.array(lower[:-1] + upper[:-1])
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def measure(pf):
    ds = yt.load(pf)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    at = np.asarray(g[("boxlib", "fire_arrival_time")])[:, :, 0]
    ros = np.asarray(g[("boxlib", "fire_ros")])[:, :, 0]
    u = np.asarray(g[("boxlib", "fire_wind_eff_u")])[:, :, 0]
    v = np.asarray(g[("boxlib", "fire_wind_eff_v")])[:, :, 0]
    h = float((ds.domain_right_edge.d[0] - ds.domain_left_edge.d[0]) / ds.domain_dimensions[0])
    b = at >= 0.0
    jc = int(YC / h)
    cols = np.nonzero(b[:, jc - 1:jc + 1].any(axis=1))[0]
    rows = np.nonzero(b.any(axis=0))[0]
    inner = b.copy()
    inner[1:-1, 1:-1] = b[1:-1, 1:-1] & b[2:, 1:-1] & b[:-2, 1:-1] & b[1:-1, 2:] & b[1:-1, :-2]
    ii, jj = np.nonzero(b & ~inner)
    corners = np.concatenate([np.stack([(ii + di) * h, (jj + dj) * h], axis=1) for di in (0, 1) for dj in (0, 1)])
    return dict(t=float(ds.current_time), head=(cols.max() + 1) * h, back=cols.min() * h,
                half=0.5 * (rows.max() + 1 - rows.min()) * h, area=b.sum() * h * h,
                convex=b.sum() * h * h / hull_area(corners), ros=float(ros.max()),
                u=(float(u.min()), float(u.max())), umean=float(u.mean()), vmax=float(np.abs(v).max()))


def main():
    variants = sys.argv[1:] or ["front_cell", "legacy"]
    rates = {}
    for var in variants:
        pfs = sorted(glob.glob(f"plt_fire_{var}_?????"))
        print(f"{var}: {len(pfs)} plotfiles")
        if len(pfs) < 3:
            check("plotfiles", False, "need at least three")
            continue
        m = [measure(pf) for pf in pfs[1:]]    # the step-0 plotfile carries no wind
        umin = min(r["u"][0] for r in m); umax = max(r["u"][1] for r in m); vmax = max(r["vmax"] for r in m)
        check("uniform steady wind", umin >= 0.99 * U_WIND and umax <= 1.01 * U_WIND and vmax <= 0.01 * U_WIND,
              f"u {umin:.3f}-{umax:.3f} m/s, |v| <= {vmax:.3f} m/s")
        R = max(r["ros"] for r in m)
        a, b, c, lw = richards(np.mean([r["umean"] for r in m]))
        print("      t [s]   head x [m]   back x [m]   half-width [m]   area [ha]   convexity")
        for r in m:
            print(f"    {r['t']:7.0f}   {r['head']:10.0f}   {r['back']:10.0f}   {r['half']:14.0f}"
                  f"   {r['area'] / 1e4:9.2f}   {r['convex']:9.3f}")
        t = np.array([r["t"] for r in m])
        fit = lambda key: np.polyfit(t, [r[key] for r in m], 1)[0]
        head, back, flank = fit("head"), -fit("back"), fit("half")
        rates[var] = dict(head=head, back=back, flank=flank, area=m[-1]["area"])
        cmin = min(r["convex"] for r in m)
        print(f"  R = {R:.3f} m/s, L/W = {lw:.2f}: Richards head {a * R:.3f}, back {c * R:.3f}, flank {b * R:.3f} m/s")
        print(f"  measured head {head:.3f} ({head / (a * R):.2f} a R), back {back:.3f} ({back / (c * R):.2f} c R), "
              f"flank {flank:.3f} ({flank / (b * R):.2f} b R) m/s; smallest area / hull {cmin:.3f}")
        if var in CHECKED:
            check("head at a R", abs(head / (a * R) - 1.0) <= TOL_HEAD, f"{head:.3f} vs {a * R:.3f} m/s ({100 * (head / (a * R) - 1):+.1f} %)")
            check("back at c R", abs(back / (c * R) - 1.0) <= TOL_SIDE, f"{back:.3f} vs {c * R:.3f} m/s ({100 * (back / (c * R) - 1):+.1f} %)")
            check("flanks at b R", abs(flank / (b * R) - 1.0) <= TOL_SIDE, f"{flank:.3f} vs {b * R:.3f} m/s ({100 * (flank / (b * R) - 1):+.1f} %)")
            check("burned region convex", cmin >= MIN_CONVEX, f"smallest area / hull {cmin:.3f}")
    if "front_cell" in rates and "legacy" in rates:
        f, l = rates["front_cell"], rates["legacy"]
        print(f"  reported: legacy / front_cell  head {l['head'] / f['head']:.2f}, back {l['back'] / f['back']:.2f}, "
              f"flank {l['flank'] / f['flank']:.2f}, burned area at the end {l['area'] / f['area']:.2f}")
    n_fail = results.count(False)
    print(f"{len(results) - n_fail} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
