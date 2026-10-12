#!/usr/bin/env python3
"""UNR_WRF_Fire: head position and rate of the UNR WRF-Fire case studies against WRF.

    python3 check_unr_wrf_fire.py case1_compressible [case2_anelastic ...]
    python3 check_unr_wrf_fire.py --dir RUN_DIR --prefix plt_fire_ case1_compressible

Reads the fire_arrival_time of the last fire plotfile of each deck
(plt_fire_<deck>_?????). The head at time t is where the arrival time crosses t,
interpolated between fire-cell centres, on the rows within 300 m of the middle of
the 1 km ignition line (y = 2500 m), scanning from the ignition line in +x; the
scan stops at the domain end, so a head that has crossed the periodic x edge is
not counted. Times are after the start of the run for cases 1 and 2 (UNR's
figures) and after ignition for case 3.

Case 1 has an independent answer: on flat ground in a uniform wind the head of a
long line fire runs at Rothermel's head rate, which WRF's formula gives as
1.173 m/s for this fuel, moisture and 6.5 m wind (3.79 m/s, from the 5 m/s
sounding at the 24.6 m first cell centre by the log law below it). The check requires the head
rate between 10 and 40 min (--rate-window) within 1 % of it, and |grad phi| at
the head at the end of that window within 10 % of 1 (the level set stays a
distance function). Cases 2 and 3 have no
independent number: the positions read from UNR's figures are printed next to
the run's, and only a missing or empty output fails.
"""
import argparse, glob, math, os, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt")

# m/s, WRF's Rothermel formula, fuel category 1, M = 0.08, at the 6.5 m wind the log law
# gives below the first cell centre: 3.79 m/s from the sounding's 5 m/s at the 24.6 m first
# cell centre and z0 = 0.1 m (5 ln(65)/ln(246.4)). It holds for these decks' vertical levels
# and wind_below_first_cell = "log" only; a 5 m/s wind at 6.5 m itself would give 2.07 m/s.
ROS_WRF_CASE1 = 1.173
TOL_RATE = 0.01         # relative, case 1 head rate
TOL_GRAD = 0.10         # | |grad phi| - 1 | at the case 1 head

# Head x [km] read from UNR's figures at (minutes, after the run start for cases 1 and 2,
# after ignition for case 3).
UNR = {1: {10: 1.65, 20: 2.35, 40: 3.75},
       2: {},
       3: {10: 2.38, 20: 3.22, 30: 3.75}}
T_IGN = {1: 10.0, 2: 10.0, 3: 1800.0}
LINE_X = {1: 1050.0, 2: 1050.0, 3: 1020.0}
Y0, BAND = 2500.0, 300.0
results = []

def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {name:34s} {'PASS' if ok else 'FAIL'}  {detail}")

def load(pf):
    ds = yt.load(pf)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    ta = np.asarray(g[("boxlib", "fire_arrival_time")])[:, :, 0]
    phi = np.asarray(g[("boxlib", "fire_phi")])[:, :, 0]
    n = ta.shape; dx = float(ds.domain_width[0]) / n[0]
    return float(ds.current_time), ta, phi, dx

def head(ta, dx, t, line_x):
    """Largest head x [m] at time t over the rows of the band; -inf where nothing burned,
    nan once the head has reached the domain end."""
    n = ta.shape; x = (np.arange(n[0]) + 0.5) * dx; y = (np.arange(n[1]) + 0.5) * dx
    i0 = int(line_x / dx); best = -math.inf
    for j in np.where(abs(y - Y0) < BAND)[0]:
        col = ta[:, j]; burnt = (col >= 0.0) & (col <= t)
        if not burnt[i0]:
            continue
        i = i0
        while i + 1 < n[0] and burnt[i + 1]:
            i += 1
        if i + 1 >= n[0]:
            return math.nan                       # the head has reached the domain end
        xh = x[i]
        if col[i + 1] > t:                        # interpolate to where the arrival time equals t
            xh += dx * (t - col[i]) / (col[i + 1] - col[i])
        best = max(best, xh)
    return best

def grad_at_head(phi, dx, line_x):
    """|grad phi| where phi crosses zero on the centre row, scanning +x from the line."""
    j = int(Y0 / dx); r = phi[:, j]; i = int(line_x / dx)
    while i + 1 < phi.shape[0] and r[i + 1] < 0.0:
        i += 1
    if i + 1 >= phi.shape[0] or r[i] >= 0.0:
        return float("nan")
    gx, gy = np.gradient(phi, dx)
    w = -r[i] / (r[i + 1] - r[i])
    return float(np.hypot((1 - w) * gx[i, j] + w * gx[i + 1, j], (1 - w) * gy[i, j] + w * gy[i + 1, j]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decks", nargs="+")
    ap.add_argument("--dir", default=".", help="folder holding the fire plotfiles")
    ap.add_argument("--prefix", default=None, help="fire plotfile prefix (default plt_fire_<deck>_)")
    ap.add_argument("--rate-window", nargs=2, type=int, default=[10, 40], metavar=("M0", "M1"),
                    help="case 1: minutes after the start between which the head rate is measured")
    a = ap.parse_args()
    for deck in a.decks:
        case = int(deck[4])
        pre = a.prefix if a.prefix is not None else f"plt_fire_{deck}_"
        files = sorted(glob.glob(os.path.join(a.dir, pre + "?????")))
        print(f"{deck}: {len(files)} fire plotfiles")
        if not files:
            check("fire plotfiles", False, f"none matching {os.path.join(a.dir, pre)}?????"); continue
        tend, ta, phi, dx = load(files[-1])
        t0 = 0.0 if case < 3 else T_IGN[case]
        print(f"  last at t = {tend:.0f} s, fire cell {dx:.1f} m; head x [km] at minutes after "
              + ("the start" if case < 3 else "ignition"))
        m0, m1 = a.rate_window
        mins = sorted(set(UNR[case]) | ({10, 20, 30, 40, 50} if case < 3 else {10, 20, 30, 60, 90}) | {m0, m1})
        heads = {}
        for m in mins:
            t = t0 + 60.0 * m
            if t > tend + 1e-6:
                continue
            heads[m] = head(ta, dx, t, LINE_X[case])
            ref = UNR[case].get(m)
            if math.isnan(heads[m]):
                print(f"    {m:3d} min  ERF at the domain end"); continue
            print(f"    {m:3d} min  ERF {heads[m] / 1e3:6.3f}" + (f"   UNR {ref:5.2f}  (ERF - UNR {heads[m] / 1e3 - ref:+.2f})" if ref else ""))
        if case == 1:
            label = f"head rate {m0}-{m1} min"
            if m0 not in heads or m1 not in heads or not all(map(math.isfinite, (heads[m0], heads[m1]))):
                check(label, False, f"the run does not reach {m1} min with the head inside the domain"); continue
            rate = (heads[m1] - heads[m0]) / (60.0 * (m1 - m0))
            unr = (UNR[1][40] - UNR[1][10]) * 1e3 / 1800.0
            check(label, abs(rate / ROS_WRF_CASE1 - 1) <= TOL_RATE,
                  f"{rate:.4f} m/s, WRF Rothermel {ROS_WRF_CASE1} ({100 * (rate / ROS_WRF_CASE1 - 1):+.2f} %); "
                  f"UNR's figures 10-40 min {unr:.3f}")
            k = int(np.argmin([abs(load_time(f) - (t0 + 60.0 * m1)) for f in files]))
            tg, _, phig, _ = load(files[k])
            gn = grad_at_head(phig, dx, LINE_X[case])
            check("|grad phi| at the head", math.isfinite(gn) and abs(gn - 1) <= TOL_GRAD, f"{gn:.3f} at t = {tg:.0f} s")
        else:
            ok = any(math.isfinite(v) and v > LINE_X[case] for v in heads.values())
            check("fire spread from the line", ok, "head positions above")
            ks = sorted(heads)
            for m0, m1 in zip(ks, ks[1:]):
                if math.isfinite(heads[m0]) and math.isfinite(heads[m1]):
                    print(f"    rate {m0:3d}-{m1:3d} min  {(heads[m1] - heads[m0]) / (60.0 * (m1 - m0)):.3f} m/s")
    print("UNR_WRF_Fire:", "PASS" if results and all(results) else "FAIL")
    return 0 if results and all(results) else 1

def load_time(pf):
    L = open(os.path.join(pf, "Header")).read().split("\n"); n = int(L[1]); return float(L[2 + n + 1])

if __name__ == "__main__":
    sys.exit(main())
