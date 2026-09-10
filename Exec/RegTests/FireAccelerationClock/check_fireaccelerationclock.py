#!/usr/bin/env python3
"""FireAccelerationClock: head of a point fire under the temporal acceleration.

    python3 check_fireaccelerationclock.py [farsite_off farsite_legacy farsite_front
                                            levelset_off levelset_legacy levelset_front]

A 50 m radius disc centred at (500, 960) m burns short grass in a uniform 3 m/s
westerly for 10 minutes. With the temporal acceleration R = R_E (1 - exp(-A t)),
A = 1 /min here, a fire that carries its ignition clock has a head rate equal to
its path's unaccelerated head rate times 1 - exp(-A t), so by time t its head
has covered the share

    S(t) = 1 - (1 - exp(-A t)) / (A t)

of the distance the unaccelerated head covers, whatever the path makes of R_E
(the legacy FARSITE update runs at about twice the rate of spread, the
level-set path at the rate). Each deck is measured against its path's "_off"
deck.

The head is read from fire_arrival_time in each deck's last plotfile along the
two rows of cells either side of y = 960 m: H(t) is the largest x reached by
time t, interpolated between the cells that set a new record, less its value at
t = 0.

front   erf.fire.accel.clock = "front": H(t) / H_off(t) within 0.03 plus one
        cell over H_off(t) of S(t) at every whole minute from 2 minutes on.
legacy  the per-cell clock before 2026-09. Reported, not checked.
"""
import glob, math, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt")

A = 1.0 / 60.0              # erf.fire.accel.A_point, 1/s
YC = 960.0
DEFAULT = ["farsite_off", "farsite_legacy", "farsite_front",
           "levelset_off", "levelset_legacy", "levelset_front"]
CHECK_TIMES = [120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 480.0, 540.0, 600.0]
REPORT_TIMES = [60.0] + CHECK_TIMES
WINDOWS = [(0.0, 120.0), (120.0, 240.0), (240.0, 360.0), (360.0, 600.0)]
TOL = 0.03
results = []


def check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    results.append(bool(ok))


def share(t):
    return 1.0 - (-math.expm1(-A * t)) / (A * t)


def window_factor(t1, t2):
    """Mean of 1 - exp(-A t) over [t1, t2]."""
    return 1.0 - (math.exp(-A * t1) - math.exp(-A * t2)) / (A * (t2 - t1))


def head_curve(variant):
    pfs = sorted(p for p in glob.glob(f"plt_fire_{variant}_?????") if "." not in p)
    if not pfs:
        return None
    ds = yt.load(pfs[-1])
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    at = np.asarray(g[("boxlib", "fire_arrival_time")])[:, :, 0]
    nx, ny = at.shape
    x0, y0 = float(ds.domain_left_edge[0]), float(ds.domain_left_edge[1])
    dx = float(ds.domain_right_edge[0] - ds.domain_left_edge[0]) / nx
    X = x0 + (np.arange(nx) + 0.5) * dx
    Y = y0 + (np.arange(ny) + 0.5) * dx
    rows = np.where(np.abs(Y - YC) <= dx)[0]
    cells = sorted((at[i, j], X[i]) for j in rows for i in range(nx) if at[i, j] >= 0.0)
    rec_t, rec_x, best = [], [], -1.0e30
    for t, x in cells:
        if x > best:
            best = x
            rec_t.append(t)
            rec_x.append(x)
    rec_t, rec_x = np.array(rec_t), np.array(rec_x)
    h0 = rec_x[rec_t <= 1.0e-9].max()
    t_end = float(ds.current_time)
    return (lambda t: float(np.interp(t, rec_t, rec_x)) - h0), t_end, dx


def main():
    variants = sys.argv[1:] or DEFAULT
    curves = {v: head_curve(v) for v in variants}
    missing = [v for v, c in curves.items() if c is None]
    if missing:
        print(f"no plotfile for {missing}")

    for path in ("farsite", "levelset"):
        ref = f"{path}_off"
        if curves.get(ref) is None:
            continue
        H_off, t_off, dx = curves[ref]
        others = [v for v in (f"{path}_legacy", f"{path}_front") if curves.get(v) is not None]
        print(f"\n{path}: head advance [m] and share of the unaccelerated advance")
        print("   t [s]    S(t)" + f"{'off':>9s}" + "".join(f"{v.split('_')[1]:>9s}{'share':>8s}" for v in others))
        for t in REPORT_TIMES:
            if t > t_off + 1e-6:
                continue
            h_off = H_off(t)
            row = f"  {t:6.0f} {share(t):7.3f}{h_off:9.1f}"
            for v in others:
                h = curves[v][0](t)
                row += f"{h:9.1f}{h / h_off:8.3f}"
            print(row)
        print(f"  head rate [m/s] and ratio to off; ideal: mean of 1 - exp(-A t) over the window")
        print("  window [s]  ideal" + f"{'off':>9s}" + "".join(f"{v.split('_')[1]:>9s}{'ratio':>8s}" for v in others))
        for t1, t2 in WINDOWS:
            if t2 > t_off + 1e-6:
                continue
            r_off = (H_off(t2) - H_off(t1)) / (t2 - t1)
            row = f"  {t1:4.0f}-{t2:4.0f} {window_factor(t1, t2):6.3f}{r_off:9.3f}"
            for v in others:
                r = (curves[v][0](t2) - curves[v][0](t1)) / (t2 - t1)
                row += f"{r:9.3f}{r / r_off:8.3f}"
            print(row)

        front = f"{path}_front"
        if curves.get(front) is not None:
            H, t_end, _ = curves[front]
            worst = 0.0
            ok = True
            for t in CHECK_TIMES:
                if t > min(t_end, t_off) + 1e-6:
                    continue
                h_off = H_off(t)
                err = abs(H(t) / h_off - share(t))
                tol = TOL + dx / h_off
                worst = max(worst, err / tol)
                ok = ok and err <= tol
            check(f"{front} follows S(t)", ok, f"largest |H/H_off - S| is {worst:.2f} of its tolerance")

    n_fail = results.count(False)
    print(f"\nFireAccelerationClock: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
