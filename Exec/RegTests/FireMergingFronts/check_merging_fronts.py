#!/usr/bin/env python3
"""FireMergingFronts: three ways fires meet, against the geometry of their ignitions.

    python3 check_merging_fronts.py                      # every plt_fire_<variant>_00040 present
    python3 check_merging_fronts.py parallel [plotfile]  # one variant

With a prescribed rate R = 1 m/s in still air, the burned region at time t is
every point within R t of the ignition (plus the ignition's own half width w),
so a point burns at T = (d - w) / R, with d its distance to the nearest
ignition over ALL the ignitions of the deck:

  coalescing  four 6 m discs from the schedule: d is the distance to the
              nearest centre and w = 6 m, after the stamp time t0 (the arrival
              time inside the discs; the schedule stamps them in the first step)
  junction    one polyline, a V of two 70 m arms at 60 degrees, w = 4 m; on the
              bisector the meeting point runs at R / sin(30 deg) = 2 m/s
  parallel    two polylines in two files, y = 40 and 80 m, w = 4 m; the strip
              between them closes on y = 60 m at 16 s

The checks, each naming the defect it guards:
  ignition stamped   every cell well inside an ignition has arrival time t0:
                     a file or schedule row that was not read leaves its cells
                     unburned (the old code read one perimeter file only)
  arrival time       mean |error| at most half a cell crossing (h/R = 2 s) and
                     the 95th percentile at most one, over the cells the fronts
                     reached: the fronts spread at the prescribed rate
  merge region       the same on the cells between the fires, where the fronts
                     meet: the neck fills at the geometric rate. The signed
                     mean error is held to a quarter of a crossing, which is
                     what tells a fire running ahead from one held back (the
                     one-file mutant is late by half a crossing on one side
                     of the strip: a signed bias of +0.5)
  not yet reached    cells more than two crossings beyond the final front are
                     unburned: nothing ignites ahead of the fronts
A pure-Python reader (erf_plotfile.py) reads the fire plotfile; no yt.
"""
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                                    # the CTest copy of the reader
sys.path.append(os.path.join(HERE, "..", "..", "CanonicalTests", "Canonical_RANS"))  # from the source folder
import erf_plotfile  # noqa: E402

TOL_MEAN = 0.5      # mean |arrival error|, in cell crossings h/R
TOL_P95 = 1.0       # 95th percentile of |arrival error|, in cell crossings
TOL_BIAS = 0.25     # |signed mean error|: a fire running ahead or held back
VARIANTS = ("coalescing", "junction", "parallel")

results = []


def check(name, ok, detail=""):
    ok = bool(ok)
    results.append(ok)
    print(f"  {name:20s} {'PASS' if ok else 'FAIL'}  {detail}")
    return ok


def segment_dist(px, py, ax, ay, bx, by):
    ux, uy = bx - ax, by - ay
    if ux * ux + uy * uy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * ux + (py - ay) * uy) / (ux * ux + uy * uy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * ux), py - (ay + t * uy))


def polyline_dist(px, py, pts):
    return min(segment_dist(px, py, a[0], a[1], b[0], b[1]) for a, b in zip(pts, pts[1:]))


def read_vertices(name):
    pts = []
    with open(name) as fh:
        for line in fh:
            line = line.replace(",", " ").strip()
            if not line or line[0] in "#!":
                continue
            x, y = line.split()[:2]
            pts.append((float(x), float(y)))
    return pts


def geometry(variant, R, t_end):
    """Distance function d(x, y) to the nearest ignition, its half width w, and
    the cells of the merge region as a predicate on (x, y)."""
    if variant == "coalescing":
        centres = []   # (cx, cy, radius) of each schedule row: time cx cy radius
        with open("coalescing.csv") as fh:
            for line in fh:
                line = line.replace(",", " ").strip()
                if not line or line[0] in "#!":
                    continue
                t, cx, cy, r = [float(v) for v in line.split()[:4]]
                centres.append((cx, cy, r))
        radius = centres[0][2]
        assert all(abs(c[2] - radius) < 1e-12 for c in centres)
        d = lambda x, y: min(math.hypot(x - cx, y - cy) for cx, cy, _ in centres)
        # the necks: between the three on y = 60 (x = 80 and 120) and between the middle one and the north one
        merge = lambda x, y: (abs(y - 60.0) < 3.0 and (abs(x - 80.0) < 5.0 or abs(x - 120.0) < 5.0)) \
            or (abs(x - 100.0) < 3.0 and abs(y - 74.0) < 5.0)
        return d, radius, merge
    if variant == "junction":
        pts = read_vertices("junction_v60.csv")
        d = lambda x, y: polyline_dist(x, y, pts)
        w = 4.0
        apex_x, apex_y = pts[1]
        # half the opening angle, from the two arms
        a0 = math.atan2(pts[0][1] - apex_y, pts[0][0] - apex_x)
        a2 = math.atan2(pts[2][1] - apex_y, pts[2][0] - apex_x)
        half = 0.5 * abs(a0 - a2)
        # the bisector inside the wedge, from where the inner edges first meet
        # (w / sin(theta/2) from the apex) to where the meeting point is at t_end
        x_first = apex_x + w / math.sin(half)
        x_last = apex_x + (w + R * t_end) / math.sin(half)
        merge = lambda x, y: abs(y - apex_y) < 3.0 and x_first + w < x < x_last
        return d, w, merge
    if variant == "parallel":
        south = read_vertices("parallel_south.csv")
        north = read_vertices("parallel_north.csv")
        d = lambda x, y: min(polyline_dist(x, y, south), polyline_dist(x, y, north))
        merge = lambda x, y: abs(y - 60.0) < 3.0 and 60.0 < x < 180.0
        return d, 4.0, merge
    raise ValueError(variant)


def run(variant, plotfile):
    print(f"{variant}: {plotfile}")
    try:
        hdr, f = erf_plotfile.read_fields(plotfile, ["fire_arrival_time", "fire_phi", "fire_ros"])
    except KeyError as e:
        check("fields", False, str(e))
        return
    at = f["fire_arrival_time"]
    nx, ny = len(at), len(at[0])
    lo, dx = hdr["prob_lo"], hdr["dx"]
    hi = hdr["prob_hi"]
    h = dx[0]
    t_end = hdr["time"]
    # the prescribed rate, from the plotfile rather than the deck
    ros = [f["fire_ros"][i][j][0] for i in range(nx) for j in range(ny)]
    R = max(ros)
    check("prescribed rate", R > 0.0 and min(ros) >= 0.0 and max(ros) - min(r for r in ros if r > 0.0) < 1e-9 * R,
          f"R = {R:.3f} m/s on every burnable cell")
    crossing = h / R
    d, w, merge = geometry(variant, R, t_end)

    X = [lo[0] + (i + 0.5) * dx[0] for i in range(nx)]
    Y = [lo[1] + (j + 0.5) * dx[1] for j in range(ny)]
    A = [[at[i][j][0] for j in range(ny)] for i in range(nx)]
    D = [[d(X[i], Y[j]) for j in range(ny)] for i in range(nx)]

    # the stamp time: 0 for a perimeter file, the first step's window end for the schedule
    inside = [A[i][j] for i in range(nx) for j in range(ny) if D[i][j] < w - h]
    stamped = [a for a in inside if a >= 0.0]
    t0 = min(stamped) if stamped else float("nan")
    check("ignition stamped", len(stamped) == len(inside) and len(inside) > 0
          and t0 <= 0.5 and max(stamped) - t0 < 1e-9,
          f"{len(stamped)}/{len(inside)} cells inside the ignitions burned, stamp time {t0:.2f} s")
    if not stamped:
        return

    def T(i, j):
        return t0 + (D[i][j] - w) / R

    def interior(x, y, m):
        return lo[0] + m < x < hi[0] - m and lo[1] + m < y < hi[1] - m

    def arrival(label, sel, tol_mean=TOL_MEAN, tol_p95=TOL_P95, tol_bias=TOL_BIAS):
        e = sorted(abs(A[i][j] - T(i, j)) / crossing for (i, j) in sel)
        if len(e) < 20:
            check(label, False, f"only {len(e)} cells to compare")
            return
        mean = sum(e) / len(e)
        p95 = e[min(len(e) - 1, int(math.ceil(0.95 * len(e))) - 1)]
        bias = sum((A[i][j] - T(i, j)) / crossing for (i, j) in sel) / len(e)
        check(label, mean <= tol_mean and p95 <= tol_p95 and abs(bias) <= tol_bias,
              f"{len(e)} cells: signed mean {bias:+.3f}, mean |e| {mean:.3f}, 95th pct {p95:.3f} "
              f"cell crossings ({crossing:.1f} s)")

    reached = [(i, j) for i in range(nx) for j in range(ny)
               if interior(X[i], Y[j], 3 * h) and D[i][j] > w + 2 * h and T(i, j) < t_end - 2 * crossing
               and A[i][j] >= 0.0]
    unburned_reached = sum(1 for i in range(nx) for j in range(ny)
                           if interior(X[i], Y[j], 3 * h) and D[i][j] > w + 2 * h
                           and T(i, j) < t_end - 2 * crossing and A[i][j] < 0.0)
    check("fronts reached", unburned_reached == 0,
          f"{unburned_reached} cells the fronts should have passed are unburned")
    arrival("arrival time", reached)
    arrival("merge region", [(i, j) for (i, j) in reached if merge(X[i], Y[j])])
    ahead = sum(1 for i in range(nx) for j in range(ny)
                if T(i, j) > t_end + 2 * crossing and A[i][j] >= 0.0)
    check("not yet reached", ahead == 0, f"{ahead} cells beyond the final front are burned")


def main():
    args = sys.argv[1:]
    if args:
        variant = args[0]
        pfs = [args[1]] if len(args) > 1 else sorted(glob.glob(f"plt_fire_{variant}_?????"))
        if not pfs:
            check(variant, False, "no plotfile")
        else:
            run(variant, pfs[-1])
    else:
        found = False
        for variant in VARIANTS:
            pfs = sorted(glob.glob(f"plt_fire_{variant}_?????"))
            if pfs:
                found = True
                run(variant, pfs[-1])
        if not found:
            check("plotfiles", False, "no plt_fire_<variant>_NNNNN found")
    n_fail = results.count(False)
    print(f"FireMergingFronts: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
