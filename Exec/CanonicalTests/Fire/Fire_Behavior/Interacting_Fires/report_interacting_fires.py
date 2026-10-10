#!/usr/bin/env python3
"""Interacting_Fires: when and where the fires meet, from the fire plotfiles.

    python3 report_interacting_fires.py [coalescing junction parallel]

For every variant and every fire plotfile plt_fire_<variant>_NNNNN: the burned
area and the arrival time at the meeting cells, the places the fronts of two
fires reach from opposite sides. The first plotfile at which a meeting cell is
burned is the meeting time (to the plotfile interval). The meeting cells:

  coalescing  the midpoints between the five spot fires and their nearest neighbours
  junction    the bisector of the V, 40 to 160 m from the apex
  parallel    the strip between the lines, y = 1000 m, every 50 m along it

It also prints the reference wind the fire saw (fire_wind_ref_u) at the end.

This reads the plotfiles with the pure-Python reader of Canonical_RANS; no yt.
"""
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "Canonical_RANS"))
import erf_plotfile  # noqa: E402

VARIANTS = ("coalescing", "junction", "parallel")


def read_schedule(name):
    rows = []
    with open(name) as fh:
        for line in fh:
            line = line.replace(",", " ").strip()
            if not line or line[0] in "#!":
                continue
            t, cx, cy, r = [float(v) for v in line.split()[:4]]
            rows.append((t, cx, cy, r))
    return rows


def meeting_points(variant):
    if variant == "coalescing":
        spots = read_schedule("coalescing.csv")
        pts = []
        for a in range(len(spots)):
            others = sorted(range(len(spots)), key=lambda b: math.hypot(spots[a][1] - spots[b][1], spots[a][2] - spots[b][2]))
            b = others[1]
            pts.append((0.5 * (spots[a][1] + spots[b][1]), 0.5 * (spots[a][2] + spots[b][2])))
        return sorted(set(pts))
    if variant == "junction":
        return [(500.0 + 40.0 * n, 1000.0) for n in range(1, 5)]
    if variant == "parallel":
        return [(550.0 + 50.0 * n, 1000.0) for n in range(0, 5)]
    raise ValueError(variant)


def report(variant):
    pfs = sorted(glob.glob(f"plt_fire_{variant}_?????"))
    if not pfs:
        print(f"{variant}: no plotfiles")
        return
    pts = meeting_points(variant)
    print(f"{variant}: {len(pfs)} plotfiles; meeting cells " + ", ".join(f"({x:.0f}, {y:.0f})" for x, y in pts))
    first = {}
    for pf in pfs:
        hdr, f = erf_plotfile.read_fields(pf, ["fire_phi", "fire_arrival_time", "fire_wind_ref_u"])
        phi, at, uref = f["fire_phi"], f["fire_arrival_time"], f["fire_wind_ref_u"]
        lo, dx = hdr["prob_lo"], hdr["dx"]
        nx, ny = len(phi), len(phi[0])
        area = sum(1 for i in range(nx) for j in range(ny) if phi[i][j][0] < 0.0) * dx[0] * dx[1]
        cells = []
        for x, y in pts:
            i, j = int((x - lo[0]) / dx[0]), int((y - lo[1]) / dx[1])
            a = at[i][j][0]
            cells.append(a)
            if a >= 0.0 and (x, y) not in first:
                first[(x, y)] = (hdr["time"], a)
        print(f"  t = {hdr['time']:7.1f} s  burned area {area / 1.0e4:8.2f} ha  meeting cells: "
              + " ".join(f"{a:7.1f}" if a >= 0 else "      -" for a in cells))
    burning = [uref[i][j][0] for i in range(nx) for j in range(ny) if phi[i][j][0] < 0.0]
    print(f"  reference wind over the burned cells at the end: {min(burning):.2f} to {max(burning):.2f} m/s")
    for (x, y), (t_pf, a) in sorted(first.items()):
        print(f"  ({x:.0f}, {y:.0f}) burned by the {t_pf:.0f} s plotfile, arrival time {a:.1f} s")
    if len(first) < len(pts):
        print(f"  {len(pts) - len(first)} meeting cell(s) not reached by {hdr['time']:.0f} s")


def main():
    for v in (sys.argv[1:] or VARIANTS):
        report(v)


if __name__ == "__main__":
    main()
