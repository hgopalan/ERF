#!/usr/bin/env python3
"""Checks on the fire output of the idealized Marshall Fire decks.

    python3 check_idealized_fire.py [RUN_DIR] [--fuel-map fuel_marshall_fbfm40.asc]
                                    [--acres LO HI] [--head-km LO HI]

RUN_DIR (default .) holds the plt_fire_????? plotfiles of one run. The script prints the
burned area and the head distance (the burned cell farthest from the ignition along the
downwind direction, 87 degrees for a wind from 267) every 5 minutes of fire, then checks:

  finite    no NaN or infinity in the level set, rate of spread, arrival time or fuel load
  ignited   cells have burned by the end
  ros       the rate of spread is non-negative and below 50 m/s
  arrival   every burned cell has an arrival time between the ignition and the plotfile
            time; every unburned cell carries the negative sentinel
  fuel      the fuel load is non-negative and never above its value at step 0
  fuel map  no cell outside the ignition disc has burned on a non-burnable code (0, 91-99)
            of the fuel map, which also catches a map read with its rows flipped
  area      the burned area never decreases between plotfiles
  acres     (with --acres) the final burned area is within [LO, HI] acres
  head      (with --head-km) the final head distance is within [LO, HI] km

The script exits 1 if any check fails.
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt: pip install numpy yt")

T_IGN, IGN, IGN_R = 1200.0, (5000.0, 11000.0), 100.0   # ignitions_spinup.csv
HEAD_DIR = 87.0                                         # downwind of a wind from 267 degrees
ROS_CAP = 50.0
ACRE = 4046.86
TOL = 1.0e-9


def load(pf):
    ds = yt.load(pf)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    names = {f for _, f in ds.field_list}
    get = lambda f: np.asarray(g[("boxlib", f)])[:, :, 0] if f in names else None
    dx = float((ds.domain_right_edge.d[0] - ds.domain_left_edge.d[0]) / ds.domain_dimensions[0])
    return float(ds.current_time), dx, get


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=".")
    ap.add_argument("--fuel-map", default="fuel_marshall_fbfm40.asc")
    ap.add_argument("--acres", nargs=2, type=float)
    ap.add_argument("--head-km", nargs=2, type=float)
    args = ap.parse_args()

    pfs = sorted(p for p in glob.glob(os.path.join(args.run, "plt_fire_?????")) if os.path.isdir(p))
    if not pfs:
        sys.exit(f"no plt_fire_????? plotfiles in {args.run}")
    times = np.array([float(yt.load(p).current_time) for p in pfs])
    ex, ey = math.sin(math.radians(HEAD_DIR)), math.cos(math.radians(HEAD_DIR))

    print(f"{args.run}: {len(pfs)} fire plotfiles, last at t = {times[-1]:.0f} s")
    print("  fire min   burned acres   head [km]")
    areas = []
    for p, t in zip(pfs, times):
        _, dx, get = load(p)
        burned = get("fire_arrival_time") >= 0.0
        areas.append(burned.sum() * dx * dx)
        m = (t - T_IGN) / 60.0
        if burned.any() and abs(m - 5.0 * round(m / 5.0)) < 0.26 and m > 0.0:
            ii, jj = np.nonzero(burned)
            head = np.max(((ii + 0.5) * dx - IGN[0]) * ex + ((jj + 0.5) * dx - IGN[1]) * ey) / 1000.0
            print(f"  {m:8.1f}   {areas[-1] / ACRE:12.0f}   {head:9.2f}")

    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"  {name:9s} {'PASS' if ok else 'FAIL'}  {detail}")

    t, dx, get = load(pfs[-1])
    phi, ros, at, fuel = get("fire_phi"), get("fire_ros"), get("fire_arrival_time"), get("fire_fuel_load")
    bad = [n for n, f in (("phi", phi), ("ros", ros), ("arrival", at), ("fuel", fuel)) if f is not None and not np.all(np.isfinite(f))]
    check("finite", not bad, "all fields finite" if not bad else f"non-finite in {bad}")

    burned = at >= 0.0
    check("ignited", bool(burned.any()), f"{int(burned.sum())} burned cells")

    if ros is not None:
        check("ros", float(ros.min()) >= -TOL and float(ros.max()) < ROS_CAP, f"range [{ros.min():.4g}, {ros.max():.4g}] m/s")

    ab = at[burned]
    ok_b = ab.size == 0 or (ab.min() >= T_IGN - 1.0 and ab.max() <= t + 1.0e-6)
    ok_u = bool(np.all(at[~burned] < 0.0))
    check("arrival", ok_b and ok_u, (f"burned in [{ab.min():.0f}, {ab.max():.0f}] s" if ab.size else "none burned")
          + ("" if ok_u else "; unburned cells with an arrival time"))

    if fuel is not None:
        _, _, get0 = load(pfs[0])
        f0 = get0("fire_fuel_load")
        grew = int((fuel > f0 + 1.0e-9).sum()) if f0 is not None else 0
        check("fuel", float(fuel.min()) >= -TOL and grew == 0, f"min {fuel.min():.4g} kg/m2, {grew} cells above their start")

    fmap = args.fuel_map if os.path.isabs(args.fuel_map) else os.path.join(args.run, args.fuel_map)
    if os.path.exists(fmap):
        codes = np.loadtxt(fmap, skiprows=6).astype(int)[::-1, :].T     # rows north first -> (i, j), j north
        if codes.shape == burned.shape:
            nx, ny = burned.shape
            X, Y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dx, indexing="ij")
            outside = np.hypot(X - IGN[0], Y - IGN[1]) > IGN_R + dx
            nb = (codes == 0) | ((codes >= 91) & (codes <= 99))
            n_bad = int((burned & nb & outside).sum())
            check("fuel map", n_bad == 0, f"{n_bad} burned cells on non-burnable codes outside the ignition disc "
                  f"({int((burned & ~nb).sum())} on burnable codes)")
        else:
            check("fuel map", False, f"{fmap} is {codes.shape}, the fire grid {burned.shape}")
    else:
        print(f"  fuel map  skipped: {fmap} not found")

    dips = sum(1 for a, b in zip(areas, areas[1:]) if b < a - 1.0e-6)
    check("area", dips == 0, f"{areas[0] / ACRE:.0f} -> {areas[-1] / ACRE:.0f} acres, {dips} decreases")

    if burned.any():
        ii, jj = np.nonzero(burned)
        head = np.max(((ii + 0.5) * dx - IGN[0]) * ex + ((jj + 0.5) * dx - IGN[1]) * ey) / 1000.0
    else:
        head = 0.0
    if args.acres:
        lo, hi = args.acres
        check("acres", lo <= areas[-1] / ACRE <= hi, f"{areas[-1] / ACRE:.0f} acres, expected [{lo:.0f}, {hi:.0f}]")
    if args.head_km:
        lo, hi = args.head_km
        check("head", lo <= head <= hi, f"{head:.2f} km, expected [{lo:.2f}, {hi:.2f}]")

    n_fail = results.count(False)
    print(f"  {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
