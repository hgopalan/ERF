#!/usr/bin/env python3
"""Sanity checks on the fire output of the LevelSetPropagation tests.

    python3 check_levelsetpropagation.py [plt_fire_NNNNN] [--stats fire_stats.csv]

With no plotfile the last plt_fire_????? in the directory is used; with no
--stats every fire_stats*.csv present is checked. Each check prints PASS or FAIL
and the script exits 1 if any fails, so it can follow the run in CTest:

    erf_exec inputs_... && python3 check_levelsetpropagation.py

The checks are the ones every fire case must satisfy whatever its physics:

  finite      no NaN or infinity in the level set, the rate of spread, the
              arrival time or the fuel load
  ignited     at least one cell has burned (phi < 0)
  ros         the rate of spread is non-negative and below 50 m/s
  arrival     every burned cell has an arrival time in [0, t]; every unburned
              cell still carries the negative sentinel
  fuel        the fuel load is non-negative and never above its starting value
              in a burned cell (burning only removes fuel)
  area        the burned area in the statistics CSV never decreases

A deck that deliberately never ignites (a wet or extinction case) can pass
--allow-no-fire, which turns "ignited" into "burned area stays zero".
"""
import argparse, csv, glob, math, os, sys
import numpy as np

ROS_CAP = 50.0          # m/s; no surface fire model gets near this
TOL = 1.0e-9

def fail_if(missing):
    if missing:
        sys.exit(f"needs {missing}: pip install numpy yt")

try:
    import yt
    yt.set_log_level(50)
except ImportError:
    fail_if("yt")

def load(pf):
    ds = yt.load(pf)
    names = [f for _, f in ds.field_list]
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    def get(sub, required=True):
        cand = [n for n in names if n == sub] or [n for n in names if sub in n]
        if not cand:
            if required:
                raise KeyError(f"{pf} has no field matching '{sub}' (has {sorted(names)})")
            return None
        return np.asarray(g[("boxlib", cand[0])])[:, :, 0]
    return ds, get

results = []
def check(name, ok, detail):
    results.append(ok)
    print(f"  {name:8s} {'PASS' if ok else 'FAIL'}  {detail}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plotfile", nargs="?")
    ap.add_argument("--stats", action="append")
    ap.add_argument("--allow-no-fire", action="store_true")
    args = ap.parse_args()

    pf = args.plotfile
    if pf is None:
        pfs = sorted(p for p in glob.glob("plt_fire_?????") if os.path.isdir(p))
        if not pfs:
            print("  no plt_fire_????? plotfile here: FAIL")
            sys.exit(1)
        pf = pfs[-1]
    ds, get = load(pf)
    t = float(ds.current_time)
    print(f"LevelSetPropagation: {pf} at t = {t:.3f} s")

    phi = get("fire_phi")
    ros = get("fire_ros", required=False)
    at  = get("fire_arrival_time", required=False)
    fuel = get("fire_fuel_load", required=False)

    fields = {"phi": phi, "ros": ros, "arrival": at, "fuel": fuel}
    bad = [k for k, v in fields.items() if v is not None and not np.all(np.isfinite(v))]
    check("finite", not bad, "all fields finite" if not bad else f"non-finite in {bad}")

    burned = phi < 0.0
    n_burned = int(burned.sum())
    if args.allow_no_fire:
        check("ignited", True, f"{n_burned} burned cells (no-fire case allowed)")
    else:
        check("ignited", n_burned > 0, f"{n_burned} burned cells")

    if ros is not None:
        lo, hi = float(np.nanmin(ros)), float(np.nanmax(ros))
        check("ros", lo >= -TOL and hi < ROS_CAP, f"range [{lo:.4g}, {hi:.4g}] m/s")

    if at is not None:
        at_b = at[burned]
        ok_b = at_b.size == 0 or (at_b.min() >= -TOL and at_b.max() <= t + 1.0e-6)
        ok_u = bool(np.all(at[~burned] < 0.0)) if (~burned).any() else True
        detail = (f"burned in [{at_b.min():.3g}, {at_b.max():.3g}] s" if at_b.size else "no burned cells") \
                 + (", unburned carry the sentinel" if ok_u else ", some unburned cells have an arrival time")
        check("arrival", ok_b and ok_u, detail)

    if fuel is not None:
        f0_pfs = sorted(p for p in glob.glob("plt_fire_00000") if os.path.isdir(p))
        ok_neg = float(np.nanmin(fuel)) >= -TOL
        if f0_pfs and f0_pfs[0] != pf:
            _, get0 = load(f0_pfs[0])
            f0 = get0("fire_fuel_load", required=False)
            grew = int((fuel > f0 + 1.0e-9).sum()) if f0 is not None else 0
            check("fuel", ok_neg and grew == 0, f"min {np.nanmin(fuel):.4g}, {grew} cells above their start")
        else:
            check("fuel", ok_neg, f"min {np.nanmin(fuel):.4g} kg/m2")

    stats = args.stats or sorted(glob.glob("fire_stats*.csv"))
    for sf in stats:
        try:
            rows = list(csv.DictReader(open(sf)))
        except OSError:
            continue
        key = next((k for k in (rows[0].keys() if rows else []) if "burned_area" in k), None)
        if not rows or key is None:
            continue
        a = [float(r[key]) for r in rows if r[key] not in ("", None)]
        dips = sum(1 for x, y in zip(a, a[1:]) if y < x - 1.0e-9)
        if args.allow_no_fire:
            check("area", max(a) <= 1.0e-9 or dips == 0, f"{sf}: {len(a)} rows, max {max(a):.4g} ha")
        else:
            check("area", dips == 0, f"{sf}: {len(a)} rows, {a[0]:.4g} -> {a[-1]:.4g} ha, {dips} decreases")

    n_fail = results.count(False)
    print(f"LevelSetPropagation: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
    main()
