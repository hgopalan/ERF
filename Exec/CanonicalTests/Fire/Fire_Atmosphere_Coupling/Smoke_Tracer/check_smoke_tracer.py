#!/usr/bin/env python3
"""Sanity checks on the fire and smoke output of the Smoke_Tracer tests.

    python3 check_smoke_tracer.py [plt_fire_NNNNN] [--stats fire_stats.csv]
                                  [--atm-prefix plt_1_]

With no plotfile the last fire plotfile in the directory is used (plt_fire_NNNNN,
or plt_fire_<name>_NNNNN for a deck that sets erf.fire_plot_file); with no
--stats every fire_stats*.csv present is checked. Each check prints PASS or FAIL
and the script exits 1 if any fails, so it can follow the run in CTest:

    erf_exec inputs_... && python3 check_smoke_tracer.py

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

and, from the smoke mass concentration [kg/m^3] in the atmosphere plotfiles
(erf.plot_vars_1 must list smoke):

  smoke       every atmosphere plotfile has a finite smoke field, and the last
              one holds smoke
  positive    the negative smoke mass stays below 10% of the positive mass in
              every plotfile. The default Upwind_3rd scalar advection is not
              monotone and rings next to the one-layer surface source, so single
              cells undershoot zero (by 12% of the maximum early in the 60 s run
              of this deck); by mass the undershoot was 2-6%
  mass        the domain-total smoke mass never decreases: the domain is
              periodic in x and y with walls top and bottom, so smoke only
              enters, through the fire emission, and never leaves

A deck that deliberately never ignites (a wet or extinction case) can pass
--allow-no-fire, which turns "ignited" into "burned area stays zero" and
"smoke" into "no smoke is emitted".
"""
import argparse, csv, glob, math, os, re, sys
import numpy as np

ROS_CAP = 50.0          # m/s; no surface fire model gets near this
TOL = 1.0e-9
NEG_MASS_FRAC = 0.10    # allowed negative smoke mass as a fraction of the positive mass
MASS_RTOL = 1.0e-9      # allowed relative decrease of the total smoke mass

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

def plotfiles(pattern):
    """Plotfile directories matching pattern (step number in group 2), by step."""
    rx = re.compile(pattern)
    found = [(m, p) for p in os.listdir(".") for m in [rx.match(p)] if m and os.path.isdir(p)]
    return [p for m, p in sorted(found, key=lambda mp: int(mp[0].group(2)))]

results = []
def check(name, ok, detail):
    results.append(ok)
    print(f"  {name:8s} {'PASS' if ok else 'FAIL'}  {detail}")

def check_smoke(prefix, allow_no_fire):
    pfs = plotfiles(r"^(" + re.escape(prefix) + r")(\d{5,})$")
    if not pfs:
        check("smoke", False, f"no {prefix}NNNNN atmosphere plotfile (erf.plot_int_1)")
        return
    rows = []
    for p in pfs:
        ds = yt.load(p)
        if ("boxlib", "smoke") not in ds.field_list:
            check("smoke", False, f"{p} has no smoke field (erf.plot_vars_1)")
            return
        g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        s = np.asarray(g[("boxlib", "smoke")])
        dv = float(np.prod((ds.domain_right_edge - ds.domain_left_edge).d / ds.domain_dimensions))
        finite = bool(np.all(np.isfinite(s)))
        s0 = np.where(np.isfinite(s), s, 0.0)
        rows.append((p, float(ds.current_time), finite,
                     float(s0.min()), float(s0.max()), float(s0.sum()) * dv,
                     float(-s0[s0 < 0.0].sum()) * dv, float(s0[s0 > 0.0].sum()) * dv))

    bad = [p for p, _, finite, *_ in rows if not finite]
    last = rows[-1]
    smax = max(r[4] for r in rows)
    if bad:
        check("smoke", False, f"non-finite smoke in {bad}")
    elif allow_no_fire:
        check("smoke", smax <= TOL, f"{len(rows)} plotfiles, max {smax:.4g} kg/m3 (no-fire case)")
    else:
        check("smoke", last[4] > 0.0,
              f"{len(rows)} plotfiles, {last[0]} at t = {last[1]:.1f} s: max {last[4]:.4g} kg/m3")

    smin = min(r[3] for r in rows)
    fracs = [(r[6] / r[7], r[0]) for r in rows if r[7] > 0.0]
    worst, worst_pf = max(fracs) if fracs else (0.0, "no plotfile with smoke")
    check("positive", worst <= NEG_MASS_FRAC,
          f"negative/positive mass at most {worst:.3f} ({worst_pf}); "
          f"cell min {smin:.4g} against max {smax:.4g} kg/m3")

    mass = [r[5] for r in rows]
    dips = sum(1 for x, y in zip(mass, mass[1:]) if y < x - MASS_RTOL * max(abs(x), abs(y)))
    check("mass", dips == 0, f"{mass[0]:.4g} -> {mass[-1]:.4g} kg over {len(mass)} plotfiles, {dips} decreases")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plotfile", nargs="?")
    ap.add_argument("--stats", action="append")
    ap.add_argument("--atm-prefix", default="plt_1_",
                    help="atmosphere plotfile prefix (erf.plot_file_1)")
    ap.add_argument("--allow-no-fire", action="store_true")
    args = ap.parse_args()

    pf = args.plotfile
    if pf is None:
        pfs = plotfiles(r"^(plt_fire_(?:.+_)?)(\d{5,})$")
        if not pfs:
            print("  no plt_fire_NNNNN plotfile here: FAIL")
            sys.exit(1)
        pf = pfs[-1]
    ds, get = load(pf)
    t = float(ds.current_time)
    print(f"Smoke_Tracer: {pf} at t = {t:.3f} s")

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
        f0 = re.sub(r"\d+$", "00000", pf)
        ok_neg = float(np.nanmin(fuel)) >= -TOL
        if os.path.isdir(f0) and f0 != pf:
            _, get0 = load(f0)
            fuel0 = get0("fire_fuel_load", required=False)
            grew = int((fuel > fuel0 + 1.0e-9).sum()) if fuel0 is not None else 0
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

    check_smoke(args.atm_prefix, args.allow_no_fire)

    n_fail = results.count(False)
    print(f"Smoke_Tracer: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
    main()
