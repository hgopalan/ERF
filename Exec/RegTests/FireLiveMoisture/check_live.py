#!/usr/bin/env python3
"""Checks on the fire plotfiles of the live-moisture regtest.

    python3 check_live.py plt_fire_legacy plt_fire_legacy_key plt_fire_fixed M_live

Each argument directory holds plt_fire_00000 and the last plt_fire_NNNNN.
Prints a table of the live classes, the dead 1-hour class and the front at
the start and the end of each run, then the checks; exits 1 if any fails.
"""
import glob, os, sys
import numpy as np
import yt
yt.set_log_level(50)

CLASSES = ("fire_fuel_mc_1hr", "fire_fuel_mc_10hr", "fire_fuel_mc_100hr", "fire_fuel_mc_lh", "fire_fuel_mc_lw")

def load(pf):
    ds = yt.load(pf)
    names = [f for _, f in ds.field_list]
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    def get(name):
        if name not in names:
            sys.exit(f"{pf} has no field {name} (has {sorted(names)})")
        return np.asarray(g[("boxlib", name)])[:, :, 0]
    out = {n: get(n) for n in CLASSES + ("fire_phi", "fire_ros")}
    dx = ds.domain_width / ds.domain_dimensions
    out["t"] = float(ds.current_time)
    out["cell_area"] = float(dx[0] * dx[1])
    return out

def first_last(d):
    pfs = sorted(p for p in glob.glob(os.path.join(d, "plt_fire_?????")) if os.path.isdir(p))
    if len(pfs) < 2:
        sys.exit(f"{d}: needs the first and the last fire plotfile, found {pfs}")
    return load(pfs[0]), load(pfs[-1])

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    legacy_dir, key_dir, fixed_dir, m_live = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    runs = {"legacy": first_last(legacy_dir), "legacy_key": first_last(key_dir), "fixed": first_last(fixed_dir)}

    print(f"{'variant':11s} {'t [s]':>6s} {'live herb min/max':>19s} {'live woody min/max':>19s} "
          f"{'1-h mean':>9s} {'burned [m2]':>11s} {'mean ROS [m/s]':>14s}")
    for v, pair in runs.items():
        for r in pair:
            burning = r["fire_phi"] < 0.0
            ros = r["fire_ros"][burning].mean() if burning.any() else 0.0
            print(f"{v:11s} {r['t']:6.1f} {r['fire_fuel_mc_lh'].min():9.5f}/{r['fire_fuel_mc_lh'].max():<9.5f} "
                  f"{r['fire_fuel_mc_lw'].min():9.5f}/{r['fire_fuel_mc_lw'].max():<9.5f} "
                  f"{r['fire_fuel_mc_1hr'].mean():9.5f} {burning.sum() * r['cell_area']:11.1f} {ros:14.4f}")
    print()

    L0, L = runs["legacy"]; K = runs["legacy_key"][1]; F0, F = runs["fixed"]
    worst = max(np.max(np.abs(L[n] - K[n])) for n in CLASSES + ("fire_phi", "fire_ros"))
    check("the default written out reproduces the historical deck (level set, rate of spread, five classes)",
          worst == 0.0, f"max abs diff {worst:.1e}")
    start = max(np.max(np.abs(r[n] - m_live)) for r in (L0, F0) for n in ("fire_fuel_mc_lh", "fire_fuel_mc_lw"))
    check(f"both runs start with the live classes at erf.fire.moisture_live = {m_live}", start < 1e-12,
          f"max abs diff {start:.1e}")
    held = max(np.max(np.abs(F[n] - m_live)) for n in ("fire_fuel_mc_lh", "fire_fuel_mc_lw"))
    check(f"fixed: the live classes are still exactly {m_live} at t = {F['t']:.0f} s", held == 0.0,
          f"max abs diff {held:.1e}")
    top = max(L[n].max() for n in ("fire_fuel_mc_lh", "fire_fuel_mc_lw"))
    check("legacy: the live classes have fallen to the dead-fuel clamp, at most 0.40", top <= 0.40,
          f"max {top:.5f}")
    dead = max(np.max(np.abs(L[n] - F[n])) for n in CLASSES[:3])
    check("the dead classes do not depend on the live setting", dead == 0.0, f"max abs diff {dead:.1e}")
    area_l = (L["fire_phi"] < 0.0).sum(); area_f = (F["fire_phi"] < 0.0).sum()
    dros = np.max(np.abs(L["fire_ros"] - F["fire_ros"]))
    check("the rate of spread differs between legacy and fixed (fuel model 2 has a live herbaceous load)",
          dros > 0.0, f"max abs ROS diff {dros:.3e} m/s, burned area legacy/fixed {area_l / max(area_f, 1):.3f}")

    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)

if __name__ == "__main__":
    main()
