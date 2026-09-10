#!/usr/bin/env python3
"""Checks on the fire plotfiles of the live-moisture regtest.

    python3 check_live.py plt_fire_legacy plt_fire_legacy_key plt_fire_fixed M_live
        [--restart plt_fire_legacy_restart plt_fire_fixed_restart]
        [--directional plt_fire_fixed_dir020 0.20 plt_fire_fixed_dir025 0.25]

The first three directories hold plt_fire_00000 and the last plt_fire_NNNNN;
the restart and directional ones need only the last. Prints a table of the
live classes, the dead 1-hour class and the front of each run, then the
checks; exits 1 if any fails.
"""
import argparse, glob, os, sys
import numpy as np
import yt
yt.set_log_level(50)

CLASSES = ("fire_fuel_mc_1hr", "fire_fuel_mc_10hr", "fire_fuel_mc_100hr", "fire_fuel_mc_lh", "fire_fuel_mc_lw")
LIVE = ("fire_fuel_mc_lh", "fire_fuel_mc_lw")

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

def plotfiles(d):
    return sorted(p for p in glob.glob(os.path.join(d, "plt_fire_?????")) if os.path.isdir(p))

def first_last(d):
    pfs = plotfiles(d)
    if len(pfs) < 2:
        sys.exit(f"{d}: needs the first and the last fire plotfile, found {pfs}")
    return load(pfs[0]), load(pfs[-1])

def last(d):
    pfs = plotfiles(d)
    if not pfs:
        sys.exit(f"{d}: no fire plotfile")
    return load(pfs[-1])

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

def row(v, r):
    burning = r["fire_phi"] < 0.0
    ros = r["fire_ros"][burning].mean() if burning.any() else 0.0
    print(f"{v:15s} {r['t']:6.1f} {r['fire_fuel_mc_lh'].min():9.5f}/{r['fire_fuel_mc_lh'].max():<9.5f} "
          f"{r['fire_fuel_mc_lw'].min():9.5f}/{r['fire_fuel_mc_lw'].max():<9.5f} "
          f"{r['fire_fuel_mc_1hr'].mean():9.5f} {burning.sum() * r['cell_area']:11.1f} {ros:14.4f}")

def max_diff(a, b, names):
    return max(np.max(np.abs(a[n] - b[n])) for n in names)

def main():
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("legacy"); ap.add_argument("legacy_key"); ap.add_argument("fixed")
    ap.add_argument("m_live", type=float)
    ap.add_argument("--restart", nargs=2)
    ap.add_argument("--directional", nargs=4)
    a = ap.parse_args()
    m_live = a.m_live

    runs = {"legacy": first_last(a.legacy), "legacy_key": first_last(a.legacy_key), "fixed": first_last(a.fixed)}
    extra = {}
    if a.restart:
        extra["legacy_restart"] = last(a.restart[0]); extra["fixed_restart"] = last(a.restart[1])
    if a.directional:
        extra["fixed_dir_a"] = last(a.directional[0]); extra["fixed_dir_b"] = last(a.directional[2])

    print(f"{'variant':15s} {'t [s]':>6s} {'live herb min/max':>19s} {'live woody min/max':>19s} "
          f"{'1-h mean':>9s} {'burned [m2]':>11s} {'mean ROS [m/s]':>14s}")
    for v, pair in runs.items():
        for r in pair:
            row(v, r)
    for v, r in extra.items():
        row(v, r)
    print()

    L0, L = runs["legacy"]; K = runs["legacy_key"][1]; F0, F = runs["fixed"]
    worst = max_diff(L, K, CLASSES + ("fire_phi", "fire_ros"))
    check("the default written out reproduces the historical deck (level set, rate of spread, five classes)",
          worst == 0.0, f"max abs diff {worst:.1e}")
    start = max(np.max(np.abs(r[n] - m_live)) for r in (L0, F0) for n in LIVE)
    check(f"both runs start with the live classes at erf.fire.moisture_live = {m_live}", start < 1e-12,
          f"max abs diff {start:.1e}")
    held = max(np.max(np.abs(F[n] - m_live)) for n in LIVE)
    check(f"fixed: the live classes are still exactly {m_live} at t = {F['t']:.0f} s", held == 0.0,
          f"max abs diff {held:.1e}")
    top = max(L[n].max() for n in LIVE)
    check("legacy: the live classes have fallen to the dead-fuel clamp, at most 0.40", top <= 0.40,
          f"max {top:.5f}")
    dead = max_diff(L, F, CLASSES[:3])
    check("the dead classes do not depend on the live setting", dead == 0.0, f"max abs diff {dead:.1e}")
    area_l = (L["fire_phi"] < 0.0).sum(); area_f = (F["fire_phi"] < 0.0).sum()
    dros = np.max(np.abs(L["fire_ros"] - F["fire_ros"]))
    check("the rate of spread differs between legacy and fixed (fuel model 2 has a live herbaceous load)",
          dros > 0.0, f"max abs ROS diff {dros:.3e} m/s, burned area legacy/fixed {area_l / max(area_f, 1):.3f}")

    if a.restart:
        LR, FR = extra["legacy_restart"], extra["fixed_restart"]
        back = max(np.max(np.abs(FR[n] - m_live)) for n in LIVE)
        check(f"fixed restart from a legacy checkpoint: the live classes are back at {m_live} at t = {FR['t']:.0f} s",
              back == 0.0, f"max abs diff {back:.1e}")
        kept = max(LR[n].max() for n in LIVE)
        check("legacy restart from the same checkpoint keeps the checkpointed live classes (at most 0.40)",
              kept <= 0.40, f"max {kept:.5f}")
        dead_r = max_diff(LR, FR, CLASSES[:3])
        check("the dead classes restart the same under both settings", dead_r == 0.0, f"max abs diff {dead_r:.1e}")

    if a.directional:
        A, B = extra["fixed_dir_a"], extra["fixed_dir_b"]
        ma, mb = float(a.directional[1]), float(a.directional[3])
        held_d = max(max(np.max(np.abs(A[n] - ma)) for n in LIVE), max(np.max(np.abs(B[n] - mb)) for n in LIVE))
        check(f"fixed on the directional path holds the live classes at {ma} and {mb}", held_d == 0.0,
              f"max abs diff {held_d:.1e}")
        dphi = np.max(np.abs(A["fire_phi"] - B["fire_phi"]))
        area_a = (A["fire_phi"] < 0.0).sum(); area_b = (B["fire_phi"] < 0.0).sum()
        check(f"the directional front at {ma} differs from the one at {mb}: the domain-average BEHAVE state is not clamped to 0.30",
              dphi > 0.0, f"max abs phi diff {dphi:.3e} m, burned area ratio {area_a / max(area_b, 1):.4f}")

    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)

if __name__ == "__main__":
    main()
