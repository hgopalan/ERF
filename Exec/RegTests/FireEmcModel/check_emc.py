#!/usr/bin/env python3
"""Checks on the FireEmcModel regtest, from the run logs and the fire plotfiles.

    python3 check_emc.py table            # moisture, surface RH/T, R0 and burning cells per variant
    python3 check_emc.py same va vb       # fire plotfiles of two variants bit-identical at 60 s
    python3 check_emc.py order            # each curve choice moves the dead classes the expected way

The logs are run_<variant>.log with erf.fire.fire_debug = true; the plotfiles
plt_fire_<variant>/plt_fire_00480. The ordering checks hold whichever
hysteresis branch the kernel takes, since both relax toward the same side.
"""
import glob
import os
import re
import sys

VARIANTS = ["legacy", "legacy_key", "van_wagner", "humid_legacy", "humid_van_wagner"]
M0 = 0.08          # deck value of every dead class
PLT = "plt_fire_00480"


def last(pattern, log):
    val = None
    with open(log, errors="replace") as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                val = m.groups()
    return val


def summary(v):
    log = f"run_{v}.log"
    mc = last(r"avg moisture: M_1hr=(\S+) M_10hr=(\S+) M_100hr=(\S+) R0=(\S+)", log)
    rh = last(r"Surface RH range:\s+min=(\S+)\s+max=(\S+)", log)
    T = last(r"Surface temp range: min=(\S+) K\s+max=(\S+) K", log)
    cells = last(r"active fire cells\D*?(\d+)\s*$", log)
    if mc is None or rh is None or T is None:
        sys.exit(f"{log}: no [FIRE DEBUG] moisture/RH lines (did the run finish with fire_debug on?)")
    return dict(m1=float(mc[0]), m10=float(mc[1]), m100=float(mc[2]), R0=float(mc[3]),
                rh=float(rh[1]), T=float(T[1]), cells=int(cells[0]) if cells else -1)


def table():
    print(f"{'variant':<18} {'RH max':>7} {'T max K':>8} {'M_1hr':>9} {'M_10hr':>9} {'M_100hr':>9} {'R0 m/s':>10} {'cells':>7}")
    for v in VARIANTS:
        if not os.path.isfile(f"run_{v}.log"):
            continue
        s = summary(v)
        print(f"{v:<18} {s['rh']:7.3f} {s['T']:8.2f} {s['m1']:9.6f} {s['m10']:9.6f} {s['m100']:9.6f} {s['R0']:10.5f} {s['cells']:7d}")


def same(va, vb):
    fa = sorted(glob.glob(f"plt_fire_{va}/{PLT}/Level_0/Cell_D_*"))
    fb = sorted(glob.glob(f"plt_fire_{vb}/{PLT}/Level_0/Cell_D_*"))
    ok = len(fa) > 0 and [os.path.basename(p) for p in fa] == [os.path.basename(p) for p in fb]
    if ok:
        for a, b in zip(fa, fb):
            with open(a, "rb") as x, open(b, "rb") as y:
                if x.read() != y.read():
                    ok = False
                    break
    print(f"  {va} and {vb}: fire plotfile data bit-identical at 60 s: {'PASS' if ok else 'FAIL'}")
    return ok


def check(name, ok):
    print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return ok


def order():
    s = {v: summary(v) for v in VARIANTS if os.path.isfile(f"run_{v}.log")}
    ok = True
    ok &= check("dry air reaches the fuel as zero RH", s["legacy"]["rh"] == 0.0 and s["van_wagner"]["rh"] == 0.0)
    ok &= check(f"humid air reaches the fuel at 30-50 % RH (max {s['humid_legacy']['rh']:.3f})",
                0.30 <= s["humid_legacy"]["rh"] <= 0.50 and s["humid_legacy"]["rh"] == s["humid_van_wagner"]["rh"])
    for v in VARIANTS:
        if v not in s:
            continue
        d = [abs(s[v][k] - M0) for k in ("m1", "m10", "m100")]
        ok &= check(f"{v}: classes moved from {M0} in lag order (1 h {d[0]:.1e}, 10 h {d[1]:.1e}, 100 h {d[2]:.1e})",
                    d[0] > d[1] > d[2] > 0.0 and all(0.01 <= s[v][k] <= 0.40 for k in ("m1", "m10", "m100")))
    # Dry air: legacy E_w/E_d = 0.035/0.060 at its 1 % RH clamp, van_wagner 0.000 at 0 % RH.
    ok &= check(f"dry air: van_wagner 1-h ({s['van_wagner']['m1']:.6f}) drier than legacy ({s['legacy']['m1']:.6f}), both below {M0}",
                s["van_wagner"]["m1"] < s["legacy"]["m1"] < M0)
    # About 40 % RH at 27 C: legacy E_w/E_d = 0.166/0.189, van_wagner 0.089/0.105, both above 0.08.
    ok &= check(f"humid air: legacy 1-h ({s['humid_legacy']['m1']:.6f}) wetter than van_wagner ({s['humid_van_wagner']['m1']:.6f}), both above {M0}",
                s["humid_legacy"]["m1"] > s["humid_van_wagner"]["m1"] > M0)
    # Wetter fuel spreads slower: the no-wind Rothermel rate follows the moisture.
    ok &= check("R0 ordered inversely to the 1-h moisture in both airs",
                s["van_wagner"]["R0"] > s["legacy"]["R0"] and s["humid_van_wagner"]["R0"] > s["humid_legacy"]["R0"])
    return ok


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "table"
    if mode == "table":
        table()
    elif mode == "same":
        sys.exit(0 if same(sys.argv[2], sys.argv[3]) else 1)
    elif mode == "order":
        sys.exit(0 if order() else 1)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
