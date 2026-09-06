#!/usr/bin/env python3
"""Checks on the fire plotfiles of the stick-moisture regtest.

    python3 check_stick.py same  plt_a plt_b          # phi and the three dead classes identical
    python3 check_stick.py order plt m1 m10 m100      # classes within bounds, moved from the deck values in lag order
"""
import sys
import numpy as np
import yt

def load(pf):
    ds = yt.load(pf); names = [f for _, f in ds.field_list]
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    def pick(sub): return g[("boxlib", [n for n in names if sub in n][0])].value[:, :, 0]
    return pick("phi"), pick("fuel_mc") if any("fuel_mc" in n for n in names) else None, g, names

def field(g, names, sub):
    return g[("boxlib", [n for n in names if sub in n][0])].value[:, :, 0]

def main():
    mode = sys.argv[1]
    if mode == "same":
        a, b = sys.argv[2], sys.argv[3]
        pa, _, ga, na = load(a); pb, _, gb, nb = load(b)
        worst = np.max(np.abs(pa - pb))
        for c in ("M_1hr", "M_10hr", "M_100hr", "1hr", "10hr", "100hr"):
            ca = [n for n in na if c in n]
            if ca:
                worst = max(worst, np.max(np.abs(field(ga, na, ca[0]) - field(gb, nb, ca[0]))))
        ok = worst == 0.0
        print(f"  {a} and {b}: phi and the dead classes identical: {'PASS' if ok else 'FAIL'} (max abs diff {worst:.1e})")
        sys.exit(0 if ok else 1)
    if mode == "order":
        pf, m1, m10, m100 = sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
        _, _, g, names = load(pf)
        cls = {}
        for key, init in (("1hr", m1), ("10hr", m10), ("100hr", m100)):
            cand = [n for n in names if key in n and "100hr" not in n] if key != "100hr" else [n for n in names if "100hr" in n]
            if key == "10hr": cand = [n for n in names if "10hr" in n and "100hr" not in n]
            if key == "1hr": cand = [n for n in names if "1hr" in n and "10hr" not in n and "100hr" not in n]
            v = field(g, names, cand[0]); cls[key] = (v.mean(), init)
        d1, d10, d100 = [abs(cls[k][0] - cls[k][1]) for k in ("1hr", "10hr", "100hr")]
        bounds = all(0.01 <= cls[k][0] <= 0.40 for k in cls)
        ok = bounds and d1 > d10 > d100 > 0.0
        print(f"  classes within bounds and moved in lag order (1h {d1:.2e}, 10h {d10:.2e}, 100h {d100:.2e} from the deck values): "
              f"{'PASS' if ok else 'FAIL'} (means 1h {cls['1hr'][0]:.5f}, 10h {cls['10hr'][0]:.5f}, 100h {cls['100hr'][0]:.5f})")
        sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
