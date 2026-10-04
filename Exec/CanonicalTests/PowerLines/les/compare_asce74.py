#!/usr/bin/env python3
"""Compare the conductor spans of an ERF LES run with ASCE Manual of Practice 74's quasi-static wire loads.

For every span the run logged (<line>_span<k>.dat under the run's diagnostics directory), over the
samples from t_start (erf.conductors.stats_start by default):

  LES   the wind at mid-span normal to the span (horizontal), its mean and its peak 3-second average,
        the gust V3; the span's wind load per metre (its drag normal to the span over the chord), its
        mean and its peak; the effective gust response factor, the peak load over (rho/2) Cf d V3^2;
        the largest swing angle either way and the peak tension.
  ASCE  74 with the same gust: the wire load Q kz V^2 Gw Cf d with kz V^2 = V3^2, the 3-second gust at
        the span's height, so the predicted peak is (rho/2) Cf d V3^2 Gw(z, L); the swing atan(load / W)
        and the end tension of the elastic catenary under the resultant load.

The span's height z, chord L, unstretched length and weight W come from <diagnostics_dir>/asce74.csv, which ERF writes
when erf.conductors.asce74_wind is given; this script checks its own kz and Gw against that file.
Writes asce74_comparison.csv and asce74_comparison.png next to the run.

Usage: compare_asce74.py RUN_DIR [--exposure C] [--t_start T] [--diameter 0.0281] [--cf 1.0]
"""
import argparse
import glob
import math
import os
import re
import sys

import numpy as np

FT = 0.3048
EXPOSURE = {"B": (7.0, 1200 * FT, 0.010, 170 * FT), "C": (9.5, 900 * FT, 0.005, 220 * FT)}
KV = 1.43


def kz(e, z):
    a, zg, _, _ = EXPOSURE[e]
    return 2.01 * (z / zg) ** (2.0 / a)


def gw(e, z, span):
    a, _, kappa, ls = EXPOSURE[e]
    E = 4.9 * math.sqrt(kappa) * (33 * FT / z) ** (1.0 / a)
    return (1.0 + 2.7 * E * math.sqrt(1.0 / (1.0 + 0.8 * span / ls))) / KV ** 2


def elastic_catenary(chord, length, w, EA):
    """End tension and sag of a level elastic catenary (ERF_ConductorInputs.H, elastic_catenary)."""
    def unstretched(H):
        a = H / w
        x = chord / (2 * a)
        if x > 300:
            return float("inf")
        return 2 * a * math.sinh(x) - H / EA * (0.5 * chord + 0.5 * a * math.sinh(chord / a))
    lo, hi = math.log(1e-6 * w * chord), math.log(1e3 * EA)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if unstretched(math.exp(mid)) > length:
            lo = mid
        else:
            hi = mid
    H = math.exp(0.5 * (lo + hi))
    a = H / w
    return H * math.cosh(chord / (2 * a)), a * (math.cosh(chord / (2 * a)) - 1)


def inputs_value(text, key, default=None):
    m = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(\S+)", text, re.M)
    return m.group(1) if m else default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--inputs", default=None, help="the run's inputs file (default: the first inputs* in RUN_DIR)")
    ap.add_argument("--exposure", default="C")
    ap.add_argument("--t_start", type=float, default=None)
    args = ap.parse_args()
    R = os.path.abspath(args.run)
    inp = args.inputs or sorted(glob.glob(os.path.join(R, "inputs*")))[0]
    text = open(inp).read()
    net = text
    for f in re.findall(r"^\s*FILE\s*=\s*(\S+)", text, re.M):
        net += open(os.path.join(R, f)).read()
    diag = os.path.join(R, inputs_value(text, "erf.conductors.diagnostics_dir", "conductors"))
    rho = float(inputs_value(text, "erf.conductors.air_density", "1.225"))
    t_start = args.t_start if args.t_start is not None else float(inputs_value(text, "erf.conductors.stats_start", "0"))
    v_design = float(inputs_value(text, "erf.conductors.asce74_wind", "0"))
    e = args.exposure.upper()
    if not v_design > 0:
        sys.exit("the run needs erf.conductors.asce74_wind for its asce74.csv (heights, chords, weights)")

    # ERF's design table, checked against this script's own formulas
    rows = [l.strip().split(",") for l in open(os.path.join(diag, "asce74.csv")).read().strip().split("\n")]
    head, rows = rows[0], rows[1:]
    col = {h: i for i, h in enumerate(head)}
    spans = {}
    for r in rows:
        z, chord = float(r[col["height"]]), float(r[col["chord"]])
        assert abs(kz(e, z) - float(r[col["kz"]])) < 1e-6 * kz(e, z), "kz differs from ERF's: wrong --exposure?"
        assert abs(gw(e, z, chord) - float(r[col["gust_response"]])) < 1e-6, "Gw differs from ERF's"
        spans[(r[0], int(r[1]))] = dict(z=z, chord=chord, W=float(r[col["weight"]]), length=float(r[col["length"]]))

    # each line's attachment points (ground.dat: line point x y ground z), for the spans' directions
    pts = {}
    for l in open(os.path.join(diag, "ground.dat")).read().strip().split("\n")[1:]:
        p = l.split()
        if p[1] == "transformer":
            continue
        pts.setdefault(p[0], []).append((float(p[2]), float(p[3])))

    out = []
    series = {}
    for (line, k), s in sorted(spans.items()):
        # the span's log: <output_root>_span<k>.dat, output_root defaulting to <diagnostics_dir>/<line>
        root = inputs_value(net, f"erf.conductors.{line}.output_root")
        f = (os.path.join(R, root) if root else os.path.join(diag, line)) + f"_span{k}.dat"
        h = open(f).readline().split()
        d = np.loadtxt(f, skiprows=1)
        d = d[d[:, 0] >= t_start]
        c = {n: i for i, n in enumerate(h)}
        (xa, ya), (xb, yb) = pts[line][k - 1], pts[line][k]
        L = math.hypot(xb - xa, yb - ya)
        n = np.array([-(yb - ya) / L, (xb - xa) / L])
        vn = d[:, c["mid_u"]] * n[0] + d[:, c["mid_v"]] * n[1]
        sign = 1.0 if vn.mean() >= 0 else -1.0
        vn *= sign
        q = sign * (d[:, c["drag_x"]] * n[0] + d[:, c["drag_y"]] * n[1]) / s["chord"]
        dt = np.median(np.diff(d[:, 0]))
        m = max(1, int(round(3.0 / dt)))
        v3 = np.convolve(vn, np.ones(m) / m, mode="valid").max()
        diam = float(inputs_value(net, f"erf.conductors.{line}.diameter"))
        cf = float(inputs_value(net, f"erf.conductors.{line}.drag_coefficient", "1.0"))
        ea = float(inputs_value(net, f"erf.conductors.{line}.axial_stiffness"))
        q0 = 0.5 * rho * cf * diam * v3 ** 2
        g = gw(e, s["z"], s["chord"])
        q_asce = q0 * g
        swing_asce = math.degrees(math.atan2(q_asce, s["W"]))
        tension_asce, _ = elastic_catenary(s["chord"], s["length"], math.hypot(q_asce, s["W"]), ea)
        out.append(dict(line=line, span=k, z=s["z"], chord=s["chord"], v_mean=vn.mean(), v3=v3, gust_factor=v3 / vn.mean(),
                        q_mean=q.mean(), q_peak=q.max(), gw_les=q.max() / q0, gw_asce=g, q_asce=q_asce,
                        swing_les=np.abs(d[:, c["swing_deg"]]).max(), swing_asce=swing_asce,
                        tension_les=d[:, c["max_tension"]].max(), tension_asce=tension_asce))
        series[(line, k)] = (d[:, 0], q, q_asce, q0)

    keys = list(out[0].keys())
    with open(os.path.join(R, "asce74_comparison.csv"), "w") as table:
        table.write(",".join(keys) + "\n")
        for o in out:
            table.write(",".join(str(o[k]) if isinstance(o[k], str) else ("%d" % o[k] if isinstance(o[k], int) else "%.6g" % o[k])
                              for k in keys) + "\n")
    print("%-5s %4s %6s %6s %6s %6s %5s %7s %7s %7s %6s %6s %6s %6s %8s %8s" % ("line", "span", "z", "L", "Vmean", "V3s", "GF",
          "q_mean", "q_peak", "q_ASCE", "Gw_LES", "Gw_74", "sw_LES", "sw_74", "T_LES", "T_74"))
    for o in out:
        print("%-5s %4d %6.1f %6.0f %6.2f %6.2f %5.2f %7.3f %7.3f %7.3f %6.3f %6.3f %6.1f %6.1f %8.0f %8.0f" % (
            o["line"], o["span"], o["z"], o["chord"], o["v_mean"], o["v3"], o["gust_factor"], o["q_mean"], o["q_peak"],
            o["q_asce"], o["gw_les"], o["gw_asce"], o["swing_les"], o["swing_asce"], o["tension_les"], o["tension_asce"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    L = np.array([o["chord"] for o in out])
    sc = ax[0].scatter(L, [o["gw_les"] for o in out], c=[o["v_mean"] for o in out], cmap="viridis", zorder=3,
                       label="LES: peak span load / ((rho/2) Cf d V3s^2)")
    fig.colorbar(sc, ax=ax[0], label="mean wind across the span (m/s)")
    for o in out:
        ax[0].annotate(f'{o["line"]}.{o["span"]}', (o["chord"], o["gw_les"]), fontsize=7)
    Ls = np.linspace(20, max(L) * 1.1, 200)
    zbar = np.mean([o["z"] for o in out])
    ax[0].plot(Ls, [gw(e, zbar, x) for x in Ls], "k-", label=f"ASCE 74 Gw at z = {zbar:.0f} m, exposure {e}")
    ax[0].set_xlabel("span (m)"); ax[0].set_ylabel("gust response factor"); ax[0].legend(fontsize=8)
    ax[0].set_ylim(0, 1.1 * max(1.2, max(o["gw_les"] for o in out)))
    ax[0].set_title("how much of the point gust the whole span feels")
    ax[1].scatter([o["q_asce"] for o in out], [o["q_peak"] for o in out], c="C0", zorder=3)
    lim = max(max(o["q_asce"] for o in out), max(o["q_peak"] for o in out)) * 1.1
    ax[1].plot([0, lim], [0, lim], "k--", lw=0.8)
    for o in out:
        ax[1].annotate(f'{o["line"]}.{o["span"]}', (o["q_asce"], o["q_peak"]), fontsize=7)
    ax[1].set_xlabel("ASCE 74 peak load from the LES gust (N/m)"); ax[1].set_ylabel("LES peak span load (N/m)")
    ax[1].set_title("peak wind load per metre of span")
    # the span the wind loads most of those at least half as long as the longest
    long_spans = [k for k in series if spans[k]["chord"] >= 0.5 * max(s["chord"] for s in spans.values())]
    longest = max(long_spans, key=lambda k: series[k][1].mean())
    t, q, qa, q0 = series[longest]
    ax[2].plot(t, q, lw=0.7, label="LES span load")
    ax[2].axhline(qa, color="k", lw=1.2, label="ASCE 74 with the LES 3-s gust")
    ax[2].axhline(q0, color="0.5", lw=0.8, ls=":", label="the 3-s gust on the whole span (Gw = 1)")
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("N/m"); ax[2].legend(fontsize=8)
    ax[2].set_title(f"{longest[0]} span {longest[1]} ({spans[longest]['chord']:.0f} m)")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "asce74_comparison.png"), dpi=110)
    print("wrote", os.path.join(R, "asce74_comparison.csv"), "and asce74_comparison.png")


if __name__ == "__main__":
    main()
