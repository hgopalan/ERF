#!/usr/bin/env python3
"""Compare the gusts on the conductor spans of a k-equation RANS run with an LES of the same lines over the same hills.

The LES run (LES_DIR) logs every span (<line>_span<k>.dat) with erf.conductors.diagnostics_int = 1; the RANS directory
(RANS_DIR) holds one run per gust type of the same network in the RANS flow, each with its own inputs file:
inputs_factor (gust_type = factor: the gust-free wind and gusts.csv), inputs_random* (gust_type = random, one per
seed, named inputs_random<seed>) and, optionally, inputs_event (gust_type = event). Per span, over each run's samples
from its erf.conductors.stats_start:

  LES      the wind at mid-span normal to the span, its mean and standard deviation, and the standard deviation of the
           horizontal speed there, sigma_res (resolved); the span's wind load per metre (its drag normal to the span
           over the chord), its mean and its peak; the peak tension and the largest swing either way.
  factor   the mean load in the RANS wind without gusts; from gusts.csv the RANS k, sigma_u, the gust factor G and its
           peak load G times the mean, and whether the linear form holds there (valid).
  random   the peak load and the peak tension of each seed's run, and their means over the seeds.
  event    the peak load and the peak tension (when inputs_event was run).
  c        sigma_u / sqrt(k) the LES gives at the span: sqrt(sigma_res^2 + (2/3) k_sgs) / sqrt(k_RANS), k_sgs the LES's
           subgrid energy at the span's height from the precursor's mean profile (--sgs_profile, its mean over
           --sgs_from on), and the resolved part alone.

Writes gust_comparison.csv and gust_comparison.png in RANS_DIR.

Inputs are read as ERF reads them: FILE = includes expanded where they stand, and the last definition of a key wins.

Usage: compare_gusts.py LES_DIR RANS_DIR [--les_inputs FILE] [--sgs_profile PATH] [--sgs_from 7200]
"""
import argparse
import glob
import math
import os
import re
import sys

import numpy as np


def expand(R, name, depth=0):
    """An inputs file with its FILE = includes expanded in place (paths relative to R), as ERF's ParmParse reads it."""
    if depth > 10:
        sys.exit(f"FILE = includes nested too deep at {name}")
    out = []
    for line in open(os.path.join(R, name)).read().split("\n"):
        m = re.match(r"^\s*FILE\s*=\s*(\S+)", line)
        out.append(expand(R, m.group(1), depth + 1) if m else line)
    return "\n".join(out)


def inputs_value(text, key, default=None):
    """The value of key, its last definition winning, as in ParmParse; default when it is not set."""
    m = re.findall(r"^\s*" + re.escape(key) + r"\s*=\s*(\S+)", text, re.M)
    return m[-1].strip('"') if m else default


def read_run(R, inputs):
    """The span load series of one run: {(line, k): dict(t, q, vn, speed, tension, swing, chord, z)}, and its diagnostics dir."""
    net = expand(R, inputs)
    diag = os.path.join(R, inputs_value(net, "erf.conductors.diagnostics_dir", "conductors"))
    t_start = float(inputs_value(net, "erf.conductors.stats_start", "0"))
    pts = {}
    for l in open(os.path.join(diag, "ground.dat")).read().strip().split("\n")[1:]:
        p = l.split()
        if p[1] != "transformer":
            pts.setdefault(p[0], []).append((float(p[2]), float(p[3]), float(p[5]) - float(p[4])))
    out = {}
    for line, P in pts.items():
        for k in range(1, len(P)):
            root = inputs_value(net, f"erf.conductors.{line}.output_root")
            f = (os.path.join(R, root) if root else os.path.join(diag, line)) + (f"_span{k}.dat" if len(P) > 2 else ".dat")
            h = open(f).readline().split()
            d = np.loadtxt(f, skiprows=1)
            d = d[d[:, 0] >= t_start]
            c = {n: i for i, n in enumerate(h)}
            (xa, ya, za), (xb, yb, zb) = P[k - 1], P[k]
            L = math.hypot(xb - xa, yb - ya)
            n = np.array([-(yb - ya) / L, (xb - xa) / L])
            vn = d[:, c["mid_u"]] * n[0] + d[:, c["mid_v"]] * n[1]
            sign = 1.0 if vn.mean() >= 0 else -1.0
            out[(line, k)] = dict(t=d[:, 0], vn=sign * vn, speed=np.hypot(d[:, c["mid_u"]], d[:, c["mid_v"]]),
                                  q=sign * (d[:, c["drag_x"]] * n[0] + d[:, c["drag_y"]] * n[1]) / L,
                                  tension=d[:, c["max_tension"]], swing=np.abs(d[:, c["swing_deg"]]), chord=L,
                                  z=0.5 * (za + zb))
    return out, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("les")
    ap.add_argument("rans")
    ap.add_argument("--les_inputs", default=None, help="the LES run's inputs file, a name in LES_DIR (default: the first inputs* there)")
    ap.add_argument("--sgs_profile", default=None, help="the precursor's mean_profiles.dat (time z u v w rho theta ke ...)")
    ap.add_argument("--sgs_from", type=float, default=7200.0)
    args = ap.parse_args()
    L_dir, R_dir = os.path.abspath(args.les), os.path.abspath(args.rans)
    les, _ = read_run(L_dir, os.path.basename(args.les_inputs or sorted(glob.glob(os.path.join(L_dir, "inputs*")))[0]))
    factor, fdiag = read_run(R_dir, "inputs_factor")
    # the random runs' inputs files, inputs_random<seed>, and nothing else of that prefix (a run script's markers)
    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(R_dir, "inputs_random*")))
    randoms = [read_run(R_dir, f)[0] for f in names if re.fullmatch(r"inputs_random\d+", f)]
    if not randoms:
        sys.exit(f"no inputs_random* runs in {R_dir}")
    # the event when its deck is there and it was run (its diagnostics directory holds the run's ground.dat)
    event = None
    if os.path.exists(os.path.join(R_dir, "inputs_event")):
        ediag = inputs_value(expand(R_dir, "inputs_event"), "erf.conductors.diagnostics_dir", "conductors")
        if os.path.exists(os.path.join(R_dir, ediag, "ground.dat")):
            event = read_run(R_dir, "inputs_event")[0]
    rows = [l.strip().split(",") for l in open(os.path.join(fdiag, "gusts.csv")).read().strip().split("\n")]
    col = {h: i for i, h in enumerate(rows[0])}
    gusts = {(r[0], int(r[1])): r for r in rows[1:]}
    ksgs = None
    if args.sgs_profile:
        p = np.loadtxt(args.sgs_profile)
        p = p[p[:, 0] >= args.sgs_from]
        zs = np.unique(p[:, 1])
        ksgs = (zs, np.array([p[p[:, 1] == z, 7].mean() for z in zs]))

    out = []
    for key in sorted(les):
        a, f, g = les[key], factor[key], gusts[key]
        sigma_res = a["speed"].std()
        k_rans = float(g[col["k"]])
        k_sub = float(np.interp(a["z"], *ksgs)) if ksgs else 0.0
        o = dict(line=key[0], span=key[1], z=a["z"], chord=a["chord"],
                 les_vn=a["vn"].mean(), les_sigma_vn=a["vn"].std(), les_sigma=sigma_res, les_ksgs=k_sub,
                 les_q_mean=a["q"].mean(), les_q_peak=a["q"].max(), les_g=a["q"].max() / a["q"].mean(),
                 les_tension=a["tension"].max(), les_swing=a["swing"].max(),
                 rans_vn=f["vn"].mean(), rans_k=k_rans, rans_q_mean=f["q"].mean(),
                 factor_g=float(g[col["gust_response"]]), factor_q_peak=float(g[col["gust_response"]]) * f["q"].mean(),
                 valid=int(g[col["valid"]]),
                 c_res=sigma_res / math.sqrt(k_rans), c_les=math.sqrt(sigma_res ** 2 + 2.0 / 3.0 * k_sub) / math.sqrt(k_rans),
                 random_q_peak=np.mean([r[key]["q"].max() for r in randoms]),
                 random_q_peak_min=min(r[key]["q"].max() for r in randoms), random_q_peak_max=max(r[key]["q"].max() for r in randoms),
                 random_tension=np.mean([r[key]["tension"].max() for r in randoms]),
                 random_swing=np.mean([r[key]["swing"].max() for r in randoms]),
                 random_sigma_vn=np.mean([r[key]["vn"].std() for r in randoms]))
        if event:
            o.update(event_q_peak=event[key]["q"].max(), event_tension=event[key]["tension"].max())
        out.append(o)

    keys = list(out[0].keys())
    with open(os.path.join(R_dir, "gust_comparison.csv"), "w") as t:
        t.write(",".join(keys) + "\n")
        for o in out:
            t.write(",".join(o[k] if isinstance(o[k], str) else ("%d" % o[k] if isinstance(o[k], int) else "%.6g" % o[k])
                             for k in keys) + "\n")
    print("%-5s %4s %5s %5s | %6s %6s %6s %6s %5s | %6s %6s %5s %6s %6s %6s %5s | %5s %5s" % (
        "line", "span", "z", "L", "Vn_LES", "q_LES", "qp_LES", "G_LES", "T_LES", "Vn_RNS", "q_RANS", "G_fac", "qp_fac",
        "qp_rnd", "T_rnd", "valid", "c_res", "c_LES"))
    for o in out:
        print("%-5s %4d %5.1f %5.0f | %6.2f %6.3f %6.3f %6.2f %5.0f | %6.2f %6.3f %5.2f %6.3f %6.3f %6.0f %5d | %5.2f %5.2f" % (
            o["line"], o["span"], o["z"], o["chord"], o["les_vn"], o["les_q_mean"], o["les_q_peak"], o["les_g"],
            o["les_tension"], o["rans_vn"], o["rans_q_mean"], o["factor_g"], o["factor_q_peak"], o["random_q_peak"],
            o["random_tension"], o["valid"], o["c_res"], o["c_les"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(out))
    lab = [f'{o["line"]}.{o["span"]}' for o in out]
    ax[0].bar(x - 0.3, [o["les_q_peak"] for o in out], 0.2, label="LES peak")
    ax[0].bar(x - 0.1, [o["factor_q_peak"] for o in out], 0.2, label="RANS factor: G x mean")
    ax[0].bar(x + 0.1, [o["random_q_peak"] for o in out], 0.2,
              yerr=[[o["random_q_peak"] - o["random_q_peak_min"] for o in out],
                    [o["random_q_peak_max"] - o["random_q_peak"] for o in out]], label="RANS random: peak (seeds)")
    if event:
        ax[0].bar(x + 0.3, [o["event_q_peak"] for o in out], 0.2, label="RANS event: peak")
    ax[0].plot(x, [o["les_q_mean"] for o in out], "k_", ms=14, mew=2, label="LES mean")
    ax[0].plot(x, [o["rans_q_mean"] for o in out], "r_", ms=14, mew=2, label="RANS mean")
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=45); ax[0].set_ylabel("span load normal to the span (N/m)")
    ax[0].legend(fontsize=7); ax[0].set_title("peak and mean wind load per metre of span")
    ax[1].bar(x - 0.2, [o["les_tension"] for o in out], 0.4, label="LES")
    ax[1].bar(x + 0.2, [o["random_tension"] for o in out], 0.4, label="RANS random (mean of seeds)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab, rotation=45); ax[1].set_ylabel("peak tension (N)")
    ax[1].set_ylim(0.9 * min(o["les_tension"] for o in out), 1.05 * max(max(o["les_tension"], o["random_tension"]) for o in out))
    ax[1].legend(fontsize=8); ax[1].set_title("peak tension")
    ax[2].plot(x, [o["c_les"] for o in out], "o", label="LES: sqrt(sigma_res^2 + 2/3 k_sgs) / sqrt(k_RANS)")
    ax[2].plot(x, [o["c_res"] for o in out], "s", mfc="none", label="resolved only")
    ax[2].axhline(2.5 * 0.5562, color="k", lw=1, label="default c = 2.5 Cmu0")
    ax[2].set_xticks(x); ax[2].set_xticklabels(lab, rotation=45); ax[2].set_ylabel("c = sigma_u / sqrt(k)")
    ax[2].legend(fontsize=8); ax[2].set_title("the LES's sigma_u over the RANS k at each span")
    fig.tight_layout()
    fig.savefig(os.path.join(R_dir, "gust_comparison.png"), dpi=110)
    print("wrote", os.path.join(R_dir, "gust_comparison.csv"), "and gust_comparison.png")


if __name__ == "__main__":
    main()
