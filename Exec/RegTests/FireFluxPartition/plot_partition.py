#!/usr/bin/env python3
"""Plot the fluxes handed to the atmosphere under both partitions.

    python3 plot_partition.py [--out partition.png]

Reads run_legacy.log, run_cfbm.log and run_cfbm_wet.log written by
run_partition.sh and draws the sensible and latent flux maxima against time
(one coupling call per step, dt from the deck), with the CFBM to legacy
ratio of the sensible flux in the lower panel.
"""
import argparse, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def series(log, pat):
    out = []
    with open(log) as f:
        for line in f:
            m = re.search(pat, line)
            if m: out.append(float(m.group(1)))
    return np.array(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="partition.png"); ap.add_argument("--dt", type=float, default=0.125)
    a = ap.parse_args()
    runs = {"legacy (M_f 0.08)": "run_legacy.log", "cfbm (M_f 0.08)": "run_cfbm.log", "cfbm (M_f 0.30)": "run_cfbm_wet.log"}
    fig, axs = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    S = {}
    for name, log in runs.items():
        s = series(log, r"Sensible flux to the atmosphere: max ([^ ]+) W"); l = series(log, r"Max latent flux: ([^ ]+) W")
        t = np.arange(len(s)) * a.dt; S[name] = s
        axs[0].plot(t, s, label=f"sensible, {name}")
        if len(l): axs[0].plot(np.arange(len(l)) * a.dt, l, "--", label=f"latent, {name}")
    axs[0].set_ylabel("flux maximum on the atmospheric grid [W/m2]"); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    ref = S["legacy (M_f 0.08)"]
    for name in ("cfbm (M_f 0.08)",):
        n = min(len(ref), len(S[name])); ok = ref[:n] > 0
        axs[1].plot(np.arange(n)[ok] * a.dt, S[name][:n][ok] / ref[:n][ok], label=f"{name} / legacy, one-way")
    try:
        r2 = series("run_legacy_2way.log", r"Sensible flux to the atmosphere: max ([^ ]+) W"); c2 = series("run_cfbm_2way.log", r"Sensible flux to the atmosphere: max ([^ ]+) W")
        n = min(len(r2), len(c2)); ok = r2[:n] > 0
        axs[1].plot(np.arange(n)[ok] * a.dt, c2[:n][ok] / r2[:n][ok], lw=0.8, alpha=0.7, label="cfbm / legacy, two-way")
    except FileNotFoundError:
        pass
    axs[1].axhline(1 / 1.08, color="k", lw=0.8, ls=":", label="1 / (1 + 0.08)")
    axs[1].set_xlabel("time [s]"); axs[1].set_ylabel("sensible flux ratio"); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)

if __name__ == "__main__":
    main()
