#!/usr/bin/env python3
"""Total fire power and fuel left under the two burnout models.

    python3 plot_burnout.py [--out burnout.png] [--dt 0.125]

Reads the run_*.log files of run_burnout.sh and draws the total power on
the fire grid and the fuel left against time for grass and timber litter
under the crossing-time and the SFIRE burn-time models.
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
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="burnout.png"); ap.add_argument("--dt", type=float, default=0.125)
    a = ap.parse_args()
    runs = (("residence", "grass, crossing time"), ("sfire_grass", "grass, SFIRE 8.2 s"),
            ("residence_litter", "litter, crossing time"), ("sfire_litter", "litter, SFIRE 1057 s"))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for name, lab in runs:
        P = series(f"run_{name}.log", r"total_power_W=([^ ]+)"); F = series(f"run_{name}.log", r"fuel_kg=([^ \n]+)")
        t = np.arange(len(P)) * a.dt
        axs[0].plot(t, P / 1e6, label=lab); axs[1].plot(t, F[0] - F, label=lab)
    axs[0].set_xlabel("time [s]"); axs[0].set_ylabel("total fire power [MW]"); axs[0].set_yscale("log"); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    axs[1].set_xlabel("time [s]"); axs[1].set_ylabel("fuel consumed [kg]"); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)

if __name__ == "__main__":
    main()
