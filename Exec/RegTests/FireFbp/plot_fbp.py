#!/usr/bin/env python3
"""FBP rate of spread against wind for the fuel types, with the run's points.

    python3 plot_fbp.py [--out fbp.png]

Left: the surface rate on flat ground against the 10 m wind for the
sixteen fuel types at FFMC 90, BUI 60 (grass 80 % cured, mixedwood at 50 %),
from fbp_reference.py. Right: the step-1 head rates of the decks against
the reference at the run's wind.
"""
import argparse, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fbp_reference import ros, COEF

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="fbp.png"); a = ap.parse_args()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    W = np.linspace(0, 12, 60)
    for t in list(COEF) + ["M1", "M2"]:
        axs[0].plot(W * 3.6, [ros(t, 90, 60, w, 80, 50, 50) * 60 for w in W], label=t, lw=1)
    axs[0].set_xlabel("10 m wind [km/h]"); axs[0].set_ylabel("rate of spread [m/min]"); axs[0].set_yscale("log")
    axs[0].legend(fontsize=6, ncol=3); axs[0].grid(alpha=0.3); axs[0].set_title("FBP surface rate, FFMC 90, BUI 60, flat")
    decks = (("fbp_c2", "C2", 90, 60, 60, 50), ("fbp_o1b", "O1B", 92, 40, 80, 50), ("fbp_m1", "M1", 90, 60, 60, 60))
    for name, t, ffmc, bui, cur, pc in decks:
        log = open(f"run_{name}.log").read()
        w = float(re.search(r"Max reference wind: ([^ ]+) m", log).group(1)); r = float(re.search(r"max_ROS=([^ ]+) ", log).group(1))
        axs[1].plot(ros(t, ffmc, bui, w, cur, pc), r, "o", label=f"{t} at {w:.1f} m/s")
    lim = axs[1].get_xlim(); axs[1].plot(lim, lim, "k:", lw=0.8)
    axs[1].set_xlabel("reference ROS [m/s]"); axs[1].set_ylabel("ERF step-1 head ROS [m/s]"); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)

if __name__ == "__main__":
    main()
