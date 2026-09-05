#!/usr/bin/env python3
"""Interior state behind the perimeter against the distance inside it.

    python3 plot_perimeter.py [--plotfile plt_fire_interior/plt_fire_00000] [--ros 0.5] [--tau 60] [--out perimeter.png]

Scatter of the fuel load and arrival time of every burned cell of the
plotfile against d = -phi, with the expected curves w0 exp(-d/(R tau)) and
-d/R.
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plotfile", default="plt_fire_interior/plt_fire_00000"); ap.add_argument("--ros", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=60.0); ap.add_argument("--out", default="perimeter.png")
    a = ap.parse_args()
    ds = yt.load(a.plotfile); names = [f for _, f in ds.field_list]
    pick = lambda s: ("boxlib", [n for n in names if s in n][0])
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    phi = g[pick("phi")].value[:, :, 0]; fuel = g[pick("fuel_load")].value[:, :, 0]; at = g[pick("arrival")].value[:, :, 0]
    inside = phi < 0; d = -phi[inside]; w0 = np.median(fuel[~inside]); dd = np.linspace(0, d.max(), 100)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(d, fuel[inside], s=6, label="cells"); axs[0].plot(dd, w0 * np.exp(-dd / (a.ros * a.tau)), "k-", label="w0 exp(-d/(R tau))")
    axs[0].set_xlabel("distance inside the perimeter d [m]"); axs[0].set_ylabel("fuel load [kg/m2]"); axs[0].legend(); axs[0].grid(alpha=0.3)
    axs[1].scatter(d, at[inside], s=6, label="cells"); axs[1].plot(dd, float(ds.current_time) - dd / a.ros, "k-", label="t_ign - d/R")
    axs[1].set_xlabel("distance inside the perimeter d [m]"); axs[1].set_ylabel("arrival time [s]"); axs[1].legend(); axs[1].grid(alpha=0.3)
    fig.suptitle(f"interior state at t = {float(ds.current_time):.0f} s, R = {a.ros} m/s, tau = {a.tau} s")
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)

if __name__ == "__main__":
    main()
