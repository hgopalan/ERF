#!/usr/bin/env python3
"""Interior state behind the perimeter against the distance inside it.

    python3 plot_perimeter.py [--fuel_plotfile plt_fire_interior/plt_fire_00000]
                              [--arrival_plotfile plt_fire_spinup_interior/plt_fire_00480 --t_ign 30]
                              [--ros 0.5] [--tau 60] [--out perimeter.png]

Left: the fuel load of every burned cell of the interior deck's step-0
plotfile against d = -phi, with w0 exp(-d/(R tau)). Right: the arrival time
of every stamped cell of the spin-up interior deck against its geometric
distance d0 inside the square, with max(t_ign - d0/R, 0).
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuel_plotfile", default="plt_fire_interior/plt_fire_00000")
    ap.add_argument("--arrival_plotfile", default="plt_fire_spinup_interior/plt_fire_00480"); ap.add_argument("--t_ign", type=float, default=30.0)
    ap.add_argument("--ros", type=float, default=0.5); ap.add_argument("--tau", type=float, default=60.0); ap.add_argument("--out", default="perimeter.png")
    a = ap.parse_args()
    def grid(pf):
        ds = yt.load(pf); names = [f for _, f in ds.field_list]
        pick = lambda s: ("boxlib", [n for n in names if s in n][0])
        g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        return (g[pick("phi")].value[:, :, 0], g[pick("fuel_load")].value[:, :, 0], g[pick("arrival")].value[:, :, 0],
                g["index", "x"].value[:, :, 0], g["index", "y"].value[:, :, 0], float(ds.current_time))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    phi, fuel, at, xc, yc, t = grid(a.fuel_plotfile)
    inside = phi < 0; d = -phi[inside]; w0 = np.median(fuel[~inside]); dd = np.linspace(0, d.max(), 100)
    axs[0].scatter(d, fuel[inside], s=6, label="cells"); axs[0].plot(dd, w0 * np.exp(-dd / (a.ros * a.tau)), "k-", label="w0 exp(-d/(R tau))")
    axs[0].set_xlabel("distance inside the perimeter d = -phi [m]"); axs[0].set_ylabel("fuel load [kg/m2]"); axs[0].legend(); axs[0].grid(alpha=0.3)
    axs[0].set_title(f"interior deck at t = {t:.0f} s", fontsize=10)
    phi, fuel, at, xc, yc, t = grid(a.arrival_plotfile)
    d0 = np.minimum(np.minimum(xc - 80.0, 120.0 - xc), np.minimum(yc - 60.0, 100.0 - yc))
    sel = (phi < 0) & (at < a.t_ign - 1e-6); dd = np.linspace(0, d0[sel].max(), 100)
    axs[1].scatter(d0[sel], at[sel], s=6, label="stamped cells"); axs[1].plot(dd, np.maximum(a.t_ign - dd / a.ros, 0.0), "k-", label="max(t_ign - d0/R, 0)")
    axs[1].set_xlabel("geometric distance inside the square d0 [m]"); axs[1].set_ylabel("arrival time [s]"); axs[1].legend(); axs[1].grid(alpha=0.3)
    axs[1].set_title(f"spin-up interior deck at t = {t:.0f} s, stamped at {a.t_ign:.0f} s", fontsize=10)
    fig.suptitle(f"interior state behind the perimeter, R = {a.ros} m/s, tau = {a.tau} s")
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)

if __name__ == "__main__":
    main()
