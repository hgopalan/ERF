#!/usr/bin/env python3
"""Burned shapes at 60 s under the three spread options.

    python3 plot_shapes.py [--out shapes.png]

Draws the phi = 0 contour of plt_fire_off, plt_fire_directional,
plt_fire_ellipse and plt_fire_ellipse_lw3 at step 480 on one map, with the
ignition point and the wind direction.
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="shapes.png"); a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, lab in (("off", "disc (historical)"), ("directional", "directional projection"), ("ellipse", "ellipse, Anderson L/W"), ("ellipse_lw3", "ellipse, L/W = 3")):
        ds = yt.load(f"plt_fire_{name}/plt_fire_00480"); names = [f for _, f in ds.field_list]
        fphi = ("boxlib", [n for n in names if "phi" in n][0])
        g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        phi = g[fphi].value[:, :, 0]; x = g["index", "x"].value[:, :, 0]; y = g["index", "y"].value[:, :, 0]
        ax.contour(x, y, phi, levels=[0.0], colors=[plt.cm.tab10(len(ax.collections) % 10)])
        ax.plot([], [], color=plt.cm.tab10((len(ax.collections) - 1) % 10), label=lab)
    ax.plot(60.0, 80.0, "k+", ms=10, label="ignition"); ax.annotate("wind", xy=(45, 105), xytext=(32, 105), arrowprops=dict(arrowstyle="->"))
    ax.set_aspect("equal"); ax.set_xlim(30, 100); ax.set_ylim(50, 110); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("fire perimeter at 60 s"); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)

if __name__ == "__main__":
    main()
