#!/usr/bin/env python3
"""The Scott-Burgan map and the burned area at 60 s.

    python3 plot_sb40.py [--out sb40.png]

Left: the fuel map (codes). Right: the phi = 0 contours of the native map
deck and the crosswalk deck over the map.
"""
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
from check_sb40 import read_map

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="sb40.png"); a = ap.parse_args()
    m, hdr = read_map("fuel_map_sb40.asc"); dx = hdr["cellsize"]
    ny, nx = m.shape; x = (np.arange(nx) + 0.5) * dx; y = (np.arange(ny) + 0.5) * dx
    fig, ax = plt.subplots(figsize=(9, 4.6))
    codes = sorted(set(m.flatten().tolist())); idx = np.vectorize({c: i for i, c in enumerate(codes)}.get)(m)
    im = ax.pcolormesh(x, y, idx, cmap="tab10", vmin=-0.5, vmax=9.5, shading="auto", alpha=0.6)
    names = {102: "GR2", 142: "SH2", 183: "TL3", 91: "NB1 urban", 98: "NB8 water"}
    for i, c in enumerate(codes):
        ax.plot([], [], "s", color=plt.cm.tab10(i), alpha=0.6, label=f"{c} {names.get(c, '')}")
    for name, lab, col in (("sb_map", "native Scott-Burgan", "k"), ("sb_map_crosswalk", "crosswalk to Anderson", "r")):
        ds = yt.load(f"plt_fire_{name}/plt_fire_00480"); fn = [f for _, f in ds.field_list if "phi" in f][0]
        g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        phi = g[("boxlib", fn)].value[:, :, 0]; X = g["index", "x"].value[:, :, 0]; Y = g["index", "y"].value[:, :, 0]
        ax.contour(X, Y, phi, levels=[0.0], colors=[col]); ax.plot([], [], color=col, label=f"perimeter at 60 s, {lab}")
    ax.plot(120.0, 80.0, "k+", ms=10); ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(fontsize=7, loc="upper right"); ax.set_title("Scott-Burgan fuel map and the fire at 60 s")
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)

if __name__ == "__main__":
    main()
