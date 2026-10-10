#!/usr/bin/env python3
"""Map movie of the MRF and YSUNew runs overlaid on one map, each with its own colormap.

    python3 make_overlay_movie.py MRF_RUN_DIR YSUNEW_RUN_DIR TAG      (env ONLY_LAST=1: last frame only)

Needs USGS imagery of the domain, ../Real_Terrain/Marshall_Fire/basemap_marshall_z14.tif, which
`python3 ../../../../Tools/make_map_movie.py --fetch usgs` writes when run in that folder.
The imagery is warped onto the model's UTM square as in Exec/Tools/make_map_movie.py. Each
run's burned area is coloured by minutes since ignition on its own one-hue ramp (MRF blue,
YSUNew orange) with its front drawn in the same hue; the ignition and two observed arrivals
from the Boulder County after-action report are marked. The legend gives each run's burned
area and head distance along HEAD_DIR (env, default 87 degrees, downwind of a wind from 267).
Writes marshall_overlay_TAG.mp4 and marshall_overlay_TAG_last.png (frames in frames_overlay_TAG/).
"""
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import yt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from pyproj import Transformer

yt.set_log_level(50)
HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, "..", "Real_Terrain", "Marshall_Fire")
if len(sys.argv) != 4:
    sys.exit(__doc__)
mrf_dir, ysu_dir, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
T_IGN, IGN = 1200.0, (5000.0, 11000.0)
HEAD_DIR = float(os.environ.get("HEAD_DIR", "87"))
FPS, DPI, RES = 6, 115, 8.0
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE = ["#fde3d3", "#f9bf9d", "#f39767", "#eb6834", "#c9501f", "#9c3b14", "#6e290c"]
RUNS = [(ysu_dir, "YSUNew", "#eb6834", LinearSegmentedColormap.from_list("ysu", ORANGE)),   # drawn first
        (mrf_dir, "MRF", "#2a78d6", LinearSegmentedColormap.from_list("mrf", BLUE))]
OBS = [("Costco, Superior", -105.1745, 39.9557, "observed 12:18 (~75 min)"),
       ("Home Depot, Louisville", -105.1687, 39.9603, "observed 12:45 (~100 min)")]

basemap = os.path.join(CASE, "basemap_marshall_z14.tif")
if not os.path.exists(basemap):
    sys.exit(f"{basemap} not found: run Exec/Tools/make_map_movie.py --fetch usgs in {CASE} first")
info = json.load(open(os.path.join(CASE, "marshall_domain.json")))
L = float(info["domain_m"])
zone = info["utm_zone"]
utm = f"EPSG:{(32600 if zone.upper().endswith('N') else 32700) + int(zone[:-1])}"
to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
x0, y0 = to_utm.transform(*info["sw_corner_lonlat"])
obs_xy = [(n, (to_utm.transform(lo, la)[0] - x0) / 1000, (to_utm.transform(lo, la)[1] - y0) / 1000, w) for n, lo, la, w in OBS]
nb = int(round(L / RES))
rgb = np.zeros((3, nb, nb), np.uint8)
with rasterio.open(basemap) as src:
    credit = src.tags().get("attribution", "Imagery: U.S. Geological Survey")
    for b in range(3):
        reproject(source=src.read(b + 1), destination=rgb[b], src_transform=src.transform, src_crs=src.crs,
                  dst_transform=from_origin(x0, y0 + L, RES, RES), dst_crs=utm, resampling=Resampling.bilinear)
rgb = rgb.transpose(1, 2, 0)

files = {d: sorted(glob.glob(f"{d}/plt_fire_?????")) for d, *_ in RUNS}
times = {d: np.array([float(yt.load(p).current_time) for p in files[d]]) for d, *_ in RUNS}
t_end = min(times[d][-1] for d in files)
t_frames = [t for t in times[mrf_dir] if T_IGN - 1 <= t <= t_end + 1 and np.min(np.abs(times[ysu_dir] - t)) < 15.0]
if os.environ.get("ONLY_LAST"):
    t_frames = t_frames[-1:]


def load(d, t):
    ds = yt.load(files[d][int(np.argmin(np.abs(times[d] - t)))])
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    return np.asarray(g[("boxlib", "fire_arrival_time")])[:, :, 0], np.asarray(g[("boxlib", "fire_phi")])[:, :, 0]


# crop to both final burned areas and the observed points, with room for the labels on the right
xs, ys = [IGN[0] / 1000] + [o[1] for o in obs_xy], [IGN[1] / 1000] + [o[2] for o in obs_xy]
for d, *_ in RUNS:
    at, _ = load(d, t_frames[-1] if t_frames else t_end)
    ii, jj = np.nonzero(at >= 0)
    dx = L / at.shape[0]
    if ii.size:
        xs += [(ii.min() + 0.5) * dx / 1000, (ii.max() + 0.5) * dx / 1000]
        ys += [(jj.min() + 0.5) * dx / 1000, (jj.max() + 0.5) * dx / 1000]
cx0, cx1, cy0, cy1 = min(xs) - 1.2, max(xs) + 2.6, min(ys) - 1.0, max(ys) + 1.0
w, h = cx1 - cx0, cy1 - cy0
if w / h < 1.45:
    cx0, cx1 = (cx0 + cx1) / 2 - 0.725 * h, (cx0 + cx1) / 2 + 0.725 * h
else:
    cy0, cy1 = (cy0 + cy1) / 2 - w / 2.9, (cy0 + cy1) / 2 + w / 2.9

ex, ey = np.sin(np.radians(HEAD_DIR)), np.cos(np.radians(HEAD_DIR))
ext = [0, L / 1000, 0, L / 1000]
out_dir = os.path.join(os.getcwd(), f"frames_overlay_{TAG}")
os.makedirs(out_dir, exist_ok=True)
for old in glob.glob(f"{out_dir}/frame_*.png"):
    os.remove(old)
halo = [pe.Stroke(linewidth=3.2, foreground="w"), pe.Normal()]
fn = None
for k, t in enumerate(t_frames):
    fig = plt.figure(figsize=(11.8, 7.2))
    ax = fig.add_axes([0.06, 0.08, 0.70, 0.82])
    ax.imshow(rgb, origin="upper", extent=ext, interpolation="bilinear")
    handles = []
    for n, (d, name, col, cmap) in enumerate(RUNS):
        at, phi = load(d, t)
        dx = L / at.shape[0]
        xc = (np.arange(at.shape[0]) + 0.5) * dx / 1000
        burned = at >= 0
        im = ax.imshow(np.ma.masked_where(~burned, (at - T_IGN) / 60).T, cmap=cmap, origin="lower", extent=ext,
                       vmin=0, vmax=45, alpha=0.62, interpolation="nearest")
        head = 0.0
        if burned.any():
            cs = ax.contour(xc, xc, phi.T, levels=[0.0], colors=[col], linewidths=1.8)
            cs.set_path_effects(halo)
            ii, jj = np.nonzero(burned)
            head = float(np.max(((ii + 0.5) * dx - IGN[0]) * ex + ((jj + 0.5) * dx - IGN[1]) * ey)) / 1000
        acres = burned.sum() * dx * dx / 4046.86
        handles.append(Line2D([], [], color=col, lw=2.4, label=f"{name}: {acres:5.0f} acres, head {head:4.2f} km"))
        cax = fig.add_axes([0.79 + 0.075 * n, 0.12, 0.018, 0.70])
        cb = fig.colorbar(im, cax=cax)
        cb.solids.set_alpha(1)
        cb.set_label(f"{name} arrival [min]", color="#0b0b0b")
        cb.ax.tick_params(colors="#52514e", labelsize=8)
    ax.plot(IGN[0] / 1000, IGN[1] / 1000, "o", mfc="none", mec="cyan", mew=1.8, ms=10)
    for oname, ox, oy, when in obs_xy:
        ax.plot(ox, oy, "D", mfc="#1baf7a", mec="k", ms=7)
        ax.annotate(f"{oname}\n{when}", (ox, oy), xytext=(8, 6), textcoords="offset points", fontsize=8, color="w",
                    bbox=dict(facecolor="k", alpha=0.55, pad=1.5, linewidth=0))
    handles += [Line2D([], [], marker="o", ls="none", mfc="none", mec="cyan", mew=1.8, ms=9, label="ignition"),
                Line2D([], [], marker="D", ls="none", mfc="#1baf7a", mec="k", ms=7, label="observed arrival")]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.85, facecolor="w", edgecolor="#c3c2b7")
    ax.set_xlim(cx0, cx1)
    ax.set_ylim(cy0, cy1)
    ax.set_xlabel("x [km east of the domain's SW corner]")
    ax.set_ylabel("y [km north]")
    ax.text(0.99, 0.01, credit, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="w",
            bbox=dict(facecolor="k", alpha=0.45, pad=1.5, linewidth=0))
    ax.set_title(f"Idealized Marshall Fire, {TAG}:  {(t - T_IGN) / 60:4.1f} min after ignition", fontsize=13, fontweight="bold", loc="left")
    fn = f"{out_dir}/frame_{k:04d}.png"
    fig.savefig(fn, dpi=DPI)
    plt.close(fig)
    if k % 15 == 0:
        print(f"frame {k}/{len(t_frames)}: {(t - T_IGN) / 60:.1f} min")
if fn is None:
    sys.exit("no frames: the runs have no fire plotfiles after the ignition")
shutil.copy(fn, f"marshall_overlay_{TAG}_last.png")
if os.environ.get("ONLY_LAST"):
    sys.exit(0)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{out_dir}/frame_%04d.png",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,tpad=stop_mode=clone:stop_duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", f"marshall_overlay_{TAG}.mp4"], check=True)
print(f"wrote marshall_overlay_{TAG}.mp4 ({len(t_frames)} frames at {FPS} fps)")
