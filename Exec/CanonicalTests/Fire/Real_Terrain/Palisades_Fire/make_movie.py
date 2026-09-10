#!/usr/bin/env python3
"""Animate the Palisades Fire run: one frame per fire plotfile, the burned area
and the fire front over a shaded relief of the terrain, the shoreline drawn from
the fuel map, and the fire-grid wind as a sparse quiver. Writes
palisades_fire.gif and .mp4 (and the frames as PNGs under frames/).

    python3 make_movie.py [--every N] [--crop x0,x1,y0,y1]
"""
import argparse, glob, json, os, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from PIL import Image
except ImportError as e:
    sys.exit(f"make_movie.py needs yt, matplotlib and Pillow: {e}")

ap = argparse.ArgumentParser()
ap.add_argument("--every", type=int, default=1, help="use every N-th plotfile")
ap.add_argument("--out", default="palisades_fire.gif")
ap.add_argument("--fps", type=float, default=4.0)
ap.add_argument("--dpi", type=int, default=130)
ap.add_argument("--tmax", type=float, default=0.0,
                help="fixed top of the arrival-time scale in minutes; 0 = the last plotfile's time")
ap.add_argument("--mp4", default="palisades_fire.mp4",
                help='H.264 video alongside the gif when ffmpeg is on PATH; "" to skip')
ap.add_argument("--crop", default="4,18,3,17", help='x0,x1,y0,y1 window in km; "" for the whole domain')
args = ap.parse_args()

files = sorted(glob.glob("plt_fire_?????"))[::args.every]
if not files:
    sys.exit("no plt_fire_????? plotfiles here")
info = json.load(open("palisades_domain.json")) if os.path.exists("palisades_domain.json") else {}

# The fire's own raster, which is the one the front actually climbs.
v = np.loadtxt("terrain_palisades_fire.txt")
nt = int(v[0]); tz = v[2 + 2 * nt:].reshape(nt, nt)
L = float(v[2 + nt - 1])
ls = LightSource(azdeg=315, altdeg=45)
relief = ls.shade(tz.T, cmap=plt.cm.terrain, blend_mode="soft", vert_exag=2,
                  dx=L / (nt - 1), dy=L / (nt - 1), vmin=-0.3 * tz.max(), vmax=1.05 * tz.max())
xt = np.linspace(0, L / 1000, nt)

# Shoreline: the non-burnable code in the fuel map.
sea = None
if os.path.exists("fuel_palisades.asc"):
    fm = np.loadtxt("fuel_palisades.asc", skiprows=6)      # rows run north first
    sea = (fm[::-1, :].T == 0).astype(float)               # -> (i, j) with j north
    xs_f = (np.arange(sea.shape[0]) + 0.5) * (L / sea.shape[0]) / 1000.0

os.makedirs("frames", exist_ok=True)
t_max_min = args.tmax if args.tmax > 0 else float(yt.load(files[-1]).current_time) / 60.0
frames = []
for k, pf in enumerate(files):
    ds = yt.load(pf)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    at = np.array(g["fire_arrival_time"])[:, :, 0]
    phi = np.array(g["fire_phi"])[:, :, 0]
    u = np.array(g["fire_wind_eff_u"])[:, :, 0]; w = np.array(g["fire_wind_eff_v"])[:, :, 0]
    t = float(ds.current_time)
    nx = at.shape[0]; dx = L / nx
    xc = (np.arange(nx) + 0.5) * dx / 1000.0
    fig, ax = plt.subplots(figsize=(8.4, 6.3))
    ax.imshow(relief, origin="lower", extent=[0, L / 1000, 0, L / 1000])
    ax.contour(xt, xt, tz.T, levels=np.arange(0, tz.max(), 50.0), colors="k", linewidths=0.3, alpha=0.5)
    if sea is not None:
        ax.contourf(xs_f, xs_f, sea.T, levels=[0.5, 1.5], colors=["#12457a"], alpha=0.85)
        ax.contour(xs_f, xs_f, sea.T, levels=[0.5], colors="#08325e", linewidths=0.8)
    burned = np.ma.masked_where(at < 0, at / 60.0)
    im = ax.imshow(burned.T, cmap="hot", origin="lower", extent=[0, L / 1000, 0, L / 1000],
                   vmin=0, vmax=t_max_min, alpha=0.95)
    ax.contour(xc, xc, phi.T, levels=[0.0], colors="red", linewidths=1.5)
    s = max(1, nx // 48)
    ax.quiver(xc[::s], xc[::s], u[::s, ::s].T, w[::s, ::s].T, color="navy", scale=400, width=0.0025)
    ig = info.get("ignition")
    if ig:
        ax.plot(ig["x"] / 1000, ig["y"] / 1000, "*", mfc="none", mec="white", mew=1.2, ms=13)
    if args.crop:
        cx0, cx1, cy0, cy1 = [float(c) for c in args.crop.split(",")]
        ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1)
    else:
        ax.set_xlim(0, L / 1000); ax.set_ylim(0, L / 1000)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
    n_b = int((at >= 0).sum())
    ha = n_b * dx * dx / 1e4
    ax.set_title(f"Palisades Fire  t = {t / 60:6.1f} min   burned {ha:7.1f} ha ({ha * 2.4711:6.0f} acres)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label("arrival time [min]")
    z_asl = info.get("floor_elevation_m_asl", 0.0)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.terrain,
                               norm=matplotlib.colors.Normalize(vmin=-0.3 * tz.max() + z_asl,
                                                                vmax=1.05 * tz.max() + z_asl))
    cbz = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.16, location="left")
    cbz.set_label("elevation [m ASL]")
    cbz.ax.set_ylim(z_asl + tz.min(), z_asl + tz.max())
    fig.tight_layout()
    fn = f"frames/frame_{k:04d}.png"; fig.savefig(fn, dpi=args.dpi); plt.close(fig)
    frames.append(Image.open(fn).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                                         dither=Image.Dither.NONE))
    print(f"{pf}: t = {t:.0f} s, burned {n_b} cells ({ha * 2.4711:.0f} acres)")

frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0)
print(f"wrote {args.out} with {len(frames)} frames")
if args.mp4:
    import shutil, subprocess
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", "frames/frame_%04d.png",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", args.mp4], check=True)
        print(f"wrote {args.mp4}")
    else:
        print("ffmpeg not on PATH, no mp4")
