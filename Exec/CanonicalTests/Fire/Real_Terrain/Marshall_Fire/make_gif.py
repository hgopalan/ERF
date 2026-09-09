#!/usr/bin/env python3
"""Animate the Marshall Fire run: one frame per fire plotfile, the burned area
and the fire front over a shaded relief of the terrain, with the fire-grid
wind as a sparse quiver. Writes marshall_fire.gif (and the frames as PNGs
under frames/).

    python3 make_gif.py [--every N] [--out marshall_fire.gif]
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
    sys.exit(f"make_gif.py needs yt, matplotlib and Pillow: {e}")

ap = argparse.ArgumentParser()
ap.add_argument("--every", type=int, default=1, help="use every N-th plotfile")
ap.add_argument("--out", default="marshall_fire.gif")
ap.add_argument("--fps", type=float, default=2.0)
ap.add_argument("--crop", default="0,10,6,14", help="x0,x1,y0,y1 window in km; \"\" for the whole domain")
args = ap.parse_args()

files = sorted(glob.glob("plt_fire_?????"))[::args.every]
if not files:
    sys.exit("no plt_fire_????? plotfiles here")
info = json.load(open("marshall_domain.json")) if os.path.exists("marshall_domain.json") else {}

# terrain raster for the relief
v = np.loadtxt("terrain_marshall.txt")
nt = int(v[0]); tz = v[2 + 2 * nt:].reshape(nt, nt)
L = float(v[2 + nt - 1])
ls = LightSource(azdeg=315, altdeg=45)
# colour-shaded elevation: hillshade over the terrain colormap, plus 50 m contours
relief = ls.shade(tz.T, cmap=plt.cm.terrain, blend_mode="soft", vert_exag=2,
                  dx=L / (nt - 1), dy=L / (nt - 1), vmin=-0.3 * tz.max(), vmax=1.05 * tz.max())
xt = np.linspace(0, L / 1000, nt)

os.makedirs("frames", exist_ok=True)
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
    fig, ax = plt.subplots(figsize=(7.7, 6.3))   # 1.22 aspect, the review slide's picture box
    ax.imshow(relief, origin="lower", extent=[0, L / 1000, 0, L / 1000])
    ax.contour(xt, xt, tz.T, levels=np.arange(0, tz.max(), 50.0), colors="k", linewidths=0.3, alpha=0.5)
    burned = np.ma.masked_where(at < 0, at / 60.0)
    im = ax.imshow(burned.T, cmap="hot", origin="lower", extent=[0, L / 1000, 0, L / 1000],
                   vmin=0, vmax=max(1.0, t / 60.0), alpha=0.95)
    ax.contour(xc, xc, phi.T, levels=[0.0], colors="red", linewidths=1.5)
    s = max(1, nx // 48)
    ax.quiver(xc[::s], xc[::s], u[::s, ::s].T, w[::s, ::s].T, color="navy", scale=250, width=0.0025)
    for ig in info.get("ignitions", []):
        ax.plot(ig["x"] / 1000, ig["y"] / 1000, "o", mfc="none", mec="white", mew=1.2, ms=8)
    if args.crop:
        cx0, cx1, cy0, cy1 = [float(c) for c in args.crop.split(",")]
        ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1)
    else:
        ax.set_xlim(0, L / 1000); ax.set_ylim(0, L / 1000)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
    n_b = int((at >= 0).sum())
    ax.set_title(f"Marshall Fire  t = {t / 60:6.1f} min   burned {n_b * dx * dx / 1e4:7.1f} ha")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label("arrival time [min]")
    fig.tight_layout()
    fn = f"frames/frame_{k:04d}.png"; fig.savefig(fn, dpi=90); plt.close(fig)
    frames.append(Image.open(fn).convert("P", palette=Image.ADAPTIVE))
    print(f"{pf}: t = {t:.0f} s, burned {n_b} cells")

frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0)
print(f"wrote {args.out} with {len(frames)} frames")
