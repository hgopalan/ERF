#!/usr/bin/env python3
"""Animate a real-terrain fire run (Marshall or Palisades) over map imagery:
the burned area coloured by arrival time, the fire front and the fire-grid
wind, on a basemap warped onto the model's UTM grid. Run it in the directory
holding the plotfiles and <case>_domain.json; writes <case>_fire_map.gif and
.mp4 (frames under frames_map/).

    cd <run directory>     # plt_fire_*, <case>_domain.json and the terrain raster
    python3 <ERF>/Exec/Tools/make_map_movie.py --fetch usgs            # download tiles once, then render
    python3 <ERF>/Exec/Tools/make_map_movie.py --basemap basemap.tif   # any RGB GeoTIFF, any CRS

Sources for --fetch: usgs (USGS imagery, public domain), usgstopo (USGS topo
map, public domain), esri (Esri World Imagery; display only, with Esri's
attribution). The credit line is printed on every frame. A new real-terrain
case needs an entry in CASES below.

--fetch stitches web-map tiles covering the domain into
basemap_<case>_z<zoom>.tif (EPSG:3857) and keeps them under basemap_tiles/, so
a rerun downloads nothing. Behind a TLS-inspecting proxy whose root is only in
the system keychain, Python's certificate bundle rejects the server; the fetch
then uses curl, which checks against the system trust store (verification
stays on either way).

The model domain is axis-aligned in UTM (zone from the domain json) with its
south-west corner at sw_corner_lonlat, so the basemap is reprojected onto
exactly that square and the fire fields are drawn in model x, y.
"""
import argparse, glob, json, math, os, shutil, subprocess, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling
    from pyproj import Transformer
    from PIL import Image
except ImportError as e:
    sys.exit(f"make_map_movie.py needs yt, matplotlib, rasterio, pyproj and Pillow: {e}")

SOURCES = {
    "usgs":     ("https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
                 "Imagery: U.S. Geological Survey"),
    "usgstopo": ("https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
                 "Map: U.S. Geological Survey"),
    "esri":     ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                 "Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community"),
}
# per-case defaults, matching each case's make_gif.py / make_movie.py
CASES = {
    "marshall":  dict(title="Marshall Fire", terrain="terrain_marshall.txt", crop="0,10,6,14", fps=2.0,
                      quiver_scale=250, acres=False),
    "palisades": dict(title="Palisades Fire", terrain="terrain_palisades_fire.txt", crop="4,18,3,17", fps=4.0,
                      quiver_scale=400, acres=True),
}
HALF = 20037508.342789244          # half the Web Mercator extent [m]

found = sorted(glob.glob("*_domain.json"))
ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=sorted(CASES), default=found[0].split("_domain.json")[0] if found else None,
                help="which case; default from the *_domain.json in this directory")
ap.add_argument("--basemap", default="", help="RGB GeoTIFF to draw under the fire (any CRS)")
ap.add_argument("--fetch", choices=sorted(SOURCES), help="download web-map tiles for the domain into a GeoTIFF first")
ap.add_argument("--zoom", type=int, default=14, help="tile zoom for --fetch; 14 is about 7-8 m per pixel")
ap.add_argument("--attribution", default="", help="credit line printed on every frame")
ap.add_argument("--res", type=float, default=8.0, help="basemap pixel size on the model grid [m]")
ap.add_argument("--every", type=int, default=1, help="use every N-th plotfile")
ap.add_argument("--fps", type=float, default=0.0, help="0 = the case default")
ap.add_argument("--dpi", type=int, default=130)
ap.add_argument("--tmax", type=float, default=0.0, help="top of the arrival-time scale [min]; 0 = last plotfile's time")
ap.add_argument("--alpha", type=float, default=0.6, help="opacity of the burned-area colours")
ap.add_argument("--contours", type=float, default=100.0, help="terrain contour interval [m]; 0 = none")
ap.add_argument("--crop", default=None, help="x0,x1,y0,y1 window in km; \"\" for the whole domain; default per case")
ap.add_argument("--out", default="", help="gif name; default <case>_fire_map.gif")
ap.add_argument("--mp4", default=None, help="video name; default <case>_fire_map.mp4; \"\" to skip")
args = ap.parse_args()
if args.case is None:
    sys.exit("no *_domain.json here; run in the case's run directory or pass --case")
C = CASES[args.case]
crop = C["crop"] if args.crop is None else args.crop
fps = args.fps or C["fps"]
out_gif = args.out or f"{args.case}_fire_map.gif"
out_mp4 = f"{args.case}_fire_map.mp4" if args.mp4 is None else args.mp4

info = json.load(open(f"{args.case}_domain.json"))
L = float(info["domain_m"])
zone = info["utm_zone"]
utm_crs = f"EPSG:{(32600 if zone.upper().endswith('N') else 32700) + int(zone[:-1])}"
to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
x0, y0 = to_utm.transform(*info["sw_corner_lonlat"])
ignitions = info.get("ignitions") or ([info["ignition"]] if "ignition" in info else [])

def fetch_basemap(src, z):
    """Stitch the tiles covering the domain into an EPSG:3857 GeoTIFF."""
    import requests
    url, credit = SOURCES[src]
    (w, s), (e, n) = info["sw_corner_lonlat"], info["ne_corner_lonlat"]
    N = 2 ** z
    def tx(lon): return int((lon + 180.0) / 360.0 * N)
    def ty(lat): return int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * N)
    xs, ys = range(tx(w), tx(e) + 1), range(ty(n), ty(s) + 1)
    print(f"{src} zoom {z}: {len(xs) * len(ys)} tiles")
    agent = "ERF-Hazard make_map_movie.py (research visualisation)"
    sess = requests.Session(); sess.headers["User-Agent"] = agent
    use_curl = False
    mosaic = np.zeros((len(ys) * 256, len(xs) * 256, 3), np.uint8)
    got = 0
    for j, ty_ in enumerate(ys):
        for i, tx_ in enumerate(xs):
            fn = os.path.join("basemap_tiles", src, str(z), str(ty_), f"{tx_}.img")
            if not os.path.exists(fn):
                os.makedirs(os.path.dirname(fn), exist_ok=True)
                u = url.format(z=z, y=ty_, x=tx_)
                if not use_curl:
                    try:
                        r = sess.get(u, timeout=30); r.raise_for_status()
                        open(fn, "wb").write(r.content)
                    except requests.exceptions.SSLError:
                        if not shutil.which("curl"):
                            raise
                        print("Python rejected the server certificate; fetching with curl (system trust store)")
                        use_curl = True
                if use_curl:
                    subprocess.run(["curl", "-fsS", "--retry", "2", "-A", agent, "-o", fn, u], check=True)
                got += 1
            mosaic[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = np.asarray(Image.open(fn).convert("RGB"))
    res = 2 * HALF / (N * 256)
    transform = from_origin(xs[0] / N * 2 * HALF - HALF, HALF - ys[0] / N * 2 * HALF, res, res)
    out = f"basemap_{args.case}_z{z}.tif"
    with rasterio.open(out, "w", driver="GTiff", width=mosaic.shape[1], height=mosaic.shape[0], count=3,
                       dtype="uint8", crs="EPSG:3857", transform=transform, compress="deflate") as dst:
        dst.write(mosaic.transpose(2, 0, 1)); dst.update_tags(attribution=credit)
    print(f"downloaded {got} new tiles, wrote {out}")
    return out

if args.fetch:
    args.basemap = fetch_basemap(args.fetch, args.zoom)
if not args.basemap:
    sys.exit("give --basemap FILE.tif or --fetch {" + ",".join(sorted(SOURCES)) + "}")

# basemap reprojected onto the model square, row 0 = north
nb = int(round(L / args.res))
rgb = np.zeros((3, nb, nb), np.uint8)
with rasterio.open(args.basemap) as src:
    credit = args.attribution or src.tags().get("attribution", "")
    for b in range(3):
        reproject(source=src.read(b + 1), destination=rgb[b], src_transform=src.transform, src_crs=src.crs,
                  dst_transform=from_origin(x0, y0 + L, args.res, args.res), dst_crs=utm_crs,
                  resampling=Resampling.bilinear)
rgb = rgb.transpose(1, 2, 0)
if not rgb.any():
    sys.exit(f"{args.basemap} does not cover the model domain")

tz = None
if args.contours > 0 and os.path.exists(C["terrain"]):
    v = np.loadtxt(C["terrain"]); nt = int(v[0])
    tz = v[2 + 2 * nt:].reshape(nt, nt); xt = np.linspace(0, L / 1000, nt)

files = sorted(glob.glob("plt_fire_?????"))[::args.every]
if not files:
    sys.exit("no plt_fire_????? plotfiles here")
t_max_min = args.tmax if args.tmax > 0 else float(yt.load(files[-1]).current_time) / 60.0
ext = [0, L / 1000, 0, L / 1000]
os.makedirs("frames_map", exist_ok=True)
for old in glob.glob("frames_map/frame_*.png"):
    os.remove(old)
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

    fig, ax = plt.subplots(figsize=(7.6, 6.3))
    ax.imshow(rgb, origin="upper", extent=ext, interpolation="bilinear")
    if tz is not None:
        ax.contour(xt, xt, tz.T, levels=np.arange(args.contours, tz.max(), args.contours),
                   colors="w", linewidths=0.4, alpha=0.45)
    burned = np.ma.masked_where(at < 0, at / 60.0)
    im = ax.imshow(burned.T, cmap="hot", origin="lower", extent=ext, vmin=0, vmax=t_max_min, alpha=args.alpha)
    ax.contour(xc, xc, phi.T, levels=[0.0], colors="red", linewidths=1.6)
    s = max(1, nx // 48)
    ax.quiver(xc[::s], xc[::s], u[::s, ::s].T, w[::s, ::s].T, color="w", edgecolor="k", linewidth=0.3,
              scale=C["quiver_scale"], width=0.0028)
    for ig in ignitions:
        ax.plot(ig["x"] / 1000, ig["y"] / 1000, "o", mfc="none", mec="cyan", mew=1.4, ms=9)
    if crop:
        cx0, cx1, cy0, cy1 = [float(c) for c in crop.split(",")]
        ax.set_xlim(cx0, cx1); ax.set_ylim(cy0, cy1)
    ax.set_xlabel("x [km east of the domain's SW corner]"); ax.set_ylabel("y [km north]")
    n_b = int((at >= 0).sum()); ha = n_b * dx * dx / 1e4
    area = f"{ha:7.1f} ha ({ha * 2.4711:6.0f} acres)" if C["acres"] else f"{ha:7.1f} ha"
    ax.set_title(f"{C['title']}  t = {t / 60:6.1f} min   burned {area}")
    if credit:
        ax.text(0.99, 0.01, credit, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="w",
                bbox=dict(facecolor="k", alpha=0.45, pad=1.5, linewidth=0))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.solids.set_alpha(1); cb.set_label("arrival time [min]")
    fig.tight_layout()
    fn = f"frames_map/frame_{k:04d}.png"; fig.savefig(fn, dpi=args.dpi); plt.close(fig)
    frames.append(Image.open(fn).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    print(f"{pf}: t = {t:.0f} s, burned {n_b} cells")

frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
print(f"wrote {out_gif} with {len(frames)} frames")
if out_mp4:
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", "frames_map/frame_%04d.png",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_mp4],
                       check=True)
        print(f"wrote {out_mp4}")
    else:
        print("ffmpeg not on PATH: no mp4 written (the GIF is complete)")
