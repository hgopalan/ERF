#!/usr/bin/env python3
"""Put a LANDFIRE fuel raster on a real-terrain case's fire grid, as the ESRI
ASCII fuel map erf.fire.fuel_map.file reads.

    cd <case directory>      # holds <case>_domain.json
    python3 <ERF>/Exec/Tools/make_landfire_fuel_map.py --fetch LF2016 --dx 25      # download the window once, then map it
    python3 <ERF>/Exec/Tools/make_landfire_fuel_map.py --raster FBFM40.tif --dx 25 # a raster you already have, any CRS

The fire grid is the case's square (domain_m on a side, south-west corner at
sw_corner_lonlat, axis-aligned in the UTM zone of the domain json) in cells of
--dx metres, the atmosphere cell divided by erf.fire.grid_ratio: 25 m for the
committed Marshall deck. ERF places the map by cell index, so the map carries
exactly one code per fire cell; its rows run from the north edge down, as ESRI
ASCII grids do, and the header's corner is written as 0 0 and not used.

--fetch downloads one window of the CONUS raster from the USGS LANDFIRE Product
Service image server (lfps.usgs.gov, public, no account) covering the domain
plus --margin, on LANDFIRE's own 30 m Albers grid (EPSG:5070), and saves it as
<edition>_<layer>_<case>.tif, so a rerun downloads nothing. The service path
Landfire_<edition>/<edition>_<layer>_CONUS/ImageServer was checked for LF2016
(LANDFIRE 2.0.0) FBFM40 on 2026-09-11; the editions and their service names
are listed at https://lfps.usgs.gov/arcgis/rest/services, and --service takes
one that is named differently. Behind a TLS-inspecting proxy whose root is only
in the system keychain, Python rejects the certificate; the fetch then uses
curl, which checks against the system trust store (verification stays on
either way).

Categorical codes are resampled nearest-neighbour. Fire cells the raster does
not cover get NODATA_value, which the reader turns into code 0. The script
prints the share of each code and the deck lines that use the map.
"""
import argparse, glob, json, math, os, shutil, ssl, subprocess, sys, urllib.error, urllib.request
import numpy as np
try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling, transform_bounds
    from pyproj import Transformer
except ImportError as e:
    sys.exit(f"make_landfire_fuel_map.py needs rasterio and pyproj: {e}")

LFPS = "https://lfps.usgs.gov/arcgis/rest/services"
LF_ORIGIN = (-2362425.0, 3177435.0)   # upper-left corner of LANDFIRE's CONUS 30 m grid [EPSG:5070 m]
LF_CELL = 30.0
NODATA = -9999

found = sorted(glob.glob("*_domain.json"))
ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--domain", default=found[0] if len(found) == 1 else "",
                help="<case>_domain.json; default the one in this directory")
ap.add_argument("--dx", type=float, required=True, help="fire cell size [m]")
src_group = ap.add_mutually_exclusive_group(required=True)
src_group.add_argument("--raster", help="fuel GeoTIFF to map (any CRS)")
src_group.add_argument("--fetch", metavar="EDITION", help="download the window from LANDFIRE first, e.g. LF2016")
ap.add_argument("--layer", default="FBFM40", choices=["FBFM40", "FBFM13"], help="LANDFIRE layer for --fetch")
ap.add_argument("--service", default="", help="image server under arcgis/rest/services for --fetch, "
                "if not Landfire_<edition>/<edition>_<layer>_CONUS")
ap.add_argument("--margin", type=float, default=500.0, help="window margin around the domain for --fetch [m]")
ap.add_argument("--out", default="", help="output map; default fuel_<case>_<layer>.asc")
ap.add_argument("--png", action="store_true", help="also write a quick-look image (needs matplotlib)")
ap.add_argument("--dry-run", action="store_true", help="with --fetch, print the request and stop")
args = ap.parse_args()
if not args.domain:
    sys.exit("found no single *_domain.json here; pass --domain")

case = os.path.basename(args.domain).split("_domain.json")[0]
layer = args.layer
info = json.load(open(args.domain))
L = float(info["domain_m"])
nx = int(round(L / args.dx))
if abs(nx * args.dx - L) > 1e-6 * L:
    sys.exit(f"--dx {args.dx} m does not divide the {L:g} m domain")
zone = info["utm_zone"]
utm = f"EPSG:{(32600 if zone[-1].upper() == 'N' else 32700) + int(zone[:-1])}"
x0, y0 = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform(*info["sw_corner_lonlat"])


def fetch_window(edition, out):
    """One exportImage request for the domain window on LANDFIRE's 30 m grid."""
    m = args.margin
    b = transform_bounds(utm, "EPSG:5070", x0 - m, y0 - m, x0 + L + m, y0 + L + m)
    ox, oy = LF_ORIGIN
    xmin = ox + math.floor((b[0] - ox) / LF_CELL) * LF_CELL
    xmax = ox + math.ceil((b[2] - ox) / LF_CELL) * LF_CELL
    ymax = oy - math.floor((oy - b[3]) / LF_CELL) * LF_CELL
    ymin = oy - math.ceil((oy - b[1]) / LF_CELL) * LF_CELL
    w, h = round((xmax - xmin) / LF_CELL), round((ymax - ymin) / LF_CELL)
    service = args.service or f"Landfire_{edition}/{edition}_{layer}_CONUS"
    url = (f"{LFPS}/{service}/ImageServer/exportImage?bbox={xmin:.0f},{ymin:.0f},{xmax:.0f},{ymax:.0f}"
           f"&bboxSR=5070&imageSR=5070&size={w},{h}&format=tiff&pixelType=S16&noData={NODATA}"
           f"&interpolation=RSP_NearestNeighbor&f=image")
    print(f"LANDFIRE window {w} x {h} cells of 30 m: {url}")
    if args.dry_run:
        sys.exit(0)
    part = out + ".part"
    try:
        with urllib.request.urlopen(url, timeout=300) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.URLError as e:
        if not isinstance(e.reason, ssl.SSLError) or not shutil.which("curl"):
            raise
        print("Python rejected the server certificate; fetching with curl (system trust store)")
        subprocess.run(["curl", "-fsS", "--max-time", "300", "-o", part, url], check=True)
    try:
        rasterio.open(part).close()
    except rasterio.errors.RasterioIOError:
        sys.exit(f"the service did not return a GeoTIFF (check the edition and --service): "
                 f"{open(part, 'rb').read(300)!r}")
    os.replace(part, out)
    print(f"saved {out}")


raster = args.raster
if args.fetch:
    raster = f"{args.fetch}_{layer}_{case}.tif"
    if os.path.exists(raster):
        print(f"using the window already downloaded: {raster}")
    else:
        fetch_window(args.fetch, raster)

# Fire cell centres of row 0 lie at the north edge: from_origin takes the
# north-west corner, so dst[0, :] is the first row the ESRI file wants.
dst = np.full((nx, nx), NODATA, np.int16)
with rasterio.open(raster) as src:
    reproject(source=rasterio.band(src, 1), destination=dst,
              src_transform=src.transform, src_crs=src.crs,
              src_nodata=src.nodata if src.nodata is not None else NODATA,
              dst_transform=from_origin(x0, y0 + L, args.dx, args.dx), dst_crs=utm,
              dst_nodata=NODATA, resampling=Resampling.nearest)

out = args.out or f"fuel_{case}_{layer.lower()}.asc"
with open(out, "w") as f:
    f.write(f"ncols {nx}\nnrows {nx}\nxllcorner 0.0\nyllcorner 0.0\ncellsize {args.dx:g}\nNODATA_value {NODATA}\n")
    np.savetxt(f, dst, fmt="%d")   # north row first


def code_name(c):
    """Short name of a fuel code: Anderson FM1-13, Scott and Burgan (2005) GR1 ... NB9."""
    if 1 <= c <= 13:
        return f"FM{c}"
    for lo, hi, group in ((91, 99, "NB"), (101, 109, "GR"), (121, 124, "GS"), (141, 149, "SH"),
                          (161, 165, "TU"), (181, 189, "TL"), (201, 204, "SB")):
        if lo <= c <= hi:
            return f"{group}{c - lo + 1}"
    return "no data" if c == NODATA else ""


codes, counts = np.unique(dst, return_counts=True)
print(f"wrote {out}: {nx} x {nx} fire cells of {args.dx:g} m from {raster}")
for c, n in sorted(zip(codes.tolist(), counts.tolist()), key=lambda t: -t[1]):
    print(f"  {c:6d} {code_name(c):8s} {100.0 * n / dst.size:6.2f} %")
print("\nDeck lines:")
print(f'erf.fire.fuel_map.file = "{out}"')
print('erf.fire.fuel_map.format = "ascii"')
if layer == "FBFM40":
    print('erf.fire.fuel_map.fuel_set = "scott_burgan40"   # the 40 codes natively; 91-99 and 0 are non-burnable')
else:
    print("erf.fire.fuel_map.nonburnable_codes = 0 91 92 93 98 99")
print("erf.fire.fuel_map.load_from_map = true           # each cell starts with its own model's load")
print("erf.fire.rothermel_per_fuel = true               # per-cell Rothermel coefficients")

if args.png:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.ma.masked_equal(dst, NODATA), origin="upper", extent=[0, L / 1000, 0, L / 1000],
                   cmap="tab20", interpolation="nearest")
    fig.colorbar(im, ax=ax, label=f"{layer} code")
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
    ax.set_title(f"{os.path.basename(raster)} on the {case} fire grid ({args.dx:g} m)")
    fig.savefig(out.replace(".asc", ".png"), dpi=110)
    print(f"wrote {out.replace('.asc', '.png')}")
