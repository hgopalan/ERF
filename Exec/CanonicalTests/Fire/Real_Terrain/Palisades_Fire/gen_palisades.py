#!/usr/bin/env python3
"""Build the Palisades Fire case inputs from SRTM1 tiles.

The case reproduces the NLR Palisades stress-test demo: a 24 km square of the
Santa Monica Mountains and the Pacific Palisades coast, a 300 m atmosphere over
a 30 m fire and terrain grid, a simplified stratified atmosphere with a
southwesterly at 38 mph, and a two-way coupled Rothermel fire started at the
reported origin above Temescal Canyon.

Writes, in the current directory:
  terrain_palisades.txt    30 m raster (801 x 801 nodes) in ERF terrain format,
                           elevation above the domain floor (sea level), with
                           the outer 1.5 km of every edge blended to that edge's
                           mean so the inflow faces are flat
  fuel_palisades.asc       ESRI ASCII fuel map on the 800 x 800 fire grid:
                           Anderson model 4 (chaparral) on land, 0 (the
                           non-burnable code) over the Pacific and the beach
  inflow_xlo.txt           z u v w for the west face, neutral log law over
  inflow_ylo.txt           z0 = 0.1 m anchored at 38 mph (16.99 m/s) 10 m above
                           that face's mean ground, from 225 degrees; likewise
                           the south face
  sounding_palisades.txt   ERF input_sounding: neutral to 1500 m, 3 K/km above
  palisades_domain.json    corner coordinates, floor elevation, ignition site,
                           sea fraction, cell counts
  palisades_terrain.png    map of the raster with the ignition site and coast

SRTM1 tiles come from the `elevation` package cache
(~/Library/Caches/elevation/SRTM1/cache/N34/N34W119.tif); pass --tiles to point
at another directory holding them. SRTM1 is a 30 m product: the demo used a
10 m DEM, so pass --dxr with a finer raster only if you substitute one.

    python3 gen_palisades.py [--tiles DIR] [--dxr 30.0]
"""
import argparse, glob, json, math, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tiles", default=os.path.expanduser("~/Library/Caches/elevation/SRTM1/cache"))
ap.add_argument("--dxr", type=float, default=30.0, help="terrain raster spacing [m]")
args = ap.parse_args()

try:
    import rasterio
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import uniform_filter
except ImportError as e:
    sys.exit(f"gen_palisades.py needs rasterio, pyproj and scipy: {e}")

# ---------------------------------------------------------------- domain
L        = 24000.0          # domain side [m]; 80 cells of 300 m, 800 fire cells of 30 m
DXR      = args.dxr         # raster spacing [m]
DXF      = 30.0             # fire cell size [m]; grid_ratio 10 under the 300 m atmosphere
BLEND    = 1500.0           # edge blend width [m]
SEA_M    = 1.0              # elevation at or below this is sea/beach: non-burnable
# Reported origin of the 7 Jan 2025 Palisades Fire, the hillside above Temescal
# Canyon near the Skull Rock trailhead. Approximate: the published coordinates
# vary by a few hundred metres between sources.
ORIGIN   = (34.0797, -118.5265)
X_ORIGIN = 9000.0           # the origin's distance from the west (inflow) edge [m]
Y_ORIGIN = 8000.0           # ... and from the south (inflow) edge [m]
WDIR_DEG = 225.0            # southwesterly, the demo's wind
U10_MPH  = 38.0

# UTM zone 11N; the domain is axis-aligned in UTM with the origin placed as above.
to_utm   = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)
from_utm = Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True)
ox, oy = to_utm.transform(ORIGIN[1], ORIGIN[0])
x0, y0 = ox - X_ORIGIN, oy - Y_ORIGIN          # SW corner in UTM

# ---------------------------------------------------------------- tiles
# The 24 km box around the origin sits inside N34W119, but take whatever of the
# 2 x 2 neighbourhood is present so a shifted origin still works.
want = [(34, 119), (34, 118), (35, 119), (35, 118)]
tiles = []
for la, lo in want:
    p = os.path.join(args.tiles, f"N{la:02d}", f"N{la:02d}W{lo:03d}.tif")
    if os.path.exists(p):
        tiles.append(p)
if not tiles:
    sys.exit(f"need SRTM1 tile N34W119 under {args.tiles}\n"
             f"  python3 -c \"import elevation; elevation.clip(bounds=(-119,34,-118,35), output='/tmp/pal.tif')\"")

res = 1.0 / 3600.0
lat_min, lon_min = 34.0, -119.0
nlat = nlon = 3601 + 3600 * (len({t for t in want if os.path.exists(
    os.path.join(args.tiles, f"N{t[0]:02d}", f"N{t[0]:02d}W{t[1]:03d}.tif"))}) > 2)
# Simple mosaic on a 1 arcsec grid spanning the available tiles.
lat_hi = 35.0 if any("N35" in t for t in tiles) else 35.0
lon_hi = -118.0 if any("W118" in t for t in tiles) else -118.0
nlon = int(round((lon_hi - lon_min) / res)) + 1
nlat = int(round((lat_hi - lat_min) / res)) + 1
mosaic = np.full((nlat, nlon), np.nan)
for f in tiles:
    r = rasterio.open(f)
    a = r.read(1).astype(float)
    a[a < -1000] = np.nan            # SRTM void
    b = r.bounds
    rows = np.arange(a.shape[0])
    lat_rows = b.top - res / 2 - rows * res
    jj = np.round((lat_rows - lat_min) / res).astype(int)
    cols = np.arange(a.shape[1])
    lon_cols = b.left + res / 2 + cols * res
    ii = np.round((lon_cols - lon_min) / res).astype(int)
    okj = (jj >= 0) & (jj < nlat)
    oki = (ii >= 0) & (ii < nlon)
    mosaic[np.ix_(jj[okj], ii[oki])] = a[np.ix_(okj, oki)]
# SRTM1 reports the ocean as 0; voids over water become 0 too.
mosaic = np.nan_to_num(mosaic, nan=0.0)

lons = lon_min + np.arange(nlon) * res
lats = lat_min + np.arange(nlat) * res
smooth = uniform_filter(mosaic, size=2, mode="nearest")
interp = RegularGridInterpolator((lats, lons), smooth, bounds_error=True)

# ---------------------------------------------------------------- raster
n = int(round(L / DXR)) + 1
xs = np.arange(n) * DXR
ys = np.arange(n) * DXR
X, Y = np.meshgrid(xs, ys, indexing="ij")
LON, LAT = from_utm.transform(x0 + X, y0 + Y)
Z = interp(np.stack([LAT.ravel(), LON.ravel()], axis=1)).reshape(n, n)
# SRTM1 carries a few metres of noise below datum over Santa Monica Bay; the
# domain floor is mean sea level, so clamp there rather than at the raw minimum
# (which would lift the whole ocean tens of metres above the floor and hide it
# from the sea test below).
Z = np.maximum(Z, 0.0)
z_floor = 0.0                                 # sea level
Zrel = Z - z_floor
raw_relief = float(Zrel.max())
sea_node = Zrel <= SEA_M

# Blend the outer BLEND metres of every edge to that edge's mean height with a
# raised cosine, so each face is flat where the inflow profile is applied and
# nothing steep sits on an outflow face.
def blend_weight(d):
    w = np.clip(d / BLEND, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(math.pi * w)     # 0 at the edge, 1 at BLEND
Zb = Zrel.copy()
for axis, edge in ((0, 0), (0, -1), (1, 0), (1, -1)):
    face = Zb[0, :] if (axis == 0 and edge == 0) else Zb[-1, :] if axis == 0 else Zb[:, 0] if edge == 0 else Zb[:, -1]
    # The median, not the mean: the south and west faces here are part ocean and
    # part coastal hillside, and a mean would lift the sea tens of metres above
    # the floor and drop the hillside onto it. The median follows whichever of
    # the two covers most of the face, so the sea stays at sea level.
    zmean = float(np.median(face))
    d = (X if axis == 0 else Y)
    d = d if edge == 0 else (L - d)
    w = blend_weight(d)
    Zb = w * Zb + (1.0 - w) * zmean

sx = np.gradient(Zb, DXR, axis=0); sy = np.gradient(Zb, DXR, axis=1)
slope = np.hypot(sx, sy)

# The fitted mesh is sampled at the 300 m atmospheric nodes and cannot carry the
# 30 m relief: at full resolution the canyon walls fold the near-ground cells
# over each other and the surface layer cannot find its query height. Box-average
# the raster over one atmospheric cell for the mesh, and keep the sharp field for
# the fire, which needs the real slope for the rate of spread.
DXA = DXF * 10                                  # atmospheric cell [m]
Za = uniform_filter(Zb, size=int(round(DXA / DXR)), mode="nearest")
sxa = np.gradient(Za, DXR, axis=0); sya = np.gradient(Za, DXR, axis=1)
slope_atm = np.hypot(sxa, sya)

# ---------------------------------------------------------------- write terrain
def write_raster(fname, field):
    with open(fname, "w") as f:
        f.write(f"{n}\n{n}\n")
        for x in xs: f.write(f"{x:.2f}\n")
        for y in ys: f.write(f"{y:.2f}\n")
        for i in range(n):
            f.write("\n".join(f"{v:.3f}" for v in field[i, :]) + "\n")
write_raster("terrain_palisades.txt", Za)        # atmosphere: fitted mesh
write_raster("terrain_palisades_fire.txt", Zb)   # fire: slope and rate of spread

# ---------------------------------------------------------------- fuel map
# Cell-centred on the fire grid: Anderson 4 (chaparral) on land, 0 over the
# Pacific and the beach. 0 is listed in erf.fire.fuel_map.nonburnable_codes,
# so the front stops at the shoreline instead of running out to sea.
nf = int(round(L / DXF))
xc = (np.arange(nf) + 0.5) * DXF
yc = (np.arange(nf) + 0.5) * DXF
XC, YC = np.meshgrid(xc, yc, indexing="ij")
LONC, LATC = from_utm.transform(x0 + XC, y0 + YC)
Zc = np.maximum(interp(np.stack([LATC.ravel(), LONC.ravel()], axis=1)).reshape(nf, nf), 0.0) - z_floor
code = np.where(Zc <= SEA_M, 0, 4).astype(int)
sea_fraction = float((code == 0).mean())
with open("fuel_palisades.asc", "w") as f:
    f.write(f"ncols {nf}\nnrows {nf}\nxllcorner 0.0\nyllcorner 0.0\n"
            f"cellsize {DXF}\nNODATA_value -9999\n")
    # ESRI ASCII rows run north first
    for j in range(nf - 1, -1, -1):
        f.write(" ".join(str(int(v)) for v in code[:, j]) + "\n")

# ---------------------------------------------------------------- inflow
# Neutral log law over z0 = 0.1 m anchored at the demo's 38 mph 10 m above
# ground, capped where the law reaches 1.75x that, from 225 degrees. ERF reads
# the file's heights as absolute (above the domain floor), so each inflow face
# gets its own file with the profile anchored at that face's mean ground: here
# the west and south faces are largely ocean, so both sit near sea level.
z0 = 0.1
u10 = U10_MPH * 0.44704
ucap = 1.75 * u10
ustar = 0.4 * u10 / math.log(10.0 / z0)
zin = [0.9, 2.0, 4.0, 6.1, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0,
       300.0, 400.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6500.0]
wdir = math.radians(WDIR_DEG)
def uv(z):
    s = min(ucap, ustar / 0.4 * math.log(max(z, z0 * 1.001) / z0))
    return -s * math.sin(wdir), -s * math.cos(wdir)     # meteorological "from" direction
z_west  = float(Zb[0, :].mean())
z_south = float(Zb[:, 0].mean())
for fname, zg in (("inflow_xlo.txt", z_west), ("inflow_ylo.txt", z_south)):
    with open(fname, "w") as f:
        if zg > 0.0:
            f.write(f"{0.0:12.4f} {0.0:11.5f} {0.0:11.5f} {0.0:6.1f}\n")
            f.write(f"{zg:12.4f} {0.0:11.5f} {0.0:11.5f} {0.0:6.1f}\n")
        for z in zin:
            u, v = uv(z); f.write(f"{zg + z:12.4f} {u:11.5f} {v:11.5f} {0.0:6.1f}\n")

with open("sounding_palisades.txt", "w") as f:
    f.write("101325.0 300.0 0.0\n")        # surface pressure [Pa] at sea level, theta, qv
    for z in zin:
        u, v = uv(z)
        th = 300.0 + max(0.0, z - 1500.0) * 0.003
        f.write(f"{z:12.4f} {th:9.2f} {0.0:6.2f} {u:11.5f} {v:11.5f}\n")

# ---------------------------------------------------------------- report
def at(field, x, y, d=DXR):
    return float(field[int(round(x / d)), int(round(y / d))])
sw = from_utm.transform(x0, y0); ne = from_utm.transform(x0 + L, y0 + L)
nx_atm = int(round(L / (DXF * 10)))
info = {
    "domain_m": L, "raster_dx_m": DXR, "nodes": n,
    "fire_dx_m": DXF, "fire_cells": nf * nf,
    "atm_dx_m": DXF * 10, "atm_cells_xy": nx_atm * nx_atm,
    "utm_zone": "11N",
    "sw_corner_lonlat": sw, "ne_corner_lonlat": ne,
    "floor_elevation_m_asl": z_floor,
    "relief_m": float(Zb.max()), "relief_before_blend_m": raw_relief,
    "max_slope_fire": float(slope.max()), "max_slope_atm": float(slope_atm.max()),
    "sea_fraction_of_fire_grid": sea_fraction,
    "west_face_ground_m": z_west, "south_face_ground_m": z_south,
    "u10_ms": u10, "wind_from_deg": WDIR_DEG,
    "ignition": {"x": X_ORIGIN, "y": Y_ORIGIN, "lonlat": list(ORIGIN[::-1]),
                 "z_m": at(Zb, X_ORIGIN, Y_ORIGIN),
                 "slope": at(slope, X_ORIGIN, Y_ORIGIN)},
}
json.dump(info, open("palisades_domain.json", "w"), indent=1)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6.4))
    ls = matplotlib.colors.LightSource(azdeg=315, altdeg=45)
    ax.imshow(ls.shade(Zb.T, cmap=plt.cm.terrain, vert_exag=2, dx=DXR, dy=DXR), origin="lower",
              extent=[0, L / 1000, 0, L / 1000])
    ax.contour(xc / 1000, yc / 1000, (code == 0).T.astype(float), levels=[0.5],
               colors="#0044aa", linewidths=1.0)
    ax.plot(X_ORIGIN / 1000, Y_ORIGIN / 1000, "*", mfc="red", mec="k", mew=0.6, ms=17)
    ax.annotate("ignition", (X_ORIGIN / 1000, Y_ORIGIN / 1000), xytext=(8, 6),
                textcoords="offset points", color="red")
    ax.annotate("", xy=(4.2, 4.2), xytext=(1.6, 1.6),
                arrowprops=dict(arrowstyle="-|>", color="#0066ff", lw=2))
    ax.annotate(f"SW {U10_MPH:.0f} mph", (2.0, 4.4), color="#0066ff", fontsize=9)
    ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]")
    ax.set_title(f"Palisades Fire terrain, floor {z_floor:.0f} m ASL, "
                 f"relief {Zb.max():.0f} m, {100*sea_fraction:.0f}% sea")
    fig.tight_layout(); fig.savefig("palisades_terrain.png", dpi=110)
except Exception as e:
    print("no map:", e)

print(json.dumps(info, indent=1))
