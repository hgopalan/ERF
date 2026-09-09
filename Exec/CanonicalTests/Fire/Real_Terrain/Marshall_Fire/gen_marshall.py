#!/usr/bin/env python3
"""Build the Marshall Fire case inputs from SRTM1 tiles.

Writes, in the current directory:
  terrain_marshall.txt     50 m raster (513 x 513 nodes) in ERF terrain format,
                           elevation above the domain floor, outer 1.5 km of
                           every edge blended to that edge's mean
  inflow_xlo.txt           z u v w for the west face, neutral log law, 20 m/s at
  inflow_yhi.txt           10 m above that face's mean ground, capped at 35 m/s,
                           from 285 degrees (a westerly with a small northerly
                           component); the north face likewise
  sounding_marshall.txt    ERF input_sounding: neutral to 1500 m, 3 K/km above
  ignitions_marshall.csv   erf.fire.ignition.schedule_file: two more starts on
                           terrain of opposite curvature to the primary one
  marshall_domain.json     corner coordinates, floor elevation, ignition sites
  marshall_terrain.png     map of the raster with the ignition sites

SRTM1 tiles come from the `elevation` package cache
(~/Library/Caches/elevation/SRTM1/cache/N39/N39W106.tif etc., downloaded with
elevation.clip); pass --tiles to point at another directory holding them.

    python3 gen_marshall.py [--tiles DIR]
"""
import argparse, glob, json, math, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tiles", default=os.path.expanduser("~/Library/Caches/elevation/SRTM1/cache"))
args = ap.parse_args()

try:
    import rasterio
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import uniform_filter, gaussian_filter
except ImportError as e:
    sys.exit(f"gen_marshall.py needs rasterio, pyproj and scipy: {e}")

# ---------------------------------------------------------------- domain
L        = 25600.0          # domain side [m]; 256 cells of 100 m, 1024 fire cells of 25 m
DXR      = 50.0             # raster spacing [m]
BLEND    = 1500.0           # edge blend width [m]
ORIGIN   = (39.953, -105.232)   # first Marshall Fire ignition, Eldorado Springs Dr at CO-93, 30 Dec 2021
X_ORIGIN = 5000.0           # the origin's distance from the west (inflow) edge [m]
Y_ORIGIN = 11000.0          # ... and from the south edge [m]

# UTM zone 13N; the domain is axis-aligned in UTM with the origin placed as above.
to_utm   = Transformer.from_crs("EPSG:4326", "EPSG:32613", always_xy=True)
from_utm = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
ox, oy = to_utm.transform(ORIGIN[1], ORIGIN[0])
x0, y0 = ox - X_ORIGIN, oy - Y_ORIGIN          # SW corner in UTM

# ---------------------------------------------------------------- tiles
tiles = sorted(glob.glob(os.path.join(args.tiles, "N39", "N39W10[56].tif")) +
               glob.glob(os.path.join(args.tiles, "N40", "N40W10[56].tif")))
if len(tiles) < 4:
    sys.exit(f"need the four SRTM1 tiles N39W106, N39W105, N40W106, N40W105 under {args.tiles}")
lon_edges, lat_edges, data = [], [], {}
for f in tiles:
    r = rasterio.open(f); a = r.read(1).astype(float)
    a[a < -1000] = np.nan
    b = r.bounds
    data[(round(b.left), round(b.bottom))] = (a, b, r.res)
# Mosaic on a common 1 arcsec grid: tiles overlap by one row/column.
res = 1.0 / 3600.0
lon_min, lat_min = -106.0, 39.0
nlon, nlat = 7201, 7201
mosaic = np.full((nlat, nlon), np.nan)
for (lx, ly), (a, b, _) in data.items():
    i0 = int(round((b.left + res / 2 - lon_min) / res))     # pixel centres
    j0 = int(round((lat_min - (b.bottom + res / 2)) / res))  # rows run north to south
    # row 0 of the tile is its northern edge
    jt = int(round((b.top - res / 2 - lat_min) / res))
    rows = np.arange(a.shape[0]); lat_rows = b.top - res / 2 - rows * res
    jj = np.round((lat_rows - lat_min) / res).astype(int)
    ii = i0 + np.arange(a.shape[1])
    ok = (jj >= 0) & (jj < nlat)
    mosaic[np.ix_(jj[ok], ii[(ii >= 0) & (ii < nlon)])] = a[ok][:, (ii >= 0) & (ii < nlon)]
lons = lon_min + np.arange(nlon) * res
lats = lat_min + np.arange(nlat) * res
if np.isnan(mosaic).any():
    # a few voids at tile seams: fill from the nearest valid row
    from scipy.ndimage import distance_transform_edt
    mask = np.isnan(mosaic)
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    mosaic = mosaic[tuple(idx)]

# Box-average the 30 m posts to about the raster spacing before sampling.
smooth = uniform_filter(mosaic, size=2, mode="nearest")
interp = RegularGridInterpolator((lats, lons), smooth, bounds_error=True)

# ---------------------------------------------------------------- raster
n = int(round(L / DXR)) + 1
xs = np.arange(n) * DXR
ys = np.arange(n) * DXR
X, Y = np.meshgrid(xs, ys, indexing="ij")
LON, LAT = from_utm.transform(x0 + X, y0 + Y)
Z = interp(np.stack([LAT.ravel(), LON.ravel()], axis=1)).reshape(n, n)
z_floor = float(np.floor(Z.min()))
Zrel = Z - z_floor
raw_relief = float(Zrel.max())

# Blend the outer BLEND metres of every edge to that edge's mean height with a
# raised cosine, so each face is flat where the inflow profile is applied and
# nothing steep sits on an outflow face.
def blend_weight(d):
    w = np.clip(d / BLEND, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(math.pi * w)     # 0 at the edge, 1 at BLEND
Zb = Zrel.copy()
for axis, edge in ((0, 0), (0, -1), (1, 0), (1, -1)):
    face = Zb[0, :] if (axis == 0 and edge == 0) else Zb[-1, :] if axis == 0 else Zb[:, 0] if edge == 0 else Zb[:, -1]
    zmean = float(face.mean())
    d = (X if axis == 0 else Y)
    d = d if edge == 0 else (L - d)
    w = blend_weight(d)
    Zb = w * Zb + (1.0 - w) * zmean
# Corners were blended twice along each axis; that is intended (both faces flat there).

# ---------------------------------------------------------------- ignitions
# Curvature of the raster (Laplacian of a 150 m Gaussian-smoothed field) picks
# terrain of opposite character within 4 km of the historical origin:
# strongly convex (a mesa rim or ridge crest) and strongly concave (a drainage).
Zs = gaussian_filter(Zb, sigma=3.0)
zxx = (np.roll(Zs, -1, 0) - 2 * Zs + np.roll(Zs, 1, 0)) / DXR**2
zyy = (np.roll(Zs, -1, 1) - 2 * Zs + np.roll(Zs, 1, 1)) / DXR**2
curv = zxx + zyy
sx = np.gradient(Zb, DXR, axis=0); sy = np.gradient(Zb, DXR, axis=1)
slope = np.hypot(sx, sy)
R_search, R_sep = 4000.0, 1500.0
near = np.hypot(X - X_ORIGIN, Y - Y_ORIGIN) < R_search
inner = (X > BLEND + 500) & (X < L - BLEND - 500) & (Y > BLEND + 500) & (Y < L - BLEND - 500)
sites = [("origin", X_ORIGIN, Y_ORIGIN)]
def pick(field, sign):
    cand = np.where(near & inner, sign * field, -np.inf)
    for (_, px, py) in sites:
        cand[np.hypot(X - px, Y - py) < R_sep] = -np.inf
    k = np.unravel_index(np.argmax(cand), cand.shape)
    return float(X[k]), float(Y[k])
cx, cy = pick(curv, -1.0); sites.append(("convex", cx, cy))
cx, cy = pick(curv, +1.0); sites.append(("concave", cx, cy))
def at(field, x, y):
    return float(field[int(round(x / DXR)), int(round(y / DXR))])

# ---------------------------------------------------------------- write
with open("terrain_marshall.txt", "w") as f:
    f.write(f"{n}\n{n}\n")
    for x in xs: f.write(f"{x:.2f}\n")
    for y in ys: f.write(f"{y:.2f}\n")
    for i in range(n):
        f.write("\n".join(f"{v:.3f}" for v in Zb[i, :]) + "\n")

# Inflow: neutral log law over z0 = 0.1 m anchored at 20 m/s at 10 m above
# ground, held at 35 m/s above the height where the law reaches it; from 285
# degrees. ERF reads the file's heights as absolute (above the domain floor),
# so each inflow face gets its own file with the profile anchored at that
# face's mean ground: the west face on the foothill shelf, the north face on
# the plain.
z0, u10, ucap = 0.1, 20.0, 35.0
ustar = 0.4 * u10 / math.log(10.0 / z0)
zin = [0.9, 2.0, 4.0, 6.1, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0,
       300.0, 400.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0]
wdir = math.radians(285.0)
def uv(z):
    s = min(ucap, ustar / 0.4 * math.log(max(z, z0 * 1.001) / z0))
    return -s * math.sin(wdir), -s * math.cos(wdir)     # meteorological "from" direction
z_west  = float(Zb[0, :].mean())
z_north = float(Zb[:, -1].mean())
for fname, zg in (("inflow_xlo.txt", z_west), ("inflow_yhi.txt", z_north)):
    with open(fname, "w") as f:
        if zg > 0.0:
            f.write(f"{0.0:12.4f} {0.0:11.5f} {0.0:11.5f} {0.0:6.1f}\n")
            f.write(f"{zg:12.4f} {0.0:11.5f} {0.0:11.5f} {0.0:6.1f}\n")
        for z in zin:
            u, v = uv(z); f.write(f"{zg + z:12.4f} {u:11.5f} {v:11.5f} {0.0:6.1f}\n")
with open("sounding_marshall.txt", "w") as f:
    f.write("84000.0 300.0 0.0\n")          # surface pressure [Pa] at 1600 m ASL, theta, qv
    for z in zin:
        u, v = uv(z)
        th = 300.0 + max(0.0, z - 1500.0) * 0.003
        f.write(f"{z:12.4f} {th:9.2f} {0.0:6.2f} {u:11.5f} {v:11.5f}\n")
    # the sounding is the interior initial state; heights are above the floor
with open("ignitions_marshall.csv", "w") as f:
    f.write("# time_s cx cy radius   (erf.fire.ignition.schedule_file)\n")
    f.write(f"{300.0:8.1f} {sites[1][1]:10.1f} {sites[1][2]:10.1f} {100.0:6.1f}\n")
    f.write(f"{600.0:8.1f} {sites[2][1]:10.1f} {sites[2][2]:10.1f} {100.0:6.1f}\n")

sw = from_utm.transform(x0, y0); ne = from_utm.transform(x0 + L, y0 + L)
info = {
    "domain_m": L, "raster_dx_m": DXR, "nodes": n, "utm_zone": "13N",
    "sw_corner_lonlat": sw, "ne_corner_lonlat": ne,
    "floor_elevation_m_asl": z_floor, "relief_m": float(Zb.max()), "relief_before_blend_m": raw_relief,
    "max_slope": float(slope[inner].max()), "west_face_ground_m": z_west, "north_face_ground_m": z_north,
    "ignitions": [{"name": nm, "x": x, "y": y, "lonlat": from_utm.transform(x0 + x, y0 + y),
                   "z_m": at(Zb, x, y), "curvature_1_per_m": at(curv, x, y), "slope": at(slope, x, y)}
                  for nm, x, y in sites],
}
json.dump(info, open("marshall_domain.json", "w"), indent=1)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    ls = matplotlib.colors.LightSource(azdeg=315, altdeg=45)
    ax.imshow(ls.shade(Zb.T, cmap=plt.cm.terrain, vert_exag=2, dx=DXR, dy=DXR), origin="lower",
              extent=[0, L / 1000, 0, L / 1000])
    for nm, x, y in sites:
        ax.plot(x / 1000, y / 1000, "o", mfc="none", mec="red", mew=2, ms=10)
        ax.annotate(nm, (x / 1000, y / 1000), xytext=(6, 6), textcoords="offset points", color="red")
    ax.set_xlabel("x [km] (wind from the west)"); ax.set_ylabel("y [km]")
    ax.set_title(f"Marshall Fire terrain, floor {z_floor:.0f} m ASL, relief {Zb.max():.0f} m")
    fig.tight_layout(); fig.savefig("marshall_terrain.png", dpi=110)
except Exception as e:
    print("no map:", e)

print(json.dumps(info, indent=1))
