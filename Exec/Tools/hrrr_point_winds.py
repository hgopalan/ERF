#!/usr/bin/env python3
"""HRRR analysis winds at points: speed and meteorological direction (degrees the wind blows FROM).

    cd <case directory>      # holds <case>_domain.json
    python3 <ERF>/Exec/Tools/hrrr_point_winds.py --date 2021-12-30 --hours 17 18 19 20 --domain marshall_domain.json \\
        --point "Costco, Superior" -105.1745 39.9557 --csv hrrr_winds.csv

Fetches only the needed GRIB messages of the HRRR surface analysis (product sfc, fxx 0) with Herbie from
NOAA's public archive (source priority --priority, default aws), cached under --save-dir, so a rerun downloads
nothing. Fields: UGRD/VGRD at 10 m and 80 m above ground, surface GUST and HPBL.

HRRR stores U and V relative to its Lambert conformal grid (GRIB flag uvRelativeToGrid = 1). They are rotated to
earth-relative components here: with the grid's central meridian LoV and tangent latitude Latin1, the angle is
alpha = sin(Latin1) * (lon - LoV), u_e = u_g cos(alpha) + v_g sin(alpha), v_e = -u_g sin(alpha) + v_g cos(alpha).
At 105.2 W that is about -4.8 degrees; skipping it biases the direction by that much.

With --domain <case>_domain.json (an ERF-Hazard real-terrain case: square of domain_m, south-west corner at
sw_corner_lonlat, axis-aligned in the UTM zone), the default points are: the west face at the first ignition's
y and at mid-face, 5 km upstream of the west face, the ignition(s) and the domain centre. --point adds more.
Values are nearest-grid-point (HRRR is 3 km).

Marshall, 30 Dec 2021, west face at the ignition y: 18 UTC 16.7 m/s at 10 m from 267 deg (29.4 m/s at 80 m from
268 deg), gust 37 m/s, HPBL 650 m; 19 UTC 17.5 m/s from 261 deg. The idea follows the HRRR-driven 1D column set-up
of github.com/hgopalan/onedterrainsolver (Herbie, UGRD/VGRD at 80 m above ground).
"""
import argparse, json, math, os, sys, warnings
import numpy as np
import pandas as pd
try:
    from herbie import Herbie
    from pyproj import Transformer
except ImportError as e:
    sys.exit(f"hrrr_point_winds.py needs herbie-data, cfgrib/xarray and pyproj: {e}")

ap = argparse.ArgumentParser()
ap.add_argument("--date", required=True, help="UTC date, YYYY-MM-DD")
ap.add_argument("--hours", type=int, nargs="+", required=True, help="UTC analysis hours")
ap.add_argument("--domain", default="", help="<case>_domain.json for the default points")
ap.add_argument("--point", nargs=3, action="append", default=[], metavar=("NAME", "LON", "LAT"))
ap.add_argument("--save-dir", default=os.path.expanduser("~/data"))
ap.add_argument("--priority", nargs="+", default=["aws"])
ap.add_argument("--csv", default="hrrr_point_winds.csv")
args = ap.parse_args()

points = []
if args.domain:
    info = json.load(open(args.domain)); L = float(info["domain_m"]); zone = info["utm_zone"]
    utm = f"EPSG:{(32600 if zone.upper().endswith('N') else 32700) + int(zone[:-1])}"
    fwd = Transformer.from_crs("EPSG:4326", utm, always_xy=True); inv = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    x0, y0 = fwd.transform(*info["sw_corner_lonlat"])
    ign = (info.get("ignitions") or [info.get("ignition")])[0]
    for name, x, y in [("west face at ignition y", 0.0, ign["y"]), ("west face mid", 0.0, L / 2),
                       ("5 km upstream of west face", -5000.0, ign["y"]), (f"ignition ({ign['name']})", ign["x"], ign["y"]),
                       ("domain centre", L / 2, L / 2)]:
        lon, lat = inv.transform(x0 + x, y0 + y); points.append((name, lon, lat, x, y))
for name, lon, lat in args.point:
    lon, lat = float(lon), float(lat)
    if args.domain:
        x, y = fwd.transform(lon, lat); points.append((name, lon, lat, x - x0, y - y0))
    else:
        points.append((name, lon, lat, np.nan, np.nan))
if not points:
    sys.exit("give --domain and/or --point")

SEARCH = ":(?:UGRD|VGRD):(?:10|80) m above ground|:GUST:surface|:HPBL:surface"
rows = []
for hour in args.hours:
    when = pd.Timestamp(f"{args.date} {hour:02d}:00")
    H = Herbie(when, model="hrrr", product="sfc", fxx=0, save_dir=args.save_dir, priority=args.priority, verbose=False)
    H.download(SEARCH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dsl = H.xarray(SEARCH, remove_grib=False)
    dsl = dsl if isinstance(dsl, list) else [dsl]
    lat2 = lon2 = None; fields = {}; gridrel = {}
    lov = latin1 = None
    for ds in dsl:
        for v in ds.data_vars:
            da = ds[v]
            if lat2 is None:
                lat2 = ds["latitude"].values; lon2 = ((ds["longitude"].values + 180.0) % 360.0) - 180.0
            if v in ("gust", "blh", "hpbl") or v[-1].isdigit() or "heightAboveGround" not in ds.coords:
                key = v                                  # cfgrib names the 10 m winds u10/v10 already
            else:
                key = f"{v}{int(ds['heightAboveGround'].values)}"   # u/v at 80 m -> u80/v80
            fields[key] = da.values
            gridrel[key] = int(da.attrs.get("GRIB_uvRelativeToGrid", 0))
            lov = da.attrs.get("GRIB_LoVInDegrees", lov); latin1 = da.attrs.get("GRIB_Latin1InDegrees", latin1)
    lov = -97.5 if lov is None else (((float(lov) + 180.0) % 360.0) - 180.0)
    latin1 = 38.5 if latin1 is None else float(latin1)
    for name, lon, lat, x, y in points:
        d2 = (lat2 - lat) ** 2 + ((lon2 - lon) * math.cos(math.radians(lat))) ** 2
        j, i = np.unravel_index(np.argmin(d2), d2.shape)
        row = dict(time_utc=str(when), point=name, lon=lon, lat=lat, x_m=x, y_m=y,
                   grid_lon=float(lon2[j, i]), grid_lat=float(lat2[j, i]))
        alpha = math.radians(math.sin(math.radians(latin1)) * (lon2[j, i] - lov))
        for hgt in (10, 80):
            ug, vg = fields.get(f"u{hgt}"), fields.get(f"v{hgt}")
            if ug is None or vg is None:
                continue
            u, v = float(ug[j, i]), float(vg[j, i])
            if gridrel.get(f"u{hgt}", 1):
                u, v = u * math.cos(alpha) + v * math.sin(alpha), -u * math.sin(alpha) + v * math.cos(alpha)
            row[f"u{hgt}_east"] = u; row[f"v{hgt}_north"] = v
            row[f"speed{hgt}"] = math.hypot(u, v); row[f"dir{hgt}_from"] = (math.degrees(math.atan2(-u, -v)) % 360.0)
        for k in ("gust", "blh", "hpbl"):
            if k in fields:
                row[k] = float(fields[k][j, i])
        rows.append(row)
    print(f"{when}: fields {sorted(fields)}; uvRelativeToGrid {sorted(set(gridrel.values()))}; LoV {lov}, Latin1 {latin1}")

df = pd.DataFrame(rows); df.to_csv(args.csv, index=False)
cols = ["time_utc", "point", "speed10", "dir10_from", "speed80", "dir80_from"] + [c for c in ("gust", "blh", "hpbl") if c in df]
with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:7.1f}".format):
    print(df[cols].to_string(index=False))
print(f"wrote {args.csv}")
