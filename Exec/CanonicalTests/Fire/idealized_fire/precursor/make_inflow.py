#!/usr/bin/env python3
"""Turn a settled precursor column into the inflow profile and starting sounding of a fire deck.

    python3 make_inflow.py RUN_DIR PBL [TARGET_FROM]      (e.g. . mrf 267)

RUN_DIR holds the plt_pre* and plt2d_pre* plotfiles of inputs_precursor_PBL; TARGET_FROM is
the direction the 10 m wind should blow from (default 267 degrees, the HRRR analysis at the
west face of the Marshall domain at 18 UTC on 30 December 2021).

- Takes the horizontal mean of u, v, theta and Kmv in every plt_pre plotfile and prints how
  much the column below 3 km still changes over its last three hours, with u* and the PBL height.
- Rotates the final column so its 10 m wind blows from TARGET_FROM. On an f-plane a
  horizontally uniform column is invariant under rotation, so this equals running the
  precursor with the geostrophic wind rotated by the same angle, which is printed for the
  fire deck's erf.abl_geo_wind.
- Writes inflow_PBL.txt ('# z u v T', heights above the ground, for <face>.inflow_profile_file)
  and sounding_PBL.txt (840 hPa; z theta qv u v, for erf.input_sounding_file with the wind and
  theta read above the ground) into the current directory.
"""
import glob
import math
import sys

import numpy as np
import yt

yt.set_log_level(50)
if len(sys.argv) < 3:
    sys.exit(__doc__)
run, pbl = sys.argv[1], sys.argv[2]
TARGET_FROM = float(sys.argv[3]) if len(sys.argv) > 3 else 267.0
GEO = (64.71703, -17.34088)          # erf.abl_geo_wind of inputs_precursor_common


def wind_from(u, v):
    return np.degrees(np.arctan2(-u, -v)) % 360.0


def column(p):
    ds = yt.load(p)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    mean = lambda f: np.asarray(g["boxlib", f]).mean(axis=(0, 1))
    return float(ds.current_time), mean("z_phys"), mean("x_velocity"), mean("y_velocity"), mean("theta"), mean("Kmv")


files = sorted(glob.glob(f"{run}/plt_pre?????"))
if not files:
    sys.exit(f"no plt_pre plotfiles in {run}")
cols = [column(p) for p in files]
t, z, u, v, th, km = cols[-1]
below = z < 3000.0
print(f"== {pbl}: {len(cols)} plotfiles, final t = {t / 3600:.2f} h")
print("  change of the column below 3 km against the final one:")
print("  t [h]   max |d speed| [m/s]   max |d dir| [deg]   max |d theta| [K]")
for tc, zc, uc, vc, thc, _ in cols[-4:-1]:
    dd = (wind_from(uc, vc) - wind_from(u, v) + 180.0) % 360.0 - 180.0
    print(f"  {tc / 3600:5.2f}   {np.abs(np.hypot(uc, vc) - np.hypot(u, v))[below].max():19.3f}"
          f"   {np.abs(dd)[below].max():17.2f}   {np.abs(thc - th)[below].max():17.3f}")

d2 = yt.load(sorted(glob.glob(f"{run}/plt2d_pre?????"))[-1])
g2 = d2.covering_grid(0, d2.domain_left_edge, d2.domain_dimensions)
ustar = float(np.asarray(g2["boxlib", "u_star"]).mean())
pblh = float(np.asarray(g2["boxlib", "pblh"]).mean())
print(f"  final u* = {ustar:.3f} m/s, PBL height = {pblh:.0f} m")

u10, v10 = np.interp(10.0, z, u), np.interp(10.0, z, v)
dir10 = float(wind_from(u10, v10))
delta = math.radians(TARGET_FROM - dir10)
c, s = math.cos(delta), math.sin(delta)
ur, vr = u * c + v * s, -u * s + v * c
gu, gv = GEO[0] * c + GEO[1] * s, -GEO[0] * s + GEO[1] * c
print(f"  10 m wind {math.hypot(u10, v10):.2f} m/s from {dir10:.1f} deg; rotated by {TARGET_FROM - dir10:+.1f} deg to {TARGET_FROM:.0f}")
print(f"  erf.abl_geo_wind = {gu:.5f} {gv:.5f} 0.0")
print("  turning with height: " + ", ".join(
    f"{zz:.0f} m {math.hypot(np.interp(zz, z, ur), np.interp(zz, z, vr)):.1f} m/s from "
    f"{float(wind_from(np.interp(zz, z, ur), np.interp(zz, z, vr))):.0f}"
    for zz in (10, 100, 500, 1000, 2000, 4000)))

with open(f"inflow_{pbl}.txt", "w") as f:
    f.write(f"# {pbl.upper()} precursor column (flat, periodic, 67 m/s geostrophic wind, 39.95 N) at t = {t / 3600:.2f} h,\n")
    f.write(f"# rotated so the 10 m wind blows from {TARGET_FROM:.0f} deg; u* {ustar:.3f} m/s, PBL height {pblh:.0f} m.\n")
    f.write("# Heights are above the local ground.\n")
    f.write("# z u v T\n")
    for row in zip(z, ur, vr, th):
        f.write("%10.3f %10.5f %10.5f %9.4f\n" % row)

with open(f"sounding_{pbl}.txt", "w") as f:
    f.write("840.0 300.0 0.0\n")
    for row in zip(z, th, ur, vr):
        f.write("%10.3f %9.4f 0.0 %10.5f %10.5f\n" % row)
    f.write("%10.3f %9.4f 0.0 %10.5f %10.5f\n" % (z[-1] + 1000.0, th[-1] + 0.003 * 1000.0, ur[-1], vr[-1]))
print(f"  wrote inflow_{pbl}.txt and sounding_{pbl}.txt")
