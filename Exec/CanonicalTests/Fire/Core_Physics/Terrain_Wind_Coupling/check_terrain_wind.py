#!/usr/bin/env python3
"""Checks for the Terrain_Wind_Coupling case. Reads the last fire plotfile
(yt) and the terrain raster, prints what it finds and exits non-zero when a
check fails. See README.md for what each check means.

Run from the case directory after the case itself:
    python3 check_terrain_wind.py [plotfile]
"""
import glob, math, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("check_terrain_wind.py needs yt (pip install yt)")

# Deck values (inputs_fire_terrain_wind) and the FM1 spotting cap the README
# quotes, written out here so the reference is independent of the code.
IGN_X, IGN_Y, IGN_R = 1350.0, 2000.0, 100.0   # erf.fire.ignition_x/y/r [m]
WIND_REF_HT = 6.1                             # erf.fire.wind_ref_ht [m]
SPOT_RADIUS = 30.0                            # erf.fire.spotting.spot_radius [m]
ROS_MAX_ALLOWED = 5.0                         # m/s; FM1 in a 22 m/s wind on a 41 degree slope stays under 2
TERRAIN_FILE = "terrain_tubbs.txt"

ok_all = True
def fail(msg):
    global ok_all
    ok_all = False
    print("  FAIL:", msg)

# ---------------------------------------------------------------- inputs
if len(sys.argv) > 1:
    pf = sys.argv[1]
else:
    files = sorted(glob.glob("plt_fire_?????"))
    if not files:
        sys.exit("no plt_fire_????? plotfile in the current directory")
    pf = files[-1]

ds = yt.load(pf)
g = ds.covering_grid(0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
def field(n):
    return np.array(g[("boxlib", n)])[:, :, 0]
nx, ny = int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1])
lx = float(ds.domain_right_edge[0] - ds.domain_left_edge[0])
ly = float(ds.domain_right_edge[1] - ds.domain_left_edge[1])
dx, dy = lx / nx, ly / ny
t_end = float(ds.current_time)
at   = field("fire_arrival_time")
ros  = field("fire_ros")
sx   = field("fire_slope_x")
sy   = field("fire_slope_y")
ez   = field("fire_extract_z")
xc = (np.arange(nx) + 0.5) * dx
yc = (np.arange(ny) + 0.5) * dy
X, Y = np.meshgrid(xc, yc, indexing="ij")
print(f"plotfile {pf}: t = {t_end:.1f} s, fire grid {nx} x {ny}, dx = {dx:.3f} m")

# Raster: max slope along each axis, from the file the case reads.
v = np.loadtxt(TERRAIN_FILE)
ntx, nty = int(v[0]), int(v[1])
tx = v[2:2 + ntx]; ty = v[2 + ntx:2 + ntx + nty]
tz = v[2 + ntx + nty:].reshape(ntx, nty)
raster_sx = np.abs(np.diff(tz, axis=0) / np.diff(tx)[:, None]).max()
raster_sy = np.abs(np.diff(tz, axis=1) / np.diff(ty)[None, :]).max()
print(f"raster: {ntx} x {nty} points, z {tz.min():.1f}..{tz.max():.1f} m, max |dz/dx| {raster_sx:.3f}, max |dz/dy| {raster_sy:.3f}")

# ---------------------------------------------------------------- slopes
# The fire grid samples the raster bilinearly, so its slopes cannot exceed
# the raster's along either axis. The last column upwind of the outflow face
# once read (0 - z) / dx from an unfilled ghost node: a 163 m cliff.
print("== terrain slopes on the fire grid")
print(f"  slope_x {sx.min():.3f}..{sx.max():.3f}, slope_y {sy.min():.3f}..{sy.max():.3f}")
if np.abs(sx).max() > 1.05 * raster_sx or np.abs(sy).max() > 1.05 * raster_sy:
    ii, jj = np.nonzero((np.abs(sx) > 1.05 * raster_sx) | (np.abs(sy) > 1.05 * raster_sy))
    fail(f"fire-grid slope exceeds the raster's in {len(ii)} cells, columns {sorted(set(ii))[:8]}")
for name, col in (("first column", 0), ("last column", nx - 1)):
    if np.abs(sx[col]).max() > 1.05 * raster_sx:
        fail(f"{name} (x = {xc[col]:.1f} m) carries a slope of {sx[col][np.abs(sx[col]).argmax()]:.3f}")

# ---------------------------------------------------------------- ROS
print("== rate of spread")
print(f"  ROS {ros.min():.3f}..{ros.max():.3f} m/s; last column {ros[nx - 1].min():.3f}..{ros[nx - 1].max():.3f} m/s")
if ros.max() > ROS_MAX_ALLOWED:
    ii, jj = np.nonzero(ros > ROS_MAX_ALLOWED)
    fail(f"ROS above {ROS_MAX_ALLOWED} m/s in {len(ii)} cells, columns {sorted(set(ii))[:8]}")

# ---------------------------------------------------------------- reach
# Nothing west of the ignition disc burns before the backing fire can get
# there. The wind is westerly, so a brand always lands east of the cell that
# launched it and the fire can only move upwind at the local ROS (isotropic on
# the level-set path); one stamp radius allows for a brand launched from the
# western flank. Downwind the fire advances by chains of spot fires, up to the
# Scott cap every spotting interval, so no such bound holds there. The columns
# that burned 500 to 1200 m upwind within 15 s came from an unfilled ghost cell
# on the outflow column, not from the ignition.
print("== reach of the fire west of the ignition disc")
burned = at >= 0.0
r_max = min(ros.max(), ROS_MAX_ALLOWED)   # a broken ROS must not widen the allowed reach
west = burned & (X < IGN_X - IGN_R)
upwind = (IGN_X - IGN_R) - X               # distance west of the disc's western edge
reach = r_max * np.maximum(at, 0.0) + SPOT_RADIUS
early = west & (upwind > reach)
n_b = int(burned.sum())
print(f"  burned cells {n_b}, west of the disc {int(west.sum())}, "
      f"furthest west x = {X[west].min() if west.any() else float('nan'):.1f} m")
print(f"  allowed upwind reach at t: {r_max:.3f} m/s x t + {SPOT_RADIUS:.0f} m")
if early.any():
    k = np.argmax((upwind - reach)[early])
    fail(f"{int(early.sum())} cells west of the disc burned before the backing fire could reach them, "
         f"e.g. ({X[early][k]:.0f}, {Y[early][k]:.0f}) m at t = {at[early][k]:.2f} s, "
         f"{upwind[early][k]:.0f} m upwind of the disc")
if burned[0].any() or burned[nx - 1].any():
    fail(f"burned cells on the inflow ({int(burned[0].sum())}) or outflow ({int(burned[nx - 1].sum())}) column")
if n_b <= int(math.pi * IGN_R ** 2 / (dx * dy)):
    fail("the fire did not grow beyond the ignition disc")

# ---------------------------------------------------------------- extraction
print("== wind extraction height")
ground = ez - WIND_REF_HT
print(f"  extraction height {ez.min():.1f}..{ez.max():.1f} m, i.e. ground {ground.min():.1f}..{ground.max():.1f} m above the floor")
if ground.min() < tz.min() - 1.0 or ground.max() > tz.max() + 1.0:
    fail("the extraction height minus wind_ref_ht leaves the raster's elevation range")

print("PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
