#!/usr/bin/env python3
"""Write the rasters of the WUI_Subdivision case: the building heightmap the
immersed forcing and the fire's structure mask read (ERF terrain text, nodal),
the fuel maps (ESRI ASCII on the fire grid) and the sounding.

Layout (metres). Domain 960 x 480, atmosphere cells 10 m, fire cells 5 m.
  x <  480          wildland, FM1 short grass
  480 <= x < 510    fuel break in the defensible-space map only (code 0)
  three rows of 20 m x 20 m, 8 m tall houses at x = 520-540, 600-620, 680-700,
  eight per row on a 60 m pitch in y: house y = 60k-60k+20, a 20 m grass lane
  60k+20-60k+40 and a 20 m east-west street (code 0) 60k+40-60k+60, so the
  lanes carry the fire through the rows and the streets run with the wind and
  block nothing; the defensible-space map also clears 10 m around every
  house, which removes the lanes. The lanes and streets are 20 m because the
  immersed forcing blanks every atmosphere cell that touches a house node:
  a 10 m gap in a 10 m grid is a wall.
"""
import numpy as np

LX, LY = 960.0, 480.0
DX_ATM = 10.0
DX_FIRE = 5.0
NX_A, NY_A = int(LX / DX_ATM), int(LY / DX_ATM)     # 96 x 48
NX_F, NY_F = int(LX / DX_FIRE), int(LY / DX_FIRE)   # 192 x 96
HOUSE, H_ROOF = 20.0, 8.0
ROWS_X = [520.0, 600.0, 680.0]
PITCH_Y = 60.0
HOUSES_Y = [PITCH_Y * j for j in range(8)]
STREET_Y0, STREET_W = 40.0, 20.0      # east-west street within each 60 m pitch
BREAK = (480.0, 510.0)
CLEAR = 10.0

def footprints():
    return [(x0, x0 + HOUSE, y0, y0 + HOUSE) for x0 in ROWS_X for y0 in HOUSES_Y]

def write_heightmap(fname):
    # Nodal heights, i-major, as ERF_FireTerrainReader reads them (z[ix*ny + iy])
    # and as the immersed-forcing reader shares the file.
    xs = np.arange(NX_A + 1) * DX_ATM
    ys = np.arange(NY_A + 1) * DX_ATM
    z = np.zeros((NX_A + 1, NY_A + 1))
    for (x0, x1, y0, y1) in footprints():
        ix = np.where((xs >= x0 - 1e-6) & (xs <= x1 + 1e-6))[0]
        iy = np.where((ys >= y0 - 1e-6) & (ys <= y1 + 1e-6))[0]
        z[np.ix_(ix, iy)] = H_ROOF
    # One value per line: the atmosphere's terrain reader takes the header as
    # two single values and the fire's reader streams the same file.
    with open(fname, "w") as f:
        f.write(f"{NX_A + 1}\n{NY_A + 1}\n")
        for v in xs: f.write(f"{v:.3f}\n")
        for v in ys: f.write(f"{v:.3f}\n")
        for v in z.ravel(order="C"): f.write(f"{v:.3f}\n")

def fuel_codes(defensible):
    xc = (np.arange(NX_F) + 0.5) * DX_FIRE
    yc = (np.arange(NY_F) + 0.5) * DX_FIRE
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    code = np.ones((NX_F, NY_F), dtype=int)
    in_rows = (X >= ROWS_X[0]) & (X < ROWS_X[-1] + HOUSE)
    street = ((Y % PITCH_Y) >= STREET_Y0) & ((Y % PITCH_Y) < STREET_Y0 + STREET_W)
    code[in_rows & street] = 0
    if defensible:
        code[(X >= BREAK[0]) & (X < BREAK[1])] = 0
        for (x0, x1, y0, y1) in footprints():
            code[(X >= x0 - CLEAR) & (X < x1 + CLEAR) & (Y >= y0 - CLEAR) & (Y < y1 + CLEAR)] = 0
    return code

def write_fuel_map(fname, code):
    # ESRI ASCII: rows from the north, so the last j row comes first.
    with open(fname, "w") as f:
        f.write(f"ncols {NX_F}\nnrows {NY_F}\nxllcorner 0.0\nyllcorner 0.0\n"
                f"cellsize {DX_FIRE}\nnodata_value -9999\n")
        for j in range(NY_F - 1, -1, -1):
            f.write(" ".join(str(int(v)) for v in code[:, j]) + "\n")

def write_sounding(fname, u=10.0):
    with open(fname, "w") as f:
        f.write("1000.  300.0  0.0\n")
        f.write(f"   0.0  300.0  0.0  {u:.1f}  0.0\n")
        f.write(f" 240.0  300.0  0.0  {u:.1f}  0.0\n")

if __name__ == "__main__":
    write_heightmap("houses_10m_96x48.txt")
    write_fuel_map("fuel_map_subdivision.asc", fuel_codes(False))
    write_fuel_map("fuel_map_defensible.asc", fuel_codes(True))
    write_sounding("input_sounding")
    n = len(footprints())
    print(f"{n} houses; heightmap {NX_A+1}x{NY_A+1} nodes; fuel maps {NX_F}x{NY_F}")
