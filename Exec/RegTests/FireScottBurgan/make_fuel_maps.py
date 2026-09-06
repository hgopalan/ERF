#!/usr/bin/env python3
"""Write the Scott-Burgan fuel map and its Anderson crosswalk twin.

    python3 make_fuel_maps.py

fuel_map_sb40.asc on the 256 x 128 fire grid (1.25 m cells, 320 x 160 m):
GR2 (102) on the west half, SH2 (142) on the east half, a TL3 (183) band
across the north (y > 140 m), an urban strip NB1 (91) at x = 200-210 m and
water NB8 (98) along the south (y < 20 m). fuel_map_anderson.asc is the same
map with the standard crosswalk applied by hand (GR2 -> 1, SH2 -> 6, TL3 -> 8,
NB -> 0), which the crosswalk deck must reproduce line for line.
"""
import numpy as np

nx, ny, dx = 256, 128, 1.25
x = (np.arange(nx) + 0.5) * dx; y = (np.arange(ny) + 0.5) * dx
X, Y = np.meshgrid(x, y)              # rows are y (north up in the file)
sb = np.where(X < 160.0, 102, 142)
sb[Y > 140.0] = 183
sb[(X >= 200.0) & (X < 210.0)] = 91
sb[Y < 20.0] = 98
cross = {102: 1, 142: 6, 183: 8, 91: 0, 98: 0}
an = np.vectorize(cross.get)(sb)

def write(fn, a):
    with open(fn, "w") as f:
        f.write(f"ncols {nx}\nnrows {ny}\nxllcorner 0.0\nyllcorner 0.0\ncellsize {dx}\nNODATA_value -9999\n")
        for row in a[::-1]:            # ESRI ASCII: first row is the north edge
            f.write(" ".join(str(int(v)) for v in row) + "\n")

write("fuel_map_sb40.asc", sb); write("fuel_map_anderson.asc", an)
codes, counts = np.unique(sb, return_counts=True)
print("cells by Scott-Burgan code:", dict(zip(codes.tolist(), counts.tolist())))
