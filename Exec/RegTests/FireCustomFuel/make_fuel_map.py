#!/usr/bin/env python3
"""Write the mixed fuel map for FireCustomFuel.

    python3 make_fuel_map.py

fuel_map_mixed.asc on the 256 x 128 fire grid (1.25 m cells, 320 x 160 m):
Scott-Burgan GR2 (102) everywhere, a deck-defined block (1000) over
x = 140-200 m around the ignition, an urban strip NB1 (91) at x = 240-250 m and
water NB8 (98) along the south (y < 20 m). The point is the mixture: one front
crosses from a published model into a deck-defined one with no seam.

fuel_map_undeclared.asc is the same map with the block at 1007, a code in the
custom range that no deck defines, which start-up has to reject.
"""
import numpy as np

nx, ny, dx = 256, 128, 1.25
x = (np.arange(nx) + 0.5) * dx
y = (np.arange(ny) + 0.5) * dx
X, Y = np.meshgrid(x, y)              # rows are y (north up in the file)


def build(block_code):
    a = np.full((ny, nx), 102)
    a[(X >= 140.0) & (X < 200.0)] = block_code
    a[(X >= 240.0) & (X < 250.0)] = 91
    a[Y < 20.0] = 98
    return a


def write(fn, a):
    with open(fn, "w") as f:
        f.write(f"ncols {nx}\nnrows {ny}\nxllcorner 0.0\nyllcorner 0.0\n"
                f"cellsize {dx}\nNODATA_value -9999\n")
        for row in a[::-1]:            # ESRI ASCII: first row is the north edge
            f.write(" ".join(str(int(v)) for v in row) + "\n")


mixed = build(1000)
write("fuel_map_mixed.asc", mixed)
write("fuel_map_undeclared.asc", build(1007))
codes, counts = np.unique(mixed, return_counts=True)
print("cells by code:", dict(zip(codes.tolist(), counts.tolist())))
