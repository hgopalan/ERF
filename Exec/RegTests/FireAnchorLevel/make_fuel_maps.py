#!/usr/bin/env python3
"""Write the two fuel maps of the FireAnchorLevel decks.

    python3 make_fuel_maps.py

fuel_region.asc covers the refined region of inputs_base (x 200-600 m, y 200-500 m,
64 x 48 fire cells of 6.25 m); fuel_full.asc covers the domain of inputs_single
(800 x 800 m, 128 x 128 cells). Both carry the same checkerboard by position,
code 4 where floor(x / 25) + floor(y / 50) is odd and code 1 elsewhere, so a map
placed one row or column off changes the fire.
"""

DX = 6.25


def code(x, y):
    return 4 if (int(x // 25.0) + int(y // 50.0)) % 2 else 1


def write(name, x0, y0, nx, ny):
    with open(name, "w") as f:
        f.write(f"ncols {nx}\nnrows {ny}\nxllcorner {x0}\nyllcorner {y0}\ncellsize {DX}\nNODATA_value -9999\n")
        for j in reversed(range(ny)):          # the first data row is the north edge
            y = y0 + (j + 0.5) * DX
            f.write(" ".join(str(code(x0 + (i + 0.5) * DX, y)) for i in range(nx)) + "\n")


write("fuel_region.asc", 200.0, 200.0, 64, 48)
write("fuel_full.asc", 0.0, 0.0, 128, 128)
