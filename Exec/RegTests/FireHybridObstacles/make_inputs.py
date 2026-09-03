#!/usr/bin/env python3
"""Write the building heightmap and fuel map for the obstacle deck.

Domain 320 x 160 m, atmosphere at 5 m, fire grid at 1.25 m (grid_ratio 4;
the 32-cell atmosphere boxes must divide by the ratio).
Three 20 m wide, 10 m tall boxes stand across the wind at x = 180-200 m with
gaps of 30 m and 15 m between them; a 5 m non-burnable strip runs along the
wind at y = 125-130 m. Both files are committed; rerun this only to change
the layout.
"""

# Building heightmap in the ERF terrain text format read by both
# erf.buildings_file_name and erf.fire.hybrid.structure_file:
#   nx, ny, x coordinates, y coordinates, then z[ix*ny + iy].
NX, NY, DX = 65, 33, 5.0            # nodes of a 64 x 32 cell atmosphere grid
BOXES = [(180.0, 200.0, 20.0, 40.0),   # x_lo, x_hi, y_lo, y_hi
         (180.0, 200.0, 70.0, 90.0),
         (180.0, 200.0, 105.0, 125.0)]
H_BOX = 10.0

with open("buildings_5m_64x32.txt", "w") as f:
    f.write(f"{NX}\n{NY}\n")
    for i in range(NX):
        f.write(f"{i*DX:.3f}\n")
    for j in range(NY):
        f.write(f"{j*DX:.3f}\n")
    for i in range(NX):
        x = i * DX
        for j in range(NY):
            y = j * DX
            inside = any(xl <= x <= xh and yl <= y <= yh for xl, xh, yl, yh in BOXES)
            f.write(f"{H_BOX if inside else 0.0:.3f}\n")

# Fuel map on the 256 x 128 fire grid (1.25 m): FM1 short grass everywhere
# except a non-burnable street (code 0) along the wind. Row 0 is the
# northernmost row.
NCOLS, NROWS, DF = 256, 128, 1.25
with open("fuel_map_street.asc", "w") as f:
    f.write(f"ncols {NCOLS}\nnrows {NROWS}\nxllcorner 0.0\nyllcorner 0.0\n"
            f"cellsize {DF}\nnodata_value -9999\n")
    for row in range(NROWS):
        y = ((NROWS - 1 - row) + 0.5) * DF
        code = 0 if 125.0 <= y < 130.0 else 1
        f.write(" ".join(str(code) for _ in range(NCOLS)) + "\n")
print("wrote buildings_5m_64x32.txt and fuel_map_street.asc")
