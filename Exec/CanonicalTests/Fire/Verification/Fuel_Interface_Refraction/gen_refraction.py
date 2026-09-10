#!/usr/bin/env python3
"""Write fuel_halves.asc (fuel code 1 for x < 200 m, code 2 beyond, on the 2 m fire
grid of the 400 x 400 m domain) and the two tilted ignition lines: line_40.csv,
whose front normal makes 40 degrees with +x, and line_25.csv at 25 degrees, both
through (60, 200) and running well past the domain."""
import math
NX = NY = 200; H = 2.0; XI = 200.0
with open("fuel_halves.asc", "w") as f:
    f.write(f"ncols {NX}\nnrows {NY}\nxllcorner 0.0\nyllcorner 0.0\ncellsize {H}\nnodata_value -9999\n")
    row = " ".join("1" if (i + 0.5) * H < XI else "2" for i in range(NX))
    for _ in range(NY):
        f.write(row + "\n")
for ang in (40, 25):
    th = math.radians(ang)
    ux, uy = -math.sin(th), math.cos(th)            # along the line, normal (cos, sin)
    with open(f"line_{ang}.csv", "w") as f:
        f.write(f"# ignition line through (60, 200) with its normal at {ang} degrees to +x\n")
        for s in (-700.0, 700.0):
            f.write(f"{60.0 + s * ux:.6f} {200.0 + s * uy:.6f}\n")
