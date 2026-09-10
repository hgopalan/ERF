#!/usr/bin/env python3
"""Write fuel_obstacle.asc: fuel code 1 everywhere on the 2 m fire grid of the
400 x 200 m domain, code 0 (non-burnable) inside a 30 m disc centred at (160, 100)."""
NX, NY, H = 200, 100, 2.0
XO, YO, A = 160.0, 100.0, 30.0
with open("fuel_obstacle.asc", "w") as f:
    f.write(f"ncols {NX}\nnrows {NY}\nxllcorner 0.0\nyllcorner 0.0\ncellsize {H}\nnodata_value -9999\n")
    for j in reversed(range(NY)):                       # row 0 is the northernmost
        y = (j + 0.5) * H
        f.write(" ".join("0" if ((i + 0.5) * H - XO) ** 2 + (y - YO) ** 2 <= A * A else "1"
                         for i in range(NX)) + "\n")
