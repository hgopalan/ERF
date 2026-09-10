#!/usr/bin/env python3
"""Write v<angle>.csv: a V-shaped polyline, apex at (40, 100), opening towards +x,
arms 180 m long at +-angle/2 from the x axis, for angle = 30, 60 and 90 degrees."""
import math
AX, AY, ARM = 40.0, 100.0, 180.0
for ang in (30, 60, 90):
    h = math.radians(ang / 2.0)
    with open(f"v{ang}.csv", "w") as f:
        f.write(f"# V ignition, {ang} degrees between the arms, apex ({AX}, {AY})\n")
        f.write(f"{AX + ARM * math.cos(h):.6f} {AY + ARM * math.sin(h):.6f}\n")
        f.write(f"{AX:.6f} {AY:.6f}\n")
        f.write(f"{AX + ARM * math.cos(h):.6f} {AY - ARM * math.sin(h):.6f}\n")
