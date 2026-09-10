#!/usr/bin/env python3
"""Write square.csv (a 60 m square) and cross.csv (a plus sign 160 m across with
40 m wide arms), both centred at (200, 200), as closed polygon ignitions."""
C = 200.0
with open("square.csv", "w") as f:
    f.write("# 60 m square centred at (200, 200)\n")
    for x, y in ((-30, -30), (30, -30), (30, 30), (-30, 30)):
        f.write(f"{C + x:.1f} {C + y:.1f}\n")
B, L = 20.0, 80.0          # half arm width, half span
cross = [(B, -L), (B, -B), (L, -B), (L, B), (B, B), (B, L),
         (-B, L), (-B, B), (-L, B), (-L, -B), (-B, -B), (-B, -L)]
with open("cross.csv", "w") as f:
    f.write("# plus sign 160 m across with 40 m arms, centred at (200, 200)\n")
    for x, y in cross:
        f.write(f"{C + x:.1f} {C + y:.1f}\n")
