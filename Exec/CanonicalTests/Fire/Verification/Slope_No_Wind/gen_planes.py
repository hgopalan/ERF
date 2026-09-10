#!/usr/bin/env python3
"""Write plane_s30.txt and plane_s60.txt: planes z = s x with s = 0.3 and 0.6 over
the 400 x 200 m domain, and plane_s30_wide.txt and plane_s60_wide.txt over the
400 x 400 m domain of the and_* decks, in the ERF terrain text format the fire
reads through erf.fire.terrain_file_name (nx, ny, the x and y node coordinates,
then z stored contiguous in y)."""
xs = [10.0 * i for i in range(41)]
for suffix, ny in (("", 21), ("_wide", 41)):
    ys = [10.0 * j for j in range(ny)]
    for tag, s in (("s30", 0.3), ("s60", 0.6)):
        with open(f"plane_{tag}{suffix}.txt", "w") as f:
            f.write(f"{len(xs)}\n{len(ys)}\n")
            f.write("".join(f"{x:.4f}\n" for x in xs))
            f.write("".join(f"{y:.4f}\n" for y in ys))
            f.write("".join(f"{s * x:.4f}\n" for x in xs for _ in ys))
