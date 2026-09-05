#!/usr/bin/env python3
"""Measure the burned shape in a fire plotfile.

    python3 check_shape.py plt_fire_xxx/plt_fire_00480 x_ign y_ign [r_ign LB tol_m]

Prints the extents of the burned region (phi < 0) along and across the wind
(x and y here): head, back and half-width from the ignition point, and the
length-to-width ratio. With r_ign, LB and tol_m given, checks the Huygens
envelope of the ignition disc: with the head travel h = head - r_ign taken
from the plotfile, the ellipse with length-to-width LB and Alexander's
head-to-back HB predicts back travel h / HB and half-width travel
h (1 + 1/HB) / (2 LB); both extents must match to tol_m (one fire cell).
"""
import sys
import numpy as np
import yt

def alexander_hb(LB):
    s = np.sqrt(max(LB * LB - 1.0, 0.0))
    return (LB + s) / (LB - s) if LB - s > 1e-12 else 1e12

def main():
    pf, x0, y0 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    ds = yt.load(pf); names = [f for _, f in ds.field_list]
    fphi = ("boxlib", [n for n in names if "phi" in n][0])
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    phi = g[fphi].value[:, :, 0]; xc = g["index", "x"].value[:, :, 0]; yc = g["index", "y"].value[:, :, 0]
    b = phi < 0.0
    dx = float((ds.domain_right_edge[0] - ds.domain_left_edge[0]) / ds.domain_dimensions[0])
    xmax, xmin = xc[b].max() + dx / 2, xc[b].min() - dx / 2
    ymax, ymin = yc[b].max() + dx / 2, yc[b].min() - dx / 2
    length, width = xmax - xmin, ymax - ymin
    print(f"  burned cells {b.sum()}: head {xmax - x0:.1f} m, back {x0 - xmin:.1f} m, half-width {(width / 2):.1f} m, "
          f"length/width {length / width:.3f} at t = {float(ds.current_time):.0f} s")
    if len(sys.argv) > 4:
        r0, LB, tol = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
        HB = alexander_hb(LB); h = (xmax - x0) - r0
        back_pred = r0 + h / HB; half_pred = r0 + h * (1.0 + 1.0 / HB) / (2.0 * LB)
        eb, ew = abs((x0 - xmin) - back_pred), abs(width / 2 - half_pred)
        ok = eb <= tol and ew <= tol
        print(f"  envelope of the {r0:.0f} m disc with LB {LB:.3f}, HB {HB:.1f}: back {x0 - xmin:.2f} vs {back_pred:.2f} m, "
              f"half-width {width / 2:.2f} vs {half_pred:.2f} m: {'PASS' if ok else 'FAIL'} (tol {tol} m)")
        sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
