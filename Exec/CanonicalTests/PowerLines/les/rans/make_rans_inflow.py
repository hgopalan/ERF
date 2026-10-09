#!/usr/bin/env python3
"""Write the RANS runs' inflow_profile (z u v w theta) and input_sounding from the LES precursor's mean profile.

The precursor's mean_profiles.dat (time z u v w rho theta ke ...) is averaged over its rows from --t_from on, level by
level; a row at z = 0 with no wind and one at the domain top (--top) with the highest level's wind close each file.
theta is 300 K throughout (the precursor is neutral); the sounding's surface pressure is 1000 hPa.

Usage: make_rans_inflow.py PRECURSOR_DIR [--t_from 7200] [--top 768]
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("precursor")
    ap.add_argument("--t_from", type=float, default=7200.0)
    ap.add_argument("--top", type=float, default=768.0)
    args = ap.parse_args()
    d = np.loadtxt(os.path.join(args.precursor, "mean_profiles.dat"))
    d = d[d[:, 0] >= args.t_from]
    zs = np.unique(d[:, 1])
    rows = [(z, d[d[:, 1] == z, 2].mean(), d[d[:, 1] == z, 3].mean()) for z in zs]
    rows = [(0.0, 0.0, 0.0)] + rows + [(args.top, rows[-1][1], rows[-1][2])]
    with open("inflow_profile", "w") as f:
        for z, u, v in rows:
            f.write("%.3f %.6f %.6f 0.0 300.0\n" % (z, u, v))
    with open("input_sounding", "w") as f:
        f.write("1000.0 300.0 0.0\n")
        for z, u, v in rows:
            f.write("%.3f 300.0 0.0 %.6f %.6f\n" % (z, u, v))
    print("wrote inflow_profile and input_sounding:", len(rows), "levels, u at the top", rows[-1][1])


if __name__ == "__main__":
    main()
