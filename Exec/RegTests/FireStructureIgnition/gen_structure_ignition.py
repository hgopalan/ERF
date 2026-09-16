#!/usr/bin/env python3
"""Write the rasters of the FireStructureIgnition suite: the building
heightmap (ERF terrain text, nodal, shared by the fire's structure mask and
the exposure ids) and the sounding.

Layout (metres). Domain 320 x 160, atmosphere cells 10 m, fire cells 5 m.
Three 20 m x 20 m, 8 m tall houses on the centreline y = 70-90:
  A at x = 100-120   the grass fire is ignited against its west wall
  B at x = 140-160   20 m east of A: within reach of A's radiation and brands
  C at x = 250-270   90 m further east: the control that must not ignite
The gap between A and B is two node spacings so that the footprints stay
separate (adjacent roof nodes are one 4-connected footprint). The heightmap is
sampled onto the fire cells by nearest node, so a footprint spans 2.5 m beyond
its nominal edge (A: 97.5-122.5 m, B: 137.5-162.5 m, C: 247.5-272.5 m).
"""
import numpy as np

LX, LY = 320.0, 160.0
DX_ATM = 10.0
NX_A, NY_A = int(LX / DX_ATM), int(LY / DX_ATM)     # 32 x 16
HOUSE, H_ROOF = 20.0, 8.0
HOUSES_X = [100.0, 140.0, 250.0]
Y0 = 70.0

def footprints():
    return [(x0, x0 + HOUSE, Y0, Y0 + HOUSE) for x0 in HOUSES_X]

def write_heightmap(fname):
    xs = np.arange(NX_A + 1) * DX_ATM
    ys = np.arange(NY_A + 1) * DX_ATM
    z = np.zeros((NX_A + 1, NY_A + 1))
    for (x0, x1, y0, y1) in footprints():
        ix = np.where((xs >= x0 - 1e-6) & (xs <= x1 + 1e-6))[0]
        iy = np.where((ys >= y0 - 1e-6) & (ys <= y1 + 1e-6))[0]
        z[np.ix_(ix, iy)] = H_ROOF
    with open(fname, "w") as f:
        f.write(f"{NX_A + 1}\n{NY_A + 1}\n")
        for v in xs: f.write(f"{v:.3f}\n")
        for v in ys: f.write(f"{v:.3f}\n")
        for v in z.ravel(order="C"): f.write(f"{v:.3f}\n")

def write_sounding(fname, u=10.0):
    with open(fname, "w") as f:
        f.write("1000.  300.0  0.0\n")
        f.write(f"   0.0  300.0  0.0  {u:.1f}  0.0\n")
        f.write(f" 120.0  300.0  0.0  {u:.1f}  0.0\n")

if __name__ == "__main__":
    write_heightmap("houses_10m_32x16.txt")
    write_sounding("input_sounding")
    print(f"{len(footprints())} houses; heightmap {NX_A+1}x{NY_A+1} nodes")
