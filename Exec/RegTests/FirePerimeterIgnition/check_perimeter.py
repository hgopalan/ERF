#!/usr/bin/env python3
"""Cell-by-cell checks of the perimeter ignition from the fire plotfiles.

    python3 check_perimeter.py interior plt_fire_00000 0.5 60.0 0.0
    python3 check_perimeter.py arrival  plt_fire_00480 0.5 30.0

interior: every burned cell (phi < 0) of the plotfile written at the stamp
          time has fuel = w0 exp(-d / (R tau)) and arrival = t_ign - d / R with
          d = -phi, w0 the fuel of the unburned cells (to 1e-6 relative).
arrival:  in a later plotfile, every cell whose arrival time is below the
          perimeter time (a stamped interior cell) obeys arrival = max(t_ign - d0 / R, 0)
          with d0 its geometric distance inside the 40 m square (80-120, 60-100),
          which the level set's later motion does not change (to 1e-6).
"""
import sys
import numpy as np
import yt

def fields(ds):
    names = [f for _, f in ds.field_list]
    def pick(sub):
        c = [n for n in names if sub in n]
        if not c: sys.exit(f"no field containing '{sub}' in {names}")
        return ("boxlib", c[0])
    return pick("phi"), pick("fuel_load"), pick("arrival")

def load(pf):
    ds = yt.load(pf)
    fphi, ffuel, fat = fields(ds)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    xc = g["index", "x"].value[:, :, 0]; yc = g["index", "y"].value[:, :, 0]
    return g[fphi].value[:, :, 0], g[ffuel].value[:, :, 0], g[fat].value[:, :, 0], xc, yc

SQUARE = (80.0, 120.0, 60.0, 100.0)   # the polygon of square_40m.csv

def dist_inside(xc, yc):
    x0, x1, y0, y1 = SQUARE
    return np.minimum(np.minimum(xc - x0, x1 - xc), np.minimum(yc - y0, y1 - yc))

def main():
    mode, pf, R = sys.argv[1], sys.argv[2], float(sys.argv[3])
    phi, fuel, at, xc, yc = load(pf)
    inside = phi < 0.0
    status = 0
    if mode == "interior":
        tau, t_ign = float(sys.argv[4]), float(sys.argv[5])
        w0 = np.median(fuel[~inside])
        d = -phi[inside]
        fuel_exp = w0 * np.exp(-d / (R * tau)); at_exp = np.maximum(t_ign - d / R, 0.0)
        ef = np.max(np.abs(fuel[inside] - fuel_exp) / w0); ea = np.max(np.abs(at[inside] - at_exp))
        ok = ef < 1e-6 and ea < 1e-6
        print(f"  interior state of {inside.sum()} burned cells: fuel = w0 exp(-d/(R tau)) and arrival = max(t_ign - d/R, 0): "
              f"{'PASS' if ok else 'FAIL'} (max fuel diff {ef:.1e} of w0 = {w0:.3f} kg/m2, max arrival diff {ea:.1e} s; "
              f"fuel range {fuel[inside].min():.3f}-{fuel[inside].max():.3f}, arrival range {at[inside].min():.1f}-{at[inside].max():.1f} s)")
        status |= (not ok)
        # the unburned cells are untouched
        ok2 = np.allclose(fuel[~inside], w0) and np.all(at[~inside] < 0.0)
        print(f"  unburned cells keep w0 and no arrival time: {'PASS' if ok2 else 'FAIL'}"); status |= (not ok2)
    elif mode == "arrival":
        t_ign = float(sys.argv[4])
        sel = inside & (at < t_ign - 1e-6)            # stamped interior cells: arrival before the perimeter time
        d0 = dist_inside(xc, yc)[sel]
        ok = sel.sum() > 0 and np.all(d0 > 0.0)
        ea = np.max(np.abs(at[sel] - np.maximum(t_ign - d0 / R, 0.0))) if sel.sum() else np.inf
        ok = ok and ea < 1e-6
        print(f"  {sel.sum()} interior cells stamped at {t_ign} s obey arrival = max(t_ign - d0/R, 0) (d0 geometric): "
              f"{'PASS' if ok else 'FAIL'} (max diff {ea:.1e} s; arrival range {at[sel].min():.2f}-{at[sel].max():.2f} s)" if sel.sum()
              else "  no stamped interior cells found: FAIL")
        status |= (not ok)
    sys.exit(status)

if __name__ == "__main__":
    main()
