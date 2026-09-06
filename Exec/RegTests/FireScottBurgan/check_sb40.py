#!/usr/bin/env python3
"""Checks of the Scott-Burgan support that need the table or the plotfile.

    python3 check_sb40.py params run_sb_gr2.log 102
    python3 check_sb40.py fuel   run_sb_map.log fuel_map_sb40.asc 1.25
    python3 check_sb40.py mask   plt_fire_sb_map/plt_fire_00480 fuel_map_sb40.asc

params: the uniform fuel parameters the code printed equal the Scott and
        Burgan (2005) table for that code (loads t/ac -> lb/ft2, SAV, depth,
        moisture of extinction, heat content), with the herbaceous curing
        transfer at the deck's live moisture.
fuel:   the initial fuel on the grid printed by the code equals the sum over
        the map of each cell's model load (kg/m2 x cell area), non-burnable
        codes contributing nothing.
mask:   no cell with a non-burnable code (91-99) is burned in the plotfile.
"""
import re, sys
import numpy as np

# Scott & Burgan (2005) Table: code: (1h, 10h, 100h, live herb, live woody [t/ac],
#   SAV 1h, SAV herb, SAV woody [1/ft], depth [ft], Mx [%], heat [BTU/lb])
SB40 = {
 101: (0.10, 0.00, 0.00, 0.30, 0.00, 2200, 2000,    0, 0.4, 15, 8000),
 102: (0.10, 0.00, 0.00, 1.00, 0.00, 2000, 1800,    0, 1.0, 15, 8000),
 103: (0.10, 0.40, 0.00, 1.50, 0.00, 1500, 1300,    0, 2.0, 30, 8000),
 104: (0.25, 0.00, 0.00, 1.90, 0.00, 2000, 1800,    0, 2.0, 15, 8000),
 105: (0.40, 0.00, 0.00, 2.50, 0.00, 1800, 1600,    0, 1.5, 40, 8000),
 106: (0.10, 0.00, 0.00, 3.40, 0.00, 2200, 2000,    0, 1.5, 40, 9000),
 107: (1.00, 0.00, 0.00, 5.40, 0.00, 2000, 1800,    0, 3.0, 15, 8000),
 108: (0.50, 1.00, 0.00, 7.30, 0.00, 1500, 1300,    0, 4.0, 30, 8000),
 109: (1.00, 1.00, 0.00, 9.00, 0.00, 1800, 1600,    0, 5.0, 40, 8000),
 121: (0.20, 0.00, 0.00, 0.50, 0.65, 2000, 1800, 1800, 0.9, 15, 8000),
 122: (0.50, 0.50, 0.00, 0.60, 1.00, 2000, 1800, 1800, 1.5, 15, 8000),
 123: (0.30, 0.25, 0.00, 1.45, 1.25, 1800, 1600, 1600, 1.8, 40, 8000),
 124: (1.90, 0.30, 0.10, 3.40, 7.10, 1800, 1600, 1600, 2.1, 40, 8000),
 141: (0.25, 0.25, 0.00, 0.15, 1.30, 2000, 1800, 1600, 1.0, 15, 8000),
 142: (1.35, 2.40, 0.75, 0.00, 3.85, 2000,    0, 1600, 1.0, 15, 8000),
 143: (0.45, 3.00, 0.00, 0.00, 6.20, 1600,    0, 1400, 2.4, 40, 8000),
 144: (0.85, 1.15, 0.20, 0.00, 2.55, 2000, 1800, 1600, 3.0, 30, 8000),
 145: (3.60, 2.10, 0.00, 0.00, 2.90,  750,    0, 1600, 6.0, 15, 8000),
 146: (2.90, 1.45, 0.00, 0.00, 1.40,  750,    0, 1600, 2.0, 30, 8000),
 147: (3.50, 5.30, 2.20, 0.00, 3.40,  750,    0, 1600, 6.0, 15, 8000),
 148: (2.05, 3.40, 0.85, 0.00, 4.35,  750,    0, 1600, 3.0, 40, 8000),
 149: (4.50, 2.45, 0.00, 1.55, 7.00,  750, 1800, 1500, 4.4, 40, 8000),
 161: (0.20, 0.90, 1.50, 0.20, 0.90, 2000, 1800, 1600, 0.6, 20, 8000),
 162: (0.95, 1.80, 1.25, 0.00, 0.20, 2000,    0, 1600, 1.0, 30, 8000),
 163: (1.10, 0.15, 0.25, 0.65, 1.10, 1800, 1600, 1400, 1.3, 30, 8000),
 164: (4.50, 0.00, 0.00, 0.00, 2.00, 2300,    0, 2000, 0.5, 12, 8000),
 165: (4.00, 4.00, 3.00, 0.00, 3.00, 1500,    0,  750, 1.0, 25, 8000),
 181: (1.00, 2.20, 3.60, 0.00, 0.00, 2000,    0,    0, 0.2, 30, 8000),
 182: (1.40, 2.30, 2.20, 0.00, 0.00, 2000,    0,    0, 0.2, 25, 8000),
 183: (0.50, 2.20, 2.80, 0.00, 0.00, 2000,    0,    0, 0.3, 20, 8000),
 184: (0.50, 1.50, 4.20, 0.00, 0.00, 2000,    0,    0, 0.4, 25, 8000),
 185: (1.15, 2.50, 4.40, 0.00, 0.00, 2000,    0,    0, 0.6, 25, 8000),
 186: (2.40, 1.20, 1.20, 0.00, 0.00, 2000,    0,    0, 0.3, 25, 8000),
 187: (0.30, 1.40, 8.10, 0.00, 0.00, 2000,    0,    0, 0.4, 25, 8000),
 188: (5.80, 1.40, 1.10, 0.00, 0.00, 1800,    0,    0, 0.3, 35, 8000),
 189: (6.65, 3.30, 4.15, 0.00, 0.00, 1800,    0,    0, 0.6, 35, 8000),
 201: (1.50, 3.00, 11.00, 0.00, 0.00, 2000,   0,    0, 1.0, 25, 8000),
 202: (4.50, 4.25, 4.00, 0.00, 0.00, 2000,    0,    0, 1.0, 25, 8000),
 203: (5.50, 2.75, 3.00, 0.00, 0.00, 2000,    0,    0, 1.2, 25, 8000),
 204: (5.25, 3.50, 5.25, 0.00, 0.00, 2000,    0,    0, 2.7, 25, 8000),
}
TPA_TO_LBFT2 = 2000.0 / 43560.0
LBFT2_TO_KGM2 = 4.88243

def params(code, m_live):
    w1, w10, w100, wlh, wlw, s1, slh, slw, depth, mx, heat = SB40[code]
    w1, w10, w100, wlh, wlw = [v * TPA_TO_LBFT2 for v in (w1, w10, w100, wlh, wlw)]
    if wlh > 0.0 and m_live >= 0.0:      # dynamic model: cured herbaceous load moves to the 1-h class
        T = min(max((1.2 - m_live) / 0.9, 0.0), 1.0)
        w1 += T * wlh; wlh *= (1.0 - T)
    return dict(w_d1=w1, w_d10=w10, w_d100=w100, w_lh=wlh, w_lw=wlw, sigma_d1=s1, delta=depth, Mx=mx / 100.0, heat=heat)

def load_kgm2(code, m_live):
    if code not in SB40: return 0.0
    p = params(code, m_live)
    return (p["w_d1"] + p["w_d10"] + p["w_d100"] + p["w_lh"] + p["w_lw"]) * LBFT2_TO_KGM2

def read_map(fn):
    with open(fn) as f:
        hdr = {}
        for _ in range(6):
            k, v = f.readline().split(); hdr[k] = float(v)
        rows = [list(map(int, line.split())) for line in f if line.strip()]
    return np.array(rows)[::-1], hdr

def main():
    mode = sys.argv[1]; status = 0
    if mode == "params":
        log, code = sys.argv[2], int(sys.argv[3])
        line = [l for l in open(log) if "Uniform fuel model" in l][-1]
        got = {k: float(v) for k, v in re.findall(r"(\w+)=([-+0-9.eE]+)", line) if k != "code"}
        m_live = got.get("M_live", -1.0)
        exp = params(code, m_live)
        worst = max(abs(got[k] - exp[k]) / max(abs(exp[k]), 1e-12) for k in exp)
        ok = worst < 1e-6
        print(f"  uniform model {code} parameters equal the Scott-Burgan table (curing at M_live {m_live:.2f}): "
              f"{'PASS' if ok else 'FAIL'} (worst rel. diff {worst:.1e}; w_d1 {got['w_d1']:.5f} vs {exp['w_d1']:.5f} lb/ft2, "
              f"w_lh {got['w_lh']:.5f} vs {exp['w_lh']:.5f}, Mx {got['Mx']} vs {exp['Mx']}, heat {got['heat']} vs {exp['heat']})")
        status |= (not ok)
    elif mode == "fuel":
        log, fn, dx = sys.argv[2], sys.argv[3], float(sys.argv[4])
        m = read_map(fn)[0]
        m_live = 0.60
        # the load the code reports at initialisation, before any depletion
        line = [l for l in open(log) if "Fuel load from the map:" in l][-1]
        got = float(re.search(r"Fuel load from the map: ([^ ]+) kg", line).group(1))
        codes, counts = np.unique(m, return_counts=True)
        exp = sum(load_kgm2(int(c), m_live) * n * dx * dx for c, n in zip(codes, counts))
        d = abs(got - exp) / exp; ok = d < 1e-6
        print(f"  initial fuel on the grid from the map's per-cell loads: {'PASS' if ok else 'FAIL'} ({got:.3f} vs {exp:.3f} kg, rel. diff {d:.1e})")
        status |= (not ok)
    elif mode == "mask":
        import yt
        pf, fn = sys.argv[2], sys.argv[3]
        m = read_map(fn)[0]
        ds = yt.load(pf); names = [f for _, f in ds.field_list]
        fphi = ("boxlib", [n for n in names if "phi" in n][0])
        g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        phi = g[fphi].value[:, :, 0].T          # to (y, x) like the map
        nb = (m >= 91) & (m <= 99)
        burned_nb = int(((phi < 0.0) & nb).sum()); burned = int((phi < 0.0).sum())
        ok = burned_nb == 0 and burned > 0
        print(f"  non-burnable codes stay unburned at 60 s: {'PASS' if ok else 'FAIL'} ({burned_nb} of {int(nb.sum())} NB cells burned; {burned} cells burned in all)")
        status |= (not ok)
    sys.exit(status)

if __name__ == "__main__":
    main()
