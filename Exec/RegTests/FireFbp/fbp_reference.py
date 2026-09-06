#!/usr/bin/env python3
"""Independent FBP rate of spread for the checks.

    python3 fbp_reference.py TYPE FFMC BUI W_mps [curing] [pc] [pdf]

Prints the surface rate of spread [m/s] on flat ground from the equations
of Forestry Canada (1992) and Wotton, Alexander and Taylor (2009).
"""
import math, sys

COEF = {"C1": (90, 0.0649, 4.5, 0.90, 72), "C2": (110, 0.0282, 1.5, 0.70, 64), "C3": (110, 0.0444, 3.0, 0.75, 62),
        "C4": (110, 0.0293, 1.5, 0.80, 66), "C5": (30, 0.0697, 4.0, 0.80, 56), "C6": (30, 0.0800, 3.0, 0.80, 62),
        "C7": (45, 0.0305, 2.0, 0.85, 106), "D1": (30, 0.0232, 1.6, 0.90, 32), "M3": (120, 0.0572, 1.4, 0.80, 50),
        "M4": (100, 0.0404, 1.48, 0.80, 50), "S1": (75, 0.0297, 1.3, 0.75, 38), "S2": (40, 0.0438, 1.7, 0.75, 63),
        "S3": (55, 0.0829, 3.2, 0.75, 31), "O1A": (190, 0.0310, 1.4, 1.0, 1), "O1B": (250, 0.0350, 1.7, 1.0, 1)}

def basic(t, isi):
    a, b, c, _, _ = COEF[t]; return a * max(0.0, 1 - math.exp(-b * isi)) ** c

def ros(t, ffmc, bui, W_mps, curing=60.0, pc=50.0, pdf=50.0):
    m = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    fF = 91.9 * math.exp(-0.1386 * m) * (1 + m ** 5.31 / 4.93e7)
    isi = 0.208 * fF * math.exp(0.05039 * 3.6 * W_mps)
    if t == "M1": rsi = pc / 100 * basic("C2", isi) + (1 - pc / 100) * basic("D1", isi); q, b0 = 0.80, 50
    elif t == "M2": rsi = pc / 100 * basic("C2", isi) + 0.2 * (1 - pc / 100) * basic("D1", isi); q, b0 = 0.80, 50
    elif t in ("M3", "M4"): rsi = pdf / 100 * basic(t, isi) + (0.2 if t == "M4" else 1.0) * (1 - pdf / 100) * basic("D1", isi); q, b0 = 0.80, 50
    elif t in ("O1A", "O1B"):
        cf = 0.005 * (math.exp(0.061 * curing) - 1) if curing < 58.8 else 0.176 + 0.02 * (curing - 58.8)
        rsi = cf * basic(t, isi); q, b0 = 1.0, 1
    else: rsi = basic(t, isi); q, b0 = COEF[t][3], COEF[t][4]
    be = 1.0 if (q >= 1.0 or bui <= 0) else math.exp(50 * math.log(q) * (1 / bui - 1 / b0))
    return rsi * be / 60.0

if __name__ == "__main__":
    a = sys.argv[1:]
    t, ffmc, bui, W = a[0].upper(), float(a[1]), float(a[2]), float(a[3])
    extra = [float(v) for v in a[4:]]
    print(f"{ros(t, ffmc, bui, W, *extra):.10g}")
