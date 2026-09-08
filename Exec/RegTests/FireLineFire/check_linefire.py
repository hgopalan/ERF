#!/usr/bin/env python3
"""Rates of spread of the line fire from the run logs, against Rothermel and Coen et al. (2013).

    python3 check_linefire.py nowind wind2p5 wind5 [wind2p5_2way]

For each variant the head and backing rates are the distance between
consecutive probe cells divided by the difference of their arrival times
([FIRE PROBE] lines). The expected rates are Rothermel (1972) for Anderson
fuel model 1 at the deck's moisture, evaluated at the effective (post wind
reduction factor) wind the fire reports ([FIRE DEBUG] Max effective wind),
with no midflame cap since the decks set use_wind_limit = false: the head at
R0 (1 + phi_w(U_eff)), the backing fire at R0. One-way variants must match to
TOL; a variant whose name ends in _2way is reported only.
"""
import math, re, sys

TOL = 0.10
M_F = 0.055                       # the decks' fuel moisture (all dead classes)
FT_MIN_TO_M_S = 0.00508
FM1 = dict(w0=0.034, sigma=3500.0, delta=1.0, Mx=0.12, h=8000.0, S_T=0.0555, S_e=0.010, rho_p=32.0)
# Coen et al. 2013, coupled LES: NoWind crept outward at 0.02 m/s on every side (= R0); Control
# ran a 0.22 m/s HEAD (backing not quoted; WRF-Fire sets it to R0); WSHi "four-fifths" faster.
# The one-way heads here are meant to sit below these: the paper's plume doubles the head wind.
COEN = {"nowind": ("NoWind", 0.02), "wind2p5": ("Control head", 0.22), "wind5": ("WSHi head", 0.40),
        "wind2p5_2way": ("Control head", 0.22)}

def rothermel_fm1(M_f, U_eff_ms):
    """Rothermel (1972) as Source/Fire/ERF_Rothermel.cpp computes it: (R0, R, phi_w) in m/s."""
    fp = FM1
    w_n = fp['w0'] * (1 - fp['S_T']); rho_b = fp['w0'] / fp['delta']; beta = rho_b / fp['rho_p']; s = fp['sigma']
    beta_op = 3.348 * s ** -0.8189; s15 = s ** 1.5; Gmax = s15 / (495 + 0.0594 * s15); A = 133 * s ** -0.7913
    br = beta / beta_op; Gp = Gmax * br ** A * math.exp(A * (1 - br))
    rm = min(M_f / fp['Mx'], 1.0); etaM = max(0.0, 1 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3)
    etas = 0.174 * fp['S_e'] ** -0.19
    IR = max(Gp * w_n * fp['h'] * etaM * etas, 0.01)
    xi = math.exp((0.792 + 0.681 * math.sqrt(s)) * (beta + 0.1)) / (192 + 0.2595 * s)
    eps = math.exp(-138 / s); Qig = 250 + 1116 * M_f
    R0 = IR * xi / (rho_b * eps * Qig) * FT_MIN_TO_M_S
    C = 7.47 * math.exp(-0.133 * s ** 0.55); B = 0.02526 * s ** 0.54; E = 0.715 * math.exp(-3.59e-4 * s)
    phi_w = C * (U_eff_ms * 196.85) ** B * br ** -E if U_eff_ms > 0 else 0.0
    return R0, R0 * (1 + phi_w), phi_w

def parse(log):
    probes, ueff, uref = {}, [], []
    for line in open(log):
        m = re.search(r"\[FIRE PROBE\] (\d+) x=([\d.eE+-]+) y=.*arrival_time_s=([\d.eE+-]+)", line)
        if m: probes[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        m = re.search(r"Max effective wind: ([\d.eE+-]+) m/s", line)
        if m: ueff.append(float(m.group(1)))
        m = re.search(r"Max reference wind: ([\d.eE+-]+) m/s", line)
        if m: uref.append(float(m.group(1)))
    return probes, ueff, uref

def rate(pts):
    """Mean rate over consecutive arrived probes, ordered by distance from the line (x = 160 m)."""
    pts = sorted(pts, key=lambda p: abs(p[0] - 160.0))
    rates = [abs(x1 - x0) / (t1 - t0) for (x0, t0), (x1, t1) in zip(pts, pts[1:]) if t1 > t0]
    return (sum(rates) / len(rates), len(rates)) if rates else (float('nan'), 0)

def main():
    variants = sys.argv[1:]
    status = 0
    hdr = f"{'variant':14s} {'U6.1':>6s} {'U_eff':>6s} {'R0':>7s} {'R_head':>7s} | {'back':>7s} {'head':>7s} {'n':>3s} | {'Coen (coupled)':>18s}"
    print(hdr); print("-" * len(hdr))
    for v in variants:
        probes, ueff, uref = parse(f"run_{v}.log")
        U = sum(ueff) / len(ueff) if ueff else 0.0
        Ur = sum(uref) / len(uref) if uref else 0.0
        R0, Rh, _ = rothermel_fm1(M_F, U)
        back = [(x, t) for x, t in probes.values() if x < 140.0]
        head = [(x, t) for x, t in probes.values() if x > 180.0]
        rb, nb = rate(back); rh, nh = rate(head)
        name, rc = COEN.get(v, ("", float('nan')))
        print(f"{v:14s} {Ur:6.2f} {U:6.2f} {R0:7.4f} {Rh:7.4f} | {rb:7.4f} {rh:7.4f} {nb + nh:3d} | {name:>12s} {rc:5.2f}")
        if v.endswith("_2way"): continue
        eb = abs(rb - R0) / R0 if nb else float('inf'); eh = abs(rh - Rh) / Rh if nh else float('inf')
        spread = (max(ueff) - min(ueff)) / U if U > 0 and ueff else 0.0
        ok_b, ok_h = eb < TOL, eh < TOL
        print(f"  backing {rb:.4f} vs R0 {R0:.4f} m/s ({eb * 100:.1f} %, {nb} pairs): {'PASS' if ok_b else 'FAIL'}")
        print(f"  head    {rh:.4f} vs R0(1+phi_w) {Rh:.4f} m/s at U_eff {U:.3f} m/s (range {spread * 100:.1f} %) "
              f"({eh * 100:.1f} %, {nh} pairs): {'PASS' if ok_h else 'FAIL'}")
        status |= (not ok_b) | (not ok_h)
    print("Coen et al. (2013), Table 1 and section 4: FM1 at 5.5 %, 40 m wide 1 km line, coupled LES with a "
          "convective boundary layer; NoWind 0.02 m/s outward, Control (2.5 m/s) 0.22 m/s head, WSHi (5 m/s) "
          "four-fifths faster than Control.")
    sys.exit(status)

if __name__ == "__main__":
    main()
