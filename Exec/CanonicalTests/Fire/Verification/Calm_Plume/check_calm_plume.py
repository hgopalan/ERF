#!/usr/bin/env python3
"""Calm_Plume: the plume over a steady prescribed heat source, against its heat budget and plume theory.

    python3 check_calm_plume.py neutral stable

A disc of radius 120 m releases q = 1.1e3 W/m2 (about 50 MW) into still air, with
an adiabatic ground and walls that conserve heat.

Checked (both decks): the heat budget. The coupling places 1 - exp(-H/alfg) of
the source power P in a column of height H (essentially all of it at 1200 m), so
Cp times the change of the integral of rho theta must equal the placed power times
the heated time, one atmospheric step shorter since the flux is lagged; to 1.5 %.

Reported, not checked: plume theory. Far above a fire the time-mean centreline
excess temperature and vertical velocity follow Heskestad's correlations

    dT0 = 9.1 (T/(g cp^2 rho^2))^(1/3) Qc^(2/3) (z - z0)^(-5/3)
    w0  = 3.4 (g/(cp rho T))^(1/3)    Qc^(1/3) (z - z0)^(-1/3)

with the virtual origin z0 = 0.083 Q^(2/5) - 1.02 D (Q in kW, D the source
diameter), and in stratification N a plume of buoyancy flux F = g P/(pi cp rho T)
rises to about 5.0 F^(1/4) N^(-3/4) above that origin and spreads at about 0.76 of
it (Morton, Taylor and Turner 1956; Briggs). The script prints, from fields
averaged over 400-1000 s (neutral) and after 700 s (stable, more than one buoyancy
period 2 pi/N = 628 s), the centreline ratios at 260-540 m, the axis velocity
profile, the level where the axis becomes neutrally buoyant, the level of strongest
radial outflow 200-600 m from the axis, and the height where the mean axis velocity
reaches zero.

At 40 m cells the 240 m source is six cells across and wide against its height:
the plume is under-entrained and still in its near field, its axis velocity grows
with height where the far-field law has it fall, and the stable plume overshoots
the theoretical top by about a factor of two. Those comparisons therefore stay
reports until a run with a smaller source on 10-20 m cells can resolve the
entrainment.
"""
import glob, math, re, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt")

G, CP, T0 = 9.81, 1004.5, 300.0
Q, XC, YC, RAD = 1.1e3, 800.0, 800.0, 120.0
ALFG, H_FIRE = 45.0, 20.0
N_STABLE = 0.01
results = []

def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {name:32s} {'PASS' if ok else 'FAIL'}  {detail}")

def load(pf, names):
    ds = yt.load(pf)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    dx = (ds.domain_right_edge.d - ds.domain_left_edge.d) / ds.domain_dimensions
    return float(ds.current_time), {n: np.asarray(g[("boxlib", n)]) for n in names}, dx

def disc_power():
    xc = (np.arange(int(1600 / H_FIRE)) + 0.5) * H_FIRE
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    return Q * np.count_nonzero(np.hypot(X - XC, Y - YC) <= RAD) * H_FIRE * H_FIRE

def main():
    P = disc_power()
    Q_kw, D = P / 1000.0, 2 * RAD
    z0 = 0.083 * Q_kw ** 0.4 - 1.02 * D
    print(f"source power P = {P / 1e6:.2f} MW over D = {D:g} m; Heskestad virtual origin z0 = {z0:.1f} m")
    for v in (sys.argv[1:] or ["neutral", "stable"]):
        pfs = sorted(glob.glob(f"plt_atm_{v}_?????")) or sorted(glob.glob("plt_atm_?????"))
        print(f"{v}: {len(pfs)} plotfiles")
        if len(pfs) < 3:
            check("plotfiles", False, "need at least three"); continue
        t0, f0, dx = load(pfs[0], ["density", "theta"])
        dV = dx[0] * dx[1] * dx[2]
        H = dx[2] * f0["theta"].shape[2]
        placed = 1.0 - math.exp(-H / ALFG)
        rt0 = (f0["density"] * f0["theta"]).sum()
        th_init = f0["theta"].mean(axis=(0, 1))
        rho_init = f0["density"].mean(axis=(0, 1))
        z = (np.arange(f0["theta"].shape[2]) + 0.5) * dx[2]
        log = glob.glob(f"run_{v}.log")
        dt_first = 0.0
        if log:
            m = re.search(r"STEP 1 ends\. TIME = [\d.eE+-]+ DT = ([\d.eE+-]+)", open(log[0]).read())
            dt_first = float(m.group(1)) if m else 0.0
        worst = 0.0
        times, fields = [], []
        for pf in pfs[1:]:
            t, f, _ = load(pf, ["density", "theta", "x_velocity", "y_velocity", "z_velocity"])
            heat = CP * ((f["density"] * f["theta"]).sum() - rt0) * dV
            expected = placed * P * (t - t0 - dt_first)
            worst = max(worst, abs(heat / expected - 1.0))
            times.append(t); fields.append(f)
        check("heat budget", worst < 0.015,
              f"Cp d(int rho theta) against the placed power times time at {len(times)} plotfiles: worst {worst * 100:.2f} %")
        ic, jc = int(XC / dx[0]), int(YC / dx[1])
        if v == "neutral":
            sel = [k for k, t in enumerate(times) if 400.0 <= t <= 1000.0]
            if len(sel) < 3:
                check("averaging window", False, f"{len(sel)} plotfiles in 400-1000 s"); continue
            w = np.mean([fields[k]["z_velocity"][ic - 1:ic + 1, jc - 1:jc + 1, :].mean(axis=(0, 1)) for k in sel], axis=0)
            dth = np.mean([fields[k]["theta"][ic - 1:ic + 1, jc - 1:jc + 1, :].mean(axis=(0, 1)) for k in sel], axis=0) - th_init
            print("  reported, not checked (Heskestad far field; see the docstring):")
            for zz in (250.0, 350.0, 450.0, 550.0):
                kk = int(zz / dx[2]); rho = rho_init[kk]; zr = z[kk] - z0
                dT_h = 9.1 * (T0 / (G * CP * CP * rho * rho)) ** (1 / 3) * P ** (2 / 3) * zr ** (-5 / 3)
                w_h = 3.4 * (G / (CP * rho * T0)) ** (1 / 3) * P ** (1 / 3) * zr ** (-1 / 3)
                print(f"    centreline at {z[kk]:.0f} m: dT {dth[kk]:.3f} K vs {dT_h:.3f} K (x{dth[kk] / dT_h:.2f}); "
                      f"w {w[kk]:.2f} m/s vs {w_h:.2f} m/s (x{w[kk] / w_h:.2f})")
            k_top = int(np.argmax(w[:int(1000.0 / dx[2])]))
            print(f"    axis w {w[int(260.0 / dx[2])]:.2f} m/s at 260 m rising to {w[k_top]:.2f} m/s at {z[k_top]:.0f} m "
                  f"(the far-field law has it falling as (z - z0)^(-1/3))")
        else:
            sel = [k for k, t in enumerate(times) if t >= 700.0]
            if len(sel) < 3:
                check("averaging window", False, f"{len(sel)} plotfiles after 700 s"); continue
            wbar = np.mean([fields[k]["z_velocity"][ic - 1:ic + 1, jc - 1:jc + 1, :].mean(axis=(0, 1)) for k in sel], axis=0)
            k_w = int(np.argmax(wbar))
            nx, ny = f0["theta"].shape[:2]
            xc = (np.arange(nx) + 0.5) * dx[0] - XC
            yc = (np.arange(ny) + 0.5) * dx[1] - YC
            XX, YY = np.meshgrid(xc, yc, indexing="ij")
            RR = np.hypot(XX, YY)
            ring = (RR >= 200.0) & (RR <= 600.0)
            ur = np.mean([((fields[k]["x_velocity"] * (XX / np.maximum(RR, 1.0))[:, :, None]
                            + fields[k]["y_velocity"] * (YY / np.maximum(RR, 1.0))[:, :, None])[ring]).mean(axis=0)
                          for k in sel], axis=0)
            k_pk = int(np.argmax(ur))
            if 0 < k_pk < len(ur) - 1:
                a, b, c = ur[k_pk - 1], ur[k_pk], ur[k_pk + 1]
                z_spread = z[k_pk] + 0.5 * dx[2] * (a - c) / (a - 2 * b + c)
            else:
                z_spread = z[k_pk]
            print(f"  peak mean axis w {wbar[k_w]:.2f} m/s at {z[k_w]:.0f} m; peak mean radial outflow {ur[k_pk]:.3f} m/s")
            F = G * P / (math.pi * CP * rho_init[0] * T0)
            z_max = 5.0 * F ** 0.25 * N_STABLE ** -0.75 + z0
            z_n = 0.76 * 5.0 * F ** 0.25 * N_STABLE ** -0.75 + z0
            thb = np.mean([fields[k]["theta"][ic - 1:ic + 1, jc - 1:jc + 1, :].mean(axis=(0, 1)) for k in sel], axis=0) - th_init
            k_nb = next((k for k in range(int(np.argmax(thb)), len(thb)) if thb[k] <= 0.0), None)
            z_nb = z[k_nb] if k_nb is not None else float("nan")
            k_w0 = next((k for k in range(k_w, len(wbar)) if wbar[k] <= 0.0), None)
            z_w0 = z[k_w0] if k_w0 is not None else float("nan")
            print("  reported, not checked (Morton-Taylor-Turner and Briggs; see the docstring):")
            print(f"    F = {F:.1f} m4/s3; 5.0 F^(1/4) N^(-3/4) = {5.0 * F ** 0.25 * N_STABLE ** -0.75:.0f} m above the virtual origin z0 = {z0:.0f} m")
            print(f"    spreading: strongest radial outflow at {z_spread:.0f} m vs {z_n:.0f} m (x{z_spread / z_n:.2f}); "
                  f"axis neutrally buoyant at {z_nb:.0f} m")
            print(f"    top: mean axis w reaches zero at {z_w0:.0f} m vs {z_max:.0f} m (x{z_w0 / z_max:.2f}); "
                  f"the axis is up to {-thb[k_nb:k_w0].min() if k_nb is not None and k_w0 is not None and k_w0 > k_nb else float('nan'):.2f} K colder than its surroundings above neutral buoyancy")
    n_fail = results.count(False)
    print(f"Calm_Plume: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
    main()
