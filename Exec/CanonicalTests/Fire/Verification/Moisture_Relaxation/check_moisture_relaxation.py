#!/usr/bin/env python3
"""Moisture_Relaxation: dead fuel drying in dry air, against the time-lag solution.

    python3 check_moisture_relaxation.py drydown

Each dead class relaxes towards the equilibrium moisture M_e of the air,

    dM/dt = (M_e - M) / tau_eff,   tau_eff = tau f(T),   f(T) = exp(-0.015 (T - 20 C)),

with tau = 1, 10 and 100 hours (Nelson 2000). In still, dry air M_e and T are
constant. M_e comes from Nelson's adsorption (wetting) and desorption (drying)
polynomials in the relative humidity, the curve chosen from the fuel's current
moisture, and the relative humidity of air without a moisture model is zero,
clamped to 1 %: E_w = 0.0351 and E_d = 0.0600.

As implemented (compute_emc_with_hysteresis in ERF_FuelMoisture.H) the choice is
the reverse of Nelson's: fuel above the adsorption curve relaxes towards it, and
fuel below the desorption curve towards that, so a fuel drying from 20 % heads for
E_w and stops when it meets E_d,

    M(t) = max(E_w + (M0 - E_w) exp(-t / tau_eff), E_d),

where Nelson's model would give E_d + (M0 - E_d) exp(-t / tau_eff), 0.106 instead
of 0.090 after an hour. The check follows the code, and prints Nelson's curve
alongside; the model advances the equation with forward Euler at the atmospheric
step, which the check also reproduces step for step.

Rothermel is rebuilt from the moisture every step. Fuel model 1 carries only
1-hour fuel, so its no-wind rate follows M_1h(t) alone: it is zero until M_1h
falls below the 12 % moisture of extinction, at

    t_x = -tau_eff ln((0.12 - M_e) / (M0 - M_e)),

and the ignition disc then grows at R0(M_1h(t)).

The checks at every plotfile: each class against the stepwise solution (to
round-off) and against the exponential; the rate of spread against Rothermel at
the 1-hour moisture, from an independent port of ERF_Rothermel.cpp; and the burned
radius against r_ig plus the integral of that rate, to half a cell.
"""
import glob, math, re, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt")

R_IG = 8.0
TAU_H = (1.0, 10.0, 100.0)
M_MIN, M_MAX = 0.01, 0.40
FT_MIN_TO_M_S = 0.00508
FM1 = dict(w0=0.034, sigma=3500.0, delta=1.0, Mx=0.12, h=8000.0, S_T=0.0555, S_e=0.010, rho_p=32.0)
results = []

def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {name:30s} {'PASS' if ok else 'FAIL'}  {detail}")

def emc_ads(RH):
    H = min(max(RH / 100.0, 0.01), 0.99)
    return max(0.0, min(0.03229 + 0.2810 * H + 0.4093 * H**2 - 1.3560 * H**3 + 1.6596 * H**4, 0.35))

def emc_des(RH):
    H = min(max(RH / 100.0, 0.01), 0.99)
    return max(0.0, min(0.05800 + 0.1985 * H + 0.6250 * H**2 - 1.1830 * H**3 + 1.0570 * H**4, 0.35))

def emc(RH, M):
    a, d = emc_ads(RH), emc_des(RH)
    return d if M < d else (a if M > a else 0.5 * (a + d))

def temp_factor(T_C):
    return min(max(math.exp(-0.015 * (T_C - 20.0)), 0.5), 2.0)

def euler(M, RH, T_C, dt_h, tau_h):
    M = min(max(M, M_MIN), M_MAX)
    Me = emc(RH, M)
    M = M + dt_h * (Me - M) / max(tau_h * temp_factor(T_C), 0.1)
    return min(max(M, M_MIN), M_MAX)

def rothermel_R0(M_f):
    fp = FM1
    w_n = fp['w0'] * (1 - fp['S_T']); rho_b = fp['w0'] / fp['delta']; beta = rho_b / fp['rho_p']; s = fp['sigma']
    beta_op = 3.348 * s ** -0.8189; s15 = s ** 1.5; Gmax = s15 / (495 + 0.0594 * s15); A = 133 * s ** -0.7913
    br = beta / beta_op; Gp = Gmax * br ** A * math.exp(A * (1 - br))
    rm = min(M_f / fp['Mx'], 1.0); etaM = max(0.0, 1 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3)
    etas = 0.174 * fp['S_e'] ** -0.19
    IR = max(Gp * w_n * fp['h'] * etaM * etas, 0.01)
    xi = math.exp((0.792 + 0.681 * math.sqrt(s)) * (beta + 0.1)) / (192 + 0.2595 * s)
    eps = math.exp(-138 / s); Qig = 250 + 1116 * M_f
    return IR * xi / (rho_b * eps * Qig) * FT_MIN_TO_M_S

def main():
    deck = open("inputs_drydown").read()
    num = lambda k: float(re.search(rf"^{re.escape(k)}\s*=\s*([\d.eE+-]+)", deck, re.M).group(1))
    dt, theta, M0 = num("erf.fixed_dt"), num("erf.theta_ref"), num("erf.fire.moisture_1hr")
    T_C = theta - 273.15      # the fire takes the k = 0 potential temperature as the air temperature
    for v in (sys.argv[1:] or ["drydown"]):
        pfs = sorted(glob.glob(f"plt_fire_{v}_?????")) or sorted(glob.glob("plt_fire_?????"))
        if len(pfs) < 2:
            check("plotfiles", False, f"found {len(pfs)}"); continue
        ds = yt.load(pfs[0]); g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        RH = 100.0 * float(np.asarray(g[("boxlib", "fire_surface_rh")]).mean())
        Me = [emc(RH, M0)] * 3
        E_d = emc_des(RH)
        tau_eff = [t_h * temp_factor(T_C) * 3600.0 for t_h in TAU_H]
        t_x = -tau_eff[0] * math.log((FM1["Mx"] - Me[0]) / (M0 - Me[0]))
        print(f"{v}: RH {RH:.2f} %, T {T_C:.2f} C, M_e {Me[0]:.4f}, tau_eff {tau_eff[0] / 3600:.4f} h, "
              f"1-hour class below extinction at t_x = {t_x:.0f} s")
        # stepwise reference, and the burned radius from the rate it gives
        n_max = int(round(float(yt.load(pfs[-1]).current_time) / dt))
        M = [[M0] * 3]; radius = [R_IG]
        for n in range(n_max):
            M.append([euler(M[-1][c], RH, T_C, dt / 3600.0, TAU_H[c]) for c in range(3)])
            radius.append(radius[-1] + rothermel_R0(M[-1][0]) * dt)
        for pf in pfs[1:]:
            ds = yt.load(pf); t = float(ds.current_time); n = int(round(t / dt))
            g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
            h = float((ds.domain_right_edge.d[0] - ds.domain_left_edge.d[0]) / ds.domain_dimensions[0])
            print(f" t = {t:6.0f} s")
            worst_step, worst_exp = 0.0, 0.0
            for c, name in enumerate(("fire_fuel_mc_1hr", "fire_fuel_mc_10hr", "fire_fuel_mc_100hr")):
                f = np.asarray(g[("boxlib", name)])
                exact = max(Me[c] + (M0 - Me[c]) * math.exp(-t / tau_eff[c]), E_d)
                worst_step = max(worst_step, float(np.abs(f - M[n][c]).max()))
                worst_exp = max(worst_exp, float(np.abs(f - exact).max()))
            check("moisture: stepwise solution", worst_step < 1e-9,
                  f"max |M - Euler| {worst_step:.1e} over the three classes (1-hour {M[n][0]:.5f})")
            check("moisture: closed form", worst_exp < 2e-4,
                  f"max |M - max(E_w + (M0 - E_w) exp(-t/tau), E_d)| {worst_exp:.1e}; "
                  f"Nelson's hysteresis would give {E_d + (M0 - E_d) * math.exp(-t / tau_eff[0]):.5f} for the 1-hour class")
            ros = float(np.asarray(g[("boxlib", "fire_ros")]).max())
            R0 = rothermel_R0(M[n][0])
            check("rate of spread", abs(ros - R0) <= 1e-6 * max(R0, 1e-3),
                  f"{ros:.6f} m/s vs Rothermel at M_1h: {R0:.6f} m/s" + ("  (above extinction)" if M[n][0] >= FM1["Mx"] else ""))
            phi = np.asarray(g[("boxlib", "fire_phi")])[:, :, 0]
            r_num = math.sqrt(np.clip(0.5 - phi / h, 0.0, 1.0).sum() * h * h / math.pi)
            check("burned radius", abs(r_num - radius[n]) <= 0.5 * h,
                  f"{r_num:.2f} m vs r_ig + integral of R0 dt = {radius[n]:.2f} m ({(r_num - radius[n]) / h:+.2f} cells)")
    n_fail = results.count(False)
    print(f"Moisture_Relaxation: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
    main()
