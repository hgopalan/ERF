#!/usr/bin/env python3
"""Slope_No_Wind: a grass fire on an inclined plane in still air, against Rothermel.

    python3 check_slope_no_wind.py iso_s30 iso_s60 dir_s30 dir_s60 ell_s30 ell_s60

Rothermel (1972) spreads a fire on a slope of tangent s at R0 (1 + phi_s) with
the slope factor phi_s = 5.275 beta^-0.3 s^2; R0 is the no-wind rate of Anderson
fuel model 1 at 5.5 % moisture, from an independent port of the same equations as
Source/Fire/ERF_Rothermel.cpp. The level set projects |grad phi| onto the terrain,
so in map view it solves phi_t + F(n) |grad phi| = 0 with
F(n) = R(n) / sqrt(1 + (s . n)^2).

Each deck is compared with the exact (viscosity) solution of its own equation.
From a disc of radius r0 about c that is the Hopf formula

    T(x) = max over unit n of ((x - c) . n - r0) / F(n),

the time at which the disc grown by t times the Wulff shape
{x : x . n <= F(n) for every n} reaches x, evaluated at the plotfile's cell
centres. The run and T are measured the same way: the time the front first
reaches each column (up and down the slope) or row (across it) against its
distance from the ignition point, from three cells beyond the ignition disc to
20 s before the plotfile. Across the slope that is the growth of the half-width.

Where that span covers 20 cells or more the fitted rate must match the exact
one to 3 %. The backs and flanks at R0 travel only 7 to 9 cells in 1000 s, and
3 % of that is a third of a cell, finer than the level set resolves; there the
run's front must lie within half a cell of the exact front on average and one
cell at worst, the convention of the other Verification cases, and the rate is
reported alongside.

iso_*  directional_ros = false: R = R0 (1 + phi_s) in every direction, so
       R0 (1 + phi_s) / sqrt(1 + s^2) up and down the slope and R0 (1 + phi_s)
       across it. All four at the Hopf solution.

dir_*  the default directional level set, R(n) = R0 (1 + 5.275 beta^-0.3
       max(s . n, 0)^2). Down the slope and across it the projection vanishes, so
       back and flanks spread at R0 (downslope reduced by the ground projection),
       at the Hopf solution.

       The head is not R0 (1 + phi_s). R(n) peaks so sharply about the upslope
       direction that its Wulff shape is not its polar plot: the head is a wedge
       of oblique facets whose tip runs at

           min over n of R(n) / (sqrt(1 + (s . n)^2) n_x),

       about 2 sqrt(phi_s) R0 once phi_s > 1: 0.183 against 0.326 m/s at s = 0.6.
       A straight line fire is not affected (its front keeps n = x), which is why
       FireLineFire meets Rothermel's head rate. The scheme, which freezes R(n)
       from central differences, lands between the two. That is the known defect
       the ellipse decks remove, so the check only requires the head between the
       Wulff tip speed and Rothermel's head rate, and reports where.

ell_*  erf.fire.directional_shape = "ellipse": R(n) is the support function of the
       ellipse with head R_h = R0 (1 + phi_s) upslope and back and flank rates R0,
       c cos(theta) + sqrt(b^2 cos^2(theta) + a^2 sin^2(theta)) with
       b = (R_h + R0)/2, c = (R_h - R0)/2, a = R0. An ellipse is its own Wulff
       shape, so the head runs at Rothermel's head rate. All four directions at
       the Hopf solution, and the head within 3 % of R0 (1 + phi_s) / sqrt(1 + s^2).
"""

import glob, math, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt")

TOL = 0.03
MIN_RATE_CELLS = 20
T_MARGIN = 20.0
SLOPES = {"iso_s30": 0.3, "iso_s60": 0.6, "dir_s30": 0.3, "dir_s60": 0.6, "ell_s30": 0.3, "ell_s60": 0.6}
XS, YS, R_IG = 80.0, 100.0, 10.0
M_F = 0.055
FT_MIN_TO_M_S = 0.00508
FM1 = dict(w0=0.034, sigma=3500.0, delta=1.0, Mx=0.12, h=8000.0, S_T=0.0555, S_e=0.010, rho_p=32.0)
N_ANGLES = 3600
results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {name:24s} {'PASS' if ok else 'FAIL'}  {detail}")


def rothermel_fm1(M_f):
    """Rothermel (1972) no-wind rate and slope-factor constant for fuel model 1: (R0 [m/s], 5.275 beta^-0.3)."""
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
    return R0, 5.275 * beta ** -0.3


def speed(kind, th, R0, phis_c, s):
    """Map-view normal speed F(theta) of each deck, theta measured from upslope (+x)."""
    cs = np.cos(th)
    Rh = R0 * (1 + phis_c * s * s)
    if kind == "iso":
        R = np.full_like(th, Rh)
    elif kind == "dir":
        R = R0 * (1 + phis_c * (s * np.maximum(cs, 0.0)) ** 2)
    else:
        b, c, a = 0.5 * (Rh + R0), 0.5 * (Rh - R0), R0
        R = c * cs + np.sqrt((b * cs) ** 2 + (a * np.sin(th)) ** 2)
    return R / np.sqrt(1 + (s * cs) ** 2)


def hopf_grid(x, y, F, th):
    """Hopf arrival time at every cell centre from the ignition disc for normal speed F(th)."""
    cs, sn = np.cos(th), np.sin(th)
    X, Y = np.meshgrid(x - XS, y - YS, indexing="ij")
    px, py = X.ravel(), Y.ravel()
    T = np.empty(px.size)
    for k in range(0, px.size, 2000):
        num = np.outer(px[k:k + 2000], cs) + np.outer(py[k:k + 2000], sn) - R_IG
        T[k:k + 2000] = np.max(num / F[None, :], axis=1)
    return np.maximum(T, 0.0).reshape(X.shape)


def first_arrival(at, x, y):
    """Distance and time at which the front first reaches each column or row, per direction."""
    a = np.where(at >= 0, at, np.inf)
    tx, ty = a.min(axis=1), a.min(axis=0)
    return {"up the slope (+x)":     (x[x > XS] - XS, tx[x > XS]),
            "down the slope (-x)":   (XS - x[x < XS], tx[x < XS]),
            "across the slope (+y)": (y[y > YS] - YS, ty[y > YS]),
            "across the slope (-y)": (YS - y[y < YS], ty[y < YS])}


def fitted(dist, T, h, t):
    return (dist > R_IG + 3 * h) & np.isfinite(T) & (T < t - T_MARGIN)


def fit_rate(dist, T, h, t):
    m = fitted(dist, T, h, t)
    if m.sum() < 4:
        return float("nan"), int(m.sum())
    return 1.0 / np.polyfit(dist[m], T[m], 1)[0], int(m.sum())


def check_direction(name, run, exact, h, t):
    """The rate to 3 % over MIN_RATE_CELLS cells or more; over a shorter span the
    front within half a cell of the exact one on average and one cell at worst."""
    (d, T), (_, T_ex) = run, exact
    rate, n = fit_rate(d, T, h, t)
    ref, _ = fit_rate(d, T_ex, h, t)
    detail = f"{rate:.5f} m/s from {n} cells vs Hopf {ref:.5f} m/s ({(rate / ref - 1) * 100:+.2f} %)"
    if n >= MIN_RATE_CELLS:
        check(name, np.isfinite(rate) and abs(rate / ref - 1) < TOL, detail)
        return rate
    m = fitted(d, T, h, t)
    err = np.abs(T[m] - T_ex[m]) * ref / h if n >= 4 else np.array([np.inf])
    check(name, bool(err.mean() <= 0.5 and err.max() <= 1.0),
          f"{detail}; front {err.mean():.2f} cells from the exact one on average, {err.max():.2f} at worst")
    return rate


def plotfiles(v):
    return sorted(glob.glob(f"plt_fire_{v}_?????")) or sorted(glob.glob("plt_fire_?????"))


def main():
    R0, phis_c = rothermel_fm1(M_F)
    th = np.linspace(0.0, 2 * np.pi, N_ANGLES, endpoint=False)
    print(f"Rothermel fuel model 1 at {M_F:.3f}: R0 = {R0:.5f} m/s, slope-factor constant 5.275 beta^-0.3 = {phis_c:.3f}")
    for v in (sys.argv[1:] or list(SLOPES)):
        s, kind = SLOPES[v], v[:3]
        head = R0 * (1 + phis_c * s * s) / math.sqrt(1 + s * s)
        u = np.linspace(1e-4, 1.0, 200001)
        wulff = float(np.min(R0 * (1 + phis_c * s * s * u * u) / (u * np.sqrt(1 + s * s * u * u))))
        print(f"{v}: tan(slope) = {s}, phi_s = {phis_c * s * s:.3f}; Rothermel head {head:.5f} m/s in map view, "
              f"Wulff tip of the projection {wulff:.5f} m/s")
        pfs = plotfiles(v)
        if not pfs:
            check("plotfiles", False, "none"); continue
        ds = yt.load(pfs[-1]); t = float(ds.current_time)
        g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        at = np.asarray(g[("boxlib", "fire_arrival_time")])[:, :, 0]
        h = float((ds.domain_right_edge.d[0] - ds.domain_left_edge.d[0]) / ds.domain_dimensions[0])
        nx, ny = at.shape
        x = (np.arange(nx) + 0.5) * h; y = (np.arange(ny) + 0.5) * h
        measured = first_arrival(at, x, y)
        exact = first_arrival(hopf_grid(x, y, speed(kind, th, R0, phis_c, s), th), x, y)
        for name in measured:
            if kind == "dir" and name.startswith("up"):
                rate, n = fit_rate(*measured[name], h, t)
                ref, _ = fit_rate(*exact[name], h, t)
                check(name + " [bracket]", np.isfinite(rate) and wulff * (1 - TOL) <= rate <= head * (1 + TOL),
                      f"{rate:.5f} m/s from {n} cells: Wulff tip {wulff:.5f}, Hopf {ref:.5f}, Rothermel head {head:.5f} m/s "
                      f"({(rate - wulff) / (head - wulff) * 100:.0f} % of the way to Rothermel)")
                continue
            rate = check_direction(name, measured[name], exact[name], h, t)
            if kind == "ell" and name.startswith("up"):
                check("head vs Rothermel", np.isfinite(rate) and abs(rate / head - 1) < TOL,
                      f"{rate:.5f} vs {head:.5f} m/s ({(rate / head - 1) * 100:+.2f} %)")
    n_fail = results.count(False)
    print(f"Slope_No_Wind: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
