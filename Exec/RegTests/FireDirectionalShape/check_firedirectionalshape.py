#!/usr/bin/env python3
"""FireDirectionalShape: a point ignition in a uniform wind, against Rothermel.

    python3 check_firedirectionalshape.py [isotropic projection projection_key ellipse]

A grass fire (Anderson fuel model 1 at 5.5 % moisture) spreads from a 10 m disc
on flat ground in a uniform westerly. Rothermel (1972) spreads its head at
R0 (1 + phi_w), phi_w = C U^B (beta/beta_op)^-E with U the midflame wind in
ft/min under the 300 ft/min cap the code applies to fine fuels, and R0 the
no-wind rate; both come from an independent port of the equations in
Source/Fire/ERF_Rothermel.cpp. The wind is read from the fire plotfiles, which
must show it uniform and steady to 1 %.

Each deck is compared with the exact (viscosity) solution of its own equation,
phi_t + F(n) |grad phi| = 0. From a disc of radius r0 about c that is the Hopf
formula

    T(x) = max over unit n of ((x - c) . n - r0) / F(n),

the time at which the disc grown by t times the Wulff shape
{x : x . n <= F(n) for every n} reaches x, evaluated at the plotfile's cell
centres. The run and T are measured the same way: the time the front first
reaches each column (head, back) or row (flanks) against its distance from the
ignition point, from three cells beyond the ignition disc to 20 s before the
last plotfile. For the flanks that is the growth of the half-width.

Where that span covers 20 cells or more the fitted rate must match the exact
one to 3 %. Backs and flanks at R0 travel only about 15 cells in 1500 s, and 3 %
of that is half a cell, about what the level set resolves; there the run's front
must lie within half a cell of the exact front on average and one cell at worst,
the convention of the Verification cases, and the rate is reported alongside.

isotropic       directional_ros = false: F = R0 (1 + phi_w), a disc at the head
                rate. Head, back and flanks at the Hopf solution.
projection      the default: F(n) = R0 (1 + phi_w(max(U . n, 0))). With
                phi_w (B - 1) > 1 this peaks so sharply that the Wulff shape is
                not the polar plot of F: the head is a wedge of oblique facets
                whose tip runs at min over n of F(n) / n_x, which is
                R0 B/(B - 1) (phi_w (B - 1))^(1/B), well below Rothermel's head
                rate. The scheme, which freezes F(n) from central differences,
                lands between the two, so the head is required to lie in that
                bracket (the check reports where); back and flanks at the Hopf
                solution.
projection_key  the same deck with erf.fire.directional_shape = "projection"
                written out; its arrival times must equal the projection deck's
                bit for bit.
ellipse         erf.fire.directional_shape = "ellipse": F is the support function
                of the ellipse with head R_h = R0 (1 + phi_w) along the wind and
                back and flank rates R0,
                    F(theta) = c cos(theta) + sqrt(b^2 cos^2(theta) + a^2 sin^2(theta)),
                    b = (R_h + R0) / 2,  c = (R_h - R0) / 2,  a = R0.
                An ellipse is its own Wulff shape, so the head runs at Rothermel's
                head rate. Head, back and flanks at the Hopf solution, and the head
                within 3 % of Rothermel's head rate.
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
XS, YS, R_IG = 100.0, 100.0, 10.0
M_F = 0.055
FT_MIN_TO_M_S = 0.00508
M_S_TO_FT_MIN = 196.85
FM1 = dict(w0=0.034, sigma=3500.0, delta=1.0, Mx=0.12, h=8000.0, S_T=0.0555, S_e=0.010, rho_p=32.0)
KIND = {"isotropic": "iso", "projection": "proj", "projection_key": "proj", "ellipse": "ell"}
N_ANGLES = 3600
results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {name:24s} {'PASS' if ok else 'FAIL'}  {detail}")


def rothermel_fm1(M_f):
    """Rothermel (1972) for fuel model 1: R0 [m/s], phi_w(U [m/s]) under the fine-fuel wind cap, B, and the cap [m/s]."""
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
    cap = 300.0 if s > 1000.0 else 500.0

    def phi_w(U):
        U_ft = np.minimum(np.asarray(U, dtype=float) * M_S_TO_FT_MIN, cap)
        return C * U_ft ** B * br ** -E
    return R0, phi_w, B, cap / M_S_TO_FT_MIN


def speed(kind, th, R0, phi_w, U):
    """Normal speed F(theta) of each deck, theta measured from the wind."""
    Rh = R0 * (1 + float(phi_w(U)))
    if kind == "iso":
        return np.full_like(th, Rh)
    if kind == "proj":
        return R0 * (1 + phi_w(U * np.maximum(np.cos(th), 0.0)))
    b, c, a = 0.5 * (Rh + R0), 0.5 * (Rh - R0), R0
    return c * np.cos(th) + np.sqrt((b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2)


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
    return {"head (+x)":  (x[x > XS] - XS, tx[x > XS]),
            "back (-x)":  (XS - x[x < XS], tx[x < XS]),
            "flank (+y)": (y[y > YS] - YS, ty[y > YS]),
            "flank (-y)": (YS - y[y < YS], ty[y < YS])}


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


def load(pf):
    ds = yt.load(pf)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    fields = {f: np.asarray(g[("boxlib", f)])[:, :, 0]
              for f in ("fire_arrival_time", "fire_wind_eff_u", "fire_wind_eff_v")}
    h = float((ds.domain_right_edge.d[0] - ds.domain_left_edge.d[0]) / ds.domain_dimensions[0])
    return float(ds.current_time), h, fields


def main():
    R0, phi_w, B, U_cap = rothermel_fm1(M_F)
    th = np.linspace(0.0, 2 * np.pi, N_ANGLES, endpoint=False)
    print(f"Rothermel fuel model 1 at {M_F:.3f}: R0 = {R0:.5f} m/s, B = {B:.3f}, midflame wind cap {U_cap:.3f} m/s")
    arrival = {}
    for v in (sys.argv[1:] or list(KIND)):
        # variants run with extra arguments may carry a suffix, e.g. projection_upwind
        kind = KIND.get(v) or next(k for p, k in KIND.items() if v.startswith(p + "_"))
        pfs = plotfiles(v)
        print(f"{v}:")
        if not pfs:
            check("plotfiles", False, "none"); continue
        t, h, f = load(pfs[-1])
        # the step-0 plotfile is written before the fire first samples the wind
        _, _, f0 = load(pfs[1] if len(pfs) > 2 else pfs[0])
        at = f["fire_arrival_time"]
        arrival[v] = at
        u, w = f["fire_wind_eff_u"], f["fire_wind_eff_v"]
        U = float(u.mean())
        dev = max(float(np.abs(u - U).max()), float(np.abs(w).max()),
                  float(np.abs(f0["fire_wind_eff_u"] - U).max()))
        check("wind uniform, steady", dev < 0.01 * U,
              f"U = {U:.4f} m/s, largest departure {dev:.2e} m/s over {len(pfs)} plotfiles")
        phw = float(phi_w(U)); head = R0 * (1 + phw)
        u01 = np.linspace(1e-4, 1.0, 200001)
        wulff = float(np.min(R0 * (1 + phi_w(U * u01)) / u01))
        print(f"  phi_w = {phw:.3f}, phi_w (B - 1) = {phw * (B - 1):.3f}; Rothermel head {head:.5f} m/s, "
              f"Wulff tip of the projection {wulff:.5f} m/s ({wulff / head * 100:.0f} % of it)")

        nx, ny = at.shape
        x = (np.arange(nx) + 0.5) * h; y = (np.arange(ny) + 0.5) * h
        measured = first_arrival(at, x, y)
        exact = first_arrival(hopf_grid(x, y, speed(kind, th, R0, phi_w, U), th), x, y)
        for name in measured:
            if kind == "proj" and name.startswith("head"):
                rate, n = fit_rate(*measured[name], h, t)
                ref, _ = fit_rate(*exact[name], h, t)
                check(name + " [bracket]", np.isfinite(rate) and wulff * (1 - TOL) <= rate <= head * (1 + TOL),
                      f"{rate:.5f} m/s from {n} cells: Wulff tip {wulff:.5f}, Hopf {ref:.5f}, Rothermel head {head:.5f} m/s "
                      f"({(rate - wulff) / (head - wulff) * 100:.0f} % of the way to Rothermel)")
                continue
            rate = check_direction(name, measured[name], exact[name], h, t)
            if kind == "ell" and name.startswith("head"):
                check("head vs Rothermel", np.isfinite(rate) and abs(rate / head - 1) < TOL,
                      f"{rate:.5f} vs {head:.5f} m/s ({(rate / head - 1) * 100:+.2f} %)")
    if "projection" in arrival and "projection_key" in arrival:
        check("key reproduces default", np.array_equal(arrival["projection"], arrival["projection_key"]),
              "projection_key arrival times equal projection's bit for bit")
    n_fail = results.count(False)
    print(f"FireDirectionalShape: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
