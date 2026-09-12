#!/usr/bin/env python3
"""FireWrfWindCoupling: a finite ignition line in a strong uniform wind, testing
erf.fire.directional_wind_coupling ("projection", the default, vs "wrf").

    python3 check_firewrfwindcoupling.py [default projection wrf]

A 1 km ignition line (short of the 3 km periodic y-extent, so the fire
develops real lateral flanks) burns FM1 (short grass) at 6 % moisture in a
uniform 4.005 m/s wind, chosen so Rothermel (1972) gives head rate
Rf = R0(1 + phi_w) = 1.701 m/s with phi_w(B - 1) = 76.8 -- sharply
non-convex. R0, phi_w and B come from an independent port of the equations in
Source/Fire/ERF_Rothermel.cpp (the same port used by FireDirectionalShape's
checker), and the wind bypasses atmospheric interpolation
(erf.fire.prescribed_wind) so all three variants see exactly the same field.

default / projection    phi_w evaluated from the wind already projected onto
                         the front normal, R(n) = R0(1 + phi_w(max(U.n,0))).
                         Non-convex: the level-set head falls from near Rf
                         towards the Wulff-shape tip rate
                         R0 B/(B-1) (phi_w(B-1))^(1/B) as the front develops
                         facets. `default` (the flag left unset) must
                         reproduce `projection` (the flag written out) bit
                         for bit.
wrf                      WRF-Fire's fire_ros: phi_w evaluated from the raw
                         wind, then the whole wind/slope factor scaled by
                         cos(theta) to the front normal, linear in
                         cos(theta) and so convex -- the head tracks Rf.

The checker reads the centerline row (nearest y=1500, the ignition line's
midpoint, farthest from the flank curvature at its ends) from every
`fire_phi` plotfile, interpolates the front's x position (head, +x; back,
-x) at each snapshot, and fits a rate over the second half of the run
(t >= 0.5 stop_time), where the projection deck's front has settled onto its
asymptotic behaviour.
"""

import glob, math, re, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt")

TOL_WRF = 0.03          # wrf head rate must be within 3% of Rf
MAX_FRAC_PROJECTION = 0.80  # projection head rate must be no more than 80% of Rf (measured ~71%)
MIN_CLOSER_MARGIN = 0.15   # wrf must land at least 15 points of Rf-fraction closer than projection
X0, Y_CENTER = 500.0, 1500.0
U = 4.005
M_F = 0.06
FT_MIN_TO_M_S = 0.00508
M_S_TO_FT_MIN = 196.85
FM1 = dict(w0=0.034, sigma=3500.0, delta=1.0, Mx=0.12, h=8000.0, S_T=0.0555, S_e=0.010, rho_p=32.0)
STOP_TIME = 2100.0
PREFIX = {"default": "plt_fire_default_", "projection": "plt_fire_projection_", "wrf": "plt_fire_wrf_"}
results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {name:28s} {'PASS' if ok else 'FAIL'}  {detail}")


def rothermel_fm1(M_f, U, use_wind_limit=False):
    """Rothermel (1972) for fuel model 1: R0, phi_w(U), B [m/s, dimensionless]."""
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
    U_ft = U * M_S_TO_FT_MIN
    if use_wind_limit:
        U_ft = min(U_ft, cap)
    phi_w = C * U_ft ** B * br ** -E
    return R0, phi_w, B


def wulff_tip_rate(R0, phi_w, B):
    """Asymptotic head rate of the projection formula's Wulff shape."""
    return R0 * B / (B - 1) * (phi_w * (B - 1)) ** (1.0 / B)


def times_and_files(prefix):
    out = []
    for f in sorted(glob.glob(f"{prefix}?????")):
        m = re.search(r'_(\d{5})$', f)
        out.append((float(int(m.group(1))), f))
    return sorted(out)


def load_phi(fname):
    ds = yt.load(fname)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    phi = np.asarray(g[("boxlib", "fire_phi")])[:, :, 0]
    nx, ny = phi.shape
    dx = float((ds.domain_right_edge[0] - ds.domain_left_edge[0]).d) / nx
    dy = float((ds.domain_right_edge[1] - ds.domain_left_edge[1]).d) / ny
    xc = float(ds.domain_left_edge[0].d) + (np.arange(nx) + 0.5) * dx
    yc = float(ds.domain_left_edge[1].d) + (np.arange(ny) + 0.5) * dy
    return xc, yc, phi, dx, dy


def front_positions(xc, yc, phi, y_center=Y_CENTER):
    """x position where the centerline row nearest y_center crosses phi=0."""
    j = int(np.argmin(np.abs(yc - y_center)))
    row = phi[:, j]
    burned = row < 0.0
    back_cross = np.where(~burned[:-1] & burned[1:])[0]
    head_cross = np.where(burned[:-1] & ~burned[1:])[0]

    def interp(i):
        x0, x1 = xc[i], xc[i + 1]; p0, p1 = row[i], row[i + 1]
        return x0 + (0.0 - p0) * (x1 - x0) / (p1 - p0)
    xb = interp(back_cross[0]) if len(back_cross) else None
    xh = interp(head_cross[-1]) if len(head_cross) else None
    return xb, xh


def track(prefix):
    ts, xheads, xbacks = [], [], []
    for t, f in times_and_files(prefix):
        xc, yc, phi, dx, dy = load_phi(f)
        xb, xh = front_positions(xc, yc, phi)
        ts.append(t); xheads.append(xh); xbacks.append(xb)
    return np.array(ts), np.array(xheads, dtype=float), np.array(xbacks, dtype=float)


def fit_rate(t, x, t_min):
    m = np.isfinite(x) & (t >= t_min)
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    return np.polyfit(t[m], x[m], 1)[0], int(m.sum())


def main():
    variants = sys.argv[1:] or list(PREFIX)
    R0, phi_w, B = rothermel_fm1(M_F, U, use_wind_limit=False)
    Rf = R0 * (1 + phi_w)
    tip = wulff_tip_rate(R0, phi_w, B)
    print(f"Rothermel FM1 at {M_F:.2f} moisture, U={U} m/s: R0={R0:.5f} m/s, phi_w={phi_w:.3f}, "
          f"B={B:.3f}, phi_w(B-1)={phi_w * (B - 1):.1f}")
    print(f"Rf (head rate) = {Rf:.5f} m/s; Wulff-shape tip of the projection formula = "
          f"{tip:.5f} m/s ({tip / Rf * 100:.1f} % of Rf)")

    tracks = {}
    for v in variants:
        if v not in PREFIX:
            check(v, False, "unknown variant"); continue
        pf = times_and_files(PREFIX[v])
        print(f"{v}:")
        if not pf:
            check("plotfiles", False, "none found"); continue
        t, xh, xb = track(PREFIX[v])
        tracks[v] = dict(t=t, xh=xh, xb=xb)
        check("head reached final plotfile", np.isfinite(xh[-1]),
              f"x_head({t[-1]:.0f}s) = {xh[-1]}")
        if not np.isfinite(xh[-1]):
            continue
        rate, n = fit_rate(t, xh, 0.5 * STOP_TIME)
        frac = rate / Rf
        detail = (f"x_head({t[-1]:.0f}s) = {xh[-1]:.2f} m; late-time rate = {rate:.5f} m/s "
                  f"from {n} snapshots (t >= {0.5 * STOP_TIME:.0f}s), {frac * 100:.1f} % of Rf")
        if v == "wrf":
            check("head rate vs Rf", abs(frac - 1.0) < TOL_WRF, detail)
        elif v in ("default", "projection"):
            check("head rate degraded", tip / Rf - 0.05 < frac < MAX_FRAC_PROJECTION, detail)
        back_rate, nb = fit_rate(t, xb, 0.0)
        print(f"  back rate = {-back_rate:.5f} m/s from {nb} snapshots vs R0 = {R0:.5f} m/s")

    if "default" in tracks and "projection" in tracks:
        # compare every common snapshot's phi field, not just the last
        common = {t: f for t, f in times_and_files(PREFIX["default"])}
        common_p = {t: f for t, f in times_and_files(PREFIX["projection"])}
        shared = sorted(set(common) & set(common_p))
        max_diff = 0.0
        for t in shared:
            _, _, phi_d, _, _ = load_phi(common[t])
            _, _, phi_p, _, _ = load_phi(common_p[t])
            max_diff = max(max_diff, float(np.max(np.abs(phi_d - phi_p))))
        check("default == projection bit for bit", max_diff == 0.0,
              f"max|phi_default - phi_projection| = {max_diff:.3e} over {len(shared)} shared plotfiles")

    if "wrf" in tracks and "projection" in tracks:
        rw, _ = fit_rate(tracks["wrf"]["t"], tracks["wrf"]["xh"], 0.5 * STOP_TIME)
        rp, _ = fit_rate(tracks["projection"]["t"], tracks["projection"]["xh"], 0.5 * STOP_TIME)
        margin = rw / Rf - rp / Rf
        check("wrf closer to Rf than projection", margin > MIN_CLOSER_MARGIN,
              f"wrf {rw / Rf * 100:.1f} % of Rf vs projection {rp / Rf * 100:.1f} % of Rf "
              f"({margin * 100:+.1f} points)")

    n_fail = results.count(False)
    print(f"FireWrfWindCoupling: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
