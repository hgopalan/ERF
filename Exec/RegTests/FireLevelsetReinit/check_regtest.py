#!/usr/bin/env python3
"""FireLevelsetReinit check: does level-set reinitialization improve
front-tracking accuracy over 120s of a short, uncoupled, single-fuel (FM1,
short grass) windy line fire, relative to no reinit at all?

Pass/fail: the reinit deck's head-position error against the theoretical
Rothermel Rf at t=120s must be smaller than the no-reinit deck's.

Theoretical Rf uses ERF's own native Rothermel formula (Albini reaction-
velocity exponent, no WRF-Fire compatibility deflation) -- this does not
depend on WRF-Fire in any way.
"""

import glob, math, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt")

Y_CENTER = 1500.0   # ignition polyline midpoint (line_finite.csv)
# NOT the ignition line's nominal x=500: erf.fire.ignition.polyline_width=25
# means the initial phi=0 front already sits at x=525 at t=0 (confirmed by
# reading the t=0 plotfile directly), so the theoretical head position must
# be anchored to the actual t=0 front, not the input-deck ignition
# coordinate, or the whole comparison is off by a fixed ~25m.
STOP_TIME = 120.0
U = 4.005
FM1 = dict(w0=0.034, sigma=3500.0, delta=1.0, Mx=0.12, h=8000.0, S_T=0.0555, S_e=0.010, rho_p=32.0)
PREFIX = {"noreinit": "plt_noreinit_", "reinit": "plt_reinit_"}


def rothermel_fm1_native(M1, U):
    """ERF's own compiled-in Rothermel path (Albini A, no wrf_bmst_compat)."""
    fp = FM1
    w0 = fp['w0']
    M_f = M1  # FM1 has only w_d1 nonzero, so the weighted dead moisture is M1 exactly
    w_n = w0 * (1 - fp['S_T'])
    rho_b = w0 / fp['delta']
    beta = rho_b / fp['rho_p']
    s = max(fp['sigma'], 100.0)
    beta_op = 3.348 * s ** -0.8189
    s15 = s ** 1.5
    Gmax = s15 / (495.0 + 0.0594 * s15)
    A = 133.0 * s ** -0.7913  # Albini (1976) form -- ERF's default
    br = beta / beta_op
    Gp = Gmax * br ** A * math.exp(A * (1 - br))
    rm = min(M_f / fp['Mx'], 1.0)
    eta_M = max(0.0, 1 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3)
    eta_s = 0.174 * fp['S_e'] ** -0.19
    I_R = max(Gp * w_n * fp['h'] * eta_M * eta_s, 0.01)
    xi = math.exp((0.792 + 0.681 * math.sqrt(s)) * (beta + 0.1)) / (192.0 + 0.2595 * s)
    eps_h = math.exp(-138.0 / s)
    Q_ig = 250.0 + 1116.0 * M_f
    R0_ftmin = (I_R * xi) / (rho_b * eps_h * Q_ig)
    C = 7.47 * math.exp(-0.133 * s ** 0.55)
    B = 0.02526 * s ** 0.54
    E = 0.715 * math.exp(-3.59e-4 * s)
    U_ftmin = U * 196.85
    phi_w = C * (U_ftmin ** B) * (br ** -E)
    R_ftmin = R0_ftmin * (1.0 + phi_w)
    return R_ftmin * 0.00508  # ft/min -> m/s


def load_phi(fname):
    ds = yt.load(fname)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    phi = np.asarray(g[("boxlib", "fire_phi")])[:, :, 0]
    nx, ny = phi.shape
    dx = float((ds.domain_right_edge[0] - ds.domain_left_edge[0]).d) / nx
    xc = float(ds.domain_left_edge[0].d) + (np.arange(nx) + 0.5) * dx
    return xc, phi


def head_position(xc, phi, j):
    row = phi[:, j]
    burned = row < 0.0
    head_cross = np.where(burned[:-1] & ~burned[1:])[0]
    if not len(head_cross):
        return None
    i = head_cross[-1]
    x0, x1 = xc[i], xc[i + 1]
    p0, p1 = row[i], row[i + 1]
    return x0 + (0.0 - p0) * (x1 - x0) / (p1 - p0)


def head_at(fname, y_center=Y_CENTER):
    xc, phi = load_phi(fname)
    ds = yt.load(fname)
    ny = phi.shape[1]
    dy = float((ds.domain_right_edge[1] - ds.domain_left_edge[1]).d) / ny
    j = int((y_center - float(ds.domain_left_edge[1].d)) / dy - 0.5)
    x_head = head_position(xc, phi, j)
    if x_head is None:
        sys.exit(f"'{fname}': no front crossing found on the centerline")
    return x_head


def head_error_pct(prefix, Rf):
    # Named exactly, not "last file in a glob" -- this directory can (and
    # during development, does) accumulate leftover plotfiles from earlier,
    # differently-scoped runs of this same regtest.
    first, last = f"{prefix}00000", f"{prefix}{int(STOP_TIME):05d}"
    if not glob.glob(first) or not glob.glob(last):
        sys.exit(f"'{first}'/'{last}' not found -- run the decks first (run_regtest.sh)")
    x0 = head_at(first)   # the actual t=0 front, not the nominal ignition x
    x_head = head_at(last)
    x_theory = x0 + Rf * STOP_TIME
    return abs(x_head - x_theory) / (Rf * STOP_TIME) * 100.0, x_head, x_theory


def main():
    Rf = rothermel_fm1_native(0.06, U)
    print(f"Theoretical Rf (native ERF Rothermel, FM1, U={U} m/s) = {Rf:.5f} m/s")

    err_noreinit, x_nr, x_th = head_error_pct(PREFIX["noreinit"], Rf)
    print(f"noreinit: head at x={x_nr:.2f} m (theory {x_th:.2f} m), error = {err_noreinit:.3f}%")

    err_reinit, x_re, _ = head_error_pct(PREFIX["reinit"], Rf)
    print(f"reinit:   head at x={x_re:.2f} m (theory {x_th:.2f} m), error = {err_reinit:.3f}%")

    if err_reinit < err_noreinit:
        print(f"PASS: reinit error ({err_reinit:.3f}%) < noreinit error ({err_noreinit:.3f}%)")
    else:
        sys.exit(f"FAIL: reinit error ({err_reinit:.3f}%) >= noreinit error ({err_noreinit:.3f}%)")


if __name__ == "__main__":
    main()
