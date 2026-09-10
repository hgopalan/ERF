#!/usr/bin/env python3
"""Slope_No_Wind: a grass fire on an inclined plane in still air, against Rothermel.

    python3 check_slope_no_wind.py iso_s30 iso_s60 dir_s30 dir_s60

Rothermel (1972) spreads a fire on a slope of tangent s at R0 (1 + phi_s) with
the slope factor phi_s = 5.275 beta^-0.3 s^2; R0 is the no-wind rate of Anderson
fuel model 1 at 5.5 % moisture, from an independent port of the same equations as
Source/Fire/ERF_Rothermel.cpp. The level set projects |grad phi| onto the terrain,
so a direction with slope s_n along it spreads at R / sqrt(1 + s_n^2) in map view.

iso_*  directional_ros = false: the whole front spreads at R0 (1 + phi_s). Along
       the slope (+x and -x) that is R0 (1 + phi_s) / sqrt(1 + s^2) in map view,
       across it (+y and -y) R0 (1 + phi_s). All four are checked to 3 %.

dir_*  the default directional level set evaluates R(n) = R0 (1 + 5.275
       beta^-0.3 max(s . n, 0)^2) along the front normal n. Down the slope and
       across it the projection vanishes, so backing and flanks spread at R0
       (downslope reduced by the ground projection); both are checked to 3 %.

       The head is not R0 (1 + phi_s). For a speed that peaks this sharply
       about one direction the level-set equation phi_t + R(n) |grad phi| = 0
       has, from a point ignition, the Wulff shape as its exact solution: the set
       x . n <= t R(n) / sqrt(1 + (s . n)^2) for every n. Its head is a wedge of
       oblique facets whose tip runs at

           min over n of R(n) / (sqrt(1 + (s . n)^2) n_x),

       about 2 sqrt(phi_s) R0 instead of R0 (1 + phi_s) once phi_s > 1: 0.183
       against 0.326 m/s at s = 0.6. A straight line fire is not affected (its
       front keeps n = x), which is why FireLineFire meets Rothermel's head rate.
       The scheme, which freezes R(n) from central differences, lands between the
       two, and the check requires exactly that, reporting where: the head rate
       must lie between the Wulff tip speed and Rothermel's head rate.

The rates come from a straight-line fit of the arrival time against distance along
each axis, from three cells beyond the ignition disc to the cells that burned 300 s
before the plotfile.
"""

import glob, math, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt")

TOL = 0.03
SLOPES = {"iso_s30": 0.3, "iso_s60": 0.6, "dir_s30": 0.3, "dir_s60": 0.6}
XS, YS, R_IG = 80.0, 100.0, 10.0
M_F = 0.055
FT_MIN_TO_M_S = 0.00508
FM1 = dict(w0=0.034, sigma=3500.0, delta=1.0, Mx=0.12, h=8000.0, S_T=0.0555, S_e=0.010, rho_p=32.0)
results = []

def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {name:22s} {'PASS' if ok else 'FAIL'}  {detail}")

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

def plotfiles(v):
    return sorted(glob.glob(f"plt_fire_{v}_?????")) or sorted(glob.glob("plt_fire_?????"))

def axis_rate(dist, at, h, t):
    m = (dist > R_IG + 3 * h) & (at >= 0) & (at < t - 3 * h / 0.02)
    if m.sum() < 4:
        return float("nan"), int(m.sum())
    slope = np.polyfit(dist[m], at[m], 1)[0]
    return 1.0 / slope, int(m.sum())

def main():
    R0, phis_c = rothermel_fm1(M_F)
    print(f"Rothermel fuel model 1 at {M_F:.3f}: R0 = {R0:.5f} m/s, slope-factor constant 5.275 beta^-0.3 = {phis_c:.3f}")
    for v in (sys.argv[1:] or list(SLOPES)):
        s = SLOPES[v]
        head = R0 * (1 + phis_c * s * s)
        if v.startswith("iso"):
            expected = {"up the slope (+x)": head / math.sqrt(1 + s * s),
                        "down the slope (-x)": head / math.sqrt(1 + s * s),
                        "across the slope (+y)": head, "across the slope (-y)": head}
        else:
            u = np.linspace(1e-4, 1.0, 200001)
            wulff = float(np.min(R0 * (1 + phis_c * s * s * u * u) / (u * np.sqrt(1 + s * s * u * u))))
            expected = {"up the slope (+x)": (wulff, head / math.sqrt(1 + s * s)),
                        "down the slope (-x)": R0 / math.sqrt(1 + s * s),
                        "across the slope (+y)": R0, "across the slope (-y)": R0}
        print(f"{v}: tan(slope) = {s}, phi_s = {phis_c * s * s:.3f}")
        pfs = plotfiles(v)
        if not pfs:
            check("plotfiles", False, "none"); continue
        ds = yt.load(pfs[-1]); t = float(ds.current_time)
        g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        at = np.asarray(g[("boxlib", "fire_arrival_time")])[:, :, 0]
        h = float((ds.domain_right_edge.d[0] - ds.domain_left_edge.d[0]) / ds.domain_dimensions[0])
        nx, ny = at.shape
        x = (np.arange(nx) + 0.5) * h; y = (np.arange(ny) + 0.5) * h
        i0, j0 = int(XS / h), int(YS / h)          # the two rows/columns straddling the centre
        rows = {"up the slope (+x)":   (x[x > XS] - XS, 0.5 * (at[x > XS, j0 - 1] + at[x > XS, j0])),
                "down the slope (-x)": (XS - x[x < XS], 0.5 * (at[x < XS, j0 - 1] + at[x < XS, j0])),
                "across the slope (+y)": (y[y > YS] - YS, 0.5 * (at[i0 - 1, y > YS] + at[i0, y > YS])),
                "across the slope (-y)": (YS - y[y < YS], 0.5 * (at[i0 - 1, y < YS] + at[i0, y < YS]))}
        for name, (d, a) in rows.items():
            rate, n = axis_rate(d, a, h, t)
            ex = expected[name]
            if isinstance(ex, tuple):
                lo, hi = ex
                check(name + " [bracket]", np.isfinite(rate) and lo * (1 - TOL) <= rate <= hi * (1 + TOL),
                      f"{rate:.5f} m/s from {n} cells: Wulff tip {lo:.5f}, Rothermel head {hi:.5f} m/s "
                      f"({(rate - lo) / (hi - lo) * 100:.0f} % of the way to Rothermel)")
            else:
                check(name, np.isfinite(rate) and abs(rate / ex - 1) < TOL,
                      f"{rate:.5f} m/s from {n} cells vs {ex:.5f} m/s ({(rate / ex - 1) * 100:+.2f} %)")
    n_fail = results.count(False)
    print(f"Slope_No_Wind: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
    main()
