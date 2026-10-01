#!/usr/bin/env python3
"""FireSplitHamiltonian: rotation invariance of the advective level-set scheme,
with and without erf.fire.directional_split_hamiltonian.

    python3 check_firesplithamiltonian.py

A 1 km ignition line burns FM1 (short grass) at 6 % moisture in a uniform
4.005 m/s wind (Rothermel head rate Rf = 1.701 m/s), with Jiang-Peng
reinitialization every level-set substep. The scenario runs twice per scheme:
wind along +x, and the same scenario rigidly rotated by 34 deg about the domain
centre (wind, ignition line and standoff all rotated). A scheme that respects
the rotation gives the same burned region in both after un-rotating; the fire
is far from the periodic edges, so the domain's square grid is the only thing
that breaks the symmetry.

Two measurements per scheme (the mismatch at every saved time from 300 s on, and
the baseline's required to exceed its bound only once the wing has formed, from
600 s):

  mismatch   Area where the 0 deg burned region (phi < 0) and the un-rotated
             34 deg burned region disagree, as a fraction of the 0 deg burned
             area. The 34 deg field is sampled bilinearly at each 0 deg cell
             centre rotated about the domain centre.
  head rate  Speed of the front along the wind ray from the ignition line's
             midpoint, fitted over t >= 600 s, against Rothermel's Rf from an
             independent port of the equations in Source/Fire/ERF_Rothermel.cpp.

The baseline scheme (R(n) from an estimated front normal times one Godunov
|grad phi|) grows a wing at the oblique angle; the split Hamiltonian does not.
"""

import functools, glob, math, re, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
    from scipy.ndimage import map_coordinates
except ImportError:
    sys.exit("needs numpy, scipy and yt")

DT = 2.0                      # erf.fixed_dt
CENTRE = 2400.0               # domain centre, x and y [m]
STANDOFF = 1200.0             # ignition line midpoint, upwind of the centre [m]
ANGLE = 34.0                  # rotation of the oblique case [deg]
U = 4.005
M_F = 0.06
T_MISMATCH_MIN = 300.0        # first time the mismatch is checked [s]
T_FIT_MIN = 600.0             # head rate is fitted over t >= this [s]

MAX_MISMATCH_SPLIT = 0.01     # split: rotated and native footprints agree to 1 % (measured ~0.1 %)
MIN_MISMATCH_BASELINE = 0.05  # baseline: at least 5 % disagreement once the wing has formed, t >= T_FIT_MIN (measured 9-13 %)
MIN_MISMATCH_RATIO = 10.0     # baseline mismatch at least 10x the split's
TOL_HEAD = 0.03               # split head rate within 3 % of Rf

FT_MIN_TO_M_S = 0.00508
M_S_TO_FT_MIN = 196.85
FM1 = dict(w0=0.034, sigma=3500.0, delta=1.0, Mx=0.12, h=8000.0, S_T=0.0555, S_e=0.010, rho_p=32.0)
results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {name:44s} {'PASS' if ok else 'FAIL'}  {detail}")


def rothermel_fm1(M_f, U):
    """Rothermel (1972) for fuel model 1, no wind cap: R0 [m/s], phi_w(U)."""
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
    C = 7.47 * math.exp(-0.133 * s ** 0.55); E = 0.715 * math.exp(-3.59e-4 * s)
    phi_w = C * (U * M_S_TO_FT_MIN) ** (0.02526 * s ** 0.54) * br ** -E
    return R0, phi_w


def snapshots(prefix):
    out = []
    for f in sorted(glob.glob(f"{prefix}?????")):
        out.append((int(re.search(r'_(\d{5})$', f).group(1)) * DT, f))
    return sorted(out)


@functools.lru_cache(maxsize=None)
def load_phi(fname):
    ds = yt.load(fname)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    phi = np.asarray(g[("boxlib", "fire_phi")])[:, :, 0]
    return phi, float((ds.domain_right_edge[0] - ds.domain_left_edge[0]).d) / phi.shape[0]


def mismatch(phi0, phi34, dx):
    """Fraction of the 0 deg burned area where the un-rotated 34 deg region disagrees."""
    n = phi0.shape[0]
    xc = (np.arange(n) + 0.5) * dx
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    th = math.radians(ANGLE)
    Xr = CENTRE + (X - CENTRE) * math.cos(th) - (Y - CENTRE) * math.sin(th)
    Yr = CENTRE + (X - CENTRE) * math.sin(th) + (Y - CENTRE) * math.cos(th)
    q = map_coordinates(phi34, [Xr / dx - 0.5, Yr / dx - 0.5], order=1, mode="nearest")
    b0, b34 = phi0 < 0.0, q < 0.0
    return (b0 ^ b34).sum() / b0.sum()


def head_distance(phi, dx, angle_deg):
    """Distance from the ignition line midpoint to the front along the wind ray."""
    th = math.radians(angle_deg)
    ux, uy = math.cos(th), math.sin(th)
    x0, y0 = CENTRE - STANDOFF * ux, CENTRE - STANDOFF * uy
    d = np.arange(0.0, 3000.0, 0.5)
    v = map_coordinates(phi, [(x0 + d * ux) / dx - 0.5, (y0 + d * uy) / dx - 0.5], order=1, mode="nearest")
    burned = v < 0.0
    idx = np.where(burned[:-1] & ~burned[1:])[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[-1]
    return d[i] + (0.0 - v[i]) * (d[i + 1] - d[i]) / (v[i + 1] - v[i])


def head_rate(prefix, angle):
    ts, ds = [], []
    for t, f in snapshots(prefix):
        phi, dx = load_phi(f)
        ts.append(t); ds.append(head_distance(phi, dx, angle))
    ts, ds = np.array(ts), np.array(ds)
    m = np.isfinite(ds) & (ts >= T_FIT_MIN)
    return float(np.polyfit(ts[m], ds[m], 1)[0]) if m.sum() >= 3 else float("nan")


def main():
    R0, phi_w = rothermel_fm1(M_F, U)
    Rf = R0 * (1.0 + phi_w)
    print(f"Rothermel FM1, M_f = {M_F}, U = {U} m/s:  R0 = {R0:.5f}  phi_w = {phi_w:.2f}  Rf = {Rf:.4f} m/s\n")

    mm = {}
    for scheme in ("baseline", "split"):
        s0, s34 = snapshots(f"plt_fire_{scheme}_0_"), snapshots(f"plt_fire_{scheme}_34_")
        if not s0 or len(s0) != len(s34):
            sys.exit(f"missing or unequal plotfiles for {scheme}")
        mm[scheme] = {}
        for (t, f0), (_, f34) in zip(s0, s34):
            if t < T_MISMATCH_MIN:
                continue
            p0, dx = load_phi(f0); p34, _ = load_phi(f34)
            mm[scheme][t] = mismatch(p0, p34, dx)
        print(f"{scheme:9s} mismatch  " + "  ".join(f"t={t:.0f}s: {100 * m:.2f}%" for t, m in mm[scheme].items()))

    print("\nRotation invariance (0 deg vs the 34 deg case un-rotated)")
    worst_split = max(mm["split"].values())
    check("split mismatch <= %.0f %% at every time" % (100 * MAX_MISMATCH_SPLIT), worst_split <= MAX_MISMATCH_SPLIT,
          f"worst {100 * worst_split:.3f} %")
    late = [t for t in mm["baseline"] if t >= T_FIT_MIN]
    lowest_base = min(mm["baseline"][t] for t in late)
    check("baseline mismatch >= %.0f %% for t >= %.0f s" % (100 * MIN_MISMATCH_BASELINE, T_FIT_MIN),
          lowest_base >= MIN_MISMATCH_BASELINE, f"lowest {100 * lowest_base:.2f} %")
    ratios = [mm["baseline"][t] / max(mm["split"][t], 1e-6) for t in late]
    check("baseline / split mismatch >= %.0fx for t >= %.0f s" % (MIN_MISMATCH_RATIO, T_FIT_MIN), min(ratios) >= MIN_MISMATCH_RATIO,
          f"smallest ratio {min(ratios):.0f}x")

    print("\nHead rate along the wind, fit over t >= %.0f s (Rothermel Rf = %.4f m/s)" % (T_FIT_MIN, Rf))
    for angle in (0, 34):
        r = head_rate(f"plt_fire_split_{angle}_", angle)
        check(f"split {angle:2d} deg head rate within {100 * TOL_HEAD:.0f} % of Rf", abs(r / Rf - 1.0) <= TOL_HEAD,
              f"{r:.4f} m/s ({100 * (r / Rf - 1.0):+.2f} %)")
    for angle in (0, 34):
        r = head_rate(f"plt_fire_baseline_{angle}_", angle)
        print(f"  (baseline {angle:2d} deg head rate {r:.4f} m/s, {100 * (r / Rf - 1.0):+.2f} % -- for reference)")

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
        return 0
    print(f"{results.count(False)} of {len(results)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
