#!/usr/bin/env python3
"""Checks on the FirePrescribed decks against their exact answers.

    python3 check_fireprescribed.py ros_circle heat_patch

Each variant reads plt_fire_<variant>_NNNNN (and plt_atm_<variant>_NNNNN), as
run_fireprescribed.sh writes them, or plt_fire_NNNNN / plt_atm_NNNNN when a
CTest directory holds a single deck; the log is run_<variant>.log or the only
*.log present.

ros_circle   a 10 m disc grown at a prescribed 1 m/s. The rate field must be
             exactly 1 m/s, the burned area must give a radius of 10 + t, and
             the arrival time at distance d from the centre must be d - 10.
heat_patch   a prescribed 1e4 W/m2 over a 30 m disc with nothing burning. The
             fire heat flux must be the flux times the disc's cells and no cell
             may burn. The coupling takes in exactly that power (energy_in)
             and places it with the profile exp(-z / alfg), whose flux
             differences telescope to 1 - exp(-H / alfg) of it for a column of
             height H, so energy_out / energy_in must be that factor to round
             off. The heat in the atmosphere, Cp times the change of the
             integral of rho theta, must then be the placed power times the
             heated time.
"""
import glob, math, re, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt")

CP_D = 1004.5
ALFG = 45.0      # erf.fire.heat_flux_alfg, the default
results = []

def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {name:22s} {'PASS' if ok else 'FAIL'}  {detail}")

def plotfiles(kind, v):
    pfs = sorted(glob.glob(f"plt_{kind}_{v}_?????")) or sorted(glob.glob(f"plt_{kind}_?????"))
    return pfs

def logfile(v):
    logs = glob.glob(f"run_{v}.log") or glob.glob("*.log")
    return logs[0] if logs else None

def fields(pf, names):
    ds = yt.load(pf)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    out = {n: np.asarray(g[("boxlib", n)]) for n in names}
    dx = (ds.domain_right_edge - ds.domain_left_edge).d / ds.domain_dimensions
    return float(ds.current_time), out, dx, ds.domain_left_edge.d

def ros_circle(v, R=1.0, r0=10.0, xc=200.0, yc=200.0):
    pfs = plotfiles("fire", v)
    if not pfs:
        check("plotfiles", False, "no fire plotfiles"); return
    for pf in pfs[1:]:
        t, f, dx, lo = fields(pf, ["fire_phi", "fire_ros", "fire_arrival_time"])
        phi, ros, at = (f[n][:, :, 0] for n in ("fire_phi", "fire_ros", "fire_arrival_time"))
        h = dx[0]
        check(f"{pf}: rate", np.allclose(ros, R, rtol=0, atol=1e-12),
              f"fire_ros in [{ros.min():.6g}, {ros.max():.6g}] m/s, prescribed {R}")
        area = np.clip(0.5 - phi / h, 0.0, 1.0).sum() * dx[0] * dx[1]
        r_num, r_ex = math.sqrt(area / math.pi), r0 + R * t
        check(f"{pf}: radius", abs(r_num - r_ex) <= 0.25 * h,
              f"radius from area {r_num:.3f} m vs 10 + t = {r_ex:.3f} m ({(r_num - r_ex) / h:+.2f} cells)")
        x = lo[0] + (np.arange(phi.shape[0]) + 0.5) * dx[0]
        y = lo[1] + (np.arange(phi.shape[1]) + 0.5) * dx[1]
        d = np.hypot(*np.meshgrid(x - xc, y - yc, indexing="ij"))
        band = (d > r0 + 2 * h) & (d < r_ex - 2 * h) & (at >= 0)
        if band.sum() > 10:
            err = at[band] - (d[band] - r0) / R
            check(f"{pf}: arrival", np.abs(err).mean() <= 0.5 * h / R and np.percentile(np.abs(err), 95) <= 1.5 * h / R,
                  f"{band.sum()} cells, error mean {err.mean():+.3f} s, mean |e| {np.abs(err).mean():.3f} s, "
                  f"95th pct {np.percentile(np.abs(err), 95):.3f} s (cell crossing {h / R:.1f} s)")

def heat_patch(v, q=1.0e4, xc=200.0, yc=200.0, r=30.0):
    pfs = plotfiles("fire", v)
    log = logfile(v)
    if not pfs or not log:
        check("output", False, f"fire plotfiles {len(pfs)}, log {log}"); return
    t, f, dx, lo = fields(pfs[-1], ["fire_phi", "fire_heat_flux"])
    phi, hf = f["fire_phi"][:, :, 0], f["fire_heat_flux"][:, :, 0]
    x = lo[0] + (np.arange(hf.shape[0]) + 0.5) * dx[0]
    y = lo[1] + (np.arange(hf.shape[1]) + 0.5) * dx[1]
    inside = np.hypot(*np.meshgrid(x - xc, y - yc, indexing="ij")) <= r
    P_exact = q * inside.sum() * dx[0] * dx[1]
    P_fire = hf.sum() * dx[0] * dx[1]
    check("fire heat flux", np.allclose(hf[inside], q) and np.all(hf[~inside] == 0.0),
          f"{inside.sum()} cells at {q:g} W/m2, power {P_fire / 1e6:.4f} MW (pi r^2 q = {math.pi * r * r * q / 1e6:.4f} MW)")
    check("nothing burned", np.all(phi > 0.0), f"min phi {phi.min():.3g} m")

    deck = open("inputs_base").read()
    z_top = float(re.search(r"geometry\.prob_hi\s*=\s*\S+\s+\S+\s+([\d.eE+-]+)", deck).group(1))
    placed = 1.0 - math.exp(-z_top / ALFG)
    e = [(float(a), float(b)) for a, b in re.findall(r"energy_in=([\d.eE+-]+) W  energy_out=([\d.eE+-]+) W", open(log).read())]
    e = [p for p in e if p[0] > 0.0]
    if not e:
        check("coupling energy", False, f"no energy_in lines with power in {log}")
    else:
        worst_in = max(abs(a - P_exact) / P_exact for a, _ in e)
        worst_ratio = max(abs(b / a - placed) for a, b in e)
        check("coupling energy", worst_in < 1e-9 and worst_ratio < 1e-9,
              f"{len(e)} stage lines: energy_in within {worst_in:.1e} of the disc power, "
              f"energy_out/energy_in within {worst_ratio:.1e} of 1 - exp(-{z_top:g}/{ALFG:g}) = {placed:.6f}")

    apfs = plotfiles("atm", v)
    if len(apfs) >= 2:
        t0, a0, dxa, _ = fields(apfs[0], ["density", "theta"])
        t1, a1, _, _ = fields(apfs[-1], ["density", "theta"])
        dV = dxa[0] * dxa[1] * dxa[2]
        heat = CP_D * ((a1["density"] * a1["theta"]).sum() - (a0["density"] * a0["theta"]).sum()) * dV
        dt = float(re.search(r"erf\.fixed_dt\s*=\s*([\d.eE+-]+)", deck).group(1))
        expected = placed * P_exact * (t1 - t0 - dt)   # lagged: the first step injects the zero initial flux
        warm = (a1["theta"] - a0["theta"]).max()
        check("atmosphere heat budget", abs(heat / expected - 1.0) < 0.005 and warm > 0.0,
              f"Cp d(int rho theta) = {heat / 1e9:.4f} GJ vs placed P (t - dt) = {expected / 1e9:.4f} GJ "
              f"({(heat / expected - 1) * 100:+.2f} %), max warming {warm:.3f} K")
    else:
        check("atmosphere heat budget", False, f"need two atmospheric plotfiles, found {len(apfs)}")

CHECKS = {"ros_circle": ros_circle, "heat_patch": heat_patch}

def main():
    variants = sys.argv[1:] or list(CHECKS)
    for v in variants:
        print(f"{v}:")
        CHECKS[v](v)
    n_fail = results.count(False)
    print(f"FirePrescribed: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
    main()
