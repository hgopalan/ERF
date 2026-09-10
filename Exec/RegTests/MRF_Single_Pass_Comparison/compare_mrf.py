#!/usr/bin/env python3
"""Tabulate PBLH, Kmv and theta of an old and a new MRF run of the same deck.

    python3 compare_mrf.py OLD_RUN_DIR NEW_RUN_DIR [--label name]

Each run directory holds the plt*/plt2d* plotfiles and mean_profiles.dat written
by run_comparison.sh. For every plotfile step present in both runs it prints

  Lturb      horizontal mean of the PBLH the K-profile used (Turb_lengthscale)
  pblh       horizontal mean of the PBLH SurfaceLayer stored (2D plotfile)
  Kmv_max    domain maximum of rho*K_m [kg/(m s)]
  Kmv@z      horizontal mean rho*K_m at the level nearest z
  th@z       horizontal mean theta at the level nearest z [K]
  th_max     domain maximum theta [K]

then the largest old-new difference of the mean theta and Kmv profiles. Kmv is
the plotfile's rho-weighted value, as the diffusion operator receives it.
"""
import argparse, glob, os, re, sys
import numpy as np

try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs yt: pip install yt")

Z_LEVELS = (100.0, 500.0)


def valid(x):
    """False for the -999 plot fill and SurfaceLayer's bogus large initial PBLH."""
    return -998.0 < x < 1.0e20


def plotfiles(run, prefix):
    out = {}
    for p in glob.glob(os.path.join(run, prefix + "[0-9]*")):
        m = re.fullmatch(re.escape(prefix) + r"(\d+)", os.path.basename(p))
        if m and os.path.isdir(p):
            out[int(m.group(1))] = p
    return out


def level0(path, names):
    ds = yt.load(path)
    cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    return float(ds.current_time), {n: np.array(cg[("boxlib", n)]) for n in names}


def z_centres(run, nz):
    """Cell-centre heights from mean_profiles.dat (column 2 of the first block)."""
    rows = np.loadtxt(os.path.join(run, "mean_profiles.dat"), usecols=(0, 1))
    return rows[:nz, 1]


def summary(run, step, nearest):
    t, f = level0(plotfiles(run, "plt")[step], ("theta", "Kmv", "Lturb"))
    _, g = level0(plotfiles(run, "plt2d")[step], ("pblh",))
    th_prof = f["theta"].mean(axis=(0, 1))
    kmv_prof = f["Kmv"].mean(axis=(0, 1))
    row = {
        "t": t,
        "Lturb": f["Lturb"][:, :, 0].mean(),
        "pblh": g["pblh"].mean(),
        "Kmv_max": f["Kmv"].max(),
        "th_max": f["theta"].max(),
    }
    for z, k in nearest.items():
        row[f"Kmv@{z:g}"] = kmv_prof[k]
        row[f"th@{z:g}"] = th_prof[k]
    return row, th_prof, kmv_prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    steps = sorted(set(plotfiles(a.old, "plt")) & set(plotfiles(a.new, "plt"))
                   & set(plotfiles(a.old, "plt2d")) & set(plotfiles(a.new, "plt2d")))
    if not steps:
        print(f"{a.label}: no common plotfiles in {a.old} and {a.new}")
        return 1

    _, f0 = level0(plotfiles(a.old, "plt")[steps[0]], ("theta",))
    z = z_centres(a.old, f0["theta"].shape[2])
    nearest = {zl: int(np.argmin(np.abs(z - zl))) for zl in Z_LEVELS}

    keys = ["Lturb", "pblh", "Kmv_max"] + [f"Kmv@{zl:g}" for zl in Z_LEVELS] \
         + [f"th@{zl:g}" for zl in Z_LEVELS] + ["th_max"]
    fmt = {"Lturb": "{:8.1f}", "pblh": "{:8.1f}", "th_max": "{:8.3f}"}
    fmt.update({f"th@{zl:g}": "{:8.3f}" for zl in Z_LEVELS})

    print(f"\n== {a.label}  (z levels used: "
          + ", ".join(f"{zl:g} m -> {z[k]:.1f} m" for zl, k in nearest.items()) + ")")
    print("   step      t[s]  run  " + " ".join(f"{k:>8s}" for k in keys))
    worst = []
    for s in steps:
        ro, th_old, km_old = summary(a.old, s, nearest)
        rn, th_new, km_new = summary(a.new, s, nearest)
        for tag, r in (("old", ro), ("new", rn)):
            vals = " ".join(fmt.get(k, "{:8.2f}").format(r[k]) if valid(r[k]) else "     n/a"
                            for k in keys)
            print(f"{s:7d} {r['t']:9.1f}  {tag}  {vals}")
        rel = " ".join(f"{100.0 * (rn[k] - ro[k]) / ro[k]:+7.2f}%"
                       if ro[k] != 0 and valid(ro[k]) and valid(rn[k]) else "     n/a"
                       for k in keys)
        print(f"{'':17s}  d%   {rel}")
        dth = np.abs(th_new - th_old)
        dkm = np.abs(km_new - km_old)
        worst.append((s, dth.max(), z[int(np.argmax(dth))], dkm.max(), z[int(np.argmax(dkm))]))
    print("   step  max|d<theta>| [K] at z[m]   max|d<Kmv>| at z[m]")
    for s, dt, zt, dk, zk in worst:
        print(f"{s:7d}  {dt:14.4f} {zt:8.1f}   {dk:11.3f} {zk:8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
