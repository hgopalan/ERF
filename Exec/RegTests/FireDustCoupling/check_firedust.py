#!/usr/bin/env python3
"""Checks on the fire-dust coupling regression (FireDustCoupling/inputs).

    python3 check_firedust.py [--fire-prefix plt_fire_] [--dust-prefix plt_dust_]
                              [--diag dust_diag.dat]

Reads the fire and dust plotfiles of the same steps (both on the 25 m grid) with
the pure-Python reader erf_plotfile.py and checks:

  crust     u*_t(burned) / u*_t(unburned) = (1 + a c (1 - r)) / (1 + a c) at every
            plotted step, with a = alpha_crust, c = crust_index, r = the crust
            reduction. The reduction is applied to the baseline once per step; a
            compounded reduction gives 1 / (1 + a c) after a few steps.
  fire_u*   dust_ustar_in >= kappa |U_fire| / ln(zref / z0) in every cell, the
            log-law u* of the fire-grid effective wind (kappa 0.4, the coupling's
            constants). An overwritten coupling leaves cells below it.
  deposit   the deposition total of dust_diag.dat at the last step is within
            15 % of the reference measured after the once-per-step fix and never
            decreases (a per-stage accumulation is 1.83x larger).

Exit 1 on any failure. The numbers are for the committed deck; the tolerances
cover box-layout round-off, not model changes.
"""
import argparse, glob, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erf_plotfile  # noqa: E402

ALPHA_CRUST = 0.5
CRUST_INDEX = 1.0
REDUCTION = 0.8
KAPPA, Z0, ZREF = 0.4, 3.0, 6.1
DEP_REF = 1.5596e-02   # deposition_total [kg/m2] at step 40 with the committed deck (2026-09-13, after the MB95 and Bagnold fixes)
DEP_TOL = 0.15

results = []
def check(name, ok, detail):
    results.append(ok)
    print(f"  {name:8s} {'PASS' if ok else 'FAIL'}  {detail}")

def steps(prefix):
    out = {}
    for p in sorted(glob.glob(prefix + "[0-9]" * 5)):
        if os.path.isdir(p):
            out[int(p[len(prefix):])] = p
    return out

def read_deck_value(key, default):
    try:
        for line in open("inputs"):
            s = line.split("#")[0].strip()
            if s.startswith(key) and "=" in s:
                return float(s.split("=")[1].split()[0])
    except OSError:
        pass
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire-prefix", default="plt_fire_")
    ap.add_argument("--dust-prefix", default="plt_dust_")
    ap.add_argument("--diag", default="dust_diag.dat")
    args = ap.parse_args()

    a = read_deck_value("erf.dust.alpha_crust", ALPHA_CRUST)
    c = read_deck_value("erf.dust.crust_index", CRUST_INDEX)
    r = read_deck_value("erf.fire_dust_crust_reduction", REDUCTION)
    z0 = read_deck_value("erf.fire_dust_wind_z0", Z0)
    zref = read_deck_value("erf.fire_dust_wind_zref", ZREF)
    expected_ratio = (1.0 + a * c * (1.0 - r)) / (1.0 + a * c)

    fire = steps(args.fire_prefix); dust = steps(args.dust_prefix)
    common = sorted(n for n in set(fire) & set(dust) if n > 0)   # step 0 has no fire wind yet
    if not common:
        sys.exit(f"no common fire/dust plotfile steps (fire {sorted(fire)}, dust {sorted(dust)})")

    for n in common:
        _, f = erf_plotfile.read_fields(fire[n], ["fire_phi", "fire_wind_eff_u", "fire_wind_eff_v"])
        _, d = erf_plotfile.read_fields(dust[n], ["dust_ustar_t", "dust_ustar_in"])
        phi, ut, ui = f["fire_phi"], d["dust_ustar_t"], d["dust_ustar_in"]
        nx, ny = len(phi), len(phi[0])
        burned = [(i, j) for i in range(nx) for j in range(ny) if phi[i][j][0] < 0.0]
        unburned = [(i, j) for i in range(nx) for j in range(ny) if phi[i][j][0] >= 0.0]
        check(f"ignited{n}", len(burned) > 10 and len(unburned) > 10,
              f"step {n}: {len(burned)} burned, {len(unburned)} unburned cells")
        if not burned or not unburned:
            continue
        ub = sum(ut[i][j][0] for i, j in burned) / len(burned)
        uu = sum(ut[i][j][0] for i, j in unburned) / len(unburned)
        ratio = ub / uu if uu > 0 else float("nan")
        check(f"crust{n}", abs(ratio - expected_ratio) < 0.01,
              f"step {n}: u*_t burned/unburned = {ratio:.4f}, expected {expected_ratio:.4f}"
              f" (compounded would give {1.0 / (1.0 + a * c):.4f})")
        # fire wind -> u*
        log_ratio = math.log(zref / z0)
        worst = 0.0; nboost = 0
        for i in range(nx):
            for j in range(ny):
                spd = math.hypot(f["fire_wind_eff_u"][i][j][0], f["fire_wind_eff_v"][i][j][0])
                us_fire = spd * KAPPA / log_ratio
                deficit = us_fire - ui[i][j][0]
                worst = max(worst, deficit)
                if us_fire > 1.0e-6:
                    nboost += 1
        check(f"fireu*{n}", worst < 1.0e-6 and nboost > 0,
              f"step {n}: max(u*_fire - dust u*) = {worst:.3e} m/s over {nboost} cells with fire wind")

    # deposition accumulator: monotone, once per step
    try:
        rows = [l.split(",") for l in open(args.diag) if l.strip() and not l.startswith("#") and not l.startswith("step")]
        dep = [float(r[3]) for r in rows]
        mono = all(b >= a - 1e-30 for a, b in zip(dep, dep[1:]))
        check("dep_mono", mono and len(dep) >= 2, f"{len(dep)} rows, deposition_total {dep[0]:.4e} -> {dep[-1]:.4e} kg/m2")
        if DEP_REF is not None:
            check("dep_ref", abs(dep[-1] - DEP_REF) <= DEP_TOL * DEP_REF,
                  f"deposition_total at the last step {dep[-1]:.6e}, reference {DEP_REF:.6e} +/- {DEP_TOL * 100:.0f}%"
                  f" (per-stage accumulation would give {1.83 * DEP_REF:.3e})")
    except OSError as e:
        check("dep_mono", False, f"cannot read {args.diag}: {e}")

    ok = all(results)
    print("ALL PASS" if ok else f"{results.count(False)} FAILED")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
