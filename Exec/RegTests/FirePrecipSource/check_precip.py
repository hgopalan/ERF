#!/usr/bin/env python3
"""Checks on the FirePrecipSource regtest: the rain that wets the dead fuel classes
comes from the right place.

    python3 check_precip.py [--dt 0.5] [--min-rain 0.1]

Reads the fire plotfiles of the variants (plt_fire_<variant>/plt_fire_00040), the
last two atmosphere plotfiles of the atmosphere variant (plt_atmosphere/plt00039,
plt00040) and the checkpoint of the chk variant, with the pure-Python reader
erf_plotfile.py (copied next to this script by CTest, or found in
Exec/CanonicalTests/Canonical_RANS). Every check prints PASS or FAIL and the
script exits 1 if any fails.

  field      the atmosphere and uniform variants write fire_precip_mm_hr, the
             legacy variant (no rain key) does not
  columns    atmosphere: fire_precip_mm_hr is above the wetting threshold on some
             fire cells and zero on others (a raining column and a dry one exist)
  wet_atm    atmosphere minus legacy: the 1-h moisture is higher on every fire
             cell under a raining column and identical elsewhere
  wet_uni    uniform minus legacy: the 1-h moisture is higher on every fire cell,
             by the same amount to 1 %, and fire_precip_mm_hr is the deck's rate
  rate       atmosphere: the rain rate of every fire cell equals the change of
             rain_accum of its column over the last step, times 3600 / dt
  snapshot   the checkpoint carries FirePrecipAccumPrev
  restart    the restarted run reproduces the straight run's level set, 1-h
             moisture and rain rate exactly
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "CanonicalTests", "Canonical_RANS"))
from erf_plotfile import read_fields, _read_header  # noqa: E402

FIRE_PLT = "plt_fire_00040"
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"  {name:9s} {'PASS' if ok else 'FAIL'}  {detail}")
    return ok


def plane(pf, field):
    """The k = 0 plane of a field as data[i][j]."""
    _, data = read_fields(pf, [field])
    return [[col[0] for col in row] for row in data[field]]


def names(pf):
    return _read_header(pf)["names"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.5, help="erf.fixed_dt of the decks [s]")
    ap.add_argument("--min-rain", type=float, default=0.1,
                    help="FuelMoistureConst::MIN_RAIN_RATE, below which rain does not wet [mm/hr]")
    ap.add_argument("--uniform-rate", type=float, default=2.0, help="erf.fire.precip_rate_mm_hr of inputs_uniform")
    args = ap.parse_args()

    pf = {v: os.path.join(f"plt_fire_{v}", FIRE_PLT) for v in ("legacy", "uniform", "atmosphere", "atmosphere_restart")}
    for v, p in pf.items():
        if not os.path.isdir(p):
            sys.exit(f"{p} not found: did the {v} run finish?")

    # field: present with a rain source, absent on the historical deck
    n_leg, n_uni, n_atm = names(pf["legacy"]), names(pf["uniform"]), names(pf["atmosphere"])
    check("field", "fire_precip_mm_hr" in n_uni and "fire_precip_mm_hr" in n_atm
          and "fire_precip_mm_hr" not in n_leg,
          f"fire_precip_mm_hr in uniform: {'fire_precip_mm_hr' in n_uni}, atmosphere: "
          f"{'fire_precip_mm_hr' in n_atm}, legacy: {'fire_precip_mm_hr' in n_leg}")
    if "fire_precip_mm_hr" not in n_atm or "fire_precip_mm_hr" not in n_uni:
        print("  the remaining checks need the field; a binary without erf.fire.precip_source")
        sys.exit(1)

    rate_atm = plane(pf["atmosphere"], "fire_precip_mm_hr")
    rate_uni = plane(pf["uniform"], "fire_precip_mm_hr")
    mc_leg = plane(pf["legacy"], "fire_fuel_mc_1hr")
    mc_uni = plane(pf["uniform"], "fire_fuel_mc_1hr")
    mc_atm = plane(pf["atmosphere"], "fire_fuel_mc_1hr")
    nx, ny = len(rate_atm), len(rate_atm[0])
    cells = [(i, j) for i in range(nx) for j in range(ny)]

    # columns: rain above the threshold somewhere, none elsewhere
    wet = [(i, j) for (i, j) in cells if rate_atm[i][j] >= args.min_rain]
    dry = [(i, j) for (i, j) in cells if rate_atm[i][j] == 0.0]
    between = len(cells) - len(wet) - len(dry)
    rmax = max(rate_atm[i][j] for (i, j) in cells)
    check("columns", len(wet) > 0 and len(dry) > 0,
          f"{len(wet)} fire cells at or above {args.min_rain} mm/hr (max {rmax:.3f}), {len(dry)} at zero, "
          f"{between} in between, of {len(cells)}")

    # wet_atm: the rain wets the 1-h class under the raining columns only
    d_atm = {(i, j): mc_atm[i][j] - mc_leg[i][j] for (i, j) in cells}
    wet_ok = all(d_atm[c] > 0.0 for c in wet)
    dry_ok = all(d_atm[c] == 0.0 for c in cells if rate_atm[c[0]][c[1]] < args.min_rain)
    dmin_wet = min((d_atm[c] for c in wet), default=0.0)
    dmax_dry = max((abs(d_atm[c]) for c in cells if rate_atm[c[0]][c[1]] < args.min_rain), default=0.0)
    check("wet_atm", len(wet) > 0 and wet_ok and dry_ok,
          f"1-h moisture minus legacy: min {dmin_wet:.3e} under rain, max |diff| {dmax_dry:.1e} elsewhere")

    # wet_uni: the uniform rate wets every cell by the same amount, and the field is the rate
    d_uni = [mc_uni[i][j] - mc_leg[i][j] for (i, j) in cells]
    lo, hi = min(d_uni), max(d_uni)
    same_rate = all(rate_uni[i][j] == args.uniform_rate for (i, j) in cells)
    check("wet_uni", lo > 0.0 and (hi - lo) <= 0.01 * hi and same_rate,
          f"1-h moisture minus legacy: min {lo:.3e} max {hi:.3e}; fire_precip_mm_hr == {args.uniform_rate} everywhere: {same_rate}")

    # rate: the fire cell's rate is its column's accumulation change over the last step
    ok = True; worst = 0.0; nchecked = 0
    a39 = plane(os.path.join("plt_atmosphere", "plt00039"), "rain_accum")
    a40 = plane(os.path.join("plt_atmosphere", "plt00040"), "rain_accum")
    C = nx // len(a40)
    for (i, j) in cells:
        expect = max(0.0, a40[i // C][j // C] - a39[i // C][j // C]) * 3600.0 / args.dt
        got = rate_atm[i][j]
        err = abs(got - expect) / max(abs(expect), 1e-30)
        if expect == 0.0:
            err = abs(got)
        worst = max(worst, err)
        nchecked += 1
        if err > 1e-10:
            ok = False
    check("rate", ok and nchecked == len(cells),
          f"fire_precip_mm_hr vs (rain_accum[40] - rain_accum[39]) * 3600 / dt on {nchecked} cells "
          f"(grid ratio {C}): worst rel. error {worst:.1e}")

    # snapshot: the checkpoint carries the previous accumulation
    snap = os.path.join("chk00027", "Level_0", "FirePrecipAccumPrev_H")
    check("snapshot", os.path.isfile(snap), f"{snap} {'present' if os.path.isfile(snap) else 'missing'}")

    # restart: straight and restarted runs identical
    worst = 0.0
    for f in ("fire_phi", "fire_fuel_mc_1hr", "fire_precip_mm_hr"):
        a = plane(pf["atmosphere"], f); b = plane(pf["atmosphere_restart"], f)
        worst = max(worst, max(abs(a[i][j] - b[i][j]) for (i, j) in cells))
    check("restart", worst == 0.0,
          f"max |straight - restarted| over phi, 1-h moisture and rain rate: {worst:.1e}")

    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
