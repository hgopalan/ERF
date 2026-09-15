#!/usr/bin/env python3
"""Checks on the FireAnchorLevel regression test.

    python3 check_anchor_level.py [parity] [heat] [restart] [mrf]

run_anchor_level.sh writes, for each variant v, run_<v>.log, fire_stats_<v>.csv,
plt_fire_<v>_NNNNN and, for the heat decks, plt_atm_<v>_NNNNN.

parity   inputs_base (the fire on level 1, erf.fire.anchor_level unset) against
         inputs_single (one level at level 1's resolution and step). The fire
         grid of inputs_base is the refined region, x 200-600 m by y 200-500 m in
         64 x 48 cells; every one of its cells has the arrival time of the
         single-level run's cell at the same position, the single-level fire
         never leaves that region, and the two statistics CSVs are identical.
         The comparison only means something for a front that grew through both
         fuel codes of the map, which is checked too.
heat     inputs_heat, an anelastic prescribed heat disc on level 1 with two-way
         coupling: the coupling on level 1 takes in the disc's power and places
         1 - exp(-400 / 45) of it (to 1e-9); Cp times the change of the level-0
         integral of rho theta equals the placed power over the heated time
         within 0.1 %; the level-1 integral equals level 0's over the refined
         region, and no heat appears outside it. The same deck with
         erf.fire.anchor_level = 0 warns that the fire grid is not on the finest
         level and loses the heat, since average-down replaces the heated cells.
restart  inputs_base restarted at step 20 from its own checkpoint: the fire
         plotfile at step 40 is byte-identical to the uninterrupted run's and the
         statistics rows after the restart are the uninterrupted run's rows.
mrf      inputs_heat with MRF and erf.pbl_mrf_fire_thermal_excess, which read the
         lagged fire flux on level 1 and in its ghost columns, restarted at step
         10: the atmosphere (both levels) and fire plotfiles at step 40 are
         byte-identical to the uninterrupted run's, and MRF changed the
         atmosphere of the heat run.

Standard library only (erf_plotfile.py from Canonical_RANS is copied beside it).
"""
import csv
import glob
import math
import os
import re
import sys

import erf_plotfile as pf

CP_D = 1004.5        # ERF's Cp_d
ALFG = 45.0          # erf.fire.heat_flux_alfg, the default
Z_TOP = 400.0        # geometry.prob_hi z of inputs_base
results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {name:34s} {'PASS' if ok else 'FAIL'}  {detail}")


def last_plotfile(kind, v):
    pfs = sorted(glob.glob(f"plt_{kind}_{v}_?????"))
    return pfs[-1] if pfs else None


def rows(v):
    path = f"fire_stats_{v}.csv"
    return list(csv.reader(open(path))) if os.path.isfile(path) else None


def plane(data, name):
    """data[name][i][j][0] as a list of columns"""
    return [[col[0] for col in row] for row in data[name]]


def parity():
    a_pf, s_pf = last_plotfile("fire", "anchor"), last_plotfile("fire", "single")
    if not a_pf or not s_pf:
        check("plotfiles", False, f"anchor {a_pf}, single {s_pf}")
        return
    ha, da = pf.read_fields(a_pf, ["fire_arrival_time", "fire_fuel_model_code"])
    hs, ds = pf.read_fields(s_pf, ["fire_arrival_time"])
    at_a, code_a = plane(da, "fire_arrival_time"), plane(da, "fire_fuel_model_code")
    at_s = plane(ds, "fire_arrival_time")

    nx, ny = ha["hi"][0] - ha["lo"][0] + 1, ha["hi"][1] - ha["lo"][1] + 1
    geom_ok = (abs(ha["prob_lo"][0] - 200.0) < 1e-9 and abs(ha["prob_lo"][1] - 200.0) < 1e-9
               and abs(ha["prob_hi"][0] - 600.0) < 1e-9 and abs(ha["prob_hi"][1] - 500.0) < 1e-9
               and (nx, ny) == (64, 48) and ha["lo"][:2] == [0, 0])
    check("fire grid is the refined region", geom_ok,
          f"x {ha['prob_lo'][0]:g}-{ha['prob_hi'][0]:g} m, y {ha['prob_lo'][1]:g}-{ha['prob_hi'][1]:g} m, "
          f"{nx} x {ny} cells from ({ha['lo'][0]}, {ha['lo'][1]})")
    log = open("run_anchor.log", errors="replace").read()
    check("start-up line names level 1", re.search(r"\[FIRE\] Fire grid on level 1 of 0-1: x 200 to 600 m, y 200 to 500 m, 64 x 48 fire cells", log) is not None,
          "[FIRE] Fire grid on level 1 of 0-1 ...")
    check("no warning on the finest level", "is not the finest" not in log and "but the finest level is" not in log, "")
    check("same output time", abs(ha["time"] - hs["time"]) < 1e-9, f"{ha['time']:g} s and {hs['time']:g} s")

    dx, dxs = ha["dx"], hs["dx"]
    worst, burned, codes, t_max = 0.0, 0, set(), -1.0
    inside = set()
    for i in range(nx):
        x = ha["prob_lo"][0] + (i + 0.5) * dx[0]
        si = int(round((x - hs["prob_lo"][0]) / dxs[0] - 0.5))
        for j in range(ny):
            y = ha["prob_lo"][1] + (j + 0.5) * dx[1]
            sj = int(round((y - hs["prob_lo"][1]) / dxs[1] - 0.5))
            inside.add((si, sj))
            worst = max(worst, abs(at_a[i][j] - at_s[si][sj]))
            if at_a[i][j] >= 0.0:
                burned += 1
                codes.add(int(round(code_a[i][j])))
                t_max = max(t_max, at_a[i][j])
    check("arrival times equal", worst <= 1e-9, f"largest difference {worst:.3g} s over {nx * ny} cells")
    check("front grew through both fuels", burned >= 100 and codes == {1, 4} and t_max > 20.0,
          f"{burned} cells burned, fuel codes {sorted(codes)}, latest arrival {t_max:.1f} s")
    outside = sum(1 for si in range(len(at_s)) for sj in range(len(at_s[0]))
                  if (si, sj) not in inside and at_s[si][sj] >= 0.0)
    check("single-level fire stays in the region", outside == 0, f"{outside} burned cells outside it")

    ra, rs = rows("anchor"), rows("single")
    same = ra is not None and rs is not None and len(ra) > 60 and ra == rs   # 80 fire steps and the header
    check("statistics CSVs identical", same,
          f"{len(ra) if ra else 0} and {len(rs) if rs else 0} rows")


def level_integral(plt, lev, region=None, ratio=(2, 2, 1)):
    """Sum of rho theta dV over the valid cells of level lev, optionally only
    the cells (i, j) with region = (ilo, ihi, jlo, jhi) in that level's indices."""
    hdr = pf._read_header(plt)
    names = hdr["names"]
    ir, it = names.index("density"), names.index("theta")
    dx = [hdr["dx"][d] / ratio[d] ** lev for d in range(3)]
    ldir = os.path.join(plt, f"Level_{lev}")
    ncomp, _, boxes, fabs = pf._read_cell_h(ldir)
    tot = 0.0
    for (blo, bhi), (fname, off) in zip(boxes, fabs):
        flo, fhi, data = pf._read_fab(os.path.join(ldir, fname), off, ncomp)
        nx, ny, nz = [fhi[d] - flo[d] + 1 for d in range(3)]
        npts = nx * ny * nz
        for k in range(blo[2], bhi[2] + 1):
            for j in range(blo[1], bhi[1] + 1):
                if region and not (region[2] <= j <= region[3]):
                    continue
                row = ((k - flo[2]) * ny + (j - flo[1])) * nx - flo[0]
                for i in range(blo[0], bhi[0] + 1):
                    if region and not (region[0] <= i <= region[1]):
                        continue
                    tot += data[ir * npts + row + i] * data[it * npts + row + i]
    return tot * dx[0] * dx[1] * dx[2]


REGION_L0 = (4, 11, 4, 9)    # the refined region, x 200-600 m by y 200-500 m, in level-0 columns


def heat_budget(v):
    """Cp d(int rho theta) on level 0 (whole and over the refined region) and on
    level 1, the fire power, the expected heat and the coupling's energy lines."""
    apfs = sorted(glob.glob(f"plt_atm_{v}_?????"))
    fpf = last_plotfile("fire", v)
    r = rows(v)
    log = f"run_{v}.log"
    if len(apfs) < 2 or not fpf or not r or len(r) < 3 or not os.path.isfile(log):
        return None, f"atmospheric plotfiles {len(apfs)}, fire plotfile {fpf}, CSV rows {len(r) if r else 0}"
    a0, a1 = apfs[0], apfs[-1]
    h0, h1 = pf._read_header(a0), pf._read_header(a1)
    out = {"L0": CP_D * (level_integral(a1, 0) - level_integral(a0, 0)),
           "L0_region": CP_D * (level_integral(a1, 0, REGION_L0) - level_integral(a0, 0, REGION_L0))}
    if h1["finest"] >= 1:
        out["L1"] = CP_D * (level_integral(a1, 1) - level_integral(a0, 1))
    hf, df = pf.read_fields(fpf, ["fire_heat_flux"])
    out["P"] = sum(sum(col[0] for col in row) for row in df["fire_heat_flux"]) * hf["dx"][0] * hf["dx"][1]
    dt_fire = float(r[2][1]) - float(r[1][1])
    out["placed"] = 1.0 - math.exp(-Z_TOP / ALFG)
    out["expected"] = out["placed"] * out["P"] * (h1["time"] - h0["time"] - dt_fire)   # lagged: the first fire step injects zero
    e = [(float(a), float(b)) for a, b in re.findall(r"energy_in=([\d.eE+-]+) W  energy_out=([\d.eE+-]+) W", open(log).read())]
    out["energy"] = [p for p in e if p[0] > 0.0]
    return out, (f"Cp d(int rho theta) on level 0 = {out['L0'] / 1e9:.5f} GJ, placed power x heated time = "
                 f"{out['expected'] / 1e9:.5f} GJ ({out['P'] / 1e6:.4f} MW, fire step {dt_fire:g} s)")


def heat():
    b, detail = heat_budget("heat")
    if b is None:
        check("heat output", False, detail)
    else:
        e = b["energy"]
        worst_in = max((abs(a - b["P"]) / b["P"] for a, _ in e), default=1.0)
        worst_out = max((abs(o / a - b["placed"]) for a, o in e), default=1.0)
        check("coupling on level 1", len(e) > 0 and worst_in < 1e-9 and worst_out < 1e-9,
              f"{len(e)} stage lines: energy_in within {worst_in:.1e} of the disc power, "
              f"energy_out / energy_in within {worst_out:.1e} of {b['placed']:.8f}")
        ratio = b["L0"] / b["expected"]
        check("level-0 heat budget", abs(ratio - 1.0) < 1e-3, f"{detail}, ratio {ratio:.5f}")
        L1 = b.get("L1", float("nan"))
        check("level 1 averages down to level 0", L1 > 0.0 and abs(b["L0_region"] - L1) <= 1e-9 * L1,
              f"level 1 {L1 / 1e9:.6f} GJ, level 0 over the refined region {b['L0_region'] / 1e9:.6f} GJ")
        check("no heat outside the refined region", abs(b["L0"] - b["L0_region"]) <= 1e-6 * b["L0"],
              f"{(b['L0'] - b['L0_region']) / 1e9:.3g} GJ outside it")

    b0, detail0 = heat_budget("heat_anchor0")
    log0 = open("run_heat_anchor0.log", errors="replace").read() if os.path.isfile("run_heat_anchor0.log") else ""
    m = re.search(r"WARNING: the fire grid is on level 0 but the finest level is 1 .*?level 1 covers ([\d.]+) % of the fire grid", log0)
    check("anchor_level 0 warns", m is not None and abs(float(m.group(1)) - 18.75) < 1e-6,
          f"level 1 covers {m.group(1) if m else '?'} % of the level-0 fire grid (400 x 300 of 800 x 800 m)")
    ok0 = b0 is not None and abs(b0["L0"] / b0["expected"]) < 0.01
    check("anchor_level 0 loses the heat", ok0,
          detail0 if b0 is None else f"{detail0}, ratio {b0['L0'] / b0['expected']:.5f}")


def restart():
    a_pf, r_pf = last_plotfile("fire", "anchor"), last_plotfile("fire", "restart")
    if not a_pf or not r_pf or os.path.basename(a_pf)[-5:] != os.path.basename(r_pf)[-5:]:
        check("restart plotfiles", False, f"straight {a_pf}, restarted {r_pf}")
        return
    fa = sorted(glob.glob(os.path.join(a_pf, "Level_0", "Cell_D_*")))
    fr = sorted(glob.glob(os.path.join(r_pf, "Level_0", "Cell_D_*")))
    same = len(fa) > 0 and [os.path.basename(p) for p in fa] == [os.path.basename(p) for p in fr] and \
        all(open(x, "rb").read() == open(y, "rb").read() for x, y in zip(fa, fr))
    check("fire plotfile after restart", same,
          f"{os.path.basename(r_pf)}: {len(fa)} data files {'byte-identical' if same else 'not all byte-identical'}")
    log = open("run_restart.log", errors="replace").read() if os.path.isfile("run_restart.log") else ""
    check("restart read the fire state", "[FIRE] Restoring fire state from checkpoint" in log, "")
    ra, rr = rows("anchor"), rows("restart")
    ok = ra is not None and rr is not None and len(rr) > 1 and rr[1:] == ra[len(ra) - (len(rr) - 1):]
    check("statistics rows after restart", ok, f"{len(rr) - 1 if rr else 0} rows against the straight run's last rows")


def same_data(a, b):
    """(every data file of every level byte-identical, number of files)"""
    fa = sorted(glob.glob(os.path.join(a, "Level_*", "Cell_D_*")))
    fb = sorted(glob.glob(os.path.join(b, "Level_*", "Cell_D_*")))
    same = len(fa) > 0 and [os.path.relpath(p, a) for p in fa] == [os.path.relpath(p, b) for p in fb] and \
        all(open(x, "rb").read() == open(y, "rb").read() for x, y in zip(fa, fb))
    return same, len(fa)


def mrf():
    for kind in ("atm", "fire"):
        s_pf, r_pf = last_plotfile(kind, "heat_mrf"), last_plotfile(kind, "heat_mrf_restart")
        if not s_pf or not r_pf or s_pf[-5:] != r_pf[-5:] or not s_pf.endswith("00040"):
            check(f"MRF {kind} plotfiles", False, f"straight {s_pf}, restarted {r_pf}")
            continue
        same, n = same_data(s_pf, r_pf)
        check(f"MRF {kind} plotfile after restart", same,
              f"{os.path.basename(r_pf)}: {n} data files {'byte-identical' if same else 'not all byte-identical'}")
    log = open("run_heat_mrf_restart.log", errors="replace").read() if os.path.isfile("run_heat_mrf_restart.log") else ""
    check("MRF restart read the fire state", "[FIRE] Restoring fire state from checkpoint" in log, "")
    # the MRF keys are read: the same deck without them gives another atmosphere
    h_pf, m_pf = last_plotfile("atm", "heat"), last_plotfile("atm", "heat_mrf")
    ok = bool(h_pf and m_pf) and h_pf[-5:] == m_pf[-5:] and not same_data(h_pf, m_pf)[0]
    check("MRF changes the atmosphere", ok, f"{h_pf} against {m_pf}")


CHECKS = {"parity": parity, "heat": heat, "restart": restart, "mrf": mrf}


def main():
    for name in (sys.argv[1:] or list(CHECKS)):
        print(f"{name}:")
        CHECKS[name]()
    n_fail = results.count(False)
    print(f"FireAnchorLevel: {len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail or not results else 0)


if __name__ == "__main__":
    main()
