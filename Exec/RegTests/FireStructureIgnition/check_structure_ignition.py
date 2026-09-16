#!/usr/bin/env python3
"""Checks for the FireStructureIgnition suite. Reads the exposure CSVs, the
logs and the fire plotfiles (standard library only; erf_plotfile.py from
Canonical_RANS is copied beside it by the CTest, or found in the tree);
prints PASS/FAIL per check and exits 1 if any fails.

    python3 check_structure_ignition.py on radiation restart
    python3 check_structure_ignition.py old off      # must FAIL on the old path

Checks (see README.md):
  on         A (id 1) ignites from the heat load at its wall; at least one other
             house ignites later; A burns out before the end and its release is
             zero then; while burning its release was positive; the plotfile
             carries the state, the ignition time and the incident flux
  radiation  no spotting, ember criterion off: A ignites by heat, B (id 2)
             ignites by heat later although the front never reached its wall
             band (house-to-house spread by radiation alone), B's band saw a
             positive incident flux before it ignited, C (id 3) stays unignited
  restart    the run restarted from step 20 reproduces the step-40 fire
             plotfile to round-off (1e-9 relative per field; the inflow
             atmosphere of this deck restarts to about 1e-11, with or without
             structure ignition) and the last CSV row of every structure
             exactly
  old        the CSV of the ignition-off run has no state column, so every
             ignition assertion fails: the check must fail on the old path
"""
import csv, glob, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "CanonicalTests", "Canonical_RANS"))
import erf_plotfile as pf

results = []
def check(name, ok, detail):
    results.append(bool(ok))
    print(f"  {name:34s} {'PASS' if ok else 'FAIL'}  {detail}")

def rows(v):
    """Rows by structure id, in time order."""
    out = {}
    with open(f"exposure_{v}.csv") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(int(r["structure_id"]), []).append(r)
    return out

def f(r, k):
    return float(r[k])

def has_ignition_columns(by_id):
    r = next(iter(by_id.values()))[0]
    return all(k in r for k in ("state", "t_ignition_s", "cause", "structure_flux_Wm2", "incident_flux_max_Wm2"))

def last_plotfile(v):
    files = sorted(glob.glob(f"plt_fire_{v}_?????"))
    return files[-1] if files else None

def fields(plotfile, names):
    """The k = 0 plane of each named field as a list of columns [i][j]; None when absent."""
    hdr = pf._read_header(plotfile)
    have = [n for n in names if n in hdr["names"]]
    _, data = pf.read_fields(plotfile, have) if have else (hdr, {})
    return {n: ([[col[0] for col in data[n][i]] for i in range(len(data[n]))] if n in have else None)
            for n in names}, hdr

def cells(mask2d, value):
    return [(i, j) for i, col in enumerate(mask2d) for j, v in enumerate(col) if abs(v - value) < 0.5]

def check_ignition_variant(v, need_second_from_radiation):
    print(f"== {v}")
    by_id = rows(v)
    if not has_ignition_columns(by_id):
        check("ignition columns in the CSV", False, "the exposure CSV has no state/cause columns (ignition off)")
        return
    check("ignition columns in the CSV", True, "state, t_ignition_s, cause, structure_flux_Wm2, incident_flux_max_Wm2")
    last = {s: rs[-1] for s, rs in by_id.items()}
    check("three structures", len(last) == 3, f"{len(last)} ids in the CSV")
    a = last.get(1); b = last.get(2); c = last.get(3)
    if a is None or b is None or c is None:
        return
    t_a = f(a, "t_ignition_s")
    check("A ignites from the heat at its wall", a["cause"] == "heat" and t_a > 0.0,
          f"cause={a['cause']} t_ignition={t_a} s heat_load_max={f(a, 'heat_load_max_MJm2'):.3f} MJ/m2")
    others = [(s, f(r, "t_ignition_s"), r["cause"]) for s, r in last.items() if s != 1 and r["state"] != "unignited"]
    later = [o for o in others if o[1] > t_a]
    check("a second structure ignites later", len(later) >= 1,
          ", ".join(f"id {s} at {t} s by {cz}" for s, t, cz in others) or "none")
    check("A burns out before the end", a["state"] == "burned_out" and f(a, "structure_flux_Wm2") == 0.0,
          f"state={a['state']} release now {f(a, 'structure_flux_Wm2'):.0f} W/m2")
    q_mid = max(f(r, "structure_flux_Wm2") for r in by_id[1])
    check("A released heat while burning", q_mid > 0.0, f"largest release in the CSV {q_mid:.0f} W/m2")
    if need_second_from_radiation:
        t_b = f(b, "t_ignition_s")
        check("B ignites by heat", b["cause"] == "heat" and t_b > t_a, f"cause={b['cause']} t_ignition={t_b} s")
        check("the front never reached B's band", f(b, "t_first_s") < 0.0 and f(b, "wall_burned_frac") == 0.0,
              f"t_first={f(b, 't_first_s')} wall_burned_frac={f(b, 'wall_burned_frac')}")
        q_inc = max(f(r, "incident_flux_max_Wm2") for r in by_id[2] if f(r, "time_s") <= t_b)
        check("B saw A's radiation before igniting", q_inc > 0.0, f"largest incident flux on B's band {q_inc:.0f} W/m2")
        check("C stays unignited", c["state"] == "unignited" and f(c, "t_ignition_s") < 0.0,
              f"state={c['state']} heat_load_max={f(c, 'heat_load_max_MJm2'):.3f} MJ/m2")
    pf = last_plotfile(v)
    if pf is None:
        check("fire plotfile", False, "none"); return
    F, _ = fields(pf, ["fire_structure_state", "fire_structure_ignition_time", "fire_structure_rad_flux", "fire_structure_id"])
    ok = all(F[n] is not None for n in F)
    check("plotfile carries the ignition fields", ok, "fire_structure_state, fire_structure_ignition_time, fire_structure_rad_flux")
    if ok:
        sid = F["fire_structure_id"]; st = F["fire_structure_state"]; ti = F["fire_structure_ignition_time"]
        foot_a = cells(sid, 1.0); off = cells(sid, 0.0)
        st_a = sorted(set(st[i][j] for i, j in foot_a)); ti_a = sorted(set(ti[i][j] for i, j in foot_a))
        check("A's footprint holds its state", len(foot_a) > 0 and st_a == [2.0] and len(ti_a) == 1 and abs(ti_a[0] - t_a) < 1e-9,
              f"state {st_a} ignition time {ti_a}")
        check("unignited cells carry the sentinel", all(ti[i][j] == -1.0 and st[i][j] == 0.0 for i, j in off),
              "state 0 and ignition time -1 off the footprints")
    # The incident flux in the step-10 plotfile (t = 5 s, A at its peak) against
    # the point-source sum recomputed here from A's footprint and the release
    # the CSV reports for that step: chi_r q dA / (2 pi max(r^2, (dx/2)^2))
    # over A's cells within the cutoff. Independent of the kernel in the code.
    pf10 = f"plt_fire_{v}_00010"
    if not glob.glob(pf10):
        check("radiation kernel at step 10", False, f"no {pf10}"); return
    G, hdr = fields(pf10, ["fire_structure_rad_flux", "fire_structure_id"])
    dx, dy = hdr["dx"][0], hdr["dx"][1]
    t10 = hdr["time"] - 0.5                     # the release is evaluated at the start of the step
    q_a = [f(r, "structure_flux_Wm2") for r in by_id[1] if abs(f(r, "time_s") - t10) < 1e-6]
    if not q_a:
        check("radiation kernel at step 10", False, f"no CSV row of A at t = {t10} s"); return
    chi, R = 0.3, 60.0                           # inputs_on: rad_fraction, rad_radius_m
    P = chi * q_a[0] * dx * dy
    sid = G["fire_structure_id"]; rad = G["fire_structure_rad_flux"]
    nx, ny = len(sid), len(sid[0])
    # every burning structure's footprint cells radiate (B is burning too by
    # step 10 in the ember variant); the release of each from its CSV row
    sources = []
    for s2 in (1, 2, 3):
        q_s = [f(r, "structure_flux_Wm2") for r in by_id[s2] if abs(f(r, "time_s") - t10) < 1e-6]
        if q_s and q_s[0] > 0.0:
            P2 = chi * q_s[0] * dx * dy
            sources += [((i + 0.5) * dx, (j + 0.5) * dy, P2) for i, j in cells(sid, float(s2))]
    r2_min = 0.25 * dx * dx
    err = 0.0; scale = 0.0
    for i in range(nx):
        for j in range(ny):
            x, y = (i + 0.5) * dx, (j + 0.5) * dy
            e = 0.0
            for sx, sy, P2 in sources:
                r2 = (x - sx) ** 2 + (y - sy) ** 2
                if r2 <= R * R:
                    e += P2 / (2.0 * math.pi * max(r2, r2_min))
            err = max(err, abs(rad[i][j] - e)); scale = max(scale, e)
    check("radiation kernel at step 10", scale > 0.0 and err <= 1e-9 * scale,
          f"A releases {q_a[0]:.0f} W/m2; incident max {scale:.0f} W/m2 recomputed, largest difference {err:.2e} W/m2")

def check_restart():
    print("== restart")
    a = last_plotfile("on"); b = last_plotfile("restart")
    if a is None or b is None:
        check("plotfiles present", False, f"on: {a} restart: {b}"); return
    names = sorted(pf._read_header(a)["names"])
    _, da = pf.read_fields(a, names); _, db = pf.read_fields(b, names)
    worst = 0.0; bad = []; exact_bad = []
    exact = ["fire_structure_state", "fire_structure_ignition_time", "fire_structure_rad_flux",
             "fire_heat_load", "fire_ember_landings", "fire_peak_intensity"]
    for n in names:
        xa = [v for col in da[n] for row in col for v in row]
        xb = [v for col in db[n] for row in col for v in row]
        scale = max(max(abs(v) for v in xa), 1.0e-30)
        diff = max(abs(u - v) for u, v in zip(xa, xb))
        rel = diff / scale
        worst = max(worst, rel)
        if rel > 1.0e-9:
            bad.append(f"{n} ({rel:.1e})")
        if n in exact and diff != 0.0:
            exact_bad.append(n)
    check("step-40 plotfile reproduced after restart", not bad,
          f"{len(names)} fields, largest relative difference {worst:.1e}" + (f"; beyond 1e-9: {bad}" if bad else ""))
    # the structure state, the accumulators and the incident flux restart exactly
    check("structure fields identical after restart", not exact_bad, ", ".join(exact) + (f"; differ: {exact_bad}" if exact_bad else ""))
    ra = rows("on"); rb = rows("restart")
    same = True; detail = []
    for s in sorted(ra):
        la = ra[s][-1]; lb = rb.get(s, [{}])[-1]
        for k in la:
            if la[k] != lb.get(k):
                same = False; detail.append(f"id {s} {k}: {la[k]} vs {lb.get(k)}")
    check("last CSV rows identical after restart", same, "; ".join(detail) if detail else f"{len(ra)} structures")

if __name__ == "__main__":
    args = sys.argv[1:] or ["on", "radiation", "restart"]
    for a in args:
        if a == "on":          check_ignition_variant("on", False)
        elif a == "radiation": check_ignition_variant("radiation", True)
        elif a == "restart":   check_restart()
        elif a == "old":       check_ignition_variant("off", True)
        elif a.startswith("old:"):  # old:<variant>
            check_ignition_variant(a.split(":", 1)[1], True)
        else:
            sys.exit(f"unknown check {a}")
    print("RESULT:", "pass" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)
