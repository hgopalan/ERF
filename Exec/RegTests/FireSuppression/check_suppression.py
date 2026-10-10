#!/usr/bin/env python3
"""Checks on the FireSuppression regtest, from the last fire plotfile and the
suppression log of a variant.

    python3 check_suppression.py                       # every variant whose suppression_<variant>.csv exists,
                                                       # read with plt_fire_00040 (plt_fire_00080 for poll)
    python3 check_suppression.py hold plt_hold_farsite_00040 suppression_hold_farsite.csv
    python3 check_suppression.py restart plt_restart_straight_levelset_00040 plt_restart_restart_levelset_00040

The deck: 2 m fire cells, a disc of radius 10 m at (200, 200) spreading at a
prescribed 1 m/s in still air, so the unsuppressed east edge is at 230 m
after the 20 s run. Every check names the defect it guards; a plotfile
without the suppression fields (a binary without the feature) fails.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import erf_plotfile  # noqa: E402

DX = 2.0
LINE_X = 221.0      # line_early, line_late, hold, restart
FAR_LINE_X = 241.0  # burnout, poll
FIELDS = ["fire_arrival_time", "fire_suppression_mask", "fire_ros_factor", "fire_line_progress"]


def centre(i):
    return (i + 0.5) * DX


def cell(x):
    """Index of the cell whose centre is x."""
    return int(round(x / DX - 0.5))


def check(name, ok, detail=""):
    ok = bool(ok)
    print(f"  {name}: {'PASS' if ok else 'FAIL'}" + (f"  ({detail})" if detail else ""))
    return ok


def load(plotfile):
    try:
        _, f = erf_plotfile.read_fields(plotfile, FIELDS)
    except KeyError as e:
        print(f"  {plotfile}: {e}: FAIL (a binary without the suppression fields)")
        return None
    return {k: [[v[i][j][0] for j in range(len(v[i]))] for i in range(len(v))] for k, v in f.items()}


def burned_max_x(at, rows):
    """Largest x-centre of a burned cell (arrival time >= 0) over the rows."""
    best = -1.0
    for i in range(len(at)):
        for j in rows:
            if at[i][j] >= 0.0:
                best = max(best, centre(i))
    return best


def events(log, event=None, ident=None):
    if not os.path.isfile(log):
        return []
    rows = list(csv.DictReader(open(log)))
    return [r for r in rows if (event is None or r["event"] == event) and (ident is None or r["id"] == ident)]


def rows_between(y0, y1):
    return [j for j in range(int(400 / DX)) if y0 <= centre(j) <= y1]


def check_line_early(f, log):
    at, m, pr = f["fire_arrival_time"], f["fire_suppression_mask"], f["fire_line_progress"]
    il = cell(LINE_X)
    ok = True
    mx = burned_max_x(at, rows_between(121, 279))
    ok &= check("no burned cell at or beyond the line in the rows it spans (the line stops the head)",
                mx < LINE_X, f"max burned x {mx:.0f} m")
    ok &= check("the front reached the cell before the line in row y = 200 m (the fire got there)",
                at[il - 1][cell(199)] >= 0.0)
    line_cells = [j for j in rows_between(101, 299) if m[il][j] == 1.0 and pr[il][j] == 1.0]
    ok &= check("every line cell holds mask 1 and line ordinal 1 (stamp and progress field)",
                len(line_cells) == len(rows_between(101, 299)), f"{len(line_cells)} cells")
    off = sum(1 for i in range(len(m)) for j in range(len(m[i])) if m[i][j] != 0.0 and i != il)
    ok &= check("no mask cell off the line (one cell wide)", off == 0, f"{off} stray cells")
    ap = events(log, "applied", "L1")
    ok &= check("log: L1 applied once on the first step", len(ap) == 1 and int(ap[0]["step"]) == 1,
                ", ".join(f"t={r['time_s']} step={r['step']}" for r in ap))
    ok &= check("log: L1 completed", len(events(log, "completed", "L1")) == 1)
    return ok


def check_line_late(f, log):
    at, m = f["fire_arrival_time"], f["fire_suppression_mask"]
    il = cell(LINE_X)
    ok = True
    mx = burned_max_x(at, rows_between(199, 201))
    ok &= check("the head ran past the late line in row y = 200 m (burned cells are not stamped)",
                mx >= 227.0, f"max burned x {mx:.0f} m")
    n_line = sum(1 for j in rows_between(99, 301) if m[il][j] == 1.0)
    full = len(rows_between(99, 301))
    ok &= check("the line holds only its unburned cells (fewer than the full line)",
                0 < n_line < full, f"{n_line} of {full}")
    ap = events(log, "applied", "L1")
    co = events(log, "completed", "L1")
    ok &= check("log: L1 applied at t = 15 s and completed with fewer cells than the full line",
                len(ap) == 1 and abs(float(ap[0]["time_s"]) - 15.0) < 1e-9
                and len(co) == 1 and 0 < int(co[0]["cells"]) < full,
                ", ".join(f"{r['event']} t={r['time_s']} cells={r['cells']}" for r in ap + co))
    return ok


def check_drop_hold(f, log):
    at = f["fire_arrival_time"]
    ok = True
    mx = burned_max_x(at, rows_between(199, 201))
    ok &= check("head at 20 s in row y = 200 m between 221 and 227 m (held 6 s, then released)",
                221.0 <= mx <= 227.0, f"max burned x {mx:.0f} m")
    ap = events(log, "applied", "D1")
    ex = events(log, "expired", "D1")
    ok &= check("log: D1 applied on step 1 and expired at t = 12 s",
                len(ap) == 1 and int(ap[0]["step"]) == 1 and len(ex) == 1 and abs(float(ex[0]["time_s"]) - 12.0) < 1e-9,
                ", ".join(f"{r['event']} t={r['time_s']}" for r in ap + ex))
    ok &= check("log: the drop covered every cell of the polygon at t = 0 (42 x 100 cells, all unburned)",
                len(ap) == 1 and int(ap[0]["cells"]) == 4200, ap[0]["cells"] if ap else "")
    return ok


def check_drop_slow(f, log):
    at, fac = f["fire_arrival_time"], f["fire_ros_factor"]
    ok = True
    mx = burned_max_x(at, rows_between(199, 201))
    ok &= check("head at 20 s in row y = 200 m between 217 and 223 m (0.3 m/s inside the drop)",
                217.0 <= mx <= 223.0, f"max burned x {mx:.0f} m")
    inside = [fac[i][j] for i in range(cell(217), cell(299) + 1) for j in rows_between(101, 299) if at[i][j] < 0.0]
    outside = [fac[i][j] for i in range(0, cell(213) + 1) for j in range(len(fac[0]))]
    ok &= check("fire_ros_factor is 0.3 in every unburned cell of the drop and 1 outside it",
                inside and all(abs(v - 0.3) < 1e-12 for v in inside) and all(v == 1.0 for v in outside),
                f"{len(inside)} inside cells")
    ok &= check("log: D1 applied on step 1, never expired",
                len(events(log, "applied", "D1")) == 1 and not events(log, "expired", "D1"))
    return ok


def check_hold(f, log):
    at, m = f["fire_arrival_time"], f["fire_suppression_mask"]
    il = cell(LINE_X)
    ok = True
    failed = [centre(j) for j in range(len(m[il])) if m[il][j] == -1.0]
    held = [centre(j) for j in range(len(m[il])) if m[il][j] == 1.0 and at[il - 1][j] >= 0.0]
    ok &= check("some line cells failed the hold test (mask -1)", len(failed) > 0, f"{len(failed)} cells")
    ok &= check("every failed cell lies north of y = 195 m (the hotter side; the limit sits at y = 198 m)",
                failed and min(failed) > 195.0, f"y {min(failed) if failed else 'n/a'}..{max(failed) if failed else 'n/a'}")
    ok &= check("line cells with a burned neighbour south of y = 195 m all held (mask 1)",
                held and min(held) < 190.0 and
                all(m[il][j] == 1.0 for j in rows_between(150, 195) if at[il - 1][j] >= 0.0),
                f"{len(held)} held cells with a burned neighbour, y {min(held) if held else 'n/a'}..{max(held) if held else 'n/a'}")
    beyond = [centre(j) for j in range(len(at[0])) if any(at[i][j] >= 0.0 for i in range(il + 1, il + 6))]
    ok &= check("the fire crossed the line (burned cells beyond it) only north of y = 190 m",
                beyond and min(beyond) > 190.0, f"crossed rows y {min(beyond) if beyond else 'n/a'}..{max(beyond) if beyond else 'n/a'}")
    hf = events(log, "hold_failed", "L1")
    ok &= check("log: hold_failed events for L1 with the cell counts summing to the failed cells",
                hf and sum(int(r["cells"]) for r in hf) == len(failed), f"{len(hf)} events")
    return ok


def check_burnout(f, log):
    at = f["fire_arrival_time"]
    il = cell(FAR_LINE_X)
    ok = True
    strip = [at[il - 1][j] for j in rows_between(1, 399)]
    ok &= check("the cell before the line burned in every row long before the main front (burnout)",
                all(0.0 <= t <= 15.0 for t in strip), f"arrival {min(strip):.1f}..{max(strip):.1f} s")
    mx = burned_max_x(at, rows_between(1, 399))
    ok &= check("nothing burned beyond the line (it spans the whole domain)", mx < FAR_LINE_X, f"max burned x {mx:.0f} m")
    bo = events(log, "burnout", "B1")
    ok &= check("log: one burnout event at t = 5 s with one ignition point per cell of the 400 m line",
                len(bo) == 1 and abs(float(bo[0]["time_s"]) - 5.0) < 1e-9 and int(bo[0]["cells"]) == 200,
                ", ".join(f"t={r['time_s']} points={r['cells']} {r['detail']}" for r in bo))
    ok &= check("log: the burnout side is west (the disc's side)", bo and "side left" in bo[0]["detail"])
    return ok


def check_poll(f, log):
    at = f["fire_arrival_time"]
    il = cell(FAR_LINE_X)
    ok = True
    rr = events(log, "reread")
    ap = events(log, "applied", "L1")
    ok &= check("log: the file was re-read with one new action", any(int(r["cells"]) == 1 for r in rr),
                ", ".join(f"step={r['step']} new={r['cells']}" for r in rr))
    ok &= check("log: L1 applied by a poll (step > 1) before the front could reach it (step <= 60)",
                len(ap) == 1 and 1 < int(ap[0]["step"]) <= 60, ", ".join(f"step={r['step']}" for r in ap))
    mx = burned_max_x(at, rows_between(121, 279))
    ok &= check("no burned cell at or beyond the line in the rows it spans at 40 s", mx < FAR_LINE_X, f"max burned x {mx:.0f} m")
    ok &= check("the front reached the cell before the line in row y = 200 m", at[il - 1][cell(199)] >= 0.0)
    return ok


def check_restart(plt_a, plt_b):
    hdr = erf_plotfile._read_header(plt_a)
    names = hdr["names"]
    ok = True
    _, a = erf_plotfile.read_fields(plt_a, names)
    _, b = erf_plotfile.read_fields(plt_b, names)
    for n in names:
        diff = max(abs(a[n][i][j][0] - b[n][i][j][0]) for i in range(len(a[n])) for j in range(len(a[n][0])))
        ok &= check(f"{n} identical", diff == 0.0, f"max |diff| {diff:g}")
    mask = a["fire_suppression_mask"]
    n_line = sum(1 for i in range(len(mask)) for j in range(len(mask[0])) if mask[i][j][0] == 1.0)
    # 97.5 m built at the start of step 40 (50 cells from y = 99 m), less the
    # rows the front had burned at x = 221 m before the line got there
    ok &= check("the line is half built at 20 s (40 to 50 unburned line cells)", 40 <= n_line <= 50, f"{n_line} cells")
    return ok


CHECKS = {"line_early": check_line_early, "line_late": check_line_late, "drop_hold": check_drop_hold,
          "drop_slow": check_drop_slow, "hold": check_hold, "burnout": check_burnout, "poll": check_poll}


def run_variant(variant, plotfile, log):
    print(f"{variant}: {plotfile}, {log}")
    if not os.path.isdir(plotfile):
        return check("plotfile exists", False, plotfile)
    f = load(plotfile)
    if f is None:
        return False
    return CHECKS[variant](f, log)


def main():
    args = sys.argv[1:]
    if args and args[0] == "restart":
        if len(args) != 3:
            sys.exit("usage: check_suppression.py restart <straight plotfile> <restarted plotfile>")
        print(f"restart: {args[1]} vs {args[2]}")
        ok = check_restart(args[1], args[2])
    elif args:
        variant = args[0]
        if variant not in CHECKS:
            sys.exit(f"unknown variant {variant}")
        plotfile = args[1] if len(args) > 1 else ("plt_fire_00080" if variant == "poll" else "plt_fire_00040")
        log = args[2] if len(args) > 2 else f"suppression_{variant}.csv"
        ok = run_variant(variant, plotfile, log)
    else:
        variants = [p[len("suppression_"):-len(".csv")] for p in sorted(glob.glob("suppression_*.csv"))]
        variants = [v for v in variants if v in CHECKS]
        if not variants:
            sys.exit("no suppression_<variant>.csv here")
        ok = all([run_variant(v, "plt_fire_00080" if v == "poll" else "plt_fire_00040", f"suppression_{v}.csv")
                  for v in variants])
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
