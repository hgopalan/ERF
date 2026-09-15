#!/usr/bin/env python3
"""Checks on the FireBoundaryGuard regtest, from the statistics CSVs and the run logs.

    python3 check_guard.py            # every variant whose fire_stats_<variant>.csv exists
    python3 check_guard.py warn far   # the named variants only

The log of a variant is run_<variant>.log when the run script wrote it, else
the newest *.log in the directory that carries the "[FIRE] Reach at ignition"
line (a CTest names the log after the test).

warn: the disc's east edge is 10 m from the wall and the centre of the first
      band cell 7 m ahead of it at 1 m/s, so the contact time is 7 s, allowed a
      fire cell either side; the band count is zero before it and positive on
      the last row, the contact time is -1 before it and constant after, and
      the log warns once at ignition for the east edge and once at the contact.
far:  190 m from every wall in a 20 s run: no warning, no contact, zero band
      count on every row.
"""
import csv
import glob
import os
import re
import sys

ROS = 1.0            # erf.fire.prescribed.ros
DX = 2.0             # fire cell [m]
GAP_M = 7.0          # disc east edge (390 m) to the centre of the first band cell (397 m)
COLS = ("edge_band_cells", "edge_contact_time_s")


def read_rows(variant):
    path = f"fire_stats_{variant}.csv"
    if not os.path.isfile(path):
        sys.exit(f"{path}: not found (did the run finish?)")
    rows = list(csv.DictReader(open(path)))
    for c in COLS:
        if c not in rows[0]:
            print(f"  {path}: no column {c} (a binary without the boundary guard): FAIL")
            sys.exit(1)
    return rows


def find_log(variant):
    if os.path.isfile(f"run_{variant}.log"):
        return f"run_{variant}.log"
    logs = [p for p in sorted(glob.glob("*.log"), key=os.path.getmtime, reverse=True)
            if "check" not in p and "Reach at ignition" in open(p, errors="replace").read()]
    return logs[0] if logs else None


def check(name, ok):
    print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return ok


def check_variant(variant):
    rows = read_rows(variant)
    log_path = find_log(variant)
    if log_path is None:
        return check(f"{variant}: a log with the reach estimate", False)
    log = open(log_path, errors="replace").read()
    band = [int(float(r["edge_band_cells"])) for r in rows]
    tc = [float(r["edge_contact_time_s"]) for r in rows]
    t = [float(r["time_s"]) for r in rows]
    reach_lines = re.findall(r"WARNING: the fire can reach the (\S+ \(\S+ \S+\)) edge", log)
    contact = re.search(r"entered the boundary guard band \((\d+) fire cells\) at the (.+?) edge of the fire grid at t = (\S+) s", log)
    print(f"{variant}: {len(rows)} rows, log {log_path}")
    ok = True
    ok &= check("reach estimate printed once at ignition", log.count("[FIRE] Reach at ignition") == 1)
    if variant == "warn":
        first = next((i for i, b in enumerate(band) if b > 0), None)
        ok &= check("band count zero before the contact and positive on the last row",
                    first is not None and all(b == 0 for b in band[:first]) and band[-1] > 0)
        if first is None:
            return False
        t_expect = GAP_M / ROS
        ok &= check(f"contact time {tc[-1]:.2f} s within a cell of {t_expect:.1f} s",
                    abs(tc[-1] - t_expect) <= DX / ROS)
        ok &= check("contact time -1 before the contact, constant after and equal to the first band row's time",
                    all(v == -1.0 for v in tc[:first]) and all(v == tc[first] for v in tc[first:]) and tc[first] == t[first])
        ok &= check("reach warning at ignition for the east edge only",
                    reach_lines == ["east (x hi)"])
        ok &= check("contact warning names the east edge at the contact time",
                    contact is not None and contact.group(2) == "east (x hi)" and abs(float(contact.group(3)) - tc[first]) < 1e-9)
        ok &= check("contact warning printed once", log.count("entered the boundary guard band") == 1)
    elif variant == "far":
        ok &= check("band count zero on every row", all(b == 0 for b in band))
        ok &= check("contact time -1 on every row", all(v == -1.0 for v in tc))
        ok &= check("no reach warning", not reach_lines)
        ok &= check("no contact warning", contact is None)
    else:
        sys.exit(f"no checks defined for variant {variant}")
    return ok


def main():
    variants = sys.argv[1:] or [p[len("fire_stats_"):-len(".csv")] for p in sorted(glob.glob("fire_stats_*.csv"))]
    variants = [v for v in variants if v in ("warn", "far")]
    if not variants:
        sys.exit("no fire_stats_warn.csv or fire_stats_far.csv here")
    ok = all([check_variant(v) for v in variants])
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
