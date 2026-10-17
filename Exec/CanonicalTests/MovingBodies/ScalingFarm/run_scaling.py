#!/usr/bin/env python3
"""Run the scaling farm for several farm sizes, rank counts and ERF binaries, and tabulate the timers.

For each farm and rank count it writes the deck (make_scaling_farm.py), runs every binary on it (in an order
that alternates from one rank count to the next, so that neither binary always runs first), reads the "Moving
bodies timing" summaries of the run log (erf.moving_bodies.timing_int) and prints one markdown table:
- the start-up (the constructor, set_ground) as the maximum over the ranks, and the first step's start-up
  checks and wake-line set-up (coverage, audits), also the maximum, from the first summary;
- each phase's mean over the ranks per step from the later summaries (n/a when the run gave only one);
- with two binaries, whether their farm logs (everything under moving_bodies/, to the 10 digits they are
  printed with) are identical.
A run that failed shows its exit code in the status column and no numbers. Each run waits until the 1-minute
load average is below --max-load, so that runs on a shared machine yield to other work.

    run_scaling.py --exe base=/path/erf_exec_a --exe new=/path/erf_exec_b --n 16 64 256 1024 --ranks 1 2 4
"""
import argparse
import filecmp
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def summaries(log):
    """Every timing summary of a run log: [first step, last step, {phase: (min, mean, max)}, turbine line]."""
    out, cur = [], None
    for line in log.splitlines():
        m = re.match(r"Moving bodies timing, steps (\d+) to (\d+) ", line)
        if m:
            cur = [int(m.group(1)), int(m.group(2)), {}, ""]
            out.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"  (\w+)\s+(\S+)\s+(\S+)\s+(\S+)$", line)
        if m:
            cur[2][m.group(1)] = tuple(float(v) for v in m.group(2, 3, 4))
        elif line.startswith("  OpenFAST step per turbine"):
            cur[3] = line.strip()
        else:
            cur = None
    return out


def wait_for_load(limit):
    while os.getloadavg()[0] >= limit:
        time.sleep(30)


def same_tree(a, b):
    """Whether two directories hold the same files with the same contents, at every depth."""
    if not (os.path.isdir(a) and os.path.isdir(b)):
        return False
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    if mismatch or errors:
        return False
    return all(same_tree(os.path.join(a, d), os.path.join(b, d)) for d in cmp.common_dirs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", action="append", required=True, help="label=path of an erf_exec; give one or two")
    ap.add_argument("--n", type=int, nargs="+", default=[16, 64, 256, 1024], help="farm sizes (squares)")
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4], help="MPI rank counts")
    ap.add_argument("--mode", nargs="+", default=["adm"], choices=["adm", "alm"])
    ap.add_argument("--farm-args", default="",
                    help="more make_scaling_farm.py options, e.g. \"--bodies ct_disk --wake-lines-xD 1 2 --stretched\"")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--timing-int", type=int, default=2)
    ap.add_argument("--work", default="scaling_runs", help="directory for the decks and runs (created)")
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--max-load", type=float, default=6.0, help="start a run only below this 1-minute load average")
    ap.add_argument("--phases", default="advance,sample,openfast,spread,diagnostics,wake,add_sources",
                    help="phases to tabulate per step (comma-separated)")
    a = ap.parse_args()
    exes = [e.split("=", 1) for e in a.exe]
    if len(exes) > 2 or any(len(e) != 2 for e in exes):
        raise SystemExit("run_scaling.py: --exe label=path, once or twice")
    phases = a.phases.split(",")

    rows = []
    turn = 0
    for mode in a.mode:
        for n in a.n:
            deck_dir = os.path.join(a.work, f"{mode}_n{n}", "deck")
            subprocess.run([sys.executable, os.path.join(HERE, "make_scaling_farm.py"), "--n", str(n), "--mode", mode,
                            "--steps", str(a.steps), "--timing-int", str(a.timing_int), "--out", deck_dir] + a.farm_args.split(),
                           check=True, stdout=subprocess.DEVNULL)
            for np_ in a.ranks:
                order = exes if turn % 2 == 0 else list(reversed(exes))
                turn += 1
                runs = {}
                for label, exe in order:
                    run = os.path.join(a.work, f"{mode}_n{n}", f"{label}_np{np_}")
                    shutil.rmtree(run, ignore_errors=True)
                    shutil.copytree(deck_dir, run)
                    wait_for_load(a.max_load)
                    t0 = time.time()
                    with open(os.path.join(run, "run.log"), "w") as log:
                        rc = subprocess.run([a.mpiexec, "-n", str(np_), exe, "farm.i"], cwd=run, stdout=log,
                                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
                    wall = time.time() - t0
                    s = summaries(open(os.path.join(run, "run.log")).read())
                    row = {"mode": mode, "n": n, "ranks": np_, "exe": label, "wall": wall, "status": "ok"}
                    if rc != 0 or not s:
                        row["status"] = f"FAILED (exit {rc})"
                        print(f"run_scaling.py: {run} failed (exit {rc}); see {run}/run.log", file=sys.stderr)
                    else:
                        first = s[0][2]
                        row["startup"] = first.get("startup", (0, 0, 0))[2]
                        row["checks"] = first.get("checks", (0, 0, 0))[2]
                        row["wake setup"] = first.get("wake", (0, 0, 0))[2]
                        steady = s[1:]
                        nsteps = sum(x[1] - x[0] + 1 for x in steady)
                        for p in phases:
                            row[p] = (sum(x[2].get(p, (0, 0, 0))[1] for x in steady) / nsteps) if nsteps else None
                    runs[label] = (run, row)
                    print(f"{mode} N={n} ranks={np_} {label}: {row['status']}, {wall:.1f} s", file=sys.stderr, flush=True)
                for label, _ in exes:
                    rows.append(runs[label][1])
                if len(exes) == 2:
                    (ra, xa), (rb, xb) = runs[exes[0][0]], runs[exes[1][0]]
                    ok = xa["status"] == "ok" and xb["status"] == "ok"
                    same = (("yes" if same_tree(os.path.join(ra, "moving_bodies"), os.path.join(rb, "moving_bodies")) else "NO")
                            if ok else "n/a")
                    xa["same"] = xb["same"] = same

    first_cols = ["startup", "checks", "wake setup"]
    head = (["mode", "N", "ranks", "binary", "status", "startup max (s)", "step-1 checks max (s)", "wake set-up max (s)"] +
            [f"{p} mean (s/step)" for p in phases])
    if len(exes) == 2:
        head.append("farm logs identical")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        cells = [r["mode"], str(r["n"]), str(r["ranks"]), r["exe"], r["status"]]
        for c in first_cols + phases:
            v = r.get(c)
            cells.append("" if r["status"] != "ok" else ("n/a" if v is None else f"{v:.3g}"))
        if len(exes) == 2:
            cells.append(r.get("same", ""))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
