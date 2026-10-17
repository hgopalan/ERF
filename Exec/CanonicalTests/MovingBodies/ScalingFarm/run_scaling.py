#!/usr/bin/env python3
"""Run the scaling farm for several farm sizes, rank counts and ERF binaries, and tabulate the timers.

For each mode, N and rank count it writes the deck (make_scaling_farm.py), runs every binary on it,
reads the "Moving bodies timing" summaries of the run log (erf.moving_bodies.timing_int) and prints one
markdown table: the start-up and each phase's mean over the ranks per step, from the steady summaries
(all but the first). With two binaries it also checks that their outputs (the farm logs) are identical.
Each run waits until the 1-minute load average is below --max-load, so that runs on a shared machine
yield to other work.

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
    """Every timing summary of a run log: (first step, last step, {phase: (min, mean, max)}, turbine line)."""
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


def outputs_identical(a, b):
    for sub in ("moving_bodies",):
        if not (os.path.isdir(os.path.join(a, sub)) and os.path.isdir(os.path.join(b, sub))):
            return False
        cmp = filecmp.dircmp(os.path.join(a, sub), os.path.join(b, sub))
        if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", action="append", required=True, help="label=path of an erf_exec; give one or two")
    ap.add_argument("--n", type=int, nargs="+", default=[16, 64, 256, 1024], help="farm sizes (squares)")
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4], help="MPI rank counts")
    ap.add_argument("--mode", nargs="+", default=["adm"], choices=["adm", "alm"])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--work", default="scaling_runs", help="directory for the decks and runs (created)")
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--max-load", type=float, default=6.0, help="start a run only below this 1-minute load average")
    ap.add_argument("--phases", default="startup,advance,sample,openfast,spread,diagnostics,add_sources",
                    help="phases to tabulate (comma-separated)")
    a = ap.parse_args()
    exes = [e.split("=", 1) for e in a.exe]
    if len(exes) > 2 or any(len(e) != 2 for e in exes):
        raise SystemExit("run_scaling.py: --exe label=path, once or twice")
    phases = a.phases.split(",")

    rows = []
    for mode in a.mode:
        for n in a.n:
            deck_dir = os.path.join(a.work, f"{mode}_n{n}", "deck")
            subprocess.run([sys.executable, os.path.join(HERE, "make_scaling_farm.py"), "--n", str(n), "--mode", mode,
                            "--steps", str(a.steps), "--out", deck_dir], check=True, stdout=subprocess.DEVNULL)
            for np_ in a.ranks:
                runs = []
                for label, exe in exes:
                    run = os.path.join(a.work, f"{mode}_n{n}", f"{label}_np{np_}")
                    shutil.rmtree(run, ignore_errors=True)
                    shutil.copytree(deck_dir, run)
                    wait_for_load(a.max_load)
                    t0 = time.time()
                    with open(os.path.join(run, "run.log"), "w") as log:
                        rc = subprocess.run([a.mpiexec, "-n", str(np_), exe, "farm.i"], cwd=run, stdout=log,
                                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
                    wall = time.time() - t0
                    text = open(os.path.join(run, "run.log")).read()
                    s = summaries(text)
                    if rc != 0 or not s:
                        print(f"run_scaling.py: {run} failed (exit {rc}); see {run}/run.log", file=sys.stderr)
                    runs.append(run)
                    first = s[0][2] if s else {}
                    steady = s[1:] if len(s) > 1 else s
                    nsteps = sum(x[1] - x[0] + 1 for x in steady)
                    row = {"mode": mode, "n": n, "ranks": np_, "exe": label, "rc": rc, "wall": wall,
                           "startup": first.get("startup", (0, 0, 0))[1]}
                    for p in phases:
                        if p == "startup":
                            continue
                        tot = sum(x[2].get(p, (0, 0, 0))[1] for x in steady)
                        row[p] = tot / max(nsteps, 1)
                    row["turbine"] = steady[-1][3] if steady else ""
                    rows.append(row)
                    print(f"{mode} N={n} ranks={np_} {label}: exit {rc}, {wall:.1f} s", file=sys.stderr, flush=True)
                if len(runs) == 2:
                    same = outputs_identical(runs[0], runs[1])
                    rows[-1]["same"] = rows[-2]["same"] = "yes" if same else "NO"

    cols = ["mode", "n", "ranks", "exe", "startup"] + [p for p in phases if p != "startup"]
    head = ["mode", "N", "ranks", "binary", "startup (s)"] + [f"{p} (s/step)" for p in phases if p != "startup"]
    if len(exes) == 2:
        head.append("outputs identical")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        cells = [str(r[c]) if c in ("mode", "n", "ranks", "exe") else f"{r[c]:.3g}" for c in cols]
        if len(exes) == 2:
            cells.append(r.get("same", ""))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
