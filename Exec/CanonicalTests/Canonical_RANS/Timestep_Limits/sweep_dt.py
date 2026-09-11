#!/usr/bin/env python3
"""Largest stable time step of three eddy-diffusivity closures under three
integrators, on a RANS-like grid (coarse in x and y, fine in z).

Usage:
  sweep_dt.py --exe ERF_EXEC [--closure kEqn|Deardorff|MRF]... [--steps N]
              [--ladder DT,DT,...] [--mpi-cmd "mpiexec -n 1"]
              [--workdir DIR] [--timeout SECONDS]

For each closure the deck (inputs_dt: 4 x 4 x 200 cells, dx = 800 m,
dz = 5 m) is spun up for 1 h at dt = 5 s with the implicit column solve,
once under the anelastic integrator and once compressible, and
checkpointed. Every rung of the step ladder restarts from that checkpoint
and takes --steps steps at the rung's dt (erf.change_max is lifted so the
new step applies at once instead of growing 10 % per step from 5 s):

  explicit anelastic     anelastic checkpoint,    erf.vert_implicit = false
  implicit anelastic     anelastic checkpoint,    erf.vert_implicit = true
  implicit compressible  compressible checkpoint, erf.vert_implicit = true,
                         acoustic substeps pinned at a fast step of 2 s

The substeps are pinned because ERF otherwise sizes them from the current
state, and a state that is going unstable asks for billions of substeps
and hangs instead of aborting.

The ladder is climbed from the bottom and stops at the first failing rung,
so the reported step is the largest one below the first failure. A rung
passes if ERF exits cleanly and its last plotfile is healthy: finite,
|u| and |v| <= 2 G, |w| <= 1e-2 m/s (the column is horizontally uniform),
theta inside the sounding range +/- 0.5 K and Kmv >= 0.

Checks per closure (the exit code is non-zero if any fails):
  every integrator passes the lowest rung  (otherwise the setup is broken)
  explicit anelastic fails on the ladder   (its limit is bracketed)
  explicit anelastic step / (dz^2 / (2 K/rho)) in [0.5, 2], with K the
      larger of Kmv and Khv anywhere in the restart state
  implicit anelastic step    >= 8 x the explicit anelastic step
  implicit compressible step >= 8 x the explicit anelastic step
When an implicit integrator passes every rung, its step is the top rung,
which is a lower bound.
"""

import argparse
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

# erf_plotfile.py and rans_checks.py live in Canonical_RANS/ next to the case
# directories; the CTest copies them beside this script, so look in both.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..")]
import erf_plotfile  # noqa: E402
from rans_checks import Report  # noqa: E402

DECK = "inputs_dt"
SOUNDING = "input_sounding"
SPINUP_DT = 5.0
SPINUP_STEPS = 720            # 1 h
FAST_DT = 2.0                 # acoustic substep; horizontal CFL about 0.9 at dx = 800 m
GEO_WIND = 10.0
THETA_RANGE = (300.0, 309.0)  # the sounding
THETA_SLACK = 0.5
W_MAX = 1.0e-2
RATIO_MIN = 8.0
PREDICT_BAND = (0.5, 2.0)
LADDER = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
FIELDS = ["density", "x_velocity", "y_velocity", "z_velocity", "theta", "Kmv", "Khv"]

CLOSURES = {
    "kEqn": ["erf.rans_type=kEqn", "erf.dirichlet_k=true", "erf.init_tke_from_ustar=true"],
    "Deardorff": ["erf.les_type=Deardorff", "erf.init_tke_from_ustar=true"],
    "MRF": ["erf.pbl_type=MRF", "erf.pbl_mrf_coriolis_freq=1e-4"],
}
ANELASTIC = ["erf.anelastic=1", "erf.use_fft=true"]
COMPRESSIBLE = ["erf.anelastic=0", "erf.use_fft=false"]
# spin-up integrator per checkpoint family; both use the implicit column solve
FAMILIES = {
    "anelastic": ANELASTIC + ["erf.vert_implicit=true"],
    "compressible": COMPRESSIBLE + ["erf.vert_implicit=true"],
}
# (integrator, checkpoint family, overrides)
MODES = [
    ("explicit anelastic", "anelastic", ANELASTIC + ["erf.vert_implicit=false"]),
    ("implicit anelastic", "anelastic", ANELASTIC + ["erf.vert_implicit=true"]),
    ("implicit compressible", "compressible", COMPRESSIBLE + ["erf.vert_implicit=true"]),
]


def substeps(dt):
    """Even acoustic substep count with a fast step of at most FAST_DT."""
    return max(4, 2 * int(math.ceil(dt / (2.0 * FAST_DT))))


def family_overrides(family, dt):
    if family == "compressible":
        return ["erf.fixed_mri_dt_ratio=%d" % substeps(dt)]
    return []


def run_erf(opts, rundir, args):
    """Run ERF in a fresh directory; return (status, last step, wall seconds)."""
    if os.path.isdir(rundir):
        shutil.rmtree(rundir)
    os.makedirs(rundir)
    shutil.copy(os.path.join(opts.deck_dir, SOUNDING), rundir)
    cmd = shlex.split(opts.mpi_cmd) + [opts.exe, os.path.join(opts.deck_dir, DECK)] + args
    with open(os.path.join(rundir, "cmd"), "w") as fh:
        fh.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
    log = os.path.join(rundir, "log")
    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=rundir, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=opts.timeout)
            status = "ok" if code == 0 else "exit %d" % code
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            status = "timeout"
    wall = time.time() - t0
    last = -1
    with open(log, errors="replace") as fh:
        for line in fh:
            m = re.match(r"Coarse STEP (\d+) ends", line)
            if m:
                last = int(m.group(1))
    return status, last, wall


def abort_reason(rundir):
    with open(os.path.join(rundir, "log"), errors="replace") as fh:
        for line in fh:
            if "is negative or NaN" in line or "Abort" in line or "Error" in line:
                return line.strip()
    return ""


def health(plotfile):
    """Return (ok, reason, hdr, data) for the last plotfile of a run."""
    hdr, d = erf_plotfile.read_fields(plotfile, FIELDS)
    flat = {f: [v for plane in d[f] for col in plane for v in col] for f in FIELDS}
    if not all(math.isfinite(v) for f in FIELDS for v in flat[f]):
        return False, "non-finite field", hdr, flat
    umax = max(max(map(abs, flat["x_velocity"])), max(map(abs, flat["y_velocity"])))
    if umax > 2.0 * GEO_WIND:
        return False, "max |u|,|v| %.3g m/s > %.3g" % (umax, 2.0 * GEO_WIND), hdr, flat
    wmax = max(map(abs, flat["z_velocity"]))
    if wmax > W_MAX:
        return False, "max |w| %.3g m/s > %.3g" % (wmax, W_MAX), hdr, flat
    theta_lo, theta_hi = min(flat["theta"]), max(flat["theta"])
    if theta_lo < THETA_RANGE[0] - THETA_SLACK or theta_hi > THETA_RANGE[1] + THETA_SLACK:
        return False, "theta range %.3f to %.3f K" % (theta_lo, theta_hi), hdr, flat
    kmin = min(flat["Kmv"])
    if kmin < 0.0:
        return False, "min Kmv %.3g < 0" % kmin, hdr, flat
    return True, "", hdr, flat


def plotfile_name(rundir, step):
    return os.path.join(rundir, "plt%05d" % step)


def spin_up(opts, closure, family):
    rundir = os.path.join(opts.workdir, closure, "spinup_" + family)
    args = CLOSURES[closure] + FAMILIES[family] + family_overrides(family, SPINUP_DT) + [
        "erf.fixed_dt=%g" % SPINUP_DT, "max_step=%d" % SPINUP_STEPS,
        "erf.check_int=%d" % SPINUP_STEPS, "erf.plot_int_1=%d" % SPINUP_STEPS]
    status, last, wall = run_erf(opts, rundir, args)
    plt = plotfile_name(rundir, SPINUP_STEPS)
    chk = os.path.join(rundir, "chk%05d" % SPINUP_STEPS)
    ok = status == "ok" and os.path.isdir(plt) and os.path.isdir(chk)
    reason = "" if ok else "%s at step %d %s" % (status, last, abort_reason(rundir))
    flat = None
    if ok:
        ok, reason, _, flat = health(plt)
    print("  spin-up %-12s %-5s %6.1f s  %s" % (family, "ok" if ok else "FAIL", wall, reason))
    if not ok:
        print("    log: %s" % os.path.join(rundir, "log"))
    return ok, chk, plt, flat


def explicit_prediction(plt):
    """dz^2 / (2 K/rho), K = max(Kmv, Khv) over the restart state."""
    hdr, d = erf_plotfile.read_fields(plt, ["density", "Kmv", "Khv"])
    dz = hdr["dx"][2]
    kmax = 0.0
    for i, plane in enumerate(d["density"]):
        for j, col in enumerate(plane):
            for k, rho in enumerate(col):
                kmax = max(kmax, d["Kmv"][i][j][k] / rho, d["Khv"][i][j][k] / rho)
    return dz * dz / (2.0 * kmax), kmax


def climb(opts, closure, mode, family, overrides, chk):
    """Largest passing rung below the first failure, the failure, and the rows."""
    rows = []
    best = None
    failure = None
    end = SPINUP_STEPS + opts.steps
    for dt in opts.ladder:
        rundir = os.path.join(opts.workdir, closure, mode.replace(" ", "_"), "dt_%g" % dt)
        args = CLOSURES[closure] + overrides + family_overrides(family, dt) + [
            "erf.restart=%s" % os.path.abspath(chk), "erf.fixed_dt=%g" % dt,
            "erf.change_max=1.0e9", "max_step=%d" % end,
            "erf.check_int=-1", "erf.plot_int_1=%d" % end]
        status, last, wall = run_erf(opts, rundir, args)
        plt = plotfile_name(rundir, end)
        if status == "ok" and os.path.isdir(plt):
            ok, reason, _, _ = health(plt)
        else:
            ok = False
            reason = "%s after %d steps: %s" % (status, max(last - SPINUP_STEPS, 0), abort_reason(rundir))
        rows.append((dt, ok, reason, wall))
        print("    %-22s dt %8g  %-4s %6.1f s  %s" % (mode, dt, "pass" if ok else "FAIL", wall, reason))
        if not ok:
            failure = dt
            print("      log: %s" % os.path.join(rundir, "log"))
            break
        best = dt
    return best, failure, rows


def sweep_closure(opts, closure, rep, summary):
    print("== %s" % closure)
    chk = {}
    plts = {}
    for family in FAMILIES:
        ok, chk[family], plts[family], _ = spin_up(opts, closure, family)
        rep.check("%s: %s spin-up healthy" % (closure, family), 1.0 if ok else 0.0, 1.0, 0.0)
        if not ok:
            return
    predicted, kmax = explicit_prediction(plts["anelastic"])
    print("  restart state: max K/rho %.3g m2/s, dz^2/(2K) = %.3g s" % (kmax, predicted))

    result = {}
    for mode, family, overrides in MODES:
        best, failure, rows = climb(opts, closure, mode, family, overrides, chk[family])
        result[mode] = (best, failure)
        rep.check("%s: %s passes dt %g" % (closure, mode, opts.ladder[0]),
                  1.0 if rows and rows[0][1] else 0.0, 1.0, 0.0)
    summary.append((closure, predicted, result))

    nan = float("nan")
    expl_best, expl_fail = result["explicit anelastic"]
    rep.check("%s: explicit anelastic fails on the ladder" % closure,
              1.0 if expl_fail is not None else 0.0, 1.0, 0.0)
    ratio = expl_best / predicted if expl_best else nan
    rep.check("%s: explicit step / dz^2/(2K)" % closure, ratio, PREDICT_BAND, 0.0, "range")
    for mode in ("implicit anelastic", "implicit compressible"):
        best = result[mode][0]
        gain = best / expl_best if (best and expl_best) else nan
        rep.check("%s: %s step / explicit step" % (closure, mode), gain, RATIO_MIN, 0.0, "min")


def fmt_step(best, failure, top):
    if best is None:
        return "none (fails %g)" % failure
    if failure is None:
        return ">= %g" % top
    return "%g (fails %g)" % (best, failure)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True, help="erf_exec")
    ap.add_argument("--closure", action="append", choices=sorted(CLOSURES),
                    help="closure to sweep (repeatable; default all three)")
    ap.add_argument("--steps", type=int, default=200, help="steps per rung after the restart")
    ap.add_argument("--ladder", default=",".join("%g" % v for v in LADDER),
                    help="comma-separated time steps, ascending")
    ap.add_argument("--mpi-cmd", default="", help='MPI launcher prefix, e.g. "mpiexec -n 1"')
    ap.add_argument("--workdir", default="dt_runs", help="directory for the runs")
    ap.add_argument("--deck-dir", default=_here, help="directory holding inputs_dt and input_sounding")
    ap.add_argument("--timeout", type=float, default=900.0, help="wall-clock limit per ERF run [s]")
    opts = ap.parse_args()
    opts.exe = os.path.abspath(opts.exe)
    opts.workdir = os.path.abspath(opts.workdir)
    opts.deck_dir = os.path.abspath(opts.deck_dir)
    opts.ladder = sorted(float(v) for v in opts.ladder.split(","))
    closures = opts.closure or ["kEqn", "Deardorff", "MRF"]

    rep = Report()
    summary = []
    t0 = time.time()
    for closure in closures:
        sweep_closure(opts, closure, rep, summary)

    top = opts.ladder[-1]
    print()
    print("Largest step [s] that runs %d steps from the 1 h state (first failing rung in brackets)" % opts.steps)
    print("%-10s %-20s %-20s %-22s %14s" % ("closure", "explicit anelastic", "implicit anelastic",
                                             "implicit compressible", "dz^2/(2K) [s]"))
    for closure, predicted, result in summary:
        cells = [fmt_step(*result[mode], top) for mode, _, _ in MODES]
        print("%-10s %-20s %-20s %-22s %14.3g" % (closure, cells[0], cells[1], cells[2], predicted))
    print()
    rep.dump()
    print("total wall time %.0f s" % (time.time() - t0))
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
