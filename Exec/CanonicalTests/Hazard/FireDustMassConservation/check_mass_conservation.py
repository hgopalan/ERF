#!/usr/bin/env python3
"""Close the dust mass budget of FireDustMassConservation.

    erf_exec inputs > run.log && python3 check_mass_conservation.py [run.log] [dust_diag.dat]
    python3 check_mass_conservation.py --self-test   # the checker on a synthetic closed and a leaky budget

The deck sets erf.dust.deposition_E0 = 0, which reduces the deposition velocity
to the settling velocity (it does not switch deposition off: dust still leaves
the air through the surface at v_s). erf.sum_interval prints "RHO DUST = M"
every 10 steps, the volume integral of the airborne dust density [kg]. The dust
diagnostics give the emission rate E [kg/s] of every step (summed over the bins
and the cells, times the cell area) and the cumulative deposited mass D [kg].
The flux computed at step n is injected during step n+1 (the documented
one-step lag), so at the step printed

    M_air(n) = sum_{m < n} E_m dt - D(n)

within 2 % of the emitted mass. Until October 2026 the case asserted nothing
and its header claimed the airborne mass was non-decreasing.
"""
import re
import sys

TOL = 0.02


def read_diag(diag):
    """dust_diag.dat rows: (step, time_s, emission_total_kg_s, deposition_total_kg)."""
    rows = []
    for line in open(diag):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("step"):
            continue
        f = s.split(",")
        rows.append((int(f[0]), float(f[1]), float(f[2]), float(f[3])))
    return rows


def read_log(log):
    """run.log: "TIME= t" then " RHO DUST   = M" -> (times, masses)."""
    times, masses = [], []
    t_cur = None
    for line in open(log):
        m = re.match(r"\s*TIME=\s*([0-9.eE+-]+)", line)
        if m:
            t_cur = float(m.group(1)); continue
        m = re.match(r"\s*RHO DUST\s*=\s*([0-9.eE+-]+)", line)
        if m and t_cur is not None:
            times.append(t_cur); masses.append(float(m.group(1)))
    return times, masses


def budget(rows, times, masses, quiet=False):
    """The three checks; returns the list of booleans."""
    results = []

    def check(name, ok, detail):
        results.append(ok)
        if not quiet:
            print(f"  {name:10s} {'PASS' if ok else 'FAIL'}  {detail}")

    by_step = {r[0]: r for r in rows}
    dt = rows[2][1] - rows[1][1]
    worst = 0.0
    emitted_total = 0.0
    for t, M in zip(times, masses):
        n = int(round(t / dt))
        if n not in by_step:
            continue
        emitted = sum(by_step[m][2] * dt for m in by_step if m < n)
        deposited = by_step[n][3]
        expected = emitted - deposited
        emitted_total = max(emitted_total, emitted)
        scale = max(emitted, 1e-300)
        err = abs(M - expected) / scale
        worst = max(worst, err)
    check("emitted", emitted_total > 0.0, f"{emitted_total:.4e} kg emitted by the last printed step")
    check("deposited", by_step[max(by_step)][3] > 0.0,
          f"{by_step[max(by_step)][3]:.4e} kg deposited (settling at v_s, E_0 = 0)")
    check("budget", worst < TOL,
          f"max |M_air - (emitted - deposited)| / emitted = {worst:.3e} over {len(masses)} prints (tol {TOL})")
    return results


def self_test():
    """The checker's own logic: a closed synthetic budget passes, a 10 % leak fails."""
    dt, E = 1.0, 70.0                      # 70 kg/s for 100 steps
    rows = [(n, n * dt, E, 0.5 * E * n * dt) for n in range(101)]   # half of the emission deposits
    times = [10.0 * k for k in range(1, 11)]
    closed = [sum(E * dt for m in range(int(t))) - 0.5 * E * t for t in times]
    ok_pass = budget(rows, times, closed, quiet=True)
    leaky = [0.9 * M for M in closed]      # 10 % of the airborne mass lost
    ok_fail = budget(rows, times, leaky, quiet=True)
    good = all(ok_pass) and ok_fail[:2] == [True, True] and ok_fail[2] is False
    print(f"  self-test  {'PASS' if good else 'FAIL'}  closed budget {sum(ok_pass)}/3, 10 % leak fails the budget check: {not ok_fail[2]}")
    sys.exit(0 if good else 1)


if "--self-test" in sys.argv[1:]:
    self_test()

args = [a for a in sys.argv[1:] if not a.startswith("--")]
log = args[0] if len(args) > 0 else "run.log"
diag = args[1] if len(args) > 1 else "dust_diag.dat"

try:
    rows = read_diag(diag)
except (OSError, ValueError, IndexError) as e:
    print(f"  mass: cannot read the diagnostics file {diag} ({e}); usage: check_mass_conservation.py [run.log] [dust_diag.dat]: FAIL")
    sys.exit(1)
if len(rows) < 3:
    print(f"  mass: {len(rows)} rows in {diag}: FAIL"); sys.exit(1)
try:
    times, masses = read_log(log)
except OSError as e:
    print(f"  mass: cannot read the run log {log} ({e}): FAIL"); sys.exit(1)
if len(masses) < 3:
    print(f"  mass: {len(masses)} RHO DUST lines in {log}: FAIL"); sys.exit(1)

results = budget(rows, times, masses)
n_pass = sum(results)
print(f"FireDustMassConservation: {n_pass}/{len(results)} checks passed")
sys.exit(0 if n_pass == len(results) else 1)
