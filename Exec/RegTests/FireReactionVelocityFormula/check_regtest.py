#!/usr/bin/env python3
"""FireReactionVelocityFormula check: for each combination of
erf.fire.reaction_velocity_formula ("albini" | "rothermel") and
erf.fire.wrf_bmst_compat (false | true), calm and at 4.005 m/s, does the head
rate ERF writes (head_ros_ms of the fire statistics CSV) equal the
closed-form Rothermel rate of that combination?

The closed form below is written from the papers (Rothermel 1972 with
Albini's 1976 A, or Rothermel's own Eq. 39 A) in ERF's single-class layout:
the 1-h SAV as the bed SAV and w_n = w_0 (1 - S_T). It checks that the two
options reach the solver on the uniform-fuel path and on the per-fuel table of
a fuel map; it is not a comparison with WRF-Fire.

Pass/fail: every run's relative error must be below TOL_PCT, and every pair of
combinations at the same wind must differ by more than SEP_PCT, so a run that
ignored an option cannot pass by sitting inside the tolerance of its neighbour.

    python3 check_regtest.py              # check the runs in this directory
    python3 check_regtest.py --self-test  # the checker's own pass/fail logic
"""

import csv
import itertools
import math
import os
import sys
import tempfile

WINDS = {"calm": 0.0, "wind": 4.005}
M1 = 0.06
TOL_PCT = 0.01   # the CSV carries six significant digits
SEP_PCT = 0.003  # the albini pair at 4.005 m/s differs by 0.006 %
FM1 = dict(w0=0.034, sigma=3500.0, delta=1.0, Mx=0.12, h=8000.0, S_T=0.0555, S_e=0.010, rho_p=32.0)
DECKS = {
    "albini":             ("albini", False),
    "albini_bmst":        ("albini", True),
    "rothermel":          ("rothermel", False),
    "rothermel_bmst":     ("rothermel", True),
    "rothermel_bmst_map": ("rothermel", True),
}


def rothermel_fm1(formula, wrf_bmst_compat, M_f, U):
    """Head rate [m/s] of FM1 (only a 1-h load, so the dead-weighted moisture
    is M1) at midflame wind U [m/s], no slope, no wind cap."""
    fp = FM1
    w0 = fp['w0']
    if wrf_bmst_compat:
        w0 = w0 * (1.0 - M_f / (1.0 + M_f))
    w_n = w0 * (1 - fp['S_T'])
    rho_b = w0 / fp['delta']
    beta = rho_b / fp['rho_p']
    s = max(fp['sigma'], 100.0)
    beta_op = 3.348 * s ** -0.8189
    s15 = s ** 1.5
    Gmax = s15 / (495.0 + 0.0594 * s15)
    A = (1.0 / (4.774 * s ** 0.1 - 7.27)) if formula == "rothermel" else (133.0 * s ** -0.7913)
    br = beta / beta_op
    Gp = Gmax * br ** A * math.exp(A * (1 - br))
    rm = min(M_f / fp['Mx'], 1.0)
    eta_M = max(0.0, 1 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3)
    eta_s = 0.174 * fp['S_e'] ** -0.19
    I_R = Gp * w_n * fp['h'] * eta_M * eta_s
    xi = math.exp((0.792 + 0.681 * math.sqrt(s)) * (beta + 0.1)) / (192.0 + 0.2595 * s)
    eps_h = math.exp(-138.0 / s)
    Q_ig = 250.0 + 1116.0 * M_f
    R0_ftmin = (I_R * xi) / (rho_b * eps_h * Q_ig)
    C = 7.47 * math.exp(-0.133 * s ** 0.55)
    B = 0.02526 * s ** 0.54
    E = 0.715 * math.exp(-3.59e-4 * s)
    phi_w = C * ((U * 196.85) ** B) * (br ** -E)
    return R0_ftmin * (1.0 + phi_w) * 0.00508


def head_ros(csv_file):
    with open(csv_file) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"'{csv_file}' has no data rows")
    return float(rows[-1]["head_ros_ms"])


def check(rates):
    """rates: {(deck, wind): head rate}. Returns the list of failures."""
    failures = []
    for (deck, wind), ros in sorted(rates.items()):
        formula, bmst = DECKS[deck]
        ref = rothermel_fm1(formula, bmst, M1, WINDS[wind])
        err = abs(ros - ref) / ref * 100.0
        flag = "ok" if err < TOL_PCT else "FAIL"
        print(f"{deck:20s} {wind:5s} formula={formula:9s} wrf_bmst_compat={bmst!s:5s} "
              f"ERF {ros:.6g} m/s  closed form {ref:.6g} m/s  error {err:.4f} %  {flag}")
        if err >= TOL_PCT:
            failures.append(f"{deck} {wind}: error {err:.4f} % >= {TOL_PCT} %")
    for wind, u in WINDS.items():
        refs = {d: rothermel_fm1(*DECKS[d], M1, u) for d in ("albini", "albini_bmst", "rothermel", "rothermel_bmst")}
        for a, b in itertools.combinations(refs, 2):
            sep = abs(refs[a] - refs[b]) / refs[a] * 100.0
            if sep <= SEP_PCT:
                failures.append(f"{a} and {b} {wind}: closed forms only {sep:.4f} % apart, test cannot tell them apart")
    return failures


def run_checks(directory):
    rates = {}
    for deck in DECKS:
        for wind in WINDS:
            path = os.path.join(directory, f"fire_stats_{deck}_{wind}.csv")
            try:
                rates[(deck, wind)] = head_ros(path)
            except (OSError, ValueError, KeyError) as e:
                return [f"cannot read {path}: {e} (run run_regtest.sh first)"]
    return check(rates)


def self_test():
    """The right rates pass; a run that ignored wrf_bmst_compat, or one that
    ignored reaction_velocity_formula, fails."""
    def write(directory, override):
        for deck in DECKS:
            for wind, u in WINDS.items():
                formula, bmst = override.get(deck, DECKS[deck])
                with open(os.path.join(directory, f"fire_stats_{deck}_{wind}.csv"), "w") as f:
                    f.write("step,head_ros_ms\n")
                    f.write(f"5,{rothermel_fm1(formula, bmst, M1, u):.6g}\n")
    cases = [("right rates", {}, True),
             ("albini_bmst ignores wrf_bmst_compat", {"albini_bmst": ("albini", False)}, False),
             ("rothermel deck ignores the formula", {"rothermel": ("albini", False)}, False),
             ("map deck ignores both options", {"rothermel_bmst_map": ("albini", False)}, False)]
    ok = True
    for name, override, should_pass in cases:
        with tempfile.TemporaryDirectory() as d:
            write(d, override)
            passed = not run_checks(d)
        print(f"self-test: {name}: {'passed' if passed else 'failed'} (expected {'pass' if should_pass else 'fail'})")
        ok = ok and (passed == should_pass)
    return ok


def main():
    if "--self-test" in sys.argv[1:]:
        sys.exit(0 if self_test() else "FAIL: the checker's own pass/fail logic is wrong")
    failures = run_checks(".")
    print()
    if failures:
        sys.exit("FAIL:\n  " + "\n  ".join(failures))
    print(f"PASS: every head rate within {TOL_PCT} % of its closed form")


if __name__ == "__main__":
    main()
