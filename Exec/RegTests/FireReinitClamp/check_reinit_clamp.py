#!/usr/bin/env python3
"""FireReinitClamp check: erf.fire.levelset.reinit_clamp, old against new.

The deck reinitialises the level set every 5th subcycle with 2 pseudo-time
steps of 0.25 dx, stronger than the default. run_reinit_clamp.sh runs it with
each clamp, and with reinit_iters = 0 under each clamp. This script reads
fire_phi of the last fire plotfile of each run and checks:

1. The two clamps give different level sets, but only through the clamp:
   the largest |phi| difference must lie between PHI_MIN and PHI_MAX (4.69 m
   measured). A FireLayer that stopped passing its start-of-subcycle field to
   the reinitialisation gives "step_start" the field of "post_advection" (0 m)
   and fails the lower bound. One that copied the reference once per
   atmospheric step instead of once per subcycle (each 5 s step takes 4
   subcycles here) clamps against a field up to 4 subcycles old and doubles
   the difference (9.94 m), failing the upper bound.
2. The "post_advection" head (the clamp against the advected field, the form
   before 2026-10) runs ahead of the "step_start" one (WRF-Fire's clamp
   against the level set at the start of the subcycle, the default) by at
   least HEAD_MIN m (0.906 m measured): with the old clamp no correction may
   hold the front back. The burned areas are printed (4308.9 and 4330.6 m^2
   here) but not checked: on other settings of this deck their difference
   changed sign during the run.
3. With reinit_iters = 0 the reinitialisation, and with it the clamp, is
   skipped, so the two clamps give the same field (to ZERO_TOL; 0 on one CPU
   rank, while two GPU runs need not agree to the bit). A FireLayer that
   still applied the clamp then cuts back every rise of the advection (at the
   burned minimum of the disc, where the viscous term lifts phi) under
   "step_start" only (0.31 m measured with the clamp left on), and fails
   here.

    python3 check_reinit_clamp.py              # check the runs in this directory
    python3 check_reinit_clamp.py --self-test  # the checker's own pass/fail logic

The exit code is the verdict.
"""
import glob
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_here, os.path.join(_here, "..", "..", "CanonicalTests", "Canonical_RANS")]

PHI_MIN = 0.5          # m, the largest |phi_post - phi_step| must reach this (4.69 m measured)
PHI_MAX = 7.0          # m, ...and stay below this (9.94 m with the reference copied once per step)
HEAD_MIN = 0.3         # m, post_advection head - step_start head (0.906 m measured)
ZERO_TOL = 1.0e-6      # m, reinit_iters = 0 runs (0.31 m with the clamp left on)
TAGS = ("step_start", "post_advection", "iters0_step_start", "iters0_post_advection")


def load_phi(tag):
    import erf_plotfile
    pfs = sorted(p for p in glob.glob("plt_fire_%s_?????" % tag) if os.path.isdir(p))
    if not pfs:
        return None
    hdr, d = erf_plotfile.read_fields(pfs[-1], ["fire_phi"])
    phi = [[d["fire_phi"][i][j][0] for j in range(len(d["fire_phi"][0]))]
           for i in range(len(d["fire_phi"]))]
    return dict(path=pfs[-1], time=hdr["time"], dx=hdr["dx"][0], lo=hdr["prob_lo"][0], phi=phi)


def burned(f):
    return sum(1 for row in f["phi"] for v in row if v < 0.0)


def burned_area(f):
    """Burned area [m^2] with each cell counted by the fraction of it behind the
    front, min(max(1/2 - phi/dx, 0), 1): exact for a straight front in a
    distance function, and it moves with sub-cell changes of the front."""
    dx = f["dx"]
    return sum(min(max(0.5 - v / dx, 0.0), 1.0) for row in f["phi"] for v in row) * dx * dx


def head_x(f):
    """x of the zero crossing furthest downwind along the middle row."""
    phi, dx, lo = f["phi"], f["dx"], f["lo"]
    j = len(phi[0]) // 2
    x = float("nan")
    for i in range(len(phi) - 1):
        a, b = phi[i][j], phi[i + 1][j]
        if a < 0.0 <= b:
            x = lo + (i + 0.5) * dx + dx * a / (a - b)
    return x


def max_diff(f, g):
    return max(abs(a - b) for ra, rb in zip(f["phi"], g["phi"]) for a, b in zip(ra, rb))


def check(fields):
    missing = [t for t in TAGS if fields.get(t) is None]
    if missing:
        print("no fire plotfile for: %s" % ", ".join(missing))
        return 1
    s, p = fields["step_start"], fields["post_advection"]
    z_s, z_p = fields["iters0_step_start"], fields["iters0_post_advection"]
    print("%-24s %8s %12s %14s %10s" % ("run", "t [s]", "burned cells", "burned [m^2]", "head x [m]"))
    for t in TAGS:
        f = fields[t]
        print("%-24s %8.2f %12d %14.2f %10.3f" % (t, f["time"], burned(f), burned_area(f), head_x(f)))
    d_head = head_x(p) - head_x(s)
    d_phi = max_diff(p, s)
    d_zero = max_diff(z_p, z_s)
    print("post_advection - step_start burned area: %.2f m^2" % (burned_area(p) - burned_area(s)))
    rows = [
        ("max |phi_post - phi_step| [m]", d_phi, PHI_MIN <= d_phi <= PHI_MAX),
        ("post_advection - step_start head [m]", d_head, d_head >= HEAD_MIN),
        ("reinit_iters = 0: max |phi difference| [m]", d_zero, d_zero <= ZERO_TOL),
    ]
    print("%-46s %14s %6s" % ("check", "value", "pass"))
    failed = 0
    for name, val, ok in rows:
        failed += 0 if ok else 1
        print("%-46s %14.6g %6s" % (name, val, "yes" if ok else "NO"))
    print("%d check(s) failed" % failed)
    return failed


def self_test():
    """Synthetic fields: the checker must pass a front that ran further with the old
    clamp and identical iters-0 fields, and fail equal clamps, an old-clamp head
    behind the new one, and iters-0 fields that differ."""
    n, dx = 40, 10.0

    def disc(r, bump=0.0):
        phi = [[((((i + 0.5) * dx - 200.0) ** 2 + ((j + 0.5) * dx - 200.0) ** 2) ** 0.5 - r)
                for j in range(n)] for i in range(n)]
        phi[20][20] += bump
        return dict(path="synthetic", time=20.0, dx=dx, lo=0.0, phi=phi)

    good = {"step_start": disc(60.0), "post_advection": disc(65.0),
            "iters0_step_start": disc(60.0), "iters0_post_advection": disc(60.0)}
    ok = check(good) == 0
    bad = []
    for name, case in (("equal clamps", dict(good, post_advection=disc(60.0))),
                       ("old clamp behind", dict(good, post_advection=disc(55.0))),
                       ("clamps too far apart", dict(good, post_advection=disc(70.0))),
                       ("iters 0 differ", dict(good, iters0_step_start=disc(60.0, bump=-0.5)))):
        print("-- self-test case: %s" % name)
        bad.append(check(case) > 0)
    passed = ok and all(bad)
    print("self-test:", "passed" if passed else "FAILED")
    return 0 if passed else 1


def main(argv):
    if "--self-test" in argv:
        return self_test()
    return 1 if check({t: load_phi(t) for t in TAGS}) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
