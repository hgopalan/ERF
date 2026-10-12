#!/usr/bin/env python3
"""Check that an OpenFAST actuator disk puts its shaft thrust into the flow, row by row.

An actuator disk (erf.moving_bodies.<name>.mode = adm) replaces each blade node by a ring of
num_points_t >= 2 points carrying the node's force along the shaft and its radial and tangential
parts turned with the ring, which cancel around the ring; the hub node's force is kept as one
point, and OpenFAST's external-inflow interface sets it to zero (ExternalInflow.f90). On a tilted
shaft (the IEA 15 MW's is 6 degrees) the momentum source therefore integrates to minus the thrust's
shaft component, (T . n) n, not to minus the full thrust vector T: the x components differ by the
tilt and the blades' net in-plane load. A model whose hub node carried a force off the shaft would
fail this check without any defect in ERF. This compares
fx of the source log with -(T . n) n_x of the turbine log, T = (thrust_x, thrust_y, thrust_z) and
n = (axis_x, axis_y, axis_z), and requires them to agree to rel_tol in every row. It also requires
-thrust_x to differ from fx by more than rel_tol in at least one row, so the check is not the plain
x identity in disguise (a turbine whose loads are all along x would pass that one).

The turbine log's first row is the initial solution before any step; the source log's first row
belongs to the state after the first step, so turbine row r + 1 pairs with source row r.

Usage: check_openfast_shaft_source.py --turbine T1_erf.csv --source moving_bodies/momentum_source.csv
                                      [--rel-tol 1e-8]
"""

import argparse
import csv
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--turbine", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--rel-tol", type=float, default=1.0e-8)
    a = p.parse_args()

    turb = list(csv.DictReader(open(a.turbine)))
    src = list(csv.DictReader(open(a.source)))
    for name, rows, cols in ((a.turbine, turb, ["time"] + ["thrust_" + c for c in "xyz"] + ["axis_" + c for c in "xyz"]),
                             (a.source, src, ["time", "fx"])):
        missing = [c for c in cols if not rows or c not in rows[0]]
        if missing:
            print(f"check_openfast_shaft_source.py: {name} lacks the column(s) {', '.join(missing)}")
            return 1
    if len(turb) != len(src) + 1:
        print(f"check_openfast_shaft_source.py: {a.turbine} has {len(turb) - 1} stepped rows but {a.source} has {len(src)}")
        return 1
    if len(src) < 2:
        print(f"check_openfast_shaft_source.py: only {len(src)} rows; the check would be trivial")
        return 1

    worst, x_differs = 0.0, False
    for r, s in enumerate(src):
        t = turb[r + 1]
        # source row r (the state after a step, at time t) pairs with turbine row r + 1 (that step's loads, t + dt)
        if not float(s["time"]) < float(t["time"]):
            print(f"check_openfast_shaft_source.py: row {r + 1}: the source time {s['time']} is not before the turbine "
                  f"time {t['time']}; the logs are not paired as expected")
            return 1
        thrust = [float(t["thrust_" + c]) for c in "xyz"]
        axis = [float(t["axis_" + c]) for c in "xyz"]
        shaft_x = sum(f * n for f, n in zip(thrust, axis)) * axis[0]
        fx = float(s["fx"])
        if shaft_x == 0.0:
            print(f"check_openfast_shaft_source.py: row {r + 1}: the shaft thrust is zero; the turbine carried no load")
            return 1
        rel = abs(fx + shaft_x) / abs(shaft_x)
        worst = max(worst, rel)
        x_differs = x_differs or abs(fx + thrust[0]) > a.rel_tol * abs(thrust[0])
        print(f"row {r + 1}: fx {fx:.10g}, -(T.n) n_x {-shaft_x:.10g} (rel {rel:.1e}), -thrust_x {-thrust[0]:.10g}")
    if worst > a.rel_tol:
        print(f"check_openfast_shaft_source.py: fx differs from -(T.n) n_x by {worst:.2e} (relative) > {a.rel_tol:g}; "
              "the disk does not put its shaft thrust into the flow")
        return 1
    if not x_differs:
        print("check_openfast_shaft_source.py: fx equals -thrust_x in every row as well; the loads lie along x and the "
              "shaft projection is not tested by this case")
        return 1
    print(f"check_openfast_shaft_source.py: fx equals -(T.n) n_x to {worst:.1e} in {len(src)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
