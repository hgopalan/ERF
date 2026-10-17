#!/usr/bin/env python3
"""Check that a rotor's vertical wake line starts on the ground under its centre.

The vertical wake line behind a rotor stops at the terrain under the line's centre (erf_actuator::WakeLines),
so its lowest point lies on the ground there. With another body's base placed at that centre, the ground
ERF found under that base (moving_bodies/ground.csv) is the ground the line must start on. The check also
refuses a vacuous case, where the ground there equals the ground under the rotor itself.

    check_wake_ground.py --wake D1_wake.csv --ground moving_bodies/ground.csv --rotor D1 --at D2 --xD 1
Exit 0 when the line starts on the ground (to 1e-6 m), 1 otherwise.
"""
import argparse
import csv
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wake", required=True, help="the rotor's <output_root>_wake.csv")
    ap.add_argument("--ground", required=True, help="moving_bodies/ground.csv")
    ap.add_argument("--rotor", required=True, help="the rotor whose wake line is checked")
    ap.add_argument("--at", required=True, help="the body whose base stands at the line's centre")
    ap.add_argument("--xD", type=float, required=True, help="the line's distance behind the rotor (diameters)")
    a = ap.parse_args()
    ground = {r["body"]: float(r["ground"]) for r in csv.DictReader(open(a.ground))}
    for b in (a.rotor, a.at):
        if b not in ground:
            print(f"check_wake_ground.py: {b} is not in {a.ground}")
            return 1
    zs = [float(r["z"]) for r in csv.DictReader(open(a.wake))
          if r["line"] == "vertical" and abs(float(r["xD"]) - a.xD) < 1e-9 and float(r["time"]) == 0.0]
    if not zs:
        print(f"check_wake_ground.py: no vertical line {a.xD:g} D behind {a.rotor} at the first time in {a.wake}")
        return 1
    if abs(ground[a.at] - ground[a.rotor]) < 1.0:
        print(f"check_wake_ground.py: the ground under {a.at} ({ground[a.at]:g} m) is the ground under {a.rotor}: "
              "the check would not tell the two apart")
        return 1
    zmin = min(zs)
    if abs(zmin - ground[a.at]) > 1.0e-6:
        print(f"check_wake_ground.py: the vertical line {a.xD:g} D behind {a.rotor} starts at z = {zmin:.10g} m, "
              f"not on the ground under {a.at} ({ground[a.at]:.10g} m)")
        return 1
    print(f"check_wake_ground.py: the vertical line {a.xD:g} D behind {a.rotor} starts on the ground under {a.at} "
          f"(z = {zmin:.10g} m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
