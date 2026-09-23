#!/usr/bin/env python3
"""Checks of the deck-defined fuel models (erf.fire.custom_fuel.*).

    python3 check_custom_fuel.py identity run_anderson1.log run_custom_grass.log
    python3 check_custom_fuel.py identity run_custom_map.log run_custom_map_altid.log I_B_max L_max
    python3 check_custom_fuel.py summary  run_custom_grass.log 1000
    python3 check_custom_fuel.py fuel     run_custom_map.log fuel_map_mixed.asc 1.25
    python3 check_custom_fuel.py slower   run_anderson1.log run_custom_heavy.log 2.0
    python3 check_custom_fuel.py crossed  plt_fire_custom_map/plt_fire_00480 1000 102 2.0
    python3 check_custom_fuel.py abort    run_bad_depth.log depth_m
    python3 check_custom_fuel.py --selftest

identity: two runs that must agree. Not bitwise: a deck carries its SI values
          to eight digits, so the comparison is relative, at IDENT_TOL. Any
          trailing arguments name `label=` quantities exempted from the
          comparison, for a known defect the run is not meant to guard; the
          check reports an exemption that no longer differs so it can be
          dropped once the defect is fixed.
summary:  the [FIRE DEBUG] custom fuel line reports back the SI the deck gave,
          which is the round trip through FuelModelParams' US units.
fuel:     the initial fuel on the grid equals the sum over the raster of each
          cell's own model load, the deck-defined code included and the
          non-burnable codes contributing nothing.
slower:   the coarse deck-defined bed spreads at most 1/factor of the grass
          baseline, so the deck's properties really drive the spread.
crossed:  in the mixed raster the front has burned cells of both codes, and the
          rate of spread inside the deck-defined block is the block's own, at
          most 1/factor of the published model's around it. Needs yt.
abort:    the run stopped with the custom fuel message naming that input.
"""
import re
import sys

import numpy as np

# Round-off of the eight-digit SI values in the decks, not a physics tolerance.
IDENT_TOL = 1.0e-9

LB_FT2_TO_KG_M2 = 4.88243
TPA_TO_LB_FT2 = 2000.0 / 43560.0

# Total oven-dry load [kg/m2] of the codes the mixed map uses. Only totals are
# needed: the herbaceous curing transfer moves load between classes, it does
# not change the sum.
MAP_LOAD_KG_M2 = {
    102: 1.10 * TPA_TO_LB_FT2 * LB_FT2_TO_KG_M2,   # Scott-Burgan GR2, 1-h + live herbaceous
    1000: 22.0,                                    # the deck block: 2.0 + 20.0 kg/m2
    91: 0.0,                                       # non-burnable urban
    98: 0.0,                                       # non-burnable water
}


def fail(msg):
    print(f"  FAIL {msg}")
    return False


def ok(msg):
    print(f"  PASS {msg}")
    return True


def debug_lines(path):
    """The per-step fire numbers a run prints, as a list of comparable strings."""
    keep = re.compile(r"active fire cells|Current max heat flux|max_ROS=")
    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if keep.search(ln)]


def numbers(line):
    return [float(t) for t in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]


def labelled_numbers(line):
    """(label, value) for every `label=value` on a line, plus bare numbers.

    The labels let a comparison exempt a named quantity without loosening the
    tolerance on everything else.
    """
    out = [(m.group(1), float(m.group(2)))
           for m in re.finditer(r"(\w+)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)]
    if out:
        return out
    return [("", v) for v in numbers(line)]


def check_identity(log_a, log_b, exempt=()):
    a, b = debug_lines(log_a), debug_lines(log_b)
    if not a:
        return fail(f"{log_a} has no fire debug lines")
    if len(a) != len(b):
        return fail(f"{log_a} has {len(a)} fire debug lines, {log_b} has {len(b)}")

    worst, worst_at, worst_label = 0.0, "", ""
    seen_exempt = set()
    for la, lb in zip(a, b):
        na, nb = labelled_numbers(la), labelled_numbers(lb)
        if len(na) != len(nb):
            return fail(f"line shape differs:\n    {la}\n    {lb}")
        for (label, xa), (_, xb) in zip(na, nb):
            denom = max(abs(xa), 1.0)
            rel = abs(xa - xb) / denom
            if label in exempt:
                if rel > IDENT_TOL:
                    seen_exempt.add(label)
                continue
            if rel > worst:
                worst, worst_at, worst_label = rel, la, label
    if worst > IDENT_TOL:
        return fail(f"{worst_label or 'a value'} differs by {worst:.3e} "
                    f"(> {IDENT_TOL:.0e}) at:\n    {worst_at}")

    note = ""
    if exempt:
        missing = [e for e in exempt if e not in seen_exempt]
        # An exemption that never fires is a stale exemption: say so rather than
        # quietly keeping it once the underlying defect is fixed.
        note = (f"; exempt and differing: {sorted(seen_exempt) or 'none'}"
                + (f"; exempt but now equal (drop them): {missing}" if missing else ""))
    return ok(f"the runs agree over {len(a)} lines "
              f"(worst relative difference {worst:.3e}){note}")


def check_summary(log, code):
    pat = re.compile(rf"code\s+{code}\s+slot\s+(\d+).*?load=([-\d.eE+]+) kg/m2"
                     r"\s+sav_1h=([-\d.eE+]+) 1/m\s+depth=([-\d.eE+]+) m"
                     r"\s+Mx=([-\d.eE+]+)\s+h=([-\d.eE+]+) J/kg"
                     r"\s+rho_p=([-\d.eE+]+) kg/m3")
    with open(log) as f:
        m = next((pat.search(ln) for ln in f if pat.search(ln)), None)
    if m is None:
        return fail(f"{log} has no custom fuel summary line for code {code}")
    slot, load, sav, depth, mx, heat, rho = (int(m.group(1)),) + tuple(float(m.group(i)) for i in range(2, 8))

    good = True
    if slot < 54:
        good = fail(f"code {code} reported slot {slot}, which is a published set's slot")
    # The deck for code 1000 in inputs_custom_grass is the Anderson 1 round trip.
    for name, got, want_v in (("load", load, 0.16600262), ("sav_1h", sav, 11482.94),
                              ("depth", depth, 0.3048), ("Mx", mx, 0.12),
                              ("heat", heat, 1.8608e7), ("rho_p", rho, 512.592)):
        if abs(got - want_v) > 1.0e-6 * max(abs(want_v), 1.0):
            good = fail(f"code {code} {name}: reported {got:.8g}, deck gave {want_v:.8g}")
    if good:
        ok(f"code {code} reports back the deck's SI at slot {slot}")
    return good


def read_asc(path):
    with open(path) as f:
        header = [f.readline().split() for _ in range(6)]
        meta = {k.lower(): v for k, v in header}
        rows = [[int(float(v)) for v in ln.split()] for ln in f if ln.strip()]
    return np.array(rows), float(meta["cellsize"])


def check_fuel(log, asc, dx):
    codes, cellsize = read_asc(asc)
    if abs(cellsize - dx) > 1.0e-9:
        return fail(f"{asc} cell size {cellsize} is not the fire cell size {dx}")

    unknown = sorted(set(codes.flatten().tolist()) - set(MAP_LOAD_KG_M2))
    if unknown:
        return fail(f"{asc} holds codes this check has no load for: {unknown}")

    want = sum(MAP_LOAD_KG_M2[c] * int((codes == c).sum()) for c in MAP_LOAD_KG_M2) * dx * dx
    m = re.search(r"Fuel load from the map: ([-\d.eE+]+) kg", open(log).read())
    if m is None:
        return fail(f"{log} never printed the fuel load from the map")
    got = float(m.group(1))

    # The deck-defined block has to be a real share of the load, or this check
    # would pass on a run that ignored it.
    share = MAP_LOAD_KG_M2[1000] * int((codes == 1000).sum()) * dx * dx / want
    if share < 0.5:
        return fail(f"the deck-defined block is only {share:.1%} of the load; "
                    "the check would not see it being ignored")
    if abs(got - want) > 1.0e-6 * want:
        return fail(f"initial fuel {got:.4f} kg, the map's models sum to {want:.4f} kg")
    return ok(f"initial fuel {got:.1f} kg matches the map's models "
              f"({share:.0%} of it from the deck-defined block)")


def max_ros(log):
    vals = re.findall(r"max_ROS=([-\d.eE+]+)", open(log).read())
    return float(vals[-1]) if vals else None


def check_slower(log_fast, log_slow, factor):
    fast, slow = max_ros(log_fast), max_ros(log_slow)
    if fast is None or slow is None:
        return fail("one of the runs printed no max_ROS")
    if not fast > 0.0:
        return fail(f"the baseline spread is {fast}, so the comparison is vacuous")
    if slow * factor > fast:
        return fail(f"the coarse deck fuel spread at {slow:.4g} m/s, not below "
                    f"{fast:.4g}/{factor:g} = {fast / factor:.4g} m/s")
    return ok(f"the coarse deck fuel spreads at {slow:.4g} m/s against the "
              f"baseline's {fast:.4g} m/s")


def check_crossed(plotfile, block_code, other_code, factor):
    import yt

    yt.set_log_level(50)
    ds = yt.load(plotfile)
    names = [f for _, f in ds.field_list]
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)

    def field(sub):
        match = [n for n in names if sub in n]
        if not match:
            raise KeyError(f"{plotfile} has no field matching {sub}")
        return grid[("boxlib", match[0])].value[:, :, 0]

    phi = field("fire_phi")
    code = field("fire_fuel_model_code")
    ros = field("fire_ros")

    block = np.abs(code - block_code) < 0.5
    other = np.abs(code - other_code) < 0.5
    if not block.any():
        return fail(f"no cell carries code {block_code}")
    if not other.any():
        return fail(f"no cell carries code {other_code}")

    burned = phi < 0.0
    n_block = int((burned & block).sum())
    n_other = int((burned & other).sum())
    if n_block == 0 or n_other == 0:
        return fail(f"the front burned {n_block} cells of code {block_code} and "
                    f"{n_other} of code {other_code}; it has to burn in both")

    r_block = float(ros[block].max())
    r_other = float(ros[other].max())
    if not r_other > 0.0:
        return fail(f"code {other_code} has no spread, so the comparison is vacuous")
    if r_block * factor > r_other:
        return fail(f"code {block_code} spreads at {r_block:.4g} m/s, not below "
                    f"{r_other:.4g}/{factor:g} = {r_other / factor:.4g} m/s: the "
                    "deck-defined properties are not reaching the kernel")
    return ok(f"the front burned {n_block} cells of the deck-defined code {block_code} "
              f"at {r_block:.4g} m/s and {n_other} of code {other_code} at "
              f"{r_other:.4g} m/s")


def check_abort(log, key):
    text = open(log).read()
    m = re.search(r"amrex::Abort.*?ERF-Fire custom fuel: (.*?) !!!", text, re.S)
    if m is None:
        return fail(f"{log} did not abort with a custom fuel message")
    msg = " ".join(m.group(1).split())
    if key not in msg:
        return fail(f"{log} aborted, but the message does not name {key}: {msg}")
    return ok(f"aborts naming {key}: {msg}")


def selftest():
    """The checker's own pass/fail logic, fed values just outside each band."""
    import os
    import tempfile

    good = True
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.log")
        b = os.path.join(d, "b.log")
        line = "[FIRE DEBUG] ... max_ROS=0.2160244347 m/s\n"
        open(a, "w").write(line)

        # identity: a difference just above the tolerance must fail
        open(b, "w").write("[FIRE DEBUG] ... max_ROS=0.2160244400 m/s\n")
        if check_identity(a, b):
            good = fail("selftest: identity passed a difference above the tolerance")
        # and one just below must pass
        open(b, "w").write("[FIRE DEBUG] ... max_ROS=0.2160244347000001 m/s\n")
        if not check_identity(a, b):
            good = fail("selftest: identity failed a difference below the tolerance")
        # a log with no fire lines is a failure, not a pass
        open(b, "w").write("nothing here\n")
        if check_identity(b, b):
            good = fail("selftest: identity passed a log with no fire debug lines")

        # slower: equal spreads must fail
        open(b, "w").write(line)
        if check_slower(a, b, 2.0):
            good = fail("selftest: slower passed two equal spreads")
        if not check_slower(a, b, 1.0):
            good = fail("selftest: slower failed at factor 1 on equal spreads")

        # abort: a clean log must fail, and a wrong key must fail
        open(b, "w").write("all fine\n")
        if check_abort(b, "depth_m"):
            good = fail("selftest: abort passed a log that did not abort")
        open(b, "w").write("amrex::Abort::0::ERF-Fire custom fuel: depth_m is bad !!!\n")
        if check_abort(b, "heat_content_J_kg"):
            good = fail("selftest: abort passed a message naming another input")
        if not check_abort(b, "depth_m"):
            good = fail("selftest: abort failed its own message")

    print("  selftest:", "PASS" if good else "FAIL")
    return good


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    if mode == "--selftest":
        return 0 if selftest() else 1
    if mode == "identity":
        return 0 if check_identity(sys.argv[2], sys.argv[3], tuple(sys.argv[4:])) else 1
    if mode == "summary":
        return 0 if check_summary(sys.argv[2], int(sys.argv[3])) else 1
    if mode == "fuel":
        return 0 if check_fuel(sys.argv[2], sys.argv[3], float(sys.argv[4])) else 1
    if mode == "slower":
        return 0 if check_slower(sys.argv[2], sys.argv[3], float(sys.argv[4])) else 1
    if mode == "crossed":
        return 0 if check_crossed(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                                  float(sys.argv[5])) else 1
    if mode == "abort":
        return 0 if check_abort(sys.argv[2], sys.argv[3]) else 1
    print(f"unknown mode {mode}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
