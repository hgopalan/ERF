#!/usr/bin/env python3
"""Minimal regression check for UCMOneWayInject.

Auto-generated Phase 3.1c placeholder. Replace with physics-specific
assertions when the canonical's expected behavior is validated.
"""
import glob
import os
import re
import sys

try:
    import yt
    try:
        yt.set_log_level("error")
    except Exception:
        pass
    import numpy as np
except ImportError as e:
    print(f"[UCMOneWayInject] SKIP — missing dependency: {e}")
    sys.exit(0)  # SKIP != FAIL for missing deps


def find_final_plotfile():
    pattern = re.compile(r"^plt_\d+$")
    candidates = sorted(
        f for f in glob.glob("plt_*")
        if pattern.match(os.path.basename(f)) and not os.path.basename(f).startswith("plt_ucm")
    )
    return candidates[-1] if candidates else None


def main():
    name = "UCMOneWayInject"
    print(f"[{name}] Regression check")

    pf = find_final_plotfile()
    if not pf:
        print(f"[{name}] FAIL: no main ATM plotfile found")
        return 1

    print(f"[{name}] Loading: {pf}")
    try:
        ds = yt.load(pf)
    except Exception as e:
        print(f"[{name}] FAIL: yt.load raised: {e}")
        return 1

    try:
        cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                              dims=ds.domain_dimensions,
                              fields=[("boxlib", "theta"),
                                      ("boxlib", "x_velocity"),
                                      ("boxlib", "y_velocity")])
        theta = np.array(cg[("boxlib", "theta")])
        u = np.array(cg[("boxlib", "x_velocity")])
        v = np.array(cg[("boxlib", "y_velocity")])
    except Exception as e:
        print(f"[{name}] FAIL: field access raised: {e}")
        return 1

    # Finite check
    for label, arr in [("theta", theta), ("u", u), ("v", v)]:
        if not np.all(np.isfinite(arr)):
            print(f"[{name}] FAIL: {label} contains NaN or Inf")
            return 1
    print(f"[{name}]   Finite check: PASS")

    # Bounds check
    if not (280 < theta.min() and theta.max() < 320):
        print(f"[{name}] FAIL: theta out of [280, 320] K: min={theta.min():.2f} max={theta.max():.2f}")
        return 1
    wmag = np.sqrt(u**2 + v**2)
    if wmag.max() > 30:
        print(f"[{name}] FAIL: wind mag > 30 m/s: max={wmag.max():.2f}")
        return 1
    print(f"[{name}]   Bounds check: PASS")

    # Non-trivial check (solver produced structure, not stuck at IC)
    theta_spread = theta.max() - theta.min()
    if theta_spread < 0.001:
        print(f"[{name}] FAIL: theta spread {theta_spread:.6f} K < 0.001 K (solver may be stuck)")
        return 1
    print(f"[{name}]   Non-trivial check: PASS (theta spread {theta_spread:.3f} K)")

    print(f"[{name}] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
