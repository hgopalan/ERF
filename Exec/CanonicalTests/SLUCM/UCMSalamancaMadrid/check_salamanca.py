#!/usr/bin/env python3
import glob
import os
import re
import sys

import numpy as np

try:
    import yt

    try:
        yt.set_log_level("error")
    except Exception:
        pass
except ImportError:
    print("ERROR: yt not found. Install with: pip install yt")
    sys.exit(1)


def list_main_plotfiles():
    pattern = re.compile(r"^plt_\d+$")
    entries = sorted(glob.glob("plt_*"))
    return [
        p
        for p in entries
        if pattern.match(os.path.basename(p))
        and not os.path.basename(p).startswith("plt_ucm")
    ]


def load_field(ds, field):
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions, fields=[field])
    return np.array(cg[field])


def main():
    plotfiles = list_main_plotfiles()
    if not plotfiles:
        print("FAIL: no main ATM plotfiles found")
        return 1

    print(f"Found {len(plotfiles)} main ATM plotfiles")

    deltas = []
    theta_profile_center = None
    z_profile = None

    for pf in plotfiles:
        ds = yt.load(pf)
        theta = load_field(ds, ("boxlib", "theta"))

        nx, ny, nz = theta.shape
        j_mid = ny // 2
        i_center = nx // 2
        i_upwind = 0
        k_near_surface = 1 if nz > 1 else 0

        t_urban = float(theta[i_center, j_mid, k_near_surface])
        t_rural = float(theta[i_upwind, j_mid, k_near_surface])
        deltas.append(t_urban - t_rural)

        theta_profile_center = theta[i_center, j_mid, :]
        dz = (ds.domain_right_edge[2] - ds.domain_left_edge[2]) / nz
        z_profile = np.array(ds.domain_left_edge[2] + (np.arange(nz) + 0.5) * dz)

    mean_delta = float(np.mean(deltas))
    print("\nUrban-rural near-surface temperature diagnostics")
    print(f"  Mean(Theta_urban - Theta_upwind) over plotfiles: {mean_delta:.4f} K")

    print("\nCenter-column theta profile (diagnostic)")
    sample_stride = max(1, len(theta_profile_center) // 8)
    for k in range(0, len(theta_profile_center), sample_stride):
        print(f"  k={k:3d} z={z_profile[k]:8.2f} m theta={theta_profile_center[k]:8.3f} K")

    if mean_delta <= 0.0:
        print("FAIL: expected urban near-surface theta to exceed upwind rural surrogate")
        return 1

    print("PASS: urban near-surface theta exceeds upwind rural surrogate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
