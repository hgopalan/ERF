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


def find_final_plotfile():
    pattern = re.compile(r"^plt_\d+$")
    candidates = [
        p
        for p in sorted(glob.glob("plt_*"))
        if pattern.match(os.path.basename(p))
        and not os.path.basename(p).startswith("plt_ucm")
    ]
    return candidates[-1] if candidates else None


def load_field(ds, field):
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions, fields=[field])
    return np.array(cg[field])


def main():
    pf = find_final_plotfile()
    if pf is None:
        print("FAIL: no main ATM plotfiles found")
        return 1

    ds = yt.load(pf)
    u = load_field(ds, ("boxlib", "x_velocity"))
    v = load_field(ds, ("boxlib", "y_velocity"))
    U = np.sqrt(u * u + v * v)

    nx, ny, nz = U.shape
    j_mid = ny // 2

    dz = float((ds.domain_right_edge[2] - ds.domain_left_edge[2]) / nz)
    z = np.array((np.arange(nz) + 0.5) * dz)

    H = 15.0
    z_over_h_targets = [0.25, 0.5, 0.75, 1.0]
    z_targets = [x * H for x in z_over_h_targets]

    stripe_info = [
        (0.11, 4),
        (0.25, 14),
        (0.33, 24),
        (0.44, 34),
    ]

    # Approximate values interpolated from Kanda et al. (2004) Fig. 6
    refs = {
        0.11: [0.30, 0.55, 0.78, 1.00],
        0.25: [0.20, 0.42, 0.70, 1.00],
        0.33: [0.15, 0.35, 0.65, 1.00],
        0.44: [0.10, 0.28, 0.58, 1.00],
    }

    erf_norm = {}

    print("lambda_p    z/H     U_ERF/U_H    U_Kanda/U_H    error_pct")
    for lam, ucm_i in stripe_info:
        atm_i = min(nx - 1, max(0, ucm_i // 4))
        prof = U[atm_i, j_mid, :]
        k_h = int(np.argmin(np.abs(z - H)))
        U_h = float(prof[k_h])
        if U_h <= 1.0e-10:
            print(f"FAIL: U_H too small for lambda_p={lam:.2f}")
            return 1

        norm_vals = []
        for zoh, zt, ref_val in zip(z_over_h_targets, z_targets, refs[lam]):
            k = int(np.argmin(np.abs(z - zt)))
            val = float(prof[k] / U_h)
            norm_vals.append(val)
            err = 100.0 * (val - ref_val) / ref_val if ref_val > 1.0e-10 else 0.0
            print(f"{lam:7.2f}  {zoh:6.2f}   {val:10.3f}    {ref_val:10.3f}    {err:8.2f}")
        erf_norm[lam] = norm_vals

    interior_levels = [0, 1, 2]
    lambdas = [0.11, 0.25, 0.33, 0.44]
    for lev in interior_levels:
        vals = [erf_norm[lam][lev] for lam in lambdas]
        if not all(vals[i] >= vals[i + 1] - 1.0e-6 for i in range(len(vals) - 1)):
            print(f"FAIL: non-monotone interior canopy speed at z/H={z_over_h_targets[lev]:.2f}")
            return 1

    print("PASS: canopy-interior U/U_H decreases monotonically with increasing lambda_p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
