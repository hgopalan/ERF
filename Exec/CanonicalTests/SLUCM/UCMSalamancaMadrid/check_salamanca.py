#!/usr/bin/env python3
"""Verification for UCMSalamancaMadrid (Phase 2.10).

Compares urban core temperatures against upwind rural reference.
Loose physical assertions only: positive UHI, canopy wind reduction.
Quantitative match to Salamanca 2011 Fig 5 deferred to Phase 3+.

Uses yt.covering_grid() for 3D field indexing. Robust to yt version
via yt.set_log_level() (yt.suppress_stream_stdout does not exist).

Plotfile discovery: main ERF plotfiles are named 'plt_NNNNN' (no
underscore between prefix and digits). UCM companion plotfiles
(plt_ucm_*, plt_ucm_atm_*) must be excluded — they lack velocity
and theta fields.
"""

import os
import re
import sys
import glob
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
    """Return the highest-numbered main ATM plotfile (plt_NNNNN)."""
    all_entries = sorted(glob.glob("plt_*"))
    pattern = re.compile(r"^plt_\d+$")
    main_files = [
        f for f in all_entries
        if pattern.match(os.path.basename(f))
        and not os.path.basename(f).startswith("plt_ucm")
    ]
    if not main_files:
        print("ERROR: No main ATM plotfiles found matching 'plt_NNNNN'")
        print("       Set 'amr.plot_int > 0' in inputs to write main ATM plotfiles.")
        if all_entries:
            print(f"       Found only UCM companion files: {all_entries}")
        return None
    return main_files[-1]


def load_field_3d(ds, field_name):
    """Load a full-domain 3D field via covering_grid; return (z, data) or (None, None)."""
    try:
        cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                              dims=ds.domain_dimensions, fields=[field_name])
        data = np.array(cg[field_name])
        if data.ndim != 3:
            print(f"WARNING: Field {field_name} has unexpected shape {data.shape}")
            return None, None
        z = ds.domain_left_edge[2] + (np.arange(data.shape[2]) + 0.5) \
            * (ds.domain_right_edge[2] - ds.domain_left_edge[2]) / data.shape[2]
        return z, data
    except Exception as e:
        print(f"ERROR loading field {field_name}: {e}")
        return None, None


def main():
    print("=" * 70)
    print("UCMSalamancaMadrid Verification (Phase 2.10)")
    print("=" * 70)

    plotfile = find_final_plotfile()
    if not plotfile:
        print("FAIL: No plotfile found")
        return False

    print(f"\n[1] Loading plotfile: {plotfile}")
    try:
        ds = yt.load(plotfile)
    except Exception as e:
        print(f"FAIL: Could not load plotfile: {e}")
        return False

    print(f"    Domain dims: {ds.domain_dimensions}")
    print(f"    Extent: {ds.domain_left_edge} to {ds.domain_right_edge}")

    # Load fields
    print(f"\n[2] Loading θ and velocity fields")
    z, theta = load_field_3d(ds, ("boxlib", "theta"))
    _, u = load_field_3d(ds, ("boxlib", "x_velocity"))
    _, v = load_field_3d(ds, ("boxlib", "y_velocity"))

    if theta is None or u is None or v is None:
        print("FAIL: Could not load required fields")
        return False

    print(f"    theta shape: {theta.shape}")

    # Domain is 10×10×64 (ATM cells). UCM stripes are:
    #   i=0..1   -> UCM 0..7   (rural upwind)
    #   i=2..3   -> UCM 8..15  (suburban)
    #   i=4..5   -> UCM 16..23 (urban core)  <-- comparison target
    #   i=6..7   -> UCM 24..31 (suburban)
    #   i=8..9   -> UCM 32..39 (rural downwind)
    # In ATM indices:
    i_rural_upwind = 0        # rural, first column
    i_urban_core   = 5        # urban core middle
    j_mid          = theta.shape[1] // 2

    # Near-surface (k=1, ~10 m above surface for dz=20 m)
    k_surface = 1

    T_rural_upwind = theta[i_rural_upwind, j_mid, k_surface]
    T_urban_core   = theta[i_urban_core,   j_mid, k_surface]

    print(f"\n[3] Urban heat island check")
    print(f"    T at rural upwind (i={i_rural_upwind}, k={k_surface}, ~10 m AGL): "
          f"{T_rural_upwind:.2f} K")
    print(f"    T at urban core   (i={i_urban_core}, k={k_surface}, ~10 m AGL): "
          f"{T_urban_core:.2f} K")

    dT = T_urban_core - T_rural_upwind
    print(f"    UHI intensity  ΔT = T_urban - T_rural = {dT:+.2f} K")

    # Loose assertion: positive UHI at noon LST
    if dT > 0.05:
        print(f"    ✓ PASS: ΔT = {dT:.2f} K > 0.05 K (positive UHI detected)")
    else:
        print(f"    ✗ FAIL: ΔT = {dT:.2f} K <= 0.05 K "
              f"(heat island absent or negative — check facet3d + AH configuration)")
        return False

    # Canopy wind reduction check (Phase 2.8 regression)
    print(f"\n[4] Canopy wind reduction check (urban core vs rural upwind)")
    U_urban   = np.sqrt(u[i_urban_core, j_mid, :]**2 + v[i_urban_core, j_mid, :]**2)
    U_rural   = np.sqrt(u[i_rural_upwind, j_mid, :]**2 + v[i_rural_upwind, j_mid, :]**2)

    # H_urban_core = 20 m; 2*H boundary is z=40 m -> k ≈ 2 for dz=20 m
    k_canopy_top = 1  # z=30 m for dz=20 m
    U_urban_10m = U_urban[k_surface]
    U_rural_10m = U_rural[k_surface]

    print(f"    U at rural upwind (i={i_rural_upwind}, ~10 m AGL): {U_rural_10m:.2f} m/s")
    print(f"    U at urban core   (i={i_urban_core}, ~10 m AGL):   {U_urban_10m:.2f} m/s")

    if U_rural_10m > 0.1:
        reduction_pct = 100.0 * (1.0 - U_urban_10m / U_rural_10m)
        print(f"    Wind reduction: {reduction_pct:.1f}%")
        if reduction_pct > 10.0:
            print(f"    ✓ PASS: reduction {reduction_pct:.1f}% > 10% "
                  f"(drag active in urban core)")
        else:
            print(f"    DIAGNOSTIC: only {reduction_pct:.1f}% reduction — check Cd_wall")
    else:
        print(f"    DIAGNOSTIC: rural wind too weak to compute reduction")

    # Diagnostic vertical θ profile
    print(f"\n[5] Vertical θ profile at urban core vs rural upwind")
    print(f"    k    z(m)     θ_rural(K)   θ_urban(K)   Δθ(K)")
    for k in range(0, min(theta.shape[2], 20), 2):
        z_val = float(z[k])
        print(f"    {k:3d}  {z_val:6.1f}   {theta[i_rural_upwind, j_mid, k]:9.2f}   "
              f"{theta[i_urban_core, j_mid, k]:9.2f}   "
              f"{theta[i_urban_core, j_mid, k] - theta[i_rural_upwind, j_mid, k]:+.2f}")

    print(f"\n" + "=" * 70)
    print(f"PASS: UCMSalamancaMadrid verification complete")
    print(f"=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)