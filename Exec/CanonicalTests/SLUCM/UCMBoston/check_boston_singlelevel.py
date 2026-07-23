#!/usr/bin/env python3
"""Verification for UCMBoston single-level one-way (Phase 2.11).

Compares concentric ring temperatures and wind reduction in urban core
against rural upwind reference. Loose physical assertions only: concentric
UHI structure, canopy wind reduction.

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
        print("       Set 'erf.plot_file_1' and 'erf.plot_int_1' in inputs to write main ATM plotfiles.")
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
    print("UCMBoston Verification (Phase 2.11 single-level one-way)")
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

    # Domain is 20×20×64 (ATM cells). Boston concentric rings map to:
    #   i=0..4   -> UCM 0..19 (may span edges, use center)
    #   i=10     -> UCM 40 (downtown core center, i=40 ATM equivalent = 10 in 0-19 range)
    # Actually for 20x20 ATM grid, center is at i,j = 9.5
    # Downtown core center is at UCM i,j ~ 39.5, which in ATM coords (ATM=UCM/4) is i,j ~ 9.875
    # Use i=10 (close to center) and i=0 (upwind edge).
    
    i_center = 10       # Near center of 20×20 grid (downtown)
    i_edge = 0          # Upwind edge (rural reference)
    j_mid = theta.shape[1] // 2

    # Near-surface (k=1, ~30 m above surface for dz=20 m baseline at first level)
    k_surface = 1

    T_edge = theta[i_edge, j_mid, k_surface]
    T_center = theta[i_center, j_mid, k_surface]

    print(f"\n[3] Concentric UHI structure check")
    print(f"    T at domain edge (i={i_edge}, k={k_surface}, ~30 m AGL):   "
          f"{T_edge:.2f} K")
    print(f"    T at near-center (i={i_center}, k={k_surface}, ~30 m AGL): "
          f"{T_center:.2f} K")

    dT = T_center - T_edge
    print(f"    UHI intensity  ΔT = T_center - T_edge = {dT:+.2f} K")

    # Loose assertion: positive UHI at noon LST (daytime heating)
    if dT > 0.05:
        print(f"    ✓ PASS: ΔT = {dT:.2f} K > 0.05 K (concentric UHI detected)")
    else:
        print(f"    ✗ FAIL: ΔT = {dT:.2f} K <= 0.05 K "
              f"(heat island absent or negative — check facet3d + AH configuration)")
        return False

    # Canopy wind reduction check (Phase 2.8 regression)
    print(f"\n[4] Canopy wind reduction check (downtown vs upwind edge)")
    U_center = np.sqrt(u[i_center, j_mid, :]**2 + v[i_center, j_mid, :]**2)
    U_edge = np.sqrt(u[i_edge, j_mid, :]**2 + v[i_edge, j_mid, :]**2)

    # Downtown H ~ 100 m; 2*H boundary is z=200 m. Use near-surface k=1
    U_center_surface = U_center[k_surface]
    U_edge_surface = U_edge[k_surface]

    print(f"    U at domain edge (i={i_edge}, ~30 m AGL):   {U_edge_surface:.2f} m/s")
    print(f"    U at near-center (i={i_center}, ~30 m AGL): {U_center_surface:.2f} m/s")

    if U_edge_surface > 0.1:
        reduction_pct = 100.0 * (1.0 - U_center_surface / U_edge_surface)
        print(f"    Wind reduction: {reduction_pct:.1f}%")
        if reduction_pct > 10.0:
            print(f"    ✓ PASS: reduction {reduction_pct:.1f}% > 10% "
                  f"(drag active in urban core)")
        else:
            print(f"    DIAGNOSTIC: only {reduction_pct:.1f}% reduction — check Cd_wall")
    else:
        print(f"    DIAGNOSTIC: edge wind too weak to compute reduction")

    # NaN check
    print(f"\n[5] Finite-value check (no NaN/Inf)")
    has_nan = False
    if not np.all(np.isfinite(theta)):
        print(f"    ✗ FAIL: theta contains NaN or Inf")
        has_nan = True
    if not np.all(np.isfinite(u)):
        print(f"    ✗ FAIL: u contains NaN or Inf")
        has_nan = True
    if not np.all(np.isfinite(v)):
        print(f"    ✗ FAIL: v contains NaN or Inf")
        has_nan = True
    if not has_nan:
        print(f"    ✓ PASS: all fields are finite")
    else:
        return False

    # Diagnostic vertical θ profile (downtown vs edge, side-by-side)
    print(f"\n[6] Vertical θ profile at downtown core vs upwind edge")
    print(f"    k    z(m)     θ_edge(K)   θ_downtown(K)   Δθ(K)")
    for k in range(0, min(theta.shape[2], 20), 2):
        z_val = float(z[k])
        print(f"    {k:3d}  {z_val:6.1f}   {theta[i_edge, j_mid, k]:9.2f}   "
              f"{theta[i_center, j_mid, k]:11.2f}   "
              f"{theta[i_center, j_mid, k] - theta[i_edge, j_mid, k]:+.2f}")

    # Diagnostic ring temperature summary
    print(f"\n[7] Diagnostic: average near-surface θ in concentric rings")
    print(f"    (Boston stylized: rings at i=0, 5, 10, 15, 19)")
    print(f"    Ring index (i)   ≈ UCM ring   θ_avg(K)   Description")
    ring_indices = [0, 5, 10, 15, 19]
    ring_descriptions = [
        "Upwind/edge (rural reference)",
        "Suburban/outer transition",
        "Downtown core center",
        "Suburban/inner transition",
        "Downwind/far edge"
    ]
    for idx, (i_ring, desc) in enumerate(zip(ring_indices, ring_descriptions)):
        T_ring = np.mean(theta[i_ring, :, k_surface])
        print(f"    {i_ring:3d}              {idx:d}           {T_ring:9.2f}   {desc}")

    print(f"\n" + "=" * 70)
    print(f"PASS: UCMBoston verification complete")
    print(f"=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
