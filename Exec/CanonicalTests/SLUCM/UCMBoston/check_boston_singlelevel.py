#!/usr/bin/env python3
"""Verification for UCMBoston single-level one-way (Phase 2.11).

Compares concentric ring temperatures and wind reduction in urban core
against rural upwind reference. Loose physical assertions only: concentric
UHI structure (aloft, above tall-building canopy), canopy wind reduction.

Uses yt.covering_grid() for 3D field indexing. Robust to yt version
via yt.set_log_level() (yt.suppress_stream_stdout does not exist).

Plotfile discovery: main ERF plotfiles are named 'plt_NNNNN' (no
underscore between prefix and digits). UCM companion plotfiles
(plt_ucm_*, plt_ucm_atm_*) must be excluded — they lack velocity
and theta fields.

Phase 2.11-fix notes (post atm_feedback split):
- Drag is now active in urban core (wind reduction ~50-80% at k=1).
- UHI signal is transported ALOFT (k=6-15, 130-290 m AGL) due to strong
  canyon blocking + 100 m building height. Surface k=1 is inside the
  drag-blocked canyon where residual flow is near-still and thermally
  coupled to surface — NOT where UHI shows up.
- Sample UHI at k=10 (~210 m AGL), above canopy top (H_max = 100 m).
- Threshold 0.02 K reflects 1-hour spin-up with no diurnal cycle.
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
    print("UCMBoston Verification (Phase 2.11 single-level one-way, post-fix)")
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

    # Domain: 20×20×64 (ATM cells). Downtown at i=10 (center),
    # rural upwind at i=0.
    i_center = 10       # Near center (downtown core)
    i_edge = 0          # Upwind edge (rural reference)
    j_mid = theta.shape[1] // 2

    # Two sampling heights:
    #   k_surface = 1  (~30 m AGL) — inside canyon, drag-blocked
    #   k_uhi     = 10 (~210 m AGL) — above 100 m canopy, where UHI plume lives
    k_surface = 1
    k_uhi = 10

    # ------------------------------------------------------------------
    # [3] UHI structure check — sample ALOFT, above canopy top
    # ------------------------------------------------------------------
    T_edge_uhi = theta[i_edge, j_mid, k_uhi]
    T_center_uhi = theta[i_center, j_mid, k_uhi]

    print(f"\n[3] Concentric UHI structure check (ABOVE canopy top)")
    print(f"    UHI plume lives aloft for tall-building canopies (H_max=100 m).")
    print(f"    Sampling at k={k_uhi} (~{float(z[k_uhi]):.0f} m AGL).")
    print(f"    T at domain edge   (i={i_edge}, k={k_uhi}): {T_edge_uhi:.2f} K")
    print(f"    T at near-center   (i={i_center}, k={k_uhi}): {T_center_uhi:.2f} K")

    dT = T_center_uhi - T_edge_uhi
    print(f"    UHI intensity ΔT = T_center - T_edge = {dT:+.2f} K")

    uhi_pass = dT > 0.02
    if uhi_pass:
        print(f"    ✓ PASS: ΔT = {dT:.2f} K > 0.02 K (urban plume detected aloft)")
    else:
        print(f"    ⚠ WARN: ΔT = {dT:.2f} K <= 0.02 K "
              f"(plume weak — check facet3d + AH configuration)")

    # Also print surface-level ΔT for reference (expected small for tall H)
    T_edge_surf = theta[i_edge, j_mid, k_surface]
    T_center_surf = theta[i_center, j_mid, k_surface]
    dT_surf = T_center_surf - T_edge_surf
    print(f"    [Reference] Surface ΔT at k={k_surface} (~{float(z[k_surface]):.0f} m): "
          f"{dT_surf:+.2f} K (expected ~0 for H_max=100 m canyon)")

    # ------------------------------------------------------------------
    # [4] Canopy wind reduction check (Phase 2.8 regression + Phase 2.11-fix)
    # ------------------------------------------------------------------
    print(f"\n[4] Canopy wind reduction check (downtown vs upwind edge)")
    U_center = np.sqrt(u[i_center, j_mid, :]**2 + v[i_center, j_mid, :]**2)
    U_edge = np.sqrt(u[i_edge, j_mid, :]**2 + v[i_edge, j_mid, :]**2)

    U_center_surface = U_center[k_surface]
    U_edge_surface = U_edge[k_surface]

    print(f"    U at domain edge (i={i_edge}, ~{float(z[k_surface]):.0f} m AGL): "
          f"{U_edge_surface:.2f} m/s")
    print(f"    U at near-center (i={i_center}, ~{float(z[k_surface]):.0f} m AGL): "
          f"{U_center_surface:.2f} m/s")

    wind_pass = True
    if U_edge_surface > 0.1:
        reduction_pct = 100.0 * (1.0 - U_center_surface / U_edge_surface)
        print(f"    Wind reduction: {reduction_pct:.1f}%")
        if reduction_pct > 10.0:
            print(f"    ✓ PASS: reduction {reduction_pct:.1f}% > 10% "
                  f"(drag active in urban core)")
        else:
            print(f"    ✗ FAIL: only {reduction_pct:.1f}% reduction — "
                  f"check atm_feedback_momentum and Cd_wall")
            wind_pass = False
    else:
        print(f"    DIAGNOSTIC: edge wind too weak to compute reduction")
        wind_pass = False

    # Sanity check: drag should not extend above canopy top
    U_center_aloft = U_center[k_uhi]
    U_edge_aloft = U_edge[k_uhi]
    print(f"    [Reference] U at k={k_uhi} (~{float(z[k_uhi]):.0f} m, above canopy): "
          f"edge={U_edge_aloft:.2f} m/s, center={U_center_aloft:.2f} m/s")

    # ------------------------------------------------------------------
    # [5] Finite-value check
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # [6] Diagnostic vertical θ profile
    # ------------------------------------------------------------------
    print(f"\n[6] Vertical θ profile at downtown core vs upwind edge")
    print(f"    k    z(m)     θ_edge(K)   θ_downtown(K)   Δθ(K)")
    for k in range(0, min(theta.shape[2], 20), 2):
        z_val = float(z[k])
        print(f"    {k:3d}  {z_val:6.1f}   {theta[i_edge, j_mid, k]:9.2f}   "
              f"{theta[i_center, j_mid, k]:11.2f}   "
              f"{theta[i_center, j_mid, k] - theta[i_edge, j_mid, k]:+.2f}")

    # ------------------------------------------------------------------
    # [7] Ring temperature summary
    # ------------------------------------------------------------------
    # ATM i index → UCM ring mapping (grid_ratio=4):
    #   i=0   → outer edge (west, suburban/rural)
    #   i=5   → residential dense (west transition)
    #   i=10  → downtown core center
    #   i=15  → residential dense (east transition)
    #   i=19  → outer edge (east, suburban/rural)
    print(f"\n[7] Diagnostic: average near-surface θ in concentric rings")
    print(f"    (Boston stylized: rings at i=0, 5, 10, 15, 19)")
    print(f"    Ring index (i)   θ_avg(K)   Description")
    ring_indices = [0, 5, 10, 15, 19]
    ring_descriptions = [
        "Outer edge (suburban/rural, west)",
        "Residential dense (west transition)",
        "Downtown core center",
        "Residential dense (east transition)",
        "Outer edge (suburban/rural, east)"
    ]
    for i_ring, desc in zip(ring_indices, ring_descriptions):
        T_ring = np.mean(theta[i_ring, :, k_surface])
        print(f"    {i_ring:3d}              {T_ring:9.2f}   {desc}")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print(f"\n" + "=" * 70)
    if uhi_pass and wind_pass:
        print(f"PASS: UCMBoston verification complete")
        print(f"      (UHI +{dT:.2f} K aloft, wind reduction {reduction_pct:.1f}% at surface)")
        print(f"=" * 70)
        return True
    else:
        print(f"PARTIAL: UCMBoston verification completed with warnings")
        print(f"         UHI pass: {uhi_pass}  |  Wind pass: {wind_pass}")
        print(f"=" * 70)
        return uhi_pass and wind_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
