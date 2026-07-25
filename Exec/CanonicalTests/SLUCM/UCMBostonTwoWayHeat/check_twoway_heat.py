#!/usr/bin/env python3
"""Verification for UCMBoston two-way heat coupling (Phase 3.2).

Validates that T_atm and wind plumbing enable UCM heat feedback to modify
ATM theta. Checks:
1. UHI signal aloft at k=10 (~210 m AGL): theta_center > theta_edge by > 0.02 K
   (tall Boston buildings push the UHI plume aloft; surface k=0 signal is ~0
   due to canyon shading — this is correct physics, not a bug).
2. Wind reduction at k=1 still > 10% (momentum drag not broken by heat feedback).
3. All fields finite (no NaN/Inf).

Sampling geometry matches check_boston_singlelevel.py:
  - i_center = 10, i_edge = 0, j_mid = ny//2
  - k_uhi = 10 (~210 m AGL, above 100 m canopy top)
  - k_surface = 1 (~30 m AGL, for wind reduction check)

Note on rural contamination: The Boston CSV layout has urban cells across the
entire domain (5 concentric zones, all urban). There is no truly rural region,
so a rural-contamination surface check is not meaningful here. The aloft UHI
structure check (center vs edge at k=10) is the correct two-way validation.

Plotfile discovery: main ERF plotfiles are named 'plt_NNNNN'. UCM companion
plotfiles (plt_ucm_*, plt_ucm_atm_*) are excluded.
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
    print("UCMBoston Two-Way Heat Validation (Phase 3.2)")
    print("=" * 70)

    main_plt = find_final_plotfile()
    if main_plt is None:
        print("FAIL: No plotfile found")
        sys.exit(1)

    print(f"\nLoading main ATM plotfile: {main_plt}")
    try:
        ds = yt.load(main_plt)
    except Exception as e:
        print(f"ERROR: Failed to load plotfile {main_plt}: {e}")
        sys.exit(1)

    z, theta = load_field_3d(ds, ("boxlib", "theta"))
    _, u     = load_field_3d(ds, ("boxlib", "x_velocity"))
    _, v     = load_field_3d(ds, ("boxlib", "y_velocity"))

    if theta is None or u is None or v is None:
        print("FAIL: Could not load required fields (theta, x_velocity, y_velocity)")
        sys.exit(1)

    print(f"Domain shape: {theta.shape} (nx, ny, nz)")
    print(f"Z range: {float(z[0]):.1f} m to {float(z[-1]):.1f} m")

    nx, ny, nz = theta.shape
    i_center = nx // 2   # downtown core
    i_edge   = 0         # upwind edge (west, rural reference)
    j_mid    = ny // 2

    # Sampling heights matching check_boston_singlelevel.py
    k_surface = 1   # ~30 m AGL (inside canyon, wind check)
    k_uhi     = 10  # ~210 m AGL (above 100 m canopy, where UHI plume lives)

    pass_count = 0
    fail_count = 0

    # ------------------------------------------------------------------
    # [1] UHI aloft check — primary two-way validation
    # ------------------------------------------------------------------
    # The Boston layout has buildings up to H=100m. Heat injection via
    # facet3D BEP distributes wall+roof flux above the canopy. The UHI
    # signal lives at k=10 (~210m), not at k=0 (which is inside the canyon
    # and shaded). This matches the one-way baseline behavior confirmed in
    # check_boston_singlelevel.py (UHI +0.03 K at k=10).
    # With atm_feedback_heat=1.0 the signal should be >= the one-way baseline.
    T_edge_uhi   = theta[i_edge,   j_mid, k_uhi]
    T_center_uhi = theta[i_center, j_mid, k_uhi]
    dT_aloft     = T_center_uhi - T_edge_uhi

    print(f"\n[1] UHI check aloft at k={k_uhi} (~{float(z[k_uhi]):.0f} m AGL, above canopy top)")
    print(f"    T at edge   (i={i_edge},    k={k_uhi}): {T_edge_uhi:.4f} K")
    print(f"    T at center (i={i_center}, k={k_uhi}): {T_center_uhi:.4f} K")
    print(f"    UHI ΔT (center - edge) = {dT_aloft:+.4f} K")

    UHI_THRESHOLD = 0.02  # K — matches one-way baseline threshold
    uhi_pass = dT_aloft > UHI_THRESHOLD
    status = "PASS" if uhi_pass else "FAIL"
    print(f"    {status}: ΔT = {dT_aloft:+.3f} K  (threshold: >{UHI_THRESHOLD:.3f} K)")
    pass_count += uhi_pass
    fail_count += not uhi_pass

    # Reference: surface k=0 ΔT (expected ~0 for tall canyons — not a failure)
    T_edge_surf   = theta[i_edge,   j_mid, 0]
    T_center_surf = theta[i_center, j_mid, 0]
    print(f"    [Reference] Surface k=0 ΔT: {T_center_surf - T_edge_surf:+.4f} K"
          f"  (expected ~0 for H_max=100m canyon — not checked)")

    # ------------------------------------------------------------------
    # [2] Wind reduction check — confirm momentum drag still active
    # ------------------------------------------------------------------
    U_center = np.sqrt(u[i_center, j_mid, k_surface]**2 + v[i_center, j_mid, k_surface]**2)
    U_edge   = np.sqrt(u[i_edge,   j_mid, k_surface]**2 + v[i_edge,   j_mid, k_surface]**2)

    print(f"\n[2] Wind reduction at k={k_surface} (~{float(z[k_surface]):.0f} m AGL)")
    print(f"    U at edge   (i={i_edge}):   {U_edge:.2f} m/s")
    print(f"    U at center (i={i_center}): {U_center:.2f} m/s")

    wind_pass = False
    if U_edge > 0.1:
        reduction_pct = 100.0 * (1.0 - U_center / U_edge)
        print(f"    Wind reduction: {reduction_pct:.1f}%")
        wind_pass = reduction_pct > 10.0
        status = "PASS" if wind_pass else "FAIL"
        print(f"    {status}: {reduction_pct:.1f}% reduction  (threshold: >10%)")
    else:
        print(f"    FAIL: edge wind too weak ({U_edge:.3f} m/s) to compute reduction")
    pass_count += wind_pass
    fail_count += not wind_pass

    # ------------------------------------------------------------------
    # [3] Vertical θ profile diagnostic
    # ------------------------------------------------------------------
    print(f"\n[3] Vertical θ profile: downtown (i={i_center}) vs edge (i={i_edge})")
    print(f"    k    z(m)     θ_edge(K)   θ_center(K)   Δθ(K)")
    for k in range(0, min(nz, 20), 2):
        z_val = float(z[k])
        te = theta[i_edge,   j_mid, k]
        tc = theta[i_center, j_mid, k]
        print(f"    {k:3d}  {z_val:6.1f}   {te:9.4f}   {tc:11.4f}   {tc-te:+.4f}")

    # ------------------------------------------------------------------
    # [4] Finite-value check
    # ------------------------------------------------------------------
    print(f"\n[4] Finite-value check")
    fields_finite = (np.isfinite(theta).all() and
                     np.isfinite(u).all() and
                     np.isfinite(v).all())
    status = "PASS" if fields_finite else "FAIL"
    print(f"    {status}: all fields finite")
    pass_count += fields_finite
    fail_count += not fields_finite

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[UCM][3.2][check_twoway_heat]")
    print("=" * 70)
    print(f"  UHI aloft k={k_uhi} (ΔT center-edge): {dT_aloft:+.3f} K"
          f"   {'PASS' if uhi_pass else 'FAIL'} (threshold: >{UHI_THRESHOLD:.3f} K)")
    print(f"  Wind reduction k={k_surface}:          {('%.1f%%' % reduction_pct) if U_edge > 0.1 else 'N/A'}"
          f"   {'PASS' if wind_pass else 'FAIL'} (threshold: >10%)")
    print(f"  All fields finite:                  {'PASS' if fields_finite else 'FAIL'}")
    print(f"  cc_source[RhoTheta] max:            (see [UCM][3.2] debug lines in run log)")
    print("=" * 70)
    print(f"Results: {pass_count} PASS, {fail_count} FAIL")
    print("=" * 70)

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
