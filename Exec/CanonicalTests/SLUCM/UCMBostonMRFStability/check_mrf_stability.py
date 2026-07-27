#!/usr/bin/env python3
"""Verification for UCMBoston MRF + UCM two-way heat stability (Phase 3.3).

Validates that MRF + UCM two-way heat coupling remains stable over 7200 time steps
(1 hour simulation). Checks:
1. Theta bounded [294, 310] K at k=10 everywhere (no blow-up)
2. UHI signal aloft at k=10: theta_center > theta_edge by > 0.02 K
   (stable UCM heating signal, not suppressed by MRF)
3. Wind reduction at k=1 still > 10% (drag still active)
4. All fields finite (no NaN/Inf)
5. MRF conflict max from log grep (informational)

Sampling geometry matches check_twoway_heat.py:
  - i_center = nx//2, i_edge = 0, j_mid = ny//2
  - k_uhi = 10 (~210 m AGL, above 100 m canopy top)
  - k_surface = 1 (~30 m AGL, for wind reduction check)

Reference: Phase 3.3 problem statement, Section 4.
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


def grep_log_for_mrf_conflict():
    """Grep run_mrf_stability.log for [UCM][3.3][mrf-conflict-check] and report max value."""
    max_val = 0.0
    pattern = re.compile(r"cc_src\[RhoTheta\] max BEFORE UCM injection = ([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)")
    
    log_files = glob.glob("run_mrf_stability.log") + glob.glob("*.log")
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if "[UCM][3.3][mrf-conflict-check]" in line or "cc_src[RhoTheta] max" in line:
                        match = pattern.search(line)
                        if match:
                            val = float(match.group(1))
                            max_val = max(max_val, abs(val))
        except Exception:
            pass
    return max_val


def main():
    print("=" * 70)
    print("UCMBoston MRF + UCM Two-Way Heat Stability Validation (Phase 3.3)")
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

    nx, ny, nz = theta.shape
    print(f"Domain shape: {theta.shape} (nx, ny, nz)")
    print(f"Z range: {float(z[0]):.1f} m to {float(z[-1]):.1f} m")

    i_center = nx // 2
    i_edge   = 0
    j_mid    = ny // 2
    k_surface = 1   # ~30 m AGL
    k_uhi     = 10  # ~210 m AGL, above canopy top

    pass_count = 0
    fail_count = 0

    # ------------------------------------------------------------------
    # [1] Theta bounded check at k=10
    # ------------------------------------------------------------------
    print(f"\n[1] Theta bounded check at k={k_uhi} (~{float(z[k_uhi]):.0f} m AGL)")
    theta_at_k10 = theta[:, :, k_uhi]
    theta_min = np.min(theta_at_k10)
    theta_max = np.max(theta_at_k10)
    
    THETA_MIN_THRESHOLD = 294.0
    THETA_MAX_THRESHOLD = 310.0
    theta_bounded = (theta_min >= THETA_MIN_THRESHOLD) and (theta_max <= THETA_MAX_THRESHOLD)
    
    print(f"    Theta bounds at k={k_uhi}: [{theta_min:.4f}, {theta_max:.4f}] K")
    print(f"    Expected range: [{THETA_MIN_THRESHOLD}, {THETA_MAX_THRESHOLD}] K")
    print(f"    {'PASS' if theta_bounded else 'FAIL'}: theta bounded within [{THETA_MIN_THRESHOLD}, {THETA_MAX_THRESHOLD}] K")
    pass_count += theta_bounded
    fail_count += not theta_bounded

    # ------------------------------------------------------------------
    # [2] UHI aloft check
    # ------------------------------------------------------------------
    T_edge_uhi   = theta[i_edge,   j_mid, k_uhi]
    T_center_uhi = theta[i_center, j_mid, k_uhi]
    dT_aloft     = T_center_uhi - T_edge_uhi

    print(f"\n[2] UHI check aloft at k={k_uhi} (~{float(z[k_uhi]):.0f} m AGL)")
    print(f"    T at edge   (i={i_edge},    k={k_uhi}): {T_edge_uhi:.4f} K")
    print(f"    T at center (i={i_center}, k={k_uhi}): {T_center_uhi:.4f} K")
    print(f"    UHI delta-T (center - edge) = {dT_aloft:+.4f} K")

    UHI_THRESHOLD = 0.02
    uhi_pass = dT_aloft > UHI_THRESHOLD
    print(f"    {'PASS' if uhi_pass else 'FAIL'}: delta-T = {dT_aloft:+.3f} K  (threshold: >{UHI_THRESHOLD:.3f} K)")
    pass_count += uhi_pass
    fail_count += not uhi_pass

    # ------------------------------------------------------------------
    # [3] Wind reduction check
    # ------------------------------------------------------------------
    U_center = float(np.sqrt(u[i_center, j_mid, k_surface]**2 + v[i_center, j_mid, k_surface]**2))
    U_edge   = float(np.sqrt(u[i_edge,   j_mid, k_surface]**2 + v[i_edge,   j_mid, k_surface]**2))

    print(f"\n[3] Wind reduction at k={k_surface} (~{float(z[k_surface]):.0f} m AGL)")
    print(f"    U at edge   (i={i_edge}):   {U_edge:.2f} m/s")
    print(f"    U at center (i={i_center}): {U_center:.2f} m/s")

    wind_pass = False
    reduction_pct = 0.0
    if U_edge > 0.1:
        reduction_pct = 100.0 * (1.0 - U_center / U_edge)
        wind_pass = reduction_pct > 10.0
        print(f"    Wind reduction: {reduction_pct:.1f}%")
        print(f"    {'PASS' if wind_pass else 'FAIL'}: {reduction_pct:.1f}% reduction  (threshold: >10%)")
    else:
        print(f"    FAIL: edge wind too weak ({U_edge:.3f} m/s)")
    pass_count += wind_pass
    fail_count += not wind_pass

    # ------------------------------------------------------------------
    # [4] Finite-value check
    # ------------------------------------------------------------------
    print(f"\n[4] Finite-value check")
    fields_finite = (np.isfinite(theta).all() and
                     np.isfinite(u).all() and
                     np.isfinite(v).all())
    print(f"    {'PASS' if fields_finite else 'FAIL'}: all fields finite")
    pass_count += fields_finite
    fail_count += not fields_finite

    # ------------------------------------------------------------------
    # [5] MRF conflict max from log grep (informational)
    # ------------------------------------------------------------------
    print(f"\n[5] MRF conflict check (informational, from log grep)")
    mrf_conflict_max = grep_log_for_mrf_conflict()
    print(f"    MRF cc_src[RhoTheta] before UCM max = {mrf_conflict_max:.2e}")
    if mrf_conflict_max > 1.0e-6:
        print(f"    WARNING: MRF wrote non-trivial value into cc_src[RhoTheta] (> 1e-6)")
        print(f"    -> This indicates MRF counter-gradient may affect UCM slot")
        print(f"    -> Not a FAIL (UCM ownership is enforced), but documented for audit trail")
    else:
        print(f"    OK: No significant MRF conflict detected")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[UCM][3.3][check_mrf_stability]")
    print("=" * 70)
    print(f"  theta bounded [294,310] K at k={k_uhi}: {theta_min:.3f}-{theta_max:.3f} K"
          f"   {'PASS' if theta_bounded else 'FAIL'}")
    print(f"  UHI aloft k={k_uhi} (delta-T center-edge):   {dT_aloft:+.3f} K"
          f"   {'PASS' if uhi_pass else 'FAIL'} (threshold: >{UHI_THRESHOLD:.3f} K)")
    print(f"  Wind reduction k={k_surface}:                        {reduction_pct:.1f}%"
          f"   {'PASS' if wind_pass else 'FAIL'} (threshold: >10%)")
    print(f"  All fields finite:                         {'PASS' if fields_finite else 'FAIL'}")
    print(f"  MRF conflict: cc_src before UCM max:       {mrf_conflict_max:.2e}  (from log grep)")
    print("=" * 70)
    print(f"Results: {pass_count} PASS, {fail_count} FAIL")
    print("=" * 70)

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
