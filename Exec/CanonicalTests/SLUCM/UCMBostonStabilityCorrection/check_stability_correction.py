#!/usr/bin/env python3
"""Verification for UCMBostonStabilityCorrection (Phase 3.4/3.5).

Validates that the Businger-Dyer stability-corrected canyon-atmosphere
heat exchange produces physically correct results. Checks:
1. UHI aloft at k=10 (~210 m AGL): delta-T (center - edge) > 0.02 K
2. Wind reduction at k=1 (~30 m AGL): > 10%
3. All fields finite (no NaN/Inf)
4. Theta bounded [294, 310] K at k=10 (no blow-up)

Plotfile discovery: main ERF plotfiles are named 'plt_NNNNN' (digits only).
UCM companion plotfiles (plt_ucm_*, plt_ucm_atm_*) are excluded — they lack
velocity and theta fields.
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
            print(f"       Found only: {all_entries}")
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
    print("UCMBoston Stability Correction Validation (Phase 3.4/3.5)")
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

    i_center  = nx // 2
    i_edge    = 0
    j_mid     = ny // 2
    k_surface = 1   # ~30 m AGL
    k_uhi     = 10  # ~210 m AGL, above canopy top

    pass_count = 0
    fail_count = 0

    # ------------------------------------------------------------------
    # [1] UHI aloft check
    # ------------------------------------------------------------------
    T_edge_uhi   = theta[i_edge,   j_mid, k_uhi]
    T_center_uhi = theta[i_center, j_mid, k_uhi]
    dT_aloft     = T_center_uhi - T_edge_uhi

    print(f"\n[1] UHI check aloft at k={k_uhi} (~{float(z[k_uhi]):.0f} m AGL)")
    print(f"    T at edge   (i={i_edge},    k={k_uhi}): {T_edge_uhi:.4f} K")
    print(f"    T at center (i={i_center}, k={k_uhi}): {T_center_uhi:.4f} K")
    print(f"    UHI delta-T (center - edge) = {dT_aloft:+.4f} K")

    UHI_THRESHOLD = 0.02
    uhi_pass = dT_aloft > UHI_THRESHOLD
    print(f"    {'PASS' if uhi_pass else 'FAIL'}: delta-T = {dT_aloft:+.3f} K  "
          f"(threshold: >{UHI_THRESHOLD:.3f} K)")
    pass_count += uhi_pass
    fail_count += not uhi_pass

    # ------------------------------------------------------------------
    # [2] Wind reduction check
    # ------------------------------------------------------------------
    U_center = float(np.sqrt(u[i_center, j_mid, k_surface]**2
                             + v[i_center, j_mid, k_surface]**2))
    U_edge   = float(np.sqrt(u[i_edge,   j_mid, k_surface]**2
                             + v[i_edge,   j_mid, k_surface]**2))

    print(f"\n[2] Wind reduction at k={k_surface} (~{float(z[k_surface]):.0f} m AGL)")
    print(f"    U at edge   (i={i_edge}):   {U_edge:.2f} m/s")
    print(f"    U at center (i={i_center}): {U_center:.2f} m/s")

    wind_pass = False
    reduction_pct = 0.0
    if U_edge > 0.1:
        reduction_pct = 100.0 * (1.0 - U_center / U_edge)
        wind_pass = reduction_pct > 10.0
        print(f"    Wind reduction: {reduction_pct:.1f}%")
        print(f"    {'PASS' if wind_pass else 'FAIL'}: {reduction_pct:.1f}% reduction  "
              f"(threshold: >10%)")
    else:
        print(f"    FAIL: edge wind too weak ({U_edge:.3f} m/s)")
    pass_count += wind_pass
    fail_count += not wind_pass

    # ------------------------------------------------------------------
    # [3] Finite-value check
    # ------------------------------------------------------------------
    print(f"\n[3] Finite-value check")
    fields_finite = (np.isfinite(theta).all() and
                     np.isfinite(u).all() and
                     np.isfinite(v).all())
    print(f"    {'PASS' if fields_finite else 'FAIL'}: all fields finite")
    pass_count += fields_finite
    fail_count += not fields_finite

    # ------------------------------------------------------------------
    # [4] Theta bounded check at k=10
    # ------------------------------------------------------------------
    theta_at_k = theta[:, :, k_uhi]
    theta_min  = float(np.min(theta_at_k))
    theta_max  = float(np.max(theta_at_k))

    THETA_MIN = 294.0
    THETA_MAX = 310.0
    theta_bounded = (theta_min >= THETA_MIN) and (theta_max <= THETA_MAX)

    print(f"\n[4] Theta bounded check at k={k_uhi} (~{float(z[k_uhi]):.0f} m AGL)")
    print(f"    Theta range: [{theta_min:.4f}, {theta_max:.4f}] K")
    print(f"    {'PASS' if theta_bounded else 'FAIL'}: theta in [{THETA_MIN}, {THETA_MAX}] K")
    pass_count += theta_bounded
    fail_count += not theta_bounded

    # ------------------------------------------------------------------
    # [5] Phase 3.5a-hotfix: Newton clamp landmine detection
    # ------------------------------------------------------------------
    print(f"\n[5] Phase 3.5a-hotfix: Newton clamp landmine check")
    log_files = glob.glob("*.log") + glob.glob("run*.log")
    max_clamped_roof = 0
    max_clamped_wall = 0
    max_clamped_road = 0
    max_diverged_roof = 0
    max_diverged_wall = 0
    max_diverged_road = 0
    
    for lf in log_files:
        try:
            with open(lf) as f:
                content = f.read()
                # Find all clamp-count blocks (multiline)
                import re
                blocks = re.findall(r'\[UCM\]\[3\.5A-hotfix\]\[clamp-count\].*?(?=\n\[|$)', content, re.DOTALL)
                for block in blocks:
                    # Extract clamped counts from line like "Clamped to T_skin_min=260K:  roof=3  wall=5  road=7"
                    match_clamped = re.search(r'Clamped to T_skin_min=260K:.*?roof=(\d+)\s+wall=(\d+)\s+road=(\d+)', block)
                    if match_clamped:
                        max_clamped_roof = max(max_clamped_roof, int(match_clamped.group(1)))
                        max_clamped_wall = max(max_clamped_wall, int(match_clamped.group(2)))
                        max_clamped_road = max(max_clamped_road, int(match_clamped.group(3)))
                    # Extract diverged counts from line like "Newton diverged (hit max_iter): roof=0  wall=0  road=0"
                    match_diverged = re.search(r'Newton diverged.*?roof=(\d+)\s+wall=(\d+)\s+road=(\d+)', block)
                    if match_diverged:
                        max_diverged_roof = max(max_diverged_roof, int(match_diverged.group(1)))
                        max_diverged_wall = max(max_diverged_wall, int(match_diverged.group(2)))
                        max_diverged_road = max(max_diverged_road, int(match_diverged.group(3)))
        except Exception:
            pass

    total_clamped = max_clamped_roof + max_clamped_wall + max_clamped_road
    total_diverged = max_diverged_roof + max_diverged_wall + max_diverged_road
    
    CLAMP_THRESHOLD = 10  # arbitrary; adjust based on domain size
    
    clamp_check_pass = None  # default: None = no data / informational
    if total_clamped > 0 or total_diverged > 0:
        print(f"    Newton clamp/diverge log entries detected:")
        if total_clamped > 0:
            print(f"      Clamped: roof={max_clamped_roof}  wall={max_clamped_wall}  road={max_clamped_road}  (total={total_clamped})")
        if total_diverged > 0:
            print(f"      Diverged: roof={max_diverged_roof}  wall={max_diverged_wall}  road={max_diverged_road}  (total={total_diverged})")
        
        if total_clamped > CLAMP_THRESHOLD:
            print(f"    FAIL: {total_clamped} cell-steps hit T_skin_min=260K (threshold: <{CLAMP_THRESHOLD})")
            clamp_check_pass = False
        elif total_diverged > 0:
            print(f"    WARN: {total_diverged} cell-steps failed to converge (Newton hit max_iter)")
            clamp_check_pass = None  # warn but don't fail
        else:
            print(f"    PASS: clamping detected but within tolerance")
            clamp_check_pass = True
    else:
        print(f"    INFO: no clamp/divergence log entries found (all Newton converged normally, or ucm_debug=0)")
    
    if clamp_check_pass is not None:
        pass_count += clamp_check_pass
        fail_count += not clamp_check_pass

    # ------------------------------------------------------------------
    # [6] Stability correction log check (informational)
    # ------------------------------------------------------------------
    print(f"\n[6] Stability correction log check (informational)")
    log_files = glob.glob("*.log") + glob.glob("run*.log")
    found_corr = False
    for lf in log_files:
        try:
            with open(lf) as f:
                for line in f:
                    if "[UCM][3.4][stability-correction]" in line:
                        found_corr = True
                        break
        except Exception:
            pass
        if found_corr:
            break
    if found_corr:
        print("    INFO: stability correction active (found in log)")
    else:
        print("    INFO: no log found or correction trace not present "
              "(run with ucm_debug=1 to enable)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[UCM][3.4][check_stability_correction]")
    print("=" * 70)
    print(f"  UHI aloft k={k_uhi} (delta-T center-edge): {dT_aloft:+.3f} K"
          f"   {'PASS' if uhi_pass else 'FAIL'} (threshold: >{UHI_THRESHOLD:.3f} K)")
    print(f"  Wind reduction k={k_surface}:               {reduction_pct:.1f}%"
          f"   {'PASS' if wind_pass else 'FAIL'} (threshold: >10%)")
    print(f"  All fields finite:                       {'PASS' if fields_finite else 'FAIL'}")
    print(f"  Theta bounded [{THETA_MIN},{THETA_MAX}] K at k={k_uhi}:"
          f"  {theta_min:.3f}-{theta_max:.3f} K"
          f"   {'PASS' if theta_bounded else 'FAIL'}")
    print(f"  Stability correction active:             {'YES' if found_corr else 'not confirmed (no log)'}")
    print("=" * 70)
    print(f"Results: {pass_count} PASS, {fail_count} FAIL")
    print("=" * 70)

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
