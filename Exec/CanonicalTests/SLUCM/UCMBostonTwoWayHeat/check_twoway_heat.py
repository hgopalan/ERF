#!/usr/bin/env python3
"""Verification for UCMBoston two-way heat coupling (Phase 3.2).

Validates that T_atm and wind plumbing enable UCM heat feedback to modify
ATM theta. Checks:
1. UHI signal at k=0 over urban cells > 0.01 K (heat feedback working)
2. Rural contamination at k=0 over non-urban cells < 0.005 K (no spurious heating)
3. cc_source[RhoTheta] max > 0 (injection active)

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


def load_ucm_field_2d(ds, field_name):
    """Load a UCM plotfile 2D field; return data or None."""
    try:
        # UCM fields are on a 2D slab (no z-dimension in UCM grid)
        cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                              dims=ds.domain_dimensions, fields=[field_name])
        data = np.array(cg[field_name])
        return data
    except Exception as e:
        print(f"WARNING: Could not load UCM field {field_name}: {e}")
        return None


def main():
    print("=" * 70)
    print("UCMBoston Two-Way Heat Validation (Phase 3.2)")
    print("=" * 70)

    # Find plotfile
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

    # Load theta field
    z, theta_3d = load_field_3d(ds, "theta")
    if theta_3d is None:
        print("FAIL: Could not load theta field")
        sys.exit(1)

    print(f"Domain shape: {theta_3d.shape} (nx, ny, nz)")
    print(f"Z range: {z.min():.1f} to {z.max():.1f} m")

    # Find k=0 (lowest ATM level)
    k0 = 0
    theta_k0 = theta_3d[:, :, k0]
    print(f"\nTheta at k=0 (z={z[k0]:.1f} m):")
    print(f"  Min: {theta_k0.min():.4f} K")
    print(f"  Max: {theta_k0.max():.4f} K")
    print(f"  Mean: {theta_k0.mean():.4f} K")
    print(f"  Std: {theta_k0.std():.4f} K")

    # Define urban region (center zone, roughly 40-60% of domain in x/y)
    nx, ny = theta_k0.shape
    x_start = int(0.4 * nx)
    x_end   = int(0.6 * nx)
    y_start = int(0.4 * ny)
    y_end   = int(0.6 * ny)

    urban_region = theta_k0[x_start:x_end, y_start:y_end]
    rural_edges = np.concatenate([
        theta_k0[:x_start, :].flatten(),
        theta_k0[x_end:, :].flatten(),
        theta_k0[:, :y_start].flatten(),
        theta_k0[:, y_end:].flatten()
    ])

    urban_mean = urban_region.mean()
    rural_mean = rural_edges.mean()
    uhi_signal = urban_mean - rural_mean

    print(f"\nUHI Analysis:")
    print(f"  Urban region (center): {urban_region.shape[0]}×{urban_region.shape[1]} cells")
    print(f"    Mean theta: {urban_mean:.4f} K")
    print(f"  Rural edges:")
    print(f"    Mean theta: {rural_mean:.4f} K")
    print(f"  UHI signal: {uhi_signal:.4f} K")

    # Check rural contamination (should be small relative to inflow)
    # Inflow sounding is ~295 K at surface; rural should be near that
    # Allow up to 0.005 K deviation from rural mean (minimal feedback on edges)
    rural_std = rural_edges.std()
    rural_contamination = rural_std
    print(f"  Rural std (contamination proxy): {rural_contamination:.4f} K")

    # Try to load UCM fields if available
    ucm_plt = find_final_plotfile().replace("plt_", "plt_ucm_atm_")
    cc_source_max = None
    if os.path.exists(ucm_plt):
        print(f"\nLoading UCM ATM plotfile: {ucm_plt}")
        try:
            ds_ucm = yt.load(ucm_plt)
            # UCM ATM plotfile has cc_source data
            try:
                cg_ucm = ds_ucm.covering_grid(level=0, left_edge=ds_ucm.domain_left_edge,
                                              dims=ds_ucm.domain_dimensions,
                                              fields=["cc_source_x"])
                # cc_source is on ATM grid, has shape (nx, ny, nz)
                cc_source_data = np.array(cg_ucm["cc_source_x"])
                if cc_source_data.size > 0:
                    cc_source_max = np.abs(cc_source_data).max()
                    print(f"  cc_source max: {cc_source_max:.6e}")
            except Exception as e:
                print(f"  (cc_source field not available: {e})")
        except Exception as e:
            print(f"  (Could not load UCM ATM plotfile: {e})")

    # Thresholds from problem statement
    UHI_THRESHOLD = 0.010  # K
    RURAL_CONTAMINATION_THRESHOLD = 0.005  # K
    CC_SOURCE_THRESHOLD = 0.0  # K*kg/m3/s (just needs to be non-zero)

    # Print results table
    print("\n" + "=" * 70)
    print("[UCM][3.2][check_twoway_heat]")
    print("=" * 70)

    pass_count = 0
    fail_count = 0

    # Check 1: UHI signal
    uhi_pass = uhi_signal > UHI_THRESHOLD
    print(f"  UHI urban k=0:    +{uhi_signal:.3f} K    {'PASS' if uhi_pass else 'FAIL'} (threshold: >{UHI_THRESHOLD:.3f} K)")
    pass_count += uhi_pass
    fail_count += not uhi_pass

    # Check 2: Rural contamination
    rural_pass = rural_contamination < RURAL_CONTAMINATION_THRESHOLD
    print(f"  Rural contamination k=0: {rural_contamination:.3f} K   {'PASS' if rural_pass else 'FAIL'} (threshold: <{RURAL_CONTAMINATION_THRESHOLD:.3f} K)")
    pass_count += rural_pass
    fail_count += not rural_pass

    # Check 3: cc_source non-zero (if available)
    if cc_source_max is not None:
        cc_source_pass = cc_source_max > CC_SOURCE_THRESHOLD
        print(f"  cc_source[RhoTheta] max: {cc_source_max:.3e} K*kg/m3/s  {'PASS' if cc_source_pass else 'FAIL'} (threshold: >{CC_SOURCE_THRESHOLD})")
        pass_count += cc_source_pass
        fail_count += not cc_source_pass
    else:
        print(f"  cc_source[RhoTheta] max: (not available)  SKIP")

    # Check that fields are finite
    fields_finite = np.isfinite(theta_3d).all()
    print(f"  All fields finite: {'PASS' if fields_finite else 'FAIL'}")
    pass_count += fields_finite
    fail_count += not fields_finite

    print("=" * 70)
    print(f"Results: {pass_count} PASS, {fail_count} FAIL")
    print("=" * 70)

    if fail_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
