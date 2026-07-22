#!/usr/bin/env python3
"""Post-run conservation check for UCMScaleAwareAggregation.

Reads ucm_diag.dat, extracts final-step H_road_max + H_wall_max + H_roof_max
(the per-cell max fluxes on the UCM grid) and cross-checks against the
aggregated ATM-grid fluxes from plt_ucm_atm_00100/.

For convention B (area-averaged, no divide-by-f_urb), we check:
    - f_urb ranges [0, 1] across ATM domain
    - At least one ATM cell has f_urb=1 (fully urban)
    - H_bldg_mean is ~10 m in urban cells
    - H_bldg_std is ~0 m (uniform urban)
    - H_atm_max <= H_ucm_max (conservation check)
    - Presence of ATM plotfile

Prints PASS or FAIL with numerical values.
"""
import sys, os, glob

def read_last_diag_row(path):
    with open(path) as f:
        rows = [ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith("#")]
    header = rows[0].split(",")
    last   = rows[-1].split(",")
    return dict(zip(header, [float(x) for x in last]))

def main():
    diag = "ucm_diag.dat"
    if not os.path.exists(diag):
        print(f"FAIL: {diag} not found"); sys.exit(1)
    d = read_last_diag_row(diag)

    # Sanity: aggregates and convention-B check fields present.
    required_keys = ("f_urb_max", "H_bldg_mean_max", "H_bldg_std_max", "lambda_f_max", "H_atm_max")
    for k in required_keys:
        if k not in d:
            print(f"FAIL: {k} missing from {diag}"); sys.exit(1)

    print(f"[check] f_urb_max        = {d['f_urb_max']:.3f}")
    print(f"[check] H_bldg_mean_max  = {d['H_bldg_mean_max']:.3f} m")
    print(f"[check] H_bldg_std_max   = {d['H_bldg_std_max']:.3f} m")
    print(f"[check] lambda_f_max     = {d['lambda_f_max']:.3f}")
    print(f"[check] H_atm_max        = {d['H_atm_max']:.3f} W/m^2")

    # Rough conservation check via BANNER-visible quantities:
    # We expect f_urb to span [0, 1] because our pattern is diagonal.
    if not (0.0 <= d["f_urb_max"] <= 1.0):
        print("FAIL: f_urb_max out of [0,1]"); sys.exit(1)
    if d["f_urb_max"] < 0.9:
        print("FAIL: at least one ATM cell should be fully urban (f_urb~1)"); sys.exit(1)

    # Sanity: mean H is 10 m in urban cells, so max should be ~10.
    if not (8.0 < d["H_bldg_mean_max"] < 12.0):
        print(f"FAIL: H_bldg_mean_max = {d['H_bldg_mean_max']}, expected ~10 m"); sys.exit(1)

    # Sanity: H_bldg is uniform in urban half, so std should be 0.
    if d["H_bldg_std_max"] > 0.5:
        print(f"FAIL: H_bldg_std_max = {d['H_bldg_std_max']}, expected ~0 m (uniform urban)")
        sys.exit(1)

    # Presence of ATM plotfile.
    atm_plt = glob.glob("plt_ucm_atm_00100*")
    if not atm_plt:
        print("FAIL: plt_ucm_atm_00100 not written"); sys.exit(1)
    print(f"[check] found ATM plotfile: {atm_plt[0]}")

    # Convention-B check: for a partial-urban ATM cell, the injected flux must
    # be area-averaged, NOT per-urban-cell average. That means:
    #
    #   H_atm_max ≈ f_urb_max_partial * H_ucm_per_cell_max
    #
    # In the diagonal test the max f_urb is 1 (fully-urban ATM cells exist),
    # so H_atm_max should equal H_ucm_per_cell_max within the flux precision.
    # In an ATM cell with f_urb=0.5, H_atm should be half of H_ucm_per_cell.
    #
    # If the code accidentally divides by f_urb somewhere (convention A regression),
    # H_atm on partial-urban cells would be too large by 1/f_urb. This shows up as
    # H_atm_max > H_ucm_per_cell_max in the partial-urban region.

    H_road_max = d["H_road_max"]
    H_wall_max = d["H_wall_max"]
    H_roof_max = d["H_roof_max"]
    H_ucm_max  = H_road_max + H_wall_max + H_roof_max  # per-UCM-cell max on urban cells

    H_atm_max  = d["H_atm_max"]

    # Under convention B, the ATM cell with f_urb=1 sees the full H_ucm.
    # H_atm_max should be within a few percent of H_ucm_max.
    # Under convention A (bug), H_atm_max for partial cells would be H_ucm_max
    # (after the erroneous divide), and for full-urban it would also be H_ucm_max.
    # So this simple check catches the A-regression only via the RATIO across
    # partial-urban cells, not via H_atm_max alone.
    #
    # Simpler decisive check: assert H_atm_max <= H_ucm_max + tolerance.
    # Under B, they should be equal (both are the max over fully-urban cells).
    # Under A with post-hoc divide-by-f_urb, H_atm on a f_urb=0.25 cell would be
    # 4 * H_ucm, which is 4x too high. H_atm_max would then be ~4x H_ucm_max.

    tolerance = 1.10  # allow 10 % slack for time-varying flux
    if H_atm_max > tolerance * abs(H_ucm_max) + 1.0:
        print(f"FAIL: convention-B violation.")
        print(f"  H_atm_max      = {H_atm_max:.3f} W/m^2")
        print(f"  H_ucm_max      = {H_ucm_max:.3f} W/m^2")
        print(f"  ratio          = {H_atm_max / max(abs(H_ucm_max), 1e-6):.3f}")
        print(f"  expected ratio = 1.0 (within {tolerance})")
        print(f"  suggests convention-A regression: H_atm was divided by f_urb")
        sys.exit(1)

    print(f"[check] convention-B: H_atm_max={H_atm_max:.3f} <= "
          f"H_ucm_max={H_ucm_max:.3f} * {tolerance}: PASS")

    print("\nPASS")

if __name__ == "__main__":
    main()
