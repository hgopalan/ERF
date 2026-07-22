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

    # Phase 2.5-fix2: Task 9 — Comprehensive conservation checks
    
    # --- Assertion 0: Basic range checks ---
    if not (0.0 <= d["f_urb_max"] <= 1.0):
        print("FAIL: f_urb_max out of [0,1]"); sys.exit(1)

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

    # --- Assertion 1: f_urb must span [0, 1] for the diagonal test. ---
    if d["f_urb_max"] < 0.99:
        print(f"FAIL: f_urb_max={d['f_urb_max']} < 0.99. "
              "Expected at least one fully-urban ATM cell. "
              "Likely CSV is_urban not propagating.")
        sys.exit(1)

    # --- Assertion 2: convention-B check. ---
    # H_atm is area-averaged (convention B). For a fully-urban ATM cell (f_urb=1),
    # H_atm should equal H_ucm_per_cell. If code accidentally divides by f_urb,
    # a partial-urban cell with f_urb=0.25 would give H_atm = 4 * H_ucm.

    H_ucm  = d["H_road_max"] + d["H_wall_max"] + d["H_roof_max"]
    H_atm  = d.get("H_atm_max")
    if H_atm is None:
        print("FAIL: H_atm_max missing from ucm_diag.dat")
        sys.exit(1)

    # Under convention B, H_atm_max ≤ H_ucm_max within tolerance.
    tol = 1.10
    if H_atm > tol * abs(H_ucm) + 1.0:
        print(f"FAIL: convention-B violation. H_atm_max={H_atm:.3f} > "
              f"{tol}*H_ucm_max={tol*H_ucm:.3f}. "
              "Likely divide-by-f_urb regression in coarsening.")
        sys.exit(1)

    # --- Assertion 3: facet-split symmetry (Bug 2 regression guard). ---
    # With uniform urban geometry (plan_frac=0.5, H_bldg=W_road=W_roof=10),
    # f_road = f_roof = f_wall = 0.5, so all three facet fluxes should be equal.
    H_road = d["H_road_max"]
    H_wall = d["H_wall_max"]
    H_roof = d["H_roof_max"] - d.get("AH_max", 0.0)   # subtract AH to isolate SEB contribution
    if abs(H_road) > 0.1 or abs(H_wall) > 0.1 or abs(H_roof) > 0.1:
        tol_sym = 0.10   # 10% pairwise
        if abs(H_road - H_wall) / max(abs(H_road), 1.0) > tol_sym or \
           abs(H_road - H_roof) / max(abs(H_road), 1.0) > tol_sym:
            print(f"FAIL: facet-split asymmetry detected.")
            print(f"  H_road (post-weight) = {H_road:.3f}")
            print(f"  H_wall (post-weight) = {H_wall:.3f}")
            print(f"  H_roof - AH          = {H_roof:.3f}")
            print(f"  All three should agree within {tol_sym*100:.0f}% for uniform geometry.")
            sys.exit(1)

    print(f"[check] all assertions PASS")
    print(f"  f_urb_max={d['f_urb_max']:.3f}")
    print(f"  H_atm_max={H_atm:.3f} <= H_ucm_max={H_ucm:.3f} * {tol}")
    print(f"  facet symmetry: road={H_road:.1f} wall={H_wall:.1f} roof-AH={H_roof:.1f}")

    print("\nPASS")

if __name__ == "__main__":
    main()
