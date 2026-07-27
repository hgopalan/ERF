#!/usr/bin/env python3
"""
validate_ucm_csvs.py
Pre-flight checker for ERF-SLUCM CSV input files.

Validates:
  - building_layout CSV : i,j,bldg_id,height_m,plan_area_frac,W_road_m,W_roof_m,
                          roof_mat_id,wall_mat_id,road_mat_id,orientation_deg,
                          ah_profile_id,is_urban  (13 columns)
  - materials CSV       : mat_id,name,albedo,emissivity,k_therm_W_per_mK,
                          rho_cp_J_per_m3K,thickness_m,description  (8 columns)

Usage:
    # Basic (checks files only):
    python validate_ucm_csvs.py --building building_hetero_mat.csv --materials materials.csv

    # With grid size check (verifies all NX*NY cells are present):
    python validate_ucm_csvs.py --building building_hetero_mat.csv --materials materials.csv --nx 16 --ny 16

    # Suppress colour output (e.g. in CI logs):
    python validate_ucm_csvs.py --building b.csv --materials m.csv --no-color
"""

import argparse
import csv
import sys
from pathlib import Path

# ── Colour helpers ─────────────────────────────────────────────────────────────
_USE_COLOR = True

def _c(code, text):
    return f"{code}{text}\033[0m" if _USE_COLOR else text

def ok(msg):
    print(f"  {_c(chr(27)+'[32m', 'OK')}  {msg}")

def warn(msg):
    print(f"  {_c(chr(27)+'[33m', 'WARN')}  {msg}")

def err(msg):
    print(f"  {_c(chr(27)+'[31m', 'ERR')}  {msg}")
    return 1


# ── Building layout CSV ────────────────────────────────────────────────────────
BUILDING_HEADER = [
    "i", "j", "bldg_id", "height_m", "plan_area_frac",
    "W_road_m", "W_roof_m",
    "roof_mat_id", "wall_mat_id", "road_mat_id",
    "orientation_deg", "ah_profile_id", "is_urban"
]


def validate_building_csv(path: Path, nx, ny):
    """
    Validates the building layout CSV.
    Returns (n_errors, set_of_mat_ids_referenced_by_urban_cells).
    """
    print(f"\n{'='*62}")
    print(f"  Building layout CSV : {path}")
    print('='*62)
    n_errors = 0

    if not path.exists():
        return err(f"File not found: {path}"), set()

    mat_refs = set()

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        # ── Header ──────────────────────────────────────────────────
        try:
            raw_header = next(reader)
        except StopIteration:
            return err("File is empty — no header row found."), set()

        header = [h.strip() for h in raw_header]
        if header != BUILDING_HEADER:
            n_errors += err(
                f"Header mismatch.\n"
                f"      Expected : {','.join(BUILDING_HEADER)}\n"
                f"      Got      : {','.join(header)}"
            )
            # Column-by-column diff
            for idx, (exp, got) in enumerate(zip(BUILDING_HEADER, header)):
                if exp != got:
                    warn(f"    Column {idx}: expected '{exp}', got '{got}'")
            extra   = len(header) - len(BUILDING_HEADER)
            missing = -extra
            if missing > 0:
                warn(f"    {missing} column(s) missing. "
                     f"Missing: {BUILDING_HEADER[len(header):]}")
            elif extra > 0:
                warn(f"    {extra} extra column(s) found.")
            return n_errors, mat_refs   # Cannot safely parse rows
        else:
            ok(f"Header OK — {len(header)} columns")

        # ── Data rows ────────────────────────────────────────────────
        seen_ij    = {}
        row_errors = 0
        all_ij     = []

        for lineno, raw in enumerate(reader, start=2):
            # Skip blank lines
            if not any(f.strip() for f in raw):
                continue

            if len(raw) != len(BUILDING_HEADER):
                row_errors += err(
                    f"Line {lineno}: expected {len(BUILDING_HEADER)} fields, "
                    f"got {len(raw)}: {raw}"
                )
                continue

            try:
                r = {k: v.strip() for k, v in zip(BUILDING_HEADER, raw)}

                i_val           = int(r["i"])
                j_val           = int(r["j"])
                bldg_id         = int(r["bldg_id"])
                height_m        = float(r["height_m"])
                plan_area_frac  = float(r["plan_area_frac"])
                W_road_m        = float(r["W_road_m"])
                W_roof_m        = float(r["W_roof_m"])
                roof_mat_id     = int(r["roof_mat_id"])
                wall_mat_id     = int(r["wall_mat_id"])
                road_mat_id     = int(r["road_mat_id"])
                orientation_deg = float(r["orientation_deg"])
                ah_profile_id   = int(r["ah_profile_id"])
                is_urban        = int(r["is_urban"])

            except (ValueError, KeyError) as exc:
                row_errors += err(f"Line {lineno}: parse error — {exc}: {raw}")
                continue

            # Duplicate (i,j) check
            ij = (i_val, j_val)
            if ij in seen_ij:
                row_errors += err(
                    f"Line {lineno}: duplicate (i,j)=({i_val},{j_val}) "
                    f"— first seen on line {seen_ij[ij]}"
                )
            else:
                seen_ij[ij] = lineno
                all_ij.append(ij)

            # is_urban must be 0 or 1
            if is_urban not in (0, 1):
                row_errors += err(
                    f"Line {lineno}: is_urban={is_urban} — must be 0 or 1"
                )

            # mat_id range rules
            if is_urban == 1:
                for col, mid in [("roof_mat_id", roof_mat_id),
                                  ("wall_mat_id", wall_mat_id),
                                  ("road_mat_id", road_mat_id)]:
                    if mid < 1:
                        row_errors += err(
                            f"Line {lineno}: urban cell ({i_val},{j_val}) "
                            f"has {col}={mid} — must be ≥1"
                        )
                    else:
                        mat_refs.add(mid)
            else:
                for col, mid in [("roof_mat_id", roof_mat_id),
                                  ("wall_mat_id", wall_mat_id),
                                  ("road_mat_id", road_mat_id)]:
                    if mid < 0:
                        row_errors += err(
                            f"Line {lineno}: non-urban cell ({i_val},{j_val}) "
                            f"has {col}={mid} — must be ≥0"
                        )

            # Physical range checks
            if height_m < 0:
                row_errors += err(
                    f"Line {lineno}: height_m={height_m} — must be ≥0"
                )
            if not (0.0 <= plan_area_frac <= 1.0):
                row_errors += err(
                    f"Line {lineno}: plan_area_frac={plan_area_frac} "
                    f"— must be in [0, 1]"
                )
            if W_road_m < 0:
                row_errors += err(
                    f"Line {lineno}: W_road_m={W_road_m} — must be ≥0"
                )
            if W_roof_m < 0:
                row_errors += err(
                    f"Line {lineno}: W_roof_m={W_roof_m} — must be ≥0"
                )

        n_rows = len(all_ij)
        n_errors += row_errors

        if row_errors == 0:
            ok(f"All {n_rows} data rows parsed without errors")

        # Urban cell summary
        urban_count = sum(
            1 for ij in seen_ij
            # Re-scan is_urban from the parsed ij dict isn't available here;
            # use a second pass via re-open
        )
        with path.open(newline="", encoding="utf-8") as f2:
            r2 = csv.DictReader(f2)
            u_count = sum(
                1 for row in r2
                if row.get("is_urban", "").strip() == "1"
            )
        ok(f"Urban cells (is_urban=1): {u_count} / {n_rows}")
        ok(f"Non-urban cells         : {n_rows - u_count} / {n_rows}")

        # Grid coverage check
        if nx is not None and ny is not None:
            expected_count = nx * ny
            if n_rows != expected_count:
                warn(
                    f"Row count {n_rows} ≠ nx×ny = {nx}×{ny} = {expected_count}. "
                    "Missing or extra cells?"
                )
            else:
                ok(f"Row count matches nx×ny = {nx}×{ny} = {expected_count}")

            expected_ij = {(i, j) for i in range(nx) for j in range(ny)}
            got_ij      = set(seen_ij.keys())
            missing_ij  = expected_ij - got_ij
            extra_ij    = got_ij - expected_ij
            if missing_ij:
                warn(
                    f"{len(missing_ij)} (i,j) pairs missing from expected grid "
                    f"(first 5: {sorted(missing_ij)[:5]})"
                )
            if extra_ij:
                warn(
                    f"{len(extra_ij)} (i,j) pairs outside [0,{nx})×[0,{ny}) "
                    f"(first 5: {sorted(extra_ij)[:5]})"
                )

    if mat_refs:
        ok(f"Material IDs referenced by urban cells: {sorted(mat_refs)}")
    else:
        warn("No urban cells found — no mat_ids referenced")

    return n_errors, mat_refs


# ── Materials CSV ──────────────────────────────────────────────────────────────
MATERIALS_HEADER = [
    "mat_id", "name", "albedo", "emissivity",
    "k_therm_W_per_mK", "rho_cp_J_per_m3K", "thickness_m", "description"
]


def validate_materials_csv(path: Path, required_mat_ids=None):
    """
    Validates the materials CSV.
    Returns n_errors.
    """
    print(f"\n{'='*62}")
    print(f"  Materials CSV : {path}")
    print('='*62)
    n_errors = 0

    if not path.exists():
        return err(f"File not found: {path}")

    defined_ids = set()

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        # ── Header ──────────────────────────────────────────────────
        try:
            raw_header = next(reader)
        except StopIteration:
            return err("File is empty — no header row found.")

        header = [h.strip() for h in raw_header]
        if header != MATERIALS_HEADER:
            n_errors += err(
                f"Header mismatch.\n"
                f"      Expected : {','.join(MATERIALS_HEADER)}\n"
                f"      Got      : {','.join(header)}"
            )
            for idx, (exp, got) in enumerate(zip(MATERIALS_HEADER, header)):
                if exp != got:
                    warn(f"    Column {idx}: expected '{exp}', got '{got}'")
            extra   = len(header) - len(MATERIALS_HEADER)
            missing = -extra
            if missing > 0:
                warn(f"    {missing} column(s) missing. "
                     f"Missing: {MATERIALS_HEADER[len(header):]}")
            elif extra > 0:
                warn(f"    {extra} extra column(s) found.")
            return n_errors
        else:
            ok(f"Header OK — {len(header)} columns")

        # ── Data rows ────────────────────────────────────────────────
        seen_ids   = {}
        row_errors = 0

        for lineno, raw in enumerate(reader, start=2):
            if not any(f.strip() for f in raw):
                continue

            if len(raw) < 7:
                row_errors += err(
                    f"Line {lineno}: expected ≥7 fields, got {len(raw)}: {raw}"
                )
                continue

            try:
                mat_id      = int(raw[0].strip())
                name        = raw[1].strip()
                albedo      = float(raw[2].strip())
                emissivity  = float(raw[3].strip())
                k_therm     = float(raw[4].strip())
                rho_cp      = float(raw[5].strip())
                thickness_m = float(raw[6].strip())
                description = raw[7].strip() if len(raw) > 7 else ""
            except (ValueError, IndexError) as exc:
                row_errors += err(
                    f"Line {lineno}: parse error — {exc}: {raw}"
                )
                continue

            # Duplicate mat_id
            if mat_id in seen_ids:
                row_errors += err(
                    f"Line {lineno}: duplicate mat_id={mat_id} "
                    f"— first on line {seen_ids[mat_id]}"
                )
            else:
                seen_ids[mat_id] = lineno
                defined_ids.add(mat_id)

            # String length limits
            if len(name) > 63:
                row_errors += err(
                    f"Line {lineno}: mat_id={mat_id} name too long "
                    f"({len(name)} chars, max 63)"
                )
            if len(description) > 127:
                row_errors += err(
                    f"Line {lineno}: mat_id={mat_id} description too long "
                    f"({len(description)} chars, max 127)"
                )

            # Physical range checks
            if not (0.0 <= albedo <= 1.0):
                row_errors += err(
                    f"Line {lineno}: mat_id={mat_id} albedo={albedo} "
                    f"— must be in [0, 1]"
                )
            if not (0.0 <= emissivity <= 1.0):
                row_errors += err(
                    f"Line {lineno}: mat_id={mat_id} emissivity={emissivity} "
                    f"— must be in [0, 1]"
                )
            if k_therm <= 0.0:
                row_errors += err(
                    f"Line {lineno}: mat_id={mat_id} k_therm_W_per_mK={k_therm} "
                    f"— must be > 0"
                )
            if rho_cp <= 0.0:
                row_errors += err(
                    f"Line {lineno}: mat_id={mat_id} rho_cp_J_per_m3K={rho_cp} "
                    f"— must be > 0"
                )
            if thickness_m <= 0.0:
                row_errors += err(
                    f"Line {lineno}: mat_id={mat_id} thickness_m={thickness_m} "
                    f"— must be > 0"
                )

        n_errors += row_errors
        if row_errors == 0:
            ok(f"All {len(defined_ids)} material rows parsed without errors")
        ok(f"Defined mat_ids: {sorted(defined_ids)}")

    # Cross-check: all mat_ids used by building CSV must be defined here
    if required_mat_ids:
        missing_ids = required_mat_ids - defined_ids
        if missing_ids:
            n_errors += err(
                f"mat_ids referenced in building CSV but NOT defined in "
                f"materials CSV: {sorted(missing_ids)}"
            )
        else:
            ok(
                f"Cross-check OK — all referenced mat_ids "
                f"{sorted(required_mat_ids)} are defined in materials CSV"
            )

    return n_errors


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight validator for ERF-SLUCM CSV input files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--building",  required=True,
                        help="Path to building layout CSV")
    parser.add_argument("--materials", required=True,
                        help="Path to materials CSV")
    parser.add_argument("--nx", type=int, default=None,
                        help="Expected grid size in x (e.g. 16)")
    parser.add_argument("--ny", type=int, default=None,
                        help="Expected grid size in y (e.g. 16)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colour output")
    args = parser.parse_args()

    global _USE_COLOR
    if args.no_color:
        _USE_COLOR = False

    building_path  = Path(args.building)
    materials_path = Path(args.materials)

    result = validate_building_csv(building_path, args.nx, args.ny)
    if isinstance(result, tuple):
        bldg_errors, mat_refs = result
    else:
        bldg_errors, mat_refs = result, set()

    mat_errors = validate_materials_csv(materials_path,
                                        required_mat_ids=mat_refs)

    total = bldg_errors + mat_errors
    print(f"\n{'='*62}")
    if total == 0:
        print(_c('\033[32m',
                 "  ALL CHECKS PASSED — CSVs are ready for ERF-SLUCM."))
    else:
        print(_c('\033[31m',
                 f"  {total} ERROR(S) FOUND — fix before running ERF."))
    print('='*62)
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
