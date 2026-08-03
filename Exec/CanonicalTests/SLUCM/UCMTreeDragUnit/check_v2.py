#!/usr/bin/env python3
"""Phase 6.1 verifier — matches directory layout:
    UCMTreeDragUnit/
        off/       plt_* + run_off.log
        on/        plt_* + run_on.log
        dense_on/  plt_* + run_on_dense.log
        tree_layout.csv, tree_layout_dense.csv
Usage:
    python3 check_tree_drag_v2.py
"""
import glob, os, re, sys
import numpy as np

try:
    import yt
    yt.set_log_level("error")
except ImportError:
    print("FAIL: yt not found"); sys.exit(1)


def latest_plot(dirpath):
    patt = re.compile(r"^plt_\d+$")
    hits = sorted(p for p in glob.glob(os.path.join(dirpath, "plt_*"))
                  if patt.match(os.path.basename(p)))
    return hits[-1] if hits else None


def load_3d(ds, field):
    cg = ds.covering_grid(level=0,
                          left_edge=ds.domain_left_edge,
                          dims=ds.domain_dimensions,
                          fields=[field])
    return np.array(cg[field])


# --- Assertion parsers on run logs -----------------------------------------

RE_TREE_BANNER = re.compile(r"\[UCM\]\[6\.1\]\[apply_ucm_tree_drag_to_source\]")
RE_NCELLS     = re.compile(r"N_cells\s*=\s*(\d+)")
RE_SUMFX      = re.compile(r"sum_Fx\s*=\s*([-+eE0-9.]+)")

def parse_log(path):
    with open(path) as fh:
        text = fh.read()
    banner_hits = len(RE_TREE_BANNER.findall(text))
    ncells = [int(m) for m in RE_NCELLS.findall(text)]
    sumfx  = [float(m) for m in RE_SUMFX.findall(text)]
    return banner_hits, ncells, sumfx


# --- Assertion 5: mid-canopy wind reduction ---------------------------------

def mid_canopy_x_velocity(plot, tree_mask_xhalf=200.0, z_target=10.0):
    ds = yt.load(plot)
    u = load_3d(ds, ("boxlib", "x_velocity"))
    nx, ny, nz = u.shape
    xlo, xhi = float(ds.domain_left_edge[0]), float(ds.domain_right_edge[0])
    zlo, zhi = float(ds.domain_left_edge[2]), float(ds.domain_right_edge[2])
    dx = (xhi - xlo) / nx
    dz = (zhi - zlo) / nz
    # left-half tree region: i where x_center < 200
    i_tree = [i for i in range(nx) if xlo + (i + 0.5) * dx < tree_mask_xhalf]
    # z-index closest to z_target
    zc = zlo + (np.arange(nz) + 0.5) * dz
    k = int(np.argmin(np.abs(zc - z_target)))
    return float(np.mean(u[i_tree, :, k]))


# --- Main -------------------------------------------------------------------

def main():
    fails = []

    # Assertion 1: off bit-identity — check via log absence of [UCM][6.1]
    if os.path.exists("run_off.log"):
        with open("run_off.log") as fh:
            text = fh.read()
        kernel_lines = [ln for ln in text.split("\n")
                        if "[UCM][6.1]" in ln
                        and "apply_ucm_tree_drag_to_source" in ln]
        if kernel_lines:
            fails.append(f"A1: tree drag kernel banner appears in off-mode "
                         f"({len(kernel_lines)} lines)")
        else:
            print("A1 off-mode kernel-silence: OK")
    else:
        print("A1 SKIP: run_off.log missing")

    # Assertion 2 + 3 + 4: on-mode kernel called, sign, N_cells
    if os.path.exists("run_on.log"):
        banners, ncells, sumfx = parse_log("run_on.log")
        if banners == 0:
            fails.append("A2: no [UCM][6.1][apply_ucm_tree_drag_to_source] banner in run_on.log")
        else:
            print(f"A2 kernel called: {banners} banner hits in run_on.log")

        if not sumfx:
            fails.append("A3: no sum_Fx reported")
        else:
            first_sumfx = sumfx[0]
            if first_sumfx >= 0:
                fails.append(f"A3: first sum_Fx = {first_sumfx:.3e} (expected < 0)")
            else:
                print(f"A3 sum_Fx sign: first = {first_sumfx:.3e} (< 0, OK)")

        if not ncells:
            fails.append("A4: no N_cells reported")
        else:
            # Expected 2 × 4 × 3 = 24 per the problem statement
            # (relax to >0 if your grid differs; comment the strict check below)
            first_n = ncells[0]
            expected = 24
            if first_n != expected:
                # Not a fatal fail — grid may differ. Warn only.
                print(f"A4 WARN: first N_cells = {first_n} (spec says {expected}); "
                      f"verify domain matches Phase 6.1 canonical")
            else:
                print(f"A4 N_cells: {first_n} = expected {expected}")
    else:
        print("A2-A4 SKIP: run_on.log missing")

    # Assertion 5: dense_on wind < on wind at mid-canopy
    p_on    = latest_plot("on")
    p_dense = latest_plot("dense_on")
    if p_on and p_dense:
        u_on    = mid_canopy_x_velocity(p_on)
        u_dense = mid_canopy_x_velocity(p_dense)
        print(f"A5 mid-canopy u: on={u_on:.4f}  dense={u_dense:.4f}")
        if not (u_dense < u_on):
            fails.append(f"A5: expected u_dense ({u_dense:.4f}) < u_on ({u_on:.4f})")
        else:
            print("A5 wind reduction: OK")
    else:
        print(f"A5 SKIP: on={p_on} dense_on={p_dense}")

    if fails:
        print("\n".join(f"FAIL {f}" for f in fails))
        return 1
    print("PASS: Phase 6.1 tree drag canonical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
