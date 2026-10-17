#!/usr/bin/env python3
"""Write a farm of N stub OpenFAST turbines on a flat domain, to time the moving bodies as the farm grows.

The turbines stand on a square grid (N must be a square: 16, 64, 256, 1024, ...), each with its own
.fst deck for the bundled OpenFAST stub (so start-up reads N models, as a farm with one deck per
turbine does), all naming one AeroDyn file that the start-up checks read. The deck runs a fixed number
of steps with erf.moving_bodies.timing_int set, so the run log carries the time of every phase.

This is a measuring tool, not a regression test: nothing compares its output with a gold.

    make_scaling_farm.py --n 64 --mode adm --out farm_n64_adm
    mpiexec -n 4 erf_exec farm_n64_adm/farm.i
"""
import argparse
import math
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="number of turbines, a square (16, 64, 256, 1024, ...)")
    ap.add_argument("--mode", choices=["adm", "alm"], default="adm", help="actuator disk or line (default adm)")
    ap.add_argument("--out", required=True, help="directory to write the deck into (created)")
    ap.add_argument("--steps", type=int, default=4, help="steps to run (default 4)")
    ap.add_argument("--timing-int", type=int, default=2,
                    help="erf.moving_bodies.timing_int (default 2: the first summary holds the start-up, the next ones the steady steps)")
    ap.add_argument("--spacing", type=float, default=350.0, help="distance between turbines (m, default 350)")
    ap.add_argument("--dx", type=float, default=50.0, help="cell size (m, default 50)")
    ap.add_argument("--rotor-radius", type=float, default=60.0, help="stub rotor radius (m, default 60)")
    ap.add_argument("--blade-points", type=int, default=10, help="num_force_points_blade (default 10)")
    ap.add_argument("--ring-points", type=int, default=8, help="num_points_t, adm only (default 8)")
    ap.add_argument("--bodies", choices=["turbine", "ct_disk"], default="turbine",
                    help="stub OpenFAST turbines (default) or prescribed-Ct disks (no OpenFAST, --mode ignored)")
    ap.add_argument("--wake-lines-xD", type=float, nargs="*", default=[],
                    help="wake lines this many diameters behind every rotor (default none)")
    ap.add_argument("--stretched", action="store_true",
                    help="a stretched mesh (erf.terrain_z_levels, so the bodies see z_phys_nd) instead of uniform dz")
    ap.add_argument("--diagnostics-format", choices=["per_turbine", "per_rank", "farm"], default="farm",
                    help="erf.moving_bodies.diagnostics_format (default farm: a few files however many turbines)")
    a = ap.parse_args()

    side = int(round(math.sqrt(a.n)))
    if side * side != a.n or a.n < 1:
        raise SystemExit(f"make_scaling_farm.py: --n {a.n} is not a square")
    if a.spacing < 2.5 * a.rotor_radius:
        raise SystemExit("make_scaling_farm.py: --spacing must be at least 2.5 rotor radii, or the rotors overlap")
    if a.timing_int < 1 or a.steps < 1:
        raise SystemExit("make_scaling_farm.py: --steps and --timing-int must be >= 1")
    os.makedirs(a.out, exist_ok=True)

    # the domain: the grid of turbines with a spacing of room on every side (a whole number of cells)
    length = a.spacing * (side + 1)
    nx = int(math.ceil(length / a.dx))
    length = nx * a.dx
    hub = 2.0 * a.rotor_radius
    nz = int(math.ceil((hub + 3.0 * a.rotor_radius) / a.dx))
    dt = 0.5
    if a.bodies == "turbine" and a.mode == "alm":
        # the stub's rotor turns at 10 rpm: its tip may sweep less than one cell per step (alm_max_tip_cells)
        cells = (10.0 * 2.0 * math.pi / 60.0) * a.rotor_radius * dt / a.dx
        if cells > 0.9:
            raise SystemExit(f"make_scaling_farm.py: the line's tip sweeps {cells:.2f} cells per step; use a larger --dx or a smaller --rotor-radius")
    for xd in a.wake_lines_xD:
        # the lines behind the last column of rotors must stay in the domain
        if not xd > 0.0 or a.spacing * side + xd * 2.0 * a.rotor_radius > length:
            raise SystemExit(f"make_scaling_farm.py: --wake-lines-xD {xd:g} must be positive and leave the lines in the domain "
                             f"(at most {(length - a.spacing * side) / (2.0 * a.rotor_radius):.2f} diameters here)")
    # a stretched mesh: cells growing by 10 % upwards from 0.6 dx at the ground, so the bodies see z_phys_nd
    levels = ""
    if a.stretched:
        z, h, zs = 0.0, 0.6 * a.dx, [0.0]
        while z < nz * a.dx:
            z += h; h *= 1.1; zs.append(z)
        nz = len(zs) - 1
        levels = "erf.terrain_z_levels = " + " ".join(f"{v:.3f}" for v in zs) + "\n"

    # the induction model the sampling needs: the disk's default sampling (disk_corrected) feeds OpenFAST the
    # free stream, so its BEM stays on; the line samples at the rotor (sampling = disk), so it is off
    wake_mod = 1 if a.mode == "adm" else 0
    with open(os.path.join(a.out, "aerodyn.dat"), "w") as f:
        f.write("------- AERODYN INPUT FILE (the lines ERF's start-up checks read) -------\n"
                f"{wake_mod}                      Wake_Mod    - Wake/induction model (switch) {{0=none, 1=BEMT}}\n"
                "0                      TwrShadow   - tower shadow off\n")
    deck = [f"""# A farm of {a.n} stub OpenFAST turbines ({side} x {side}, {a.spacing:g} m apart, mode = {a.mode}), written by
# make_scaling_farm.py to time the moving bodies; not a regression test.
max_step = {a.steps}
stop_time = {a.steps * dt}
erf.fixed_dt = {dt}

geometry.prob_lo     = 0. 0. 0.
geometry.prob_hi     = {length:g} {length:g} {(float(levels.split()[-1]) if levels else nz * a.dx):g}
amr.n_cell           = {nx} {nx} {nz}
amr.max_level        = 0
amr.max_grid_size    = 32
geometry.is_periodic = 0 1 0

xlo.type     = "Inflow"
xlo.velocity = 10.0 0. 0.
xhi.type     = "Outflow"
zlo.type     = "SlipWall"
zhi.type     = "SlipWall"

erf.anelastic       = 1
erf.use_fft         = true
erf.molec_diff_type = "None"
erf.les_type        = "None"
erf.init_type       = "uniform"
erf.prob_name       = "ABL"
prob.rho_0 = 1.0
prob.T_0   = 300.0
prob.U_0   = 10.0

erf.sum_interval = -1
erf.check_int    = -1
erf.plot_int_1   = -1

{levels}erf.moving_bodies.diagnostics_dir    = moving_bodies
erf.moving_bodies.diagnostics_format = {a.diagnostics_format}
erf.moving_bodies.timing_int         = {a.timing_int}
""" + (f"erf.moving_bodies.wake.lines_xD = {' '.join(f'{x:g}' for x in a.wake_lines_xD)}\nerf.moving_bodies.wake.num_points = 21\n" if a.wake_lines_xD else "")]
    names = []
    for i in range(side):
        for j in range(side):
            name = f"T{len(names) + 1}"
            names.append(name)
            if a.bodies == "turbine":
                with open(os.path.join(a.out, f"{name}.fst"), "w") as f:
                    f.write(f"""# stub turbine {name} (Source/MovingBodies/OpenFAST/Stub); the two OpenFAST-format lines name the AeroDyn file
2                CompAero    - aerodynamic loads from AeroDyn
"aerodyn.dat"    AeroFile    - the AeroDyn file the start-up checks read
dt = 0.01
num_blades = 3
num_blade_nodes = 10
num_tower_nodes = 0
rotor_radius = {a.rotor_radius:g}
hub_height = {hub:g}
rotor_speed_rpm = 10.0
ct = 0.8
cp = 0.45
air_density = 1.0
""")
            x = a.spacing * (i + 1)
            y = a.spacing * (j + 0.5) + 0.5 * (length - a.spacing * side)
            if a.bodies == "ct_disk":
                deck.append(f"""erf.moving_bodies.{name}.type         = ct_disk
erf.moving_bodies.{name}.base_pos     = {x:g} {y:g} 0.
erf.moving_bodies.{name}.rotor_radius = {a.rotor_radius:g}
erf.moving_bodies.{name}.hub_height   = {hub:g}
erf.moving_bodies.{name}.ct           = 0.75
erf.moving_bodies.{name}.epsilon      = 2.0
erf.moving_bodies.{name}.air_density  = 1.0
erf.moving_bodies.{name}.output_root  = turbines/{name}
""")
                continue
            deck.append(f"""erf.moving_bodies.{name}.type                   = openfast_turbine
erf.moving_bodies.{name}.mode                   = {a.mode}
erf.moving_bodies.{name}.fst_file               = {name}.fst
erf.moving_bodies.{name}.base_pos               = {x:g} {y:g} 0.
erf.moving_bodies.{name}.num_force_points_blade = {a.blade_points}
erf.moving_bodies.{name}.epsilon                = 2.0
erf.moving_bodies.{name}.air_density            = 1.0
erf.moving_bodies.{name}.output_root            = turbines/{name}
""" + (f"erf.moving_bodies.{name}.num_points_t           = {a.ring_points}\n" if a.mode == "adm" else
       f"erf.moving_bodies.{name}.fllc                   = false\n"))
    deck.append("erf.moving_bodies.bodies = " + " ".join(names) + "\n")
    with open(os.path.join(a.out, "farm.i"), "w") as f:
        f.write("\n".join(deck))
    kind = f"turbines ({a.mode})" if a.bodies == "turbine" else "Ct disks"
    print(f"make_scaling_farm.py: {a.n} {kind} in {a.out}/farm.i, domain {nx} x {nx} x {nz} cells")


if __name__ == "__main__":
    main()
