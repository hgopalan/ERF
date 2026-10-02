#!/usr/bin/env python3
"""Generate the hills-and-transformers power-line case: terrain, inflow, sounding and network.

Writes, into the output directory:
  terrain_hills.txt   ERF's ASCII terrain file (nx, ny, the x values, the y values, z with x fastest)
  inflow_profile      the xlo Dirichlet profile, z u v w theta: a neutral log law under a capping inversion
  input_sounding      the same profile as ERF's input sounding (surface pressure in hPa)
  network.inputs      the erf.conductors block: transformers and the lines between them

Hills are Gaussian bumps at random places; transformers stand on some hilltops and on flat ground
at random places, and a minimum spanning tree of lines joins them. Each line is a section dead-ended
on its two transformers, hanging from insulator strings on suspension towers at most max_span
apart, every span strung to the same horizontal tension; a span that would come closer to the ground than min_clearance in still air gets a tower
at its lowest point. Everything is drawn from one seed, so the same arguments give the same case.
"""

import argparse
import math
import os

import numpy as np

KAPPA = 0.4


def hills_field(hills, x, y):
    X, Y = np.meshgrid(x, y, indexing="ij")   # [i, j] = (x_i, y_j): flattened, y runs fastest
    z = np.zeros_like(X)
    for (xc, yc, h, s) in hills:
        z += h * np.exp(-((X - xc) ** 2 + (Y - yc) ** 2) / s ** 2)
    return z


def height(hills, x, y):
    return sum(h * math.exp(-((x - xc) ** 2 + (y - yc) ** 2) / s ** 2) for (xc, yc, h, s) in hills)


def make_hills(rng, a):
    hills = []
    while len(hills) < a.hills:
        xc = rng.uniform(a.lx * 0.27, a.lx * 0.8)
        yc = rng.uniform(a.ly * 0.2, a.ly * 0.8)
        if any(math.hypot(xc - p[0], yc - p[1]) < a.hill_spacing for p in hills):
            continue
        hills.append((xc, yc, rng.uniform(*a.hill_height), rng.uniform(*a.hill_radius)))
    return hills


def make_transformers(rng, a, hills):
    xr = (a.lx * 0.17, a.lx * 0.87)
    yr = (a.ly * 0.15, a.ly * 0.85)
    pts = []
    # on hilltops: the summits of the first few hills, nudged off the crest a little
    for (xc, yc, h, s) in hills[:a.on_hills]:
        pts.append((xc + rng.uniform(-20, 20), yc + rng.uniform(-20, 20), "hill"))
    # on flat ground: away from the hills and from the other transformers
    tries = 0
    while len(pts) < a.transformers:
        tries += 1
        if tries > 100000:
            raise SystemExit("cannot place the transformers; loosen the spacing or enlarge the domain")
        x, y = rng.uniform(*xr), rng.uniform(*yr)
        if height(hills, x, y) > a.flat_height:
            continue
        if any(math.hypot(x - p[0], y - p[1]) < a.transformer_spacing for p in pts):
            continue
        pts.append((x, y, "ground"))
    return pts


def spanning_tree(pts):
    """Prim's minimum spanning tree over the transformers, by horizontal distance."""
    n = len(pts)
    inside, edges = {0}, []
    while len(inside) < n:
        best = None
        for i in inside:
            for j in range(n):
                if j in inside:
                    continue
                d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                if best is None or d < best[0]:
                    best = (d, i, j)
        edges.append((best[1], best[2]))
        inside.add(best[2])
    return edges


def end_on_box(p, q, a):
    """The point on transformer p's top facing q, inset from the footprint's edge."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    r = math.hypot(dx, dy)
    ux, uy = dx / r, dy / r
    t = a.end_inset * min(0.5 * a.box[0] / max(abs(ux), 1e-9), 0.5 * a.box[1] / max(abs(uy), 1e-9))
    return (p[0] + t * ux, p[1] + t * uy)


W = 1.628 * 9.81   # the conductor's weight per unit length (N/m)


def route(a, hills, pa, pb):
    """The suspension points between the two dead ends: evenly spaced, split where a span is too low."""
    def top(x, y, ht):
        return height(hills, x, y) + ht

    def clear(p, q):
        # the lowest clearance along the still-air span p-q (heights above ground at each end given)
        za = top(p[0], p[1], p[2]) - (a.insulator if p[3] else 0.0)
        zb = top(q[0], q[1], q[2]) - (a.insulator if q[3] else 0.0)
        h = math.hypot(q[0] - p[0], q[1] - p[1])
        d = W * h * h / (8.0 * a.stringing_tension)
        worst = (1e30, 0.5)
        for k in range(1, 40):
            s = k / 40.0
            x, y = p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1])
            z = za + s * (zb - za) - 4.0 * d * s * (1.0 - s)
            g = z - height(hills, x, y)
            if g < worst[0]:
                worst = (g, s)
        return worst

    L = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
    n = max(1, math.ceil(L / a.max_span))
    pts = [(pa[0], pa[1], a.end_height, False)]
    for k in range(1, n):
        s = k / n
        pts.append((pa[0] + s * (pb[0] - pa[0]), pa[1] + s * (pb[1] - pa[1]), a.tower_height, True))
    pts.append((pb[0], pb[1], a.end_height, False))
    for _ in range(50):
        changed = False
        for i in range(len(pts) - 1):
            g, s = clear(pts[i], pts[i + 1])
            if g < a.min_clearance:
                p, q = pts[i], pts[i + 1]
                pts.insert(i + 1, (p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1]), a.tower_height, True))
                changed = True
                break
        if not changed:
            break
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--lx", type=float, default=3000.0)
    ap.add_argument("--ly", type=float, default=2000.0)
    ap.add_argument("--lz", type=float, default=800.0)
    ap.add_argument("--terrain_dx", type=float, default=25.0, help="spacing of the terrain file (m)")
    ap.add_argument("--hills", type=int, default=4)
    ap.add_argument("--hill_height", type=float, nargs=2, default=[60.0, 120.0])
    ap.add_argument("--hill_radius", type=float, nargs=2, default=[150.0, 250.0], help="e-folding radius (m)")
    ap.add_argument("--hill_spacing", type=float, default=550.0)
    ap.add_argument("--transformers", type=int, default=6)
    ap.add_argument("--on_hills", type=int, default=3)
    ap.add_argument("--transformer_spacing", type=float, default=450.0)
    ap.add_argument("--flat_height", type=float, default=2.0, help="highest terrain counted as flat ground (m)")
    ap.add_argument("--box", type=float, nargs=3, default=[8.0, 5.0, 6.0], help="transformer length, width, height (m)")
    ap.add_argument("--end_inset", type=float, default=0.7, help="the line ends lie this fraction of the way to the box's edge")
    ap.add_argument("--allowable_force", type=float, default=2.5e4, help="above a dead end's stringing tension (N)")
    ap.add_argument("--allowable_moment", type=float, default=2.5e5, help="(N m)")
    ap.add_argument("--end_height", type=float, default=10.0, help="dead ends above the terrain (m): the box top plus the bushings")
    ap.add_argument("--tower_height", type=float, default=30.0)
    ap.add_argument("--insulator", type=float, default=2.5)
    ap.add_argument("--tower", type=float, nargs=5, default=[6.0, 1.5, 0.2, 12.0, 1.2],
                    help="lattice towers: base width, top width, solidity, cross-arm length and depth (m)")
    ap.add_argument("--max_span", type=float, default=280.0)
    ap.add_argument("--stringing_tension", type=float, default=2.0e4, help="still-air horizontal tension every span is strung to (N)")
    ap.add_argument("--min_clearance", type=float, default=8.0)
    ap.add_argument("--u_ref", type=float, default=18.0, help="inflow speed at z_ref (m/s)")
    ap.add_argument("--z_ref", type=float, default=30.0)
    ap.add_argument("--z0", type=float, default=0.1)
    ap.add_argument("--inversion", type=float, default=500.0, help="base of the capping inversion (m)")
    ap.add_argument("--lapse", type=float, default=0.01, help="theta gradient above the inversion (K/m)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    hills = make_hills(rng, a)
    nx, ny = int(round(a.lx / a.terrain_dx)) + 1, int(round(a.ly / a.terrain_dx)) + 1
    x, y = np.linspace(0.0, a.lx, nx), np.linspace(0.0, a.ly, ny)
    z = hills_field(hills, x, y)
    with open(os.path.join(a.out, "terrain_hills.txt"), "w") as f:
        f.write(f"{nx}\n{ny}\n")
        f.write("\n".join(f"{v:.3f}" for v in x) + "\n")
        f.write("\n".join(f"{v:.3f}" for v in y) + "\n")
        f.write("\n".join(f"{v:.4f}" for v in z.ravel()) + "\n")

    # a neutral log law under a capping inversion, held above it
    ustar = a.u_ref * KAPPA / math.log((a.z_ref + a.z0) / a.z0)
    zs = np.arange(0.0, a.lz + 1e-6, 10.0)
    zl = np.minimum(zs, a.inversion)
    u = ustar / KAPPA * np.log((zl + a.z0) / a.z0)
    th = 300.0 + a.lapse * np.maximum(zs - a.inversion, 0.0)
    with open(os.path.join(a.out, "inflow_profile"), "w") as f:
        for zz, uu, tt in zip(zs, u, th):
            f.write(f"{zz:.1f} {uu:.6f} 0.0 0.0 {tt:.4f}\n")
    with open(os.path.join(a.out, "input_sounding"), "w") as f:
        f.write("1000.0 300.0 0.0\n")
        for zz, uu, tt in zip(zs, u, th):
            f.write(f"{zz:.1f} {tt:.4f} 0.0 {uu:.6f} 0.0\n")

    pts = make_transformers(rng, a, hills)
    edges = spanning_tree(pts)
    names = [f"T{i + 1}" for i in range(len(pts))]
    lines = []
    for n, (i, j) in enumerate(edges):
        pa, pb = end_on_box(pts[i], pts[j], a), end_on_box(pts[j], pts[i], a)
        lines.append((f"L{n + 1}", i, j, route(a, hills, pa, pb)))
    # the lines leaving one transformer must leave it in different directions
    for i, p in enumerate(pts):
        dirs = []
        for (_, ia, ib, _) in lines:
            if i in (ia, ib):
                q = pts[ib if ia == i else ia]
                dirs.append(math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])))
        for m in range(len(dirs)):
            for k in range(m + 1, len(dirs)):
                gap = abs((dirs[m] - dirs[k] + 180.0) % 360.0 - 180.0)
                if gap < 25.0:
                    raise SystemExit(f"two lines leave {names[i]} only {gap:.0f} deg apart; try another --seed")

    with open(os.path.join(a.out, "network.inputs"), "w") as f:
        f.write(f"# generated by make_case.py --seed {a.seed}: {len(pts)} transformers, {len(lines)} lines\n")
        for (xc, yc, h, s) in hills:
            f.write(f"#   hill at ({xc:.0f}, {yc:.0f}), height {h:.0f} m, radius {s:.0f} m\n")
        f.write("erf.conductors.spans        = " + " ".join(l[0] for l in lines) + "\n")
        f.write("erf.conductors.transformers = " + " ".join(names) + "\n")
        f.write("erf.conductors.tower_types  = lattice\n\n")
        tb, tt, ts, al, ad = a.tower
        f.write("# the suspension towers: square lattice, loaded by the wind on their members\n")
        f.write(f"erf.conductors.lattice.base_width = {tb:g}\n")
        f.write(f"erf.conductors.lattice.top_width  = {tt:g}\n")
        f.write(f"erf.conductors.lattice.solidity   = {ts:g}\n")
        f.write(f"erf.conductors.lattice.arm_length = {al:g}\n")
        f.write(f"erf.conductors.lattice.arm_depth  = {ad:g}\n\n")
        for nm, p in zip(names, pts):
            f.write(f"# {nm}: on {'a hilltop' if p[2] == 'hill' else 'flat ground'}, ground at {height(hills, p[0], p[1]):.1f} m\n")
            f.write(f"erf.conductors.{nm}.position         = {p[0]:.2f} {p[1]:.2f}\n")
            f.write(f"erf.conductors.{nm}.size             = {a.box[0]:g} {a.box[1]:g} {a.box[2]:g}\n")
            f.write(f"erf.conductors.{nm}.allowable_force  = {a.allowable_force:g}\n")
            f.write(f"erf.conductors.{nm}.allowable_moment = {a.allowable_moment:g}\n")
        for (nm, i, j, r) in lines:
            f.write(f"\n# {nm}: {names[i]} to {names[j]}, {len(r) - 1} span(s)\n")
            f.write(f"erf.conductors.{nm}.end_a  = {r[0][0]:.2f} {r[0][1]:.2f} {r[0][2]:g}\n")
            if len(r) > 2:
                f.write(f"erf.conductors.{nm}.towers = " + "  ".join(f"{p[0]:.2f} {p[1]:.2f} {p[2]:g}" for p in r[1:-1]) + "\n")
            f.write(f"erf.conductors.{nm}.end_b  = {r[-1][0]:.2f} {r[-1][1]:.2f} {r[-1][2]:g}\n")
            # ERF sets each span's length from its chord on ERF's own terrain
            f.write(f"erf.conductors.{nm}.stringing_tension = {a.stringing_tension:g}\n")
            f.write(f"erf.conductors.{nm}.diameter         = 0.0281\n")
            f.write(f"erf.conductors.{nm}.mass_per_length  = 1.628\n")
            f.write(f"erf.conductors.{nm}.axial_stiffness  = 3.0e7\n")
            if len(r) > 2:
                f.write(f"erf.conductors.{nm}.tower_type       = lattice\n")
                f.write(f"erf.conductors.{nm}.insulator_length = {a.insulator:g}\n")
                f.write(f"erf.conductors.{nm}.insulator_mass   = 60.\n")
    print(f"u* = {ustar:.4f} m/s, inflow KE 3.3 u*^2 = {3.3 * ustar ** 2:.4f} m^2/s^2")
    for (nm, i, j, r) in lines:
        print(f"{nm}: {names[i]} ({pts[i][2]}) to {names[j]} ({pts[j][2]}), {len(r) - 1} spans")


if __name__ == "__main__":
    main()
