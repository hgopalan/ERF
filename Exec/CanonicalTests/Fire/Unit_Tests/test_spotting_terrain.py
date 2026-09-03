#!/usr/bin/env python3
"""
Terrain-aware firebrand trajectory unit tests.

A firebrand is lofted a height H_z above the ground it starts on and falls at
its terminal velocity while the wind carries it. The ground it lands on is not
the height it left: a brand drifting downslope falls further and lands further
out, and one drifting upslope meets the rising ground sooner. These tests cover
the descent, its reduction to the flat-earth trajectory when there is no
terrain, and the step size that has to cover the longest possible fall.

Reference implementation extracted from:
  - ERF_AlbiniSpotting.H: compute_albini_spotting() trajectory loop,
                          albini_lofting_height()

Run: python3 test_spotting_terrain.py
"""

import sys
import math


def lofting_height(I_B_kW, I_B_min=100.0):
    """
    Albini (1983) lofting height, H_z = 12.2 I_B^(1/3), zero below threshold.
    """
    if I_B_kW < I_B_min:
        return 0.0
    return 12.2 * I_B_kW ** (1.0 / 3.0)


def fly(H_z, z_src, ground_of_x, u_wind, w_terminal, n_steps=20,
        z_ground_min=0.0, x0=0.0):
    """
    Integrate a firebrand from its source to the ground.

    Mirrors the trajectory loop: the step size comes from the longest fall the
    brand could make, down to the lowest ground in the domain, so a fixed step
    count always covers the descent. The brand lands when its altitude drops to
    the ground beneath it.

    Returns (landing_x, flight_time, steps_taken).
    """
    z_brand = z_src + H_z
    t_max = (z_brand - z_ground_min) / w_terminal if w_terminal > 0.0 else 0.0
    dt = t_max / n_steps if n_steps > 0 else 0.0
    dz = w_terminal * dt

    x = x0
    for ns in range(n_steps):
        x += u_wind * dt
        z_brand -= dz
        if z_brand <= ground_of_x(x):
            return x, (ns + 1) * dt, ns + 1
    return x, n_steps * dt, n_steps


def check(failures, name, condition, detail=""):
    status = "✓" if condition else "✗"
    print(f"{status} {name}")
    if not condition:
        if detail:
            print(f"    {detail}")
        failures.append(name)


def test_flat_matches_the_old_flight_time():
    """With no terrain the trajectory is the flat-earth one, unchanged."""
    failures = []
    H_z, u, w = 60.0, 8.0, 1.5
    flat = lambda x: 0.0

    x_land, t_flight, steps = fly(H_z, 0.0, flat, u, w, n_steps=20)

    # The superseded model flew for exactly H_z / w_terminal and drifted u * t
    t_expected = H_z / w
    check(failures, "Flat terrain reproduces the flat-earth flight time",
          abs(t_flight - t_expected) < 1.0e-12,
          f"got {t_flight}, expected {t_expected}")
    check(failures, "Flat terrain reproduces the flat-earth landing point",
          abs(x_land - u * t_expected) < 1.0e-9,
          f"got {x_land}, expected {u * t_expected}")
    check(failures, "Flat terrain uses every trajectory step", steps == 20,
          f"took {steps} steps")
    return failures


def test_downslope_flies_further():
    """Drifting onto lower ground extends the fall and the reach."""
    failures = []
    H_z, u, w = 60.0, 8.0, 1.5

    # Ground descending at 10% in the direction of drift
    downslope = lambda x: -0.10 * x
    flat = lambda x: 0.0

    x_flat, t_flat, _ = fly(H_z, 0.0, flat, u, w, n_steps=400, z_ground_min=0.0)
    x_down, t_down, _ = fly(H_z, 0.0, downslope, u, w, n_steps=400,
                            z_ground_min=-200.0)

    check(failures, "Downslope drift lands further than flat",
          x_down > x_flat + 1.0,
          f"downslope {x_down:.1f} m vs flat {x_flat:.1f} m")
    check(failures, "Downslope drift stays aloft longer",
          t_down > t_flat,
          f"{t_down:.1f} s vs {t_flat:.1f} s")
    return failures


def test_upslope_lands_sooner():
    """Rising ground cuts the fall short."""
    failures = []
    H_z, u, w = 60.0, 8.0, 1.5

    upslope = lambda x: 0.10 * x
    flat = lambda x: 0.0

    x_flat, t_flat, _ = fly(H_z, 0.0, flat, u, w, n_steps=400)
    x_up, t_up, _ = fly(H_z, 0.0, upslope, u, w, n_steps=400)

    check(failures, "Upslope drift lands closer than flat",
          x_up < x_flat - 1.0, f"upslope {x_up:.1f} m vs flat {x_flat:.1f} m")
    check(failures, "Upslope drift spends less time aloft",
          t_up < t_flat, f"{t_up:.1f} s vs {t_flat:.1f} s")
    return failures


def test_ridge_to_valley_reach():
    """A brand leaving a ridge crest reaches well beyond the flat estimate."""
    failures = []
    # 60 m loft from a 150 m crest, falling into ground 150 m lower
    H_z, u, w = 60.0, 12.0, 1.5
    z_src = 150.0
    valley = lambda x: max(0.0, 150.0 - 0.30 * x)

    x_land, t_land, _ = fly(H_z, z_src, valley, u, w, n_steps=800,
                            z_ground_min=0.0)
    x_flat, t_flat, _ = fly(H_z, 0.0, lambda x: 0.0, u, w, n_steps=800)

    check(failures, "Ridge-to-valley brand outruns the flat-earth estimate",
          x_land > 1.5 * x_flat,
          f"terrain {x_land:.0f} m vs flat {x_flat:.0f} m")
    return failures


def test_step_size_covers_the_longest_fall():
    """The step count always covers the descent to the lowest ground."""
    failures = []
    H_z, u, w = 60.0, 8.0, 1.5
    n = 20
    z_src, z_min = 150.0, 0.0

    # Worst case: the brand falls from z_src + H_z all the way to z_min
    t_max = (z_src + H_z - z_min) / w
    dt = t_max / n
    total_drop = w * dt * n
    check(failures, "Fixed step count spans the full possible descent",
          abs(total_drop - (z_src + H_z - z_min)) < 1.0e-9,
          f"covers {total_drop} m of {z_src + H_z - z_min} m")

    # And a brand over the lowest ground does land within the step budget
    _, _, steps = fly(H_z, z_src, lambda x: 0.0, u, w, n_steps=n,
                      z_ground_min=z_min)
    check(failures, "Brand lands within the step budget", steps <= n,
          f"took {steps} of {n}")
    return failures


def test_lofting_height_threshold():
    """No loft below the minimum intensity; cube-root growth above it."""
    failures = []
    check(failures, "No loft below the intensity threshold",
          lofting_height(50.0) == 0.0)
    h1 = lofting_height(100.0)
    h8 = lofting_height(800.0)
    check(failures, "Loft grows as the cube root of intensity",
          abs(h8 / h1 - 2.0) < 1.0e-12, f"ratio {h8 / h1}")
    return failures


def main():
    print("=" * 70)
    print("Terrain-Aware Firebrand Trajectory Unit Tests")
    print("=" * 70)

    all_failures = []
    all_failures.extend(test_flat_matches_the_old_flight_time())
    all_failures.extend(test_downslope_flies_further())
    all_failures.extend(test_upslope_lands_sooner())
    all_failures.extend(test_ridge_to_valley_reach())
    all_failures.extend(test_step_size_covers_the_longest_fall())
    all_failures.extend(test_lofting_height_threshold())

    print("\n" + "=" * 70)
    if not all_failures:
        print("FINAL RESULT: All tests PASSED")
        return 0
    print(f"FINAL RESULT: {len(all_failures)} test(s) FAILED")
    for failure in all_failures:
        print(f"  ✗ {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
