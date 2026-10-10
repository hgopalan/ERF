# Fuel_Interface_Refraction

A straight front crossing from one fuel into another. Huygens' principle bends
it like light at an interface: a plane front with its normal at theta1 to the
boundary normal leaves it at theta2, with Snell's law

    sin(theta2) / R2 = sin(theta1) / R1,

and on the far side the arrival time is the plane

    T2 = T1(200, y_q) + (x - 200) / (R2 cos theta2),   y_q = y - (x - 200) tan theta2.

```
python3 gen_refraction.py                                                     # fuel map and lines (committed)
MPIRUN="mpirun -np 2" ./run_fuel_interface_refraction.sh /path/to/erf_exec
```

## The case

400 x 400 m, still air, 2 m fire cells. Fuel code 1 for x < 200 m and code 2
beyond (`fuel_halves.asc`), with `erf.fire.prescribed.by_fuel` giving each its
rate. A straight 6 m wide ignition line through (60, 200). Both decks run 400 s.

| deck | R1 | R2 | theta1 | Snell theta2 |
|---|---|---|---|---|
| `fast_to_slow` | 1.0 m/s | 0.5 m/s | 40 deg | 18.747 deg |
| `slow_to_fast` | 0.5 m/s | 1.0 m/s | 25 deg | 57.697 deg |

`check_fuel_interface_refraction.py` checks the rate field in each fuel, the
arrival time in each fuel, and fits a plane to the second fuel's arrival times,
whose gradient must have magnitude 1/R2 (1 %) and point along theta2 (0.5 deg).
With a constant rate on each side the arrival time is a minimum over the source
points of the line, so a point is compared only if its own optimal source lies on
the stretch of the line inside the domain and the first fuel (the walls cut the
rest off). The 40 degree line runs into the second fuel near the bottom of the
domain; that stretch ignites it directly, and the check takes the earlier of the
two arrivals there.

## Expected Results

On two ranks, arrival errors in cell-crossing times h/R:

| deck | first fuel mean \|e\| / 95th pct | second fuel mean \|e\| / 95th pct | transmitted \|grad T\| | transmitted angle |
|---|---|---|---|---|
| `fast_to_slow` | 0.317 / 0.720 | 0.991 / 1.425 (fails the half-cell check) | 2.00628 s/m (1/R2 = 2) | 18.964 deg |
| `slow_to_fast` | 0.237 / 0.490 | 0.181 / 0.427 | 1.00081 s/m (1/R2 = 1) | 58.045 deg |

Re-measured 2026-10-09: 7 of the 8 checks pass. The refracted front leaves at
Snell's angle to 0.35 deg and at the second fuel's rate to 0.3 %, but behind
the interface where the rate halves it runs one crossing of the slow fuel
late (mean +0.99, 4 s), beyond the half-cell tolerance; the code before this
one fails the same check at +0.93 (its other errors 0.235 / 0.607, 0.186 /
0.433 and 0.133 / 0.328), and the 0.100 / 0.228 and 0.108 / 0.170 this
README carried until then predate both. The lag at a rate step is open.
