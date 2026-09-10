# Junction_Fire

Two fire lines meeting at an angle, the geometry of a junction or "jump" fire.
With a constant rate R the burned region at time t is every point within
w + R t of the ignition polyline (w its half width), so every point burns at
(distance to the V - w) / R. Inside the wedge the two inner fronts meet on the
bisector, and the meeting point runs at R / sin(theta/2): faster the narrower
the angle.

```
python3 gen_junction.py                                         # v30.csv, v60.csv, v90.csv (committed)
MPIRUN="mpirun -np 2" ./run_junction_fire.sh /path/to/erf_exec  # three decks, then the checks
```

## The case

400 x 200 m, still air, 2 m fire cells, a prescribed 1 m/s. A V of two 180 m
arms, apex at (40, 100), opening towards +x, stamped as an 8 m wide polyline.

| deck | angle | meeting point R / sin(theta/2) | run |
|---|---|---|---|
| `v30` | 30 deg | 3.864 m/s | 60 s |
| `v60` | 60 deg | 2.000 m/s | 110 s |
| `v90` | 90 deg | 1.414 m/s | 150 s |

`check_junction_fire.py` compares the arrival time over the grid wherever the
nearest point of the V is inside the domain, along the bisector inside the
wedge, and fits a line to the bisector arrival times for the meeting-point speed.

Viegas et al. (2012) measured junction fires without wind or slope in the
laboratory and found the meeting point faster still than the geometry, from the
convective interaction of the two flames: a one-way model with a prescribed rate
leaves that out by design, and a coupled run would be the place to look for it.

## Expected Results

On two ranks, arrival errors in cell-crossing times (h/R = 2 s):

| deck | whole field mean \|e\| / 95th pct | bisector mean \|e\| / 95th pct | meeting-point speed |
|---|---|---|---|
| `v30` | 0.172 / 0.395 | 0.147 / 0.387 | 3.896 m/s (+0.85 %) |
| `v60` | 0.159 / 0.334 | 0.172 / 0.183 | 2.000 m/s (-0.02 %) |
| `v90` | 0.305 / 0.423 | 0.340 / 0.488 | 1.416 m/s (+0.15 %) |

The fronts lag the exact ones by a fifth to a third of a cell, the stamping of
the thin line and the artificial viscosity of the default level set on the
rounded outer corner; the meeting point runs at the geometric speed to within 1 %.
The checks allow half a cell on average and one cell at the 95th percentile
(1.5 along the bisector), and 2 % on the speed.

## References

Viegas, D. X., J. R. Raposo, D. A. Davim and C. G. Rossa (2012). Study of the
jump fire produced by the interaction of two oblique fire fronts. Part 1.
Analytical model and validation with no-slope laboratory experiments.
International Journal of Wildland Fire, 21, 843-856.
