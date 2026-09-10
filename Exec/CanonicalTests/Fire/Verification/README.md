# Verification

Fire cases with an independent answer: an exact solution of the equations the
model solves, or a published result. Each folder holds the decks, the script that
generates any rasters or ignition files (their output is committed), a
`run_<case>.sh` that runs every deck and then the check, and a `check_<case>.py`
that compares the output with the reference and exits non-zero on any failure.
Every README records the numbers measured with the committed decks.

```
MPIRUN="mpirun -np 2" ./run_<case>.sh /path/to/erf_exec   # runs, then checks
SKIP_RUN=1 ./run_<case>.sh x                              # checks only
```

The geometry cases use `erf.fire.ros_model = "prescribed"`, a rate set in the
deck with no wind, slope or moisture dependence and no direction, so the level set
solves the eikonal equation |grad T| = 1/R as posed and the arrival time is a
shortest travel time. The atmosphere is still air under slip walls in all but
Calm_Plume, and the fire grid is 2 m.

| case | reference | what it checks |
|---|---|---|
| `Level_Set_Advection` | circle, r = r0 + R t | the level-set schemes (first order, WENO5-Z, hybrid) and reinitialisation |
| `Obstacle_Shadow` | taut string around a disc | the non-burnable mask and diffraction behind it |
| `Junction_Fire` | distance to a V; meeting point at R / sin(theta/2) | fronts meeting at an angle (Viegas et al. 2012 for the laboratory) |
| `Polygon_Growth` | Steiner's formula; offset area of a cross | convex corners rounding, inner corners staying sharp |
| `Merging_Fires` | union of two discs | two fires joining, the neck filling |
| `Fuel_Interface_Refraction` | Snell's law, sin theta2 / R2 = sin theta1 / R1 | a front crossing a fuel boundary, per-fuel rates |
| `Speed_Gradient` | eikonal travel time in a linear speed gradient | a spatially varying rate |
| `Slope_No_Wind` | Rothermel's slope factor; the Wulff shape | slope along the front normal, the ground projection, the ellipse shape of the directional rate |
| `Moisture_Relaxation` | time-lag ODE solution; Rothermel at the moisture | dynamic dead-fuel moisture and extinction |
| `Calm_Plume` | heat budget (checked); Heskestad, Morton-Taylor-Turner and Briggs (reported) | the heat the coupling injects; the plume it raises, under-entrained at 40 m |

The coupled no-wind line fire of Coen et al. (2013) is `Exec/RegTests/FireLineFire`
(`nowind_2way`), next to the wind decks of the same case, and the two options these
cases rely on have a regression test of their own in `Exec/RegTests/FirePrescribed`.

What the cases found, recorded in their READMEs:

- `Slope_No_Wind`: the directional level set cannot give a point fire Rothermel's
  head rate once the slope (or wind) factor is large; the exact solution of the
  equation it solves is the Wulff shape, whose head is slower. The opt-in
  `erf.fire.directional_shape = "ellipse"` takes the rate from a convex shape with
  the same head, back and flank rates and restores the head (the `ell_*` decks;
  `Exec/RegTests/FireDirectionalShape` does the same in wind).
- `Moisture_Relaxation`: the equilibrium-moisture hysteresis picks the adsorption
  and desorption curves the wrong way round.
- `Obstacle_Shadow`: the front that wraps a masked obstacle lags half a cell
  beyond what the cell-resolved obstacle explains.
