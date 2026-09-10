# Slope_No_Wind

A grass fire on an inclined plane in still air, against Rothermel's slope factor.
Rothermel (1972) spreads a fire on a slope of tangent s at R0 (1 + phi_s), with
phi_s = 5.275 beta^-0.3 s^2; the level set projects the gradient onto the terrain,
so a direction with slope s_n along it spreads at R / sqrt(1 + s_n^2) in map view.

```
python3 gen_planes.py                                             # plane_s30.txt, plane_s60.txt (committed)
MPIRUN="mpirun -np 2" ./run_slope_no_wind.sh /path/to/erf_exec   # four decks, then the checks
```

## The case

400 x 200 m, flat still atmosphere; the fire reads its own terrain,
`erf.fire.terrain_file_name`, the plane z = s x. Anderson fuel model 1 at 5.5 %
moisture (R0 = 0.02403 m/s, 5.275 beta^-0.3 = 41.146), a 10 m ignition disc at
(80, 100), 1000 s.

| deck | slope | level set |
|---|---|---|
| `iso_s30`, `iso_s60` | tan = 0.3, 0.6 | `directional_ros = false`: the whole front at R0 (1 + phi_s) |
| `dir_s30`, `dir_s60` | tan = 0.3, 0.6 | the default directional level set |

`check_slope_no_wind.py` fits a line to the arrival time along each axis and
compares the map-view rate with the expected one, from an independent port of
the Rothermel equations.

On the isotropic path every direction spreads at R0 (1 + phi_s), reduced by the
ground projection along the slope. On the directional path the model is
evaluated with the slope along the front normal, clamped at zero downslope, so
backing and flanks spread at R0.

The directional head is the finding of this case. For a rate that peaks as
sharply about one direction as R(n) = R0 (1 + 41 max(s n_x, 0)^2), the exact
solution of the level-set equation from a point ignition is the Wulff shape, whose
head is a wedge of oblique facets running at min over n of
R(n) / (n_x sqrt(1 + (s . n)^2)): about 2 sqrt(phi_s) R0 rather than R0 (1 + phi_s).
A straight line fire keeps n = x and is not affected, which is why FireLineFire
meets Rothermel's head rate; a point or finite fire is. The scheme, which freezes
R(n) from central differences, lands between the two, and the check only requires
that, printing where. The same argument applies to the wind factor wherever
phi_w (B - 1) > 1.

## Expected Results

On two ranks, map-view rates [m/s]:

| deck | up the slope | down the slope | across |
|---|---|---|---|
| `iso_s30` | 0.10808 (-0.16 %) | 0.10808 (-0.17 %) | 0.11278 (-0.22 %) |
| `iso_s60` | 0.32557 (-0.08 %) | 0.32530 (-0.17 %) | 0.37921 (-0.21 %) |
| `dir_s30` | 0.10175: Wulff 0.09136, Rothermel 0.10826 (61 % of the way) | 0.02292 (-0.42 %) | 0.02397 (-0.24 %) |
| `dir_s60` | 0.25824: Wulff 0.18272, Rothermel 0.32585 (53 % of the way) | 0.02051 (-0.46 %) | 0.02397 (-0.24 %) |

The slope factor and the ground projection are exact to a quarter of a percent;
the directional head of a point fire falls 6 % and 21 % short of Rothermel's.
