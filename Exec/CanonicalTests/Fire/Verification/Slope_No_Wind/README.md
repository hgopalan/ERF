# Slope_No_Wind

A grass fire on an inclined plane in still air, against Rothermel's slope factor.
Rothermel (1972) spreads a fire on a slope of tangent s at R0 (1 + phi_s), with
phi_s = 5.275 beta^-0.3 s^2; the level set projects the gradient onto the terrain,
so a direction with slope s_n along it spreads at R / sqrt(1 + s_n^2) in map view.

```
python3 gen_planes.py                                             # plane_s30.txt, plane_s60.txt and the _wide pair (committed)
MPIRUN="mpirun -np 2" ./run_slope_no_wind.sh /path/to/erf_exec   # eight decks, then the checks
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
| `ell_s30`, `ell_s60` | tan = 0.3, 0.6 | the directional level set with `directional_shape = "ellipse"` |
| `and_s30`, `and_s60` | tan = 0.3, 0.6 | the same with `directional_ellipse_lw = "anderson"`, on a 400 x 400 m domain ignited at (80, 200) |

`check_slope_no_wind.py` compares every deck with the exact solution of its own
level-set equation, phi_t + F(n) |grad phi| = 0 with F(n) = R(n) / sqrt(1 + (s . n)^2),
from an independent port of the Rothermel equations. From the ignition disc that
solution is the Hopf formula T(x) = max over n of ((x - c) . n - r0) / F(n),
evaluated at the plotfile's cell centres. The run and T are fitted the same way:
the time the front first reaches each column (up and down the slope) or row
(across it) against its distance from the ignition point. Across the slope that
is the growth of the half-width.

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
phi_w (B - 1) > 1 (`Exec/RegTests/FireDirectionalShape`).

The `ell_*` decks remove the shortfall. `erf.fire.directional_shape = "ellipse"`
takes R(n) from the support function of the ellipse with head R0 (1 + phi_s)
upslope and back and flank rates R0, the projection's own rates at those three
normals. An ellipse is convex and so its own Wulff shape, and the head runs at
Rothermel's head rate. The check requires all four directions at the Hopf rates
and the head at R0 (1 + phi_s) / sqrt(1 + s^2), each to 3 %.

## Expected Results

On two ranks, map-view rates [m/s]. In brackets: the rate against the exact
solution of the deck's own equation where the fit spans 20 cells or more, and
otherwise the mean distance of the run's front from the exact front (the backs
and flanks at R0 travel 7 to 9 cells, where the check is on position):

| deck | up the slope | down the slope | across |
|---|---|---|---|
| `iso_s30` | 0.10809 (-0.17 %) | 0.10808 (-0.20 %) | 0.11280 (-0.22 %) |
| `iso_s60` | 0.32563 (-0.07 %) | 0.32530 (-0.19 %) | 0.37921 (-0.24 %) |
| `dir_s30` | 0.10175: Wulff 0.09136, Rothermel 0.10826 (61 % of the way) | 0.02293 (0.04 cell) | 0.02399 (0.03 cell) |
| `dir_s60` | 0.25676: Wulff 0.18272, Rothermel 0.32585 (52 % of the way) | 0.02052 (0.04 cell) | 0.02401 (0.02 cell) |
| `ell_s30` | 0.10768 (-0.54 % of Rothermel) | 0.02414 (0.13 cell) | 0.02401 (0.01 cell) |
| `ell_s60` | 0.32498 (-0.27 % of Rothermel) | 0.02147 (0.13 cell) | 0.02402 (0.01 cell) |
| `and_s30` | 0.10798 (-0.26 % of Rothermel) | 0.02296 (0.03 cell) | 0.06569 (-0.24 %) |
| `and_s60` | 0.32534 (-0.16 % of Rothermel) | 0.02057 (0.03 cell) | 0.13352 (-0.08 %) |

The slope factor and the ground projection are exact to a quarter of a percent.
The default directional head of a point fire falls 6 % and 21 % short of
Rothermel's; the ellipse shape brings it within 0.6 %.

The `and_*` decks keep those heads and backs and widen the flanks to Anderson's
length-to-width ratio at the effective wind speed. At tan 0.3 that speed is
0.958 m/s and the ratio 1.04, so the fire is nearly round. At tan 0.6 the speed
reaches the 1.524 m/s fine-fuel wind limit and the ratio is 1.51. The flank rates
match the exact ones to a quarter of a percent.

The ellipse's back is pointed (a radius of curvature a^2/b, one to two cells
here), and for a few hundred seconds the rows either side of the centre line
burn one cell ahead of it. Over the 7 to 9 cells the back travels that reads as a
fitted downslope rate 4 to 5 % high, while the front itself never strays more
than 0.26 cell from the exact one.
