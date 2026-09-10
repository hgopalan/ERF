# Speed_Gradient

A point fire where the rate of spread varies linearly in space. The eikonal
equation |grad T| = 1/R with R(p) = R0 + g . (p - s) has the closed-form travel
time from a point source s, the constant-velocity-gradient result of seismology,

    T(p) = (1/|g|) acosh(1 + |g|^2 |p - s|^2 / (2 R(s) R(p))):

rays are circular arcs and the fronts are circles whose centres move up the
gradient. Along the gradient it reduces to ln(R(p)/R(s)) / |g|. It is the one
case here whose answer is not straight-line geometry.

```
MPIRUN="mpirun -np 2" ./run_speed_gradient.sh /path/to/erf_exec
```

## The case

400 x 400 m, still air, 2 m fire cells, `erf.fire.prescribed.gradient = 0.004 0.0`
about (200, 200): R = 1 + 0.004 (x - 200) m/s, from 0.2 m/s at the west edge
to 1.8 m/s at the east edge. A 4 m ignition disc at the centre, 150 s.
`check_speed_gradient.py` checks the rate field, the arrival time over the grid
and separately up and down the gradient (subtracting r_ig / R(s) for the
ignition disc, exact to 0.06 s), and the front's extent along y = 200 at 150 s.

## Expected Results

On two ranks:

- the rate field equals R(x) to 2e-16;
- arrival time over 19680 cells: mean error +0.063, mean |e| 0.080, 95th
  percentile 0.171 cell crossings (2 s at R0); up the gradient 0.085 / 0.176,
  down it 0.071 / 0.153;
- the front along the centre line at x = 399.0 m to the east (it has reached
  the edge) and 85.0 m to the west, both within 0.01 cells of ln(R/R0)/g.
