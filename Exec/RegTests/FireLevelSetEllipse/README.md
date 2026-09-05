# FireLevelSetEllipse

The wind-driven spread ellipse on the level-set path,
`erf.fire.levelset.ellipse`, the Huygens construction the FARSITE family
uses (Finney 1998; ELMFIRE, Prometheus, Cell2Fire): every point of the front
spreads as an ellipse with Anderson's (1983) length-to-width ratio from the
midflame wind, Alexander's (1985) head-to-back ratio, and the rate-of-spread
model's rate at the head; the level set is advanced at the ellipse's normal
speed, its support function. Off by default. The projection-based
`erf.fire.directional_ros` remains the recommended way to shape the front
when the flow around the fire is resolved (see the theory page for why the
ellipse double counts a resolved wind); the ellipse is for comparison with
those models and for the calibration they carry.

A grass fire on flat ground under a westerly, one-way, five decks:

- `off`: the historical disc; `off_key` writes the key out and must
  reproduce it line for line.
- `directional`: the projection, for comparison.
- `ellipse`: Anderson's ratio from the wind, which the code prints each step.
- `ellipse_lw3`: fixed ratio 3 (`erf.fire.levelset.ellipse_lw`).

```
MPIRUN="mpirun -np 4" ./run_ellipse.sh /path/to/erf_exec
python3 plot_shapes.py
```

The script measures the burned region at 60 s from the fire plotfile (head,
back and half-width from the ignition point). The 6 m ignition disc
dominates a 13 m head run, so the bounding-box ratio cannot reach L/W in
60 s; the check is the Huygens envelope of the disc instead: from the
measured head travel h, the ellipse predicts a back travel h/HB and a
half-width travel h (1 + 1/HB)/(2 L/W), and both extents must match to one
fire cell (1.25 m) for the fixed ratio 3 and for the time-mean Anderson
ratio the code printed. The script also checks that the ellipse leaves the
head extent of the disc unchanged to 10 % and that its back extent is
smaller. The unit test
`Tests/Unit/Fire/ERF_GTestLevelSetEllipse.cpp` checks the rates at the head,
flanks and back exactly. The plot overlays the four perimeters.
