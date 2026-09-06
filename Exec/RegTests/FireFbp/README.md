# FireFbp

The Canadian Forest Fire Behavior Prediction System rate of spread
(Forestry Canada 1992; Wotton, Alexander and Taylor 2009) as
`erf.fire.ros_model = fbp`: sixteen fuel types (C1-C7, D1, M1-M4, S1-S3,
O1a, O1b) with `RSI = a (1 - exp(-b ISI))^c`, the ISI from the FFMC and
the 10 m wind, the buildup effect from the BUI, the grass curing factor,
the mixedwood weighting by percent conifer or dead fir, and slope through
the system's equivalent wind. The inputs are the daily indices,
`erf.fire.fbp.*`, uniform over the domain; the wind is the reference-height
wind by default, so `erf.fire.wind_ref_ht` is set to 10 m here. The
level-set path's `erf.fire.directional_ros` evaluates the rate along the
front normal, as for the other models.

Five one-way decks under a uniform 8 m/s westerly: Rothermel on short grass
(for the table), C-2 at FFMC 90 and BUI 60, O-1b at 80 % curing, M-1 at
60 % conifer, and C-2 with the directional rate.

```
MPIRUN="mpirun -np 4" ./run_fbp.sh /path/to/erf_exec
python3 plot_fbp.py
```

At the first step the wind is 8 m/s at every height, so the largest rate
of spread the code prints is the FBP head rate at that wind on flat ground.
The script checks it against `fbp_reference.py`, an independent
implementation of the equations, for the three fuel types (to 1e-8), and
that the directional deck's head rate equals the isotropic one while it
burns less area. The unit test `ERF_GTestFbp` checks the invariants of the
equations: the buildup effect at BUI = BUI0, the curing factor at 100 %,
the mixedwood limits, and the slope-equivalent wind reproducing the
zero-wind rate on the slope.
