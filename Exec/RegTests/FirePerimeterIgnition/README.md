# FirePerimeterIgnition

Starting a fire from an observed perimeter with a spin-up, WRF-SFIRE's
perimeter time as the Community Fire Behavior Model runs use it (Jiménez y
Muñoz et al. 2026). A grass fire on flat ground is started from a 40 m square
perimeter (`square_40m.csv`), one-way so the atmosphere is identical in every
deck:

- `t0`: polygon stamped at initialisation, interior ignited with its fuel
  intact (the historical behaviour); `t0_key` writes the default keys out and
  must reproduce it line for line.
- `spinup`: `erf.fire.ignition.polygon_time = 30`, no fire for 30 s.
- `interior`: `erf.fire.ignition.polygon_interior_ros = 0.5` and
  `polygon_interior_tau = 60`: every cell at distance d inside the perimeter
  gets arrival time -d/R and fuel w0 exp(-d/(R tau)).
- `spinup_interior`: both.

```
MPIRUN="mpirun -np 4" ./run_perimeter.sh /path/to/erf_exec
python3 plot_perimeter.py
```

The script checks the line-for-line reproduction, the absence of fire before
the stamp, the burned cells one step after the stamp against the historical
deck, the interior fuel and arrival time cell by cell from the step-0 fire
plotfile (to 1e-6), the arrival times of the spin-up interior deck at 30 s
(to 0.3 s, the front having moved a step), and the centre probe's arrival
time (-40 s and -10 s). The plot shows the fuel and arrival time of every
interior cell against the distance inside the perimeter with the expected
curves.
