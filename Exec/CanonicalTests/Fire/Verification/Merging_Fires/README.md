# Merging_Fires

Two fires growing into each other at a constant rate. At time t the burned
region is the union of two discs of radius r = r_ig + R t, so the arrival time is
(the smaller distance to the two centres - r_ig) / R and, once the discs overlap
(r > d/2), the area is 2 pi r^2 minus the lens 2 r^2 acos(d/2r) - (d/2)
sqrt(4 r^2 - d^2). Where they meet, the front is a pair of inward cusps that
must fill at the geometric rate without the fire running ahead of them.

```
MPIRUN="mpirun -np 2" ./run_merging_fires.sh /path/to/erf_exec
```

## The case

400 x 200 m, still air, 2 m fire cells, a prescribed 1 m/s for 80 s. Two 6 m
discs at (160, 100) and (240, 100), 80 m apart, both at t = 0 from the ignition
schedule `two_discs.csv`; they touch at t = 34 s. `check_merging_fires.py`
compares the burned area with the union formula at every plotfile, the arrival
time over the grid, and the arrival time within 10 m of the midpoint, the neck.
It takes the time the schedule stamped the discs from the arrival time inside
them (0.00 s here).

## Expected Results

On two ranks:

- burned area 0.23 to 0.33 cell widths of perimeter short of the union
  formula from 10 s to 80 s;
- arrival time over 8272 cells: mean error +0.167, 95th percentile 0.283 cell
  crossings (2 s); across the neck, 744 cells, +0.161 and 0.283.

The discs start small (6 m on 2 m cells) and the default level set's artificial
viscosity slows a tightly curved front most, which is where the area deficit
comes from; the neck fills with the same small lag as the rest of the front.
The checks allow half a cell width of perimeter on the area and half a cell
(one at the 95th percentile) on the arrival time.
