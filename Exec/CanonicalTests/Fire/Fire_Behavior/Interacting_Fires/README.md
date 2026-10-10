# Interacting_Fires

The three ways fires meet, on short grass in a wind: spot fires coalescing,
two lines meeting at an angle (a junction fire) and two parallel lines burning
towards each other. Anderson fuel model 1 at 8 % moisture on flat ground,
Rothermel's rate with Anderson's ellipse (head, flank and backing rates from
the wind), 10 m fire cells over a 2 km square. The fire is given a constant
5 m/s westerly reference wind (`erf.fire.prescribed_wind`), reduced to midflame
height by the wind adjustment factor, so the rates do not change through the
900 s run and each deck is a pure fire-spread example in a still, one-way
atmosphere. For a coupled run replace the prescribed wind with a sounding or
geostrophic wind and set `erf.fire.coupling_type = "lagged"`.

```
MPIRUN="mpirun -np 2" ./run_interacting_fires.sh /path/to/erf_exec    # 5 to 6 min per deck on two ranks, Release
```

| deck | ignition | what happens |
|---|---|---|
| `inputs_coalescing` | five 15 m spot fires from `coalescing.csv`, lit at 0, 60, 120, 180 and 240 s, 40 to 60 m apart | each grows into an ellipse with its head to the east; the heads run into the spots downwind within a few minutes, the flanks into the spots beside them more slowly |
| `inputs_junction` | one 20 m wide polyline, `junction_v60.csv`: a V of two 400 m arms at 60 degrees, apex (500, 1000) m, opening downwind | the inner fronts meet on the bisector and their meeting point runs east along it at 0.24 m/s, twice the inner fronts' own normal rate |
| `inputs_parallel` | two 20 m wide lines along the wind, 60 m apart (y = 970 and 1030 m), from `parallel_south.csv` and `parallel_north.csv` | the heads run east together; the inner flanks close the 40 m strip between the lines at about 720 s |

The parallel pair is set up with two vertex files in one key,
`erf.fire.ignition.polygon_file = "parallel_south.csv" "parallel_north.csv"`;
every file is one perimeter, stamped with the merging rule of every ignition.

`report_interacting_fires.py` reads every fire plotfile and prints the burned
area, the arrival time at the meeting cells (the places the fronts of two
fires reach from opposite sides) and the reference wind the fire saw.

## Measured results

Two ranks, the Release build; the meeting cells and their arrival times:

| deck | meeting cells (x, y) m | arrival time [s] | burned area at 900 s |
|---|---|---|---|
| `coalescing` | (575, 980), (545, 1020), (605, 985), (515, 1020) | 115, 225, 254, 620 | 3.5 ha |
| `junction` | bisector at x = 540, 580, 620, 660 | 62, 222, 390, 559 | 11.6 ha |
| `parallel` | strip at x = 550, 600, 650, 700, 750 | 775, 725, 723, 723, 726 | 6.5 ha |

The reference wind over the burned cells is 5.00 m/s at the end of every run.
Along the junction's bisector the meeting point covers 40 m in 160 to 170 s
(0.24 m/s), the geometric R / sin(30 deg) with R the inner fronts' normal
rate, which for an edge at 60 degrees to a 5 m/s wind is about half the head
rate. The strip between the parallel lines closes from the flanks, the
slowest part of the ellipse, so it takes most of the run; the first meeting
cell (x = 550 m) is later because the lines' west ends are rounded.

The convective interaction between the fires is not in these one-way runs;
the regression suite `Exec/RegTests/FireMergingFronts` checks the same three
geometries at a prescribed rate in still air against their exact arrival
times.
