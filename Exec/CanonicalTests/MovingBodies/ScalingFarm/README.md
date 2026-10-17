# Scaling farm

A tool for timing the moving bodies as a farm grows, from a few turbines to a thousand or more. It is
not a regression test: nothing here compares output with a gold, and CTest does not run it.

- `make_scaling_farm.py` writes a farm of N bodies (N a square: 16, 64, 256, 1024, ...) on a flat
  domain with inflow and outflow in x:
  - stub OpenFAST turbines as actuator disks (`--mode adm`) or lines (`alm`), each with its own `.fst`
    deck for the bundled stub, all naming one AeroDyn file, so start-up reads N models as a farm with
    one deck per turbine does;
  - or prescribed-Ct disks (`--bodies ct_disk`, no OpenFAST);
  - optionally with wake lines behind every rotor (`--wake-lines-xD`);
  - optionally on a stretched mesh (`--stretched`), so the bodies see `z_phys_nd` as on terrain and the
    start-up checks take their terrain paths.

  The deck sets `erf.moving_bodies.timing_int`, so the run log carries the time of every phase, and writes
  the logs to a few farm files (`diagnostics_format = farm`) however many bodies there are.
- `run_scaling.py` writes the decks for several N, runs one or two ERF binaries on each at several rank
  counts, and prints a markdown table:
  - from the first timing summary, as the maximum over the ranks: the start-up (the constructor,
    `set_ground`), and the first step's start-up checks (coverage, audits) and wake-line set-up;
  - each phase's mean seconds per step over the ranks, from the later summaries;
  - a status column, so a failed run shows its exit code instead of numbers;
  - with two binaries, whether their farm logs (everything under `moving_bodies/`) are identical.

  The logs are printed to 10 digits, so identical logs show the answers agree to that precision, not bit for
  bit. The two binaries take turns to run first. Each run waits until the 1-minute load average is below
  `--max-load`, so a shared machine is left free for other work first; the load can still rise during a
  run, so repeat a sweep before reading much into small differences.

Build ERF with the stub (`-DERF_ENABLE_OPENFAST=ON -DERF_OPENFAST_USE_STUB=ON -DERF_ENABLE_FFT=ON`, MPI
on). Then, for example, to compare a branch with its base:

```bash
Exec/CanonicalTests/MovingBodies/ScalingFarm/run_scaling.py --exe base=/path/to/base/erf_exec --exe new=/path/to/branch/erf_exec --n 16 64 256 1024 --ranks 1 2 4 --mode adm alm --work scaling_runs
```

or, for disks with wake lines on a stretched mesh:

```bash
Exec/CanonicalTests/MovingBodies/ScalingFarm/run_scaling.py --exe base=/path/to/base/erf_exec --exe new=/path/to/branch/erf_exec --farm-args "--bodies ct_disk --wake-lines-xD 1 2 --stretched" --work scaling_disks
```

The defaults are small, so that a sweep runs on a laptop:
- rotors of 60 m radius, 350 m apart, on 50 m cells;
- 10 force points per blade and 8 points per ring;
- 4 steps of 0.5 s.

`make_scaling_farm.py --help` lists the options. Two differences from a production farm matter when reading
the numbers:
- A production rotor has about 50 points per blade and 16 per ring (ERF's defaults), which makes the
  spreading about 10 times as costly per turbine.
- The kernel width is 2 cells, so a kernel reaches 3 x 100 m + 50 m = 350 m, the spacing of the rotors:
  every kernel reaches its neighbours' boxes. That inflates the spreading and the number of ranks each body
  exchanges with, compared with a farm of rotors 5 to 10 diameters apart.
