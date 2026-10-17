# Scaling farm

A tool for timing the moving bodies as a farm grows, from a few turbines to a thousand or more. It is
not a regression test: nothing here compares output with a gold, and CTest does not run it.

- `make_scaling_farm.py` writes a farm of N stub OpenFAST turbines (N a square: 16, 64, 256, 1024, ...)
  on a flat domain with inflow and outflow in x, as actuator disks (`--mode adm`) or lines (`alm`).
  - Each turbine has its own `.fst` deck for the bundled stub, all naming one AeroDyn file. Start-up
    therefore reads N models, as a farm with one deck per turbine does.
  - The deck sets `erf.moving_bodies.timing_int`, so the run log carries the time of every phase.
  - The logs go to a few farm files (`diagnostics_format = farm`) however many turbines there are.
- `run_scaling.py` writes the decks for several N, runs one or two ERF binaries on each at several rank
  counts, and prints a markdown table:
  - the start-up time, from the first timing summary;
  - each phase's mean seconds per step over the ranks, from the later summaries;
  - with two binaries, whether their outputs (the farm logs) are identical.

  Each run waits until the 1-minute load average is below `--max-load`, so a shared machine is left free
  for other work first.

Build ERF with the stub (`-DERF_ENABLE_OPENFAST=ON -DERF_OPENFAST_USE_STUB=ON -DERF_ENABLE_FFT=ON`, MPI
on). Then, for example, to compare a branch with its base:

```bash
Exec/CanonicalTests/MovingBodies/ScalingFarm/run_scaling.py --exe base=/path/to/base/erf_exec --exe new=/path/to/branch/erf_exec --n 16 64 256 1024 --ranks 1 2 4 --mode adm alm --work scaling_runs
```

The defaults keep a 1024-turbine run to a minute or so on a laptop:
- rotors of 60 m radius, 350 m apart, on 50 m cells;
- 10 force points per blade and 8 points per ring;
- 4 steps of 0.5 s.

`make_scaling_farm.py --help` lists the options. A production farm uses about 50 points per blade and
16 per ring (ERF's defaults), which makes the spreading about 10 times as costly per turbine.
