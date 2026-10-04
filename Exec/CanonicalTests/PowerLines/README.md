# Power lines over hills

Conductor lines (MoorDyn-C) in a neutral boundary layer over several hills, dead-ended on
transformers that stand on hilltops and on flat ground. The flow is the k-equation RANS
(Reynolds-averaged) closure on
a terrain-following mesh with the implicit anelastic MidPoint scheme and the FFT pressure solve;
a log law under a capping inversion comes in through the x-low face and leaves through the x-high
one. Each line is a section on insulator strings over lattice suspension towers, strung to one
horizontal tension; the towers carry the wind's drag on their members and the lines' pull, their
footings are checked for uplift and compression, and they bend under those loads in their first
mode, MoorDyn moving the cross-arms the lines hang from; the transformers take the lines' pull and are checked against an allowable horizontal
force and overturning moment, and every conductor is watched for how close it comes to each box.
See `Docs/sphinx_doc/theory/Conductors.rst` ("Conductor lines in the wind").

## Files

- `make_case.py` writes the terrain (`terrain_hills.txt`), the inflow profile and sounding
  (`inflow_profile`, `input_sounding`) and the network (`network.inputs`) from one seed:
  `python3 make_case.py --seed 2026` gives the committed files. Its options set the domain, the
  hills, the number of transformers and how many stand on hilltops, the conductor's stringing
  tension, the towers' height, spacing, lattice, foundation and sway (`--tower_sway 0 0 0` for rigid
  towers), and the inflow speed. A suspension tower in a dip, where the spans either side would pull
  it up, is raised until the line weighs on it (`--min_weight_span`, at most `--max_tower_height`).
- `flow.inputs` holds the flow, shared by the two stages.
- `inputs_spinup` runs the flow alone for 600 s to a checkpoint.
- `inputs_lines` restarts from it with the lines (the checkpoint holds no conductor state, so the
  lines start in their still-air shape) and runs 120 s more.

## Running

Build with `-DERF_ENABLE_MOORDYN=ON` against MoorDyn-C (`Build/setup_moordyn.sh`),
`-DERF_ENABLE_FFT=ON` and MPI, then

    mpiexec -n 4 erf_exec inputs_spinup
    mpiexec -n 4 erf_exec inputs_lines

The mesh has 1.2 million cells. The lines write their logs under `conductors/`: a log per span
and per set of strings, `transformers.dat` with every transformer's load, flags and clearance,
`towers.dat` with every tower's drag, line pull, foundation loads and cross-arm displacement,
`separation.dat` with the closest approach of every pair of lines, `ground.dat` with where every
attachment and transformer stands, `coupling.dat` with the iterations each step's tower coupling
took, `total_load.dat` with the air's drag on all lines and towers, and the running statistics of all of them from
`stats_start` on.

## The same lines in a turbulent wind

`les/` runs a circuit network over hills in a large-eddy simulation: a periodic precursor over flat
land, then the hills as an immersed boundary fed by the precursor's boundary planes. See
`les/README.md`.
