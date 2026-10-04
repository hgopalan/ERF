# Power lines over hills in a turbulent wind (LES)

The power-line network of the parent directory (`../README.md`) in the time-varying wind of a large-eddy simulation, as ERF drives
turbines with turbulent inflow: a periodic precursor over flat land, then a run over hills fed by
its boundary planes. The hills are an immersed boundary (`erf.terrain_type = ImmersedForcing`) on
the precursor's flat mesh, so the precursor's checkpoint seeds the run; the conductors stand on the
hills' surface, read from the same terrain file. Each connection (two transformers joined by
the spanning tree) is a circuit, three phases on
insulator strings across the cross-arms and a shield wire on the peaks of one row of lattice towers
that bend, the lines moving with them as MoorDyn coupled points.

## Files

- `inputs_precursor`: the periodic neutral precursor over land (z0 = 0.1 m, u* ~ 1 m/s, about
  14 m/s at 30 m), 3072 x 1536 x 768 m on 16 m cells, Deardorff LES. 7200 s of spin-up, with
  boundary planes every 1.5 s from 7190 s (10 s before the checkpoint the lines run starts from)
  to 9000 s.
- `input_sounding`: its initial log law (surface pressure in hPa).
- `inputs_lines`: the run over the hills from the precursor's checkpoint at 7200 s, to 9000 s,
  with the inflow from the planes, pressure outflow, y periodic.

The network and the terrain come from `../make_case.py`:

    python3 ../make_case.py --out . --seed 1 --lx 3072 --ly 1536 --lz 768 --terrain_dx 16 \
        --hills 3 --hill_height 60 100 --hill_radius 120 170 --hill_spacing 500 --hill_margin 0.3 \
        --transformers 4 --on_hills 2 --transformer_spacing 400 --circuit

(it also writes a RANS inflow profile and sounding, which these runs do not use). The hills stay
0.3 of the width from the y faces, so that the periodic seam is flat.

## Running

Build with `-DERF_ENABLE_MOORDYN=ON` against MoorDyn-C, `-DERF_ENABLE_FFT=ON` and MPI. In a
directory `precursor` with `inputs_precursor` and `input_sounding`:

    mpiexec -n 4 erf_exec inputs_precursor

then in a sibling directory `lines` with `inputs_lines`, `input_sounding` and the generated
`network.inputs` and `terrain_hills.txt`:

    mpiexec -n 4 erf_exec inputs_lines

Every rank runs every MoorDyn line, so the lines' cost does not fall with more ranks: the 12 lines
add a fixed cost per step that more ranks do not reduce. The lines write their logs under `conductors/`
every step: the spans, the strings, `towers.dat` with every tower's loads and sway,
`transformers.dat`, `separation.dat`, and `coupling.dat` with the iterations the moving towers'
coupling took.
