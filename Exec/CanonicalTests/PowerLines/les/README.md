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

    python3 ../make_case.py --out . --network_only --seed 1 --lx 3072 --ly 1536 --lz 768 --terrain_dx 16 \
        --hills 3 --hill_height 60 100 --hill_radius 120 170 --hill_spacing 500 --hill_margin 0.3 \
        --transformers 4 --on_hills 2 --transformer_spacing 400 --circuit

(`--network_only` keeps this directory's `input_sounding`; without it the script would replace it
with the canonical case's RANS sounding). The hills stay 0.3 of the width from the y faces, so that
the periodic seam is flat.

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

## Comparing with ASCE 74

`compare_asce74.py` compares the spans of a run with the quasi-static wire load of ASCE Manual of
Practice 74 (`ERF_ASCE74.H`; the conductor theory's "ASCE 74 design check"). For conductors without
towers, keep one phase of each circuit in `network.inputs` (the `b` phases), drop the
`erf.conductors.tower_types` block and each line's `tower_type`, `share_towers` and insulator keys,
so that every line hangs clamped at fixed points where its towers stood, and give the run

    erf.conductors.asce74_wind     = 40.
    erf.conductors.asce74_exposure = C
    stop_time = 8100.0
    max_step  = 27100

so that ERF writes `conductors/asce74.csv` (each span's effective height, chord, length and weight with
the design check) and keeps 600 s of statistics from `stats_start` (7500 s), the window of the numbers
in the conductor theory.

Then, in the run's directory,

    python3 compare_asce74.py . --exposure C

For every span it takes the wind at mid-span normal to the span from the span's log, its mean and
its peak 3-second average V3 (the gust at the span's height), and compares the span's peak load per
metre with ASCE 74's (rho/2) Cf d V3^2 Gw, and its peak swing and tension with the quasi-static ones
under that load (the swing about the span's inclined chord, the upper end's tension, as in `asce74.csv`), over the samples from `erf.conductors.stats_start`. It writes
`asce74_comparison.csv` and `asce74_comparison.png`.

## The same lines in RANS gusts

`rans/` runs the clamped network of the ASCE 74 comparison in a k-equation RANS of the same hills, with the
gusts from its k (`erf.conductors.gust_type`; the conductor theory's "Gusts from the RANS turbulence"), so
that the gusts can be compared span by span with the LES. The mesh is terrain-fitted, of the LES's domain and
spacing, because the k-equation's wall distance is measured from the mesh's bottom, which an immersed terrain
leaves flat. In a directory with the decks of `rans/`, the clamped `network.inputs` and `terrain_hills.txt`:

    python3 make_rans_inflow.py ../precursor          # inflow_profile and input_sounding from the precursor's mean
    mpiexec -n 4 erf_exec inputs_spinup               # the RANS flow alone, 1500 s: chk03750
    mpiexec -n 4 erf_exec inputs_factor               # the lines without gusts in the wind, and gusts.csv
    mpiexec -n 4 erf_exec inputs_random1              # random gusts, one run per seed (1, 2 and 3)
    mpiexec -n 4 erf_exec inputs_event                # one travelling 1 - cos gust

Each lines run restarts from `chk03750` and logs every step into its own `conductors_<run>` directory, with
statistics over the 600 s from 1800 s, the ASCE 74 run's window length (the event's from 1600 s to its end at
2000 s). Then, with the clamped ASCE 74 run in `../lines`,

    python3 ../compare_gusts.py ../lines . --les_inputs inputs_lines \
        --sgs_profile ../precursor/mean_profiles.dat

compares every span with the LES run's (its mean and peak load per metre normal to the span, peak tension and
swing, and the wind's fluctuation at mid-span) and gives sigma_u / sqrt(k) the LES implies at each span, its
resolved and subgrid parts over the RANS k. It writes `gust_comparison.csv` and `gust_comparison.png`. To run
the gusts with another c, add `erf.conductors.gust_sigma_factor` to the lines decks in a directory of their own,
with the flow files (`flow.inputs`, `inflow_profile`, `input_sounding`, `network.inputs`, `terrain_hills.txt`),
`amr.restart` pointing at the spin-up's `chk03750`, and `inputs_event` only if it is run there.
