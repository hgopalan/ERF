# Moving-bodies cases

Regression cases for the moving-bodies framework (`erf.moving_bodies.*`:
OpenFAST turbines as actuator disks and lines, prescribed-Ct disks), laid out
like `../Canonical_RANS`: one directory per case with the input deck, the
bundled OpenFAST stub's turbine file where a turbine is involved, and the
gold logs the case is compared with. They are registered in
`Tests/CTestList.cmake` (`TEST_SOURCE_ROOT` points the registration here) and
their gold plotfiles stay in `Tests/ERFGoldFiles/<case>`. The theory and the
inputs are in `Docs/sphinx_doc/theory/MovingBodies.rst` and
`Docs/sphinx_doc/Inputs.rst`; what each test checks is in
`Docs/sphinx_doc/RegressionTests.rst`.

Every case builds with `ERF_ENABLE_OPENFAST=ON`, `ERF_OPENFAST_USE_STUB=ON`,
`ERF_ENABLE_FFT=ON` and MPI (`Actuator_UniformCtDisk` needs only
`ERF_ENABLE_MOVING_BODIES=ON`), runs the anelastic solver on a small uniform
box for ten steps or so, and is checked by the runner named below.

| case | body | runner | what it checks |
| --- | --- | --- | --- |
| `OpenFAST_DriverOnly` | stub turbine, `mode = none` | `RunOpenFASTADM.cmake` | the flow is unchanged (the gold is the deck run without bodies); the turbine log matches its gold; no momentum source is written |
| `Actuator_Sampling` | stub turbine, `mode = none` | `RunActuatorSampling.cmake` | sampled velocities at the nodes against the analytic shear |
| `Actuator_UniformCtDisk` | prescribed-Ct disk | `RunCtDisk.cmake`, `RunRestartParity.cmake` | plotfile and disk log against their golds; the disk's force integrates back exactly; restart parity with a later stop time; a changed `air_density` stops a restart |
| `OpenFAST_ADM_Uniform` | stub turbine, disk | `RunOpenFASTADM.cmake` | plotfile, sampled flow log, `fx == -thrust_x` |
| `OpenFAST_ADM_Upstream` | stub turbine, disk, velocities sampled 1 D upstream | `RunOpenFASTADM.cmake` | plotfile, sampled flow log, `fx == -thrust_x` |
| `OpenFAST_ADM_Terrain` | stub turbine, disk, on a Witch-of-Agnesi ridge (fitted mesh) | `RunOpenFASTADM.cmake` | ground heights log, plotfile, `fx == -thrust_x` |
| `OpenFAST_ADM_DiskCorrected` | stub turbine, disk, free stream recovered with the filtered-disk factor | `RunOpenFASTADM.cmake`, `RunRestartParity.cmake` | plotfile, flow log, `fx == -thrust_x`; restart parity of every log; with `correction_relax = 1` (`OpenFAST_ADM_DiskCorrected_OldUpdate`) the gold of the unrelaxed update |
| `OpenFAST_ADM_Wake` | stub turbine, disk, wake lines | `RunOpenFASTADM.cmake` | wake running average |
| `OpenFAST_ADM_Restart` | stub turbine, disk | `RunRestartParity.cmake` | restart parity of the plotfile and every log, with rows written past the checkpoint dropped; the restart aborts (changed body key, step, later stop time, added body) |
| `OpenFAST_ADM_LES` | stub turbine, disk, precursor planes | `RunPrecursorInflow.cmake` | turbulent inflow from a precursor; body statistics |
| `OpenFAST_ADM_CPM` | stub turbine, disk, cell perturbations | `RunOpenFASTADM.cmake` | turbulent inflow from the perturbation method; statistics |
| `OpenFAST_ADM_TwoTurbines` | two stub turbines, disks | `RunOpenFASTADM.cmake` | a farm: `fx == -load_x` of `total_load.csv`; ownership parity |
| `OpenFAST_ADM_TwoLevel` | stub turbine, disk, on level 1 of two | `RunOpenFASTADM.cmake` | the anchor level: sampled and forced on the fine patch, stepped with its step |
| `OpenFAST_ADM_TwoLevel_Restart` | as above | `RunRestartParity.cmake` | restart parity with the fine level, every log |
| `OpenFAST_ALM_Uniform` | stub turbine, line | `RunOpenFASTADM.cmake` | the rotating line; tip-travel limit |
| `OpenFAST_ALM_Tower` | stub turbine, line, tower, nacelle | `RunOpenFASTADM.cmake` | `fx == -load_x` with the tower and nacelle |
| `OpenFAST_ALM_FLLC` | stub turbine, line, lifting-line correction | `RunOpenFASTADM.cmake` | the correction log |
| `OpenFAST_ALM_FLLC_Restart` | as above | `RunRestartParity.cmake` | restart parity of the correction and every log |

Each case also has a box-parity twin registered next to it (one box on one
rank against a split domain on several ranks) where the feature adds a
decomposition-sensitive step.

The start-up checks have one abort test each (`add_test_abort`, label
`moving-bodies`), run on the decks above with one input changed: an anchor
level above the finest or missing at the start, a stop time that is not a
whole number of steps (of the single level, and of level 0 of two), a
prescribed-Ct disk's `air_density` off ERF's density, a finer level over the
bodies, floating-point traps with a turbine, an out-of-range
`correction_relax`, and an actuator-line tip that sweeps more cells per step
than `alm_max_tip_cells` allows. `MovingBodies_StopDatetime` runs `OpenFAST_ADM_Restart` to
a `stop_datetime` past the deck's `stop_time` and requires it to get there.

The decks that do not set `sampling` run the default `disk_corrected`; on the
50 m decks its filter width is about two rotor radii, past the range the factor
was fitted for, which is fine for a regression of the feature each deck tests.
`OpenFAST_ADM_DiskCorrected` is the set-up inside that range.

## Rules

- The stub turbine file (`stub_turbine.fst`) is the bundled stub's `key = value`
  deck, not an OpenFAST model; the start-up audit skips the model-file checks
  for it and says so.
- A gold log is the run's own output copied from a two-rank run; regenerate
  it only when the physics changes, and say why in the pull request.
- New cases go here with a row in this table and an entry in
  `Docs/sphinx_doc/RegressionTests.rst`.
