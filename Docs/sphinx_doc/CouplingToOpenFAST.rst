
 .. role:: cpp(code)
    :language: c++

 .. _CouplingToOpenFAST:

Coupling To OpenFAST
====================

ERF drives OpenFAST wind turbines as actuator disks or actuator lines through
the moving-bodies framework (``erf.moving_bodies.*``). Each step ERF samples
the wind at the turbine's nodes, hands it to OpenFAST through the
FAST_Library C interface (the ExtInfw inflow path), steps the turbine in
lockstep with its own time step, and spreads the returned loads into the flow
as a momentum source. The physics and the choices behind it are described in
:doc:`the theory chapter <theory/MovingBodies>`;
this page is the how-to.

Building
--------

Build OpenFAST 5.0 (4.x also works) with its C++ API, which produces the
shared library and ``FAST_Library.h``:

.. code-block:: bash

   cmake -DBUILD_OPENFAST_CPP_API=ON -DBUILD_SHARED_LIBS=ON -DDOUBLE_PRECISION=ON \
         -DCMAKE_INSTALL_PREFIX=$HOME/opt/openfast-5.0.0 ..
   make -j4 install

Then build ERF with the coupling. The moving-bodies framework is switched on
by ``ERF_ENABLE_OPENFAST``; the decks use the anelastic solver with the FFT
Poisson solve:

.. code-block:: bash

   cmake -DERF_ENABLE_MPI=ON -DERF_ENABLE_FFT=ON -DERF_ENABLE_OPENFAST=ON \
         -DOPENFAST_DIR=$HOME/opt/openfast-5.0.0 ..

At run time the OpenFAST library must be on the loader path
(``DYLD_LIBRARY_PATH`` on macOS, ``LD_LIBRARY_PATH`` on Linux). Without an
OpenFAST installation, ``-DERF_OPENFAST_USE_STUB=ON`` builds against a bundled
stub library with the same C API; the regression tests run on it, and so does
the ``Linux GCC OpenFAST stub`` CI job.

Floating-point traps must stay off in a coupled run
(``amrex.fpe_trap_invalid = 0``): OpenFAST's own arithmetic raises them.

Preparing the turbine model
---------------------------

The OpenFAST model is a standard set of files (``.fst``, ElastoDyn, AeroDyn,
ServoDyn or a fixed rotor speed). Four settings matter for the coupling:

- ``CompInflow = 2`` in the ``.fst``: the inflow comes from ERF.
- ``AirDens`` in the ``.fst`` equal to ERF's density at the hub; the start-up
  audit aborts when they differ by more than ``erf.moving_bodies.density_tolerance``.
- ``Wake_Mod`` in the AeroDyn file: ``0`` for ``sampling = disk`` (the resolved
  flow carries the induction), ``1`` for ``sampling = disk_corrected`` (the
  default) and ``sampling = upstream``, where OpenFAST applies its own
  induction to the free stream ERF hands it. The audit checks this.
- Tower shadow off (``TwrShadow = 0``) when tower force points are used: the
  tower's wake reaches the blades through the flow.

ERF's fixed time step must be a whole multiple of OpenFAST's ``DT``; the
coupling sub-steps OpenFAST accordingly.

A minimal case
--------------

A single IEA 15 MW rotor as an actuator disk in a uniform inflow on a
20 m mesh, the configuration calibrated against the standalone OpenFAST
solution:

.. code-block:: text

   erf.anelastic      = 1
   erf.anelastic_type = MidPoint     # the implicit vertical solve is honoured
   erf.vert_implicit  = true
   erf.use_fft        = true
   erf.fixed_dt       = 0.2          # a whole multiple of OpenFAST's DT
   amrex.fpe_trap_invalid = 0

   erf.moving_bodies.bodies                     = T1
   erf.moving_bodies.T1.type                    = openfast_turbine
   erf.moving_bodies.T1.mode                    = adm          # or alm
   erf.moving_bodies.T1.fst_file                = IEA15-land.fst
   erf.moving_bodies.T1.base_pos                = 750. 600. 0. # z above the terrain surface
   erf.moving_bodies.T1.num_force_points_blade  = 50
   erf.moving_bodies.T1.num_points_t            = 24
   erf.moving_bodies.T1.epsilon                 = 2.0          # kernel width in cells
   erf.moving_bodies.T1.air_density             = 1.225
   erf.moving_bodies.T1.output_root             = T1
   erf.moving_bodies.diagnostics_dir            = moving_bodies

The complete list of inputs, with their defaults, is in the
"Moving Bodies (OpenFAST turbines)" section of :doc:`Inputs <Inputs>`.
The canonical decks under ``Exec/CanonicalTests/MovingBodies`` are the
runnable reference for every feature (disk, line, tower, farm, wake lines,
turbulent inflow, two levels, terrain, restart).

Choosing the model and its settings
-----------------------------------

The settings below reproduced the standalone OpenFAST (BEM) loads of the
IEA 15 MW within the stated margins in a uniform 10.59 m/s inflow.

======================================================  ==================  ==================
Set-up                                                  thrust / BEM        power / BEM
======================================================  ==================  ==================
disk, ``disk_corrected`` (default), 20 m or 10 m cells  0.997 to 1.001      0.993 to 1.000
disk, ``upstream`` sampling two diameters ahead          0.993               0.999
disk, plain ``disk`` sampling, 20 m cells                1.093               1.328
line with FLLC, 2.5 m cells, kernel 2 m (0.8 cells)      0.989               0.983
line with FLLC, 10 m cells, kernel 2 cells               1.045               1.146
======================================================  ==================  ==================

- **Disk or line.** The corrected disk is the default and matches BEM on
  every grid tried; use it for farms and ABL studies. The actuator line needs
  cells of a few metres at the rotor.
- **Kernel width.** For the corrected disk keep the filter width
  :math:`\sqrt{6}\,\epsilon` within 1.25 rotor radii (a 1.5-cell kernel on
  40 m cells for a 120 m rotor); the start-up log warns otherwise. For the
  line the kernel is an absolute length set by the blade, about 2 m for the
  IEA 15 MW, not a cell count: two 2.5 m cells over-predicted power by 44 %.
- **FLLC** is on by default for the line (``fllc = false`` switches it off)
  and is the generalized variable-chord form.
- **Grids.** The rotor should span at least 8 cells (the audit notes fewer).
  On a run with refinement the bodies live on the anchor level, the finest
  by default, whose grids must cover every node with the kernel's reach.
- **Boundary conditions.** Inflow at ``xlo`` and outflow at ``xhi`` with a
  sounding or a precursor profile; a periodic box lets the wake feed the
  inflow. No sponge layer around a single turbine. CFL at or below 0.5.
- **Terrain.** On a terrain-fitted mesh ``base_pos`` z is the height above
  the surface at the base; the bodies are placed on the terrain at start-up.

Outputs
-------

Per turbine, under ``output_root``:

- ``<root>_erf.csv``: rotor speed, thrust along the shaft and along x,
  torque, power, tower and nacelle forces, total load, every step.
- ``<root>_flow.csv``: the velocities handed to OpenFAST at the hub and the
  blade mean.
- ``<root>_correction.csv`` (``disk_corrected``): disk velocity, inferred
  thrust coefficient, correction factor and recovered free stream.
- ``<root>_fllc.csv`` (line with FLLC): the correction's magnitude.
- ``<root>_wake_avg.csv`` (``erf.moving_bodies.wake.*``): running-averaged
  velocity along lateral and vertical lines downstream.
- ``<root>_stats.csv``: running statistics from ``avg_start`` on.
- ``<diagnostics_dir>/total_load.csv``, ``momentum_source.csv``,
  ``ground.csv``: farm totals, the integrated source (equal to minus the
  thrust), the terrain height under each body.

OpenFAST writes its own ``.out`` and summary files next to the ``.fst``.
Checkpoints hold the OpenFAST state and every running quantity, so a restart
continues bit for bit.

Tests
-----

.. code-block:: bash

   ctest -L moving-bodies            # the canonical cases (stub or real OpenFAST)
   ctest -L restart-parity -R OpenFAST
   Tests/Unit/erf_unit_tests --gtest_filter='OpenFAST*:Actuator*:FLLC*:WakeLines*:RunningStats*:PrescribedCtDisk.*:MovingBodiesInputs.*'

Known limits
------------

- The corrected disk's inferred thrust coefficient sits at its 0.96 clamp
  when the rotor spans only 6 cells; the recovered free stream is then a
  few percent low.
- A fine level whose boundary coincides with the inflow face gave an
  anomalous stable-ABL result; keep refinement patches away from the inflow.
- With a Dirichlet theta profile at the inflow, do not set ``<face>.density``
  in an anelastic run: the inflow theta is scaled by the density ratio.
- The k-equation length-scale cap from the PBL height reads each level's own
  diagnosis; pin ``erf.max_geom_lscale`` on multi-level ABL runs.

Release notes
-------------

The ERF-MovingBodies branch (2026-10) adds: the moving-bodies framework
(velocity sampling and force spreading on uniform, stretched and
terrain-fitted meshes), the OpenFAST turbine driver with lockstep stepping,
restart and a start-up audit of the model and geometry, actuator disk and
actuator line modes with tower and nacelle forces, the filtered lifting-line
correction, upstream and corrected-disk sampling, prescribed-Ct disks,
multi-turbine farms with round-robin ownership, wake lines and running
statistics, turbulent inflow from precursor planes or cell perturbations,
the anchor level on multi-level grids, bodies on terrain, eighteen canonical
cases with golds, box-parity and restart-parity tests, and the OpenFAST-stub
CI job.
