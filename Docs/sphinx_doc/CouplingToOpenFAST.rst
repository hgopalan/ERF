
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
shared library and ``FAST_Library.h``. ``Build/setup_openfast.sh`` does this
in one command (``--force`` rebuilds an existing install); by hand:

.. code-block:: bash

   cmake -DBUILD_OPENFAST_CPP_API=ON -DBUILD_SHARED_LIBS=ON -DDOUBLE_PRECISION=ON \
         -DCMAKE_INSTALL_PREFIX=$HOME/opt/openfast-5.0.0 ..
   make -j4 install

Then build ERF with the coupling. The moving-bodies framework is switched on
by ``ERF_ENABLE_OPENFAST``; prescribed-Ct disks alone need only
``ERF_ENABLE_MOVING_BODIES`` and no OpenFAST. The decks use the anelastic
solver with the FFT Poisson solve:

.. code-block:: bash

   cmake -DERF_ENABLE_MPI=ON -DERF_ENABLE_FFT=ON -DERF_ENABLE_OPENFAST=ON \
         -DOPENFAST_DIR=$HOME/opt/openfast-5.0.0 ..

At run time the OpenFAST library must be on the loader path
(``DYLD_LIBRARY_PATH`` on macOS, ``LD_LIBRARY_PATH`` on Linux). Without an
OpenFAST installation, ``-DERF_OPENFAST_USE_STUB=ON`` builds against a bundled
stub library with the same C API; the regression tests run on it, and so does
the ``Linux GCC OpenFAST stub`` CI job.

Floating-point traps must stay off in a run with an OpenFAST turbine
(``amrex.fpe_trap_invalid``, ``amrex.fpe_trap_zero`` and
``amrex.fpe_trap_overflow`` all 0): OpenFAST's own arithmetic raises them.

Preparing the turbine model
---------------------------

The OpenFAST model is a standard set of files (``.fst``, ElastoDyn, AeroDyn,
ServoDyn or a fixed rotor speed). Four settings matter for the coupling:

- ``CompInflow = 2`` in the ``.fst``: the inflow comes from ERF.
- ``AirDens`` in the ``.fst`` equal to ERF's density at the hub; the start-up
  audit aborts when they differ by more than ``erf.moving_bodies.density_tolerance``.
- ``Wake_Mod`` in the AeroDyn file: ``0`` for ``sampling = disk``, which is
  also what the actuator line uses (the resolved flow carries the induction),
  non-zero (``1``, BEMT) for ``sampling = disk_corrected`` (the disk's
  default) and ``sampling = upstream``, where OpenFAST applies its own
  induction to the free stream ERF hands it. The start-up check aborts on the wrong pairing, so one
  model file serves either the disk or the line, not both.
- Tower shadow off (``TwrShadow = 0``) when tower force points are used: the
  tower's wake reaches the blades through the flow.

ERF's fixed time step must be a whole multiple of OpenFAST's ``DT``; the
coupling sub-steps OpenFAST accordingly.

A minimal case
--------------

A single IEA 15 MW rotor as an actuator disk in a uniform inflow on a
20 m mesh, the configuration calibrated against the standalone OpenFAST
solution. Its AeroDyn file sets ``Wake_Mod = 1``, as the default
``sampling = disk_corrected`` needs:

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
   erf.moving_bodies.T1.output_root             = T1
   erf.moving_bodies.diagnostics_dir            = moving_bodies

The complete list of inputs, with their defaults, is in the
"Moving Bodies (OpenFAST turbines)" section of :doc:`Inputs <Inputs>`.
The canonical decks under ``Exec/CanonicalTests/MovingBodies`` are the
runnable reference for every feature (disk, line, tower, farm, wake lines,
turbulent inflow, two levels, terrain, restart).

Choosing the model and its settings
-----------------------------------

The table below shows how close each set-up came to the standalone OpenFAST
(BEM) loads of the IEA 15 MW in a uniform 10.59 m/s inflow.

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
  every grid tried; use it for farms and RANS or ABL studies. The actuator
  line, with the FLLC, is the choice for LES that resolves the blades; it needs
  cells of a few metres at the rotor.
- **The corrected disk's update.** The recovered free stream is updated each
  step from the last one, through the induction the thrust implies. Above a
  thrust coefficient of about 0.75 that update overshoots and grows step by
  step; and the flow answers a change of thrust only after a delay, which can
  make the update ring at a period of seconds. By default it is therefore
  under-relaxed by a factor taken from its own loop gain, and limited so that
  an error decays on the time scale of the rotor radius over the free stream
  (``correction_relax = -1``, ``correction_time = -1``); the relaxation changes
  how the free stream is reached, not where it settles, at the cost of a
  start-up of tens of seconds instead of a few steps. ``correction_relax = 1`` gives the unrelaxed update
  for comparison. The theory chapter has the derivation.
- **Kernel width.** For the corrected disk keep the filter width
  :math:`\sqrt{6}\,\epsilon` within 1.25 rotor radii (a 1.5-cell kernel on
  40 m cells for a 120 m rotor); the start-up log warns otherwise. For the
  line the kernel is an absolute length set by the blade, about 2 m for the
  IEA 15 MW, not a cell count: a kernel of two 2.5 m cells (5 m)
  over-predicted power by 44 %. With the FLLC the kernel may be narrower than a
  cell: the spreading still puts exactly the force into the flow, but below
  about 0.75 cells the kernel's shape depends on where the point sits in its
  cell (see the theory chapter).
- **FLLC** is on by default for the line (``fllc = false`` switches it off)
  and is the generalized variable-chord form.
- **Grids.** The rotor should span at least 8 cells (the audit notes fewer).
  On a run with refinement the bodies live on the anchor level
  (``amr.max_level`` by default), whose grids must cover every node with the
  kernel's reach, and no finer level may cover them.
- **Boundary conditions.** Inflow at ``xlo`` and outflow at ``xhi`` with a
  sounding or a precursor profile; a periodic box lets the wake feed the
  inflow. No sponge layer around a single turbine. CFL at or below 0.5.
- **Terrain.** On a terrain-fitted mesh ``base_pos`` z is the height above
  the surface at the base; the bodies are placed on the terrain at start-up.

Outputs
-------

Per turbine, under ``output_root``, every ``diagnostics_int`` steps:

- ``<root>_erf.csv``: rotor speed (signed like the torque about the shaft),
  thrust vector, torque, power, hub axis, tower and nacelle forces and total
  load.
- ``<root>_flow.csv``: the velocities handed to OpenFAST at the hub and the
  blade mean, after the sampling correction and the FLLC.
- ``<root>_correction.csv`` (``disk_corrected``): disk velocity, inferred
  thrust coefficients, induction, correction factor, recovered free stream,
  the update's loop gain and the relaxation used.
- ``<root>_fllc.csv`` (line with FLLC): the correction's magnitude.
- ``<root>_wake.csv`` and ``<root>_wake_avg.csv`` (``erf.moving_bodies.wake.*``):
  instantaneous and running-averaged velocity along lateral and vertical lines
  downstream.
- ``<root>_stats.csv``: running statistics from ``avg_start`` on, rewritten
  whole at each write.
- ``<diagnostics_dir>/total_load.csv``, ``momentum_source.csv``,
  ``ground.csv``: farm totals, the integrated source (equal to minus the
  loads), the terrain height under each body.

The rows of ``_erf.csv``, ``_flow.csv`` and ``total_load.csv`` carry the time
at the end of their step; the others carry the time at the start of the step
whose forcing they describe. The full list, with the prescribed-Ct disk's
file, is under the input table in :doc:`Inputs <Inputs>`. OpenFAST writes its
own ``.out`` and summary files next to the ``.fst``.

Restarting
----------

Checkpoints hold the OpenFAST state and every running quantity, so a restart
continues bit for bit. It must continue the same bodies with the same inputs,
the same ``erf.fixed_dt`` and a stop time no later than the one the turbines
were started with (OpenFAST keeps that stop time in its own checkpoint); the
run aborts otherwise, naming the input. The diagnostics files drop any rows
written after the checkpoint (by a run that went on past it) and continue from
there. To run on past the first run's stop time, start the turbines again:
remove the ``moving_bodies`` directory from a copy of the checkpoint and restart
from that copy, which starts OpenFAST afresh at the checkpoint's time (and also
starts the wake averages, the statistics and the FLLC afresh). A run with
prescribed-Ct disks only may simply restart with a later stop time. A run started from a precursor checkpoint written
without bodies starts the turbines at the checkpoint's time; a run that stops
at ``stop_datetime`` hands OpenFAST the seconds to the stop date.

Tests
-----

.. code-block:: bash

   ctest -L moving-bodies            # the canonical cases and the start-up aborts
   ctest -L restart-parity -R "OpenFAST|MovingBodies"
   ctest -L box-parity -R "OpenFAST|Actuator"
   Tests/Unit/erf_unit_tests --gtest_filter='OpenFAST*:Actuator*:FLLC*:WakeLines*:RunningStats*:PrescribedCtDisk.*:MovingBodiesInputs.*'

The CTests run on the stub library only (``-DERF_OPENFAST_USE_STUB=ON``);
their golds hold the stub's loads. With a real OpenFAST the same decks run, but
the loads, and so the golds, differ.

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
