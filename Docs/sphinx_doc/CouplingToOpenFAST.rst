
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

The CMake build records the OpenFAST library directory in ``erf_exec``'s run-time
search path; a moved installation or another build system needs it on the loader
path (``DYLD_LIBRARY_PATH`` on macOS, ``LD_LIBRARY_PATH`` on Linux). Without an
OpenFAST installation, ``-DERF_OPENFAST_USE_STUB=ON`` builds against a bundled
stub library with the same C API; most regression tests run on it, and so does
the ``Linux GCC OpenFAST stub`` CI job. Two cases run the real library on the
IEA 15 MW reference turbine (below); the ``Linux GCC OpenFAST 5`` CI job builds
OpenFAST 5.0.0 with ``Build/setup_openfast.sh`` (cached between runs) for them.

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
  whole at each write (every ``stats_int`` steps, at checkpoints and at the
  end, when ``stats_int`` is set).
- ``<diagnostics_dir>/total_load.csv``, ``momentum_source.csv``,
  ``ground.csv``: farm totals, the integrated source (equal to minus the
  loads; for an actuator disk on a tilted shaft, minus the thrust's shaft
  component (T . n) n, since its rings cancel the in-plane load), the terrain
  height under each body.

The rows of ``_erf.csv``, ``_flow.csv`` and ``total_load.csv`` carry the time
at the end of their step; the others carry the time at the start of the step
whose forcing they describe. The full list, with the prescribed-Ct disk's
file, is under the input table in :doc:`Inputs <Inputs>`. OpenFAST writes its
own ``.out`` and summary files next to the ``.fst``.

``erf.moving_bodies.diagnostics_format`` gathers the per-turbine time series
and statistics into one file per rank (``per_rank``) or one per log for the
whole farm (``farm``, written by the I/O rank), with the body's name as a
column; the default ``per_turbine`` writes the files above. In every format
each file is opened once and kept open, and its rows are flushed at every
checkpoint and at the end of the run.

Running on many ranks
---------------------

Each turbine is owned by one rank, which alone runs its OpenFAST model and
writes its logs: a rank whose boxes it stands on, unless that would give one
rank more turbines than another or mix two OpenFAST time steps, in which case a
nearby rank with fewer turbines takes it. Its points go only to the ranks whose
boxes they touch, so a rank far from every turbine samples and spreads nothing
for it (it still joins two integer reductions per step). The run log names each
turbine's and disk's owner and the ranks it, and the wake lines behind it,
exchange with. The turbines a rank owns are
stepped in turn, so they must use the same OpenFAST ``DT`` (turbines of
different ``DT`` get different ranks whenever there are at least as many ranks
as distinct ``DT``), and a rank that owns several turbines steps their models
one after the other: run a farm on at least as many ranks as it has turbines. A check on a turbine's points stops
the run from its owner's rank, whose message is the one to read. Each
prescribed-Ct disk has an owner too, chosen by the same rule among the disks
alone (a disk runs no model, so it may share a rank with a turbine), which
samples the flow for it, sets its thrust and writes its log and statistics.
The wake lines behind a rotor belong to the rotor's owner, which samples them,
keeps their running average and writes their files; sampling them adds one
integer reduction at each step that writes them.

At start-up each distinct ``.fst`` (a farm usually shares a few models, or
gives each turbine its own) is read by one rank only, the distinct files
spread over the ranks, each of its files once: the ``Wake_Mod`` and
``TwrShadow`` checks and the rotor's radius and time step. One gather hands
every rank the results, so a check fails on every rank alike, with the same
message. A turbine's owner reads its model once more, each file once, for the
start-up audits. At each diagnostics step the
farm's totals (``total_load.csv``) and, in the ``farm`` format, every body's
rows reach the I/O rank in one reduction.

For large farms (hundreds of turbines and more):

- Write the diagnostics less often than every step: ``diagnostics_int`` of 10
  to 100 steps (a few seconds of simulated time) resolves the rotor dynamics
  the logs are used for, and OpenFAST's own ``.out`` files keep the full rate.
  Set ``avg_start`` so the statistics, rewritten at each write (or every
  ``stats_int`` steps), start after the spin-up.
- Sample wake lines rarely: their exchange costs a collective every
  ``wake.int`` steps. Use a multiple of ``diagnostics_int``, 50 to 200 steps
  for averaged wake profiles, and only behind the turbines whose wakes are
  studied.
- Use ``diagnostics_format = farm`` (or ``per_rank`` when the farm file's
  gather to one rank is too large): a few files in place of several per
  turbine.
- Rewrite the summary files rarely: each statistics file (``_stats``) and
  each wake-average file (``_wake_avg``) is written afresh in full, by
  default at every diagnostics step and every wake sample. ``stats_int`` of
  a few hundred steps rewrites them that often, and also at every checkpoint
  and at the end of the run, so they hold every sample at those points.
- Each rank keeps its logs open from one row to the next, at most
  ``max_open_logs`` of them (by default a quarter of the process's limit on
  open files). An owner of more logs than that finds about
  ``max_open_logs - 1`` of them still open at their next row and closes and
  opens the others again, which is slower but writes the same files; raise the limit
  (``ulimit -n``) or use ``per_rank`` or ``farm`` for many bodies per rank.
- Measure before tuning: ``timing_int`` prints the time of each phase (the
  sampling, OpenFAST, the spreading, the diagnostics, the wake lines) over
  the ranks every few steps, and a build with
  ``-DERF_ENABLE_TINY_PROFILE=ON`` breaks it down further, down to each
  message round of the exchanges (see the theory section on timing).

Restarting
----------

Checkpoints hold the OpenFAST state and every running quantity, so a restart
continues bit for bit. It must continue the same bodies with the same inputs,
the same ``erf.fixed_dt`` and a stop time no later than the one the turbines
were started with (OpenFAST keeps that stop time in its own checkpoint); the
run aborts otherwise, naming the input. The diagnostics files drop any rows
written after the checkpoint (by a run that went on past it) and continue from
there (in the ``per_rank`` format, the files of every rank of either run; a
restart in another ``diagnostics_format`` starts that format's files and leaves
the old ones as they are). To run on past the first run's stop time, start the turbines again:
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
   Tests/Unit/erf_unit_tests --gtest_filter='OpenFAST*:Actuator*:BodyExchange.*:FLLC*:WakeLines*:RunningStats*:PrescribedCtDisk.*:MovingBodiesInputs.*'
   ctest -L parallel -R "Actuator|BodyExchange"   # the same exchanges on 1, 2 and 4 ranks

These CTests run on the stub library (``-DERF_OPENFAST_USE_STUB=ON``); their
golds hold the stub's loads. With a real OpenFAST the same decks run, but the
loads, and so the golds, differ. A build against the real library
(``-DERF_OPENFAST_USE_STUB=OFF``) registers two other cases instead, label
``openfast-real``: the IEA 15 MW reference turbine (the model is in
``Exec/CanonicalTests/MovingBodies/IEA15_land_model``) on the Askervein hill as
an actuator disk and as an actuator line with its tower and nacelle:

.. code-block:: bash

   ctest -L openfast-real

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
- On a terrain-fitted mesh, refined levels are not usable yet for turbines. In
  the Askervein runs that led to the single-level real-OpenFAST cases, the
  anelastic projection (an FFT-preconditioned GMRES solve) left a maximum
  divergence of 1e-10 to 1e-9 on level 0, but on level 1 and finer 1e3 to 1e7
  times the constant a refined level keeps by design (the mean it subtracts
  from its right-hand side, printed as "Subtracting ... from rhs" with
  ``erf.mg_v = 1``); a three-level actuator-line run blew up in its first step.
  ERF prints no warning when this happens: compare the "divergence after" lines
  with that constant. A refined level on a terrain-fitted mesh must also span
  the full height, or the FFT preconditioner aborts.

Release notes
-------------

The ERF-MovingBodies branch (2026-10) adds: the moving-bodies framework
(velocity sampling and force spreading on uniform, stretched and
terrain-fitted meshes), the OpenFAST turbine driver with lockstep stepping,
restart and a start-up audit of the model and geometry, actuator disk and
actuator line modes with tower and nacelle forces, the filtered lifting-line
correction, upstream and corrected-disk sampling, prescribed-Ct disks,
multi-turbine farms whose turbines and disks are owned by ranks they stand on where the
balance allows and whose points (and wake lines) go only to the ranks they reach, wake lines and running
statistics, turbulent inflow from precursor planes or cell perturbations,
the anchor level on multi-level grids, bodies on terrain, the canonical
cases with golds, box-parity and restart-parity tests, the OpenFAST-stub CI
job, and two cases with the real OpenFAST library and the IEA 15 MW reference
turbine on the Askervein hill, with their own CI job.
