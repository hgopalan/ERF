
 .. role:: cpp(code)
    :language: c++

 .. _CouplingToMoorDyn:

Coupling To MoorDyn
===================

ERF couples to `MoorDyn-C <https://moordyn.readthedocs.io>`_ (the C++ MoorDyn,
version 2) to simulate flexible lines in the wind: conductors and shield
wires dead-ended at fixed attachment points, as single spans or as sections
hanging from insulator strings at suspension towers. MoorDyn
integrates the lumped-mass line dynamics; ERF supplies the fluid velocity at
MoorDyn's line nodes through its external wave-kinematics interface, so the
"water" MoorDyn's lines hang in is ERF's air. The actuator core of
:ref:`sec:ActuatorCore` provides the sampling of the wind at the nodes. This
page is the how-to for building ERF with MoorDyn and for what the coupling
layer in ``Source/MovingBodies/MoorDyn`` checks.

Building
--------

Build and install MoorDyn-C 2.7.1 (any 2.3 or newer release works) with the
script in the ERF tree, which clones the release tag, configures the C/C++
library only (no Python, MATLAB, Fortran or Rust wrappers, no tests, no docs,
the bundled Eigen) and installs it under ``$HOME/opt/moordyn-<version>``:

.. code-block:: bash

   Build/setup_moordyn.sh --version 2.7.1

Then build ERF with the coupling. ``ERF_ENABLE_MOORDYN`` switches on the
actuator core as well (``ERF_ENABLE_MOVING_BODIES``); ``MOORDYN_DIR`` is the
install prefix, where CMake finds MoorDyn's package configuration
(``lib/cmake/moordyn``) and the headers (``include/moordyn``):

.. code-block:: bash

   cmake -DERF_ENABLE_MPI=ON -DERF_ENABLE_FFT=ON -DERF_ENABLE_MOORDYN=ON \
         -DMOORDYN_DIR=$HOME/opt/moordyn-2.7.1 ..

The configure step records the MoorDyn version it found and refuses a MoorDyn
whose ``MoorDyn2.h`` does not declare the external wave-kinematics calls
(``MoorDyn_ExternalWaveKinInit``, ``MoorDyn_ExternalWaveKinGetCoordinates``,
``MoorDyn_ExternalWaveKinSet``), which replaced the version 1 names in 2.3.
The version is compiled into ERF and reported by
:cpp:`erf_moordyn::library_version()`; a run prints it at start-up with its
bodies.

MoorDyn is a shared library. ERF's executables carry the link path of
``libmoordyn`` in their run path, so no loader variable is needed when the
install stays where it was built; if the library is moved, set
``DYLD_LIBRARY_PATH`` (macOS) or ``LD_LIBRARY_PATH`` (Linux) to its ``lib``
directory.

Without a MoorDyn installation, ``-DERF_MOORDYN_USE_STUB=ON`` builds the
coupling against a bundled stub library with the same C API
(``Source/MovingBodies/MoorDyn/Stub``). The stub reads the same input file
and has the geometry, ordering and data flow of the real interface but no
line dynamics: each line hangs as an elastic parabola, as long as its
unstretched length stretched by its tension (slack or taut), swings to the
quasi-static blowout angle of the wind it is given with a one-second lag,
and carries that tension along its tangent. A coupled point moves as MoorDyn
moves it. A free point hangs from the shortest line that ties it to a fixed or
coupled point (an insulator string from its tower), plumb below it and swung
across the line by the wind across the other line attached to it (on that
line's weight, with the same lag), and that line runs straight down to it. Its saved state holds
each line's swing angles, the last wind it was given, which sets the direction
of the swing, and where the free and coupled points are, so a restored line is where
the saved one was. The
unit tests and the ``Linux GCC MoorDyn`` CI workflow runs them on it in one
job; a second job installs MoorDyn-C 2.7.1 with ``Build/setup_moordyn.sh``
and runs the same tests, the verification tests that need real line dynamics
and the coupled regression tests against the real library.

The GNU make build takes ``USE_MOORDYN = TRUE`` with ``MOORDYN_HOME`` set to
the install prefix (and ``MOORDYN_VERSION`` to the version string it should
report).

What the coupling layer does
----------------------------

:cpp:`erf_moordyn::MoorDynSystem` wraps one MoorDyn system (one input file)
behind the version 2 C API with ERF's error handling. Creating the system
and initialising it report their failure in a message that names the input
file, so that a body can abort with its own name; every later call aborts
with the MoorDyn function and the error code's name
(``MOORDYN_INVALID_INPUT``, ``MOORDYN_NAN_ERROR``, ...), since nothing
sensible can follow a failed line step. The wrapper exposes the coupled
degrees of freedom (none when every attachment is fixed), the external
kinematics (the number of points MoorDyn wants the fluid velocity at, their
current coordinates, and the velocity and acceleration to set there before
each step), the step itself (MoorDyn sub-steps internally with its own
``dtM`` within the step it is given), the line node positions, velocities,
tensions and net forces, the end and maximum tensions, the attachment points' positions and
forces, and MoorDyn's save and load of the whole system state for restarts.

MoorDyn-C is not clean under floating-point traps: its stationary
initial-condition solver overflows intermediate values, so a run with
``amrex.fpe_trap_overflow = 1`` ends with SIGILL inside ``MoorDyn_Init``
(2.7.1; the invalid and zero traps pass). As for OpenFAST, a body that runs
MoorDyn refuses to start with any ``amrex.fpe_trap_*`` input on;
:cpp:`erf_moordyn::fpe_traps_requested()` is the check.

MoorDyn reports the net force of the points it integrates (free and coupled
points) only; a fixed attachment reports zero, and its pull is the net force
MoorDyn finds on the end node of each line attached to it
(``MoorDyn_GetLineNodeForce``): the end segment's tension with the node's
share of the weight and the drag, which the fixed point holds still. A
coupled point is one the caller moves: each ``MoorDyn_Step`` takes its
position at the start of the step and a velocity, moves it linearly over the
step, and returns the net force of the attached lines' end nodes on it, the
same sum (without the end nodes' inertia). Moving towers use this: their
cross-arms are coupled points (:ref:`sec:Conductors`). A system's whole state
can also be kept in memory and restored (``MoorDyn_Serialize`` and
``MoorDyn_Deserialize``, the stub too), so that a coupled step can be redone:
the coupling with the moving towers iterates each step until the towers and the
lines agree.

Two properties of MoorDyn's input matter for lines in air:

- the ``OPTIONS`` must set ``WaveKin = 1`` (the fluid kinematics come through
  the API); the wrapper reports a system that takes no external kinematics;
- MoorDyn applies its fluid loads to nodes below ``z = 0`` only, its free
  surface, and the flat bottom lies at ``-WtrDpth``. Lines in air therefore
  live below ``z = 0`` in MoorDyn's frame, with ``WtrDnsty`` the air density
  and ``WtrDpth`` deep enough that the bottom is below the ground; the body
  that writes the input shifts ERF's heights accordingly.

A minimal case
--------------

A 300 m span of 795 kcmil ACSR (Drake) with 1.5 m of slack, 30 m above the
ground, across a uniform 15 m/s anelastic crosswind; the wind handed to
MoorDyn is ERF's velocity at the line's nodes (the regression test
``Conductors_FlowWind``):

.. code-block:: text

   erf.anelastic      = 1
   erf.anelastic_type = MidPoint
   erf.vert_implicit  = true
   erf.use_fft        = true
   erf.fixed_dt       = 0.5
   geometry.is_periodic = 1 0 0
   ylo.type     = "Inflow"
   ylo.velocity = 0. 15.0 0.
   yhi.type     = "Outflow"

   erf.conductors.lines               = S1
   erf.conductors.S1.end_a            = 600. 500. 30.    # z above the terrain surface
   erf.conductors.S1.end_b            = 900. 500. 30.
   erf.conductors.S1.length           = 301.5           # unstretched, more than the 300 m chord
   erf.conductors.S1.diameter         = 0.0281
   erf.conductors.S1.mass_per_length  = 1.628
   erf.conductors.S1.axial_stiffness  = 3.0e7
   erf.conductors.air_density         = 1.0

The physics and the diagnostics are in :ref:`sec:Conductors`; the inputs,
with their defaults and ranges, in the "Conductor lines" section of
:doc:`Inputs <Inputs>`.

The unit test ``MoorDynSystem`` (``Tests/Unit/MovingBodies``) runs a 300 m
fixed-fixed span of 795 kcmil ACSR with 1.5 m of slack in air against
whichever library the build links: the span initialises with no coupled
degree of freedom, its end nodes on the attachment points and the catenary
sag at mid-span; the external kinematics points start with the line's nodes;
a 20 m/s crosswind blows the span out towards the quasi-static angle
``atan(q / w)`` of the drag per unit length ``q`` over the weight per unit
length ``w`` and raises the end tension; and a saved state restored into a
fresh system created from the same input continues identically.
