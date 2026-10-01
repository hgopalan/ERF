
 .. role:: cpp(code)
    :language: c++

 .. _CouplingToMoorDyn:

Coupling To MoorDyn
===================

ERF couples to `MoorDyn-C <https://moordyn.readthedocs.io>`_ (the C++ MoorDyn,
version 2) to simulate flexible lines in the wind: conductor spans, shield
wires and insulator strings hanging between fixed attachment points. MoorDyn
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
line dynamics: each line hangs as a parabola with the catenary sag of its
slack, swings to the quasi-static blowout angle of the wind it is given with
a one-second lag, and carries the catenary tension. The unit tests and the
``Linux GCC MoorDyn stub`` CI job run on it; the regression tests of the
coupled bodies will too.

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
``dtM`` within the step it is given), the line node positions, velocities and
tensions, the end and maximum tensions, the attachment points' positions and
forces, and MoorDyn's save and load of the whole system state for restarts.

MoorDyn-C is not clean under floating-point traps: its stationary
initial-condition solver overflows intermediate values, so a run with
``amrex.fpe_trap_overflow = 1`` ends with SIGILL inside ``MoorDyn_Init``
(2.7.1; the invalid and zero traps pass). As for OpenFAST, a body that runs
MoorDyn refuses to start with any ``amrex.fpe_trap_*`` input on;
:cpp:`erf_moordyn::fpe_traps_requested()` is the check.

MoorDyn reports the net force of the points it integrates (free and coupled
points) only; a fixed attachment reports zero, and its pull is the tension
vector at the end node of each line attached to it.

Two properties of MoorDyn's input matter for lines in air:

- the ``OPTIONS`` must set ``WaveKin = 1`` (the fluid kinematics come through
  the API); the wrapper reports a system that takes no external kinematics;
- MoorDyn applies its fluid loads to nodes below ``z = 0`` only, its free
  surface, and the flat bottom lies at ``-WtrDpth``. Lines in air therefore
  live below ``z = 0`` in MoorDyn's frame, with ``WtrDnsty`` the air density
  and ``WtrDpth`` deep enough that the bottom is below the ground; the body
  that writes the input shifts ERF's heights accordingly.

The unit test ``MoorDynSystem`` (``Tests/Unit/MovingBodies``) runs a 300 m
fixed-fixed span of 795 kcmil ACSR with 1.5 m of slack in air against
whichever library the build links: the span initialises with no coupled
degree of freedom, its end nodes on the attachment points and the catenary
sag at mid-span; the external kinematics points start with the line's nodes;
a 20 m/s crosswind blows the span out towards the quasi-static angle
``atan(q / w)`` of the drag per unit length ``q`` over the weight per unit
length ``w`` and raises the end tension; and a saved state restored into a
fresh system created from the same input continues identically.
