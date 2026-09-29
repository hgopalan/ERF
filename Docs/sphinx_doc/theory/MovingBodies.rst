.. _sec:MovingBodies:

Moving Bodies: OpenFAST Turbines
================================

The moving-bodies framework (``Source/MovingBodies``) represents wind turbines
whose aerodynamics and structural response are computed by
`OpenFAST <https://openfast.readthedocs.io>`_ while ERF supplies the wind. It
is separate from the wind farm parameterizations of :ref:`sec:WindFarmModels`,
which model turbines with thrust curves; here the loads come from OpenFAST's
blade-element solution on the turbine's own blade and tower nodes.

Coupling
--------

ERF talks to OpenFAST through its external-inflow C interface (``ExtInfw``,
OpenFAST 4). Every ERF step:

#. the flow velocity at OpenFAST's velocity nodes (hub, blade and tower
   structural nodes) is sampled from ERF's velocity field and handed to
   OpenFAST (or, for testing, the uniform ``erf.moving_bodies.prescribed_velocity``);
#. OpenFAST advances its own, smaller, time step ``n`` times, where ``n`` is the
   ratio of ``erf.fixed_dt`` to the OpenFAST ``DT``; ERF refuses to start when
   the ratio is not a whole number;
#. OpenFAST returns the positions of its actuator force points and the
   aerodynamic force on each. OpenFAST reports every position in the
   turbine's own frame, with the tower base at the origin; ERF adds
   ``erf.moving_bodies.<name>.base_pos`` to put them in the domain.

The models are initialised at start-up, so their inputs are checked then, but
OpenFAST's first solution is taken at the first step, once the flow exists to be
sampled at the nodes. Each turbine is owned by one MPI rank, which is the only rank that calls
OpenFAST; the node positions and forces are broadcast afterwards so that every
rank sees the same turbine. The OpenFAST model must set ``CompInflow = 2``
(external inflow) so that the velocities come from ERF rather than from
InflowWind.

Velocity sampling
-----------------

Each velocity component is interpolated to a node from its own staggered
grid: ``u`` from the x faces, ``v`` from the y faces and ``w`` from the z
faces, bilinearly in the horizontal between the four surrounding columns and
linearly in the physical height within each column, using the face heights of
``z_phys_nd`` on a stretched or terrain-following mesh. A field linear in
``x``, ``y`` and the physical height is therefore reproduced exactly, on every
mesh type. Every MPI rank samples the nodes whose containing cell lies in one
of its boxes, reading neighbours from the ghost cells, and the values are
summed across ranks, so the result does not depend on the domain
decomposition; a node outside the domain aborts the run with its coordinates.

Force spreading and momentum source
-----------------------------------

The force a body exerts on the fluid is carried by actuator points (the disk
points of a prescribed-Ct disk in this version) and spread onto ERF's
face-centred momentum sources with a 3-D Gaussian kernel of width
``epsilon`` (given in units of dx), cut off beyond three widths. The kernel is
normalised discretely on each staggered grid, with the face volumes
(``dx dy dz detJ`` on a terrain-following mesh), so the source integrates back
to the point force exactly for every point and component, whatever the
resolution and however much of the kernel the ground or the domain top cuts
off; the kernel's shape is Gaussian only where it is resolved. Distances take
the minimum image in periodic directions, so a kernel wraps across a periodic
boundary. Faces on a non-periodic domain boundary get no source. The sources are computed once per
step, from the velocities sampled at the start of the step, and added to the
momentum right-hand side in every stage of the step, before the anelastic
projection, which removes their divergent part.

Prescribed uniform-Ct disk
--------------------------

``type = ct_disk`` is the classical test rotor: a uniformly loaded disk of
radius ``R`` and normal ``n`` (yawed in the horizontal plane) covered by a
polar grid of points. The free-stream speed ``U`` is the area-weighted mean of
the normal velocity sampled on the same grid ``sample_diameters_upstream``
diameters upstream, the thrust is ``T = 1/2 rho Ct U^2 pi R^2``, and each disk
point carries the share of ``-T n`` proportional to its area. The disk's
diagnostics file records the upstream speed, the disk-averaged speed, the
thrust, the power ``T U_d`` and the integrated momentum source projected on
the normal, which equals the thrust when the disk is the only body.

Solver requirements
-------------------

This version supports the anelastic solver only, with a fixed time step and a
single level, and the AMReX floating-point traps must be off: OpenFAST's
initialisation raises exceptions of its own, and a trapped run dies inside the
library. The requirements are checked at start-up.

Diagnostics
-----------

Every turbine writes ``<output_root>_erf.csv`` with the time, rotor speed, the
thrust vector (the sum of the blade-node forces, as OpenFAST reports them: the
force of the fluid on the structure, so along the inflow), the aerodynamic
torque about the hub axis and the power (torque times rotor speed), and
``<output_root>_flow.csv`` with the velocity the flow supplied at the hub node
and its mean over the blade nodes. OpenFAST also writes its own output files as
configured in the ``.fst`` file.

Inputs are listed in :ref:`sec:MovingBodiesInputs`.
