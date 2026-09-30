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
   ``erf.moving_bodies.<name>.base_pos`` to put them in the domain. The hub's
   orientation matrix gives the shaft axis (its first row, the hub frame's x
   axis in the global frame), which the actuator-disk representation below
   needs;
#. the forces, with the sign reversed to act on the fluid, are spread onto the
   momentum sources that ERF adds during the step it is about to take.

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

The force a body exerts on the fluid is carried by actuator points (the rings
of an OpenFAST rotor's actuator disk, the disk points of a prescribed-Ct disk)
and spread onto ERF's face-centred momentum sources with a 3-D Gaussian kernel of width
``epsilon`` (given in units of dx), cut off beyond three widths. The kernel is
normalised discretely on each staggered grid, with the face volumes
(``dx dy dz detJ`` on a terrain-following mesh), so the source integrates back
to the point force exactly for every point and component, whatever the
resolution and however much of the kernel the ground or the domain top cuts
off; the kernel's shape is Gaussian only where it is resolved. Each point is
visited over the faces within three widths of it, and again as its periodic
images, so a kernel wraps across a periodic boundary and the cost grows with
the number of points, not with the mesh. Faces on a non-periodic domain
boundary get no source. The sources are computed once per step, from the
velocities sampled at the start of the step (the disks) or from the OpenFAST
step just taken (the turbines), and added to the momentum right-hand side in
every stage of the step, before the anelastic projection, which removes their
divergent part.

.. note::

   **Why the normalisation is discrete.** Actuator codes in the AMR-Wind and
   SOWFA lineage (Kynema among them) divide each point's force by the analytic
   Gaussian volume, ``epsilon^3 pi^(3/2)``, and then sum the kernel over the
   cells. The sum over cell volumes equals that analytic volume only when the
   kernel lies wholly inside the domain, is resolved by the mesh and covers
   cells of one size. Otherwise the momentum the fluid receives is not the
   force the point carries, and nothing reports it:

   * the part of a kernel below the ground or above the domain top is never
     deposited; a point at height ``h`` loses the fraction ``erfc(h/epsilon)/2``
     of its force, about 14 % for the lowest blade tip of the IEA 15 MW rotor
     with ``epsilon = 40 m`` and up to half for a tower point near the base;
   * with ``epsilon`` below about two cells the cell sum of a Gaussian is no
     longer its integral, so the injected momentum is off by a few per cent and
     changes with the resolution;
   * on a stretched or terrain-following mesh the cell volumes vary across the
     kernel, which no constant can account for.

   The symptoms are a wake deficit slightly too weak, a thrust felt by the
   flow that depends on the grid spacing and on how low the rotor sits, and
   momentum budgets that do not close. ERF pays a second pass per point to
   sum the kernel over the faces it actually reaches, with their volumes, so
   the integrated source equals the force exactly in every case; the
   regression tests check this identity in every row of the diagnostics.

OpenFAST rotor as an actuator disk
----------------------------------

With ``mode = adm`` the loads OpenFAST computes on its blade force nodes are put
into the flow as a disk rather than as rotating lines. Each blade node, at
radius ``r`` from the hub in the plane normal to the shaft axis, is replaced by
a ring of ``num_points_t`` points at that radius, equally spaced in azimuth,
each carrying ``1 / num_points_t`` of the node's force with the axial component
kept and the radial and tangential components rotated with the ring azimuth.
The force along the shaft and the torque about it are therefore preserved
exactly; the net force in the rotor plane, which a three-bladed rotor carries
only through the differences between its blades' loads at an instant, is
averaged away with the azimuth, as an actuator disk does (OpenFAST still sees
the true blade positions, only the flow does not). The integrated momentum
source therefore equals minus the thrust vector for a rotor whose blade loads
are alike, and minus its shaft component in general. The hub node keeps its own point. At the first step ERF checks that
the blade nodes lie within about 20 degrees of the plane normal to the axis it
took from the hub orientation, so that a mismatched orientation convention
aborts rather than spreading the load along the blades. The velocities ERF
samples then already contain the rotor's induction, so AeroDyn's own wake model
must be off: ERF reads the AeroDyn file named in the ``.fst`` and refuses to
start when its ``Wake_Mod`` is not 0 (``mode = none`` leaves it to the model).
The thrust in the diagnostics is the sum of the hub and blade node forces, so
its shaft component equals minus that of the integrated source. With ``mode = none``
the turbine is driven by the flow but puts no force into it (one-way
coupling, as for a loads analysis in a precomputed flow). Tower forces and
the actuator-line representation (``mode = alm``) are not available in this
version and are refused at start-up.

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

Wake diagnostics
----------------

``erf.moving_bodies.wake.lines_xD`` puts sampling lines behind every rotor:
at each listed distance in rotor diameters along the horizontal projection
of the rotor axis through the hub (the shaft axis of an OpenFAST turbine,
the normal of a prescribed-Ct disk; a wake follows the wind at hub height,
and a shaft tilt of a few degrees would otherwise carry the far lines into
the ground), one horizontal line normal to that direction and one vertical line, each of
``num_points`` points spanning ``half_width`` diameters either side of the
axis; the vertical line is clipped at the ground, so its lowest point may
sit closer to the axis than ``half_width`` (each point's offset ``s`` is in
the files). The lines are built at the first sample, from the rotor's
diameter (the outermost blade force node, or the disk radius), and every
point must lie in the domain otherwise. Every ``wake.int`` steps the velocity is sampled at the
points with the actuator sampler, written to ``<output_root>_wake.csv``
(``time, xD, line, s, x, y, z, u, v, w`` with ``s`` in diameters) and, from
``wake.avg_start`` on, accumulated into a running time average that
``<output_root>_wake_avg.csv`` always holds (``samples`` is the number of
samples in it). The running sums are part of the checkpoint, so the average
continues across a restart. Nothing is sampled when a prescribed velocity
replaces the flow.

Checkpoint and restart
----------------------

An ERF checkpoint carries the bodies' state under ``<chk>/moving_bodies``:
a ``state`` file with the step count and each turbine's OpenFAST time index,
each OpenFAST turbine's own checkpoint ``<name>.chkp``, written through
``FAST_CreateCheckpoint`` by the rank that owns the turbine, and the wake
lines' running sums. On a restart the
turbines are restored with ``FAST_ExtInfw_Restart`` instead of being
initialised, ERF checks that the time index OpenFAST reports is the one its
own checkpoint expects, and the run continues with the next step; the
momentum sources are not stored but rebuilt at that step from the restored
loads, as they would have been in the run being continued. The diagnostics
files are appended to (a restart in a clean directory starts them afresh
with their headers). The ``erf.moving_bodies`` block of the restarted run
must name the same bodies as the run that wrote the checkpoint. A checkpoint
written by a run without bodies (a precursor) can be restarted with bodies:
they then start afresh at the restart.

Solver requirements
-------------------

This version supports the anelastic solver only, with a fixed time step and a
single level, and the AMReX floating-point traps must be off: OpenFAST's
initialisation raises exceptions of its own, and a trapped run dies inside the
library. The requirements are checked at start-up.

Diagnostics
-----------

Every turbine writes ``<output_root>_erf.csv`` with the time, rotor speed, the
thrust vector (the sum of the hub and blade node forces, as OpenFAST reports
them: the force of the fluid on the structure, so along the inflow), the
aerodynamic torque about the hub axis, the power (torque times rotor speed)
and the unit hub axis (the shaft direction, which follows yaw, tilt and the
tower's deflection), and
``<output_root>_flow.csv`` with the velocity the flow supplied at the hub node
and its mean over the blade nodes. OpenFAST also writes its own output files as
configured in the ``.fst`` file. Whenever a body puts a force into the flow,
``<diagnostics_dir>/momentum_source.csv`` records the time and the momentum
source integrated over the domain, which equals the sum of the forces on the
fluid (minus the turbines' thrust vectors, plus the disks' ``-T n``).

Inputs are listed in :ref:`sec:MovingBodiesInputs`.
