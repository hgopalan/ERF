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
OpenFAST 5; the interface is the same in 4.2). Every ERF step:

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
coupling, as for a loads analysis in a precomputed flow).

OpenFAST rotor as an actuator line
----------------------------------

With ``mode = alm`` the blade force nodes themselves are the actuator points:
after every OpenFAST step the positions it reports for the rotating, yawed and
deflected blades (``num_force_points_blade`` per blade, root to tip, plus the
hub node) carry minus its node forces into the Gaussian spreading, so the flow
sees the three moving lines of force and the tip and root vortices they shed,
which the disk averages away. The integrated momentum source equals minus the
full thrust vector, the torque about the shaft is preserved, and no in-plane
force is lost. The kernel width ``epsilon`` is in units of ``dx``; the usual
choice is ``epsilon = 2 dx`` with cells small enough that the kernel is no
wider than a few chord lengths near the tip (about 4 to 5 m cells for a 240 m
rotor), since a kernel wider than the chord smears the tip loading and raises
the power for a given inflow. The line must not jump over cells between two
steps: at every step ERF computes how many cells the blade tip sweeps,
``rotor_speed * tip_radius * dt / min(dx, dy, dz)``, with the tip radius the
largest distance of a blade node from the hub axis, and aborts when it exceeds
``erf.moving_bodies.alm_max_tip_cells`` (1 by default), naming the largest
``erf.fixed_dt`` that passes. For the IEA 15 MW at rated speed (tip speed
about 90 m/s) one cell per step means ``dt <= dx / 90``. The check runs every
step because the rotor speed is OpenFAST's to change. The velocities OpenFAST
receives are still sampled at its structural nodes, and the same ``Wake_Mod = 0``
requirement and hub-axis check apply as for the disk.

Filtered lifting-line correction
--------------------------------

An actuator line spread with a kernel of width ``epsilon`` (two cells or so,
tens of metres for a 240 m rotor) sees at its own points a weaker induced
velocity than the vortex sheet of a real blade, whose kernel is of the order
of the chord: the blades then see too much wind and the line over-predicts
power, more so for wider kernels. The filtered lifting-line correction
(Martinez-Tossas and Meneveau, 2019), ``fllc = true`` with ``mode = alm``,
computes the velocity the trailing vorticity of the line's own lift
distribution induces at the line for the kernel actually used and for the
optimal one, ``epsilon_opt = fllc_eps_chord * chord`` (a quarter chord by
default, the chord being OpenFAST's at each force node), and adds the relaxed
difference to the velocities OpenFAST is given at its blade nodes. For each
blade, with the force on the fluid per unit density ``F``, the relative
velocity ``v`` (flow minus blade motion) and the node width ``dr``, the lift
force per unit span is ``G = (F - v (F . v) / |v|^2) / dr`` and the induced
velocity of the sources ``G / |v|`` is

.. math::

   u(r) = \frac{1}{2\pi} \sum_j \frac{G_j}{|v_j|} \, k(|r - r_j|, \epsilon_j) \, dr_j,
   \qquad k(r, \epsilon) = \frac{e^{-r^2/\epsilon^2}}{\epsilon^2} + \frac{e^{-r^2/\epsilon^2} - 1}{2 r^2},

the derivative form of the Gaussian-filtered vortex kernel (its far field is
the unfiltered ``-1 / (2 r^2)``). The sum runs over a fine span grid with
spacing ``epsilon_opt / fllc_eps_dr`` onto which ``G / |v|`` and
``epsilon_opt`` are interpolated, since the optimal kernel is far narrower
than the node spacing. The correction ``du = (1 - f) du + f (u_opt - u_les)``
with ``f = fllc_relax`` is a downwash (against the lift) when the run's kernel
is wider than the optimal one, and vanishes when they coincide. It is
computed at the force nodes from the loads of the step just taken, applied
after the sampling by linear interpolation along the span between OpenFAST's
velocity and force nodes, starts at ``fllc_start_time``, and is checkpointed
(it is a relaxed state). ``<output_root>_fllc.csv`` records its largest and
root-mean-square value over the blades. This follows Kynema's variable-chord
operator; the loads are divided by the body's ``air_density``, which must be
the model's ``AirDens``.

Tower and nacelle
-----------------

Both are off unless asked for, in either rotor mode. With
``num_force_points_tower > 0`` OpenFAST is asked for that many tower force
nodes (base to top); AeroDyn computes the tower's drag from the velocities ERF
samples at its tower nodes (``TwrAero`` must be on in the AeroDyn file, and the
stub applies a cylinder drag with its own ``tower_diameter`` and ``tower_cd``),
and ERF spreads each tower node's force on the fluid with the same Gaussian
kernel as the rotor points, so the tower wake and its shadow on the rotor
appear in the flow. AeroDyn's own tower-shadow correction of the blade inflow
(``TwrShadow``) would then count the shadow twice, so ERF reads it from the
AeroDyn file and prints a warning when it is not 0. With ``nacelle_cd > 0``
and ``nacelle_area`` ERF adds one drag point at the hub node,
``-1/2 rho cd area |u| u`` on the fluid with ``rho`` the body's
``air_density`` and ``u`` the velocity sampled at the hub node, corrected for
the point's own kernel: a Gaussian force of width ``epsilon`` induces a
velocity at its own centre that lowers the sampled one by the factor
``1 - cd area / (4 pi epsilon^2)``, so the sampled velocity is divided by it
(the correction Kynema applies with its own nacelle kernel width; here the
run's kernel width is used, and the run aborts if the factor is not positive,
which needs a kernel narrower than the nacelle). The tower force and the
nacelle drag are written in the turbine diagnostics with the thrust, and the
integrated momentum source equals minus their sum.

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
``erf.moving_bodies.avg_start`` on, accumulated into a running time average that
``<output_root>_wake_avg.csv`` always holds (``samples`` is the number of
samples in it). The running sums are part of the checkpoint, so the average
continues across a restart. Nothing is sampled when a prescribed velocity
replaces the flow.

Body statistics
---------------

From ``erf.moving_bodies.avg_start`` on, every step adds the state of each
body to its running statistics, and ``<output_root>_stats.csv`` always holds
the current sample count, the first and last sample times and, per quantity,
the mean, the root mean square, the minimum and the maximum. For an OpenFAST
turbine the quantities are the thrust along the shaft and along x, the torque,
the power, the rotor speed, the hub-node velocity and the blade-mean
streamwise velocity; for a prescribed-Ct disk the upstream and disk speeds,
the thrust and the power. The sums are part of the checkpoint, so the
statistics continue across a restart. These are the numbers a turbulent-inflow
LES is compared on, where a single instant means little.

Turbulent inflow
----------------

Nothing in the coupling depends on the lateral boundary conditions: a
turbine can run in a periodic box, or with ERF's inflow and outflow
boundaries fed from a precursor's boundary planes
(``erf.input_bndry_planes``), or with a steady inflow profile and ERF's cell
perturbation method (``erf.perturbation_type = CPM``) making the turbulence
just inside the inflow face, which needs no precursor. The anelastic FFT
Poisson solve handles the non-periodic direction. The regression tests
``OpenFAST_ADM_LES`` (precursor planes) and ``OpenFAST_ADM_CPM`` (cell
perturbation) run the rotor both ways.

Anchor level on multi-level grids
---------------------------------

On a run with refinement the bodies live on one level, the anchor: the finest
level unless ``erf.moving_bodies.anchor_level`` names a coarser one. Their
nodes are sampled from that level's velocities, their momentum sources are
spread onto that level's faces with the kernel width in that level's cells,
and OpenFAST is stepped with that level's fixed step (``erf.fixed_dt`` divided
by the sub-cycling ratios down to it), so a fine patch around the rotor gives
an actuator line its 4 to 5 m cells without refining the whole domain.
Coarser levels carry no source of their own: they see the rotor through ERF's
average-down of the state after every coarse step, as any fine-level physics
does; finer levels than the anchor are not supported. Everything the bodies
read or write on the level must lie on its grids: at the first step every
node with the kernel's reach, every disk point, and every wake sampling
point is checked against the anchor level's box union (periodic directions
wrap), and a point outside aborts naming it, since the sampler and spreader
read only the level's own cells. Enlarge the refinement region or choose a
coarser anchor. The refinement region must also let the FFT solver run on
the level, that is be a rectangular union of boxes, as for any anelastic run.

Farms
-----

Any number of bodies can be listed in ``erf.moving_bodies.bodies``; each has
its own input block, its own diagnostics files (``<output_root>_*.csv``) and,
for an OpenFAST turbine, its own OpenFAST instance. Turbine ``i`` (in the
order of the list) is owned by MPI rank ``i`` modulo the number of ranks, so a
farm of up to as many turbines as ranks runs one OpenFAST instance per rank
and a larger one spreads evenly; only the owner rank calls OpenFAST, and the
node positions, loads and velocities are broadcast after every step so that
every rank samples and spreads for every body. The result does not depend on
the rank count or on which rank owns which turbine. OpenFAST keeps a single
step counter for all the turbines of a process, so the turbines a rank owns
are stepped in turn, one OpenFAST time step at a time; they must therefore
use the same OpenFAST ``DT`` (the run aborts at start-up otherwise), and the
stub enforces the same order. All bodies share one
spreading width and their forces are spread onto the same momentum sources.
``<diagnostics_dir>/total_load.csv`` records the sum of every body's load on
the structure (the turbines' thrust, tower and nacelle forces and the disks'
thrust) and the turbines' total aerodynamic power, at the times of the turbine
logs; minus that load is what the integrated momentum source must equal. Wake
lines are built behind every rotor, so a downstream turbine's lines start
behind it, not behind the first.

Checkpoint and restart
----------------------

An ERF checkpoint carries the bodies' state under ``<chk>/moving_bodies``:
a ``state`` file with the step count and each turbine's OpenFAST time index,
each OpenFAST turbine's own checkpoint ``<name>.chkp``, written through
``FAST_CreateCheckpoint`` by the rank that owns the turbine, the wake
lines' running sums and the bodies' statistics. On a restart the
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

Start-up audit
--------------

At the first step, once the node layout, the flow at the nodes and ERF's
density at the hubs are known, every OpenFAST turbine is audited against the
ERF set-up and the findings are printed together under ``OpenFAST input audit
for <name>``; a fatal finding aborts the run after the whole list has been
printed. Fatal: the model's air density (``AirDens`` of the primary file, or
of the AeroDyn file when the primary says default) differs from the body's
``air_density``; ERF's density sampled at the hub differs from the model's by
more than ``erf.moving_bodies.density_tolerance`` (5 % by default; the loads
OpenFAST computes with its density go into a flow of ERF's, so the two must
agree, and an ABL's base state thins with height, hence the tolerance);
``CompAero = 0`` while the loads are meant to go into the flow; a rotor that
reaches below the ground, above the domain top or across a non-periodic side;
a base outside the domain; two rotors whose swept discs intersect. Warnings:
the model's gravity differs from ERF's; a forced tower with AeroDyn's
``TwrAero`` off; fewer than 8 cells across the rotor diameter; a kernel
narrower than a cell; an actuator line whose points are farther apart than
the kernel; a kernel cut by a non-periodic boundary; a rotor facing away from
the wind at its hub or yawed more than 30 degrees from it. The bundled stub's
deck is not an OpenFAST primary file, so its model-file checks are skipped
with a note. The checks made at start-up before the flow exists remain: the
fixed step a whole multiple of the OpenFAST step, external inflow on,
``Wake_Mod = 0``, ``TwrShadow = 0`` with a forced tower, and later the
tip-travel limit and the hub-axis convention.

Solver requirements
-------------------

This version supports the anelastic solver only, with a fixed time step, on
one anchor level of a possibly refined grid, and the AMReX floating-point
traps must be off: OpenFAST's
initialisation raises exceptions of its own, and a trapped run dies inside the
library. The requirements are checked at start-up; the actuator line's tip
travel per step is checked at every step, since it depends on the rotor speed.

Diagnostics
-----------

Every turbine writes ``<output_root>_erf.csv`` with the time, rotor speed, the
thrust vector (the sum of the hub and blade node forces, as OpenFAST reports
them: the force of the fluid on the structure, so along the inflow), the
aerodynamic torque about the hub axis, the power (torque times rotor speed),
the unit hub axis (the shaft direction, which follows yaw, tilt and the
tower's deflection), the tower force (the sum over its force nodes, zero
without them), the nacelle drag (zero unless asked for) and the total load
(thrust plus tower plus nacelle, all on the structure), and
``<output_root>_flow.csv`` with the velocity the flow supplied at the hub node
and its mean over the blade nodes. OpenFAST also writes its own output files as
configured in the ``.fst`` file. Whenever a body puts a force into the flow,
``<diagnostics_dir>/momentum_source.csv`` records the time and the momentum
source integrated over the domain, which equals the sum of the forces on the
fluid (minus the turbines' total loads, plus the disks' ``-T n``), and
``<diagnostics_dir>/total_load.csv`` the bodies' total load and power (see
Farms).

Inputs are listed in :ref:`sec:MovingBodiesInputs`.
