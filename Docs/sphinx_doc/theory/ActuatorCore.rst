.. _sec:ActuatorCore:

Actuator core for bodies in the flow
====================================

The actuator core (``Source/MovingBodies/Core``, namespace ``erf_actuator``)
is the part of ERF that connects a body in the flow to the mesh: it samples the
flow velocity at a set of points, spreads point forces back onto the momentum
equations, keeps sampling lines in the body's wake and accumulates running
statistics of the body's diagnostics. It knows nothing about what the body is.
A body model supplies the positions of its points every step, receives the
velocity sampled there, and hands back the force each point exerts on the
fluid; the core does the rest on uniform, stretched and terrain-following
meshes alike. The core is built with ``-DERF_ENABLE_MOVING_BODIES=ON``, which
defines ``ERF_USE_MOVING_BODIES``; the default build does not compile it.

Actuator points
---------------

``ActuatorPoints`` flattens the points of every body into one list, so that the
sampler and the spreader see one set of points whatever the bodies are. Body
``b`` owns the points ``[begin(b), end(b))``; the positions are refreshed every
step from the bodies and the velocities are whatever the flow sampled at them.

Velocity sampling
-----------------

``sample_velocity`` interpolates each velocity component to a point from its own
staggered grid: ``u`` from the x faces, ``v`` from the y faces and ``w`` from
the z faces, bilinearly in the horizontal between the four surrounding columns
and linearly in the physical height within each column, using the face heights
of ``z_phys_nd`` on a stretched or terrain-following mesh (``face_height`` in
``ERF_ActuatorGeometry.H``, shared with the spreader so that both see the same
face positions). A field linear in ``x``, ``y`` and the physical height is
therefore reproduced exactly, on every mesh type. Every MPI rank samples the
points whose containing cell lies in one of its boxes, reading neighbours from
the ghost cells (the velocity fields need one filled ghost cell), and the
values are summed across ranks, so the result does not depend on the domain
decomposition and every rank holds the same velocities on return. A point
outside the domain, or one found by no rank or by more than one, aborts the
run with its coordinates: the caller owns the points and must keep them inside
the domain.

``sample_cell_scalar`` does the same for one component of a cell-centred field
(bilinear between the four cell centres around the point, linear in the
physical height between the two cell centres that bracket it in the column).
``terrain_heights`` returns the height of the terrain surface under each
point: the ``k = 0`` node plane of ``z_phys_nd``, bilinear between the four
nodes around the point's ``(x, y)``, or the domain's lower ``z`` on a
uniform-dz mesh. A body model uses it to place a body's base at a height above
the ground and to measure clearances.

``points_covered_by`` tells whether every point, with the cells within a given
reach around it, lies inside the union of boxes of a level: the sampler needs
the neighbouring cells and the spreader the kernel's reach, and on a refined
level that is not the whole domain. Periodic directions wrap; the first point
outside is named.

Force spreading and momentum source
-----------------------------------

``spread_forces`` carries the force each point exerts on the fluid onto ERF's
face-centred momentum sources with a 3-D Gaussian kernel of width ``epsilon``
(in metres), cut off beyond three widths. The kernel is normalised discretely
on each staggered grid, with the face volumes (``dx dy dz detJ`` on a
terrain-following mesh, from ``face_volume``), so the source integrates back to
the point force exactly for every point and component, whatever the resolution
and however much of the kernel the ground or the domain top cuts off; the
kernel's shape is Gaussian only where it is resolved. Each point is visited
over the faces within three widths of it, and again as its periodic images, so
a kernel wraps across a periodic boundary and the cost grows with the number
of points, not with the mesh. Every rank accumulates the faces its boxes own,
each physical face once however the boxes share their boundary faces, and the
normalisations are summed across ranks, so the result does not depend on the
domain decomposition. The sources are set, not added to, on every valid face;
faces on a non-periodic domain boundary get no source. ``integrate_source``
returns the integral of a face-centred source over the mesh, which the
regression tests of a body model compare with the force it carries.

.. note::

   **Why the normalisation is discrete.** Actuator codes in the AMR-Wind and
   SOWFA lineage divide each point's force by the analytic Gaussian volume,
   ``epsilon^3 pi^(3/2)``, and then sum the kernel over the cells. The sum over
   cell volumes equals that analytic volume only when the kernel lies wholly
   inside the domain, is resolved by the mesh and covers cells of one size.
   Otherwise the momentum the fluid receives is not the force the point
   carries, and nothing reports it:

   * the part of a kernel below the ground or above the domain top is never
     deposited; a point at height ``h`` loses the fraction ``erfc(h/epsilon)/2``
     of its force, up to half for a point near the ground;
   * with ``epsilon`` below about two cells the cell sum of a Gaussian is no
     longer its integral, so the injected momentum is off by a few per cent and
     changes with the resolution;
   * on a stretched or terrain-following mesh the cell volumes vary across the
     kernel, which no constant can account for.

   The symptoms are a wake deficit slightly too weak, a force felt by the flow
   that depends on the grid spacing and on how low the body sits, and momentum
   budgets that do not close. ERF pays a second pass per point to sum the
   kernel over the faces it actually reaches, with their volumes, so the
   integrated source equals the force exactly in every case.

Bodies on terrain
-----------------

The sampler and the spreader work in the physical heights of a fitted mesh
(``z_phys_nd``) and with the cell volumes (``detJ``), so a body's points and
the kernel follow the terrain without further change. A body model that
places its base at a height above the ground adds the terrain height from
``terrain_heights`` once at start-up; the vertical wake line of a body stops
at the terrain under its centre.

Wake sampling lines
-------------------

``WakeLines`` puts sampling lines behind one body: at each requested distance,
in units of the body's diameter along the horizontal projection of the body's
axis through its centre (a wake follows the wind at the body's height, and a
tilted axis would otherwise carry the far lines into the ground), one
horizontal line normal to that direction and one vertical line, each of
``num_points`` points spanning ``half_width`` diameters either side of the
axis. The vertical line is clipped at the ground height given, so its lowest
point may sit closer to the axis than ``half_width``; each point's offset ``s``
along its line, in diameters, is in the files. The velocity sampled at the
points with the actuator sampler is written as it is to ``<output_root>_wake.csv``
(``time, xD, line, s, x, y, z, u, v, w``) and accumulated into a running time
average that ``<output_root>_wake_avg.csv`` always holds (``samples`` is the
number of samples in it). The running sums, with the sampling geometry, are
written to and read from a checkpoint directory, so the average continues
across a restart.

Running statistics
------------------

``RunningStats`` keeps, for a named body and a list of named scalar
quantities, the mean, the root mean square, the minimum and the maximum of
every sample added since the averaging start, with the sample count and the
first and last sample times; ``<output_root>_stats.csv`` always holds the
current values, one row per quantity. The sums are written to and read from a
checkpoint directory, so the statistics continue across a restart. These are
the numbers a turbulent-inflow run is compared on, where a single instant
means little.

Diagnostics files
-----------------

``open_log`` opens a CSV diagnostics file for writing, creating its directory.
A fresh run truncates the file; a restarted run appends to what the run it
continues wrote, and the header is written only when the file is new or empty.
