.. _sec:Conductors:

Conductor spans in the wind
===========================

A conductor span (``Source/MovingBodies/Conductors``, inputs ``erf.conductors.*``)
is a flexible line hanging between two fixed attachment points, an overhead
power-line conductor, shield wire or insulator string, whose motion in the
wind is computed by MoorDyn-C (:doc:`../CouplingToMoorDyn`) while ERF supplies
the wind. The questions it answers are those of a line exposed to a fire wind:
how far the span blows out, how its clearance to the ground and to its
neighbours changes, and what tension the attachments carry.

Model
-----

MoorDyn integrates the lumped-mass line dynamics: the line is a chain of
``segments`` elastic segments of axial stiffness ``EA`` with internal damping,
each node carrying its share of the mass per unit length, the weight, the
added mass and the drag of the fluid relative to it. The fluid is ERF's air:
MoorDyn's "water" density is ``erf.conductors.air_density``, its free surface
and flat bottom are placed far above and below the span (``surface_offset``),
and the fluid velocity at every line node comes from ERF through MoorDyn's
external wave-kinematics interface instead of a wave model. The attachments
are fixed points, so the line has no coupled degree of freedom and MoorDyn
needs nothing back from ERF but the wind.

For a span of chord :math:`c` and unstretched length :math:`L > c` the
still-air shape is close to a parabola of mid-span sag
:math:`s = \sqrt{3 c (L - c) / 8}`, with the horizontal tension
:math:`H = w c^2 / (8 s)` for a weight :math:`w` per unit length; MoorDyn's
stationary solver finds the elastic catenary itself, and the start-up log
prints both the solved sag and this estimate. A steady crosswind of speed
:math:`U` normal to the span loads it with :math:`q = \tfrac12 \rho C_d D U^2`
per unit length, so the span swings out of its vertical plane towards the
quasi-static blowout angle :math:`\phi = \arctan(q / w)` and its tension rises
with the effective weight :math:`\sqrt{w^2 + q^2}`. The diagnostics report
the swing angle of the middle node about the chord against exactly this
reference.

Lockstep with ERF
-----------------

The spans live on their anchor level (``anchor_level``, the finest level by
default). Every step of that level, the wind is handed to MoorDyn at the
line nodes (MoorDyn also lists its fixed entries after them, the attachments
and one entry at its own origin, which get no wind), valid at the middle of
the step, and the line is advanced by ERF's step in ``substeps``
MoorDyn calls; MoorDyn sub-steps internally with its own time step, set by
the Courant factor ``moordyn_cfl`` and bounded further by ``moordyn_dt`` when
that is given. MoorDyn's clock starts at zero when the spans are created, at
ERF's time zero in a fresh run, and ERF's clock and MoorDyn's (plus ERF's time
at its zero) must agree at the end of every step.

The wind handed to MoorDyn is ERF's velocity at the start of the step,
sampled with the actuator core at the points where the line is at that
moment, not where it hung: a span blown out by several metres sits in a
different part of the flow, which matters on a ridge or near a plume. Each
component is interpolated from its own staggered grid, bilinearly in the
horizontal and linearly in the physical height, so a field linear in
``x``, ``y`` and the height is handed over exactly. The fluid acceleration
MoorDyn also accepts enters only its added-mass load, which in air is about
a thousandth of the line's own inertia, and is left at zero. Before every
sampling the points are checked: a point that has left the domain, or whose
surrounding cells are not on the anchor level's grids (a refined patch that
does not cover the whole span), stops the run with the span, the point and
its position. ``prescribed_velocity`` replaces the sampled wind by a uniform
one, for testing.

The Courant factor matters more than its name suggests. For a 300 m span of
795 kcmil ACSR with 20 segments, MoorDyn's own default of 0.5 (an internal
step of 11 ms) and 0.25 make the line integration diverge within half a
second in a 20 m/s wind; 0.2 runs but more than doubles the still-air sag
(36.6 m instead of 16.4 m); 0.15 and 0.1 agree to the millimetre. ERF
therefore writes 0.1 by default and refuses values above 0.15. A MoorDyn
step that still diverges aborts with ``MOORDYN_NAN_ERROR`` and the advice to
reduce the internal step.

Verification
------------

Against the real MoorDyn-C 2.7.1, a 300 m span of 795 kcmil ACSR with 1.5 m
of slack and 20 segments in air of density 1.2 kg/m^3 (the unit tests
``ConductorVerification``):

* in a steady crosswind the mean swing angle over 20 to 60 s is the
  quasi-static blowout angle :math:`\arctan(q/w)`: 6.023 against 6.029
  degrees at 10 m/s, 22.893 against 22.903 at 20 m/s and 43.519 against
  43.548 at 30 m/s;
* released after a one-second gust into still air, the span swings with the
  first out-of-plane period of a cable, :math:`T = 2c/\sqrt{H/m}` (Irvine,
  *Cable Structures*, 1981), which does not depend on the sag: 6.658 s
  measured over five periods against 6.655 s.

Both agree within 0.1 %; the tests allow 1 %.

* in still air the span hangs in the elastic catenary: the parameter
  :math:`a = H/w` solves :math:`2a\sinh(c/2a) = L_s`, where the stretched
  length :math:`L_s` exceeds the unstretched one by the strain integrated
  along the line, :math:`(H/EA)\,(c/2 + (a/2)\sinh(c/a))`. For the test
  span this gives a sag of 13.584 m, a horizontal tension of 13 256 N and
  an end tension of 13 473 N; MoorDyn gives 13.599 m and 13 430 N with 20
  segments and converges to 13.5838 m with 160 (the parabola estimate,
  12.99 m, and the inextensible catenary, 13.01 m, are both too low). The
  start-up log prints MoorDyn's sag and end tension against the elastic
  catenary for every level span.

The verification tests skip on the stub library, which has no line
dynamics and hangs a parabola.

Placement on terrain
--------------------

The ``z`` of each attachment is its height above the terrain surface at its
``(x, y)``: the surface height is read from the mesh with the actuator core's
``terrain_heights`` (the ``k = 0`` node plane, bilinear between the nodes) and
added once at start-up, so on a flat mesh ``z`` is the absolute height. The
attachments must lie inside the domain. ``<diagnostics_dir>/ground.dat`` lists
each end with the terrain height found under it and its resulting absolute
height.

Drag on the flow
----------------

With ``drag_on_flow`` the lines act on the air as well: after each MoorDyn
step the air's drag on every node, reversed, is spread onto ERF's
face-centred momentum sources with the actuator core's Gaussian of width
``epsilon`` cells, discretely normalised so that the source integrates back
to the force exactly, and added during the next step. Each node carries its
share of the line's drag, so the force per unit length is resolved as long
as the node spacing stays below the kernel width. A conductor's drag is
small next to the momentum of the wind it crosses (4 to 5 N/m in a 17 m/s
wind for 795 kcmil ACSR), which is why the switch is off by default; it is
there for dense bundles, many spans in a small domain, and to close the
momentum budget. The plot variables ``conductor_fx``, ``conductor_fy`` and
``conductor_fz`` hold the source averaged to the cell centres (N/m^3) on
the anchor level and are available only with the switch on.

Diagnostics
-----------

Every ``diagnostics_int`` steps each span appends a row to
``<output_root>.dat`` (whitespace separated, like ERF's data logs): the time,
the middle node's position, its sag below the chord, its lateral offset from
the chord's vertical plane, the swing angle in degrees, the tension at the
first and last node, the largest tension on the line, the wind handed to
MoorDyn at the middle node, the smallest clearance of any node above the
terrain under it with that node's horizontal position, and the air's total
drag on the line. The clearance uses the terrain surface under each node
(``terrain_heights``, the ``k = 0`` node plane), so a blown-out span on a
slope is measured against the ground it now hangs over.

From ``stats_start`` on each span keeps running statistics, the mean, root
mean square, minimum and maximum of its swing angle, mid-span offset, end
and maximum tensions, minimum clearance and crosswind drag, in
``<output_root>_stats.csv``; these are the numbers a turbulent-wind run is
judged on. With ``node_output_int`` every node of every span (position,
clearance, tension, wind and drag) is written to ``<output_root>_nodes.dat``
for plotting the line's shape. ``<diagnostics_dir>/total_load.dat`` holds
the air's drag on all lines, the force the lines put into the air (zero
without ``drag_on_flow``) and the integral of the momentum source, which
equals that force. The MoorDyn input
file the span was built from is kept next to it
(``<diagnostics_dir>/<name>.moordyn.txt``) for inspection or for running
MoorDyn on its own.

Restart
-------

A checkpoint carries the spans under ``<chk>/conductors``: each line's whole
MoorDyn state (node positions and velocities, internal forces and the time
integrator's state, through MoorDyn's own save), the running statistics, the
step count and the time. On a restart the line is created from the same
inputs, initialised without the initial-shape solve and given that state, so
it continues blown out exactly where the checkpoint left it, on MoorDyn's
clock; the statistics go on accumulating, and the diagnostics continue on the
step count of the original run. With ``drag_on_flow`` the restored lines' drag
is spread into the momentum sources again at once, so a plotfile written at
the restart shows the source of the checkpointed step. The span logs,
``<output_root>_nodes.dat`` and ``total_load.dat`` are appended to, after
the rows a run wrote beyond the checkpoint time are dropped, so a run that
went on past its last checkpoint and is restarted from it leaves no
duplicated stretch. The spans of a restart must be those of the run that
wrote the checkpoint, in the same order and with the same number of
segments; anything else stops the run naming the span. A checkpoint without
conductor state, from a run without spans, starts them afresh from their
still-air shape, with MoorDyn's clock at zero at the restart time.

Checks at start-up
------------------

A run with spans refuses to start with any ``amrex.fpe_trap_*`` input on
(MoorDyn's initial-condition solver overflows an intermediate value), with an
anchor level that is not a level of the run, with a span whose length does not
exceed its chord, or with any span value outside its documented range. Every
message names the input key.
