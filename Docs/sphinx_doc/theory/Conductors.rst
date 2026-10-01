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
points it asks for (the line nodes first, then the attachments), valid at the
middle of the step, and the line is advanced by ERF's step in ``substeps``
MoorDyn calls; MoorDyn sub-steps internally with its own time step, set by
the Courant factor ``moordyn_cfl`` and bounded further by ``moordyn_dt`` when
that is given. ERF's clock and MoorDyn's must agree at the end of every step.

The Courant factor matters more than its name suggests. For a 300 m span of
795 kcmil ACSR with 20 segments, MoorDyn's own default of 0.5 (an internal
step of 11 ms) and 0.25 make the line integration diverge within half a
second in a 20 m/s wind; 0.2 runs but more than doubles the still-air sag
(36.6 m instead of 16.4 m); 0.15 and 0.1 agree to the millimetre. ERF
therefore writes 0.1 by default and refuses values above 0.15. A MoorDyn
step that still diverges aborts with ``MOORDYN_NAN_ERROR`` and the advice to
reduce the internal step. In this version the wind at
the nodes is the uniform ``prescribed_velocity`` (still air when it is not
given); sampling the flow at the nodes' current positions comes with the next
step of the coupling.

Placement on terrain
--------------------

The ``z`` of each attachment is its height above the terrain surface at its
``(x, y)``: the surface height is read from the mesh with the actuator core's
``terrain_heights`` (the ``k = 0`` node plane, bilinear between the nodes) and
added once at start-up, so on a flat mesh ``z`` is the absolute height. The
attachments must lie inside the domain. ``<diagnostics_dir>/ground.dat`` lists
each end with the terrain height found under it and its resulting absolute
height.

Diagnostics
-----------

Every ``diagnostics_int`` steps each span appends a row to
``<output_root>.dat`` (whitespace separated, like ERF's data logs): the time,
the middle node's position, its sag below the chord, its lateral offset from
the chord's vertical plane, the swing angle in degrees, the tension at the
first and last node and the largest tension on the line. The MoorDyn input
file the span was built from is kept next to it
(``<diagnostics_dir>/<name>.moordyn.txt``) for inspection or for running
MoorDyn on its own.

Checks at start-up
------------------

A run with spans refuses to start with any ``amrex.fpe_trap_*`` input on
(MoorDyn's initial-condition solver overflows an intermediate value), with an
anchor level that is not a level of the run, with a span whose length does not
exceed its chord, or with any span value outside its documented range. Every
message names the input key.
