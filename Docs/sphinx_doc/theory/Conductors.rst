.. _sec:Conductors:

Conductor lines in the wind
===========================

A conductor line (``Source/MovingBodies/Conductors``, inputs ``erf.conductors.*``)
is an overhead power-line conductor or shield wire hanging between two
dead-end attachments, either as a single span or as a section of spans over
suspension towers, where it hangs from insulator strings or is clamped. Its
motion in the wind is computed by MoorDyn-C (:doc:`../CouplingToMoorDyn`)
while ERF supplies the wind. The questions it answers are those of a line
exposed to a fire wind: how far the spans blow out, how their clearance to
the ground and to the neighbouring lines changes, and what tension the
attachments carry, down to the transformers the lines end on.

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

Sections and insulator strings
------------------------------

On a real line the conductor runs over many towers. At a suspension tower it
hangs from an insulator string, a chain of porcelain or glass discs a few
metres long that swings freely, and the spans on either side pull along the
line with nearly the same force, so the string only swings across the line
with the wind. At a dead-end (strain) tower the conductor is anchored. A
single span between two fixed points cannot hang from suspension strings:
nothing would balance its horizontal tension, some 13 kN for the test
conductor, and the strings would be dragged along the line until nearly
horizontal. The ``towers`` of a line are therefore the suspension points
between its two dead-ends: the line becomes a section of one span more than
the towers, each span with its own unstretched ``length``, written as one
MoorDyn system. With ``insulator_length`` the conductor hangs at each tower
from a string, a short MoorDyn line from a fixed point at the tower down to a
free point where the spans either side meet, with the string's mass, the
wind on its disc diameter, an axial stiffness of 1e7 N (a 2.5 m string gives
4 mm under a 15 kN conductor) and two segments; without it the conductor is
clamped to the towers, as a shield wire is at the tower peaks.

The span quantities (sag, offset, swing, tensions, clearance, drag) are
reported per span, measured from the chord between the span's attachment
points at the towers, so the sag and the blowout of a span on strings
include the drop and the swing of its strings: the blowout is the conductor's
displacement from the vertical plane of the tower attachments, the number
that sets its clearance. Each string's angle from the vertical, its angle
across the line (positive to the left of the direction from ``end_a`` to
``end_b``, the side a positive offset is on) and the tension at its top are
reported too.

In a steady crosswind each string of a long section swings to the angle of
the transverse load over the vertical load it carries, the wind span over
the weight span: with equal spans of length :math:`L` either side,

.. math::

   \tan\theta = \frac{q L + q_i L_i / 2}{w L + W_i / 2},

where :math:`q_i = \rho C_d D_i U^2 / 2` is the wind load per unit length of
the string of length :math:`L_i` and diameter :math:`D_i`, and :math:`W_i` its
weight net of buoyancy.

A string takes only the weight span: a span strung to the horizontal tension
:math:`H` pulls down on its end by :math:`w h / 2 + H \Delta z / h`, with
:math:`h` its horizontal length and :math:`\Delta z` the height of the end
above the other, so a tower in a dip, the spans either side rising away from
it, can be pulled up. Its string then carries none of the conductor and flips
over the cross-arm, the line slackening on either side; in practice such a
point needs a strain tower or a taller one. The start-up log warns, naming the
line and the tower, about every string that carries less than a tenth of the
weight of the half spans either side in still air. Against the real MoorDyn, a
section whose dead ends stand 80 m above its two towers, strung to 20 kN (each
span rising 80 m over 300 m pulls up 5.3 kN a side against 2.4 kN of weight),
flags both strings, and a level section neither, its strings carrying one
span's weight each.

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
  measured over five periods against 6.655 s;
* a section of three such spans over two suspension towers, hanging from
  2.5 m strings of 60 kg, swings its strings in a steady 20 m/s crosswind to
  22.382 degrees on average over 30 to 60 s, against 22.446 degrees from the
  wind span over the weight span (0.3 %), and at most 0.11 degrees along
  the line, the spans either side balancing (a probe gave 5.904 against
  5.896 degrees at 10 m/s and 42.56 against 42.91 at 30 m/s);
* in still air the span hangs in the elastic catenary: the parameter
  :math:`a = H/w` solves :math:`2a\sinh(c/2a) = L_s`, where the stretched
  length :math:`L_s` exceeds the unstretched one by the strain integrated
  along the line, :math:`(H/EA)\,(c/2 + (a/2)\sinh(c/a))`. For the test
  span this gives a sag of 13.584 m, a horizontal tension of 13 256 N and
  an end tension of 13 473 N; MoorDyn gives 13.599 m and 13 430 N with 20
  segments and converges to 13.5838 m with 160 (the parabola estimate,
  12.99 m, and the inextensible catenary, 13.01 m, are both too low). The
  start-up log prints MoorDyn's sag and end tension against the elastic
  catenary for every level span between fixed points.
* the dead ends of that span carry its weight and the catenary's horizontal
  tension: the pulls MoorDyn gives on the two fixed points add up to
  4812.82 N downwards against the line's weight of 4812.96 N, and each
  pulls 13 234 N along the chord against the catenary's 13 256 N (0.17 %).

The blowout and swing period agree within 0.1 %, the string swing within
0.3 % and the still-air sag within 0.11 %; the tests allow 1 % (0.5 % for the
catenary). The verification tests skip on the stub library, which has no line
dynamics and hangs an elastic parabola.

Stringing
---------

A line is either given an unstretched ``length`` per span or strung to a
``stringing_tension``, the horizontal tension :math:`H` every span carries in
still air, as a line crew strings a section. A section strung so hangs its
strings plumb, the spans either side of every tower pulling alike, whatever
their lengths; equal sag ratios instead would leave a short span far slacker
than a long one and drag the string between them along the line. Once the
attachments stand on the terrain, each span's length is set from the points
the conductor hangs from (the bottoms of the strings at the towers), a chord
:math:`c` over a horizontal distance :math:`h`: the parabola of horizontal
tension :math:`H` under the weight :math:`w` per unit length is

.. math::

   l = c + \frac{w^2 h^4}{24 H^2 c}

long, the tension along the chord :math:`H c / h` stretches the line by that
over :math:`EA`, and the unstretched length is
:math:`l / (1 + H c / (EA\,h))`. A short span may then be shorter than its
chord unstretched, held taut by its stretch, which MoorDyn handles like any
other. On the terrain test case MoorDyn's still-air solve gives every span
of the two lines strung to 20 kN end tensions of 19.8 to 20.9 kN, the
tension along the line being a little above the horizontal one on the
inclined spans.

Placement on terrain
--------------------

The ``z`` of each attachment, the ends and the towers, is its height above
the terrain surface at its ``(x, y)``: the surface height is read from the
mesh with the actuator core's ``terrain_heights`` (the ``k = 0`` node plane,
bilinear between the nodes) and added once at start-up, so on a flat mesh
``z`` is the absolute height, and a tower on a hill top holds its conductor
that much higher. Each span's ``length`` must exceed the distance between
its attachments where they stand: on a slope a dead end 10 m above a summit
and a tower 30 m above the hillside below it can be closer, or further
apart, than their heights above the ground suggest, so the slack is checked
once they are placed. The attachments must lie inside the domain.
``<diagnostics_dir>/ground.dat`` lists each attachment (``a``, the towers
``t1``, ``t2``, ..., ``b``) with the terrain height found under it and its
resulting absolute height, and each transformer (point ``transformer``) with
the terrain height under its centre and the height of its top.

With an immersed terrain (``erf.terrain_type = ImmersedForcing``) the mesh is
flat and the hills are solid cells inside it, so the mesh's bottom says nothing
about the ground. The lines then take the terrain's height from the surface the
immersed boundary is built from (``erf.terrain_file_name`` or the problem's
own terrain, at the nodes of the anchor level), bilinear between its nodes as on
a fitted mesh's bottom: the ends, the towers, the transformers and the ground
under every node for the clearance. The wind is sampled at the nodes' absolute
heights, which on the flat mesh are the mesh's own. Placed on an immersed ramp,
a section and its towers stand exactly where they stand on the same ramp as a
fitted mesh. A precursor run on the flat mesh can therefore seed a run over
immersed hills from its checkpoint. An anelastic run takes no acoustic substeps,
so its immersed forcing must act on the slow step (``erf.immersed_forcing_substep``
false, the default for anelastic runs), and the point-implicit form
(``erf.if_implicit_drag = true``) keeps it stable at the flow's step: explicitly,
the drag on a solid cell, :math:`C_d |u| / \Delta z`, about 60 s\ :sup:`-1` for 20
m/s on 16 m cells, is far beyond a 0.3 s step.

Drag on the flow
----------------

With ``drag_on_flow`` the lines act on the air as well: after each MoorDyn
step the air's drag on every node, reversed (and on every node of the lattice
towers, see `Towers`_), is spread onto ERF's
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
``<output_root>.dat`` for a single span, or ``<output_root>_span<k>.dat`` for
span ``k`` of a section (whitespace separated, like ERF's data logs): the time,
the middle node's position, its sag below the chord, its lateral offset from
the chord's vertical plane, the swing angle in degrees, the tension at the
first and last node, the largest tension on the line, the wind handed to
MoorDyn at the middle node, the smallest clearance of any node above the
terrain under it with that node's horizontal position, and the air's total
drag on the span. A line on strings also writes
``<output_root>_insulators.dat``: for each tower ``t<j>`` the string's swing
from the vertical and across the line in degrees and the tension at its top.
The clearance uses the terrain surface under each node
(``terrain_heights``, the ``k = 0`` node plane), so a blown-out span on a
slope is measured against the ground it now hangs over.

From ``stats_start`` on each span keeps running statistics, the mean, root
mean square, minimum and maximum of its swing angle, mid-span offset, end
and maximum tensions, minimum clearance and crosswind drag, in
``<output_root>_stats.csv`` (``<output_root>_span<k>_stats.csv`` in a
section), and the strings theirs in ``<output_root>_insulators_stats.csv``;
these are the numbers a turbulent-wind run is judged on. With
``node_output_int`` every node of every line (the spans' in order, then the
strings'; position,
clearance, tension, wind and drag) is written to ``<output_root>_nodes.dat``
for plotting the line's shape. ``<diagnostics_dir>/total_load.dat`` holds
the air's drag on all lines, the force the lines put into the air (zero
without ``drag_on_flow``) and the integral of the momentum source, which
equals that force. The MoorDyn input
file the span was built from is kept next to it
(``<diagnostics_dir>/<name>.moordyn.txt``) for inspection or for running
MoorDyn on its own.

Clearance between lines
-----------------------

Every pair of lines is watched for how close their conductors come: the
closest approach of the two polylines through their span nodes (the
strings left out, since a string belongs to its own phase), found exactly
from the closest points of every pair of segments. Every
``diagnostics_int`` steps ``<diagnostics_dir>/separation.dat`` gets, for
each pair ``A-B``, that distance, the point midway between the closest
points and a clash flag, 1 when the distance is below
``flashover_distance``, the gap an arc is assumed to jump. The flashover
distance depends on the voltage and the line's insulation coordination; the
default of 1 m is a placeholder to be set for the line studied. From
``stats_start`` each pair keeps the mean, root mean square, minimum and
maximum of the distance and of the flag, whose mean is the fraction of the
time spent in a clash, in ``<diagnostics_dir>/separation_A-B_stats.csv``.
The start-up log prints how far apart each pair hangs in still air and
warns when a pair starts inside the flashover distance. Two identical
parallel lines are almost equally far apart along their whole length, so
where they come closest can move by a span on a millimetre's difference;
the distance itself is well defined.

Transformers
------------

The lines of a network end somewhere: at a substation, on a transformer. A
transformer (``erf.conductors.transformers``, a block of its own per name) is
a box of ``size`` (length along ``x``, width along ``y``, height) standing on
the terrain: its base is at the terrain height under the centre of its
footprint, ``position``. Every line end, ``end_a`` or ``end_b``, whose
``(x, y)`` lies on a footprint (edges included) is dead-ended on that
transformer, at the end's own height above the terrain, which must clear the
box's top; the MoorDyn line is unchanged, its end still a fixed point. Two
transformers may not overlap under an end, and a transformer on which no line
ends is refused.

A dead end takes the pull of its line: the net force MoorDyn finds on the
line's end node, which the fixed point holds still, that is the end
segment's tension with the node's share of the weight and the drag. The
load on a transformer is the sum :math:`\mathbf{F}` of those pulls and
their moment about the centre of its base,

.. math::

   \mathbf{M} = \sum_e (\mathbf{x}_e - \mathbf{x}_b) \times \mathbf{F}_e ,

where :math:`\mathbf{x}_e` is the dead end and :math:`\mathbf{x}_b` the base
centre. The horizontal force :math:`F_h = |(F_x, F_y)|` and the overturning
moment :math:`M_h = |(M_x, M_y)|`, the one that tips the box, are checked
against ``allowable_force`` and ``allowable_moment``; the flag is up when
either is exceeded, and an allowable of 0 (the default) is not checked.
Lines pulling from different sides partly cancel, so the check is on the
vector sum, which is what the foundation and the bushings carry. In still air
two lines ending on a transformer from opposite sides balance; in a crosswind
they no longer do.

Every conductor is also watched for how close it comes to each box: the
closest approach of the polyline through its span nodes to the box, found by
a golden-section search along each segment (the distance from a point moving
along a segment to a box is convex). The lines ending on a transformer count
too: their ends clear its top by their standoff, and a conductor dropping
steeply from its bushing towards a valley comes closer. The clearance is
flagged against ``flashover_distance`` like a pair of lines.

Every ``diagnostics_int`` steps ``<diagnostics_dir>/transformers.dat`` gets,
for each transformer ``T``, the force components ``T_Fx``, ``T_Fy``,
``T_Fz``, ``T_Fh``, the moment ``T_Mx``, ``T_My``, ``T_Mh``, the allowable
flag ``T_over``, the clearance ``T_clearance`` and its flag ``T_clash``; from
``stats_start`` the horizontal force, the overturning moment, both flags and
the clearance are kept as running statistics in
``<diagnostics_dir>/transformer_T_stats.csv``. The start-up log prints, for
each transformer, the height of its base, the line ends on it, the still-air
load and the closest conductor.

Towers
------

The suspension towers of a line with a ``tower_type`` are lattice towers
loaded by the wind (``erf.conductors.tower_types``, a block of its own per
type). A tower stands on the terrain under its suspension point: a square
lattice body tapering linearly from ``base_width`` at the ground to
``top_width`` at the cross-arm (with an optional ``peak`` above it at the top
width), and a lattice cross-arm of ``arm_length`` and face depth
``arm_depth`` centred on the body at the height the conductor hangs from,
across the line (normal to the mean horizontal direction of the spans either
side). The towers carry the wind's drag on their members and the pull of the
line hanging from them, and stand on a foundation of four footings; they are
rigid unless their type has a ``frequency`` (see `Moving towers`_).

A circuit's phases and its shield wire hang from one row of towers: a line
with ``share_towers = <line>`` hangs from the towers of that line, which has
the ``tower_type``, each at its own point on them, its ``towers`` points. The
phases sit across the cross-arm and the shield wire on the ``peak`` above it;
a point off the tower (further from its axis than half the cross-arm, or above
its top) stops the run naming the line and the tower. A shared tower stands
under the owning line's point, and the other lines' points are placed above
its base, not above the ground under each point, so that a cross-arm on a slope
stays level. The tower takes each line's pull at its own point, the
foundation all of them, and the lines that share a moving tower step with it
together.

Each member is cut into drag nodes at the middle of equal segments
(``segments`` up the body, four along the arm). A node stands for a length
:math:`L` of member with axis :math:`\mathbf{e}`, and the wind loads it as a
slender member does, by the flow normal to it, relative to the node's own
velocity :math:`\mathbf{v}` (zero for a rigid tower):

.. math::

   \mathbf{F} = \tfrac{1}{2} \rho\, C_f\, w\, L\, |\mathbf{U}_n| \mathbf{U}_n ,
   \qquad \mathbf{U}_n = (\mathbf{U} - \mathbf{v}) - \big((\mathbf{U} - \mathbf{v})\cdot\mathbf{e}\big)\mathbf{e} ,

with :math:`w` the face width times the ``solidity`` :math:`\phi` (the
members' area over the face's outline) and :math:`C_f` the ``drag_coefficient``
or, by default, the force coefficient of a square lattice tower of
flat-sided members on the projected area of one face,
:math:`C_f = 4\phi^2 - 5.9\phi + 4` (ASCE 7), which counts the windward and
the leeward face together: 2.98 at :math:`\phi = 0.2`. A rotor-less tower in
AeroDyn is the same law with :math:`w` the tower's diameter. The wind is
ERF's velocity sampled at the nodes with the lines' sampler at the start of
each step, or the prescribed velocity; a node just above sloping ground,
below the averaged bottom face of its cell, is read from the bottom cell. The
aerodynamics sit behind a narrow interface (node positions, axes and
velocities and the wind in, a force per node out), so another model of the
members' loads can stand in for the ERF drag.

The line pulls on the tower where it hangs from it: the net force MoorDyn
finds on the top node of the insulator string there, or, for a conductor
clamped to the tower, on the end nodes of the two spans meeting there. In
still air a tower of a level section carries the weight of one span (half of
each span either side) and its string, the spans' pulls along the line
cancelling; in a crosswind it also takes the wind on that length, the wind
span. Drag and pull together load the foundation: the base shear (the
horizontal force), the overturning moment about the centre of the base and
the downward load :math:`P`, the tower's ``weight`` included. The four legs
stand at :math:`(\pm s/2, \pm s/2)` in the frame of the line and the
cross-arm, :math:`s` the ``leg_spacing`` (the base width by default), and,
as a rigid square, share :math:`P` equally and the overturning moment
:math:`\mathbf{M}` linearly,

.. math::

   N_i = \frac{P}{4} + \frac{M_y x_i - M_x y_i}{4 (s/2)^2} ,

compression positive, so that the reactions balance the loads. The largest
compression and the largest uplift (the most negative :math:`N_i`) are
flagged against ``allowable_compression`` and ``allowable_uplift``, a
footing's bearing and pull-out capacity (0, the default, is not checked). A
pull along the diagonal loads the corner leg :math:`\sqrt{2}` times more
than the same pull face-on.

Every ``diagnostics_int`` steps ``<diagnostics_dir>/towers.dat`` gets, for
each tower ``<line>_t<k>``, the drag ``drag_Fx``, ``drag_Fy``, ``drag_Fz``,
the line's pull ``line_Fx``, ``line_Fy``, ``line_Fz``, the ``shear``, the
``overturning`` moment, the ``vertical`` load, the ``max_compression`` and
``max_uplift`` of the legs and the allowable flag ``over``, at the time the
step starts (the flow the drag was found from and the line as it was then);
from ``stats_start`` each tower keeps the statistics of its horizontal drag
and pull, shear, overturning moment, leg compression and uplift and the flag
in ``<diagnostics_dir>/tower_<line>_t<k>_stats.csv``. The start-up log prints
each tower's still-air pull and leg loads. With ``drag_on_flow`` the
towers' drag goes into the momentum sources with the lines', and
``total_load.dat`` counts both. A checkpoint carries the statistics and the
last node forces, so the drag a restart puts into the flow is the
checkpointed step's.

Verification (unit tests ``MemberDrag``, ``Tower``): a node takes only the
normal component of the relative wind (none along its axis, none moving with
the wind); in a uniform wind a 30 m tower of 6 m base, 1.5 m top width and
solidity 0.2 carries exactly :math:`q C_f \phi (b_0 + b_t) H / 2` on its body
(the width is linear, so the segments' midpoint sum is exact) and
:math:`q C_f \phi d L_a` on its cross-arm, and the base moment equals the
hand integral up to the midpoint sum's known :math:`H^3/(12 n^2)` on the
quadratic part; wind along the cross-arm loads only the body; in a log-law
wind ten segments give the drag and base moment of a 20 000-point integral
within 1 %. The legs' reactions are in equilibrium with the loads for any
orientation, face-on and diagonal pulls give the hand values, and each
allowable flags on its own. Against the real MoorDyn, the middle tower of four
300 m spans on 2.5 m strings carries 5403.6 N down in still air against the
weight span's 5400.1 N; in a steady 20 m/s crosswind it takes 2149 N across
the line against :math:`qL` plus the swung string's drag, 2154 N, and 5343 N
down against the weight span less the string drag's lift, 5350 N. (The string,
swung :math:`\theta` across the line, sees the normal wind
:math:`U\cos\theta`, along :math:`(\cos\theta, \sin\theta)` in the plane
across the line, so its drag :math:`q_i L_i \cos^2\theta` lifts as well as
pushes.) A tower next to a dead end differs: the span from the dead end
drops to the string's bottom, and the lower end takes less of its weight. On the terrain test case the hilltop tower in a 25 m/s wind
carries 25.6 kN, against 25.1 kN from :math:`q C_f \phi` times its face at
that speed.

Moving towers
-------------

A tower type with a ``frequency`` (Hz, its first bending frequency on a rigid
foundation) bends under its loads, and its line hangs from a cross-arm that
moves: MoorDyn takes the cross-arms as coupled points, which ERF moves and on
which MoorDyn hands back the line's pull, the way OpenFAST couples MoorDyn to
a platform. The tower is a single mode, the same in :math:`x` and :math:`y`
(a square tower), on a foundation that can tilt and slide
(``foundation_rotational_stiffness`` :math:`k_r` in N m/rad and
``foundation_lateral_stiffness`` :math:`k_l` in N/m, rigid when 0, the
default), with the structural ``damping_ratio`` :math:`\zeta` (0.02 by
default). Its mass, ``weight`` over gravity, is spread over the drag nodes by
the length of member each stands for. A node at height :math:`z` above the
base moves horizontally by :math:`\varphi(z)` times the cross-arm's
displacement :math:`q`,

.. math::

   \varphi(z) = a_b \left(\frac{z}{H}\right)^2 + a_r \frac{z}{H} + a_l ,
   \qquad a_b = \frac{1/K_b}{c},\; a_r = \frac{H^2/k_r}{c},\; a_l = \frac{1/k_l}{c},\;
   c = \frac{1}{K_b} + \frac{H^2}{k_r} + \frac{1}{k_l} ,

the cantilever's bending, the footing's tilt and its slide, each in
proportion to its compliance under a load at the cross-arm (height
:math:`H`). The generalized mass is :math:`M = \sum m_i \varphi_i^2` and the
stiffness :math:`K = 1/c`, the strain energy of that shape exactly, with the
bending stiffness :math:`K_b = (2\pi f)^2 \sum m_i (z_i/H)^4` that gives the
type's frequency :math:`f` on a rigid foundation; a foundation that gives
lowers the frequency to :math:`\sqrt{K/M}/2\pi`. In each horizontal direction

.. math::

   M \ddot q + 2 \zeta \omega M \dot q + K q = Q ,
   \qquad Q = \sum_i \varphi_i F_i + F_{\rm line} ,
   \qquad \omega^2 = K/M ,

with :math:`F_i` the members' drag and :math:`F_{\rm line}` the line's pull
at the cross-arm (:math:`\varphi = 1`). A load held over a step is
integrated exactly, so any step is stable. The tower does not twist, rise or
sink, and its weight does not add to the overturning as it leans
(:math:`P`-:math:`\Delta`).

A line whose towers move steps in coupling steps, at least ``substeps`` and
at least 20 over the period of its fastest tower; lines that share towers step
together. In each, every tower advances under its members' drag and the mean of
the lines' pull at the start and at the end of the coupling step; then MoorDyn
moves each line's points on the towers from where they were to where the towers
have taken them, at a constant velocity over the call (MoorDyn moves a coupled
point linearly), and hands back their pull. The pull at the end is not known
before MoorDyn's call, so the coupling step is iterated: the lines' MoorDyn
states and the towers' are kept at its start and restored for each iteration
(MoorDyn's in-memory serialization), and the estimate of the end pull is
updated with Aitken's relaxation until it changes by less than
:math:`10^{-4}` of the largest pull (at most 50 iterations). A single exchange,
the towers stepping with the pull from the end of the last call, is only
conditionally stable: a short, nearly taut span pulls back in milliseconds,
and a pull that lags a coupling step acts on the tower as negative damping,
:math:`k h / 2` for a span of stiffness :math:`k` and a step :math:`h`, which
outgrows the tower's own damping once :math:`k` exceeds the tower's stiffness.
Iterated, the coupling is the implicit one; it takes about four iterations
where the towers carry strings and a clamped shield wire, and one once they are
still. ``<diagnostics_dir>/coupling.dat`` logs, every ``diagnostics_int``
steps, the most iterations a coupling step took and how many did not converge
(the run warns once and goes on with the last iterate). The members' drag is
found again in every coupling step from the wind of the ERF step and the
members' current velocity, so the wind damps the sway: a member moving with the
wind feels less of it. The foundation takes the loads
less the inertia of the nodes (each node's mass times its acceleration), so
a tower swaying freely still loads its footings. MoorDyn's pull on a coupled
point leaves out the inertia of the line's end node, a few kilograms against
the tower's tonnes.

``towers.dat`` gets the cross-arm's displacement ``arm_dx`` and ``arm_dy``
for each moving tower, and its statistics the size of that displacement,
``arm_displacement``. The start-up log prints each moving tower's frequency
on its foundation, generalized mass and stiffness. A tower starts upright and
at rest, so a run that starts in a wind rings down from the sudden load, over
about :math:`1/(\zeta\omega)`: 4 s at 2 Hz and 2 %. A checkpoint carries each
tower's sway, and the line restarts with its cross-arms where the towers had
taken them. The tower model sits behind a narrow interface (the node loads
and the line's pull in, the motion of every node and of the cross-arm and the
nodes' inertia out), so a frame model of the lattice can stand in its place.

Verification (unit tests ``OneModeTower``, ``ConductorSpan``,
``Conductors``): a load held over a step is integrated exactly however the
time is cut, and a step of a thousand periods lands on the static
deflection; released, the tower rings down with the damped period to
:math:`10^{-4}` and the logarithmic decrement of its damping ratio to
:math:`10^{-3}`; a load :math:`F` at the cross-arm deflects it by :math:`Fc`
with the foundation's compliances in series, and a load spread up the body
by the hand value of the midpoint sum of :math:`p (z/H)^2`; swaying freely,
its base shear is :math:`\omega^2 q \sum m_i \varphi_i`. Swaying in a steady
20 m/s wind, a 30 m tower at 2 Hz with 0.5 % structural damping rings down at
0.9047 % against the 0.5 % plus the 0.4047 % of the linearised relative drag,
:math:`\sum \varphi_i^2 \rho C_f w_i L_i U / (2 M \omega)`. Against both
MoorDyn libraries, the pull MoorDyn hands back on a coupled point is the one
the tower is reported to carry, a coupled step moves the cross-arm by its
velocity times the step, and a coupled point held still is a fixed one to
:math:`10^{-9}` m. In a 10 m/s crosswind the towers of a clamped section
settle 2.6 mm downwind, where their stiffness balances the 4023 N of drag on
the body (weighted by the shape) and the line's pull of 672 N; that pull is
within 2 % of the rigid towers'. A circuit's shield wire clamped 40 m from
its dead end to the peak of a tower at 2 Hz with 2 % damping, in the same wind,
settles at 4 mm when the coupling steps are iterated; with a single exchange per
coupling step the tower is still growing through 18 mm after 15 s. On the moving-towers test case the towers
bend at 1.67 Hz (2 Hz lowered by footings of :math:`10^9` N m/rad) and the
hilltop tower leans about 3 cm in its 25 m/s wind.

A network over hills
--------------------

``Exec/CanonicalTests/PowerLines`` puts the pieces together: ``make_case.py``
draws four hills of 87 to 99 m on a 3 km by 2 km domain, six transformers
(three on hilltops, three on flat ground) and the five lines of a minimum
spanning tree between them, each a section of three spans on insulator
strings with lattice towers placed for ground clearance, all strung to 20 kN, and a
neutral log-law inflow of 18 m/s at 30 m. A tower in a dip, which the spans
either side would pull up, is raised until the line weighs on it: four of the
ten stand 36 to 46 m tall rather than 30 m. The k-equation RANS flow spins up
for 600 s (1.2 million cells), then the lines run 120 s from its checkpoint
with the real MoorDyn. The wind 30 m above the ground speeds up to about
25 m/s over the hilltops and slows in their lee. The lines settle within
about 30 s, every span at 20 to 22 kN: the spans running across the wind blow
out up to 2.3 m at mid-span (swings of 12 to 22 degrees), the one running along
it barely moves (2 degrees), and the transformers' horizontal loads settle
between 10.8 kN (three lines from different sides) and 22.1 kN, under their
25 kN allowable. The lattice towers (6 m base, 1.5 m top, solidity 0.2, 12 m
cross-arm) carry 18.7 to 20.6 kN of wind drag on the hilltops, 10.5 to 15 kN on
the raised towers and about 7 kN on the 30 m towers on flat ground. With the
line's pull and a 60 kN tower weight on 6 m footings, the towers on the hills
and the raised ones lift a footing by 11 to 22 kN (the allowable is 50 kN),
their worst legs carrying 42 to 53 kN in compression against 29 kN on flat
ground. The towers bend, at 1.67 Hz (2 Hz on footings of :math:`10^9` N
m/rad), and settle leaning 20 to 32 mm downwind, 10 mm on flat ground. With a
steady RANS wind the lines hold a steady blowout; their gust response needs a
turbulent inflow.

The same lines in a turbulent wind (``les/`` in the same directory): a periodic
precursor over flat land (Deardorff LES, 16 m cells, 3072 by 1536 by 768 m,
:math:`z_0` = 0.1 m, :math:`u_*` about 1 m/s) spins up for 7200 s and writes
boundary planes every 1.5 s; a run over three immersed hills restarts from its
checkpoint and takes its inflow from the planes, with three circuits (three
phases and a shield wire each, 12 lines) on five shared towers that bend. Over
1500 s of statistics the wind at the conductors' middles peaks about 1.3 times
its mean (16.6 and 21.4 m/s on the line across the wind); that line swings 15
degrees on average and 22 at its peaks, the lines along the wind 3 and 6; the
conductors' tension moves by about 1 %, the strings taking up the swing; the
towers' sway and base shear peak at 1.4 to 2 times their means, up to 35 mm and
20 kN; and the footings' uplift varies most, a tower lifting 9 kN on average
reaching 33 kN in a gust. Each coupling step converges in at most four
iterations.

Restart
-------

A checkpoint carries the lines under ``<chk>/conductors``: each line's whole
MoorDyn state (node and free-point positions and velocities, internal forces
and the time integrator's state, through MoorDyn's own save), the running
statistics of every span, string set, pair of lines, transformer and tower,
each moving tower's sway, the step count and the time. On a restart the line is created from the same
inputs, initialised without the initial-shape solve and given that state, so
it continues blown out exactly where the checkpoint left it, on MoorDyn's
clock; the statistics go on accumulating, and the diagnostics continue on the
step count of the original run. With ``drag_on_flow`` the restored lines' drag
is spread into the momentum sources again at once, so a plotfile written at
the restart shows the source of the checkpointed step. The span and string
logs, ``<output_root>_nodes.dat``, ``total_load.dat``, ``separation.dat`` and
``transformers.dat`` are appended to after the rows a run wrote beyond the
checkpoint time are dropped, so a run that went on past its last checkpoint
and is restarted from it leaves no duplicated stretch; ``towers.dat``, whose
rows carry the time a step starts at, loses the row at the checkpoint time as
well, since the restarted run writes it again. The lines of a restart must be those of the run that
wrote the checkpoint, in the same order and with the same spans, towers,
segments and strings, and the same transformers and tower types; anything else stops the run naming the line or the transformer. A checkpoint without
conductor state, from a run without spans, starts them afresh from their
still-air shape, with MoorDyn's clock at zero at the restart time.

Checks at start-up
------------------

A run with spans refuses to start with any ``amrex.fpe_trap_*`` input on
(MoorDyn's initial-condition solver overflows an intermediate value), with an
anchor level that is not a level of the run, with a span whose length does not
exceed the distance between its attachments placed on the terrain, with a length missing for a span of a section, with
insulator strings on a line without towers or longer than a tower is high,
with a line end on two overlapping transformers or not above the top of the
transformer it ends on, with a transformer no line ends on, or with any other
value outside its documented range. Every message names the input key.
