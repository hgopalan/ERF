.. role:: cpp(code)
   :language: c++

.. _sec:FireSuppression:

Suppression
===========

Suppression actions change where the fire can spread while the run is
going: fire lines built at a rate, retardant drops that stop or slow the
spread for a while, a hold test that lets a line fail under a hot enough
fire, and burnout ignitions fired from a built line. They are read from an
action file that the run watches, so a run can be steered while it is
running and reproduced afterwards from its log. Everything is off by default
and switched on with :cpp:`erf.fire.suppression.enable`.

Mechanism
---------

Both propagation paths (:ref:`sec:FirePropagation`) already read one
non-burnable mask, ``fire_nonburnable``, whose cells get zero rate of
spread, a level set held outside the burned region, and no FARSITE target;
structures, non-burnable fuel codes and masked firebreaks feed it at
start-up. Suppression adds a second, time-dependent source. Once per fire
step, before the rate of spread is used, the active actions are stamped
into a suppression mask on the fire grid and ``fire_nonburnable`` is rebuilt
as the OR of the static mask and that one; the level set is then clamped in
the mask cells as it always is. When the run has no static source, an
all-zero mask is allocated so that the suppression cells have somewhere to
go (an all-zero mask changes nothing on either path). The suppression
fields are rebuilt from the active actions on every step, so an expired
action's cells revert by themselves.

A retardant drop with a rate factor between zero and one does not enter the
mask. It fills a second field, ``fire_ros_factor``, with which ``fire_ros``
is multiplied just before the front update, after the acceleration ramp and
the crown-fire enhancement. The level-set paths that rebuild the rate inside
every Runge-Kutta stage (the directional, Balbi and hybrid paths) take the
factor through the per-stage scale they already apply, so the reduction
reaches them too.

**Lines.** A line is a polyline built from its first vertex at
:math:`\mathrm{rate}` metres per second from ``start_s``. A cell is covered
when the built part comes within half a fire-cell spacing of its centre or
passes through the cell's interior, so the built cells form a barrier one
cell wide that no four-neighbour path crosses, whichever way the line runs
(the FARSITE front update reads four neighbours; the level-set stencils are
wider, see the limitations). ``expiry_s`` removes the line that many
seconds after its start; ``-1`` keeps it.

**Drops.** A drop is a polygon. With ``ros_factor = 0`` its cells enter the
mask; with :math:`0 < f < 1` the rate of spread inside it is multiplied by
:math:`f`. Both last ``expiry_s`` seconds after ``start_s`` (``-1`` for
ever) and then revert.

**Burned cells are never stamped.** A line through a burning cell or
retardant on burned ground does nothing to the front, so a cell with an
arrival time, or a negative level set, is skipped when an action is applied
and counted out of the action's cell count. A line started after the front
has passed therefore has a gap where it crossed the fire, and the head runs
on through it: ``Exec/RegTests/FireSuppression`` ``inputs_line_late`` shows
this.

**Hold test.** A line with a ``flame_limit_m`` is tested every step against
the flame length of the previous step: a built cell with a four-neighbour
whose flame length exceeds the limit reverts to burnable, stays out of the
line for the rest of the run, and is logged as ``hold_failed``. The plotted
mask shows such cells as :math:`-1`. ``-`` means the line always holds.

**Burnout.** A burnout follows a line (``ref=<line id>``) and, from its
``start_s``, ignites one point per fire cell along the part of the line that
is already built, ``offset`` metres from it on the side of the nearest
burning cell (the side is decided per segment of the polyline). Each point
is a scheduled ignition of radius three quarters of a cell, applied by the
same machinery as the ignition schedule, and only where the level set is
still positive. As the line is built further the burnout follows it. When
no cell is burning yet the burnout waits and tries again on the next step.

**Reverted cells.** When a drop expires or a hold test fails, the cells go
back to burnable with whatever level set the clamp left them, zero where
the front was pressing against them. On the level-set path the next advance
moves the front through them at the rate of spread and the periodic
reinitialisation repairs the distance function around them; on the FARSITE
path the front-cell update ignites a reverted cell whose neighbour has been
burning, with the head-rate distance accumulated since that neighbour
ignited, so a front held at a barrier for a while moves into the released
cells at once. ``inputs_drop_hold`` measures both.

Action file
-----------

One action per line, whitespace separated, ``#`` comments, geometry in
fire-grid metres, every id unique::

   # id  type     start_s  geometry                    param            expiry_s  flame_limit_m
   L1    line     3600     500 200 900 600 950 650     rate=0.5         -1        2.4
   D1    drop     5400     poly:x1,y1;x2,y2;x3,y3      ros_factor=0.0   1800      -
   B1    burnout  4000     ref=L1                      offset=20        -         -

- ``line``: the vertices as a flat list (at least two), then
  ``rate=<m/s>`` (:math:`> 0`), ``expiry_s`` (:math:`-1` or :math:`> 0`) and
  ``flame_limit_m`` (``-`` or :math:`> 0`).
- ``drop``: one ``poly:`` token with at least three ``x,y`` vertices separated
  by ``;``, then ``ros_factor=<f>`` in :math:`[0, 1)`, ``expiry_s`` and ``-``
  (the hold test applies to lines only).
- ``burnout``: ``ref=<line id>`` (a line in the file or read earlier), then
  ``offset=<m>`` (:math:`\ge 0`) and ``-`` ``-``.

A malformed line stops the run at the read with a message that names the
file, the line number, the reason and the line itself; so does a duplicate
id and a burnout that references an unknown line. ``start_s`` earlier than
the current time applies the action at once.

Polling and reproducibility
---------------------------

The file is read on the IO rank at start-up and, when
:cpp:`erf.fire.suppression.poll_interval` is positive, every that many fire
steps: the rank compares the file's modification time and size with those
of the last read, and when either changed reads the text and broadcasts it,
after which every rank parses the same bytes. Ids already known are
skipped, so an action is never applied twice, and an edited line of a known
id has no effect: to change an action, add a new id (the old one expires
when its ``expiry_s`` passes, or holds for ever). ``poll_interval = 0``
reads the file once at start-up.

Every event is appended to :cpp:`erf.fire.suppression.log` as
``time_s,step,id,type,event,cells,detail``: ``applied`` when an action
starts (with the cells it covers and its parameters), ``completed`` when a
line is fully built, ``expired`` when a timed action is removed (with the
cells that revert), ``hold_failed`` with the number of newly failed cells,
``burnout`` with the number of ignition points placed and the side taken,
and ``reread`` when a poll found new actions. The time is that of the start
of the fire step the event was applied in. A fresh run starts the log over;
a restart appends to it. Together with the action file the log gives every
action and the step it took effect, so a run that was steered live can be
rerun with ``poll_interval = 0`` and a file holding the same actions with
the logged start times.

Checkpoint and restart
----------------------

The checkpoint carries the suppression mask, the rate factor, the line
progress and the hold-failed cells (``FireSuppressionMask``,
``FireSuppressionFactor``, ``FireSuppressionProgress``,
``FireSuppressionFailed``) and, in ``FireSuppression``, every action with
its geometry and state: whether it was applied, expired or completed, the
built length of a line, the length of a line a burnout has fired along, and
the last cell count. A restart continues a line where it was and reapplies
nothing. The file is re-read at start-up as always, but for ids the
checkpoint knows the checkpoint's copy wins; only new ids are added.

Output
------

The fire plotfile gains ``fire_suppression_mask`` (1 in cells suppressed
on this step, :math:`-1` where a line cell failed its hold test and nothing
else covers it, 0 elsewhere), ``fire_ros_factor`` (the multiplier, 1
outside drops) and ``fire_line_progress`` (the ordinal, in file order, of
the line a built cell belongs to; 0 elsewhere). ``fire_nonburnable`` is
present whenever suppression is on, as the combined mask.

Inputs
------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - Input
     - Meaning
     - Default
   * - ``erf.fire.suppression.enable``
     - Read and apply the action file
     - false
   * - ``erf.fire.suppression.file``
     - The action file; required when enabled
     - none
   * - ``erf.fire.suppression.poll_interval``
     - Fire steps between modification-time checks; 0 reads once at start-up
     - 10
   * - ``erf.fire.suppression.log``
     - The event log
     - ``suppression_log.csv``

Verification
------------

``Exec/RegTests/FireSuppression`` runs every scenario on both propagation
paths on a still-air deck with a prescribed rate of spread of 1 m/s and 2 m
fire cells, so every arrival time follows from the geometry: a line built in
time stops the head and the same line started late is overrun; a drop with
``ros_factor = 0`` holds until its expiry and the head crosses afterwards; a
drop with ``ros_factor = 0.3`` slows the head to the expected distance; a
line with a low flame limit fails where the hotter cells of a rate gradient
arrive and holds elsewhere; a burnout fires the strip along a line long
before the main front; a writer process appends the line to an empty file
while the run polls, on one rank and on two; and a restart with a line under
construction reproduces the straight run's fire plotfile in every field.
Each check fails on a binary without the feature (the plotfile lacks the
suppression fields). The action-file parser and the geometry kernels have
their own unit tests (``Tests/Unit/Fire/ERF_GTestFireSuppression.cpp``).

Limitations
-----------

- Lines are one cell wide. The FARSITE front-cell update, which reads four
  neighbours, cannot cross one; the level-set gradient stencils read up to
  three cells across a line, and with strong spread a small amount of the
  level set can leak past a one-cell line before the clamp catches it. The
  regression decks hold on the level-set path with the default stencil; for
  a production case set :cpp:`erf.fire.levelset.wall_extrapolate` so the
  line acts as a wall of the stencil.
- There is no direct attack that walks along the front, no per-fuel
  production rate or crew type, and no coupling of lines to structures.
- The FARSITE ``legacy`` front update is not tested with the rate factor;
  the run warns once when both are set.
- The hold test compares the flame length of the previous step, so a line
  fails one step after the flame exceeded the limit.
- Actions are geometry in fire-grid metres; there is no map projection.
