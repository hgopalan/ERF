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
OpenFAST and the only rank that holds the turbine's node positions and loads;
each point goes from there only to the ranks whose boxes it touches (see
"Farms" below). The OpenFAST model must set ``CompInflow = 2``
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
mesh type. A node is sampled by the rank whose box holds its containing cell,
reading neighbours from the ghost cells, so the result does not depend on the
domain decomposition; a node outside the domain aborts the run with its
coordinates. The owner of an OpenFAST turbine or a prescribed-Ct disk sends
each of its points to the ranks whose boxes may hold it and gets the velocity
back from the one that does; the wake lines behind a rotor are sampled the same
way for the rotor's owner.

Force spreading and momentum source
-----------------------------------

The force a body exerts on the fluid is carried by actuator points (the rings
of an OpenFAST rotor's actuator disk or the nodes of its actuator line, its
tower force nodes and nacelle drag point, the disk points of a prescribed-Ct disk)
and spread onto ERF's face-centred momentum sources with a 3-D Gaussian kernel of width
``epsilon`` (given in units of dx), cut off beyond three widths. Every point has its own width:
each body its own ``epsilon``, and an OpenFAST turbine's tower force nodes and nacelle drag point
their own ``epsilon_tower`` and ``epsilon_nacelle`` (``epsilon`` unless given; see "Kernel widths"
below). The kernel is
normalised discretely on each staggered grid, with the face volumes
(``dx dy dz detJ`` on a terrain-following mesh), so the source integrates back
to the point force exactly for every point and component, whatever the
resolution and however much of the kernel the ground or the domain top cuts
off; the kernel's shape is Gaussian only where it is resolved. A kernel cut by
the ground is renormalised over what remains, which moves the centre of its
force up: for the lowest blade tip of the IEA 15 MW rotor (30 m above the
ground) by 0.6 m with ``epsilon = 20 m``, 7.5 m with 40 m and 17 m with 60 m.
The force is kept, but the moment the spread source exerts about the hub
differs from the point loads': the whole rotor's torque by 0.4 % at 40 m and
1.7 % at 60 m. Each point is
visited over the faces within three widths of it, and again as its periodic
images, so a kernel wraps across a periodic boundary and the cost grows with
the number of points and the cube of their widths, not with the mesh. Faces on a non-periodic domain
boundary get no source. The normalisation of a turbine's kernel is the sum of
the parts the ranks its kernel reaches compute on their own boxes, added by
the owner of the turbine or disk in rank order and sent back with the force. The sources are computed once per step, from the
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
   * with ``epsilon`` below about three quarters of a cell the cell sum of a
     Gaussian is no longer its integral, so the injected momentum depends on
     where the point sits in its cell: the 3-D sum over the analytic volume
     lies within 0.03 % of 1 at one cell, between 0.98 and 1.02 at 0.75 cells
     and between 0.6 and 1.6 at half a cell;
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
samples at the rotor contain the rotor's induction once its loads act on the
flow, so with ``sampling = disk`` AeroDyn's own wake model must be off, and with
the two samplings that hand OpenFAST the free stream (below) it must be on: ERF
reads the AeroDyn file named in the ``.fst`` and refuses to start when its
``Wake_Mod`` is not 0 for ``disk``, or is 0 for ``upstream`` and
``disk_corrected``, which need an induction model on (``Wake_Mod = 1``, BEMT;
``mode = none`` leaves it to the model).
The thrust in the diagnostics is the sum of the hub and blade node forces, so
its shaft component equals minus that of the integrated source. With ``mode = none``
the turbine is driven by the flow but puts no force into it (one-way
coupling, as for a loads analysis in a precomputed flow).

**Where the velocities are sampled.** With ``sampling = disk`` the
node velocities handed to OpenFAST are sampled at the nodes, inside the rotor's
own induction zone, and AeroDyn's induction model is off (``Wake_Mod = 0``): the
resolved flow supplies the induction. On a grid that does not resolve the smeared
disk (``epsilon`` of two 20 m cells for a 240 m rotor) the sampled velocity lies
well above the momentum-theory disk velocity, and the loads, computed by OpenFAST
without induction from that velocity, exceed the blade-element-momentum values
(the IEA 15 MW disk on 20 m cells with a two-cell kernel measured 33 % high in
power, 29 % with a 1.5-cell kernel). With
``sampling = upstream`` every velocity node is shifted ``sample_diameters_upstream``
diameters ahead of the hub along the horizontal projection of the shaft axis (so
a tilted shaft keeps every node at its own height), where the flow is close to
the free stream (in vortex-cylinder theory the rotor slows the flow one diameter
ahead by 9 % of its induction averaged over the disk, two diameters ahead by
3 %), and
AeroDyn keeps its induction model (``Wake_Mod`` must not be 0; 1 is BEMT): the loads are
then the BEM loads for that free stream whatever the grid, and the resolved flow
receives them as rings. The shifted points keep their absolute height, so on a
slope they may come closer to the ground, or below it, than the nodes; and a
rotor yawed out of the wind samples off to the side, two diameters times the
sine of the misalignment. The shifted sampling points must be covered by the anchor
level like the nodes. With ``sampling = disk_corrected`` (the default for
``mode = adm``) the velocities stay at the nodes and the free stream is recovered
from them: the velocity a Gaussian-smeared
disk samples is not the thin-disk velocity :math:`U_\infty (1 - a)` but that
velocity divided by the factor of Shapiro, Gayme and Meneveau (2019, eq. 25),

.. math::

   M = \left( 1 + \frac{C_T'}{4} \frac{\Delta / R}{\sqrt{3 \pi}} \right)^{-1},
   \qquad C_T' = \frac{4 a}{1 - a} = \frac{C_T}{(1 - a)^2},

where :math:`\Delta` is the width of their filter
:math:`G = (6 / \pi \Delta^2)^{3/2} \exp(-6 r^2 / \Delta^2)`; ERF's kernel
:math:`\exp(-r^2 / \epsilon^2)` is that filter with :math:`\Delta = \sqrt{6}\,\epsilon`.
Their factor is for the disk velocity filtered twice, once by the force kernel
and once more by a sampling kernel of the same width. ERF samples point values
at the blade nodes, which are filtered once; the same derivation then gives the
coefficient :math:`1/\sqrt{6\pi}` instead of :math:`1/\sqrt{3\pi}`. ERF keeps
the twice-filtered form: with it the corrected disk matches the standalone BEM
loads within 1 % (the table below), while the once-filtered form recovers a free
stream 3 to 4 % high on 20 m cells. The agreement is therefore a calibration
against BEM, not a consequence of the theory alone.
Each step the radius-weighted axial velocity over the blade nodes :math:`u_d` is
sampled (each node weighted by its radius times its share of the span),
:math:`C_T` is evaluated from the previous step's shaft thrust and previous free
stream (clamped to 0.96), :math:`a = (1 - \sqrt{1 - C_T}) / 2`, and the hub and
blade node velocities are scaled by :math:`U_\infty / u_d` with
:math:`U_\infty = M u_d / (1 - a)`; AeroDyn then applies its own induction
(``Wake_Mod`` must not be 0; 1 is BEMT). The tower nodes keep the resolved flow. The
recovered free stream is checkpointed and logged in ``<output_root>_correction.csv``;
the factor was derived for :math:`\Delta / R` up to about 1.25, and a wider kernel
is warned about: keep :math:`\sqrt{6}\,\epsilon \le 1.25 R`, i.e. :math:`\epsilon \le 0.5 R`,
so on cells coarser than a quarter of the radius use a kernel narrower than the
usual two cells (1.5 cells of 40 m for a 120 m rotor). The 0.96 clamp is reached
when the rotor's own thrust coefficient is that high (a rotor at high tip-speed
ratio, below rated at fixed speed, or in a wake), not because of the kernel;
:math:`a` then stays at 0.4, the recovered free stream is low, and the first
clamped step prints a warning. The actuator line keeps disk sampling: it
resolves its own induction, and the filtered lifting-line correction accounts
for the kernel.

**Why the update is under-relaxed.** The free stream is recovered from the last
one, through the thrust coefficient it implies. Once the flow has answered the
thrust, a small error in the previous free stream returns multiplied by the
loop gain

.. math::

   G = -\frac{2 a (1 - a)(1 - \delta)}{(1 - 2 a)(1 - a + \delta a)},
   \qquad \delta = \frac{\Delta / R}{\sqrt{3 \pi}},

which is negative (the update overshoots) and reaches :math:`|G| = 1` at
:math:`C_T \approx 0.75 + 0.11\,\Delta / R`. The gain does not depend on how
OpenFAST's thrust follows the speed it is fed. Above that threshold the plain
update alternates and grows from step to step until the clamp stops it. A
second mode comes from the flow itself: the disk velocity answers a change of
thrust only after a delay, as the induced flow builds up, and an update that
follows faster than that can ring at the delay's period; an 8 m LES of the
IEA 15 MW rang with a period of about 5 s. The update is therefore relaxed,
:math:`U_\infty^{new} = (1 - \omega)\,U_\infty^{old} + \omega\,U_\infty^{update}`,
which multiplies the one-step gain to :math:`1 - \omega (1 - G)`.

**What the new options do.** ``correction_relax = -1`` (the default) takes
:math:`\omega = 1 / (1 - G)`, clamped to [0.2, 1], from the current gain, which
cancels the step-to-step overshoot to first order, and holds :math:`\omega` at
most :math:`\Delta t / (\tau (1 - G))` as well, so an error of the estimate
decays as :math:`e^{-t/\tau}` whatever the gain, with :math:`\tau` the time scale
``correction_time`` (by default the rotor radius over the free stream, 11 s for
the IEA 15 MW at 10.6 m/s). In a model of the loop with the flow answering the
thrust after a delay and a lag, the gain factor alone left the free stream
ringing by 9 to 11 % for a 2 s delay (Ct 0.85, :math:`\Delta / R` 0.16); a bound
of :math:`\Delta t / \tau` without the gain still rang by 3 to 7 % at the 0.96
clamp for delays of 6 to 10 s; with :math:`\Delta t / (\tau (1 - G))` every case
settled, within 1 % in 8 to 34 s. The price is a slower start: from 30 % off the
free stream comes within 1 % in about 3.4 time scales (some 40 s for the
IEA 15 MW) instead of a few steps.
``correction_time = 0`` keeps the gain factor alone; a fixed
``correction_relax`` in (0, 1] is used as given, and 1 is the unrelaxed update.
The relaxation changes how the free stream is reached, not where it settles,
and the gain and the relaxation used are logged in
``<output_root>_correction.csv``.

The IEA 15 MW rotor in a uniform 10.59 m/s inflow (open boundaries, no sponge,
no turbulence closure) against the standalone OpenFAST BEM solution (aerodynamic
thrust 2.482 MN, power 15.79 MW, momentum-theory disk velocity 7.67 m/s):

================================================  =========  ========  =====================
Disk set-up                                       thrust     power     sampled velocity (m/s)
================================================  =========  ========  =====================
disk sampling, 20 m cells, kernel 1.5 dx          1.084      1.288     8.40
disk sampling, 20 m cells, kernel 2 dx            1.093      1.328     8.47
disk sampling, 20 m cells, kernel 3 dx            1.112      1.403     8.63
disk sampling, 20 m + 10 m anchor patch           1.095      1.316     8.64
disk sampling, 10 m cells, kernel 2 dx            1.072      1.243     8.36
upstream sampling 1 D, BEM on, 20 m and 10 m      0.977      0.954     10.43
upstream sampling 2 D, BEM on, 20 m               0.993      0.999     10.58
disk_corrected, 20 m cells, kernel 2 dx           0.997      1.000     8.74 -> 10.77
disk_corrected, 10 m cells, kernel 2 dx           1.001      0.993     8.44 -> 10.82
disk_corrected, 20 m cells, kernel 3 dx           0.991      0.994     8.99 -> 10.67
================================================  =========  ========  =====================

The ratios do not depend on the advection scheme (third- or fifth-order upwind,
WENO, or blended upwind give the same loads within 0.1 %) nor on the boundary
conditions (a periodic box changes them by 1 to 2 %). The one-diameter sample
sits 1.5 % below the free stream because of the rotor's own upstream induction,
and the loads follow that velocity squared and cubed; at two diameters the bias
is below 1 %, which is the turbine default. The corrected disk recovers the loads
within 1 % on every grid and kernel tried (filter widths of 0.40, 0.81 and 1.21
rotor radii) from the disk velocity alone, with no upstream points to cover: the
recovered free stream (the second velocity in the last three rows) sits 1 to 2 %
above the true value and the thrust coefficient it implies (0.76 against the BEM
0.80) correspondingly below, and the two biases cancel in the loads. In these
uniform-inflow runs (thrust coefficients 0.76 to 0.80, made with the unrelaxed
update) the correction settled from its start-up transient within about 50 s
with no oscillation.

Bodies on terrain
-----------------
On a terrain-fitted mesh the z of ``base_pos`` is the height of the body's base
above the terrain surface at its (x, y): the surface is the k = 0 node plane of
the mesh, interpolated bilinearly between the four nodes around the base, and
that height is added to the given z once at start-up, before OpenFAST is
initialised or restored (on a uniform-dz mesh nothing is added, so z is the
absolute height). The sampler and the spreader already work in the physical
heights of the fitted mesh (``z_phys_nd``) and with the cell volumes
(``detJ``), so the nodes, the rings and the kernel follow the terrain without
further change. The start-up audit measures the rotor's ground clearance from
the terrain height under the hub, and the vertical wake line of each rotor
stops at the terrain under its hub. The heights added are written to
``<diagnostics_dir>/ground.csv`` (body, x, y, terrain height, base z).

OpenFAST rotor as an actuator line
----------------------------------

With ``mode = alm`` the blade force nodes themselves are the actuator points:
after every OpenFAST step the positions it reports for the rotating, yawed and
deflected blades (``num_force_points_blade`` per blade, root to tip, plus the
hub node) carry minus its node forces into the Gaussian spreading, so the flow
sees the three moving lines of force and the tip and root vortices they shed,
which the disk averages away. The integrated momentum source equals minus the
full thrust vector, the torque about the shaft is preserved by the point loads
(see the ground-cut kernel above), and no in-plane force is lost. The kernel
width ``epsilon`` is given in units of the anchor level's ``dx`` but is best
chosen as a length at the scale of the chord: about 2 m for the IEA 15 MW with
the FLLC, which on 2.5 m cells is 0.8 cells. A kernel wider than the chord
smears the tip loading and raises the power for a given inflow: two 2.5 m cells
(5 m) over-predicted the power by 44 %. The line must not jump over cells between two
steps: at every step ERF computes how many cells the blade tip sweeps,
``rotor_speed * tip_radius * dt / min(dx, dy, dz)``, with the tip radius the
largest distance of a blade node from the hub axis and, on a terrain-following
or stretched mesh, ``dz`` the shortest physical cell height (the spacing of
``z_phys_nd``'s nodes) in the box of the hub plus and minus the tip radius,
not the nominal one, and aborts when it exceeds
``erf.moving_bodies.alm_max_tip_cells`` (1 by default), naming the largest
``erf.fixed_dt`` that passes. The count is a safe bound rather than an exact
one: it pairs the full tip speed with the shortest cell anywhere in the box,
although the shortest cells usually lie at the bottom of the disk where the tip
moves sideways, and it measures heights vertically, while on a slope the
spacing normal to the coordinate surfaces is a little shorter (41.5 m against
42.2 m in the rotor box of the Askervein test). The start-up audit's cells across
the rotor still use the nominal spacing. For the IEA 15 MW at rated speed (tip
speed about 95 m/s) one cell per step means
``dt <= min(dx, dy, dz) / 95``: 0.44 s on the 50 m Askervein mesh, whose cells
under the rotor are 42.2 m tall. The check runs every
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
(Martinez-Tossas and Meneveau, 2019, in the generalized form for blades whose
chord varies along the span of Martinez-Tossas, Sakievich, Churchfield and
Meneveau, Wind Energy 27:101, 2024), on by default with ``mode = alm``
(``fllc = false`` switches it off), computes the velocity the trailing vorticity of the line's own lift
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
with ``f = fllc_relax`` is a downwash (against the lift) when the line's own
kernel (``epsilon``) is wider than the optimal one, and vanishes when they coincide. It is
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
and ERF spreads each tower node's force on the fluid with a Gaussian kernel of
width ``epsilon_tower``, so the tower wake and its shadow on the rotor
appear in the flow. AeroDyn's own tower-shadow correction of the blade inflow
(``TwrShadow``) would then count the shadow twice, so ERF reads it from the
AeroDyn file and prints a warning when it is not 0. With ``nacelle_cd > 0``
and ``nacelle_area`` ERF adds one drag point at the hub node,
``-1/2 rho cd area |u| u`` on the fluid with ``rho`` the body's
``air_density`` and ``u`` the velocity sampled at the hub node (at the hub
itself also with ``sampling = upstream``, whose velocity nodes lie ahead of the
rotor), corrected for
the point's own kernel: a Gaussian force of width ``epsilon_nacelle`` induces a
velocity at its own centre that lowers the sampled one by the factor
``1 - cd area / (4 pi epsilon_nacelle^2)``, so the sampled velocity is divided by it
(the correction Kynema applies with its own nacelle kernel width). The
start-up audit aborts when the factor is under 0.5 (see "Kernel widths"). The
tower force and the
nacelle drag are written in the turbine diagnostics with the thrust, and the
integrated momentum source equals minus their sum.

Kernel widths
-------------

The width that suits a point depends on what it stands for, so each kind of
point has its own, in units of the anchor level's ``dx``, and each body has
its own set:

* ``epsilon``: the rotor's points (the blade and hub nodes of an actuator line,
  the rings of an actuator disk, a ``ct_disk``'s points). For an actuator line
  it is best of the order of the chord, about 2 m for the IEA 15 MW with the
  FLLC (see above); the FLLC and the ``disk_corrected`` filtered-disk factor
  both use it, as they correct for the width the rotor is spread with.
* ``epsilon_tower``: the tower force nodes. A tower node samples the velocity
  its own drag has already lowered: by the streamline argument that gives the
  nacelle's factor, a line of drag ``1/2 rho cd d U^2`` per unit length spread
  with a Gaussian of width ``eps`` slows the flow on its axis by the fraction
  ``cd d / (4 sqrt(pi) eps)`` in linear theory, about 11 % for an 8 m tower
  (Cd 1) at ``eps = 10 m`` and 6 % at 20 m; with the drag taken from the
  lowered velocity the tower drag comes out about 18 % and 10 % low. No
  correction is applied for it, so a narrow tower kernel costs tower drag:
  keeping the velocity deficit near 5 % (the drag about 10 % low) takes a
  width of about 2.8 ``cd d`` (some 23 m
  for that tower), far wider than an actuator line's blades want on fine
  cells. ``epsilon_tower`` lets the tower keep such a width while the blades
  are narrower. The start-up audit warns when the tower's force points lie
  farther apart than its kernel.
* ``epsilon_nacelle``: the nacelle drag point, and the width in its kernel
  correction ``1 - cd area / (4 pi epsilon_nacelle^2)``: for a 60 m^2 nacelle
  with Cd 1 it is 0.24 at 2.5 m and 0.81 at 5 m. The correction is a linear
  estimate: along the axis Bernoulli gives a sampled speed of
  ``U sqrt(1 - 2 cd area / (4 pi epsilon_nacelle^2))``, which stagnates where
  the linear factor reaches 0.5, and below 0.5 the drag fed back through the
  corrected velocity is no longer a small change. The start-up audit therefore
  aborts when the factor is under 0.5, i.e. when ``cd area`` exceeds
  ``2 pi epsilon_nacelle^2`` (the 2.5 m example aborts; about 3.1 m is the
  least for that nacelle). That limit is where the correction stops making
  sense, not a target: near it the linear factor and the Bernoulli speed part
  ways (the drag they give differs by 44 % at a factor of 0.6 and 5 % at 0.81).
  The velocity interpolated at the hub also sees only part of the deficit the
  correction removes, depending on where the hub sits in its cell: 60 to 100 %
  with the kernel one cell wide, 88 to 100 % at two cells, so the corrected
  drag can come out up to about twice the true one at the limit on one cell and
  25 % high on two. A factor of 0.8 or more on two cells or more keeps both
  effects to a few per cent; Kynema's choice ``sqrt(2 cd area / pi)`` gives a
  factor of 7/8 (6.2 m for that nacelle).

  The nacelle therefore usually wants its own width beside an actuator line:
  for the IEA 15 MW (``cd area`` of about 50 to 60 m^2) the 2 m blade width of
  the FLLC runs above leaves the factor between 0.02 and -0.19, which aborts, while
  ``epsilon_nacelle`` of two 2.5 m cells gives 0.81 to 0.84.

Unset, ``epsilon_tower`` and ``epsilon_nacelle`` take ``epsilon``, which
spreads every point of a body with one width. With ``mode`` adm or alm, a width
given for a tower or nacelle that is not forced aborts when its switch
(``num_force_points_tower``, ``nacelle_cd``) was never set, and warns when it
was set to zero; with ``mode = none``, which forces nothing, it warns. Bodies may
use different widths, a narrow actuator line beside a wide actuator disk, say;
their sources add on the same fields. The widths are part of the restart
record: a restart that changes one aborts, naming it.

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
the normal, which equals the thrust when the disk is the only body. The
upstream sample still feels the disk: in vortex-cylinder theory the
disk-averaged sample one diameter ahead reads 0.977 of the free stream at
``Ct = 0.75``, so the thrust is 4.6 % below nominal, and 1.4 % below at two
diameters. ``air_density`` must match ERF's density at the disk centre within
``erf.moving_bodies.density_tolerance``; the run aborts otherwise, since the
thrust is computed with one density and put into a flow of another.

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

On a run with refinement the bodies live on one level, the anchor: level
``amr.max_level`` unless ``erf.moving_bodies.anchor_level`` names a coarser one. Their
nodes are sampled from that level's velocities, their momentum sources are
spread onto that level's faces with the kernel width in that level's cells,
and OpenFAST is stepped with that level's fixed step (``erf.fixed_dt`` divided
by the sub-cycling ratios down to it), so a fine patch around the rotor gives
an actuator line its few-metre cells without refining the whole domain.
Coarser levels carry no source of their own: they see the rotor through ERF's
average-down of the state after every coarse step, as any fine-level physics
does (with ``erf.coupling_type = TwoWay``, the default; with ``OneWay`` the
coarser levels do not see the bodies at all). A finer level than the anchor may exist elsewhere, but not over the
bodies: the average-down would overwrite their force there, so before every
coarse step the run aborts if a finer level reaches a rotor, with its kernel's
reach, or a disk. Everything the bodies read or write on the level must lie on
its grids: at the first step and whenever the level's grids change, every node
with the kernel's reach, every disk point, and every wake sampling point is
checked against the anchor level's box union (periodic directions wrap), and a
point outside aborts naming it, since the sampler and spreader read only the
level's own cells. Enlarge the refinement region or choose a
coarser anchor. The refinement region must also let the FFT solver run on
the level, that is be a rectangular union of boxes, as for any anelastic run.

Farms
-----

Any number of bodies can be listed in ``erf.moving_bodies.bodies``; each has
its own input block, its own diagnostics files (``<output_root>_*.csv``) and,
for an OpenFAST turbine, its own OpenFAST instance.

**Which rank owns a turbine.** Each OpenFAST turbine is owned by one MPI rank,
chosen at start-up (and again on a restart, from that run's boxes), turbines
taken in the order of the list. Its footprint is the column of cells under its
base, widened by the rotor radius its model files give (``TipRad`` of the
ElastoDyn file, or the stub's ``rotor_radius``) and two cells. The candidates
are the least loaded ranks whose turbines step with the same OpenFAST ``DT``
(or own none). If one of them has boxes of the anchor level under the
footprint, the turbine goes there (to the one holding the most of it, then the
lowest); otherwise it goes to the candidate whose boxes come nearest. So the
turbines spread over the ranks as evenly as dealing them out would (one per
rank while there are as many ranks as turbines), they stand on their owner's
boxes wherever that costs no balance, and turbines of different time steps
share a rank only when there are more time steps than ranks (an empty rank is
kept for each time step that has none yet). Only the owner calls
OpenFAST, holds the turbine's nodes and loads, applies the sampling corrections
and the FLLC, and writes the turbine's logs, statistics and checkpoint files.
The run log names each owner, with the footprint radius and time step it read.
Each prescribed-Ct disk has an owner chosen by the same rule among the disks
alone, with its ``rotor_radius`` for the footprint and one time step for all:
a disk runs no model, so it may share a rank with a turbine without holding up
its OpenFAST steps. Only the disk's owner knows the flow at its points and its
thrust, and writes its ``_disk.csv`` and statistics. The wake lines behind a
rotor (a turbine or a disk) belong to the rotor's owner, which samples them,
keeps their running average and writes their files and checkpoint state; every
rank keeps their geometry, so a restart on other ranks reads the running sums
wherever the new owner is.

**What goes where.** Every step the owner sends each node to sample only to the
ranks whose boxes may hold its cell, and each point it spreads only to the
ranks whose boxes lie within the reach of its kernel; those ranks send back the
velocity at the node, then their part of the kernel's normalisation, and get
the force over the whole normalisation in return. A rank whose boxes no point
of a turbine reaches never receives any of its data (it still takes part in the
two integer reductions of each step and in the gathers below). The disks take
part in the same two exchanges as the turbines, after them, and the wake lines
in one of their own at each step that writes them. To know whom to expect
messages from, every rank keeps each body's envelope: a box of space that
holds its points and their kernels' reach, widened by its tip radius and two
cells so that a turning blade or a yawing rotor stays inside (a disk's and the
wake lines' by two cells, since they stand still). Each exchange
starts with one integer reduction over all ranks that says whether some
body's points have left its envelope; only then are the envelopes widened
and shared, which a turbine that stays in place does in its first step only
(once for the points it samples and, when its kernels reach farther than the
margin, again for the points it spreads). The run log
lists, for each turbine, disk and set of wake lines, the ranks it exchanges with. Every turbine's numbers
are gathered in four places only, a few numbers per turbine each: the
farm's totals in ``total_load.csv`` (at the diagnostic steps), the checkpoint's
``state`` file (at each checkpoint), and the start-up audit and the wake lines'
geometry (once); every disk's, in ``total_load.csv`` only (its thrust, in a
build with OpenFAST, which writes that file), and once at the first step, when
ERF's density at every disk centre is sampled by all ranks for the density
check.

The result does not depend on which rank owns which turbine or disk: on one or two
ranks it is the value that sampling and spreading every point on every rank,
with sums across all ranks, gives (bit for bit, but for the sign of an exact
zero); with more ranks the kernels' normalisations can differ in the last bits,
since the owner adds the ranks' parts in rank order and an MPI reduction may
not. The checks on a turbine's points (coverage, the tip's travel, the hub
axis, a kernel that reaches no face) and on a disk's or a wake line's while
they are sampled and spread (a point outside the domain, a kernel that reaches
no face) run on its owner, so on several ranks their abort message comes from
that rank.

OpenFAST keeps a single step counter for all the turbines of a process, so
the turbines a rank owns are stepped in turn, one OpenFAST time step at a
time; they must therefore use the same OpenFAST ``DT``. The owners are chosen
so that this holds whenever there are at least as many ranks as distinct time
steps (see above); the run aborts at start-up otherwise. The stub enforces the
same order. Each body is spread with its own
widths, and all the forces go onto the same momentum sources.
``<diagnostics_dir>/total_load.csv`` records the sum of every body's load on
the structure (the turbines' thrust, tower and nacelle forces and the disks'
thrust) and the turbines' total aerodynamic power, at the times of the turbine
logs; minus that load is what the integrated momentum source must equal. Wake
lines are built behind every rotor, so a downstream turbine's lines start
behind it, not behind the first.

Checkpoint and restart
----------------------

An ERF checkpoint carries the bodies' state under ``<chk>/moving_bodies``:
a ``state`` file with the step count, the time the bodies started, OpenFAST's
stop time, the anchor level and its step, one record of every body's placing
and forcing inputs, and each turbine's OpenFAST time index,
each OpenFAST turbine's own checkpoint ``<name>.chkp``, written through
``FAST_CreateCheckpoint`` by the rank that owns the turbine, with the
turbine's node loads (``<name>_nodes.dat``) and lifting-line correction
(``<name>_fllc.dat``) written by that rank too, the wake lines' running sums
and the bodies' statistics. A restart may run on another number of ranks: the
owners are chosen again from its boxes, and each reads its turbines' files
(tested with the stub library, from two ranks to one). On a restart the
turbines are restored with ``FAST_ExtInfw_Restart`` instead of being
initialised, ERF checks that the time index OpenFAST reports is the one its
own checkpoint expects, and the run continues with the next step; the
momentum sources are not stored but rebuilt at that step from the restored
loads, as they would have been in the run being continued. The diagnostics
files keep their rows up to the checkpoint, dropping any that a run which went
on past it wrote, and are appended to from there (a restart in a clean
directory starts them afresh with their headers). The restarted run must name
the same bodies with the same inputs, keep the anchor level and its step, and
stop no later than the stop time the turbines were started with, which
OpenFAST keeps in its own checkpoint; each check aborts before OpenFAST is
restarted, naming the input. A checkpoint written by a run without bodies (a
precursor) can be restarted with bodies: they then start afresh at the
checkpoint's time, and OpenFAST runs from there to the stop time.

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
a base outside the domain; two rotors whose swept discs intersect; a nacelle
drag area ``nacelle_cd * nacelle_area`` above ``2 pi epsilon_nacelle^2`` (in
metres), which takes its kernel correction factor under 0.5. Warnings:
the model's gravity differs from ERF's; a forced tower with AeroDyn's
``TwrAero`` off; fewer than 8 cells across the rotor diameter; a kernel in use
(``epsilon``, and ``epsilon_tower`` and ``epsilon_nacelle`` when the tower and
the nacelle are forced) narrower than a cell, named by its key; an actuator
line whose points are farther apart than ``epsilon``; tower force points
farther apart than ``epsilon_tower``; ``num_force_points_tower > 0`` with a
model that has no tower force nodes; the rotor's kernel cut by a non-periodic
boundary; a rotor facing away from
the wind at its hub or yawed more than 30 degrees from it. The bundled stub's
deck is not an OpenFAST primary file, so its model-file checks are skipped
with a note. The checks made at start-up before the flow exists remain: the
fixed step a whole multiple of the OpenFAST step, external inflow on,
``Wake_Mod`` paired with the sampling, ``TwrShadow = 0`` with a forced tower,
and later the tip-travel limit and the hub-axis convention. For a disk the
audit also notes rings whose points lie more than two kernel widths apart,
where the ring's force breaks into separate blobs.

Solver requirements
-------------------

This version supports the anelastic solver only, with a fixed time step, on
one anchor level of a possibly refined grid, and with an OpenFAST turbine the
AMReX floating-point traps must be off: OpenFAST raises exceptions of its own,
and a trapped run dies inside the library. The requirements are checked at start-up; the actuator line's tip
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
``<output_root>_flow.csv`` with the velocity handed to OpenFAST at the hub node
and its mean over the blade nodes, after the sampling correction and the FLLC.
Both are written every ``diagnostics_int`` steps with the time at the end of
the step; their first row is the start-up state. The files that describe a
step's forcing (``momentum_source.csv``, ``_correction.csv``, ``_fllc.csv``,
``_disk.csv``, ``_wake.csv``) carry the time at the start of that step. OpenFAST also writes its own output files as
configured in the ``.fst`` file. Whenever a body puts a force into the flow,
``<diagnostics_dir>/momentum_source.csv`` records the time and the momentum
source integrated over the domain, which equals the sum of the forces on the
fluid (minus the turbines' total loads, or for a turbine run as an actuator
disk minus its thrust's shaft component, whose rings cancel the in-plane load;
plus the prescribed-Ct disks' ``-T n``), and
``<diagnostics_dir>/total_load.csv``, in a build with OpenFAST, the
bodies' total load and power (see Farms).

Inputs are listed in :ref:`sec:MovingBodiesInputs`.
