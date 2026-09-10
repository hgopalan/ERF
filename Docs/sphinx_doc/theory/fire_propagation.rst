.. role:: cpp(code)
   :language: c++

.. _sec:FirePropagation:

Front Propagation
=================

The rate-of-spread models (:ref:`sec:ROS_Models`) give a speed at every fire
cell. This page describes how the burned region is advanced at that speed.
Two methods are available through :cpp:`erf.fire.propagation_method`:
``"farsite"`` (default), a cell-by-cell Huygens scheme in the spirit of FARSITE,
and ``"levelset"``, a Hamilton-Jacobi solver. Both act on the same normalised
level set ``fire_phi`` and both record ``fire_arrival_time``, so everything
downstream (fuel consumption, heat flux, diagnostics, output) is independent
of the choice.

Level set and ignition
----------------------

``fire_phi`` is a normalised signed distance: zero on the front, negative
inside the burned region, positive outside, clamped to :math:`[-1, 1]`. The
disc ignition sets

.. math::

   \phi = -\frac{r - d}{r} \quad (d \le r), \qquad \phi = +1 \quad (d > r),

where :math:`d` is the distance from the ignition centre
:cpp:`erf.fire.ignition_x`, :cpp:`erf.fire.ignition_y` and :math:`r` is
:cpp:`erf.fire.ignition_r`. Polygon, polyline and scheduled ignitions
(:ref:`sec:MultiIgnition`) and ember landings (:ref:`sec:FireSpottingCrown`)
stamp negative values with the same convention. Firebreaks
(:ref:`sec:SpatialFuel`) stamp a large positive sentinel.

``fire_arrival_time`` starts at :math:`-1` everywhere and is set to the
current time on the step a cell's :math:`\phi` first becomes negative. It is
never reset, so it is the cumulative burned region and the field to use for
burned area, perimeter and arrival statistics.

FARSITE path
------------

The default path grows the burned region by Huygens' principle, as Finney's
FARSITE does, but on the fire grid rather than on a free polygon, and with the
Richards (1990) spread shape below. Each fire subcycle of length
:math:`\Delta t_f`:

1. **Ellipse shape.** The length-to-width ratio of the local spread ellipse
   follows Anderson (1983) from the midflame wind :math:`U` in mph,

   .. math::

      L/W = 0.936\, e^{0.2566 U} - 0.397 \sqrt{U}, \qquad 1 \le L/W \le 8,

   and is converted to the Richards (1990) coefficients :math:`a = 1`,
   :math:`c = 0.2a`, :math:`b = (a + c) / (2\, L/W)` when
   :cpp:`erf.fire.farsite.use_anderson_lw` is 1. Setting it to 0 uses
   :cpp:`erf.fire.farsite.coeff_a`, ``coeff_b`` and ``coeff_c`` directly.
   Head, flank and backing rates are the head rate scaled by these
   coefficients, oriented along the wind: the normal speed of a front whose
   normal makes the angle :math:`\theta` with the wind is
   :math:`R\,(a\cos\theta + b\,|\sin\theta|)` ahead and
   :math:`R\,(b\,|\sin\theta| - c\cos\theta)` behind. That is the support
   function of the rectangle :math:`[-cR, aR] \times [-bR, bR]` in the wind
   frame, the shape a point fire grows into. Without wind the shape is the
   disc of radius :math:`R`.
2. **Front cells** are the unburned, burnable cells with a burned neighbour
   across a face.
3. **Arrival time.** A front cell burns when the shape grown from its burned
   neighbours reaches its centre. For each quadrant, with :math:`T_x` and
   :math:`T_y` the arrival times of its neighbours along :math:`x` and
   :math:`y`,

   .. math::

      T = \min_{0 \le \lambda \le 1} \left[ (1-\lambda)\, T_x + \lambda\, T_y
          + \frac{\gamma\big((1-\lambda)\,\Delta x,\ \lambda\,\Delta y\big)}{\bar R} \right],

   where :math:`\gamma(\mathbf d)` is the time the shape takes to cover the map
   vector :math:`\mathbf d` at unit head rate, lengthened by
   :math:`\sqrt{1 + (\nabla z \cdot \hat{\mathbf d})^2}` on a slope. This is
   the Hopf-Lax update of the arrival time, taken over the four quadrants (one
   neighbour alone gives the end point). If :math:`T` falls inside the subcycle
   the cell burns and :math:`T` becomes its ``fire_arrival_time``. For a planar
   front it is exact: rows burn one at a time, a row spacing along the normal
   over the normal speed apart.
4. **Rate.** :math:`\bar R` is the mean, since the first neighbour burned, of
   the larger of the cell's own rate of spread and those of its burned
   neighbours, accumulated in ``fire_disp_accum``. The burned side carries the
   crown-fire rate, which is set only in burned cells; the cell's own rate keeps
   the front moving where a burned cell's rate has dropped.
5. **Level set.** :math:`\phi` is rebuilt as :math:`-1` in burned cells and
   :math:`+1` elsewhere.

The update reads a cell and its four neighbours only, so the arrival times do
not depend on the box decomposition or the number of ranks. The subcycle
length is :cpp:`erf.fire.farsite.cfl_fire` times the cell size over the maximum
rate of spread. Because the directionality comes from the Anderson ellipse, the
rate-of-spread models need only supply the head-fire rate on this path;
:cpp:`erf.fire.directional_ros` has no effect here.

:cpp:`erf.fire.farsite.front_update` selects the update. ``"front_cell"`` is
the default and the one described above. ``"legacy"`` is the update used before
September 2026: every cell with :math:`\phi \le`
:cpp:`erf.fire.farsite.phi_threshold` and a nonzero gradient of :math:`\phi`
accumulated :math:`R\,\Delta t_f` along its normal and, once that reached one
cell, stamped a burned target one cell ahead
(:cpp:`erf.fire.farsite.gaussian_sigma` chooses a single cell, an automatic or a
fixed stamp radius). Since :math:`\phi` was rebuilt as 0 on every unburned cell,
the first unburned row stamped along with the last burned row, and the front
advanced two rows per cell of travel: about twice the rate of spread at the
head, flanks and back. ``Exec/RegTests/FarsiteFrontUpdate`` runs both against
the Richards rates. ``phi_threshold`` and ``gaussian_sigma`` apply to the
legacy update only.

The startup acceleration of :ref:`fire_acceleration` rescales the rate before
this step. The size-based model scales every cell. The temporal model with
:cpp:`erf.fire.accel.clock = "front"` writes its factor to the burned cells and
to the unburned cells the front moves into, so both updates advance the front
at the accelerated rate. With the default ``"legacy"`` clock only burned cells
carry the factor, and a cell starts again from zero rate when it burns. The
``"front_cell"`` update then takes the front cell's own equilibrium rate as the
larger one and is not slowed at all, and the ``"legacy"`` update still advances
the first unburned row at the equilibrium rate
(``Exec/RegTests/FireAccelerationClock``).

Level-set path
--------------

Setting :cpp:`erf.fire.propagation_method = "levelset"` solves

.. math::

   \frac{\partial \phi}{\partial t} = -R(x, y)\,\bigl(|\nabla \phi| - \varepsilon\, \Delta \phi\bigr)

with the Godunov upwind Hamiltonian for :math:`|\nabla\phi|` (Osher and
Sethian: for an expanding front the backward difference counts only where
it is positive and the forward difference only where it is negative, so the
update always reads the burned side) and a three-stage strong stability
preserving Runge-Kutta step. The one-sided derivatives that feed the
Hamiltonian follow WRF-Fire and the Community Fire Behavior Model
(Munoz-Esparza et al. 2018; Jimenez y Munoz et al. 2026): by default,
:cpp:`erf.fire.levelset.gradient = "weno5z_front"`, they are the
fifth-order Hamilton-Jacobi WENO of Jiang and Peng (2000) with the Z
weights of Borges et al. (2008) within :cpp:`erf.fire.levelset.weno_band_cells`
(default 4) fire cells of the front, where the value of the level set
matters, and first-order differences elsewhere, where it does not.
``"weno5z"`` uses WENO everywhere and ``"upwind"`` first order everywhere,
the scheme before 2026-09-05; the Godunov choice between the two sides is
the same in all three. The reinitialisation uses the same derivatives.
First-order differences dissipate the front: in the line-ignition test of
Jimenez y Munoz et al. (2026) the first-order run lags the theoretical
front the most and WENO5 removes most of the gap. Next to a masked wall
the wide stencil falls back to the first-order difference through the
wall stencil. The ``Level_Set_Advection`` canonical case compares the three.

WRF-Fire and CFBM also carry two artificial viscosities, one near the front
and one elsewhere, both 0.4 by default; in the line-ignition test of
Jimenez y Munoz et al. (2026) lowering the near-front value to 0.1 brings
the WENO5 front onto the theoretical one. The same option is
:cpp:`erf.fire.levelset.eps_visc_front` (WRF-Fire ``fire_viscosity_bg``),
0.1 by default as in the paper: it is the coefficient within
:cpp:`erf.fire.levelset.visc_front_cells` (default 2, ``fire_viscosity_ngp``)
fire cells of the front, blending linearly to :cpp:`eps_visc` (0.4) over
the next :cpp:`erf.fire.levelset.visc_transition_cells` (default 2,
WRF-Fire's ``fire_viscosity_band`` times the advection band). A negative
value keeps the single :cpp:`eps_visc` everywhere. On the 5 m grid of the
WUI wildland case the near-front value of 0.1 leaves the head rate
unchanged and widens the flanks by about 7% against the single value
(4.96 ha burned at 1200 s against 4.83 ha): the viscosity acts where the
front is curved. The Laplacian term is an artificial
viscosity with coefficient :cpp:`erf.fire.levelset.eps_visc` (default 0.4)
that keeps the front smooth at the grid scale. When terrain slopes are
available, :math:`|\nabla \phi|` is projected onto the terrain surface so that
:math:`R` is a rate along the ground rather than in map view.

The subcycle length is :cpp:`erf.fire.levelset.cfl` (default 0.4) times the
cell size over the maximum rate of spread. The field is periodically
reinitialised, see below.

Direction-dependent spread
~~~~~~~~~~~~~~~~~~~~~~~~~~

Handed one scalar :math:`R` per cell (:cpp:`erf.fire.directional_ros =
false`), the level set grows a disc at the head-fire rate: flanks and backing
fire advance as fast as the head. By default
(:cpp:`erf.fire.directional_ros = true`, as in WRF-Fire) the wind and slope are projected onto
the front normal :math:`\hat n = \nabla\phi / |\nabla\phi|` and the selected
model is evaluated with the projected scalars,

.. math::

   R(\hat n) = \text{model}\bigl(\max(\mathbf U \cdot \hat n, 0),\ \max(\nabla z \cdot \hat n, 0)\bigr),

inside every Runge-Kutta stage, so the head, flanks and backing fire each get
the rate the model gives for their own orientation. Backing and downslope
components are clamped at zero rather than reversed, since the empirical
models take magnitudes. The FARSITE ellipse is deliberately not imposed on top
of a resolved wind field: its length-to-width fit stands in for a flow field
that ERF resolves, and imposing both would double count the wind. The
projection is also what the hybrid model uses on this path, and Balbi has an
equivalent switch :cpp:`erf.fire.balbi.directional` that additionally carries
its per-cell couplings. ``Exec/RegTests/FireRosComparison`` tabulates the
effect: the head rate is unchanged and the burned area falls, since the flanks
no longer run at the head rate.

A normal speed is not a spread rate in a direction, though, and from a point
ignition the difference shows. The exact (viscosity) solution of
:math:`\phi_t + R(\hat n)|\nabla\phi| = 0` from a point is the Wulff shape

.. math::

   W(t) = \{\mathbf x : \mathbf x\cdot\hat n \le t\,R(\hat n)\ \text{for every}\ \hat n\},

whose extent along a direction :math:`\hat d` is
:math:`\min_{\hat n\cdot\hat d>0} R(\hat n)/(\hat n\cdot\hat d)`. That equals
:math:`R(\hat d)` only when :math:`R` is the support function of a convex set.
Rothermel's projected rate :math:`R_0(1 + \phi_w (U n_x)^B)` is not once
:math:`\phi_w (B-1) > 1`, nor :math:`R_0(1 + \phi_s (s n_x)^2)` once
:math:`\phi_s > 1`: the head of a point fire becomes a wedge of oblique
facets whose tip runs at
:math:`R_0\,\tfrac{B}{B-1}\,(\phi_w(B-1))^{1/B}` in wind and about
:math:`2\sqrt{\phi_s}\,R_0` on a slope, well below the head rate
:math:`R_0(1+\phi_w+\phi_s)`. A better scheme would not help, since it is the
exact solution that falls short; the level set, which freezes
:math:`R(\hat n)` from central differences, lands between the tip speed and
the head rate. A straight line fire keeps :math:`\hat n` along the wind and is
unaffected. ``Exec/RegTests/FireDirectionalShape`` (point ignition in a uniform
wind) and ``Exec/CanonicalTests/Fire/Verification/Slope_No_Wind`` measure it: in
a uniform 1.5 m/s wind on short grass (:math:`\phi_w = 9.4`) the head runs at
0.185 m/s against Rothermel's 0.249 m/s and a Wulff tip of 0.142 m/s, between
0.153 and 0.198 m/s depending on the gradient scheme and viscosity.

:cpp:`erf.fire.directional_shape = "ellipse"` (default ``"projection"``)
removes the shortfall by taking the normal speed from the support function of
an ellipse built from the same model. The ellipse is convex, so it is its own
Wulff shape and spreads a point fire's head at the head rate. With
:math:`R_0` the model's no-wind, no-slope rate and
:math:`\Delta R_w = \text{model}(|\mathbf U|, 0) - R_0`,
:math:`\Delta R_s = \text{model}(0, |\nabla z|) - R_0` the wind and slope
increments, the head runs at

.. math::

   R_h = R_0 + g\,\bigl|\Delta R_w\,\hat{\mathbf w} + \Delta R_s\,\hat{\mathbf s}\bigr|,
   \qquad g = \frac{\text{model}(|\mathbf U|, |\nabla z|) - R_0}{\Delta R_w + \Delta R_s},

along that vector (:math:`\hat{\mathbf w}` and :math:`\hat{\mathbf s}` the
unit wind and upslope vectors: BEHAVE's vector addition, with :math:`g = 1` for
the additive Rothermel form), and the back and flanks at :math:`R_0`. The
semi-axes are :math:`b = (R_h + R_0)/2` along the head and :math:`a = R_0`
across it, with centre offset :math:`c = (R_h - R_0)/2`, and the normal speed
is the support function given under Spread ellipse below. With aligned wind
and slope the head, back and flank rates are the projection's own; only the
directions between them change. The length-to-width ratio
:math:`(R_h/R_0 + 1)/2` is not capped, as the projection's is not. The option
covers Rothermel, BEHAVE, MacArthur, Cheney-Gould and FBP; Balbi and the hybrid
keep the projection, and :cpp:`erf.fire.levelset.ellipse` cannot be combined
with it. The unit test ``ERF_GTestDirectionalShape`` checks the rates, the
vector addition and the Wulff extents of both forms.

Flanks at :math:`R_0` are the projection's claim, not an observation, and give
a length-to-width ratio far above the observed one: 5.7 for short grass in a
1.5 m/s wind, where Anderson (1983) gives 1.5.
:cpp:`erf.fire.directional_ellipse_lw = "anderson"` (default ``"model"``)
keeps the head rate, its direction and the back rate :math:`R_0`, and sets the
semi-minor rate to :math:`a = b/(L/W)`. Here :math:`L/W` is Anderson's ratio (as
in the spread ellipse below), capped at
:cpp:`erf.fire.directional_ellipse_lw_max` (default 8), evaluated at the
effective wind speed :math:`U_\text{eff}`. That speed is the wind that alone
would give the head rate, :math:`\text{model}(U_\text{eff}, 0) = R_h`, which is
BEHAVE's definition, so a slope elongates the fire as the equivalent wind
would. For Rothermel and BEHAVE it is the inverted wind factor
:math:`((R_h/R_0 - 1)/(C (\beta/\beta_\text{op})^{-E}))^{1/B}`, capped at the
model's own wind limit; the other models find it by bisection. The ellipse
stays convex, so the head still runs at :math:`R_h`.

Spread ellipse
~~~~~~~~~~~~~~

The alternative the FARSITE family uses (Finney 1998; ELMFIRE, Prometheus,
Cell2Fire) is available as an option, :cpp:`erf.fire.levelset.ellipse`,
off by default and exclusive with the projection above. Every point of the
front spreads as a Huygens ellipse: the length-to-width ratio :math:`L/W`
follows Anderson (1983) from the midflame wind (capped at
:cpp:`erf.fire.levelset.ellipse_lw_max`, default 8, or fixed with
:cpp:`erf.fire.levelset.ellipse_lw`), the head-to-back ratio follows
Alexander (1985),

.. math::

   H/B = \frac{L/W + \sqrt{(L/W)^2 - 1}}{L/W - \sqrt{(L/W)^2 - 1}},

and with the model's rate :math:`R` as the head rate the ellipse has
semi-major rate :math:`b = (R + R/(H/B))/2`, semi-minor rate
:math:`a = b/(L/W)` and centre offset :math:`c = b - R/(H/B)`. The level set
needs the normal speed of that envelope, the support function of the
ellipse,

.. math::

   R_n(\theta) = c\cos\theta + \sqrt{b^2\cos^2\theta + a^2\sin^2\theta},

with :math:`\theta` the angle between the front normal and the wind, so the
head runs at :math:`R`, the flanks at :math:`a` and the back at
:math:`R/(H/B)`. Unlike the projection this reproduces the saturation of the
observed length-to-width ratio and a backing rate below the no-wind rate,
which are calibration the empirical ellipse carries; it also double counts
a resolved wind for the reason given above, so it is the right choice when
comparing with those models and the projection is the right choice when the
flow around the fire is resolved. Slope enters through :math:`R` only; the
ellipse is oriented by the wind alone. ``Exec/RegTests/FireLevelSetEllipse``
measures the burned shape against :math:`L/W` and the unit test
``ERF_GTestLevelSetEllipse`` checks the rates at the head, flanks and back.

Reinitialisation
~~~~~~~~~~~~~~~~

Advection steepens and flattens :math:`\phi`, so every
:cpp:`erf.fire.levelset.reinit_every` subcycles (default 5) it is restored to
a signed distance by :cpp:`erf.fire.levelset.reinit_iters` (default 10)
pseudo-time iterations of a band-normalised Sussman update,

.. math::

   \frac{\partial \phi}{\partial \tau} = \operatorname{sgn}(\phi_0)\,\frac{1 - L\,|\nabla\phi|}{L},

whose fixed point is :math:`|\nabla \phi| = 1/L`: :math:`\phi` varies linearly
from 0 at the front to :math:`\pm 1` at the band half-width :math:`L`, which is
:cpp:`erf.fire.levelset.reinit_band_m` or three cells when that is not
positive. Cells whose neighbourhood straddles the interface use the Russo and
Smereka (2000) subcell correction, which fixes the front from :math:`\phi_0`
instead of letting the iteration move it; without it every pass would erode
the burned area, and the level-set path never rebuilds :math:`\phi` from the
arrival time. The pseudo-timestep :cpp:`erf.fire.levelset.reinit_dtau`
defaults to a quarter of the cell size, half the Sussman stability limit.
:math:`\phi` is clamped to :math:`[-1, 1]` after every iteration.

Non-burnable cells
------------------

A non-burnable mask on the fire grid marks cells the fire may never enter.
It is built once at initialisation from up to three sources, each off by
default:

- **structures**, with :cpp:`erf.fire.structures.enable`: every cell whose
  height in the building heightmap :cpp:`erf.fire.structures.file` exceeds
  :cpp:`erf.fire.structures.min_height`. The file is in the ERF terrain text
  format, sampled onto fire cell centres by nearest point so footprints keep
  their edges, and defaults to the hybrid selector's file and then to
  :cpp:`erf.buildings_file_name`, so one heightmap can drive the
  immersed-forcing buildings, the hybrid ``structure`` selector and the mask;
- **fuel codes** listed in :cpp:`erf.fire.fuel_map.nonburnable_codes`, for
  example ``0`` (nodata) and the Scott and Burgan non-burnable classes
  ``91``-``99``. Without the list an unknown code still falls through to fuel
  model 1 in the Rothermel table and burns as short grass;
- **firebreaks**, with :cpp:`erf.fire.firebreak.use_mask`. Firebreaks are
  otherwise stamped into :math:`\phi` once as a large positive sentinel,
  which the FARSITE path rebuilds away every subcycle and the level-set
  reinitialisation clamps; the mask makes them permanent on both paths.

The mask acts in five places, so that no path around it is left open:

1. the rate of spread is zero in mask cells, on the isotropic field and inside
   every Runge-Kutta stage of the direction-dependent drivers;
2. the level set is clamped at zero there, after every advection subcycle,
   after reinitialisation and after any scheduled ignition. The clamp is only
   a guard: with a zero rate of spread a masked cell's value does not evolve
   during advection, and reinitialisation keeps its sign, so masked cells
   keep a consistent signed distance to the real front. They are not lifted
   to a fixed positive level, which would let the footprint edge act like a
   front of its own;
3. on the FARSITE path a mask cell is never a front cell (and a legacy marker
   target that lands in one is dropped), so the front stops at the footprint
   instead of crossing it;
4. ember landings on a mask cell are discarded and the spot disc never stamps
   into one;
5. the fuel load is zero in mask cells from the start, so they produce no
   heat flux, intensity or flame diagnostics, and the arrival time is never
   set.

**Walls in the level-set stencils.** A masked cell keeps the level-set value
it had before the front arrived, so once its open neighbour burns the two
differ by many metres across one cell. The Godunov norm takes the larger
one-sided difference, which is then the one into the wall, and the cells
along a wall burn down far faster than the spread rate; the front normal
from the central difference points into the wall as well, so the directional
models evaluate a head-fire rate there. On flat ground this drives the flank
along a wall about 20% fast (``FireHybridObstacles``, probe ``u3``).
:cpp:`erf.fire.levelset.wall_extrapolate` extrapolates the level set into
the mask inside every stencil: a masked stencil point takes the centre
cell's value in the gradient, the Laplacian, the front normal and the
reinitialisation, which itself leaves masked cells untouched. The wall is
then a zero-gradient boundary of the distance function, the normal next to
it runs along it, and the flank arrives at the unmasked rate.
``Exec/RegTests/FireNearWall`` measures this. Off by default, so existing
masked results are unchanged; it has no effect without a mask.

The mask is written to the fire plotfile as ``fire_nonburnable``. A fire
approaching a masked footprint goes around it through whatever burnable
cells remain; ``Exec/RegTests/FireHybridObstacles`` compares the same
obstacle deck with the mask off and on.

Choosing a path
---------------

The FARSITE path is the reference behaviour, carries the Anderson ellipse as
calibration, and is the path the canonical FARSITE tests and the acceleration,
spotting and crown-fire options were developed on. The level-set path is the
one to use when the wind is resolved and direction-dependent spread from the
model itself is wanted, when the Balbi couplings are in use, or when the
hybrid model is run on the directional path. The two are not comparable cell
for cell: the ellipse reproduces neither a backing rate below the no-wind rate
nor the saturation of the length-to-width ratio, and the projection reproduces
neither of the empirical calibrations the ellipse carries.

The temporal acceleration reaches the level-set speed only with
:cpp:`erf.fire.accel.clock = "front"`, whose factor also multiplies the rate the
directional, hybrid and Balbi paths rebuild in every Runge-Kutta stage. Those
paths never read the accelerated ``fire_ros``, so under the ``"legacy"`` clock
they spread as if acceleration were off (``Exec/RegTests/FireAccelerationClock``).

Restart
-------

The checkpoint stores ``fire_phi``, ``fire_arrival_time``, ``fire_ros``,
``fire_fuel_load``, ``fire_fuel_mc``, the FARSITE displacement accumulator,
the lagged fluxes waiting to be injected and, when crown fire is on, the
crown state and load. On restart the fire
layer is initialised from the inputs as on a clean start (fuel map,
firebreaks, hybrid weights, structure mask) and these fields are then read
back, so the front, burned area and consumed fuel continue exactly.

References
----------

- Finney, M. A. (2004). FARSITE: Fire Area Simulator model development and evaluation. USDA Forest Service RMRS-RP-4 Revised.
- Anderson, H. E. (1983). Predicting wind-driven wild land fire size and shape. USDA Forest Service Research Paper INT-305.
- Richards, G. D. (1990). An elliptical growth model of forest fire fronts and its numerical solution. International Journal for Numerical Methods in Engineering, 30(6), 1163-1179.
- Osher, S. and Fedkiw, R. (2003). Level Set Methods and Dynamic Implicit Surfaces. Springer.
- Borges, R., Carmona, M., Costa, B. and Don, W. S. (2008). An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws. Journal of Computational Physics, 227(6), 3191-3211.
- Sussman, M., Smereka, P. and Osher, S. (1994). A level set approach for computing solutions to incompressible two-phase flow. Journal of Computational Physics, 114(1), 146-159.
- Russo, G. and Smereka, P. (2000). A remark on computing distance functions. Journal of Computational Physics, 163(1), 51-67.


Perimeter ignition with spin-up
-------------------------------

A fire can be started from an observed perimeter instead of a point:
:cpp:`erf.fire.ignition.polygon_file` lists the vertices and the level set
is set to the signed distance from that polygon. By default the polygon is
stamped at initialisation. :cpp:`erf.fire.ignition.polygon_time` stamps it
at that time instead, so the atmosphere spins up before the fire exists;
this is WRF-SFIRE's perimeter time, the way the Community Fire Behavior
Model runs start from mapped perimeters (Jiménez y Muñoz et al., 2026, with
about three hours of spin-up). With
:cpp:`erf.fire.ignition.polygon_interior_ros` :math:`= R > 0` the interior
is given the state of a fire that reached the perimeter after spreading
outward at :math:`R`: a cell at distance :math:`d` inside the perimeter gets
the arrival time :math:`t_{ign} - d/R`, clamped at the simulation start
because a negative arrival time reads as unburned, and its fuel load is
reduced by
:math:`\exp(-d/(R\tau))`, the exponential burnout the heat-flux step
applies, with :math:`\tau` = :cpp:`erf.fire.ignition.polygon_interior_tau`
(the cell crossing time :math:`\Delta x / R` when 0). Without it the whole
interior ignites at the perimeter time with its fuel intact, which releases
the heat of the entire burnt area at once. The regression test
``Exec/RegTests/FirePerimeterIgnition`` checks the interior state cell by
cell from the fire plotfile.
