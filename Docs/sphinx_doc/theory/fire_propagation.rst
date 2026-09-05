.. role:: cpp(code)
   :language: c++

.. _sec:FirePropagation:

Front Propagation
=================

The rate-of-spread models (:ref:`sec:ROS_Models`) give a speed at every fire
cell. This page describes how the burned region is advanced at that speed.
Two methods are available through :cpp:`erf.fire.propagation_method`:
``"farsite"`` (default), a Lagrangian marker scheme in the spirit of FARSITE,
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

The default path advances the front with markers, in the manner of Finney's
FARSITE, but on the fire grid rather than on a free polygon. Each fire
subcycle of length :math:`\Delta t_f`:

1. **Front cells** are those with :math:`\phi \le` :cpp:`erf.fire.farsite.phi_threshold`
   (default 0.1) and a positive rate of spread.
2. **Ellipse shape.** The length-to-width ratio of the local spread ellipse
   follows Anderson (1983) from the midflame wind :math:`U` in mph,

   .. math::

      L/W = 0.936\, e^{0.2566 U} - 0.397 \sqrt{U}, \qquad 1 \le L/W \le 8,

   and is converted to the Richards (1990) coefficients :math:`a = 1`,
   :math:`c = 0.2a`, :math:`b = (a + c) / (2\, L/W)` when
   :cpp:`erf.fire.farsite.use_anderson_lw` is 1. Setting it to 0 uses
   :cpp:`erf.fire.farsite.coeff_a`, ``coeff_b`` and ``coeff_c`` directly.
   Head, flank and backing rates are the head rate scaled by these
   coefficients, oriented along the wind, with an upslope correction from the
   terrain slope.
3. **Displacement accumulation.** Every front cell accumulates the displacement
   :math:`R\,\Delta t_f` along its spread direction in ``fire_disp_accum``.
   When the accumulated length reaches one fire cell, the target position is
   recorded and the accumulator is reset. Positions are gathered across MPI
   ranks so every rank stamps the same set.
4. **Stamping.** Each recorded position is stamped into :math:`\phi` as a
   burned cell. :cpp:`erf.fire.farsite.gaussian_sigma` selects a single-cell
   stamp (negative), an automatic radius from the grid spacing (zero) or a fixed
   Gaussian radius in metres (positive).

The subcycle length is :cpp:`erf.fire.farsite.cfl_fire` times the cell size
over the maximum rate of spread, so the front never crosses more than a
fraction of a cell per subcycle. Because the directionality comes from the
Anderson ellipse, the rate-of-spread models need only supply the head-fire
rate on this path; :cpp:`erf.fire.directional_ros` has no effect here.

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

Handed one scalar :math:`R` per cell, the level set grows a disc at the
head-fire rate: flanks and backing fire advance as fast as the head. With
:cpp:`erf.fire.directional_ros = true` the wind and slope are projected onto
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
3. FARSITE marker targets that land in a mask cell are dropped, so the front
   stops at the footprint instead of crossing it;
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
