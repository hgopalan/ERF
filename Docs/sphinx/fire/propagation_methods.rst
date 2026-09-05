Propagation Methods
===================

The fire front propagation method is selected via::

    erf.fire.propagation_method = farsite   # default
    erf.fire.propagation_method = levelset

FARSITE Ellipse (default)
-------------------------

Lagrangian point-stamping scheme using Richards (1990) elliptical spread
geometry with Anderson (1983) L/W ratio. Points are stamped at the fire front
boundary and advanced according to the local rate of spread, creating a
discrete representation of the fire perimeter.

**Characteristics:**
- Lagrangian point-based tracking of fire front
- Deterministic and stable
- Backwards compatible with all existing input files
- Controlled by ``erf.fire.farsite.*`` parameters

**Parameters:**

- ``erf.fire.farsite.cfl_fire`` (default: 0.5)
  — CFL number for fire subcycle timestep control

- ``erf.fire.farsite.coeff_a``, ``coeff_b``, ``coeff_c`` (default: 0.5, 0.25, 0.1)
  — Richards ellipse coefficients (automatically derived from Anderson L/W if ``use_anderson_lw=1``)

- ``erf.fire.farsite.gaussian_sigma`` (default: −1, auto-derived)
  — Gaussian stamping radius [m]

- ``erf.fire.farsite.phi_threshold`` (default: 0.1)
  — Level-set contour value for front detection


Level-Set Advection (WENO5-Z + SSP-RK3 + Sussman Reinitialization)
-------------------------------------------------------------------

PDE-based propagation that solves the level-set advection equation:

.. math::

    \frac{\partial \phi}{\partial t} = -R(\mathbf{x}) \left( |\nabla \phi| - \varepsilon \Delta \phi \right)

where φ is the signed-distance function (negative inside fire, positive outside),
R(x) is the local rate of spread, ε is an artificial viscosity coefficient,
and Δφ is the Laplacian.

**Numerical scheme:**
- **Spatial:** WENO5-Z (5th-order Weighted Essentially Non-Oscillatory with Z-weighting)
  for reconstruction of flux derivatives
- **Temporal:** Strong Stability Preserving RK3 (3-stage SSP-RK3) for time integration
- **Reinitialization:** Sussman's pseudo-time method to maintain signed-distance property

**Characteristics:**
- Eulerian grid-based implicit level-set tracking
- Topologically clean signed-distance function (φ ≈ ±1 away from front)
- Better mass conservation of fire area compared to Lagrangian stamping
- Suitable for AMR refinement and curvature-dependent models
- Higher computational cost per timestep than FARSITE

**Parameters:**

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``erf.fire.levelset.cfl``
     - 0.4
     - CFL number for fire subcycle timestep (dt_ls = cfl × min(dx,dy) / max_ROS)
   * - ``erf.fire.levelset.eps_visc``
     - 0.4
     - Artificial viscosity coefficient ε in the level-set RHS
   * - ``erf.fire.levelset.reinit_every``
     - 5
     - Reinitialization period: reinitialize φ every N fire subcycles
   * - ``erf.fire.levelset.reinit_iters``
     - 10
     - Number of Sussman reinitialization pseudo-time iterations per reinit
   * - ``erf.fire.levelset.reinit_dtau``
     - −1 (auto)
     - Pseudo-timestep for reinitialization; negative means auto = 0.5·min(dx, dy)
   * - ``erf.fire.levelset.gradient``
     - "weno5z_front"
     - One-sided derivatives of the level set: "upwind" (first order everywhere), "weno5z" (HJ-WENO5-Z everywhere) or "weno5z_front" (WENO within the front band, first order elsewhere; the WRF-Fire hybrid)
   * - ``erf.fire.levelset.weno_band_cells``
     - 4
     - Half-width of the front band, in fire cells, that uses WENO with "weno5z_front"
   * - ``erf.fire.levelset.eps_visc_front``
     - −1 (off)
     - Artificial viscosity within ``visc_front_cells`` of the front (WRF-Fire ``fire_viscosity_bg``); negative keeps ``eps_visc`` everywhere
   * - ``erf.fire.levelset.visc_front_cells``
     - 2
     - Half-width, in fire cells, of the near-front viscosity band (WRF-Fire ``fire_viscosity_ngp``)
   * - ``erf.fire.levelset.visc_transition_cells``
     - 2
     - Width, in fire cells, of the linear blend from ``eps_visc_front`` to ``eps_visc``


When to use which method
------------------------

**Use FARSITE (default) if:**
- You have existing simulations or benchmarks with FARSITE
- You want maximum stability and predictability
- Computational cost is a primary concern
- The application does not require a strict signed-distance function
- You are using point tracking for visualization or diagnostics

**Use Level-Set if:**
- A topologically clean signed-distance function is required
  (e.g., for curvature-dependent models, signed-distance-based diagnostics)
- Better fire-front area conservation is desired
- You are using AMR mesh refinement and need to resolve the fire at multiple levels
- You prefer Eulerian implicit tracking over Lagrangian point stamping
- Computational cost is not a critical constraint

**Interoperability:**
Both methods share the same underlying ROS computation (Rothermel, MacArthur, etc.)
and are compatible with all other ERF-Fire features (spotting, terrain coupling,
heat flux, etc.). You can switch between methods by changing the
``erf.fire.propagation_method`` parameter without modifying the rest of the setup.


Anisotropy of the two paths
---------------------------

The two propagation methods do not spread the same way, and the difference is
larger than the numerics.

The FARSITE path applies the Anderson length-to-width ellipse, so the head
advances at the full rate of spread while the flanks and the backing fire run at
fractions of it. With the Richards coefficients used here the head factor is 1,
the backing factor is 0.2, and the flank factor is :math:`(a + c)/(2\,L/W)`. The
length-to-width ratio saturates at its cap of 8 above roughly 5 m/s of midflame
wind, so the flanks then run at 7.5% of the head rate and the burned area grows
as a downwind lobe.

The level-set path has no ellipse, and does not need one: it propagates a front
at whatever normal speed it is given, so the shape follows from the rate of
spread rather than from an imposed template. What it needs is a rate that depends
on direction. By default it is handed a single scalar, which it applies in all
directions, so the front grows at the head-fire rate everywhere; on an otherwise
identical case that covers roughly five times the area of the FARSITE path.

:cpp:`erf.fire.directional_ros = true` supplies the missing direction-dependence
by projecting the driving vectors onto the front normal
:math:`\hat{n} = \nabla\phi/|\nabla\phi|` and evaluating the selected model with
the projected scalars:

.. math::

   R(\hat{n}) = \text{model}\left(\max(\mathbf{U}\cdot\hat{n},\,0),\;
                                   \max(\nabla z\cdot\hat{n},\,0)\right)

At the head the normal is aligned with the wind, the model sees the full wind and
upslope, and :math:`R` is the head rate. On the flanks the projections vanish and
:math:`R` falls back to the model's no-wind, no-slope rate. Backing is the same:
a wind blowing out of the unburned fuel does not drive the front into it, and
Rothermel's slope factor is quadratic and so cannot represent downslope
retardation, so both are clamped at zero rather than reversed. Because
:math:`R(\hat{n})` never exceeds the head rate, the isotropic field that sets the
level-set CFL stays a conservative bound.

This is preferred over borrowing the ellipse, for a specific reason: the Anderson
length-to-width ratio is an empirical function of midflame wind speed, a stand-in
for a flow field that ERF resolves. Imposing it on top of a resolved wind is the
same double counting as applying the FARSITE terrain wind factors over resolved
terrain flow.

Two limits are worth knowing. The projection reproduces neither the saturation of
the observed length-to-width ratio, which the Anderson fit caps at 8, nor a
backing rate below the no-wind rate; both are calibration the empirical ellipse
carries and the projection does not. And making :math:`R` depend on the normal
turns the level-set equation into a general Hamilton-Jacobi problem, while the
Godunov upwinding used for :math:`|\nabla\phi|` is derived for a
direction-independent speed. Freezing :math:`R(\hat{n})` from central differences
at each Runge-Kutta stage, as done here, is the usual practical approximation
rather than the correct anisotropic flux.

Balbi keeps its own switch, :cpp:`erf.fire.balbi.directional`, which additionally
carries that model's per-cell moisture, surface temperature and heat-flux
couplings; either flag enables direction-dependent spread when
:cpp:`ros_model = "balbi"`.

``Exec/RegTests/FireRosComparison`` runs both settings against both Balbi
formulations and Rothermel on one fixed scenario and tabulates the burned area.


Terrain projection
------------------

Every ROS model returns a spread rate **along the terrain surface**, while both
propagation methods advance the front in map (x, y) coordinates. A surface step of
:math:`ds` up a slope of angle :math:`\theta` in the spread direction covers only
:math:`ds\cos\theta` horizontally, so the map-view front speed is

.. math::

   F_{map} = R\cos\theta = \frac{R}{\sqrt{1 + (\nabla z \cdot \hat{n})^2}}

with :math:`\hat{n}` the front normal. Both paths apply this factor:

- The level set folds it into the gradient operator,
  :math:`|\nabla\phi|_{surface} = |\nabla\phi|^2 / \sqrt{|\nabla\phi|^2 + (\nabla z\cdot\nabla\phi)^2}`,
  which is the directional form of the arc-length correction and reduces exactly to
  the component-wise spacing :math:`dx\sqrt{1 + s_x^2}` in one dimension.
- The FARSITE path divides its displacement :math:`ds` by the same factor before
  accumulating it.

A front running along the contour is already horizontal and is not projected; a
head-on upslope run on a 30 degree slope is reduced by :math:`\cos 30^\circ = 0.866`.
The factor is a cosine, so it can only slow map-view spread, never accelerate it.

Slope enhancement of the rate of spread is a separate effect and is unchanged: it
enters through the ROS model itself (Rothermel :math:`\phi_s`, Balbi flame tilt),
which is why a steeper slope still spreads faster overall despite the projection.

The projection is inactive on flat terrain, where the factor is exactly one.


Wind on the fire grid
---------------------

The fire grid is ``erf.fire.grid_ratio`` times finer than the atmospheric grid, so
the wind has to be mapped down. ``erf.fire.wind_interp`` selects how:

- ``"bilinear"`` (default) blends the four atmospheric columns surrounding the
  fire cell, with weights from the cell centre's position in atmospheric index
  space, :math:`g = (i_f + 1/2)/C - 1/2`.
- ``"nearest"`` takes the single column containing the cell, :math:`i_a = i_f / C`.

Under ``"nearest"`` every fire cell in an atmospheric column carries the same wind
vector, so the field the fire sees is piecewise constant with a step of a full
atmospheric cell of shear at each column edge. On an analytic wind field varying
linearly from 1.1 to 3.3 m/s across five atmospheric cells, ``"nearest"`` is off by
up to 0.4 m/s and steps by 1.0 m/s at each edge, while ``"bilinear"`` reproduces
the field to machine precision and steps by the true gradient over one fire cell.

Each column is sampled at the reference height above **its own** ground before the
blend, not at a single absolute height. Near the surface the profile is anchored
to the local terrain, and neighbouring column grounds can differ by tens of metres
on a slope: on the 30 degree flanks of the ROS_Slope_Effects ridge the difference
is 28 m, so blending at one absolute height would sample an upslope neighbour
about 35 m above its ground while asking for 6 m.

Within a column the bracketing levels are found by bisection on the
terrain-following cell-centre heights. A target below the lowest cell centre uses
the lowest level, and one above the highest uses the top level, rather than being
extrapolated.


Firebrand trajectories over terrain
-----------------------------------

A firebrand is lofted to :math:`H_z = 12.2\, I_B^{1/3}` above the ground **at its
source cell** and descends at its terminal velocity while the wind carries it.
The ground it lands on is not the height it left, so the descent is integrated
against the terrain: the brand lands where its altitude first meets the ground
beneath it,

.. math::

   z_{brand}(t) = z_{src} + H_z - w_{term} t
   \quad \text{lands when} \quad
   z_{brand} \le z_{ground}(x(t), y(t))

A brand drifting onto lower ground therefore falls further, stays aloft longer
and lands further out; one drifting upslope meets the rising ground sooner and
lands closer. On flat ground every elevation is zero and this reduces exactly to
the previous flat-earth flight time :math:`H_z / w_{term}`.

The trajectory step size is set from the longest fall the brand could make — down
to the lowest ground in the domain — so the fixed step count always covers the
descent.

The ground elevation used is the atmospheric column's, the same datum the wind
extraction measures from, so spotting sees terrain at atmospheric resolution.
Two simplifications remain: the brand is carried by the fire-grid reference wind
rather than by the wind at its own altitude, and the terminal velocity is a
single input rather than a distribution over brand sizes.


References
----------
- Richards, G. D. (1990). "An elliptical growth model of forest fire fronts and its
  numerical solution." International Journal for Numerical Methods in Engineering, 30(6).
- Anderson, H. E. (1983). "Predicting wind-driven wild fires." USDA For. Serv. Gen.
  Tech. Rep. INT-143.
- Sussman, M. (2003). "A second-order coupled level set and volume-of-fluid method
  for computing growth and collapse of vapor bubbles." Journal of Computational
  Physics, 187(1).
- Osher, S., & Sethian, J. A. (1988). "Fronts propagating with curvature-dependent
  speed: Algorithms based on Hamilton-Jacobi formulations." Journal of Computational
  Physics, 79(1).
