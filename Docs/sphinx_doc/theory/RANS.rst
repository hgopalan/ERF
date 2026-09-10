
 .. role:: cpp(code)
    :language: c++

.. _RANS:

One-equation RANS closure
=========================

ERF offers a Reynolds-averaged (RANS) turbulence closure alongside its LES
and PBL schemes: the one-equation :math:`k` model of Axell and Liungman
(2001, *Environ. Fluid Mech.* **1**, 71-106; AL01 below), selected with
:cpp:`erf.rans_type = kEqn`. The turbulent kinetic energy :math:`k` is
prognostic and the turbulent length scale :math:`l` is algebraic, built from
the distance to the ground and the local stratification. The model is
meant for coarse grids on which the boundary layer is not resolved, i.e.
horizontal spacings of tens of metres and up, where the closure carries
the whole turbulent transport. The regression cases in
``Exec/CanonicalTests/Canonical_RANS`` exercise it on flat ground (neutral,
stable and convective) and over 2D and 3D hills.

Eddy viscosity and diffusivity
------------------------------

The Kolmogorov-Prandtl relations (AL01 Eqs. 11-12) give the eddy viscosity
and the eddy diffusivity of heat

.. math::

   \nu_t = c_\mu \, k^{1/2} \, l, \qquad \nu_t' = c_\mu' \, k^{1/2} \, l,

with the stability functions :math:`c_\mu` and :math:`c_\mu'` of the next
section. ERF stores :math:`\rho \nu_t` in the vertical and horizontal
momentum diffusivities (plot variables ``Kmv``, ``Kmh``) and
:math:`\rho \nu_t'` in the vertical heat diffusivity (``Khv``). By
default the horizontal heat diffusivity and the scalar and moisture
diffusivities are the eddy viscosity divided by :math:`Pr_t` and
:math:`Sc_t`, as for the LES closures; with
:cpp:`erf.rans_consistent_diffusivities = true` they all follow
:math:`\rho \nu_t'`, so that heat, scalars and moisture share the AL01
stability function in every direction. The diffusivity of :math:`k`
itself is :math:`\nu_t / \sigma_k` with :math:`\sigma_k = 1` (AL01
Table I), which the code sets unless ``erf.sigma_k`` is given.

Transport equation for :math:`k`
--------------------------------

The prognostic equation (AL01 Eq. 13) is the Deardorff form ERF already
integrates for LES,

.. math::

   \frac{\partial \rho k}{\partial t} + \nabla \cdot (\rho \mathbf{u} k)
   = \nabla \cdot \left( \frac{\rho \nu_t}{\sigma_k} \nabla k \right)
   + P_s + P_b - \rho \varepsilon,

with the shear production :math:`P_s = 2 \rho \nu_t S_{ij} S_{ij}`, the
buoyancy production :math:`P_b = (g / \theta_0) \, \overline{w' \theta'}`
from the modelled heat flux :math:`-\rho \nu_t' \, \partial \theta / \partial z`
(AL01 Eq. 15, dry air; :math:`\theta_0` is ``erf.theta_ref`` or the local
:math:`\theta`), and the dissipation (AL01 Eq. 19)

.. math::

   \varepsilon = (c_\mu^0)^3 \, \frac{k^{3/2}}{l}, \qquad c_\mu^0 = 0.5562.

The proportionality constant :math:`(c_\mu^0)^3` is what makes the
logarithmic law hold in a constant-stress layer (AL01 Sect. 4.1). The
dissipation is explicit in the update by default;
:cpp:`erf.implicit_tke_dissipation = true` treats it as
:math:`(c_\mu^0)^3 k_{old}^{1/2} / l` times the new :math:`k`, folded into
the time update, which removes the dissipation time-step restriction. In
the plotfile ``diss`` holds :math:`\rho \varepsilon` from the start of the
last step. A floor ``erf.tke_floor`` (default: machine epsilon on
:math:`\rho k`) bounds :math:`k` from below.

Length scale
------------

Near a wall the length scale is the geometric one (AL01 Eq. 22),

.. math::

   l_g = \kappa \, (d + z_0),

with :math:`d` the distance to the ground and :math:`z_0` the roughness
length of the surface layer. ERF caps it harmonically at
``erf.max_geom_lscale`` (30 m by default, about :math:`\kappa \times 0.1 z_i`),

.. math::

   l_g \leftarrow \frac{l_{max} \, l_g}{l_{max} + l_g},

or, with :cpp:`erf.rans_lscale_from_pblh = true`, at
:math:`\kappa \times 0.1 z_i` from the surface layer's boundary-layer height
diagnostic (``erf.most.pblh_calc = MYNN25``), clamped between
``erf.rans_lscale_min`` and ``erf.max_geom_lscale``.

Stratification shortens or lengthens :math:`l` through the buoyancy
frequency :math:`N^2 = (g/\theta_0) \, \partial \theta / \partial z`. In stable
air (AL01 Eq. 26) the geometric and buoyancy length scales combine
harmonically,

.. math::

   \frac{1}{l^2} = \frac{1}{l_g^2} + \frac{N^2}{c_b^2 \, k}, \qquad c_b = 0.35,

so that :math:`l \to c_b k^{1/2} / N` far from the wall under strong
stratification. In unstable air AL01 rewrite the same relation with the
dissipation of Eq. 19 on the right-hand side (their Eq. 28),

.. math::

   l = l_g \left[ 1 - (c_\mu^0)^6 \, c_b^{-2} \, R_t \right]^{1/2},
   \qquad R_t = \frac{k^2 N^2}{\varepsilon^2} = \frac{l_g^2 N^2}{(c_\mu^0)^6 \, k},

with :math:`R_t < 0` the turbulent Richardson number. ERF evaluates
Eq. 28 once, with :math:`R_t` from the geometric length and passed through
the smoothing of the next section, which bounds the unstable length at
:math:`l_g \, (1 + (c_\mu^0)^6 c_b^{-2} |R_t^{min}|)^{1/2}`, about
:math:`1.31 \, l_g` for the default constants. Iterating Eq. 28 instead
(dissipation from the new length, length from the new :math:`R_t`) is the
fixed-point map of Eq. 26 with :math:`N^2 < 0`, which has no fixed point
once :math:`l_g^2 |N^2|` exceeds :math:`c_b^2 k` (AL01 p. 78), so the length
would grow without bound in strongly convective, weakly turbulent air.

Stability functions
-------------------

With the length in hand the turbulent Richardson number is recomputed as
:math:`R_t = l^2 N^2 / ((c_\mu^0)^6 k)` and the stability functions of
Launder, as recalibrated by AL01 against the Högström data set, are

.. math::

   c_\mu = \frac{c_\mu^0 + 0.108 \, R_t}{1 + 0.308 \, R_t + 0.00837 \, R_t^2},
   \qquad
   c_\mu' = \frac{c_\mu^0}{1 + 0.277 \, R_t}

(AL01 Eqs. 31-32; the turbulent Prandtl number is :math:`c_\mu / c_\mu'`).
Both have poles near :math:`R_t = -3.6`. Below a critical value
:math:`R_t^c` (``erf.Rt_crit``, default :math:`-1`) the smoothing of
Burchard and Petersen maps :math:`R_t` onto :math:`(R_t^{min}, R_t^c)`
with :math:`R_t^{min}` = ``erf.Rt_min`` (default :math:`-3`):

.. math::

   \tilde R_t = R_t - \frac{(R_t - R_t^c)^2}{R_t + R_t^{min} - 2 R_t^c}
   = R_t^c + \frac{a \, x}{x + a}, \qquad x = R_t - R_t^c, \; a = R_t^{min} - R_t^c.

ERF uses the second, algebraically identical form: the first cancels two
terms of order :math:`|R_t|` and returned meaningless values once
:math:`|R_t|` exceeded about :math:`10^{15}`, which happens wherever
:math:`k` sits at its floor under an unstable :math:`N^2`. The inputs are
validated at start-up: :math:`R_t^{min} < R_t^c \le 0` and
:math:`R_t^{min} > -3.6`. The smoothed :math:`R_t` and the two functions
can be written to the plotfile as ``Rt``, ``cmu`` and ``cmu_prime``.

Wall condition on :math:`k`
---------------------------

Under a surface layer (``zlo.type = surface_layer``) and with
:cpp:`erf.dirichlet_k = true`, the first cell above the ground takes the
equilibrium value of AL01 Eq. 16,

.. math::

   k = \left[ \frac{u_*^3}{(c_\mu^0)^3} + \frac{\max(B, 0) \, \kappa \, d_1}{(c_\mu^0)^3} \right]^{2/3},

with :math:`u_*` and the buoyancy flux :math:`B = -(g/\theta_0) u_* \theta_*`
from the surface-layer model and :math:`d_1` the distance of the cell
centre to the ground; only a destabilising flux enters. The value is set
at the start of each step and held through every Runge-Kutta stage,
including the implicit vertical diffusion solve, and the ghost cell
below the wall carries the same value so no diffusive flux of :math:`k`
crosses the wall face. Without the flag the wall flux of :math:`k` is zero
and the first cell settles at about half the equilibrium value, because
the coarse first cell cannot resolve the shear production there; the
mean wind still follows the logarithmic law since the surface layer
supplies the stress, but everything fed by near-wall :math:`k` is off. ERF
warns at start-up in that configuration. With
:cpp:`erf.init_tke_from_ustar = true` the initial :math:`k` profile tapers
linearly from :math:`u_*^2` at the surface.

Wall distance
-------------

On a flat or vertically stretched mesh the distance :math:`d` is the
height of the cell centre above the ground. On a terrain-fitted mesh two
choices are offered through ``erf.wall_dist_type``:

* ``poisson`` (default): the differential-equation distance of Tucker
  (2003, *J. Comput. Phys.* **190**, 229-248). A Poisson equation
  :math:`\nabla^2 \phi = -1` is solved in terrain-following coordinates
  with :math:`\phi = 0` on the ground and zero normal gradient elsewhere,
  and :math:`d = -|\nabla \phi| + (|\nabla \phi|^2 + 2 \phi)^{1/2}`. The
  gradient is formed at cell centres from centred differences with the
  cell-centre metrics. On the canonical hills the mean error against the
  exact distance is about 1 %, with the largest errors, 5 to 10 %, in
  the first cells over a convex crest.

* ``terrain_height``: the height of the cell centre above the local
  surface, projected on the surface normal,
  :math:`d = (z - z_s) / (1 + h_\xi^2 + h_\eta^2)^{1/2}` with
  :math:`h_\xi, h_\eta` the terrain slopes. It is exact on a plane, within
  a few percent of the true distance for slopes below about 0.3 (mean
  errors of 0.02 % and 0.01 % on the canonical ridge and hill), and needs
  no linear solve. It cannot see side walls or overhangs, which the
  Poisson distance can.

The distance is available as the plot variable ``walldist``. It is
computed once at initialisation, also on a restart; it is not recomputed
after a regrid.

Implicit vertical diffusion
---------------------------

The column tridiagonal solve of ERF (``erf.vert_implicit``, on by default
with the compressible integrator) removes the explicit limit
:math:`\Delta t < \Delta z^2 / (2K)` for the quantities it covers.
Under the anelastic integrator it is opt-in: give
:cpp:`erf.vert_implicit = true` (or an explicit
:cpp:`erf.vert_implicit_fac`) and :math:`u`, :math:`v`, :math:`\theta`,
:math:`k` and moisture are all solved implicitly. The momentum solve is
folded into the slow tendency before the stage update, so the momenta are
diffused and then projected, which is the order the divergence constraint
needs; :math:`w` stays explicit unless ERF is built with
``ERF_IMPLICIT_W``, as in the compressible path. The anelastic update is
trapezoidal: the second stage averages the first-stage tendency,
recovered from the state difference, with the new one, so the first
stage's implicit increment is already half-counted and the implicit
operator on the second stage acts with half the step. The solve takes
each box's vertical extent as the whole column, so no level-0 box may be
cut in :math:`z` (the z entry of ``amr.max_grid_size`` must reach
``amr.n_cell``); ERF checks this at start-up.

On the neutral flat case the solve carries the time step from 5 s to 60 s
under the anelastic integrator, 6 times the explicit limit of about 10 s,
and the 12 h profiles stay within 3e-3 m/s of the explicit run. On a
terrain case the advective Courant number binds well below the explicit
diffusion limit, so the solve buys no time step there; it reproduces the
explicit answer to 1e-5 relative.

One caveat: with the implicit :math:`\theta` solve the answer is no
longer invariant to the box decomposition at round-off. The column solve
couples the whole column at once, so a one-unit-in-the-last-place
difference reaches every cell in one step, and in a near-neutral layer
:math:`\partial\theta/\partial z` is a difference of nearly equal
numbers, so the buoyancy term amplifies it. On the neutral case the
spread between one box and four saturates near 1e-5 m/s in wind, 1e-6 in
relative terms, against 1e-15 for the explicit run. It is a sensitivity
of the closure in a neutral layer rather than an inconsistency: with
Smagorinsky in place of the :math:`k` equation the same solve stays at
1e-12, and the implicit momentum and :math:`k` solves are decomposition
invariant to round-off on their own.

The buoyancy term of the :math:`k` equation uses the vertical heat flux
the closure computes at the start of the step,
:math:`-K_h \, \partial\theta/\partial z` from the cell-centred gradient
(AL01 Eq. 15; the Kynema implementation forms it the same way), with the
surface-layer flux in the first cell. It does not depend on how
:math:`\theta` is then advanced. Before this the explicit diffusion
operator overwrote that flux with the face flux scaled by the explicit
fraction, which put the term half a cell low and, whenever the implicit
solve was on, silently dropped the buoyancy production and destruction
of :math:`k` in every cell above the first (the convective case then
carried a third of its turbulence kinetic energy).

Limitations
-----------

* The buoyancy production and :math:`N^2` are dry: no virtual or
  liquid-water potential temperature.
* All levels must use the closure; a hybrid RANS-LES set-up is refused.
* Embedded and thin-body boundaries are not supported by the Poisson wall
  distance.
* The time step is bounded by :math:`\Delta z^2 / (2 K)` unless the
  implicit column solve is on (see below); with the eddy viscosities a
  convective boundary layer produces (tens of m\ :sup:`2`/s) that is the
  binding constraint, not the closure. With the solve on, the limit is
  the advective Courant number, and the vertical velocity, which stays
  explicit, is diffused at :math:`2 K_m`.
* The closure is local: a convective layer keeps a superadiabatic lapse
  of order :math:`-F/K_h` through its depth where a countergradient
  scheme or LES would mix it out.

Inputs
------

+----------------------------------------+------------------------------------------------------------+------------------+
| Parameter                              | Definition                                                 | Default          |
+========================================+============================================================+==================+
| **erf.rans_type**                      | ``None`` or ``kEqn``                                       | ``None``         |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.Cmu0**, **erf.Cb**               | :math:`c_\mu^0`, :math:`c_b`                               | 0.5562, 0.35     |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.Rt_crit**, **erf.Rt_min**        | smoothing bounds on :math:`R_t`                            | -1, -3           |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.max_geom_lscale**                | cap on :math:`l_g` [m]                                     | 30               |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.sigma_k**                        | Schmidt number of :math:`k`                                | 1 (RANS)         |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.theta_ref**                      | :math:`\theta_0`; 0 uses the local :math:`\theta`          | 0                |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.dirichlet_k**                    | wall value of :math:`k` from :math:`u_*`, AL01 Eq. 16      | false            |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.init_tke_from_ustar**            | initial :math:`k` profile from :math:`u_*`                 | false            |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.tke_floor**                      | floor on :math:`k` [m\ :sup:`2`/s\ :sup:`2`]               | 0 (epsilon)      |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.implicit_tke_dissipation**       | linearised implicit dissipation                            | false            |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.rans_consistent_diffusivities**  | heat, scalar and moisture diffusivities from :math:`c_\mu'`| false            |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.rans_lscale_from_pblh**          | cap :math:`l_g` at :math:`\kappa 0.1 z_i`                  | false            |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.rans_lscale_min**                | floor of that cap [m]                                      | 1                |
+----------------------------------------+------------------------------------------------------------+------------------+
| **erf.wall_dist_type**                 | ``poisson`` or ``terrain_height`` on a fitted mesh         | ``poisson``      |
+----------------------------------------+------------------------------------------------------------+------------------+

See :ref:`sec:Inputs` for the full descriptions.
