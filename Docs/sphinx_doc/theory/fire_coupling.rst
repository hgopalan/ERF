.. role:: cpp(code)
   :language: c++

.. _sec:FireCoupling:

Fire-Atmosphere Coupling
========================

The atmosphere drives the fire through the wind, and the fire drives the
atmosphere through a surface heat flux. This page covers both directions:
how the wind reaches the fire grid and is adjusted to the flame zone, how fuel
is consumed and the heat flux formed, how that flux is injected into the
energy equation, and the smoke tracer.

Wind extraction
---------------

The rate-of-spread models take a midflame wind. It is built in three stages.

**Sampling.** For each fire cell the atmospheric column is sampled at the
reference height :cpp:`erf.fire.wind_ref_ht` (default 6.1 m) above the local
ground. The ground is the mean of the four surface nodes of the atmospheric
cell, taken from the atmospheric terrain even when
:cpp:`erf.fire.terrain_file_name` supplies a finer terrain for the slopes,
since the wind profile being interpolated belongs to that column. The
face-staggered velocities are averaged to cell centres and interpolated
linearly in height between the two cell centres that bracket the target.
With :cpp:`erf.fire.use_per_fuel_wind_ht` the height comes instead from a
per-fuel table in the WRF-SFIRE ``fcwh`` convention (all 6.096 m by default).

**Horizontal mapping.** :cpp:`erf.fire.wind_interp` selects how atmospheric
columns map to the finer fire cells. ``"bilinear"`` (default) blends the four
columns surrounding each fire cell, every one sampled at the reference height
above *its own* ground, so a cell midway between columns gets the wind of that
position. ``"nearest"`` takes only the column containing the cell, which makes
the fire-grid wind piecewise constant on atmospheric cells with a full cell of
shear at every block edge. The result is ``fire_wind_ref`` and the sampling
height is ``fire_wind_extract_z``.

Next to immersed-forcing buildings the columns inside a footprint carry the
relaxed in-building velocity, and the bilinear blend hands a share of it to
every fire cell within one atmospheric cell of the wall. With
:cpp:`erf.fire.structures.wind_open_columns` each column's weight is
multiplied by its open fraction (the part of the column not covered by
non-burnable structure cells, from the structure heightmap) whenever its
sampled roof is above the cell's wind height, and the four weights are
renormalised, so the fire reads the wind of the open columns only. A cell
whose four columns are all closed keeps the plain weights; it is inside a
footprint and masked anyway. Off by default; needs
:cpp:`erf.fire.structures.enable` and has no effect with ``"nearest"``.

**Adjustment to the flame zone.** ``fire_wind_eff`` is the reference wind
multiplied by the wind adjustment factor and modified by the terrain
corrections:

- The wind adjustment factor reduces the reference wind to the midflame
  height of the fuel bed. :cpp:`erf.fire.waf_formula = "andrews"` (default)
  uses the unsheltered logarithmic form of Andrews (2012) driven by the fuel
  bed depth of the domain fuel model, and ``"behaviorplus"`` the linear
  BehavePlus form; :cpp:`erf.fire.use_waf = false` passes the reference wind
  through. The Balbi model can bypass the factor with
  :cpp:`erf.fire.balbi.wind_source = "reference"`, since it normalises the
  wind by its own vertical velocity scale.
- The FARSITE terrain corrections (:cpp:`erf.fire.use_terrain_wind`, default
  true) scale and rotate the wind from the slope and curvature on the fire
  grid: a ridge speed-up :cpp:`erf.fire.k_ridge`, lee sheltering
  :cpp:`erf.fire.k_shelter`, valley channelling :cpp:`erf.fire.k_valley` in
  concave terrain, and a deflection :cpp:`erf.fire.k_deflect` of the wind
  vector toward the slope. These are empirical stand-ins for flow that a
  resolved LES already contains, so switch them off when the atmosphere
  resolves the terrain.
- The Rothermel kernel additionally caps the effective wind at the maximum
  effective wind speed of Rothermel (1972), 300 ft/min for fine fuels
  (surface-area-to-volume ratio above 1000 ft⁻¹) and 500 ft/min otherwise,
  when :cpp:`erf.fire.use_wind_limit` is true (default). Wind-driven
  conflagrations exceed this cap; turn it off, or hand the high-wind regime to
  a model without a cap through the hybrid ``wind`` selector.

Coupling modes
--------------

:cpp:`erf.fire.coupling_type` selects which wind the fire sees and whether its
heat returns to the atmosphere:

- ``"passive"``: the fire spreads on the wind from the start of the step and
  its heat flux is computed but never injected. Use it for fast screening runs
  and for regression tests where the atmosphere should not respond.
- ``"lagged"`` (default): the fire spreads on the wind from the start of the
  step; the flux it produces is stored and injected during the next step's
  Runge-Kutta stages. This one-step explicit lag is the WRF-SFIRE approach.
- ``"synchronous"``: the fire spreads on the wind *after* the dynamical core
  has advanced, so the spread at step :math:`n+1` already reflects the
  momentum response to the heat injected from step :math:`n`; the flux still
  enters at the next step.

The fire runs once per atmospheric step in all three modes.

Fuel consumption and heat flux
------------------------------

Once a cell has burned (:math:`\phi < 0`) its remaining fuel load
:math:`w` decays exponentially with the cell residence time,

.. math::

   w_{n+1} = w_n\, e^{-\Delta t / \tau_{res}}, \qquad
   \tau_{res} = \max\!\left(\frac{\Delta x_f}{R},\ \tau_{SAV}\right),

where :math:`\Delta x_f / R` is the time the front takes to cross one fire
cell and :math:`\tau_{SAV} = 301\,\rho_p/\sigma` (Rothermel 1983, seconds
with :math:`\rho_p` in lb/ft³ and :math:`\sigma` in ft⁻¹) is the particle
burnout time used as a floor. :cpp:`erf.fire.tau_residence_s` overrides the
residence time with a fixed value when positive. The sensible flux is the
instantaneous power of that decay,

.. math::

   Q = \frac{w\, h}{\tau_{res}} \quad [\mathrm{W/m^2}],

with :math:`h` the heat content of the fuel model. On a spatial fuel map both
the initial load and :math:`h` are those of each cell's own fuel code. Byram
intensity, flame length and the flame diagnostics are built from the same
quantities (:ref:`sec:FireOutput`).

The latent flux follows WRF-SFIRE,

.. math::

   Q_{lat} = \frac{L_v}{h}\,\bigl(b + (1 - b)\, f_w\bigr)\, Q, \qquad
   b = \frac{M_f}{1 + M_f},\ f_w = 0.56,

where :math:`M_f` is the fuel moisture and :math:`f_w` the water yield of
combustion per unit dry fuel.

Injection into the atmosphere
-----------------------------

The fire-grid fluxes are area-averaged onto the atmospheric columns and
distributed vertically with an exponential profile,

.. math::

   H(z) = \frac{Q}{c_p}\, e^{-z/\alpha_g}, \qquad
   \frac{\partial (\rho\theta)}{\partial t} = -\rho\, \frac{\partial H}{\partial z},

where :math:`z` is height above the local terrain and
:math:`\alpha_g` = :cpp:`erf.fire.heat_flux_alfg` (default 45 m) is the
e-folding depth: 37% of the flux remains at 45 m and 1% at 225 m. The latent
flux is distributed the same way into the vapour equation when
:cpp:`erf.fire.inject_latent` is true and moisture is active.

.. note::

   Before 2026-09-03 the default tendency carried an extra density factor,
   :math:`\partial(\rho\theta)/\partial t = -\rho\, \partial H/\partial z`
   with :math:`H = Q/c_p`, so the heat it injects is :math:`\rho` times the
   fire flux, about 10 to 15% too much at sea level. Heating gives
   :math:`\partial(\rho\theta)/\partial t = -(1/c_p)\, \partial Q/\partial z`
   with no density factor, and that is now the default;
   :cpp:`erf.fire.heat_tendency_density = true` restores the historical form
   for comparison with older coupled results. The energy diagnostic printed
   with :cpp:`erf.fire.fire_debug` reads exactly
   :math:`1 - e^{-z_{top}/\alpha_g}` with the default and about
   :math:`\rho` times that with the historical form.

:cpp:`erf.fire.fire_atm_feedback` multiplies both fluxes before injection;
zero gives one-way coupling with the fire still responding to the wind, one
gives the full feedback.

The cell source is rebuilt by the atmosphere at every Runge-Kutta stage and
the fire tendency is applied after that rebuild. :cpp:`erf.fire.source_mode`
selects how: ``"overwrite"`` (default, the historical behaviour) replaces the
potential-temperature and vapour slots, which keeps the injected energy
independent of the number of stages but discards any other source already
in those slots, such as the immersed-forcing scalar relaxation, radiative
heating or Rayleigh damping; ``"add"`` accumulates the fire tendency into the
freshly rebuilt source, once per stage, so nothing is lost and nothing is
double counted. Use ``"add"`` whenever another source writes into those
slots: immersed forcing applied on the slow step
(``erf.immersed_forcing_substep = false``, the anelastic default), radiative
heating, Rayleigh damping. With immersed forcing on the acoustic substeps
(the compressible default) the scalar relaxation never meets the slow-step
source and the two modes are identical, as they are with no other source.

Heat placement around buildings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fire-grid flux is area-averaged onto each atmospheric column. With the
structure mask on (:ref:`sec:FirePropagation`) the fire cells under a
building carry no flux, so a column fully covered by a footprint receives no
heat at all, and a column that straddles a footprint edge receives a flux
diluted by its open fraction. What the plain profile gets wrong in those
partial columns is where the heat goes: it is spread over the cells below
the roof as if the column were open. With
:cpp:`erf.fire.heat_open_fraction` (requires
:cpp:`erf.fire.structures.enable`) every column carries its open fraction
:math:`f` and the mean roof height :math:`H` of its structure cells, both
area-averaged from the fire grid at initialisation. A cell's share of the
exponential profile is scaled by :math:`f` below the roof, by one above it,
and linearly across the cell that holds the roof, and the column is
renormalised over the shares that remain,

.. math::

   w_k = \frac{\bigl(e^{-z_k/\alpha_g} - e^{-z_{k+1}/\alpha_g}\bigr)\, s_k}
              {\sum_j \bigl(e^{-z_j/\alpha_g} - e^{-z_{j+1}/\alpha_g}\bigr)\, s_j}
         \,\bigl(1 - e^{-z_{top}/\alpha_g}\bigr),

so the energy injected per column is exactly what the plain profile injects
and the share placed below the roof falls from :math:`1 - e^{-H/\alpha_g}`
to that times the open fraction, the rest being lifted above the roof. On a
5 m atmosphere grid with 20 m buildings the partial columns are the ring of
cells along each footprint edge, so the option is a refinement of the edge
rather than a large change; it matters more as buildings approach the grid
spacing. With :cpp:`erf.fire.fire_debug` the coupling prints the
column-integrated heating against the flux handed in, whose ratio is
:math:`1 - e^{-z_{top}/\alpha_g}` (one for any realistic domain top, and
about :math:`\rho` times that with the legacy tendency form), the largest
below-roof share among partial columns, and the largest potential
temperature of the state under a roof.

Smoke tracer
------------

With :cpp:`erf.fire.smoke_enable` a passive tracer ``RhoSmoke`` is added to
the state and fed at the surface with

.. math::

   F_{smoke} = \epsilon\, \frac{Q}{H_c} \quad [\mathrm{kg\,m^{-2}\,s^{-1}}],

where :math:`\epsilon` = :cpp:`erf.fire.smoke_emission_factor` (kg of smoke
per kg of fuel, default 0.02) and :math:`H_c` =
:cpp:`erf.fire.smoke_heat_of_comb` (J per kg of fuel, default
1.87 × 10⁷). The emission uses the same lagged, coarsened heat flux as the
energy injection and is placed in the lowest atmospheric cell. The tracer is
advected and diffused like any other scalar and is available in the
atmospheric plotfile.

References
----------

- Andrews, P. L. (2012). Modeling wind adjustment factor and midflame wind speed for Rothermel's surface fire spread model. USDA Forest Service General Technical Report RMRS-GTR-266.
- Rothermel, R. C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service Research Paper INT-115.
- Rothermel, R. C. (1983). How to predict the spread and intensity of forest and range fires. USDA Forest Service General Technical Report INT-143.
- Mandel, J., Beezley, J. D. and Kochanski, A. K. (2011). Coupled atmosphere-wildland fire modeling with WRF 3.3 and SFIRE 2011. Geoscientific Model Development, 4, 591-610.
- Finney, M. A. (1998). FARSITE: Fire Area Simulator model development and evaluation. USDA Forest Service Research Paper RMRS-RP-4.
