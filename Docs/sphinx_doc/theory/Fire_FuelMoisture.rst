.. role:: cpp(code)
   :language: c++

.. _sec:FireFuelMoisture:

Fuel Moisture
=============

Fuel moisture enters every rate-of-spread model through the moisture damping
of the reaction intensity and, for Balbi, through the energy needed to bring
the fuel to ignition. ERF-Fire carries five moisture classes per fire cell in
``fire_fuel_mc``: dead 1-hour, 10-hour and 100-hour fuels, and live
herbaceous and live woody fuels. The dead classes can be held fixed or
advanced with the atmosphere; the live classes are fixed.

Static moisture
---------------

With :cpp:`erf.fire.moisture_dynamic = false` the dead classes stay at
:cpp:`erf.fire.moisture_1hr`, :cpp:`erf.fire.moisture_10hr` and
:cpp:`erf.fire.moisture_100hr` (fractions of dry mass, defaults 0.08, 0.08
and 0.10) for the whole run, and the live classes at
:cpp:`erf.fire.moisture_live` (default 0.60). This is the configuration
the canonical rate-of-spread tests use.

Dynamic dead-fuel moisture
--------------------------

With :cpp:`erf.fire.moisture_dynamic = true` (the default in the parameter
struct, so a deck that does not set it gets the dynamic model) each dead
class in every fire cell follows the time-lag equation of Nelson (2000),
advanced by forward Euler once per atmospheric step:

.. math::

   M_{n+1} = M_n + \Delta t\,\Bigl[\frac{M_e - M_n}{\tau\, f_T} + P\Bigr],

with :math:`\Delta t` in hours, :math:`\tau` the class time lag of 1, 10 or
100 hours, and the result clamped to :math:`[0.01, 0.40]`.

**Equilibrium moisture** :math:`M_e` is Nelson's fourth-degree polynomial in
the relative humidity fraction :math:`H`, on the adsorption (wetting) curve
:math:`0.0323 + 0.281 H + 0.409 H^2 - 1.356 H^3 + 1.660 H^4` or the
desorption (drying) curve
:math:`0.0580 + 0.199 H + 0.625 H^2 - 1.183 H^3 + 1.057 H^4`, each clamped
to :math:`[0, 0.35]`. The curve is chosen by hysteresis: a fuel wetter than
the desorption value dries toward it, one drier than the adsorption value
wets toward it, and one between the two relaxes to their mean.

**Temperature correction.** The time lag is scaled by
:math:`f_T = \exp(-0.015\,(T - 20^\circ\mathrm{C}))`, clamped to
:math:`[0.5, 2]`, so warm fuel responds faster.

**Precipitation.** A uniform rate :cpp:`erf.fire.precip_rate_mm_hr` adds a
wetting term :math:`P = 0.01 \times` rate per hour once the rate exceeds
0.1 mm/h; there is no rain from the atmosphere yet.

The drivers are the potential temperature and relative humidity of the
lowest atmospheric cell, sampled onto the fire grid each step and also
written as ``fire_surface_temp_K`` and ``fire_surface_rh``. Moisture is
advanced before the rate of spread is evaluated, so the spread responds
within the same step. The Rothermel, Balbi and Cheney-Gould coefficients are
rebuilt each step from the domain-average moisture; Balbi can instead take
the per-cell value with :cpp:`erf.fire.balbi.use_cell_moisture`, and the
BEHAVE path uses the per-cell dead classes and the live classes through its
dynamic live-to-dead transfer, controlled by
:cpp:`erf.fire.behave.dynamic_transfer_lo` and ``_hi``.

Stick model
-----------

:cpp:`erf.fire.moisture_model = "stick"` (default ``"timelag"``) replaces
the single time constant with the diffusion of moisture through a
cylindrical fuel particle, the framework of Nelson (2000): for each dead
class in every fire cell,

.. math::

   \frac{\partial M}{\partial t} = \frac{1}{r}\frac{\partial}{\partial r}
   \Bigl(r D \frac{\partial M}{\partial r}\Bigr), \qquad 0 < r < R_c,

on :cpp:`erf.fire.stick.n_shells` shells (default 6), advanced implicitly,
with the surface held at the air's equilibrium moisture (the same
adsorption/desorption hysteresis, applied to the surface shell) or at
:cpp:`erf.fire.stick.rain_surface_moisture` (0.35) while it rains. The
diffusivity is set so that the slowest radial mode has the class's time lag,
:math:`D_c = R_c^2 / (\lambda_1^2 \tau_c)` with :math:`\lambda_1 = 2.405`
the first zero of :math:`J_0`, scaled by the temperature factor above and by
:cpp:`erf.fire.stick.diffusivity_scale`; the radii
:cpp:`erf.fire.stick.radius_cm` default to 0.15, 0.635 and 2.5 cm (the
10-h value is the standard stick). The class moisture the rate-of-spread
models see is the volume average of the shells, so the long-time response
equals the time-lag model's while the short-time response is diffusive: the
surface follows the air within minutes and the core lags by hours, which
is the diurnal shape Nelson measured. The shells are carried in
``fire_stick_mc`` and checkpointed. Not reproduced from Nelson: his species
constants for the diffusivity, bound-water isotherm and surface exchange,
and the solar heating of the stick. ``Exec/RegTests/FireStickMoisture``
checks the plumbing and the restart, and the unit test
``ERF_GTestFuelMoistureStick`` the relaxation, the calibrated lag, the
surface-before-core ordering and the rain condition.

Moisture of extinction
----------------------

Every Anderson fuel model carries a moisture of extinction :math:`M_x`, the
dead moisture at which spread stops. The Rothermel and BEHAVE kernels use
the fuel model's tabulated value in the moisture damping coefficient, and
Balbi zeroes its rate at and above that value when
:cpp:`erf.fire.balbi.use_moisture_extinction` is set. A surface-area-to-volume
dependent estimate,

.. math::

   M_x = 0.12 + 0.28\, (\sigma / 1739)^{-0.3},

clamped to :math:`[0.12, 0.45]`, is evaluated per cell from the load-weighted
:math:`\sigma` of the local fuel and held in the field ``fire_mext``. It is
presently diagnostic: the kernels still read the tabulated value, and
:cpp:`erf.fire.use_dynamic_mext` is accepted but not yet consumed.

Limitations
-----------

- Moisture is uniform within a fire cell and there is no fuel-bed depth
  profile.
- Rain comes only from the uniform input rate; atmospheric precipitation is
  not yet passed to the fuel.
- Live moisture is constant. Curing of the live herbaceous load is available
  to the Balbi 2020 form through :cpp:`erf.fire.balbi.herb_curing`.
- The forward Euler step is accurate while the atmospheric step is much
  shorter than the shortest time lag (one hour), which holds for every
  practical ERF step.

References
----------

- Nelson, R. M. (2000). Prediction of diurnal change in 10-h fuel stick moisture content. Canadian Journal of Forest Research, 30, 1071-1087.
- Van Wagner, C. E. (1972). Equilibrium moisture contents of some fine forest fuels in eastern Canada. Canadian Forestry Service Information Report PS-X-36.
- Rothermel, R. C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service Research Paper INT-115.
- Anderson, H. E. (1970). Forest fuel ignitability. Fire Technology, 6(4), 312-319.
