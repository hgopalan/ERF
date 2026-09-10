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
advanced with the atmosphere; how the live classes move with them is set by
:cpp:`erf.fire.moisture_live_model`.

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

**Equilibrium moisture** :math:`M_e` comes from a wetting (adsorption) curve
:math:`E_w` and a drying (desorption) curve :math:`E_d`, chosen by
:cpp:`erf.fire.emc_model`:

- ``"legacy"`` (default): fourth-degree polynomials in the relative humidity
  fraction :math:`H`, :math:`E_w = 0.0323 + 0.281 H + 0.409 H^2 - 1.356 H^3 + 1.660 H^4`
  and :math:`E_d = 0.0580 + 0.199 H + 0.625 H^2 - 1.183 H^3 + 1.057 H^4`,
  each clamped to :math:`[0, 0.35]`, with :math:`H` clamped to
  :math:`[0.01, 0.99]` and no temperature dependence. They were ported from
  ``wildfire_levelset`` with an attribution to Nelson (2000), which does not
  contain them: Nelson fits Hart's (1977) isotherm to Blackmarr's (1971)
  adsorption and desorption data and averages the two into one isotherm. The
  constant and linear terms are those of Simard's (1968) equation for
  RH below 10 % (RH and EMC in percent, temperature in °F) applied to
  fractions; no source is known for the rest. The curves run 60-80 % above
  the published ones between 10 and 60 % RH, cross near 66 % RH and both
  reach the 0.35 cap above 70 %.
- ``"van_wagner"``: the drying and wetting equilibrium moisture of the
  Canadian Fine Fuel Moisture Code (Van Wagner and Pickett, 1985; Van Wagner,
  1987, eqs. 2a-2b; Viney, 1991, eqs. 7-8), the pair WRF-SFIRE uses
  (Mandel et al., 2014). With :math:`h` the relative humidity in percent and
  :math:`T` the temperature in °C, in percent of dry mass,

  .. math::

     E_d = 0.942\, h^{0.679} + 11\, e^{(h-100)/10} + 0.18\,(21.1 - T)\,(1 - e^{-0.115 h}),

     E_w = 0.618\, h^{0.753} + 10\, e^{(h-100)/10} + 0.18\,(21.1 - T)\,(1 - e^{-0.115 h}),

  divided by 100 and floored at zero; :math:`h` is clamped to
  :math:`[0, 100]`. (WRF-SFIRE's ``module_fr_sfire_phys.F`` writes the
  exponential terms as ``0.4994e-4*exp(0.1*H)`` and ``0.4540e-4*exp(0.1*H)``,
  a tenth of :math:`11 e^{-10}` and :math:`10 e^{-10}`; ERF follows the
  published form, as the ``cffdrs`` R package does.)

At 20 °C the two choices give:

.. list-table::
   :header-rows: 1

   * - RH [%]
     - legacy :math:`E_w`
     - legacy :math:`E_d`
     - van_wagner :math:`E_w`
     - van_wagner :math:`E_d`
   * - 0 (dry air)
     - 0.035
     - 0.060
     - 0.000
     - 0.000
   * - 10
     - 0.063
     - 0.083
     - 0.036
     - 0.046
   * - 20
     - 0.097
     - 0.115
     - 0.061
     - 0.074
   * - 40
     - 0.166
     - 0.189
     - 0.102
     - 0.118
   * - 60
     - 0.270
     - 0.284
     - 0.139
     - 0.156
   * - 80
     - 0.350
     - 0.350
     - 0.183
     - 0.202
   * - 95
     - 0.350
     - 0.350
     - 0.253
     - 0.276

A deck without an atmospheric moisture model hands the fuel zero relative
humidity: the legacy curves clamp it to 1 % and hold the dead classes near
0.035-0.060, while ``van_wagner`` lets them dry to the 0.01 floor.
``Exec/RegTests/FireEmcModel`` runs both choices in dry air and at about 40 %
RH, and the unit test ``ERF_GTestFuelMoistureEMC`` checks the curves against
the published equations and the long-time equilibrium of both update models.

The curve is chosen by sorption hysteresis, as in the WRF-SFIRE dead-fuel
model (Vejmelka et al. 2016): a fuel wetter than :math:`E_d` dries toward it,
a fuel drier than :math:`E_w` wets toward it, and a fuel between the two does
not change (:math:`M_e = M_n`, so only rain moves it). Nelson (2000) draws the
same line, desorption when the fuel is wetter than its equilibrium and
adsorption when it is drier. Where the legacy polynomials cross (66-70 %
relative humidity) the band is taken between the lower and the upper value.
In constant dry air a fuel starting at :math:`M_0 > E_d` therefore follows
:math:`M(t) = E_d + (M_0 - E_d)\,e^{-t/(\tau f_T)}`, which the unit test
``ERF_GTestFuelMoisture`` and the verification case
``Exec/CanonicalTests/Fire/Verification/Moisture_Relaxation`` check.

.. note::

   Until September 2026 the choice was reversed: a fuel above the adsorption
   curve relaxed toward it and one below the desorption curve toward that, so
   a drying fuel headed for the lower curve and was then held at the upper
   one, and a fuel drier than both wetted toward the upper one. Runs with
   dynamic moisture made before the fix differ from runs made after it.

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
dynamic live-to-dead herbaceous transfer, whose moisture window is
:cpp:`erf.fire.behave.dynamic_transfer_lo` and ``_hi``
(:ref:`sec:ROS_Behave`).

Live moisture
-------------

With dynamic moisture the live herbaceous and live woody classes follow
:cpp:`erf.fire.moisture_live_model`:

- ``"legacy"`` (the default, so existing runs reproduce) passes each live
  class through the dead-fuel update above with the 100-hour lag and no rain,
  then bounds it to :math:`[0.30, 2.50]`. The dead-fuel update clamps its input
  to :math:`[0.01, 0.40]` and relaxes toward the dead-fuel equilibrium of
  :cpp:`erf.fire.emc_model`, so a live moisture above 0.40 (the default
  :cpp:`erf.fire.moisture_live` is 0.60) drops to 0.40 or below on the first
  step, even in saturated air, and then decays toward the 0.30 bound over tens
  of hours. This is not a live-moisture model; it is kept for backward
  compatibility.
- ``"fixed"`` holds the live classes where they start, at
  :cpp:`erf.fire.moisture_live`, or at the checkpointed value on a restart.
  Live moisture follows the plant's water status over days to weeks, not the
  air over the length of a fire run, so this is the recommended setting.

The evolving live classes reach only the BEHAVE model, per cell or as the
domain average: through the live moisture damping and through the transfer
of live herbaceous load to the dead class, which grows as the live herbaceous
moisture falls. A fuel without a live load (Anderson model 1, for example) is
unaffected. Rothermel, Balbi and the Scott-Burgan curing transfer read
:cpp:`erf.fire.moisture_live` directly.
``Exec/RegTests/FireLiveMoisture`` runs the two settings side by side, and
``ERF_GTestFuelMoistureLive`` checks both updates.

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
- There is no live-moisture model: the live classes are held
  (``"fixed"``) or carried through the dead-fuel update (``"legacy"``).
  Curing of the live herbaceous load is available to the Balbi 2020 form
  through :cpp:`erf.fire.balbi.herb_curing`.
- The forward Euler step is accurate while the atmospheric step is much
  shorter than the shortest time lag (one hour), which holds for every
  practical ERF step.

References
----------

- Nelson, R. M. (2000). Prediction of diurnal change in 10-h fuel stick moisture content. Canadian Journal of Forest Research, 30, 1071-1087.
- Van Wagner, C. E. (1972). Equilibrium moisture contents of some fine forest fuels in eastern Canada. Canadian Forestry Service Information Report PS-X-36.
- Van Wagner, C. E., and Pickett, T. L. (1985). Equations and FORTRAN program for the Canadian Forest Fire Weather Index System. Canadian Forestry Service, Forestry Technical Report 33.
- Vejmelka, M., Kochanski, A. K., and Mandel, J. (2016). Data assimilation of dead fuel moisture observations from remote automated weather stations. International Journal of Wildland Fire, 25, 558-568.
- Van Wagner, C. E. (1987). Development and structure of the Canadian Forest Fire Weather Index System. Canadian Forestry Service, Forestry Technical Report 35.
- Viney, N. R. (1991). A review of fine fuel moisture modelling. International Journal of Wildland Fire, 1(4), 215-234.
- Simard, A. J. (1968). The moisture content of forest fuels. Part III: moisture content variations of fast responding fuels below the fibre saturation point. Canadian Forest Service Information Report FF-X-16.
- Mandel, J., et al. (2014). Recent advances and applications of WRF-SFIRE. Natural Hazards and Earth System Sciences, 14, 2829-2845.
- Rothermel, R. C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service Research Paper INT-115.
- Anderson, H. E. (1970). Forest fuel ignitability. Fire Technology, 6(4), 312-319.
