.. role:: cpp(code)
   :language: c++

.. _sec:FireSpottingCrown:

Ember Spotting and Crown Fire
=============================

Both extensions are off by default and add fields, plotfile variables and
checkpoint state only when enabled.

Ember spotting
--------------

:cpp:`erf.fire.spotting.enable` turns on a stochastic firebrand model after
Albini (1983). Every :cpp:`erf.fire.spotting.spotting_interval` fire
subcycles (default every subcycle), on the host and with the same random
draws on every rank:

1. Front cells are identified: :math:`\phi` at or below the FARSITE threshold
   and a positive rate of spread.
2. The Byram fireline intensity :math:`I_B` [kW/m] of each front cell is
   formed from the rate of spread, the fuel load and the heat content.
3. Cells below :cpp:`erf.fire.spotting.I_B_min` (default 100 kW/m) launch
   nothing. Otherwise the lofting height is

   .. math::

      H_z = 12.2\, I_B^{1/3} \quad [\mathrm{m}],

   Albini's equation 1 for wind-driven surface fires.
4. A brand is launched with probability :cpp:`erf.fire.spotting.P_base` per
   front cell per application (default 0.01).
5. The brand falls at :cpp:`erf.fire.spotting.terminal_velocity` (default
   0.5 m/s) while drifting with the effective wind, integrated by forward
   Euler over :cpp:`erf.fire.spotting.n_traj_steps` sub-steps (default 20)
   from :math:`H_z` to the terrain surface, which is interpolated under the
   brand so that drift downslope lands farther away.
6. The landing distance is capped at the per-fuel maximum spotting distance
   of Scott (2006) and Albini for the fuel system selected by
   :cpp:`erf.fire.spotting.fuel_system`, ``"13"`` (Anderson) or ``"40"``
   (Scott and Burgan).
7. The brand ignites with probability :cpp:`erf.fire.spotting.P_catch`
   (default 1) and only if the landing cell still holds at least
   :cpp:`erf.fire.spotting.reentry_fuel_thresh` (default 0.05) of its initial
   fuel load, so brands landing in the burned area or on consumed fuel are
   discarded.
8. A disc of radius :cpp:`erf.fire.spotting.spot_radius` (default 10 m) is
   stamped negative into :math:`\phi` at the landing point, in the
   convention of the propagation path: normalised by the radius on the
   FARSITE path, as a signed distance in metres on the level-set path. A
   brand that lands on a non-burnable cell is dropped and, with the
   exposure diagnostics on, counted on that cell.

.. note::

   Until the exposure work the spotting step clamped :math:`\phi` to
   :math:`[-1, 1]` on every call regardless of the path. On the level-set
   path, where :math:`\phi` is a distance in metres, that squashed the
   distance function every spotting step and changed the spread even when
   no brand landed. Level-set runs with spotting on therefore differ from
   earlier builds; FARSITE runs do not.

:cpp:`erf.fire.spotting.random_seed` fixes the draws for reproducibility;
zero seeds from the clock. The four-component diagnostic field
``fire_albini_data`` records, per cell, the lofting height at source cells,
the number of brands launched, the maximum landing distance reached from
that cell, and a flag marking cells that received a spot ignition; the
statistics CSV adds the number of spot fires per step and the largest
landing distance. The spotting sources and landings are on the fire grid
and use the fire-grid wind, so brands do not see the three-dimensional
plume; the terrain reader supplies the ground under the trajectory.

Crown fire
----------

:cpp:`erf.fire.crown.enable` adds the transition from surface fire to crown
fire and an active crown rate of spread. The canopy is described by uniform
parameters: base height :cpp:`erf.fire.crown.canopy_base_ht` (m), bulk
density :cpp:`erf.fire.crown.canopy_bulk_den` (kg/m³), depth
:cpp:`erf.fire.crown.canopy_depth` (m), foliar moisture
:cpp:`erf.fire.crown.foliar_moisture` (fraction) and heat content
:cpp:`erf.fire.crown.h_crown_BTU_lb`. Cruz's regression was fitted to North
American conifer stands and is not meant for grass or shrub fuel models.

**Initiation.** A surface fire crowns where its Byram intensity exceeds the
Van Wagner (1977) critical intensity,

.. math::

   I_{crit} = 0.010\, C_{BH}\, \bigl(460 + 25.9\, (M_{fol} - M_c) \times 100\bigr) \quad [\mathrm{kW/m}],

with :math:`C_{BH}` the canopy base height and :math:`M_c` =
:cpp:`erf.fire.crown.M_c` (default 0.30) the critical foliar moisture; a
foliar moisture at or below :math:`M_c` makes crowning unattainable. Once a
cell has crowned it stays crowned.

**Active crown rate of spread.** :cpp:`erf.fire.crown.ros_model` selects

- ``"cruz"`` (default), Cruz, Alexander and Wakimoto (2005),

  .. math::

     R_{crown} = \frac{11.02}{60}\, U_{10}^{0.90}\, C_{BD}^{0.19}\, e^{-0.17\, M_{10}} \quad [\mathrm{m/s}],

  with :math:`U_{10}` the 10 m wind in km/h (derived from the effective wind,
  or fixed by :cpp:`erf.fire.crown.wind_10m_kmh` when positive) and
  :math:`M_{10}` the 10-hour dead moisture in percent;
- ``"rothermel1991"``, :math:`R_{crown} = 3.34\, R_{surface}`;
- ``"van_wagner_proxy"``, :math:`R_{crown} = (3 / C_{BD})\, f(M_{fol}) / 60`.

With :cpp:`erf.fire.crown.use_passive_blend` the transition is continuous
through the crowning fraction :math:`CF = \min\bigl((I_B/I_{crit})^{2/3}, 1\bigr)`,
:math:`R = (1 - CF)\, R_{surface} + CF\, R_{crown}`; otherwise the crown rate
replaces the surface rate in crowned cells. The crown fraction burned of
Scott and Reinhardt (2001), :math:`(R - R_{surface}) / (R_{crown} - R_{surface})`,
is written as ``fire_crown_fraction_burned``.

**Heat release.** Crowned cells consume the canopy load
:math:`C_{BD} \times` depth at the surface residence time and add its heat
to the sensible flux, with the same exponential injection as the surface
flux but the deeper canopy e-folding height. ``fire_crown_active`` and
``fire_crown_load`` are checkpointed so crowned cells stay crowned across a
restart.

References
----------

- Albini, F. A. (1983). Potential spotting distance from wind-driven surface fires. USDA Forest Service Research Paper INT-309.
- Scott, J. H. (2006). An analytical model for estimating spotting distance from surface fires. In: Fuels Management: How to Measure Success, USDA Forest Service RMRS-P-41.
- Van Wagner, C. E. (1977). Conditions for the start and spread of crown fire. Canadian Journal of Forest Research, 7(1), 23-34.
- Cruz, M. G., Alexander, M. E. and Wakimoto, R. H. (2005). Development and testing of models for predicting crown fire rate of spread in conifer forest stands. Canadian Journal of Forest Research, 35(7), 1626-1639.
- Rothermel, R. C. (1991). Predicting behavior and size of crown fires in the northern Rocky Mountains. USDA Forest Service Research Paper INT-438.
- Scott, J. H. and Reinhardt, E. D. (2001). Assessing crown fire potential by linking models of surface and crown fire behavior. USDA Forest Service Research Paper RMRS-RP-29.
