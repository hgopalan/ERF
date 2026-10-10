.. role:: cpp(code)
  :language: c++

.. _sec:WUIStructureIgnition:

Structure Ignition and House-to-House Spread
============================================

With :cpp:`erf.fire.structures.ignition.enable` the structures of the exposure
diagnostics (:ref:`sec:FireOutput`) stop being passive targets. Each one
carries a state, unignited, burning or burned out; it ignites from the
accumulators the exposure diagnostics already keep, and once burning it is a
heat source, an ember source and a radiator that loads the accumulators of
the structures around it. House-to-house spread therefore needs no rule of
its own: it is the same exposure path with a burning house at the other end.
The option needs :cpp:`erf.fire.exposure.enable` (and so
:cpp:`erf.fire.structures.enable`); the ember source additionally needs
:cpp:`erf.fire.spotting.enable`. Everything is off by default and a deck
without the option runs exactly as before.

Ignition rule
-------------

At the end of every fire step each unignited structure is tested against
three criteria on its *wall band* (the burnable cells within
:cpp:`erf.fire.exposure.ring` fire cells of its footprint) and its footprint:

1. **Heat load.** The largest accumulated heat load :math:`\int (Q + q_r)\,dt`
   in the band reaches :cpp:`erf.fire.structures.ignition.heat_load_J_m2`,
   where :math:`Q` is the fire's own release at the ground there and
   :math:`q_r` the incident radiant flux of burning structures described below.
2. **Embers.** The number of brands that landed on the footprint reaches
   :cpp:`erf.fire.structures.ignition.ember_count`.
3. **Intensity.** The largest fireline intensity in the band has spent
   :cpp:`erf.fire.structures.ignition.residence_s` above
   :cpp:`erf.fire.structures.ignition.intensity_kW_m`. The time above the
   threshold accumulates over the run and is not reset when the intensity
   drops, so an intermittent exposure counts in full.

The first criterion met, in that order, is the recorded cause. A threshold
that is zero or negative turns its criterion off; at least one must be on.
The ignition time is the end of the step, the convention of the arrival time.

The thresholds are **placeholders to be calibrated**, not measured constants:

- ``heat_load_J_m2 = 6.0e6``: 20 kW/m² held for five minutes. Babrauskas
  (2003) gives about 12.5 kW/m² as the critical flux for the piloted ignition
  of wood, and Cohen (2004) found that wood walls ignited only under fluxes of
  that order sustained for minutes; a heat-load criterion folds flux and time
  into one number, which the ignition of thick wood does not obey exactly.
- ``ember_count = 50`` brands on the footprint: no source; SWUIFT
  (Masoudvaziri et al. 2021) ignites structures from embers with a
  probability per landed brand rather than a count.
- ``intensity_kW_m = 1000`` for ``residence_s = 60``: a fireline intensity of
  1 MW/m corresponds to a Byram flame length of about 3.5 m, a wall height;
  the minute is a guess.

Heat release of a burning structure
-----------------------------------

A burning structure releases heat on every cell of its footprint following
the EN 1991-1-2 Annex E.4 heat-release curve per unit floor area: a t-squared growth
:math:`q = q_{peak} (\tau / t_g)^2` from the ignition time to
:cpp:`erf.fire.structures.ignition.growth_time_s`, a plateau at
:cpp:`erf.fire.structures.ignition.peak_flux_W_m2` until 70 % of
:cpp:`erf.fire.structures.ignition.fuel_load_J_m2` has been released, and a
linear decay to zero that releases the remaining 30 %. The growth phase
releases :math:`q_{peak} t_g / 3`, so the run stops at start-up if that
exceeds 70 % of the load. When the curve ends the structure is burned out.

The defaults are the Eurocode dwelling: 250 kW/m² for the peak (Table E.5,
rate of heat release per unit floor area for dwellings) and 780 MJ/m² for the
load (Table E.4, average fire load density for dwellings). Together they burn
for 74 minutes. The time to the peak, 600 s, is a placeholder: the Eurocode
prescribes a growth to 1 MW in :math:`t_\alpha = 300` s for dwellings, whose
time to the peak of a whole house depends on its floor area (about 30 minutes
for 150 m²), and ten minutes to full involvement is within the range NIST
reported for single-family homes in WUI fires.

The release is added to ``fire_heat_flux`` on the footprint cells, so it goes
to the atmosphere through the coupling of :ref:`sec:FireCoupling` like the
fire's own heat, with two consequences worth knowing. With
:cpp:`erf.fire.heat_open_fraction` the heat is placed above the roof of the
columns the footprint blocks; without it the legacy profile starts at the
ground inside the blocked column. And the full release is injected, radiant
fraction included, as it is for burning fuel (the radiative fraction of the
fire's release, backlog row 4, is future work); the radiant fraction below is
used for the exposure of the neighbours only, not subtracted from the
injection.

Radiation onto the neighbours
-----------------------------

Every footprint cell of a burning structure is a ground-level point source of
radiant power :math:`P = \chi_r\, q\, \Delta x\, \Delta y`, with
:math:`\chi_r` = :cpp:`erf.fire.structures.ignition.rad_fraction`. The
incident flux on a fire cell at horizontal distance :math:`r` is

.. math::

   q_r = \sum_{\text{sources}} \frac{P}{2 \pi \max(r^2, (\Delta x / 2)^2)}

over the sources within :cpp:`erf.fire.structures.ignition.rad_radius_m`
(the hemisphere above the ground; the floor keeps a source's own cell
finite). It is written to ``fire_structure_rad_flux`` and added to the
heat-load accumulator of every cell, which is how a burning house loads the
wall bands of the houses around it. The radiant fraction default, 0.3, is in
the 0.2-0.4 range the SFPE Handbook gives for most fuels and is a placeholder
here; the cutoff of 100 m is numerical. The point source is the simplest
form: a view-factor model of a flame panel of the wall height, which Cohen
(2004) used, would be the next step, and the flux is evaluated at the ground
rather than on a vertical wall. Nothing blocks the radiation, so a house
shadows nothing.

Embers from a burning structure
-------------------------------

With spotting on, the footprint cells of a burning structure are launch
sites of the Albini model (:ref:`sec:FireSpottingCrown`) with the equivalent
fireline intensity :math:`I = q\, \Delta x` (the release per unit length
across one fire cell of the footprint), so the lofting height follows the
curve: 250 kW/m² on 5 m cells gives 1250 kW/m and a lofting height of 131 m.
The launch probability and the trajectory are the fire's. A brand that lands
on burnable fuel ignites a spot fire; one that lands on a footprint counts in
that structure's ember accumulator and can ignite it by the ember criterion.

State, output and restart
-------------------------

The state lives on the fire grid as a four-component field on the footprint
cells: the state code (0 unignited, 1 burning, 2 burned out), the ignition
time (-1 until ignited), the time above the intensity threshold and the cause
code (0 none, 1 heat, 2 ember, 3 intensity). The fire plotfile gains
``fire_structure_state``, ``fire_structure_ignition_time`` and
``fire_structure_rad_flux``; the checkpoint carries the field as
``FireStructureState`` and a restart rebuilds the per-structure state from
it. The exposure CSV gains five columns, only when the option is on so that
positional readers of the plain CSV keep working: ``state``,
``t_ignition_s``, ``cause``, ``structure_flux_Wm2`` (the release now) and
``incident_flux_max_Wm2`` (the largest incident radiant flux in the band). A
line ``[FIRE STRUCTURE] structure n ignited at t= cause=`` is printed at each
ignition with the accumulator values that fired it, and one at burnout; the
exposure summary line adds the burning and burned-out counts.

Verification
------------

``Exec/RegTests/FireStructureIgnition`` lights a grass fire against the wall
of house A, with house B 20 m downwind and house C 90 m further, and runs
40 steps of 0.5 s with the old path (ignition off), the new path with the
heat and ember criteria, a radiation-only variant (no spotting, ember
criterion off) and a restart from step 20. Its thresholds and curve are
scaled to the 20 s run and are not the defaults above. The checks: A ignites
from the heat load at its wall at 1 s; a second house ignites later (B at
3.5 s by embers, C at 12 s by embers in the ember variant; B at 7 s by heat in
the radiation variant, where the front never reaches its band and C stays
unignited); A releases its 250 kW/m² while burning and burns out at 13 s
with zero release after; the plotfile state agrees with the CSV; the incident
flux in the step-10 plotfile equals the point-source sum recomputed from A's
footprint in the checker; the restart reproduces the last CSV rows; and the
same checker run on the old path fails, since that CSV has no state column.
``FireStructureIgnition_on`` is the CTest smoke test and
``FireStructureIgnition`` (MPI builds) runs the script;
``FireStructureIgnition_no_exposure_abort`` and
``FireStructureIgnition_curve_abort`` prove the start-up checks. The
``ignition`` variant of :ref:`sec:WUIValidation` runs the documented defaults
on the subdivision.

The unit tests (``Tests/Unit/Fire/ERF_GTestStructureIgnition.cpp``) check the
shape and the energy budget of the burn curve (its integral is the fuel load
to 1e-6), the rejected inputs, the point-source kernel and the precedence and
switches of the ignition rule.

Limitations
-----------

- The thresholds are placeholders; the curve is the Eurocode dwelling, not
  a WUI structure. Calibration against an observed event is open.
- Radiation is a ground point source without shadowing or a view factor; the
  target is the ground cell, not a wall, and the radiant fraction is not
  removed from the heat injected into the atmosphere.
- The ember source is the surface-fire spotting model with an equivalent
  intensity; no distribution of brand sizes or fall speeds (backlog row 10).
- The per-structure reductions run on the host every step over the wall bands,
  as the exposure CSV does; with thousands of structures on a large fire grid
  they cost as much as the spotting model's host copy.
- A structure's own state does not change the fire around it: its footprint
  stays non-burnable and its fuel zero, so its release is not a fuel budget.

References
----------

- EN 1991-1-2 (2002). Eurocode 1: Actions on structures. Part 1-2: General actions, actions on structures exposed to fire. Annex E. CEN, Brussels.
- Babrauskas, V. (2003). Ignition Handbook. Fire Science Publishers, Issaquah.
- Cohen, J. D. (2004). Relating flame radiation to home ignition using modeling and experimental crown fires. Canadian Journal of Forest Research, 34, 1616-1626.
- Masoudvaziri, N., Szasdi Bardales, F., Keskin, O. K., Sarreshtehdari, A., Sun, K., Elhami-Khorasani, N. (2021). Streamlined wildland-urban interface fire tracing (SWUIFT): Modeling wildfire spread in communities. Environmental Modelling & Software, 143, 105097.
- Albini, F. A. (1983). Potential spotting distance from wind-driven surface fires. USDA Forest Service Research Paper INT-309.
