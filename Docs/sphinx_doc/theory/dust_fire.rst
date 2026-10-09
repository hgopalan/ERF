.. role:: cpp(code)
   :language: c++

.. _sec:DustFire:

Fire-Dust Coupling
==================

ERF-Hazard couples the fire model of :ref:`sec:Fire` to the dust model
through three one-way interactions, switched on together by
:cpp:`erf.fire_dust_coupling` (an ``erf`` key, not ``erf.dust``) and then
selected individually. Both models must be enabled and
:cpp:`erf.dust.grid_ratio` must equal :cpp:`erf.fire.grid_ratio`; the code
aborts otherwise. Fire fields live on the fire grid and are mapped to dust
cells by physical position, copied first onto the dust distribution so ranks
never read remote boxes. All three interactions were verified on the
``Hazard/FireDust*`` cases; the ``FireDustBaseline`` case runs both models
with the coupling off as the control.

Burned area removes the crust
-----------------------------

Every dust step the crust index is reset to its baseline (the input value,
the crust raster, or the last PHREEQC update) and
then, in every dust cell whose fire cell has a negative level set (burned),
reduced to

.. math::

   C_I \leftarrow \max\big(C_I\, (1 - r),\, 0\big), \qquad r = \texttt{erf.fire\_dust\_crust\_reduction}

before the threshold is recomputed, so the lower threshold follows the current
burned area rather than accumulating. With :math:`r = 0.99` and a fully
crusted surface the threshold in burned cells drops by the factor
:math:`(1 + 0.5 \cdot 0.01)/(1 + 0.5)`, about a third, and emission rises
there. ``FireDustInteraction1`` isolates this path.

Fire outflow wind raises the friction velocity
----------------------------------------------

With :cpp:`erf.fire_dust_wind_to_dust` (on by default once the coupling is
on) the fire's wind at :cpp:`erf.fire.wind_ref_ht` (``fire_wind_ref``, the
reference wind before the wind adjustment factor; it is the datum
:cpp:`erf.fire_dust_wind_zref` names) is copied cell for cell onto the dust
grid and converted to a friction velocity by the log law

.. math::

   u_{*,\mathrm{fire}} = \frac{\kappa\, |U_\mathrm{fire}|}{\ln(z_\mathrm{ref}/z_0)}

with :cpp:`erf.fire_dust_wind_zref` and :cpp:`erf.fire_dust_wind_z0`; the
larger of it and the surface-layer value is kept. Cells at and downwind of
the perimeter therefore see the fire's wind where it exceeds the ambient
one; with no fire modification of the near-surface flow the two friction
velocities agree, so the fire path wins only where the fire accelerates the
flow. (Until October 2026 the WAF-reduced midflame wind ``fire_wind_eff`` was
handed over as if it were the wind at ``wind_ref_ht``, 0.36x for grass, so
the fire path never exceeded the surface layer's :math:`u_*`.)
``FireDustInteraction2`` uses the weak-wind sounding and
``FireDustWindStrength`` varies the ambient wind.

Fire heat lofts the dust
------------------------

With :cpp:`erf.fire_dust_lofting_enabled` the emission flux of every bin is
multiplied, inside the dust step right after the blast and haul-road
additions and before the critical-material budget, the super-particle
release and the diagnostics read it (until October 2026 it was applied after
the dust step, so the budget and the particles booked a third of what the
atmosphere received in burning cells), by

.. math::

   1 + \min\!\left(k\, \frac{\max(Q_\mathrm{fire} - Q_\mathrm{thr}, 0)}{Q_\mathrm{ref}},\; k\right)

with :math:`k` = :cpp:`erf.fire_dust_lofting_k_loft`, :math:`Q_\mathrm{thr}` =
:cpp:`erf.fire_dust_lofting_Q_threshold`, :math:`Q_\mathrm{ref}` =
:cpp:`erf.fire_dust_lofting_Q_ref` and :math:`Q_\mathrm{fire}` the fire's
sensible heat flux [W/m²] on the dust cell. The factor saturates at
:math:`1 + k` once the heat flux exceeds :math:`Q_\mathrm{thr} + Q_\mathrm{ref}`,
which every fuel model does in the canonical cases (fuel model 1 peaks near
3.5 kW/m², chaparral near 290 kW/m²), so the cap rather than the slope sets
the response there; ``FireDustLoftingScaling`` is the place to vary the
three constants. This is an emission enhancement standing in for the
convective column above the front; the plume itself is whatever the
atmosphere resolves, and the fire's heat is injected into it (or not) by the
fire's own :cpp:`erf.fire.coupling_type`.

Combined cases
--------------

``FireDustInteractions12`` and ``FireDustInteractions123`` switch the paths
on together, ``FireDustMassConservation`` checks the dust budget with the
fire on, ``FireDustTerrainCoupled`` adds terrain, and ``FireSmokeDustCoupled``
runs the fire's smoke tracer alongside the dust scalar.
