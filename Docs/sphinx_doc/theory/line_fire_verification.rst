.. role:: cpp(code)
   :language: c++

.. _sec:LineFireVerification:

Line Fire Verification
======================

``Exec/RegTests/FireLineFire`` is the idealized surface line fire of Coen et
al. (2013), the WRF-Fire paper, run without and with an ambient wind. It is
the verification the FIRE-SMART proposal named for the fire module, and it
checks the whole one-way chain, wind extraction, wind reduction, Rothermel and
the level set, against an answer that is known in closed form.

The case
--------

Short grass (Anderson fuel model 1) at 5.5 % moisture on flat ground, the
control moisture of the paper. A 40 m wide line fire is stamped at
initialisation across the whole y extent of a 400 by 80 by 160 m box that is
periodic in x and y, so the fire is a pair of straight fronts: a head fire
spreading with the wind and a backing fire against it. The atmosphere runs at
5 m with a Smagorinsky closure and a MOST surface layer over a 0.03 m
roughness length (the paper's drag coefficient of 0.005), the fire grid at
2.5 m with the level set, the directional Rothermel model and 0.25 s steps.
The WRF-Fire choices are mirrored: the wind is interpolated to 6.1 m and
reduced by the fuel model's wind reduction factor (WRF-Fire's ``windrf(1) =
0.36`` for fuel model 1; here Andrews' unsheltered factor, 0.362 for the 1 ft
grass bed), and there is no midflame wind cap (``erf.fire.use_wind_limit =
false``; WRF-Fire caps R at 6 m/s only). Seven decks share one base: no wind,
a 2.5 m/s sounding (the paper's Control), a 5 m/s sounding (WSHi), the 5 m/s
sounding with ERF's default midflame wind cap turned back on (``wind5_cap``),
and the three winds with the heat coupled back.

Probe cells sit on the line y = 40 m at 1.25 and 3.75 m behind the west edge
of the line and at 1.25, 3.75, 8.75, 18.75, 38.75 and 58.75 m ahead of its
east edge. The rates of spread are the distances between consecutive probes
divided by the differences of their arrival times, so the initial transient
of the stamped line does not enter.

The expected rates
------------------

For an infinite straight line the directional Rothermel model gives the head
fire :math:`R = R_0\,(1 + \phi_w(U_{\rm eff}))` with :math:`U_{\rm eff}` the
wind component along the front normal after the reduction factor, and the
backing fire :math:`R_0`, since the wind component into the fire is clipped
at zero (Coen et al. 2013 make the same choice: "the backing rate of spread
in these experiments is the zero-wind rate of spread"). ``check_linefire.py``
carries an independent port of the Rothermel equations of
``Source/Fire/ERF_Rothermel.cpp`` and evaluates them at the effective wind the
fire reports each step. The surface layer slows the sounding wind at 6.1 m as
the run proceeds (by 15 % over 180 s at 2.5 m/s and 28 % at 5 m/s), and
:math:`\phi_w` grows about as the square of the wind, so each pair of probes is
held to the head rate averaged over its own arrival window rather than to the
rate at the run-mean wind; at 5 m/s the two differ by 15 %. The one-way decks
must match to 10 %.

Results
-------

.. list-table:: Rates of spread [m/s] at 180 s, four ranks, 0.25 s steps
   :widths: 16 10 10 12 12 12 12 22
   :header-rows: 1

   * - Deck
     - U at 6.1 m
     - U_eff
     - R0
     - Backing
     - Head expected
     - Head measured
     - Coen et al. (2013), coupled LES
   * - nowind
     - 0.00
     - 0.00
     - 0.0240
     - 0.0240
     - 0.0240
     - 0.0240
     - NoWind: 0.02 outward, both fronts
   * - wind2p5
     - 2.30
     - 0.83
     - 0.0240
     - 0.0240
     - 0.0960
     - 0.0957
     - Control: 0.22 head (backing not quoted; WRF-Fire sets it to R0)
   * - wind5
     - 4.31
     - 1.56
     - 0.0240
     - 0.0240
     - 0.3097
     - 0.3104
     - WSHi: about 0.40 head
   * - wind5_cap
     - 4.31
     - 1.49 (capped)
     - 0.0240
     - 0.0240
     - 0.2541
     - 0.2541
     - WSHi: about 0.40 head
   * - nowind_2way
     - 1.34 (max)
     - 0.48
     - 0.0240
     - 0.0240
     - (coupled)
     - 0.0240
     - NoWind: 0.02 outward
   * - wind2p5_2way
     - 5.43 (max)
     - 1.96
     - 0.0240
     - 0.0240
     - (coupled)
     - 0.3555
     - Control: 0.22 head
   * - wind5_2way
     - 6.19 (max)
     - 2.24
     - 0.0240
     - 0.0299
     - (coupled)
     - 0.5559
     - WSHi: about 0.40 head

Reading the table: the paper's column is its head fire in a coupled LES,
except the no-wind row, where the fire crept outward at the same rate on
every side. The backing fire is never quoted in the paper, and WRF-Fire sets
it to :math:`R_0` as ERF does, so the like-for-like pairs are backing against
NoWind, the one-way heads against Rothermel at the sampled wind, and the
two-way heads against Control and WSHi.

The backing fire moves at :math:`R_0` to the last digit in every deck but the
coupled 5 m/s one (0.0299 m/s over its single probe pair), the no-wind fire
at :math:`R_0` in both directions, and the one-way head fire within 0.3 % of
Rothermel over its arrival windows in both winds, with the cap or without it.
The no-wind rate itself, 0.024 m/s, is the paper's 0.02 m/s.

The heads of the one-way decks are below the paper's 0.22 and about
0.40 m/s, and are meant to be: those are coupled results in a turbulent
convective boundary layer, where the fire's plume draws the near-surface
wind into the head ("near-fire horizontal winds varied from 2 to 4 m/s" for
the 2.5 m/s Control). At 2.5 m/s the one-way chain is 2.5 m/s in the
sounding, 2.3 m/s at 6.1 m once the surface layer has acted, 0.83 m/s after
the 0.36 reduction factor, a wind factor of 2.8 and a head of 0.09 m/s;
doubling the midflame wind raises the wind factor by about four and the head
to the paper's value.

The midflame wind cap
~~~~~~~~~~~~~~~~~~~~~

Until ``erf.fire.use_wind_limit = false`` was honoured (the flag was parsed
and ignored), every deck here ran with Rothermel's maximum effective wind
speed cap, 300 ft/min (1.52 m/s) for fuel model 1, which holds the head at or
below 0.257 m/s at 5.5 % moisture. ``wind5_cap`` keeps that cap and shows the
ceiling: its first three probe pairs move at 0.256 to 0.258 m/s while the wind
is above the cap, and the last at 0.245 m/s once the wind has fallen below it.
Its head is the 0.2541 m/s that ``wind5`` reported before; uncapped, ``wind5``
runs at 0.3104 m/s.

The two-way decks
~~~~~~~~~~~~~~~~~

The two-way decks are reported, not checked. With the cap, both coupled heads
ran at 0.24 m/s, and that equality was the cap: both runs sample midflame
winds above 1.52 m/s. Uncapped, the coupled heads run at 0.36 m/s at 2.5 m/s
and 0.56 m/s at 5 m/s, against the paper's 0.22 and about 0.40 m/s, and the
two winds evolve differently.

At 5 m/s the head crosses the probes at 0.43 to 0.80 m/s in its first 30 s,
then at 0.26 m/s between 30 and 107 s, and has not reached the probe 58.75 m
ahead of the line by 180 s. The line here spans the whole periodic y extent,
so its plume is two-dimensional and the inflow it draws from the downwind side
has nowhere to come from but against the ambient wind; the directional model
clips a wind component into the fire at zero, which drops the head towards the
no-wind rate. Fire plotfiles of the capped run showed that inflow reversing the
6.1 m wind at the head by 150 s (-1.8 m/s at the head, -2.2 m/s 10 m ahead of
it). The slowdown here is consistent with it, but the uncapped fields have not
been re-examined. At 2.5 m/s the head instead speeds up: 0.25 to 0.34 m/s
until 79 s, then 0.38 m/s to 131 s and 0.54 m/s to 168 s, as the largest
effective wind in the domain rises from 1.6 to 3.7 m/s.

The paper's 1 km line in a 5 km box lets the flow go around the flanks and
feed the head instead. Reproducing the coupled rates needs a finite line and a
domain the plume can turn over in, which is the paper's LES and not this
regression case.

References
----------

Coen, J. L., M. Cameron, J. Michalakes, E. G. Patton, P. J. Riggan and
K. M. Yedinak (2013). WRF-Fire: Coupled weather-wildland fire modeling with
the Weather Research and Forecasting model. *J. Appl. Meteor. Climatol.* 52,
16-38. Table 1 and section 4.

Rothermel, R. C. (1972). A mathematical model for predicting fire spread in
wildland fuels. USDA Forest Service Research Paper INT-115.
