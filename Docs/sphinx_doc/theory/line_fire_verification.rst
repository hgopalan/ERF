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
false``; WRF-Fire caps R at 6 m/s only). Five decks share one base: no wind,
a 2.5 m/s sounding (the paper's Control), a 5 m/s sounding (WSHi), and the
two winds with the heat coupled back.

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
fire reports each step, averaged over the run, since the surface layer slows
the sounding wind at 6.1 m as the run proceeds (by 15 % over 180 s at 2.5 m/s
and 28 % at 5 m/s). The one-way decks must match to 10 %.

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
     - 0.0908
     - 0.0957
     - Control: 0.22 head (backing not quoted; WRF-Fire sets it to R0)
   * - wind5
     - 4.31
     - 1.56
     - 0.0240
     - 0.0240
     - 0.2689
     - 0.2541
     - WSHi: about 0.40 head
   * - wind2p5_2way
     - 4.80 (max)
     - 1.74
     - 0.0240
     - 0.0240
     - (coupled)
     - 0.2427
     - Control: 0.22 head
   * - wind5_2way
     - 5.55 (max)
     - 2.01
     - 0.0240
     - 0.0240
     - (coupled)
     - 0.2383
     - WSHi: about 0.40 head

Reading the table: the paper's column is its head fire in a coupled LES,
except the no-wind row, where the fire crept outward at the same rate on
every side. The backing fire is never quoted in the paper, and WRF-Fire sets
it to :math:`R_0` as ERF does, so the like-for-like pairs are backing against
NoWind, the one-way heads against Rothermel at the sampled wind, and the
two-way heads against Control and WSHi.

The backing fire moves at :math:`R_0` to the last digit in every deck, the
no-wind fire at :math:`R_0` in both directions, and the head fire within
5.5 % of Rothermel at the sampled wind in both winds, the residual being the
wind's drift over the run. The no-wind rate itself, 0.024 m/s, is the
paper's 0.02 m/s.

The heads of the one-way decks are well below the paper's 0.22 and about
0.40 m/s, and are meant to be: those are coupled results in a turbulent
convective boundary layer, where the fire's plume draws the near-surface
wind into the head ("near-fire horizontal winds varied from 2 to 4 m/s" for
the 2.5 m/s Control). At 2.5 m/s the one-way chain is 2.5 m/s in the
sounding, 2.3 m/s at 6.1 m once the surface layer has acted, 0.83 m/s after
the 0.36 reduction factor, a wind factor of 2.8 and a head of 0.09 m/s;
doubling the midflame wind raises the wind factor by about four and the head
to the paper's value.

The two-way decks are reported, not checked, and the two winds show why.
At 2.5 m/s the coupled head runs at 0.24 m/s, the paper's 0.22. At 5 m/s it
runs at 0.24 m/s as well, below the one-way 0.25 and far below the paper's
0.40. The fire plotfiles show the reason: with the heat on, the 6.1 m wind at
the head is 5.0 m/s at 50 s (4.6 one-way), 3.7 m/s at 100 s (4.2 one-way)
and reversed, -1.8 m/s at the head and -2.2 m/s 10 m ahead of it, at 150 s.
The line here spans the whole periodic y extent, so its plume is
two-dimensional and the inflow it draws from the downwind side has nowhere to
come from but against the ambient wind, which it overpowers; the directional
model then clips the wind component into the fire at zero and the head drops
to the no-wind rate. The paper's 1 km line in a 5 km box lets the flow go
around the flanks and feed the head instead. Reproducing the coupled rates
needs a finite line and a domain the plume can turn over in, which is the
paper's LES and not this regression case; the 2.5 m/s agreement is the
transient enhancement before the reversal sets in, not the paper's
mechanism.

References
----------

Coen, J. L., M. Cameron, J. Michalakes, E. G. Patton, P. J. Riggan and
K. M. Yedinak (2013). WRF-Fire: Coupled weather-wildland fire modeling with
the Weather Research and Forecasting model. *J. Appl. Meteor. Climatol.* 52,
16-38. Table 1 and section 4.

Rothermel, R. C. (1972). A mathematical model for predicting fire spread in
wildland fuels. USDA Forest Service Research Paper INT-115.
