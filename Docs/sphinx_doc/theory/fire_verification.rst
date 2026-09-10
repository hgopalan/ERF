.. role:: cpp(code)
   :language: c++

.. _sec:FireVerification:

Fire Verification Cases
=======================

``Exec/CanonicalTests/Fire/Verification`` collects fire cases whose answer is
known independently of the code: an exact solution of the equations the fire
module solves, or a published result. Each case directory holds its decks, the
script that writes any rasters or ignition files (their output is committed), a
``run_<case>.sh`` that runs the decks and then the check, and a
``check_<case>.py`` that compares the output with the reference and exits
non-zero on a failure. Every README records the agreement measured with the
committed decks; the numbers below are from those runs.

.. code-block:: bash

   MPIRUN="mpirun -np 2" ./run_<case>.sh /path/to/erf_exec   # runs, then checks
   SKIP_RUN=1 ./run_<case>.sh x                              # checks only

Two options exist for these cases, both off by default.
``erf.fire.ros_model = "prescribed"`` sets the rate of spread in the deck,

.. math::

   R(x, y) = \max\left(R_{\min},\; R_b + g_x (x - x_0) + g_y (y - y_0)\right),

with :math:`R_b` given by ``erf.fire.prescribed.ros`` or per fuel code by
``erf.fire.prescribed.by_fuel``. It has no wind, slope or moisture dependence
and no direction, so ``directional_ros`` is turned off and the level set solves
the eikonal equation :math:`|\nabla T| = 1/R` as posed: the arrival time is the
shortest travel time from the ignition. ``erf.fire.prescribed_heat.flux`` adds a
constant heat flux over a disc to the fire heat flux, which the coupling then
injects like fire heat. ``Exec/RegTests/FirePrescribed`` checks both.

The geometry cases run on still air under slip walls with 2 m fire cells, so
arrival errors are quoted in cell-crossing times :math:`h/R`.

The level set against geometry
------------------------------

.. list-table::
   :widths: 22 38 40
   :header-rows: 1

   * - Case
     - Exact answer
     - Measured
   * - ``Obstacle_Shadow``
     - arrival :math:`(L - r_{ig})/R` with :math:`L` the taut string around a
       non-burnable disc
     - lit region: mean :math:`|e|` 0.07, 95th percentile 0.16; shadow: +0.44
       against a disc half a cell larger than the nominal one, which the
       cell-resolved mask makes it (+0.52 with ``levelset.wall_extrapolate``)
   * - ``Junction_Fire``
     - arrival (distance to the V :math:`-\,w)/R`; meeting point at
       :math:`R/\sin(\theta/2)`
     - mean :math:`|e|` 0.16 to 0.31 over the grid; meeting-point speed within
       0.85 %, 0.02 % and 0.15 % at 30, 60 and 90 degrees
   * - ``Polygon_Growth``
     - area :math:`A_0 + P_0 r + \pi r^2` (square) and
       :math:`A_0 + P_0 r + (2\pi - 4) r^2` (cross), arrival = distance / R
     - area within 0.21 (square) and 0.28 (cross) cell widths times the
       perimeter at 60 s; arrival mean :math:`|e|` 0.14; the cross's inner
       corners stay sharp
   * - ``Merging_Fires``
     - union of two discs of radius :math:`r_{ig} + R t`
     - area 0.23 to 0.33 cell widths of perimeter short; arrival mean
       :math:`|e|` 0.17, the same across the neck
   * - ``Fuel_Interface_Refraction``
     - Snell's law :math:`\sin\theta_2 / R_2 = \sin\theta_1 / R_1` and the
       transmitted plane front
     - transmitted angle 19.00 against 18.75 degrees (fast to slow) and 57.75
       against 57.70 (slow to fast); :math:`|\nabla T|` within 0.7 % of
       :math:`1/R_2`; arrival mean :math:`|e|` 0.10 to 0.15
   * - ``Speed_Gradient``
     - :math:`T = \operatorname{acosh}\left(1 + |g|^2 |p - s|^2 / (2 R(s) R(p))\right)/|g|`
     - arrival mean :math:`|e|` 0.08, 95th percentile 0.17; the front's extent
       along the gradient within 0.01 cells

The level set lags the exact fronts by a fraction of a cell: its artificial
viscosity slows a curved front by :math:`R\,\varepsilon\,\kappa`, about
:math:`\varepsilon \ln(r_1/r_0)` as a circle grows from :math:`r_0` to
:math:`r_1`, and thin ignitions are stamped to within a fraction of a cell.
The checks allow half a cell-crossing time on average and one cell at the 95th
percentile.

Rothermel on a slope
--------------------

``Slope_No_Wind`` burns Anderson fuel model 1 at 5.5 % on the planes
:math:`z = 0.3\,x` and :math:`z = 0.6\,x` in still air. On the isotropic path
(``directional_ros = false``) every direction spreads at
:math:`R_0 (1 + \phi_s)`, reduced by :math:`\sqrt{1 + s_n^2}` for the slope
:math:`s_n` along it by the ground projection; the four axis rates agree to
0.24 %. On the default directional path the backing fire and the flanks spread
at :math:`R_0`, their fronts within 0.05 cell of the exact ones.

Each deck is compared with the exact (viscosity) solution of its own level-set
equation, evaluated with the Hopf formula
:math:`T(\mathbf x) = \max_{\mathbf n} ((\mathbf x - \mathbf c)\cdot\mathbf n - r_0)/F(\mathbf n)`
for the map-view normal speed :math:`F`. Rates are checked to 3 % where the fit
spans 20 cells or more, and fronts that travel fewer cells to half a cell on
average and one at worst.

The directional head does not reach :math:`R_0 (1 + \phi_s)`. The directional
path evaluates :math:`R(\mathbf n) = R_0 (1 + 5.275\,\beta^{-0.3}
\max(\mathbf s \cdot \mathbf n, 0)^2)`, and for a rate that peaks this sharply
about one direction the exact solution of
:math:`\phi_t + R(\mathbf n) |\nabla\phi| = 0` from a point ignition is the Wulff
shape. Its head is a wedge of oblique facets whose tip runs at

.. math::

   \min_{\mathbf n} \frac{R(\mathbf n)}{n_x \sqrt{1 + (\mathbf s \cdot \mathbf n)^2}}
   \approx 2 \sqrt{\phi_s}\, R_0,

0.183 m/s instead of 0.326 m/s at :math:`s = 0.6`. A straight line fire keeps
:math:`\mathbf n = \hat{\mathbf x}` and is not affected, which is why the line
fire of :ref:`sec:LineFireVerification` meets Rothermel's head rate. The scheme
lands between the two, 0.257 m/s (52 % of the way from the Wulff tip speed to
Rothermel's head rate) at :math:`s = 0.6` and 0.102 m/s (61 %) at
:math:`s = 0.3`, and the check requires only that. The same argument applies to
the wind factor wherever :math:`\phi_w (B - 1) > 1`.

The opt-in :cpp:`erf.fire.directional_shape = "ellipse"` removes the shortfall. It
takes the normal speed from the support function of the ellipse with the same
head, back and flank rates, and an ellipse is its own Wulff shape. Its decks
``ell_s30`` and ``ell_s60`` spread the head at 0.1077 and 0.3250 m/s, within
0.6 % of Rothermel's head rate. Their backs and flanks stay within 0.26 cell of
the exact fronts, although the pointed back reads 4 to 5 % fast as a fitted rate
over the 7 to 9 cells it travels. ``Exec/RegTests/FireDirectionalShape`` measures
the same in a uniform 1.5 m/s wind on the same grass (:math:`\phi_w = 9.4`). There
the default head runs at 0.185 m/s and the ellipse's at 0.248 m/s, against
Rothermel's 0.249 m/s and a Wulff tip of 0.142 m/s.

Fuel moisture
-------------

``Moisture_Relaxation`` dries fuel model 1 from 20 % in still, dry air at 300 K
for two hours with ``moisture_dynamic = true``. Each dead class relaxes as
:math:`\mathrm dM/\mathrm dt = (M_e - M)/\tau_{\mathrm{eff}}`, and Rothermel,
rebuilt from the moisture every step, holds the ignition still until the 1-hour
class falls below the 12 % moisture of extinction. Every class matches the
model's forward-Euler steps to :math:`10^{-16}` and the closed form to
:math:`5 \times 10^{-5}`; the rate of spread matches Rothermel at the 1-hour
moisture to six digits once it crosses extinction at 2754 s; the burned radius
follows the integral of that rate to 0.09 cells, reaching 76.2 m at two hours.

With the adsorption and desorption equilibria :math:`E_w = 0.0351` and
:math:`E_d = 0.0600` of this air, the hysteresis in
``compute_emc_with_hysteresis`` sends fuel wetter than :math:`E_d` towards it,
fuel drier than :math:`E_w` towards that, and leaves fuel between the two alone
(Nelson 2000; Vejmelka et al. 2016), so drying fuel follows
:math:`M = E_d + (M_0 - E_d) e^{-t/\tau}`: 0.1062 after one hour and 0.0753
after two. The check was written against the reversed choice ERF carried until
September 2026, which sent drying fuel towards :math:`E_w` and held it at
:math:`E_d`, :math:`\max(E_w + (M_0 - E_w) e^{-t/\tau}, E_d)` = 0.0895 after one
hour; it still prints that value, and a binary from before the fix fails 44 of
its 48 checks.

References
----------

Nelson, R. M. (2000). Prediction of diurnal change in 10-h fuel stick moisture
content. Canadian Journal of Forest Research, 30, 1071-1087.

Rothermel, R. C. (1972). A mathematical model for predicting fire spread in
wildland fuels. USDA Forest Service Research Paper INT-115.

Vejmelka, M., A. K. Kochanski and J. Mandel (2016). Data assimilation of dead
fuel moisture observations from remote automated weather stations.
International Journal of Wildland Fire, 25, 558-568.

Viegas, D. X., J. R. Raposo, D. A. Davim and C. G. Rossa (2012). Study of the
jump fire produced by the interaction of two oblique fire fronts. Part 1.
Analytical model and validation with no-slope laboratory experiments.
International Journal of Wildland Fire, 21, 843-856.
