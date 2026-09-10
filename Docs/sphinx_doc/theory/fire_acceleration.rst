.. _fire_acceleration:

Fire Acceleration at Startup (Phase 12)
========================================

Physical Background
--------------------

Small fires do not immediately achieve their quasi-steady-state (QSS) rate of spread due to energy and momentum limitations. The startup acceleration period depends on fire size, wind conditions, and fuel type; accurate representation of startup behavior is critical for predictions of early spread extent and spotfire timing. This module implements two optional models to scale the rate of spread during the acceleration phase until the fire reaches equilibrium.

Size-Based Model
-----------------

The size-based model is the simpler option, appropriate when only the current fire extent is needed as a predictor. It scales the rate of spread by a factor that depends on the current burned area:

.. math::

    \alpha = 1 - \exp\!\left(-\frac{r_{\text{fire}}}{L_{\text{acc}}}\right)

    R_{\text{eff}} = \alpha \cdot R_{\text{QSS}}

where:

- :math:`r_{\text{fire}} = \sqrt{A_{\text{fire}} / \pi}` is the effective fire radius derived from burned area [m]
- :math:`A_{\text{fire}}` is the total burned area [m²]
- :math:`L_{\text{acc}}` is the acceleration length scale [m] (ParmParse key: ``erf.fire.accel.L_acc``)
- :math:`R_{\text{QSS}}` is the quasi-steady-state ROS from the active spread model [m/s]

When :math:`r_{\text{fire}} \gg L_{\text{acc}}`, :math:`\alpha \to 1` and the full ROS is used without modification.

References:

- Rothermel, R.C. (1983). How to predict the spread and intensity of forest and range fires. USDA Forest Service General Technical Report INT-143.
- Catchpole, E.A., de Mestre, N.J. &amp; Gill, A.M. (1992). Intensity of fire at its perimeter. *Australian Journal of Ecology*, 17(1), 1–4.

Temporal Model
---------------

The temporal model uses the acceleration equation of McAlpine and Wakimoto (1991):

.. math::

    R(t) = R_E \cdot \left(1 - \exp(-A \cdot t)\right)

where:

- :math:`t` is the elapsed time [s], whose meaning is set by the clock below
- :math:`A` is the acceleration constant [min⁻¹], converted to [s⁻¹] internally
- :math:`R_E` is the equilibrium ROS from the active spread model [m/s]

The acceleration constant is selected based on fire perimeter length:

- :math:`A = A_{\text{point}}` when perimeter length &lt; :math:`L_{\text{perim}}` (point ignition)
- :math:`A = A_{\text{line}}` when perimeter length ≥ :math:`L_{\text{perim}}` (line fire)

The perimeter is the number of burned cells with an unburned neighbour, times the cell diagonal. Alexander et al. (1992) calibrated values are :math:`A_{\text{point}} = 0.115\ \text{min}^{-1}` and :math:`A_{\text{line}} = 0.886\ \text{min}^{-1}`.

The equation describes a fire growing from its ignition, so :math:`t` belongs to the fire, not to whichever cell happens to be burning. From a point ignition at a steady :math:`R_E`, the head of a fire that carries its ignition clock covers

.. math::

    S(t) = 1 - \frac{1 - \exp(-A t)}{A t}

of the distance it would cover without acceleration. :cpp:`erf.fire.accel.clock` selects how :math:`t` is kept.

Front clock
~~~~~~~~~~~

With ``erf.fire.accel.clock = "front"``, :math:`t` is the time since ignition, carried with the front. Each cell holds the progress :math:`s = \int A\, dt` of the fire that burned it (:math:`A t` while :math:`A` is constant):

- a burned cell advances its own :math:`s` by :math:`A \Delta t`;
- an unburned cell within :math:`n = 2 + \lceil R_{E,\max} \Delta t / \Delta x \rceil` cells of the burned region takes the largest :math:`s` of its burned, or already reached, neighbours. This covers the cells the front can reach within the step plus the neighbours the propagation stencils read. A cell the front burns therefore continues the clock of the cells that ignited it;
- a cell further out holds :math:`s = 0`. An ignition there (a spot landing, a scheduled, polygon or threshold ignition) starts a clock of its own; one inside the band joins the fire beside it;
- a change of :math:`R_E`, in time or from cell to cell, leaves :math:`s` alone, and the rate follows :math:`R_E` at once;
- when :math:`A` switches from :math:`A_{\text{point}}` to :math:`A_{\text{line}}`, :math:`s`, and so the rate, is continuous.

The rate written to ``fire_ros`` in the burned cells and the band is :math:`R_E` times the mean of :math:`1 - e^{-s}` over the step,

.. math::

    \overline{f} = 1 - e^{-s_0}\, \frac{1 - e^{-A \Delta t}}{A \Delta t},

so a front at constant :math:`R_E` covers exactly :math:`R_E \left(t - (1 - e^{-A t})/A\right)` whatever the step. Cells beyond the band take the largest :math:`s` of any burned cell, so the level set away from the front translates with it. Before any cell has burned the rate is zero.

The factor reaches every path. The FARSITE update and the isotropic and ellipse level-set paths read ``fire_ros``. The directional, hybrid and Balbi level-set paths multiply the rate they rebuild in every Runge-Kutta stage by it. The crown-fire rate of the Cruz and Van Wagner proxy models takes it too; Rothermel (1991) scales the surface rate, which already carries it.

FARSITE restarts the acceleration of a perimeter vertex from its current rate when the vertex's equilibrium rate rises (Finney 1998). The front clock does not, because ERF's equilibrium rate changes at every step under two-way coupling or a resolved turbulent wind. For a fully accelerated fire with white step-to-step noise on :math:`R_E` (:math:`\Delta t` = 0.213 s), continuing from the current rate on every rise and dropping at once on every fall holds the mean rate at 0.997, 0.97 and 0.85 of the mean :math:`R_E` for 0.1 %, 1 % and 5 % noise with :math:`A_{\text{point}}` (0.998, 0.98 and 0.89 with :math:`A_{\text{line}}`). The fire would run towards the lower edge of its own noise.

Legacy clock
~~~~~~~~~~~~

With ``erf.fire.accel.clock = "legacy"`` (the default), each cell keeps its own clock. The clock starts when the cell burns and restarts whenever the cell's equilibrium rate changes by more than :math:`10^{-6}` of the larger of its new and previous values (or of 1 m/s); only burned cells are written. Consequences:

- A cell starts again from :math:`R = 0` when it burns, and unburned cells keep :math:`R_E`. The FARSITE ``front_cell`` update takes the larger of a front cell's own rate and its burned neighbours', and the legacy FARSITE update also advances the first unburned row, so neither head is slowed.
- The directional, hybrid and Balbi level-set paths never read ``fire_ros`` and spread as if acceleration were off.
- On the isotropic level set, the newly burned side restarts while the unburned side runs at :math:`R_E`. A planar front from a line of burned cells lagged the curve by 11 % after 100 s (35.8 m for 40.1 m, unit test ``FireAcceleration.LevelSetFrontCoversTheAcceleratedDistance``).
- When :math:`R_E` changes at every step, burned cells run at about :math:`A \Delta t\, R_E` (0.0004 :math:`R_E` with :math:`A_{\text{point}}` at :math:`\Delta t` = 0.213 s).

It is kept to reproduce results from before 2026-09, and it is the only clock the wind lag below works with.

Measured
~~~~~~~~

``Exec/RegTests/FireAccelerationClock`` burns a 50 m disc of short grass in a uniform 3 m/s wind with :math:`A` = 1 min⁻¹. It measures each head against the same path without acceleration, the FARSITE path on its default ``front_cell`` update. The table gives the share of the unaccelerated advance covered by time :math:`t`:

.. list-table::
   :header-rows: 1

   * - t [s]
     - :math:`S(t)`
     - FARSITE, legacy
     - FARSITE, front
     - level set, legacy
     - level set, front
   * - 120
     - 0.568
     - 1.000
     - 0.569
     - 1.000
     - 0.621
   * - 240
     - 0.755
     - 1.000
     - 0.755
     - 1.000
     - 0.795
   * - 600
     - 0.900
     - 1.000
     - 0.897
     - 1.000
     - 0.909

On the legacy FARSITE update the front clock gives 0.599, 0.797 and 0.907, and the legacy clock 1.039, 1.017 and 1.000.

Wind-Lag Sub-Section
~~~~~~~~~~~~~~~~~~~~~

With the legacy clock, when ``erf.fire.accel.enable_wind_lag = true`` and the equilibrium rate increases, the target equilibrium ROS is not updated instantaneously. It approaches the new equilibrium with time constant :math:`\tau_{\text{wind}}` [s]:

.. math::

    R_{\text{target}}(t) = R_{\text{prev}} + (R_E - R_{\text{prev}}) \cdot \left(1 - \exp\!\left(-\frac{\Delta t}{\tau_{\text{wind}}}\right)\right)

Wind decreases are applied immediately (no lag). The lag restarts the cell's clock on every rise, and the front clock follows :math:`R_E` at once instead, so the run aborts at start-up when ``enable_wind_lag`` is combined with ``clock = "front"``.

References:

- McAlpine, R.S. &amp; Wakimoto, R.H. (1991). The acceleration of fire from point source to equilibrium spread. *Forest Science*, 37(5), 1314–1337.
- Alexander, M.E., Stocks, B.J. &amp; Lawson, B.D. (1992). Fire behavior in Black Spruce-lichen woodland. Information Report NOR-X-310. Canadian Forest Service.
- Finney, M.A. (1998/2004). FARSITE: Fire Area Simulator — Model Development and Evaluation. USDA Forest Service Research Paper RMRS-RP-4.

Input Parameters
-----------------

.. list-table:: Fire Acceleration Parameters
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``erf.fire.accel.enable``
     - bool
     - false
     - Master on/off switch; when false, acceleration is disabled
   * - ``erf.fire.accel.use_temporal``
     - bool
     - false
     - false = size-based model, true = temporal model
   * - ``erf.fire.accel.L_acc``
     - Real [m]
     - 50.0
     - Size-based: acceleration length scale [m]
   * - ``erf.fire.accel.A_point``
     - Real [1/min]
     - 0.115
     - Temporal: acceleration constant for point ignitions [1/min]
   * - ``erf.fire.accel.A_line``
     - Real [1/min]
     - 0.886
     - Temporal: acceleration constant for line ignitions [1/min]
   * - ``erf.fire.accel.perim_limit``
     - Real [m]
     - 500.0
     - Temporal: perimeter length threshold [m] for switching from A_point to A_line
   * - ``erf.fire.accel.clock``
     - string
     - "legacy"
     - Temporal: "legacy" a clock per burned cell, written to burned cells only; "front" the time since ignition, carried with the front and applied to the rate that moves it
   * - ``erf.fire.accel.enable_wind_lag``
     - bool
     - false
     - Temporal, legacy clock only: apply exponential lag on wind speed increases
   * - ``erf.fire.accel.tau_wind``
     - Real [s]
     - 60.0
     - Temporal: wind-lag time constant [s]

Usage Example
--------------

Size-based model (simpler, uses fire area only)::

   erf.fire.accel.enable       = true
   erf.fire.accel.use_temporal = false
   erf.fire.accel.L_acc        = 100.0

Temporal model with the clock carried by the front::

   erf.fire.accel.enable          = true
   erf.fire.accel.use_temporal    = true
   erf.fire.accel.clock           = "front"
   erf.fire.accel.A_point         = 0.115
   erf.fire.accel.A_line          = 0.886
   erf.fire.accel.perim_limit     = 500.0

Limitations
-----------

- Both models are disabled by default (``accel.enable = false``); enabling has zero runtime cost when disabled.
- The size-based model uses a single global scaling factor; cells near the fire centre and cells at the expanding perimeter receive the same alpha.
- The acceleration constant is one value for the whole domain, switched by the total burned perimeter.
- Front clock: an ignition within the band of an existing fire (a few cells) joins that fire's clock instead of starting its own. ``fire_ros`` holds the accelerated rate, zero everywhere before the first cell burns.
- Legacy clock: the clock restarts whenever the equilibrium ROS changes by more than a threshold (relative threshold: 1.0e-6 times the maximum of the new and previous ROS values, or of 1 m/s); this threshold applies to both wind changes and fuel changes. The wind-lag model applies only to wind speed increases; wind decreases update the target ROS immediately.
- The temporal model requires allocating a 3-component per-cell ``fire_accel_state`` MultiFab, checkpointed as ``FireAccelState``: the accelerated rate, the elapsed time (legacy) or progress :math:`A t` (front), and the equilibrium rate (legacy) or the factor applied (front). A restart must use the clock the checkpoint was written with. The size-based model does not allocate any additional storage.
- Neither model adjusts the fire perimeter shape — only the magnitude of ROS is scaled.

References
-----------

- Rothermel, R.C. (1983). How to predict the spread and intensity of forest and range fires. USDA Forest Service General Technical Report INT-143.
- Catchpole, E.A., de Mestre, N.J. &amp; Gill, A.M. (1992). Intensity of fire at its perimeter. *Australian Journal of Ecology*, 17(1), 1–4.
- McAlpine, R.S. &amp; Wakimoto, R.H. (1991). The acceleration of fire from point source to equilibrium spread. *Forest Science*, 37(5), 1314–1337.
- Alexander, M.E., Stocks, B.J. &amp; Lawson, B.D. (1992). Fire behavior in Black Spruce-lichen woodland. Information Report NOR-X-310. Canadian Forest Service.
- Finney, M.A. (1998/2004). FARSITE: Fire Area Simulator — Model Development and Evaluation. USDA Forest Service Research Paper RMRS-RP-4.
