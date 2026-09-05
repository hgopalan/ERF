.. role:: cpp(code)
   :language: c++

.. _sec:Fire:

Fire Model
==========

Overview
--------

ERF-Fire simulates a surface wildfire on a two-dimensional fire grid that sits
on the atmosphere's level-0 mesh and is refined by an integer factor in each
horizontal direction. The atmosphere supplies wind, near-surface temperature
and humidity; the fire model returns a surface heat flux, an optional latent
flux and an optional smoke tracer. Terrain enters through slopes on the fire
grid and through the height above ground at which the wind is sampled.

The model is built from independently selectable pieces:

- a **rate-of-spread model** that fills the field ``fire_ros`` from the local
  wind, slope, fuel and moisture (Rothermel, BEHAVE, MacArthur, Cheney-Gould,
  Balbi, or a per-cell hybrid of two of them), see :ref:`sec:ROS_Models`;
- a **front propagation method** that advances the burned region at that
  rate, either the FARSITE Lagrangian marker scheme or a level-set solver, see
  :ref:`sec:FirePropagation`;
- **fuel**, either one Anderson model everywhere or a spatial fuel map with
  firebreaks, and a dead-fuel moisture model driven by the atmosphere, see
  :ref:`sec:SpatialFuel` and :ref:`sec:FireFuelMoisture`;
- a **non-burnable mask** from building footprints, listed fuel codes and
  firebreaks, which the fire goes around, see :ref:`sec:FirePropagation`;
- **ignition**, a disc, a polygon or polyline perimeter, or a timed schedule
  of events, see :ref:`sec:MultiIgnition`;
- **coupling** to the atmosphere through wind extraction and heat injection,
  see :ref:`sec:FireCoupling`;
- optional **behaviour extensions**: startup acceleration, ember spotting and
  crown fire, see :ref:`fire_acceleration` and :ref:`sec:FireSpottingCrown`;
- **diagnostics and output**: fireline intensity, flame length, flame
  temperature and tilt, fire plotfiles, a statistics CSV, arrival-time probes
  and checkpoint state, see :ref:`sec:FireOutput`.

Every option is off, or set to its historical behaviour, by default. A deck
that sets only ``erf.fire.enable``, a fuel model and an ignition runs
Rothermel on the FARSITE path with static moisture and lagged heat coupling.
The complete list of inputs with defaults is in :ref:`sec:FireInputs`, and
``Exec/CanonicalTests/Fire/inputs_fire_master_reference`` carries every input
with a comment as a reference deck.

.. toctree::
   :maxdepth: 1

   fire_propagation
   ros_models
   Fire_FuelMoisture
   spatial_fuel
   multi_ignition
   fire_coupling
   wui_validation
   fire_acceleration
   fire_spotting_crown
   fire_output

Fire grid
---------

The fire grid is created from the atmosphere's level-0 box array and
distribution map by refining both by ``erf.fire.grid_ratio`` in x and y and
collapsing z to one cell. Every fire box therefore lives on the rank that
owns its parent atmosphere box, and the map from a fire cell to its column is
integer division by the ratio. Two constraints follow:

- the x and y lengths of every atmosphere box must be divisible by the ratio
  (set ``amr.max_grid_size`` accordingly);
- the atmosphere must not be decomposed in z (``amr.max_grid_size_z`` at
  least the number of vertical cells), since the fire model interpolates
  through whole columns.

The fire model runs on level 0 only. Coordinates on the fire grid are the
physical x and y of the atmosphere domain, so ignition points, probes,
firebreaks and structure files are all given in metres.

One fire step
-------------

``FireLayer::advance`` is called once per atmospheric time step, after the
dynamical core. In order it:

1. samples the near-surface temperature and relative humidity and, when
   ``erf.fire.moisture_dynamic`` is on, advances the dead-fuel moisture
   classes (:ref:`sec:FireFuelMoisture`);
2. extracts the wind at the reference height above ground on the fire grid,
   applies the wind adjustment factor and any terrain correction
   (:ref:`sec:FireCoupling`);
3. applies any scheduled ignitions due in this step;
4. evaluates the selected rate-of-spread model into ``fire_ros``, blends at
   fuel boundaries, and applies the acceleration and crown-fire adjustments
   when enabled;
5. propagates the front with CFL-limited subcycles of the FARSITE or
   level-set scheme, reinitialising the level set periodically, and applies
   ember spotting when enabled;
6. updates the arrival-time field, consumes fuel and computes the surface
   heat flux and the flame diagnostics;
7. reports probes, appends the statistics CSV, and stores the coarsened heat
   flux for injection into the atmosphere at the next step.

The level set ``fire_phi`` is normalised: it is :math:`-1` well inside the
burned region, :math:`+1` well outside, zero on the front, and varies
linearly between them over a band a few fire cells wide. Cells with
:math:`\phi < 0` are burning or burned; ``fire_arrival_time`` records when
each cell first crossed zero and is :math:`-1` where it has not.

Fields on the fire grid
-----------------------

The principal state and diagnostic fields, all cell-centred on the fire grid:

.. list-table::
   :widths: 26 10 64
   :header-rows: 1

   * - Field
     - Comp.
     - Meaning
   * - ``fire_phi``
     - 1
     - Normalised level set, negative where burned
   * - ``fire_arrival_time``
     - 1
     - Time a cell first burned [s], :math:`-1` if unburned
   * - ``fire_ros``
     - 1
     - Isotropic (head-fire) rate of spread [m/s]
   * - ``fire_wind_ref``, ``fire_wind_eff``
     - 2 each
     - Wind at the reference height, and after WAF and terrain corrections [m/s]
   * - ``fire_wind_extract_z``
     - 1
     - Height at which the wind was sampled [m]
   * - ``fire_slopes``, ``fire_curvature``
     - 2, 1
     - Terrain slope components and curvature on the fire grid
   * - ``fire_fuel_load``
     - 1
     - Remaining fuel load [kg/m²]
   * - ``fire_fuel_mc``
     - 5
     - Dead 1-, 10-, 100-hour and live herbaceous and woody moisture [fraction]
   * - ``fire_fuel_model``
     - 1
     - Fuel code per cell, present only with a spatial fuel map
   * - ``fire_surface_temp``, ``fire_surface_rh``
     - 1 each
     - Near-surface temperature [K] and relative humidity [0-1] from the atmosphere
   * - ``fire_heat_flux``, ``fire_latent_flux``
     - 1 each
     - Sensible and latent surface flux [W/m²]
   * - ``fire_fireline_intensity``, ``fire_flame_length``, ``fire_flame_temp``, ``fire_flame_tilt``
     - 1 each
     - Byram intensity [kW/m], Thomas flame length [m], flame temperature [K], tilt [deg]
   * - ``fire_ros_weight``, ``fire_structure_height``
     - 1 each
     - Hybrid model weight and sampled building height, present with the hybrid model or structures
   * - ``fire_nonburnable``
     - 1
     - Non-burnable mask, present when structures, non-burnable fuel codes or masked firebreaks are configured
   * - ``fire_crown_active``, ``fire_crown_load``, ``fire_crown_fraction_burned``
     - 1 each
     - Crown-fire state, present only with crown fire enabled
   * - ``fire_albini_data``
     - 4
     - Spotting diagnostics, present only with spotting enabled

Which of these reach the fire plotfile, and under what names, is listed in
:ref:`sec:FireOutput`.

Source layout
-------------

All fire sources are in ``Source/Fire``. The entry points are
``ERF_FireLayer.H`` and ``ERF_FireLayer.cpp`` (the ``FireLayer`` class that
owns the fields and runs a step) and ``ERF_FireParams.H`` (every
``erf.fire.*`` input, read once from ParmParse). The remaining headers each
hold one model or one stage of the step and are named for it:
``ERF_Rothermel``, ``ERF_BalbiModel``, ``ERF_BehaveModel``,
``ERF_MacArthurModel``, ``ERF_CheneyGouldModel``, ``ERF_DirectionalRos`` and
``ERF_HybridRos`` for the rate of spread; ``ERF_FarsiteEllipse``,
``ERF_LevelSetAdvection``, ``ERF_NumericalSchemes`` and ``ERF_Reinitialize``
for propagation; ``ERF_FireWindExtract``, ``ERF_FuelWindHeight`` and
``ERF_TerrainSlope`` for the wind and terrain; ``ERF_FuelMoisture``,
``ERF_MoistureExtinction``, ``ERF_FuelMap``, ``ERF_LcpReader``,
``ERF_FuelBlending`` and ``ERF_FireBreak`` for fuel; ``ERF_FireIgnition``,
``ERF_IgnitionSchedule`` and ``ERF_PolygonIgnition`` for ignition;
``ERF_FireHeatFlux``, ``ERF_FireAtmCoupling`` and ``ERF_FireSmokeEmission``
for coupling; ``ERF_AlbiniSpotting``, ``ERF_ScottSpottingTable``,
``ERF_CrownFire`` and ``ERF_FireAcceleration`` for the behaviour extensions;
and ``ERF_FireDiagnostics``, ``ERF_FireStatsOutput``, ``ERF_FirePlotfile``
and ``ERF_FirePlotfileCatalog`` for output. The fire module is compiled when
``ERF_ENABLE_FIRE`` is on, which is the CMake default.

Tests
-----

``Exec/CanonicalTests/Fire`` holds the canonical fire cases, grouped by theme
(core physics, FARSITE and level-set propagation, fire-atmosphere coupling,
fire behaviour options, heat-flux diagnostics, mesh refinement) with a README
in every directory, plus Python unit tests under ``Unit_Tests`` for the
Rothermel kernel, the FARSITE ellipse, the ROS models, the fuel map reader,
the ignition schedule, spotting, crown fire, acceleration, wind interpolation
and terrain projection. The canonical cases have no recorded reference values, so they are smoke
tests: they show a feature runs and behaves qualitatively as documented.

The regression suites under ``Exec/RegTests`` are the quantitative checks.
Each is a directory of input decks sharing one base, a script that runs every
deck and prints a table, and a README that records the reference values and
explains what each row should show:

- ``FireRosComparison``: every rate-of-spread model on the isotropic and
  direction-dependent level-set paths, the hybrid selectors with their
  identity checks, per-fuel Rothermel coefficients, the wind mapping, and the
  Balbi wind-source and extinction options, all on one flat grass fire.
- ``FireHybridObstacles``: the hybrid structure selector, arrival-time probes
  and the interaction with immersed-forcing buildings.
- ``FireRestart``: a checkpoint written mid-run and a restart from it, on both
  propagation paths and with lagged heat coupling, which must reproduce the
  uninterrupted run exactly.
- ``FireHeatPlacement``: the fire tendency's source mode and the open-fraction
  placement around buildings, with and without immersed-forcing buildings,
  checked through the coupling's energy diagnostic.
- ``FireNearWall``: the level-set wall extrapolation and the open-column wind
  weights next to masked buildings, measured by the flank's arrival along a
  wall against the unmasked reference.
- ``FireExposure``: the per-structure exposure CSV (arrival and residence of
  the front along each wall, peak intensity, heat load, embers) with and
  without immersed-forcing buildings and with spotting.

Where each feature is exercised:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Feature
     - Test
   * - Rothermel, BEHAVE, MacArthur, Cheney-Gould, Balbi 2009 and 2020
     - ``FireRosComparison``; canonical ``Fire_Behavior/ROS_Models``, ``Core_Physics``
   * - Direction-dependent level set, every model
     - ``FireRosComparison`` (``*_directional`` rows)
   * - Hybrid: region, fuel, wind selectors, blend width, directional, non-Balbi members
     - ``FireRosComparison`` (``hybrid_*`` rows)
   * - Hybrid: structure selector, probes, immersed-forcing buildings
     - ``FireHybridObstacles``
   * - Non-burnable mask: structures and fuel codes
     - ``FireHybridObstacles`` (``*_mask`` rows), ``FireRosComparison`` (``rothermel_code0_*``)
   * - Per-fuel Rothermel coefficients, spatial fuel map, blending, firebreaks
     - ``FireRosComparison`` (``rothermel_fuelmap``, ``hybrid_fuel``); canonical ``Fire_Behavior/Spatial_Fuel``
   * - Wind mapping (bilinear, nearest), per-fuel wind height, WAF formulas
     - ``FireRosComparison`` (``rothermel_nearest``); canonical ``ROS_Models``, ``Core_Physics/Wind_Adjustment_Factor``; ``Unit_Tests/test_wind_interpolation.py``
   * - Balbi couplings: reference wind, moisture extinction
     - ``FireRosComparison`` (``balbi2020_reference_wind``, ``balbi2020_extinction_wet``)
   * - FARSITE ellipse, level-set advection and reinitialisation
     - ``FireRestart``; canonical ``FARSITE_Propagation``, ``Level_Set_Propagation``; ``Unit_Tests/test_farsite_ellipse.py``
   * - Checkpoint and restart of the fire state
     - ``FireRestart``
   * - Dynamic fuel moisture
     - canonical ``Core_Physics/Fuel_Moisture_Sensitivity``, ``ROS_Models/behave_dynamic``
   * - Ignition schedule, polygon and polyline ignition
     - canonical ``Fire_Behavior/Ignition_Patterns``; ``Unit_Tests/test_ignition_schedule.py``
   * - Startup acceleration
     - canonical ``Fire_Behavior/Acceleration``; ``Unit_Tests/test_fire_acceleration.py``
   * - Ember spotting, crown fire
     - canonical ``Fire_Behavior/Spotting``, ``Fire_Behavior/Crown_Fire``; ``Unit_Tests/test_albini_spotting.py``, ``test_crown_fire.py``
   * - Coupling modes, heat injection, smoke tracer
     - canonical ``Fire_Atmosphere_Coupling``
   * - Additive source mode, open-fraction heat placement, fire heat with immersed-forcing buildings
     - ``FireHeatPlacement``
   * - Level-set wall extrapolation, open-column wind weights
     - ``FireNearWall``
   * - Structure exposure diagnostics, their checkpoint and restart
     - ``FireExposure``; ``FireRestart`` (``exposure`` row)
   * - Flame temperature, tilt, intensity
     - canonical ``Heat_Flux_Diagnostics``
   * - Terrain slopes and terrain wind corrections
     - canonical ``Core_Physics/ROS_Slope_Effects``, ``Terrain_Wind_Coupling``; ``Unit_Tests/test_terrain_projection.py``
   * - The WUI features together: structure mask, wall extrapolation, open-column wind, exposure, spotting, immersed buildings with heat coupling
     - canonical ``WUI_Subdivision`` (:ref:`sec:WUIValidation`)

Not yet covered by any test: restart of the spotting and crown-fire state.
The fire-dust coupling has its own cases under ``Exec/CanonicalTests/Hazard``
(:ref:`sec:DustFire`).

References
----------

- Rothermel, R. C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service Research Paper INT-115.
- Finney, M. A. (2004). FARSITE: Fire Area Simulator model development and evaluation. USDA Forest Service Research Paper RMRS-RP-4 Revised.
- Mandel, J., Beezley, J. D. and Kochanski, A. K. (2011). Coupled atmosphere-wildland fire modeling with WRF 3.3 and SFIRE 2011. Geoscientific Model Development, 4, 591-610.
