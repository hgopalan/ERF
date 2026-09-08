
.. role:: cpp(code)
   :language: c++

.. _sec:MultiIgnition:

Fire Model - Multi-Ignition Schedule and Polygon Ignition
===========================================================

Overview
--------

The multi-ignition feature enables complex fire scenarios with multiple ignition sources
at different times and locations. Two complementary mechanisms are provided:

1. **Time-Scheduled Ignitions**: Discrete ignition events occurring at specified times with
   specified locations, allowing simulation of lightning strikes, spot fires from flying
   embers, or prescribed burn operations in sequence.

2. **Polygon and Polyline Ignitions**: Initial fire perimeter defined by a closed polygon
   or open polyline, allowing initialization from arbitrary geometric boundaries representing
   initial burn areas or fuel breaks.

Both mechanisms use the same signed-distance field (SDF) representation as the primary
ignition circle, with implicit merging of multiple fire fronts via the min(phi, new_val)
stamping convention.

Reference: Finney, M. A. (2004) FARSITE: Fire Area Simulator model development and evaluation.
Res. Paper RMRS-RP-4 Revised, USDA Forest Service, Rocky Mountain Research Station.

Point Ignition Schedule
-----------------------

Time-Window Firing Mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Events are applied when their specified time falls in the time window (t_prev, t_current],
where t_prev and t_current are the times at the start and end of the current atmospheric
timestep. This ensures:

- Each event fires exactly once at the intended time
- Multiple events within the same timestep are ordered by priority (higher priority first)
- Events are sorted by time then priority before application

The fired flag on each event prevents re-application after firing.

CSV File Format
~~~~~~~~~~~~~~~

The ignition schedule is specified in CSV format with one event per line:

.. code-block:: text

   time_s  cx  cy  radius  [source_type]  [priority]  [suppress_if_burning]

**Field Definitions:**

- **time_s** (Real): Time at which event fires [seconds]
- **cx, cy** (Real): Ignition center location [m]
- **radius** (Real): Ignition sphere radius [m]
- **source_type** (integer, optional, default=0): 0=sphere, 1=polyline (for future use)
- **priority** (integer, optional, default=5): Priority ordering; higher fires first in same timestep
- **suppress_if_burning** (integer, optional, default=0): If 1, skip cells already burning (phi < 0)

Lines beginning with # or ! are treated as comments and ignored.

Example Schedule File
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   # Phase 11 ignition schedule test file.
   # Primary ignition at t=0, secondary at t=300 s, tertiary at t=600 s
   0      500.0  1000.0  30.0  sphere  10
   300    1400.0 1000.0  20.0  sphere   5
   600    1000.0  400.0  15.0  sphere   5

Fire Front Merging
~~~~~~~~~~~~~~~~~~

When multiple ignitions occur at different locations, their fire fronts merge implicitly
through the SDF stamping mechanism. Each ignition applies:

.. code-block:: text

   phi(i,j,k) = min(phi(i,j,k), -(radius - distance))

The min() operation ensures:

- Cells already burning (phi < 0) are not made unburned by a new ignition
- Cells at different distances from multiple ignition centers take the value from the
  closest ignition center
- Fire fronts that overlap share the same phi field

Polygon Ignition
----------------

Closed Polygon Mode
~~~~~~~~~~~~~~~~~~~

A closed polygon defines the initial burned area via its interior and the initial fire
front via its boundary. The level-set field is set using:

- **Interior**: phi = -distance_to_boundary (burned cells)
- **Boundary**: phi ≈ 0 (fire front)
- **Exterior**: phi = +distance_to_boundary (unburned cells)

Interior/exterior distinction is determined by the 2D winding number test: a point is
inside the polygon if its winding number (ray-crossing count) is non-zero.

Polyline Mode
~~~~~~~~~~~~~

An open polyline (sequence of connected line segments) defines a line fire. The level-set
field is set using:

- **Within half_width**: phi = -half_width (burned cells)
- **Outside half_width**: phi = +distance_to_nearest_segment (unburned cells)

The half-width parameter sets the initial burn width around the line.

CSV Vertex File Format
~~~~~~~~~~~~~~~~~~~~~~

Polygon and polyline vertices are specified in CSV format with one vertex per line:

.. code-block:: text

   x  y

**Field Definitions:**

- **x, y** (Real): Vertex coordinates [m]

Lines beginning with # or ! are treated as comments and ignored.

Example Polygon File
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   # 200x100 m ellipse centered at (1000, 1000)
   1200.0 1000.0
   1178.0 1058.0
   1118.0 1095.0
   1000.0 1100.0
   882.0  1095.0
   822.0  1058.0
   800.0  1000.0
   822.0  942.0
   882.0  905.0
   1000.0 900.0
   1118.0 905.0
   1178.0 942.0

Example Polyline File
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   # East-west line fire from x=400 to x=1600 m at y=1000 m
   400.0  1000.0
   700.0  1000.0
   1000.0 1000.0
   1300.0 1000.0
   1600.0 1000.0

Temperature-Threshold Ignition
------------------------------

The third source is the atmosphere itself. With
``erf.fire.ignition.threshold_enable = true``, every burnable, unburned fire
cell whose near-surface air temperature exceeds
``erf.fire.ignition.threshold_temp`` is ignited on the fire step in which it
first does so. The temperature is the :math:`k = 0` potential temperature the
fuel-moisture update already maps onto the fire grid (``fire_surface_temp``
in the fire plotfiles), so what the threshold sees depends on the coupling:
in a two-way run the fire's own plume, advected over unburned fuel, can start
new fire ahead of the front; in a one-way run only the initial atmosphere
can. The default of 573.15 K (300 C) is the piloted ignition temperature of
fine dead fuel; the FIRE-SMART proposal that ERF-Hazard grew from asked for
"a simple temperature-threshold-based ignition model" as the ignition to pair
with the heat feedback, and this is that model.

Each hot cell is stamped as a disc of radius
``erf.fire.ignition.threshold_radius`` (one fire cell when 0) with the same
``min()`` semantics as the scheduled ignitions, so existing fire is never
overwritten and neighbouring hot cells merge. The stamp is the signed distance
to the disc, in metres on the level-set path and normalised to :math:`[-1, 1]`
on the FARSITE path, matching the primary ignition. Cells in the non-burnable
mask (structures, non-burnable fuel codes, firebreaks) neither ignite nor
receive a stamp. ``erf.fire.ignition.threshold_start_time`` holds the
threshold off until that time so the atmosphere can spin up first. The
ignition is stateless: an ignited cell is an ordinary burning cell from then
on, its arrival time is set by the propagation step, and nothing is added to
the checkpoint. With ``erf.fire.fire_debug = true`` every fire step prints
the number of cells ignited by threshold and the maximum surface temperature.

``Exec/RegTests/FireThresholdIgnition`` runs a two-way grass fire with the
threshold off and on (at 315 K, so the plume of a small disc fire crosses
it), on both propagation paths, with a start time and with a 5 m disc, and
checks that the default changes nothing, that cells ignite by threshold and
the fire ends larger, that nothing ignites before the start time, and that
the disc ignites at least as much as the one-cell stamp.

Input Parameters
----------------

All multi-ignition parameters use the ``erf.fire.ignition.*`` prefix in the ParmParse input file.

.. list-table:: Multi-Ignition Parameters
   :widths: 25 20 35 20
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Default
   * - ``ignition.schedule_file``
     - string
     - Path to ignition schedule CSV file; empty = disabled
     - ""
   * - ``ignition.polygon_file``
     - string
     - Path to polygon vertex CSV file; empty = disabled
     - ""
   * - ``ignition.polygon_type``
     - string
     - "polygon" (closed) or "polyline" (open line fire)
     - "polygon"
   * - ``ignition.polyline_width``
     - Real
     - Half-width of polyline ignition zone [m]
     - 10.0
   * - ``ignition.threshold_enable``
     - bool
     - Ignite unburned, burnable cells whose k = 0 potential temperature exceeds ``threshold_temp``
     - false
   * - ``ignition.threshold_temp``
     - Real
     - Ignition temperature [K]
     - 573.15
   * - ``ignition.threshold_radius``
     - Real
     - Radius [m] of the disc stamped around each hot cell; 0 = one fire cell
     - 0.0
   * - ``ignition.threshold_start_time``
     - Real
     - No threshold ignition before this time [s]
     - 0.0

Example Input File Snippet
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   # Primary ignition (always applied at t=0)
   erf.fire.ignition_x = 500.0
   erf.fire.ignition_y = 1000.0
   erf.fire.ignition_r = 30.0

   # Schedule-based secondary ignitions
   erf.fire.ignition.schedule_file = "Supporting_Files/ignition_schedule_phase11.csv"

   # Alternative: polygon ignition
   erf.fire.ignition.polygon_file = "Supporting_Files/polygon_phase11.csv"
   erf.fire.ignition.polygon_type = "polygon"

   # Or: polyline ignition with 20 m half-width
   erf.fire.ignition.polygon_type = "polyline"
   erf.fire.ignition.polyline_width = 20.0

Limitations
-----------

- **Schedule file size**: No practical limit; all events are loaded and stored in memory.
  Tested with up to 1000 events in a single schedule file.

- **Polygon complexity**: Tested with polygons up to ~100 vertices; no inherent limit but
  GPU memory may constrain very large vertex arrays.

- **Polyline half-width**: Independent of fire intensity. The half-width is constant
  regardless of fuel type or wind speed. For variable-width initialization, use polygon
  mode with a more complex vertex description.

- **Schedule event cancellation**: Events cannot be cancelled once loaded. To disable an
  event, either remove it from the CSV file or set its time outside the simulation duration.

- **2D only**: Polygon and polyline ignition are 2D (fire grid is 2D). Ignition is applied
  uniformly to all vertical levels of the fire grid.

- **No temporal variation**: Polygon and polyline ignitions are applied only at t=0
  during initialization. For time-varying perimeter changes, use schedule-based ignitions
  with multiple sphere events or implement custom logic.

Implementation Files
--------------------

**Header files:**

- ``Source/Fire/ERF_IgnitionSchedule.H``: Schedule loading and application
- ``Source/Fire/ERF_PolygonIgnition.H``: Polygon and polyline initialization

**Modified files:**

- ``Source/Fire/ERF_FireParams.H``: Added ``IgnitionParams`` struct
- ``Source/Fire/ERF_FireLayer.H``: Added private members for schedule state
- ``Source/Fire/ERF_FireLayer.cpp``: Initialization and advance logic for scheduled ignitions
- ``Source/Fire/Make.package``: Added header files to build system

**Test files:**

- ``Exec/CanonicalTests/Fire/Unit_Tests/test_ignition_schedule.py``: Unit tests for schedule parsing, winding number, distance formulas
- ``Exec/CanonicalTests/Fire/Fire_Behavior/Ignition_Patterns/inputs_fire_phase11_schedule``: Regression test with schedule-based ignitions
- ``Exec/CanonicalTests/Fire/Fire_Behavior/Ignition_Patterns/inputs_fire_phase11_polygon``: Regression test with polygon ignition
- ``Exec/CanonicalTests/Fire/Fire_Behavior/Ignition_Patterns/inputs_fire_phase11_polyline``: Regression test with polyline ignition

**Supporting files:**

- ``Exec/CanonicalTests/Fire/Supporting_Files/ignition_schedule_phase11.csv``
- ``Exec/CanonicalTests/Fire/Supporting_Files/polygon_phase11.csv``
- ``Exec/CanonicalTests/Fire/Supporting_Files/polyline_phase11.csv``

References
----------

Finney, M. A. (1998). FARSITE: Fire Area Simulator - Model Development and Evaluation.
Res. Paper RMRS-RP-4, USDA Forest Service, Rocky Mountain Research Station.

Finney, M. A. (2004). FARSITE: Fire Area Simulator model development and evaluation.
Res. Paper RMRS-RP-4 Revised, USDA Forest Service, Rocky Mountain Research Station.
