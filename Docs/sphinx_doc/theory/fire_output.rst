.. role:: cpp(code)
   :language: c++

.. _sec:FireOutput:

Fire Diagnostics and Output
===========================

Flame diagnostics
-----------------

Computed every fire step on burned cells (:math:`\phi < 0`) and zero
elsewhere:

- **Byram fireline intensity** :math:`I_B = h\, w_{consumed}\, R` [kW/m],
  with :math:`h` the heat content [kJ/kg], :math:`w_{consumed}` the fuel
  consumed so far [kg/m²] and :math:`R` the rate of spread. It drives ember
  lofting, crown-fire initiation and the flame diagnostics below.
- **Thomas flame length** :math:`L = 0.0775\, I_B^{0.46}` [m].
- **Flame temperature**, by :cpp:`erf.fire.flame_temp_method`:
  ``"byram_radiant"`` (default) :math:`T = T_a + 800\,(I_B/1000)^{0.25}`;
  ``"mcalpine_heat"``, the adiabatic estimate of McAlpine and Xanthopoulos
  (1989) from the heat content and moisture, clipped to
  :math:`T_a + 100 < T < 1500` K; ``"nelson_emissivity"``, the
  Stefan-Boltzmann inversion :math:`T = (1000\, I_B / (\sigma \epsilon))^{0.25}`
  with :math:`\epsilon = 0.9`. :math:`T_a` is
  :cpp:`erf.fire.flame_temp_T_amb`.
- **Flame tilt** from vertical, :math:`\arctan(U / v_b)` in degrees, from
  the effective wind speed and a buoyancy velocity built from the intensity,
  flame length and radiant flux with :cpp:`erf.fire.flame_tilt_rho_air` and
  :cpp:`erf.fire.flame_tilt_T_amb`. Off unless
  :cpp:`erf.fire.compute_flame_tilt` is set; a diagnostic only.

Fire plotfiles
--------------

The fire grid has its own plotfile stream, separate from the atmospheric
plotfiles, written to ``erf.fire_plot_file`` (default ``plt_fire_``) every
``erf.fire_plot_int`` steps or every ``erf.fire_plot_per`` seconds. Each is
a single-level AMReX plotfile on the fire grid with a small JSON sidecar
giving the grid ratio and variable count. Variables appear in this fixed
order; the optional blocks are present only when their feature is on:

.. list-table::
   :widths: 30 12 58
   :header-rows: 1

   * - Variable
     - Unit
     - Present when
   * - ``fire_phi``
     - -
     - always
   * - ``fire_ros``
     - m/s
     - always
   * - ``fire_wind_eff_u``, ``fire_wind_eff_v``
     - m/s
     - always
   * - ``fire_wind_ref_u``, ``fire_wind_ref_v``
     - m/s
     - always
   * - ``fire_extract_z``
     - m
     - always
   * - ``fire_slope_x``, ``fire_slope_y``
     - -
     - always
   * - ``fire_fuel_mc_1hr``, ``fire_fuel_mc_10hr``, ``fire_fuel_mc_100hr``
     - fraction
     - always
   * - ``fire_surface_temp_K``, ``fire_surface_rh``
     - K, 0-1
     - always
   * - ``fire_heat_flux``
     - W/m²
     - always
   * - ``fire_fuel_load``
     - kg/m²
     - always
   * - ``fire_fireline_intensity``, ``fire_flame_length``
     - kW/m, m
     - always
   * - ``fire_arrival_time``
     - s
     - always
   * - ``fire_fuel_mc_lh``, ``fire_fuel_mc_lw``
     - fraction
     - live moisture components present (BEHAVE path)
   * - ``fire_lofting_height``, ``fire_spot_brand_count``, ``fire_spot_max_dist``, ``fire_spot_active``
     - m, -, m, -
     - ``erf.fire.spotting.enable``
   * - ``fire_crown_active``, ``fire_crown_load``, ``fire_crown_fraction_burned``
     - -, kg/m², -
     - ``erf.fire.crown.enable``
   * - ``fire_flame_tilt``
     - deg
     - crown fire on and ``erf.fire.compute_flame_tilt``
   * - ``fire_flame_temp``
     - K
     - ``erf.fire.crown.enable``
   * - ``fire_fuel_model_code``
     - -
     - a spatial fuel map is loaded
   * - ``fire_ros_model_weight``
     - 0-1
     - ``erf.fire.ros_model = hybrid``
   * - ``fire_structure_height``
     - m
     - ``erf.fire.structures.enable`` or the hybrid ``structure`` selector
   * - ``fire_nonburnable``
     - 0/1
     - structures, ``fuel_map.nonburnable_codes`` or ``firebreak.use_mask`` configured

Fire statistics CSV
-------------------

When :cpp:`erf.fire.write_fire_stats_csv` is true (default) one line per
fire step is appended to :cpp:`erf.fire.fire_stats_csv_file` (default
``fire_stats.csv``) with the columns

``step, time_s, burned_area_ha, perimeter_km, active_front_cells, head_ros_ms,
major_axis_m, minor_axis_m, heat_flux_max_Wm2, spot_fires_this_step,
max_spot_dist_m``.

Burned area and perimeter come from the arrival-time field, the head rate is
the maximum rate of spread over burning cells, and the axes are those of the
burned region's bounding ellipse. The last two columns are zero unless
spotting is on.

Arrival-time probes
-------------------

:cpp:`erf.fire.probes = x1 y1 x2 y2 ...` [m] lists points at which the
arrival time is reported. Each probe prints one line,
``[FIRE PROBE] n x= y= cell=(i,j) arrival_time_s=``, on the fire step its
cell first burns, and nothing afterwards; a probe outside the domain is
reported once and ignored. The obstacle regression suite uses these to
tabulate arrival at building faces.

Debug output
------------

:cpp:`erf.fire.fire_debug` prints, every step, the wind extraction range,
the rate-of-spread maximum and mean over burning cells, the number of
subcycles, the count of active fire cells, the flux and intensity maxima,
and, where relevant, the hybrid weight summary and the coupling tendency
check. The regression scripts parse these lines, so their formats are
stable.

Checkpoint and restart
----------------------

The atmospheric checkpoint carries the fire state as ``FirePhi``,
``FireArrivalTime``, ``FireROS``, ``FireFuelLoad``, ``FireFuelMC``,
``FireDispAccum``, the lagged flux buffers ``FireQAtmPrev`` and
``FireQLatAtmPrev`` that the next step injects, and, with crown fire,
``FireCrownActive`` and ``FireCrownLoad``. On restart the fire layer is initialised from the inputs
first, so a spatial fuel map, firebreaks, a hybrid weight or a structure
mask must still be available, and the checkpointed fields are then read
over the initial ones. Diagnostics are recomputed on the first step.

References
----------

- Byram, G. M. (1959). Combustion of forest fuels. In: Forest Fire: Control and Use, McGraw-Hill, 61-89.
- Thomas, P. H. (1963). The size of flames from natural fires. Ninth Symposium (International) on Combustion, 844-859.
- McAlpine, R. S. and Xanthopoulos, G. (1989). Predicted vs. observed fire spread rates in Ponderosa pine fuel beds. Proceedings of the 10th Conference on Fire and Forest Meteorology.
- Nelson, R. M. (1980). Flame characteristics for fires in southern fuels. USDA Forest Service Research Paper SE-205.
