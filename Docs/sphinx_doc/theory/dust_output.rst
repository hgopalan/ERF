.. role:: cpp(code)
   :language: c++

.. _sec:DustOutput:

Dust Diagnostics and Output
===========================

Dust plotfile
-------------

The dust grid has its own box array and geometry, so its fields are written
as a separate single-level plotfile, ``<prefix><step:05d>`` with
:cpp:`erf.dust.dust_plot_prefix`, every :cpp:`erf.dust.dust_plot_int` steps
and at the final step (``-1`` disables, ``0`` writes only the final step).
Besides the AMReX ``Header`` and ``Level_0`` data it carries
``DustMetadata.json`` with the time, step, grid ratio and field count. The
21 fields, in order:

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Field
     - Units
     - Meaning
   * - ``dust_emission_flux``
     - kg/m²/s
     - Bin-0 emission flux (wind, blasts, roads, lofting).
   * - ``dust_ustar_in``
     - m/s
     - Friction velocity used by the emission model.
   * - ``dust_ustar_t``
     - m/s
     - Threshold friction velocity.
   * - ``dust_deposition_rate``
     - kg/m²
     - Cumulative dry deposition.
   * - ``dust_conc_sfc``
     - kg/m³
     - Dust density of the lowest atmosphere cell.
   * - ``dust_surf_moist``
     - -
     - Moisture inhibition used by the threshold.
   * - ``dust_suppression``, ``dust_retreat_flag``
     - -
     - Suppression coverage and re-treatment flag.
   * - ``dust_pm25_ug_m3``, ``dust_pm10_ug_m3``
     - µg/m³
     - Instantaneous PM2.5 and PM10.
   * - ``dust_pm25_24h_ug_m3``, ``dust_pm10_24h_ug_m3``
     - µg/m³
     - 24-hour running averages.
   * - ``dust_pm25_exceed``, ``dust_pm10_exceed``
     - -
     - 1 where the 24-hour average exceeds 35 or 150 µg/m³.
   * - ``dust_msha_dose_mg_m3_h``, ``dust_msha_twa_mg_m3``
     - mg/m³·h, mg/m³
     - Shift dose and 8-hour TWA.
   * - ``dust_msha_exceed``, ``dust_msha_shift_twa``
     - -, mg/m³
     - PEL flag and TWA at the last shift end.
   * - ``dust_source_map``
     - kg/m²
     - Deposition attributed to the source cell by super-particles (zero
       without particles).
   * - ``dust_site_id``
     - -
     - Mine site index.
   * - ``dust_cm_flux_kg_CM_m2_s``
     - kg/m²/s
     - Critical-material emission flux.

Statistics CSV
--------------

:cpp:`erf.dust.dust_diag_file` gets one row per step: ``step``, ``time_s``,
``emission_total_kg_s`` (domain integral of the flux),
``deposition_total_kg_m2``, ``ustar_max_m_s``, ``flux_max_kg_m2_s`` and
``conc_sfc_max_kg_m3``. Rank 0 writes every CSV below; all ranks take part in
the reductions.

EPA NAAQS
---------

Each bin contributes its whole mass to PM10 when its diameter in
:cpp:`erf.dust.bin_diameters` is at most 10 µm and to PM2.5 when at most
2.5 µm (with a single transported scalar the bin-0 diameter decides). The
24-hour averages are exponential running means,

.. math::

   \bar C \leftarrow \bar C\, \frac{T - \Delta t}{T} + C\, \frac{\Delta t}{T}, \qquad T = 86400\ \mathrm{s}

and the flags mark cells above the 40 CFR 50 thresholds of 35 µg/m³ (PM2.5)
and 150 µg/m³ (PM10). :cpp:`erf.dust.dust_naaqs_file` gets per step the
instantaneous and 24-hour maxima of both classes and the number of cells
above each threshold.

MSHA worker exposure
--------------------

The respirable concentration is PM10 in mg/m³. Per cell the shift dose
:math:`D = \int C\, dt` [mg/m³·h] accumulates, the 8-hour time-weighted
average is :math:`D / 8`, and the flag marks cells above
:cpp:`erf.dust.msha_pel_mg_m3` (30 CFR 56.5001). When
:math:`\lfloor t / T_\mathrm{shift} \rfloor` increases, with
:cpp:`erf.dust.msha_shift_duration_s`, the peak TWA of the ending shift is
recorded and the dose reset. Files: :cpp:`erf.dust.msha_exposure_file`
(``step, time_s, TWA_max_mg_m3, n_cells_exceed, dose_max_mg_m3_h``),
:cpp:`erf.dust.msha_shift_file` (``shift_number, shift_end_time_s,
TWA_peak_mg_m3, n_cells_exceed``) and ``msha_receptor_<name>.csv`` for every
entry of :cpp:`erf.dust.msha_receptor_names` at (``msha_receptor_x``,
``msha_receptor_y``), sampling PM10 at the nearest cell. This is the exposure
of a fixed point, not of a worker who moves.

Visibility, silica and short-term exposure
------------------------------------------

Three CSV-only diagnostics computed from PM10 every step:

- **Visibility** (:cpp:`erf.dust.visibility_enable`): Koschmieder's
  :math:`V = 3.912 / (k_\mathrm{ext}\, \mathrm{PM10})` with
  :cpp:`erf.dust.visibility_k_ext` in m²/kg and PM10 in kg/m³, capped at
  100 km. Cells below :cpp:`erf.dust.visibility_road_closure_m` and
  ``visibility_warning_m`` are counted; the file holds
  ``step, time_s, visibility_min_m, visibility_mean_m,
  haul_road_closure_cells, reduced_visibility_cells``.
- **Respirable crystalline silica** (:cpp:`erf.dust.silica_enable`):
  :math:`\mathrm{RCS} = \mathrm{PM10}\, f_\mathrm{silica}` with
  :cpp:`erf.dust.silica_fraction_rcs`, compared with
  :cpp:`erf.dust.silica_osha_pel_mg_m3` (29 CFR 1910.1053); the file holds
  ``step, time_s, rcs_max_mg_m3, rcs_mean_mg_m3, osha_pel_exceeded_cells``.
- **Short-term exposure limit** (:cpp:`erf.dust.stel_enable`): the running
  mean of PM10 over :cpp:`erf.dust.stel_averaging_s` (15 minutes by default),
  compared with :cpp:`erf.dust.stel_threshold_mg_m3` (29 CFR 1910.1000); the
  file holds ``step, time_s, stel_max_mg_m3, stel_mean_mg_m3,
  stel_exceeded_cells``.

Critical materials
------------------

With :cpp:`erf.dust.cm_fractions` set, the critical-material flux is
:math:`F_\mathrm{cm} = \sum_b F_b\, f_b` per cell and goes to the plotfile;
:cpp:`erf.dust.cm_budget_file` gets per step one row per site plus a domain
total (``site_index`` -1): ``step, time_s, site_index, site_name,
cm_emission_kg_s``.

PHREEQC feedback
----------------

Every :cpp:`erf.dust.phreeqc_feedback_interval_s` seconds, and at the end of
the run, the cumulative deposition is written for the next offline PHREEQC
run: :cpp:`erf.dust.phreeqc_feedback_file` is overwritten with one row per
dust cell, ``x_centre_m  y_centre_m  deposition_kg_m2``, after comment lines
with the time and step, and :cpp:`erf.dust.phreeqc_site_summary_file` is
appended with ``time_s, site_index, site_name, total_deposition_kg`` per
site (site 0 being the unassigned cells). An interval of 0 writes only the
final state. The intended loop is: run, let PHREEQC turn the deposition into
new crust and efflorescence, point ``phreeqc_output_file`` or the rasters at
its result, and run again.

Lagrangian super-particles
--------------------------

With ``ERF_ENABLE_PARTICLES`` and :cpp:`erf.dust.enable_particles`, every
:cpp:`erf.dust.particle_release_interval` steps each emitting cell releases
one particle carrying the mass :math:`F\, A_\mathrm{cell}\, \Delta t`, its
Stokes settling velocity, its release time and its source cell. Particles move
with the nearest-cell velocity plus the settling velocity in an Euler step,
and a particle that falls below half the lowest cell height is removed and
its mass added to ``dust_source_map`` at its source cell. The map is never
reset, so it accumulates the deposition attributable to each source over the
run.
