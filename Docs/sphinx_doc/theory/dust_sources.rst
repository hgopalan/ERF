.. role:: cpp(code)
   :language: c++

.. _sec:DustSources:

Dust Emission and Surface State
===============================

Threshold friction velocity
---------------------------

Emission starts where the friction velocity exceeds a threshold
:math:`u_{*t}` set by the grain size and everything that binds the surface.
The base value is Bagnold's

.. math::

   u_{*t,\mathrm{base}} = A \sqrt{\frac{\rho_p\, g\, d}{\rho_a}}

with :math:`A` = :cpp:`erf.dust.threshold_A_coeff` (0.0123), :math:`\rho_p`
= :cpp:`erf.dust.particle_density`, :math:`\rho_a` =
:cpp:`erf.dust.rho_air` and :math:`d` the diameter of bin 0 in
:cpp:`erf.dust.bin_diameter_um`; :cpp:`erf.dust.ustar_t_base` overrides it
when non-negative. The per-cell threshold (``compute_ustar_t_full`` in
``ERF_DustThreshold.H``) is then

.. math::

   u_{*t} = u_{*t,\mathrm{base}}\; \frac{f_\mathrm{chem}}{f_\mathrm{moist}\, f_\mathrm{supp}}\; f_\mathrm{slope}

.. math::

   f_\mathrm{chem}  = (1 + \alpha_c\, C_I)(1 + \alpha_e\, E_f), \qquad
   f_\mathrm{moist} = 1 + 4\, w, \qquad
   f_\mathrm{supp}  = 1 + 6\, s

where :math:`C_I` is the crust index, :math:`E_f` the efflorescence
fraction, :math:`w` the moisture inhibition and :math:`s` the suppression
coverage, all clamped to [0, 1], with :math:`\alpha_c` =
:cpp:`erf.dust.alpha_crust` and :math:`\alpha_e` =
:cpp:`erf.dust.alpha_efflor`. Crust and efflorescence *raise* the threshold
(a bound surface is harder to erode), which is why the fire coupling lowers
emission by removing crust. Moisture and suppression divide the base in the
code but are themselves greater than one, so they also raise the threshold.
The slope factor is the upslope correction of Iversen and White (1982),

.. math::

   f_\mathrm{slope} = \sqrt{\max\!\left(0.1,\; \cos\beta + \frac{|\nabla z|}{\tan 35^\circ}\right)},
   \qquad \cos\beta = \frac{1}{\sqrt{1 + |\nabla z|^2}}

from the terrain slope on the dust grid. The result is clamped to
[0.001, 5] m/s. Two feedbacks from the atmosphere follow when enabled: the
Shao (2001) loading feedback :math:`u_{*t} \to u_{*t}(1 + \alpha_L C_\mathrm{sfc})`
with :math:`\alpha_L` = :cpp:`erf.dust.loading_feedback_coeff` and
:math:`C_\mathrm{sfc}` the dust density of the lowest atmosphere cell, and the
Fecan (1999) dynamic moisture factor of :ref:`sec:DustCoupling`.

Saltation and vertical flux
---------------------------

Where :math:`u_* > u_{*t}` the horizontal saltation flux follows Owen (1964)
as used by Marticorena and Bergametti (1995),

.. math::

   Q_s = C_s\, \frac{\rho_a}{g}\, u_*^3 \left(1 - \frac{u_{*t}}{u_*}\right)
         \left(1 + \frac{u_{*t}}{u_*}\right)^2, \qquad C_s = 2.61

and the vertical flux of every bin is the same sandblasting fraction of it,

.. math::

   F_i = \alpha\, f_\mathrm{silt}\, Q_s, \qquad
   \log_{10}\alpha = 0.134\, f_\mathrm{clay} - 6, \qquad
   f_\mathrm{clay} = 0.2\, f_\mathrm{silt}

with :math:`f_\mathrm{silt}` the silt fraction of the cell. Each bin's flux
is clamped to :math:`10^{-3}` kg/m²/s. No emission occurs below
:math:`u_*/u_{*t} = 1.001`.

Blasting
--------

:cpp:`erf.dust.blast_schedule_file` lists timed events, one per line
(``#`` or ``!`` starts a comment):

.. code-block:: text

   time_s  cx  cy  radius  mass_kg_m2  [mineral_type]  [priority]

Events with a time in the current step's interval fire in order of
descending priority then ascending time. Every dust cell whose centre lies
within ``radius`` of (``cx``, ``cy``) receives, in every bin,

.. math::

   F_\mathrm{blast} = \frac{m\, r_b}{\Delta t\, N_\mathrm{bins}}

with :math:`m` the mass per unit area, :math:`r_b` =
:cpp:`erf.dust.blast_reactivity` and :math:`N_\mathrm{bins}` =
:cpp:`erf.dust.n_size_bins`, clamped to :math:`10^{-2}` kg/m²/s per bin. The
mineral type (0 quartz tailings, 1 lithium brine, 2 rare-earth tailings,
3 copper tailings) is carried for diagnostics. Rank 0 reads the file and
broadcasts it.

Haul roads
----------

:cpp:`erf.dust.road_schedule_file` lists road segments:

.. code-block:: text

   road_name  x_lo_m  y_lo_m  x_hi_m  y_hi_m  road_width_m  vehicle_weight_t  silt_pct  vmt_per_h  start_s  end_s

A road is active from ``start_s`` to ``end_s`` (``-1`` means for the whole
run); overlapping entries for one road give shift patterns. The emission
factor is the unpaved-road relation of EPA AP-42 13.2.2,

.. math::

   E = 2.6 \left(\frac{s}{12}\right)^{0.8} \left(\frac{W}{3}\right)^{0.4}
   \quad [\mathrm{g/VKT}]

with :math:`s` the silt content in percent and :math:`W` the vehicle mass in
tons, spread over the road as

.. math::

   F_\mathrm{road} = \frac{10^{-3}\, E\, \mathrm{VMT}}{W_\mathrm{road}\, L_\mathrm{road}\, 3600}
   \quad [\mathrm{kg/m^2/s}]

where :math:`L_\mathrm{road}` is the longer side of the bounding box. The flux
is added to bin 0 of every cell in the box and clamped to :math:`10^{-3}`
kg/m²/s; each active road appends a row to :cpp:`erf.dust.road_diag_file`.
Wind, blast and road fluxes add in the same emission field.

Suppression agents
------------------

The coverage :math:`C` of water or a chemical suppressant decays every step,

.. math::

   C \leftarrow C \exp\!\left(-\frac{\Delta t}{\tau}\right), \qquad
   \tau = \frac{\tau_\mathrm{base}}{f_T\, f_\mathrm{wind}}, \qquad
   f_T = e^{0.05 (T_s - 293.15)}, \quad f_\mathrm{wind} = 1 + 0.1\, U

with :math:`\tau_\mathrm{base}` = :cpp:`erf.dust.supp_tau_base_s`, the
surface temperature :math:`T_s` in K and the wind :math:`U` at
:cpp:`erf.dust.zref` in m/s (the domain maxima of both). Coverage below 0.01
is set to zero, and cells below 0.2 raise the re-treatment flag. The initial
coverage comes from :cpp:`erf.dust.suppression_file`; PHREEQC can also update
it.

Surface property rasters
------------------------

Five optional ESRI ASCII rasters give the surface state:
:cpp:`erf.dust.soil_type_file`, :cpp:`erf.dust.silt_fraction_file`,
:cpp:`erf.dust.crust_index_file`, :cpp:`erf.dust.moisture_flag_file` and
:cpp:`erf.dust.suppression_file`; an empty path keeps the uniform value. The
header is the standard six lines (``ncols``, ``nrows``, ``xllcorner``,
``yllcorner``, ``cellsize``, ``nodata_value``), rows run north to south in
the file and are reversed on reading, rank 0 reads and broadcasts, and the
values are interpolated bilinearly to the dust cell centres. A ``.nc`` path
selects the NetCDF reader, which is not implemented and aborts.

Soil type codes: 0 undefined (no emission), 1-16 the STATSGO classes,
100 mine tailings, 101 lithium brine pond, 102 rare-earth tailings, 103
copper tailings, 104 unpaved haul road.

PHREEQC geochemistry
--------------------

PHREEQC computes the mineral crust, salt efflorescence and metal content of
the surface offline, on timescales of days, and its output table is re-read
every :cpp:`erf.dust.phreeqc_update_interval_s` seconds from
:cpp:`erf.dust.phreeqc_output_file`. The file is a CSV whose first row names
the columns; the rows follow the dust grid in row-major order (all i for
j = 0, then j = 1, and so on). The columns named by
:cpp:`erf.dust.phreeqc_crust_var`, ``phreeqc_silt_var``,
``phreeqc_efflor_var``, ``phreeqc_supp_var`` and ``phreeqc_metal_var``
replace the crust index, silt fraction, efflorescence, suppression modifier
and bin-0 metal fraction, after which the threshold is recomputed with the
factors above. A ``.nc`` path aborts. The deposition written back for the
next PHREEQC run is described in :ref:`sec:DustOutput`.

Mine sites
----------

:cpp:`erf.dust.site_names` with the matching ``site_phreeqc_files`` and the
bounding boxes ``site_x_lo``, ``site_y_lo``, ``site_x_hi``, ``site_y_hi``
assign every dust cell a site index (0 outside every box; the last site
listed wins where boxes overlap), stored in ``dust_site_id`` and written to
the plotfile. Each site can carry its own PHREEQC table, an empty entry
falling back to the global one, and the deposition and critical-material
budgets are reported per site.
