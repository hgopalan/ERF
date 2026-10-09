.. role:: cpp(code)
   :language: c++

.. _sec:Dust:

Dust Model
==========

Overview
--------

The dust model computes wind-blown and activity-driven particulate emission
from bare mineral surfaces such as tailings impoundments, open pits, haul
roads and evaporation ponds, injects that mass into a passive scalar of the
atmosphere, follows its settling and dry deposition, and reports the
regulatory quantities built on it: EPA NAAQS PM2.5 and PM10 averages, MSHA
worker exposure, visibility, respirable silica, short-term exposure limits
and critical-material budgets. It is compiled in with ``ERF_ENABLE_DUST=ON``
and switched on with :cpp:`erf.dust.enable`.

The emission physics lives on a two-dimensional surface grid that can be
finer than the atmosphere in the horizontal, in the same way as the fire grid
of :ref:`sec:Fire`. Each step the surface grid takes the friction velocity,
the wind at a reference height, the surface temperature and the boundary-layer
height from the atmosphere, works out where the wind exceeds the local
threshold friction velocity, adds blasting and haul-road traffic, and hands the
resulting flux back to the atmosphere. The surface state that sets the
threshold (silt, crust, efflorescence, moisture, suppression agents) is read
from rasters and from the output tables of the geochemistry code PHREEQC, which
runs offline. Two optional couplings sit on top: the fire model of ERF-Hazard
can strip crust in burned cells, drive emission with its outflow wind and loft
dust in its convective column, and Lagrangian super-particles can attribute
deposition to its source cells.

.. code-block:: text

   PHREEQC (offline)  --[tables, re-read at an interval]-->  dust grid (2D)
                                                              |         ^
                              emission flux, one-step lag     |         |  u*, wind at zref,
                              (coarsened to the atmosphere)   |         |  T_sfc, PBL height,
                                                              v         |  surface concentration,
                                                         atmosphere (3D)   surface moisture flux
                                                              ^
                             fire grid (ERF-Hazard): burned area, outflow wind, heat flux

The pages below give the physics and the file formats:

.. toctree::
   :maxdepth: 1

   dust_sources
   dust_coupling
   dust_fire
   dust_output

The complete list of inputs with defaults is in :ref:`sec:DustInputs`, and
``Exec/CanonicalTests/Dust/inputs_dust_master_reference`` carries every input
at its default with a comment.

Enabling the model
------------------

The dust model needs the following from the rest of ERF, and checks them once
at startup, aborting with a message that names the input to change:

- a surface layer (:cpp:`zlo.type = "surface_layer"`) with a roughness length
  :cpp:`erf.most.z0`, which supplies :math:`u_*`, the surface temperature and
  the boundary-layer height;
- no domain decomposition in the vertical (:cpp:`amr.max_grid_size_z` at least
  the number of vertical cells), so every rank owns full columns;
- :cpp:`erf.dust.grid_ratio` of at least 1, with every atmosphere box length in
  x and y divisible by it;
- a distribution mapping the same size as the box array, a domain whose
  z index starts at 0 and a positive domain height.

Scalar transport (:cpp:`erf.transport_scalar = true`) is needed for the dust to
move at all, and the MRF scheme (:cpp:`erf.pbl_type = "MRF"`) is the one that
diagnoses the boundary-layer height and carries the scalar diffusivity used by
the dust; the canonical cases use it. Lagrangian particles additionally need
``ERF_ENABLE_PARTICLES=ON``.

Dust grid
---------

The dust grid is a slab one cell deep covering the level-0 domain, refined
horizontally by :cpp:`erf.dust.grid_ratio` (:math:`C` below). With
:math:`C = 1` a dust cell is an atmosphere cell; with :math:`C > 1` each
atmosphere cell holds :math:`C^2` dust cells, which resolves pit walls, road
segments and pond edges below the atmosphere's resolution. The dust box array
is the atmosphere's refined in x and y and its distribution mapping is the
atmosphere's, so each rank owns the same horizontal tiles on both grids and
the exchanges below need no communication. The grid carries the atmosphere's
periodicity. The map between the grids is the integer division
:math:`(i, j) \to (i/C, j/C)` in both directions: emission is averaged down to
the atmosphere and atmosphere fields are copied up to every dust cell of the
column. With the fire coupling on, the fire and dust grids must have the same
ratio.

One dust step
-------------

The dust layer advances once per atmosphere step from ``ERF::Evolve`` with
the atmosphere's :math:`\Delta t`, in this order (``DustLayer::advance`` in
``Source/Dust/ERF_DustLayer.cpp``, with the fire calls around it in
``Source/ERF.cpp``):

1. **Fire pre-step** (ERF-Hazard, when :cpp:`erf.fire_dust_coupling` is on):
   the burned area of the fire level set reduces the crust index and, if
   :cpp:`erf.fire_dust_wind_to_dust` is on, the fire's effective wind raises
   the friction velocity of the cells it covers (:ref:`sec:DustFire`).
2. **Atmosphere fields**: :math:`u_*` from the surface layer, the wind at
   :cpp:`erf.dust.zref` interpolated from the lowest cells, the surface
   temperature and the boundary-layer height. With
   :cpp:`erf.dust.use_terrain_wind` the wind gets the FARSITE terrain
   correction and :math:`u_*` is scaled by the same factor. Without a
   coupled atmosphere the ``test_*`` placeholders are used instead
   (:ref:`sec:DustCoupling`).
3. **PHREEQC** tables are re-read when :cpp:`erf.dust.phreeqc_update_interval_s`
   has elapsed, updating crust, silt, efflorescence and suppression.
4. **Suppression** coverage decays with the surface temperature and wind.
5. **Crust reset**: with the fire coupling on, the crust index is reset to its
   uniform input value and the burned-area reduction is applied again, so the
   crust follows the current fire perimeter rather than decaying step after
   step.
6. **Threshold friction velocity** from the Shao-Lu base (Bagnold with
   :cpp:`erf.dust.threshold_model = bagnold`), the chemistry,
   moisture, suppression and slope factors, then the loading feedback when
   enabled (:ref:`sec:DustSources`).
7. **Emission flux** per bin from the saltation model where
   :math:`u_* > u_{*t}`, plus the blast events due in this step and the active
   haul roads.
8. **Fire lofting** (when :cpp:`erf.fire_dust_lofting_enabled`): the flux is
   multiplied by the convective factor of the fire heat flux.
9. **Diagnostics on the surface grid**: critical-material flux and budget, PM
   classification with its 24-hour averages, MSHA dose, and the release and
   advance of super-particles, all from the lofted flux. The flux and the
   friction velocity are coarsened to the atmosphere columns here, once per
   step.
