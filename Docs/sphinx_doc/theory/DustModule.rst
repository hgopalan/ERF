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
- no domain decomposition in the vertical (:cpp:`amrex.max_grid_size_z` equal
  to the number of vertical cells), so every rank owns full columns;
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
   correction and :math:`u_*` is recomputed from it by a log law. Without a
   coupled atmosphere the ``test_*`` placeholders are used instead
   (:ref:`sec:DustCoupling`).
3. **PHREEQC** tables are re-read when :cpp:`erf.dust.phreeqc_update_interval_s`
   has elapsed, updating crust, silt, efflorescence and suppression.
4. **Suppression** coverage decays with the surface temperature and wind.
5. **Crust reset**: with the fire coupling on, the crust index is reset to its
   uniform input value and the burned-area reduction is applied again, so the
   crust follows the current fire perimeter rather than decaying step after
   step.
6. **Threshold friction velocity** from the Bagnold base, the chemistry,
   moisture, suppression and slope factors, then the loading feedback and the
   dynamic moisture inhibition when enabled (:ref:`sec:DustSources`).
7. **Emission flux** per bin from the saltation model where
   :math:`u_* > u_{*t}`, plus the blast events due in this step and the active
   haul roads.
8. **Fire lofting** (when :cpp:`erf.fire_dust_lofting_enabled`): the flux is
   multiplied by the convective factor of the fire heat flux.
9. **Diagnostics on the surface grid**: critical-material flux and budget, PM
   classification with its 24-hour averages, MSHA dose, and the release and
   advance of super-particles.
10. **Coarsening** of the total flux to the atmosphere grid. It is injected at
    the lowest cell in the slow right-hand side of the *next* step, together
    with the settling tendency and the deposition boundary condition, so the
    coupling has a one-step lag like the fire's.
11. **Return fields** after the slow right-hand side: the surface dust
    concentration and the surface moisture flux come back to the dust grid for
    the next step's threshold.
12. **Output** at the end of the step: the statistics CSV every step, the dust
    plotfile at :cpp:`erf.dust.dust_plot_int`, the PHREEQC feedback files at
    their interval, and the visibility, silica and STEL diagnostics
    (:ref:`sec:DustOutput`).

Fields on the dust grid
-----------------------

All fields have one ghost cell in x and y. The number of components of the
emission flux is :cpp:`erf.dust.n_size_bins`; everything else is one
component unless noted.

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Field
     - Units
     - Meaning
   * - ``dust_ustar_t``
     - m/s
     - Threshold friction velocity after every modifier.
   * - ``dust_ustar_base``
     - m/s
     - Bagnold base threshold from bin 0, or :cpp:`erf.dust.ustar_t_base`.
   * - ``dust_ustar_in``
     - m/s
     - Friction velocity seen by the emission model: from the surface layer,
       the terrain-corrected log law, or the fire wind.
   * - ``dust_wind_ref``
     - m/s
     - Wind components at :cpp:`erf.dust.zref` (2 components).
   * - ``dust_tsfc``, ``dust_pblh``
     - K, m
     - Surface temperature and boundary-layer height from the surface layer.
   * - ``dust_slopes``, ``dust_curvature``
     - -, 1/m
     - Static terrain slope (2 components) and curvature for the terrain
       correction and the slope factor.
   * - ``dust_soil_type``
     - -
     - Surface type code (1-16 STATSGO, 100-104 mine surfaces).
   * - ``dust_silt_fraction``
     - -
     - Surface silt mass fraction.
   * - ``dust_crust_index``, ``dust_efflor``
     - -
     - Crust strength and efflorescence fraction in [0, 1], from rasters,
       PHREEQC and the fire coupling.
   * - ``dust_moisture_flag``, ``dust_surf_moist``
     - -
     - Static moisture inhibition in [0, 1] and the per-step copy of it that
       the threshold uses.
   * - ``dust_suppression``, ``dust_retreat_flag``
     - -
     - Suppression agent coverage in [0, 1] and the flag raised where it has
       fallen below the re-treatment threshold.
   * - ``dust_emission_flux``
     - kg/m²/s
     - Vertical emission flux per bin: saltation plus blasts plus roads,
       times the fire lofting factor.
   * - ``dust_flux_atm``
     - kg/m²/s
     - Total flux on the atmosphere grid, injected at the next step.
   * - ``dust_deposition_rate``
     - kg/m²
     - Cumulative dry deposition, never reset.
   * - ``dust_conc_sfc``
     - kg/m³
     - Dust density of the lowest atmosphere cell.
   * - ``dust_site_id``
     - -
     - Mine site index (0 unassigned, then 1 to N in input order).
   * - ``dust_pm25``, ``dust_pm10``, ``..._24h``, ``..._exceed``
     - µg/m³
     - Instantaneous and 24-hour PM classes and their NAAQS flags.
   * - ``dust_msha_dose``, ``dust_msha_twa``, ``dust_msha_exceed``, ``dust_msha_shift_twa``
     - mg/m³·h, mg/m³
     - Worker exposure dose, 8-hour TWA, PEL flag and last-shift TWA.
   * - ``dust_cm_flux``
     - kg/m²/s
     - Critical-material emission flux.
   * - ``dust_source_map``
     - kg/m²
     - Deposited mass attributed to the source cell by the super-particles.

Source layout
-------------

- ``Source/Dust/ERF_DustParams.H`` reads every ``erf.dust`` key.
- ``ERF_DustGrid``, ``ERF_DustPrerequisites`` build the grid and check the
  requirements above.
- ``ERF_DustLayer`` owns the fields and the per-step sequence;
  ``ERF_DustThreshold.H``, ``ERF_DustEmission``, ``ERF_DustBlastSchedule.H``,
  ``ERF_DustRoadSchedule.H``, ``ERF_DustSuppression.H`` are the emission
  physics; ``ERF_DustSurfaceReader``, ``ERF_PhreeqcReader``,
  ``ERF_DustSiteRegistry.H`` the surface state and its inputs.
- ``ERF_DustWindExtract``, ``ERF_DustTerrainSlope``, ``ERF_DustAtmCoupling``,
  ``ERF_DustSettling.H``, ``ERF_DustDeposition.H``, ``ERF_DustAtmReturn.H``
  are the two-way coupling with the atmosphere, plus the dust scalar
  diffusivity in ``Source/PBL/ERF_ComputeDiffusivityMRF.cpp``.
- ``ERF_DustPlotfile``, ``ERF_DustPlotfileCatalog.H``, ``ERF_DustStatsOutput.H``,
  ``ERF_DustPM.H``, ``ERF_DustNAAQSOutput.H``, ``ERF_DustMSHA.H``,
  ``ERF_DustMSHAOutput.H``, ``ERF_DustVisibility.H``, ``ERF_DustSilica.H``,
  ``ERF_DustSTEL.H``, ``ERF_DustCriticalMaterials.H``,
  ``ERF_DustPHREEQCWriter.H`` are the outputs.
- ``Source/FireDust/ERF_FireDustCoupling`` and
  ``Source/Dust/ERF_DustFireLofting`` are the fire coupling, and
  ``Source/Particles/ERF_DustPC`` the super-particles.
- ``Source/Dust/DUST_DEVELOPMENT.md`` is the development log with the phase
  history behind the file names.

Tests
-----

Every case is under ``Exec/CanonicalTests``; the Dust cases exercise one
piece each on a neutral ABL (3 km square, 8 by 8 by 64 cells, 15 m/s
geostrophic wind, :math:`u_* \approx 0.56` m/s), and the Hazard cases combine
pieces.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Feature
     - Cases
   * - Grid, prerequisites, surface rasters
     - ``Dust/DustScaffold``, ``DustGrid``, ``DustSurfaceReader``
   * - Threshold and saltation emission
     - ``Dust/DustThreshold``, ``DustEmission``; ``Hazard/DustTerrainFlat``
   * - Blasting, haul roads, suppression
     - ``Dust/DustBlast``, ``DustRoadSchedule``, ``DustSuppression``
   * - Wind extraction, injection, settling, deposition, return fields, MRF diffusion
     - ``Dust/DustWindExtract``, ``DustAtmCoupling``, ``DustSettling``,
       ``DustDeposition``, ``DustAtmReturn``, ``DustMRFDiffusion``
   * - Terrain slopes and the terrain wind correction
     - ``Hazard/DustTerrainSlopeEffect``, ``DustGaussianHill``, ``DustGaussianPit``
   * - Plotfile and CSV output, NAAQS, MSHA
     - ``Dust/DustOutput``, ``DustNAAQS``, ``DustMSHA``
   * - Visibility, silica, STEL
     - ``Hazard/DustVisibility``, ``DustSilica``, ``DustSTEL``, ``DustHealthIntegration``
   * - Multi-site PHREEQC, feedback writer, critical materials, particles
     - ``Dust/DustMultiSite``, ``DustPHREEQCFeedback``, ``DustCriticalMaterials``,
       ``DustParticles``
   * - Fire coupling: crust, outflow wind, lofting, combinations
     - ``Hazard/FireDustBaseline``, ``FireDustInteraction1`` to ``3``,
       ``FireDustInteractions12``, ``FireDustInteractions123``,
       ``FireDustLoftingScaling``, ``FireDustWindStrength``,
       ``FireDustMassConservation``, ``FireDustTerrainCoupled``,
       ``FireSmokeDustCoupled``
   * - Everything together
     - ``Dust/DustIntegration``

Limitations
-----------

- **No checkpoint of the dust state.** Nothing on the dust grid is written to
  or read from a checkpoint, so a restarted run starts the deposition
  accumulator, the 24-hour PM averages, the MSHA dose, the suppression
  coverage and the fired blast events from scratch, while the dust already in
  the atmosphere is restored with the conserved state.
- The NetCDF branches of the raster and PHREEQC readers abort; use ESRI ASCII
  and CSV. :cpp:`erf.dust.surface_map_file` is read but not consumed.
- With :cpp:`erf.dust.transport_bins_separately` only bin 0 is returned to the
  surface as the concentration; the other bins are injected and transported
  but do not feed the loading feedback.
- PM classes are assigned by the nominal bin diameter; a bin that straddles
  2.5 or 10 µm goes entirely to one side.
- The MSHA exposure is that of a fixed point, not of a moving worker.
- Super-particles use nearest-cell velocities and an Euler step.
- The haul-road source uses the unpaved-road factor of AP-42 13.2.2 only, the
  road length being the longer side of the road's bounding box.
- All cases run with :cpp:`amr.max_level = 0`; dust transport on refined
  levels is not exercised, and the MRF diffusivity blending for fine levels
  is not implemented.
- The PHREEQC coupling is an offline file exchange, not a runtime call.

References
----------

- Bagnold, R. A. (1941). *The Physics of Blown Sand and Desert Dunes*. Methuen, London.
- Marticorena, B. and Bergametti, G. (1995). Modeling the atmospheric dust cycle: 1. Design of a soil-derived dust emission scheme. *J. Geophys. Res.*, 100, 16415-16430. https://doi.org/10.1029/95JD00690
- Owen, P. R. (1964). Saltation of uniform grains in air. *J. Fluid Mech.*, 20, 225-242. https://doi.org/10.1017/S0022112064001173
- Shao, Y. and Lu, H. (2000). A simple expression for wind erosion threshold friction velocity. *J. Geophys. Res.*, 105, 22437-22443. https://doi.org/10.1029/2000JD900304
- Shao, Y. (2001). A model for mineral dust emission. *J. Geophys. Res.*, 106, 20239-20254. https://doi.org/10.1029/2001JD900171
- Iversen, J. D. and White, B. R. (1982). Saltation threshold on Earth, Mars and Venus. *Sedimentology*, 29, 111-119. https://doi.org/10.1111/j.1365-3091.1982.tb01713.x
- Fecan, F., Marticorena, B. and Bergametti, G. (1999). Parametrization of the increase of the aeolian erosion threshold wind friction velocity due to soil moisture for arid and semi-arid areas. *Ann. Geophys.*, 17, 149-157. https://doi.org/10.1007/s00585-999-0149-7
- Gillies, J. A., Etyemezian, V., Kuhns, H., Nikolic, D. and Gillette, D. A. (2005). Effect of vehicle characteristics on unpaved road dust emissions. *Atmos. Environ.*, 39, 2341-2347. https://doi.org/10.1016/j.atmosenv.2004.05.064
- Hong, S.-Y. and Pan, H.-L. (1996). Nonlocal boundary layer vertical diffusion in a medium-range forecast model. *Mon. Wea. Rev.*, 124, 2322-2339. https://doi.org/10.1175/1520-0493(1996)124<2322:NBLVDI>2.0.CO;2
- Allen, M. D. and Raabe, O. G. (1985). Slip correction measurements of spherical aerosol particles at known Stokes numbers. *Aerosol Sci. Technol.*, 4, 269-286. https://doi.org/10.1080/02786828508959055
- Zhang, L., Gong, S., Padro, J. and Barrie, L. (2001). A size-segregated particle dry deposition scheme for an atmospheric aerosol module. *Atmos. Environ.*, 35, 549-560. https://doi.org/10.1016/S1352-2310(00)00326-5
- Seinfeld, J. H. and Pandis, S. N. (2006). *Atmospheric Chemistry and Physics*, 2nd ed. Wiley.
- Koschmieder, H. (1924). Theorie der horizontalen Sichtweite. *Beitr. Phys. freien Atmos.*, 12, 33-55.
- Parkhurst, D. L. and Appelo, C. A. J. (2013). Description of input and examples for PHREEQC version 3. *USGS Techniques and Methods*, 6-A43. https://pubs.usgs.gov/tm/06/a43/
- U.S. EPA. AP-42, Compilation of Air Emission Factors, Chapter 13.2.2, Unpaved Roads. https://www.epa.gov/air-emissions-factors-and-quantification/ap-42-compilation-air-emission-factors
- U.S. EPA. National Ambient Air Quality Standards for Particulate Matter, 40 CFR Part 50. https://www.epa.gov/pm-pollution
- U.S. MSHA. 30 CFR Part 56, Safety and Health Standards, Surface Metal and Nonmetal Mines. https://www.ecfr.gov/current/title-30/chapter-I/subchapter-K/part-56
- U.S. OSHA. 29 CFR 1910.1053, Respirable Crystalline Silica; 29 CFR 1910.1000 Table Z-1. https://www.osha.gov/silica-crystalline
- U.S. DOE (2023). Critical Materials Assessment. https://www.energy.gov/eere/vehicles/articles/us-doe-critical-materials-assessment
- U.S. NRCS. STATSGO2 soil data. https://www.nrcs.usda.gov/resources/data-and-reports/ssurgo/statsgo2-data
