.. _DustModule:

==============================
ERF-Dust: Dust Transport Layer
==============================

The ERF-Dust module simulates transport and emission of atmospheric dust particles
from terrestrial sources. Phase 3 implements core dust physics (emission computation,
roughness effects, surface properties). Phase 4 couples dust physics with PHREEQC
geochemical output to update mineral crust and salt efflorescence states.

Phase 3 Overview
================

Phase 3 manages:

- **Dust grid discretization**: Coarse dust grid (grid_ratio) independent of atmospheric grid.
- **Threshold friction velocity** (u*_t): Minimum surface stress to mobilize dust, computed via Bagnold formula.
- **Surface properties**: Crust index, silt fraction, suppression agent coverage.
- **Particle size bins**: Logarithmically spaced bins for size-dependent emission.
- **Emission computation**: Dust mass flux per bin and grid cell (Phase 5+ implementation).

**MultiFab inventory (Phase 3):**

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Field
     - Symbol
     - Units
     - Description
   * - dust_ustar_t
     - u*_t
     - m/s
     - Threshold friction velocity. Updated by Phase 4 chemistry.
   * - dust_crust_index
     - C_I
     - [0,1]
     - Mineral crust strength index. Updated by PHREEQC.
   * - dust_silt_fraction
     - f_silt
     - [-]
     - Silt (clay + fine silt) mass fraction [0,1]. Updated by PHREEQC.
   * - dust_suppression
     - S
     - [-]
     - Suppression agent (water, tackifier) coverage [0,1].
   * - dust_emission_flux
     - Q
     - kg/m²/s
     - Dust mass flux per size bin. Component 0..N for bins 0..N.

Threshold Friction Velocity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The threshold friction velocity is the minimum friction velocity required to initiate
dust emission. Phase 3 computes it from the Bagnold formula:

.. math::

   u^*_t = A \sqrt{\frac{\rho_p g d}{\rho_a}}

where:
  - A = Bagnold coefficient [dimensionless] (default 0.0123)
  - :math:`\rho_p` = particle density [kg/m³] (default 2650)
  - g = gravitational acceleration [m/s²]
  - d = representative particle diameter [m]
  - :math:`\rho_a` = air density [kg/m³] (default 1.225)

References:
  - Bagnold, R.A. (1941). The Physics of Blown Sand and Desert Dunes. Methuen, London.
  - Marticorena, B., & Bergametti, G. (1995). J. Geophys. Res., 100, 16415.
    https://doi.org/10.1029/95JD00690

Surface Property Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dust grid surface properties (crust_index, silt_fraction) are initialized from:

1. Constant values specified in input parameters (erf.dust.crust_index, erf.dust.silt_fraction)
2. Spatial maps from files (crust_index_file, silt_fraction_file)
3. Soil type classification (soil_type_file) → property lookup

Phase 3 initialization sequence:

  1. Create dust grid (BoxArray, DistributionMapping) based on grid_ratio.
  2. Allocate MultiFabs on dust grid.
  3. Set u*_t from Bagnold formula.
  4. Set crust_index, silt_fraction from parameters or files.
  5. Initialize suppression to zero (updated later by water/tackifier models).
  6. Zero-initialize emission_flux (computed each timestep).

PHREEQC Geochemical Coupling
-----------------------------

PHREEQC (Parkhurst & Appelo 2013) is used offline to compute geochemical
speciation, mineral crust strength, salt efflorescence state, and toxic
metal composition of mine tailings surfaces. ERF-Dust reads PHREEQC
output files at prescribed intervals and updates the corresponding dust
grid MultiFabs.

Reference:
  Parkhurst, D.L., & Appelo, C.A.J. (2013). PHREEQC version 3.
  USGS Techniques and Methods, book 6, chap. A43.
  https://pubs.usgs.gov/tm/06/a43/

Timescale Justification
~~~~~~~~~~~~~~~~~~~~~~~~

Geochemical processes (crust formation, efflorescence, salt precipitation)
evolve over days to weeks. The ERF atmospheric timestep is seconds. Periodic
file-based updates at intervals of hours to days are sufficient to capture
the evolution of surface chemistry. No runtime PHREEQC calls are made inside
the ERF timestepping loop.

Threshold Friction Velocity Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After reading crust_index and efflorescence from PHREEQC output,
update_ustar_t_from_chemistry applies:

.. math::

   u^*_{t,\mathrm{new}} = u^*_{t,\mathrm{base}}
     \times (1 - \alpha_c \cdot C_I)
     \times (1 - \alpha_e \cdot E_f)

where:
  - :math:`C_I` = crust index [0,1]
  - :math:`E_f` = efflorescence fraction [0,1]
  - :math:`\alpha_c` = crust reduction coefficient (default 0.5)
  - :math:`\alpha_e` = efflorescence reduction coefficient (default 0.3)
  - u*_{t,base} = Bagnold u*_t (stored at initialization)

The result is clamped to USTAR_T_MIN = 0.05 m/s.

Physical interpretation: mineral crusts and salt efflorescence increase
surface cohesion, increasing the threshold for dust mobilization. A fully
crusted surface (C_I=1) increases u*_t by 50%; full efflorescence (E_f=1)
increases it by 30%.

Reference:
  Marticorena, B., & Bergametti, G. (1995). J. Geophys. Res., 100, 16415.
  https://doi.org/10.1029/95JD00690

PHREEQC Input File Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CSV format (.csv):**

  - First row: comma-separated header with column names.
  - Subsequent rows: one row per dust grid cell in row-major order.
    Ordering: j=0 (southernmost row), i=0..nx-1, then j=1, etc.
  - Missing values: cells with value -9999 are replaced with nodata_fill.

Example (8×8 dust grid):

.. code-block:: text

   i,j,crust_index,silt_fraction,efflorescence,suppression_mod,metal_as_bin0
   0,0,0.0,0.10,0.0,0.0,0.0
   1,0,0.0,0.10,0.0,0.0,0.0
   ...
   7,7,0.0,0.10,0.0,0.0,0.0

**NetCDF format (.nc):**

  - Requires ERF_ENABLE_NETCDF=ON.
  - Variable names specified by column/variable name parameters.
  - Coordinate variables (lon, lat, x, y) used for bilinear interpolation.
  - Full implementation deferred to Phase 5+; currently aborts with "use CSV format".

PHREEQC ParmParse Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Description
   * - ``erf.dust.phreeqc_output_file``
     - Path to PHREEQC output file (.csv or .nc). Empty = no update.
   * - ``erf.dust.phreeqc_update_interval_s``
     - Interval between PHREEQC file reads [s]. Default = 86400 (1 day).
   * - ``erf.dust.alpha_crust``
     - Crust reduction coefficient for u*_t [-]. Default = 0.5.
   * - ``erf.dust.alpha_efflor``
     - Efflorescence reduction coefficient for u*_t [-]. Default = 0.3.
   * - ``erf.dust.phreeqc_crust_var``
     - CSV column / NetCDF variable name for crust index. Default = "crust_index".
   * - ``erf.dust.phreeqc_silt_var``
     - CSV column / NetCDF variable name for silt fraction. Default = "silt_fraction".
   * - ``erf.dust.phreeqc_efflor_var``
     - CSV column / NetCDF variable name for efflorescence fraction. Default = "efflorescence".
   * - ``erf.dust.phreeqc_supp_var``
     - CSV column / NetCDF variable name for suppression modifier. Default = "suppression_mod".
   * - ``erf.dust.phreeqc_metal_var``
     - CSV column / NetCDF variable name for toxic metal mass fraction (bin 0). Default = "metal_as_bin0".

Input Deck Example
------------------

Enable dust with PHREEQC coupling:

.. code-block:: makefile

   erf.dust.enable                    = true
   erf.dust.n_size_bins               = 3
   erf.dust.grid_ratio                = 2
   erf.dust.particle_density          = 2650.0
   erf.dust.z0_dust                   = 0.01
   erf.dust.silt_fraction             = 0.10
   erf.dust.threshold_A_coeff         = 0.0123
   erf.dust.crust_index               = 0.0
   
   # PHREEQC coupling (Phase 4)
   erf.dust.phreeqc_update_interval_s = 86400.0
   erf.dust.phreeqc_output_file       = "phreeqc_output.csv"
   erf.dust.phreeqc_crust_var         = "crust_index"
   erf.dust.phreeqc_silt_var          = "silt_fraction"
   erf.dust.phreeqc_efflor_var        = "efflorescence"
   erf.dust.phreeqc_supp_var          = "suppression_mod"
   erf.dust.phreeqc_metal_var         = "metal_as_bin0"
   erf.dust.alpha_crust               = 0.5
   erf.dust.alpha_efflor              = 0.3

Regression Test: PhreeqcReader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PhreeqcReader test (`Exec/CanonicalTests/Dust/PhreeqcReader/inputs`)
validates:

1. CSV file read and column lookup.
2. Rank-0 read + ParallelDescriptor::Bcast + GPU fill pattern.
3. PHREEQC update interval tracking in DustLayer::advance.
4. Marticorena & Bergametti u*_t reduction formula.
5. Consistency between dust_ustar_t and dust_ustar_base (u*_t_base preserved read-only).

Test configuration: 8×8×16 atmospheric grid, 8×8×1 dust grid (grid_ratio=1),
max_step=2, phreeqc_update_interval_s=1.0 (triggers on every timestep).

Building with Dust Support
---------------------------

Enable dust module during CMake configuration:

.. code-block:: bash

   cd Build
   cmake -DERF_ENABLE_DUST=ON ..
   make -j4

Build without dust:

.. code-block:: bash

   cmake -DERF_ENABLE_DUST=OFF ..
   make -j4

All code wrapped in `#ifdef ERF_USE_DUST` / `#endif`.

References
----------

Bagnold, R.A. (1941). The Physics of Blown Sand and Desert Dunes.
  Methuen, London.

Marticorena, B., & Bergametti, G. (1995). J. Geophys. Res., 100, 16415.
  https://doi.org/10.1029/95JD00690

Shao, Y., & Lu, H. (2000). J. Geophys. Res., 105, 22437.
  https://doi.org/10.1029/2000JD900304

Parkhurst, D.L., & Appelo, C.A.J. (2013). PHREEQC version 3.
  USGS Techniques and Methods, book 6, chap. A43.
  https://pubs.usgs.gov/tm/06/a43/
