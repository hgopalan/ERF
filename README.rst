ERF-Fire: wildfire, smoke and dust hazard modeling built on ERF
----

`ERF-Fire` is a stand-alone atmospheric hazard code built on the
`Energy Research and Forecasting (ERF) <https://github.com/erf-model/ERF>`__ model,
which in turn is built upon the `AMReX <https://amrex-codes.github.io/amrex/>`__ software framework
for massively parallel block-structured applications.

ERF-Fire lives on the ``ERF-Fire`` branch of this repository. It is not merged back
into upstream ERF; instead, upstream ``development`` is merged into ``ERF-Fire``
regularly so that the atmospheric core, build system and inputs stay compatible
with ERF. The original ERF README follows below the ERF-Fire sections.

What ERF-Fire adds
~~~~~~~~~~~~~~~~~~

On top of the ERF atmosphere (compressible and anelastic dynamics, terrain-fitted
meshes, LES and PBL closures, MOST surface layer, radiation, microphysics and
land-surface models), ERF-Fire adds:

* a two-dimensional surface fire spread model (``Source/Fire``): level-set
  front propagation, Rothermel and BEHAVE-style rate of spread with the
  standard fuel models, dead and live fuel moisture, fuel maps from
  ESRI ASCII and FARSITE ``.lcp`` files, spotting and crown fire, fire
  acceleration, and two-way coupling of heat, moisture and smoke to the atmosphere;
* wildland-urban interface (WUI) capabilities: building obstacles, structure
  exposure and heat-flux diagnostics;
* a dust emission, transport and deposition module (``Source/Dust``) with
  road, blast and wind-erosion sources, health and visibility diagnostics, and
  coupling to fire lofting (``Source/FireDust``);
* a one-equation (k) RANS closure and terrain-following inflow profiles for
  fire-weather simulations over real terrain;
* canonical, verification and regression cases for all of the above under
  ``Exec/CanonicalTests/Fire``, ``Exec/CanonicalTests/Dust``,
  ``Exec/CanonicalTests/Hazard``, ``Exec/CanonicalTests/Canonical_RANS`` and
  ``Exec/RegTests``.

ERF-Fire Test Status
~~~~~~~~~~~~~~~~~~~~

=================  ================
Regression Tests    |firetests|
=================  ================

.. |firetests| image:: https://github.com/hgopalan/ERF/actions/workflows/ci.yml/badge.svg?branch=ERF-Fire

Building and running ERF-Fire
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ERF-Fire is built with CMake exactly like ERF; see the ERF Getting Started page
linked below for system requirements and the basic build and run instructions.
Clone this repository with its submodules and check out the ``ERF-Fire`` branch::

   git clone --recursive --branch ERF-Fire https://github.com/hgopalan/ERF.git ERF-Fire
   cd ERF-Fire
   cmake -S . -B build -DERF_ENABLE_MPI=ON
   cmake --build build -j

The fire module is enabled by default (``ERF_ENABLE_FIRE=ON``); the dust module is
opt-in (``ERF_ENABLE_DUST=ON``). Each case directory under ``Exec/CanonicalTests``
contains a README with the inputs, expected behaviour and a check script.
ERF-Fire specific tools (terrain and fuel-map preparation, fire maps and
animations) are in ``Exec/Tools`` and next to the cases that use them.

ERF-Fire documentation
~~~~~~~~~~~~~~~~~~~~~~

The theory and inputs of the fire, dust, WUI and RANS capabilities are documented
in the Sphinx sources under ``Docs/sphinx_doc`` (``theory/Fire.rst``,
``theory/DustModule.rst``, ``theory/RANS.rst`` and the pages they link to),
alongside the ERF documentation. Build them locally with::

   pip install -r Docs/sphinx_doc/requirements.txt
   sphinx-build -b html Docs/sphinx_doc build/docs

ERF-Fire development model
~~~~~~~~~~~~~~~~~~~~~~~~~~

Development happens on topic branches off ``ERF-Fire`` with pull requests back
into ``ERF-Fire`` in this repository. New physics is added as opt-in options
with the existing behaviour left as the default, and every change to the fire
model comes with a regression or canonical case. Fixes to the shared
atmospheric core are contributed to upstream ERF and reach ``ERF-Fire`` through
the periodic upstream merges. The ERF coding conventions in CONTRIBUTING.md
apply here as well.

ERF-Fire is distributed under the ERF license below; when using ERF-Fire, please
cite the ERF publications listed at the end of this page.

Energy Research and Forecasting (ERF): An atmospheric modeling code
----

`ERF` is built upon the `AMReX <https://amrex-codes.github.io/amrex/>`_ software framework
for massively parallel block-structured applications.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8102984.svg
   :target: https://doi.org/10.5281/zenodo.8102984

Test Status
~~~~~~~~~~~

=================  =============
Regression Tests    |regtests|
=================  =============

.. |regtests| image:: https://github.com/erf-model/ERF/actions/workflows/ci.yml/badge.svg?branch=development

Getting Started
~~~~~~~~~~~~~~~

See `Getting Started <https://erf.readthedocs.io/en/latest/GettingStarted.html>`_ for instructions as to how to clone the ERF
and AMReX codes, and for how to build and run an ERF example.  Minimum requirements for system software are also given there.

Python tools for pre- and post-processing ERF can be found in the companion `erftools repository <https://github.com/erf-model/erftools/>`_.

Documentation
~~~~~~~~~~~~~~~~~

Documentation of the ERF theory and implementation is available `here <https://erf.readthedocs.io/en/latest/>`_.

In addition, there is doxygen documentation of the ERF Code available `here <https://erf-model.github.io/docs/index.html>`_.

Development model
~~~~~~~~~~~~~~~~~

See CONTRIBUTING.md for how to contribute to ERF development.

Acknowledgments
~~~~~~~~~~~~~~~

The development of the Energy Research and Forecasting (ERF) code is funded by the Wind Energy Technologies Office (WETO),
part of the U.S. Department of Energy (DOE)'s Office of Energy Efficiency & Renewable Energy (EERE).

ERF is built on the `AMReX <https://github.com/AMReX-codes/AMReX>`_ library.

License
~~~~~~~~~

ERF Copyright (c) 2022, The Regents of the University of California,
through Lawrence Berkeley National Laboratory, National Renewable Energy Laboratory,
Lawrence Livermore National Laboratory and Argonne National
Laboratory (subject to receipt of any required approvals from the
U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this
software, please contact Berkeley Lab's Innovation & Partnerships
Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the
U.S. Department of Energy and the U.S. Government consequently retains
certain rights. As such, the U.S. Government has been granted for
itself and others acting on its behalf a paid-up, nonexclusive,
irrevocable, worldwide license in the Software to reproduce,
distribute copies to the public, prepare derivative works, and perform
publicly and display publicly, and to permit other to do so.

The license for ERF can be found in the LICENSE.md file.

Citation
~~~~~~~~~

To cite ERF, please see the following publications:

|JOSS Image|

.. |JOSS Image| image:: https://joss.theoj.org/papers/10.21105/joss.05202/status.svg
   :target: https://doi.org/10.21105/joss.05202

::

   @article{ERF_JOSS,
       title   = {ERF: Energy Research and Forecasting},
       journal = {Journal of Open Source Software},
       author  = {Ann Almgren and Aaron Lattanzi and Riyaz Haque and Pankaj Jha and Branko Kosovic and Jeffrey Mirocha and Bruce Perry and Eliot Quon and Michael Sanders and David Wiersema and Donald Willcox and Xingqiu Yuan and Weiqun Zhang},
       doi     = {10.21105/joss.05202},
       url     = {https://doi.org/10.21105/joss.05202},
       year    = {2023},
       publisher = {The Open Journal of Open Source Software},
       volume  = {8},
       number  = {87},
       pages   = {5202},
   }

|JAMES Image|

.. |JAMES Image| image:: https://zenodo.org/badge/DOI/10.1029/2024MS004884.svg
  :target: https://doi.org/10.1029/2024MS004884

::

   @article{ERF_JAMES,
   author = {Lattanzi, Aaron and Almgren, Ann and Quon, Eliot and Natarajan, Mahesh and Kosovic, Branko and Mirocha, Jeffrey and Perry, Bruce and Wiersema, David and Willcox, Donald and Yuan, Xingqiu and Zhang, Weiqun},
   title = {ERF: Energy Research and Forecasting Model},
   journal = {Journal of Advances in Modeling Earth Systems},
   volume = {17},
   number = {11},
   pages = {e2024MS004884},
   doi = {https://doi.org/10.1029/2024MS004884},
   year = {2025}
   }
