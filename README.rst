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

Documentation
~~~~~~~~~~~~~~~~~

Documentation of the ERF theory and implementation is available `here <https://erf.readthedocs.io/en/latest/>`_ .

In addition, there is doxygen documentation of the ERF Code available `here <https://erf-model.github.io/docs/index.html>`_

Development model
~~~~~~~~~~~~~~~~~

See CONTRIBUTING.md for how to contribute to ERF development.

Acknowledgments
~~~~~~~~~~~~~~~

The development of the Energy Research and Forecasting (ERF) code is funded by the Wind Energy Technologies Office (WETO), part of the U.S. Department of Energy (DOE)'s Office of Energy Efficiency & Renewable Energy (EERE).

The developers of ERF acknowledge and thank the developers of the AMReX-based
`PeleC <https://github.com/AMReX-combustion/PeleC>`_ ,
`FHDeX <https://github.com/AMReX-FHD/FHDeX>`_ and
`AMR-Wind <https://github.com/Exawind/amr-wind>`_ codes.  In the spirit of open source code
development, the ERF project has ported sections of code from each of these projects rather
than writing them from scratch.
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

To cite ERF, please use |JOSS Image|

.. |JOSS Image| image:: https://joss.theoj.org/papers/10.21105/joss.05202/status.svg
   :target: https://doi.org/10.21105/joss.05202

::

   @article{ERF_JOSS,
       title   = {ERF: Energy Research and Forecasting},
       journal = {Journal of Open Source Software}
       author  = {Ann Almgren and Aaron Lattanzi and Riyaz Haque and Pankaj Jha and Branko Kosovic and Jeffrey Mirocha and Bruce Perry and Eliot Quon and Michael Sanders and David Wiersema and Donald Willcox and Xingqiu Yuan and Weiqun Zhang},
       doi     = {10.21105/joss.05202},
       url     = {https://doi.org/10.21105/joss.05202},
       year    = {2023},
       publisher = {The Open Journal of Open Source Software},
       volume  = {8},
       number  = {87},
       pages   = {5202},
   }
