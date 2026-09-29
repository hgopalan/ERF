.. _sec:MovingBodies:

Moving Bodies: OpenFAST Turbines
================================

The moving-bodies framework (``Source/MovingBodies``) represents wind turbines
whose aerodynamics and structural response are computed by
`OpenFAST <https://openfast.readthedocs.io>`_ while ERF supplies the wind. It
is separate from the wind farm parameterizations of :ref:`sec:WindFarmModels`,
which model turbines with thrust curves; here the loads come from OpenFAST's
blade-element solution on the turbine's own blade and tower nodes.

Coupling
--------

ERF talks to OpenFAST through its external-inflow C interface (``ExtInfw``,
OpenFAST 4). Every ERF step:

#. the flow velocity at OpenFAST's velocity nodes (hub, blade and tower
   structural nodes) is handed to OpenFAST;
#. OpenFAST advances its own, smaller, time step ``n`` times, where ``n`` is the
   ratio of ``erf.fixed_dt`` to the OpenFAST ``DT``; ERF refuses to start when
   the ratio is not a whole number;
#. OpenFAST returns the positions of its actuator force points and the
   aerodynamic force on each.

Each turbine is owned by one MPI rank, which is the only rank that calls
OpenFAST; the node positions and forces are broadcast afterwards so that every
rank sees the same turbine. The OpenFAST model must set ``CompInflow = 2``
(external inflow) so that the velocities come from ERF rather than from
InflowWind.

Solver requirements
-------------------

This version supports the anelastic solver only, with a fixed time step and a
single level, and the AMReX floating-point traps must be off: OpenFAST's
initialisation raises exceptions of its own, and a trapped run dies inside the
library. The requirements are checked at start-up.

Diagnostics
-----------

Every turbine writes ``<output_root>_erf.csv`` with the time, rotor speed, the
thrust vector (the sum of the blade-node forces, as OpenFAST reports them: the
force of the fluid on the structure, so along the inflow), the aerodynamic
torque about the hub axis and the power (torque times rotor speed). OpenFAST
also writes its own output files as configured in the ``.fst`` file.

Inputs are listed in :ref:`sec:MovingBodiesInputs`.
