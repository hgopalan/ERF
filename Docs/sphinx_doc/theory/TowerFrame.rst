
.. _sec:TowerFrame:

Lattice tower frame model
=========================

The frame model (``Source/MovingBodies/Towers/ERF_Frame.H``) is a linear,
three-dimensional finite-element model of a lattice tower: every member is a
two-node beam with six degrees of freedom per node, the base joints are fixed
or stand on six-component springs, and the static response to nodal loads and
gravity, the natural modes and the time response are solved in double
precision (``ERF_Frame.H``, ``ERF_FrameDynamics.H``). Its element stiffness and
mass, local axes, section properties, concentrated masses and gravity loads
follow OpenFAST's SubDyn module, and it reads SubDyn input files, so that a tower
described for SubDyn has the same stiffness and natural frequencies in ERF.
The frame model is a library with unit tests; towers in an ERF run use
the one-mode tower of :ref:`sec:Conductors` (``erf.conductors.<type>.frequency``).

Conventions
-----------

This section fixes the frame, units and ordering every other section uses.

- Coordinates are those of the input file, in m, with :math:`z` up; gravity acts
  along :math:`-z`.
- Every node has six degrees of freedom in the order :math:`u_x, u_y, u_z` (m),
  then rotations :math:`\theta_x, \theta_y, \theta_z` (rad). Loads use the same order:
  forces (N), then moments (N m). Node :math:`n` (0-based) holds degrees of
  freedom :math:`6n` to :math:`6n+5`.
- Nodes are the joints in input order, then the interior nodes of members that
  are divided into more than one element (SubDyn's ``NDiv``), member by member,
  from the member's first joint towards its second.

Elements
--------

This section gives the element stiffness and its axes.

An element of length :math:`L` (m) from node :math:`a` to node :math:`b` has
local axes :math:`(x, y, z)` with :math:`z` along the element. As SubDyn's
``GetDirCos`` builds them, with :math:`(\Delta x, \Delta y, \Delta z)` the
element's extent and :math:`\Delta_{xy} = \sqrt{\Delta x^2 + \Delta y^2}`:

.. math::

   \hat x = \left(\frac{\Delta y}{\Delta_{xy}}, -\frac{\Delta x}{\Delta_{xy}}, 0\right), \quad
   \hat y = \left(\frac{\Delta x \Delta z}{L \Delta_{xy}}, \frac{\Delta y \Delta z}{L \Delta_{xy}}, -\frac{\Delta_{xy}}{L}\right), \quad
   \hat z = \frac{(\Delta x, \Delta y, \Delta z)}{L},

so a horizontal element has a horizontal :math:`\hat x`. A vertical element
pointing up has the frame's axes; one pointing down has :math:`\hat y` and
:math:`\hat z` reversed. The member's spin (SubDyn's ``MSpin``, degrees on input)
then turns :math:`\hat x` and :math:`\hat y` about :math:`\hat z`.

In its local axes the element is SubDyn's ``ElemK_Beam``: an Euler-Bernoulli
beam (``FEMMod`` 1), or a Timoshenko beam (``FEMMod`` 3) whose bending terms carry
the shear factors

.. math::

   \Phi_x = \frac{12 E I_{yy}}{G \kappa_x A L^2}, \qquad
   \Phi_y = \frac{12 E I_{xx}}{G \kappa_y A L^2},

where :math:`E` and :math:`G` are the Young's and shear moduli (Pa), :math:`A` the
area (m\ :sup:`2`), :math:`I_{xx}` and :math:`I_{yy}` the second moments about the local
:math:`x` and :math:`y` axes (m\ :sup:`4`) and :math:`\kappa_x A`, :math:`\kappa_y A` the
shear areas along local :math:`x` and :math:`y`. :math:`\Phi_x` softens bending in the
local :math:`x`-:math:`z` plane, :math:`\Phi_y` in the :math:`y`-:math:`z` plane;
both are 0 for Euler-Bernoulli beams. The axial stiffness is :math:`EA/L` and the
torsional stiffness :math:`G J_t / L`, with :math:`J_t` the torsion constant
(m\ :sup:`4`). The element's stiffness in the frame's axes is
:math:`T K_\mathrm{local} T^T`, with :math:`T` four copies of the direction cosine
matrix on its diagonal.

Sections
--------

This section gives the beam properties of each of SubDyn's three cross-section
tables, as SubDyn's ``SetElementProperties`` derives them.

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Shape
     - Properties
     - Timoshenko shear coefficient
   * - Circular (1c)
     - :math:`D, t` (m): :math:`A`, :math:`I_{xx} = I_{yy}`, :math:`J_t = J_0 = 2 I_{xx}`; :math:`t = 0` is solid
     - Steinboeck et al. (SubDyn theory manual, eq. 13) with the diameter ratio :math:`(D-2t)/D`, so 1 for a
       solid section, as SubDyn takes it
   * - Rectangular (1r)
     - :math:`S_a` along local :math:`x`, :math:`S_b` along local :math:`y`, :math:`t` (m):
       :math:`I_{xx} = S_a S_b^3/12`, :math:`I_{yy} = S_a^3 S_b/12`, less the hollow; :math:`J_t` for a solid or
       a thin-walled section
     - solid: :math:`10(1+\nu)/(12+11\nu)`; hollow: SubDyn's thin-wall formula; :math:`\nu = E/(2G) - 1`
   * - Arbitrary (4)
     - :math:`A, A_{sx}, A_{sy}, I_{xx}, I_{yy}, J_0, J_t` as given
     - :math:`\kappa_x = A_{sx}/A`, :math:`\kappa_y = A_{sy}/A`

:math:`J_0` (the polar moment) enters only the rotary inertia; the stiffness uses
:math:`J_t`. A lattice tower's angle members are best given as arbitrary sections,
with their principal axes set by the spin.

Supports and gravity
--------------------

This section describes the boundary conditions and the loads from weight.

A support (SubDyn's base reaction joint) fixes any of its joint's six degrees
of freedom (flag 1) and leaves the others free (flag 0). The free ones may stand
on a symmetric 6 x 6 spring read from the row's SSI file: the 21 entries of its
upper triangle, column by column, named ``Kxx Kxy Kyy Kxz Kyz Kzz Kxtx Kytx Kztx
Ktxtx Kxty Kyty Kzty Ktxty Ktyty Kxtz Kytz Kztz Ktxtz Ktytz Ktztz`` (N/m, N/rad,
N m/rad), each line holding a value then its name.

The weight of an element, :math:`w = \rho A g` per length (N/m), is applied as
SubDyn's ``ElemG`` applies it: :math:`-wL/2` along :math:`z` at each node and the end
moments :math:`\mp w L^2/12` of a uniformly loaded fixed-fixed beam about the
horizontal axes. With these consistent loads the nodal displacements of a
uniformly loaded beam are exact. A concentrated mass adds its weight at its joint
and the moment of that weight about the joint when its centre is offset.

Solution
--------

This section states what a static solve returns and in which frame.

The stiffness of the free degrees of freedom, springs included, is factored once
by a dense Cholesky factorisation. A degree of freedom that nothing holds stops
the factorisation; the frame is refused with a message naming that degree of
freedom and its joint (for example "the rotation about z of joint 2").
A solve returns:

- the displacement of every degree of freedom (m, rad), in the frame's axes;
- each support's reaction, the force and moment on the frame by the ground at
  its joint (N, N m), in the frame's axes, springs included;
- each element's end forces, the forces and moments on the element at its first
  node and then at its second (N, N m), in the element's local axes: the axial
  force (tension positive) is the local :math:`z` force at the second node.

The model is linear: no geometric (stress) stiffness and no large rotations,
as in SubDyn's beams.

Mass
----

This section gives the mass matrix the modes and the time response use.

An element's mass is SubDyn's ``ElemM_Beam``: the consistent mass of the cubic
beam, :math:`\rho A L` in translation, the rotary inertia of the cross-section
(:math:`\rho I_{xx}`, :math:`\rho I_{yy}`) and the torsional inertia :math:`\rho J_0 L`,
rotated to the frame's axes as the stiffness is. A concentrated mass adds
SubDyn's rigid-body matrix at its joint: the mass :math:`m` at the offset
:math:`\mathbf r = (x, y, z)` of its centre from the joint, with the inertia tensor
entries :math:`J_{xx}, J_{yy}, J_{zz}, J_{xy}, J_{xz}, J_{yz}` about its centre,

.. math::

   M_{66} = \begin{pmatrix} m I & -m [\mathbf r]_\times \\ m [\mathbf r]_\times & J + m (|\mathbf r|^2 I - \mathbf r \mathbf r^T) \end{pmatrix},

with :math:`[\mathbf r]_\times` the cross-product matrix of :math:`\mathbf r`. A support's SSI
file adds its 6 x 6 mass (entries ``Mxx`` ... ``Mtztz``, in the order of the
stiffness entries) to the support's joint.

Natural modes
-------------

This section describes how the lowest natural modes are found and checked.

``frame_modes`` finds the lowest :math:`p` solutions of
:math:`K \boldsymbol\phi = \omega^2 M \boldsymbol\phi` on the free degrees of freedom by
subspace iteration with :math:`q = \max(2p, p + 8)` trial vectors (Bathe): each
iteration solves :math:`K Y = M X` with the factored stiffness and solves the
projected :math:`q \times q` problem by Jacobi rotations, through the projected
stiffness so that a degree of freedom without mass gives an infinite frequency
instead of a singular matrix. The iteration stops when the first :math:`p + 1`
eigenvalues change by less than :math:`10^{-12}` of themselves. A Sturm sequence
count then checks that no mode was missed: the number of negative pivots of
:math:`K - \sigma M`, :math:`\sigma` between the :math:`p`-th and the next
eigenvalue, must be :math:`p`. The modes are returned with their frequencies (Hz)
and their shapes, mass-normalised (:math:`\boldsymbol\phi^T M \boldsymbol\phi = 1`) on
every degree of freedom (0 on the fixed ones).

Time response
-------------

This section gives the time integration and its properties.

``FrameDynamics`` advances :math:`M \ddot{\mathbf u} + C \dot{\mathbf u} + K \mathbf u = \mathbf f(t)`
by Newmark's average-acceleration method (:math:`\beta = 1/4`, :math:`\gamma = 1/2`)
with Rayleigh damping :math:`C = a_0 M + a_1 K`. Over a step :math:`h` (s), with the
loads at the end of the step,

.. math::

   \left(K + \tfrac{2}{h} C + \tfrac{4}{h^2} M\right) \mathbf u_{n+1} =
   \mathbf f_{n+1} + M\left(\tfrac{4}{h^2} \mathbf u_n + \tfrac{4}{h} \dot{\mathbf u}_n + \ddot{\mathbf u}_n\right)
   + C\left(\tfrac{2}{h} \mathbf u_n + \dot{\mathbf u}_n\right),

   \ddot{\mathbf u}_{n+1} = \tfrac{4}{h^2} (\mathbf u_{n+1} - \mathbf u_n) - \tfrac{4}{h} \dot{\mathbf u}_n - \ddot{\mathbf u}_n, \qquad
   \dot{\mathbf u}_{n+1} = \dot{\mathbf u}_n + \tfrac{h}{2} (\ddot{\mathbf u}_n + \ddot{\mathbf u}_{n+1}).

The effective stiffness on the left is factored once per step size. The method
is unconditionally stable and second-order accurate and adds no numerical
damping: an undamped free vibration keeps its energy
:math:`\tfrac12 \dot{\mathbf u}^T M \dot{\mathbf u} + \tfrac12 \mathbf u^T K \mathbf u` exactly, and a
mode of angular frequency :math:`\omega` advances by the phase
:math:`2 \arctan(\omega h / 2)` per step (a period lengthened by about
:math:`(\omega h)^2/12`). ``rayleigh_coefficients`` gives :math:`a_0` (1/s) and
:math:`a_1` (s) for damping ratios at two frequencies; a mode of angular frequency
:math:`\omega` then has the ratio :math:`(a_0/\omega + a_1 \omega)/2`. A run starts at rest
in static equilibrium, or from a given displacement and velocity (the
acceleration then follows from the equation of motion), and its state (the
time, displacement, velocity and acceleration) can be saved and restored.

Reading SubDyn input files
--------------------------

This section lists the part of a SubDyn input file (OpenFAST 5.0 layout) the
reader takes, and what it refuses.

=========================================== ===============================================================
SubDyn section                              Read as
=========================================== ===============================================================
FEA and Craig-Bampton parameters            ``FEMMod`` 1 or 3 (2 and 4 refused), ``NDiv`` :math:`\ge 1`
Structure joints                            id and position; ``JointType`` must be 1 (cantilever)
Base reaction joints                        six flags (1 fixed, 0 free) and an optional SSI file, relative
                                            to the SubDyn file's directory
Interface joints                            ids, kept for reference
Members                                     ``MType`` 1/1c, 1r or 4 with the same property set at both
                                            ends and ``MSpin`` (deg); cables (2), rigid links (3) and
                                            springs (5) are refused
Circular, rectangular, arbitrary properties every row
Cable, rigid link, spring, cosine matrices  skipped
Joint additional concentrated masses        mass, inertia and the centre's offset
=========================================== ===============================================================

Every check names the file, and the line or the item: an unknown joint, a
property set missing from its member type's table, a zero-length member, a
joint on no member, a non-positive modulus or section value, a support listed
twice, an SSI file on a support with no free degree of freedom.

Verification
------------

This section lists what the unit tests ``DirectionCosines``, ``BeamElement``,
``Frame`` and ``FrameSubDyn`` (``Tests/Unit/MovingBodies``) compare with.

- Beam theory, exact for these elements at the nodes, with one and five elements
  per member: a cantilever's tip deflection :math:`PL^3/(3EI)` (plus
  :math:`PL/(\kappa G A)` with shear deformation) in both bending planes, its tip
  rotation, :math:`PL/(EA)` and :math:`TL/(G J_t)`; the sag of a cantilever under its
  own weight, :math:`wL^4/(8EI)` (plus :math:`wL^2/(2\kappa G A)`); a vertical member's
  sink under its weight and a mass, :math:`\rho g L^2/(2E) + m g L/(EA)`.
- A spring base: the tip moves by the beam's bending plus the base's translation
  and tilt from the spring's own coupled solve.
- A fixed portal frame's sway, :math:`P h^2 (2a + 3b)/(12 a (a + 6b))` with
  :math:`a = E I_c/h` and :math:`b = E I_b/W` (slope-deflection).
- A triangulated truss of slender members: the pin-jointed member forces and
  reactions.
- Equilibrium: on a lattice tower with a spring base, the reactions balance
  arbitrary joint loads and gravity in force and in moment.
- SubDyn stiffness: two SubDyn input files of a 25 m lattice tower
  (``Tests/test_files/FrameSubDynTower``) read by ERF's reader give the stiffness
  condensed to the tower's peak, a 6 x 6 matrix, that SubDyn writes as ``KBBt``
  (OpenFAST 5.0.0, double precision), to the 7 digits SubDyn prints. One case
  has Euler-Bernoulli arbitrary sections; the other Timoshenko circular,
  rectangular and spun arbitrary sections, two elements per member and a base
  joint on a coupled spring.
- Mass: an element moves :math:`\rho A L` in every translation and :math:`\rho J_0 L` in a
  twist; a concentrated mass's 6 x 6 matrix gives the kinetic energy of the
  offset rigid body for any motion.
- Modes: a cantilever's bending (two modes in each plane), axial and torsional
  frequencies from beam theory; mass orthonormality and the eigenproblem's
  residual; a stiff post on a spring base with masses ringing at the base's own
  coupled frequencies; the Sturm count of modes below a shift.
- SubDyn modes: the 20 lowest natural frequencies of the two cases above and of a
  third with concentrated masses at the cross-arm tips (offset centres and
  products of inertia) and a mass on the spring base equal those SubDyn writes
  as ``Full_frequencies``, to its 7 digits; case B's cluster of eight modes within
  0.25 % of each other included.
- Time response: a cantilever released in its first mode follows
  :math:`\cos(2 n \arctan(\omega h/2))` and keeps its energy to :math:`10^{-11}`; with Rayleigh
  damping the mode follows Newmark's recursion for the damped oscillator of its
  frequency and ratio; a damped tower under a sudden load and gravity settles at
  the static solution; a saved and restored state continues bit for bit.
