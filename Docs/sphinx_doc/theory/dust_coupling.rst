.. role:: cpp(code)
   :language: c++

.. _sec:DustCoupling:

Dust-Atmosphere Coupling
========================

Fields taken from the atmosphere
--------------------------------

At the start of each dust step the surface layer's friction velocity,
surface temperature and boundary-layer height are copied to every dust cell
of the column, and the horizontal wind is interpolated linearly in the
vertical to the height :cpp:`erf.dust.zref` above the local surface from the
cell-centred velocities, clamped to the lowest and highest cell centres (a
``zref`` below half the first cell thickness takes the first cell's wind and
a start-up warning says so; until October 2026 that case fell through to the
second-highest cell of the domain). :cpp:`erf.dust.zref` should equal
:cpp:`erf.most.zref`. Without a coupled atmosphere (the placeholder path)
:cpp:`erf.dust.test_ustar`, ``test_surf_temp_K`` and ``test_wind_speed`` are
used instead, which is how the emission physics is tested in isolation.

Terrain
-------

Slopes on the dust grid are centred differences of the atmosphere's nodal
terrain, or of the finer raster :cpp:`erf.dust.terrain_file` when given, and
the curvature follows from them. They enter the threshold's slope factor
always. With :cpp:`erf.dust.use_terrain_wind` the wind at ``zref`` also gets
the FARSITE terrain correction shared with the fire model (ridge speed-up
:cpp:`erf.dust.k_ridge`, lee sheltering ``k_shelter``, valley channelling
``k_valley`` and deflection toward the slope ``k_deflect``; see
:ref:`sec:FireCoupling`), and the friction velocity follows by the same
factor, :math:`u_* \to u_*\, |\mathbf{U}_\mathrm{corrected}| / |\mathbf{U}|`
(:cpp:`erf.dust.terrain_ustar = scale`, the default: :math:`u_*` is linear in
the wind in a neutral log law, so a flat cell keeps the surface layer's
value). :cpp:`erf.dust.terrain_ustar = loglaw` is the form used until October
2026, which re-derived

.. math::

   u_* = \frac{\kappa\, U}{\ln(z_\mathrm{ref}/z_0)}, \qquad z_0 = \texttt{erf.dust.z0\_dust}

in every cell and so replaced the surface layer's stability-corrected
:math:`u_*` on :cpp:`erf.most.z0` by a neutral one on another roughness:
with ``zref`` 24 m, ``most.z0`` 0.1 m and ``z0_dust`` 0.01 m that is 0.70x
on flat ground, emission 0.35x, before any terrain factor. As for the fire,
the factors are empirical stand-ins for flow that a resolved simulation
already contains; the flat-terrain cases set the four factors to 1.

Injection into the atmosphere
-----------------------------

The dust rides in the passive scalar slot after the first
(``RhoScalar_comp + 1``), one slot per bin when
:cpp:`erf.dust.transport_bins_separately` is true and a single total
otherwise. The per-bin emission flux is summed, averaged down to the
atmosphere grid, and added to the slow right-hand side of the lowest cell as

.. math::

   \left.\frac{\partial \rho_\mathrm{dust}}{\partial t}\right|_{k=0}
     = f_\mathrm{atm}\, \frac{F_\mathrm{dust}}{h_0}, \qquad h_0 = J_0\, \Delta z

with :math:`h_0` the thickness of the lowest cell (:math:`J` the cell volume
factor, 1 on a flat grid; the centre-to-centre spacing above the cell was
used until October 2026, which lost 5 % of the flux at a 1.1 stretching
ratio) and :math:`f_\mathrm{atm}` = :cpp:`erf.dust.atm_feedback`; setting it to 0
keeps the surface diagnostics running without changing the atmosphere. The
flux computed in step :math:`n` is injected in step :math:`n+1`, the same
lag as the fire coupling. The scalar starts at zero even when the sounding
initialisation fills the other components.

Settling
--------

Each bin settles at the Stokes velocity with the Cunningham slip correction,

.. math::

   v_s = \frac{(\rho_p - \rho_a)\, g\, d^2}{18\, \mu_a}\, C_c, \qquad
   C_c = 1 + \frac{2\lambda}{d}\left(1.257 + 0.400\, e^{-0.55\, d/\lambda}\right)

with :math:`\lambda` = 0.066 µm, :math:`\mu_a` = 1.81e-5 Pa s, the local
air density, and :math:`d` from :cpp:`erf.dust.bin_diameters` (metres; the
one entry per bin), capped at 1 m/s. The tendency is the
first-order upwind divergence of the downward flux :math:`v_s \rho_\mathrm{dust}`
through the cell faces, with :math:`\rho_\mathrm{dust}` the dust density of the
state: cell :math:`k` gains :math:`v_s \rho_\mathrm{dust}(k+1)/h_k` from above
and loses :math:`v_s \rho_\mathrm{dust}(k)/h_k` to the cell below, with
:math:`h_k = J_k \Delta z` the cell thickness; the loss through the bottom face
of the first cell is the dry deposition below. With the bins transported as
one scalar the settling velocity is the mean of the bins' velocities weighted
by their shares of the mass emitted so far (:cpp:`erf.dust.lumped_settling =
mean`; the shares are checkpointed; ``bin0`` is the form until October 2026,
which settled the 50 µm third of the default bins at the 7 µm velocity, a
residence of 5850 s in a 23 m cell instead of 117 s). The shares are those of
the emission, not of the air: the coarse bins fall out first, so the airborne
mix grows finer than the emitted one while the mean velocity stays at the
emitted shares. A check at every step aborts when the explicit settling of
the coarsest bin is unstable at that step's :math:`\Delta t` on the thinnest
first cell, :math:`\max v_s\, \Delta t / h_0 > 1` (until October 2026 the check
ran once on the first RK stage's :math:`\Delta t / 3`, so it fired only above
3). (Before October 2026 the kernel
multiplied :math:`v_s` into the tendency instead of the density and took the
neighbour from below, so the dust did not settle; the deposition flux had the
same error.)

Dry deposition
--------------

At the lowest cell the deposition velocity of Zhang et al. (2001) combines
settling with the aerodynamic and surface resistances,

.. math::

   v_d = v_s + \frac{1}{r_a + r_s + r_a r_s v_s}, \qquad
   r_a = \frac{1}{\kappa\, u_*}, \qquad r_s = \frac{1}{E_0\, u_*}

with :math:`E_0` = :cpp:`erf.dust.deposition_E0`, never below :math:`v_s`
and capped like it at 1 m/s (a 0.1 m/s cap until October 2026 sat below the
settling velocity of 50 µm dust, 0.2 m/s, so coarse dust arrived in the lowest
cell faster than it could leave and piled up to twice the physical
concentration). The flux
:math:`v_d \rho_\mathrm{dust}(k=0)` leaves the atmosphere as a sink of the
lowest cell and accumulates on the dust grid in ``dust_deposition_rate``
[kg/m²], which is never reset and feeds the MSHA diagnostics, the PHREEQC
feedback files and the super-particle source map.

Fields returned to the surface
------------------------------

After the slow right-hand side two fields come back to every dust cell of
the column: the dust density of the lowest atmosphere cell,
``dust_conc_sfc``, which drives the loading feedback of
:ref:`sec:DustSources`, and the surface moisture flux from the microphysics
(``Q1fx3`` at the bottom face), which is zero without a moisture scheme.
The flux is carried to the dust plotfile as ``dust_surf_moist`` and is not
used by the threshold: the Fecan (1999) factor needs the gravimetric soil
moisture, which a surface flux cannot give (the former dynamic-moisture
option divided the flux by :math:`L_v \rho_a` and
compared the result, of order :math:`10^{-11}`, with a 0.3 % moisture, so it
never did anything and was removed in September 2026). The static moisture
raster is all that
acts.

Turbulent diffusion in the MRF scheme
-------------------------------------

The MRF scheme sets the vertical scalar diffusivity equal to the heat
diffusivity, so the dust is mixed through the boundary layer by the same
:math:`K_h(z) = w_* \kappa h\, (z/h)(1 - z/h)^2` profile as heat, with no
countergradient term because dust has no prescribed surface flux gradient.
:cpp:`erf.dust_mrf_Sc_t` (note the ``erf`` prefix) scales it by
:math:`Pr_t / Sc_t`; 0 or a negative value keeps the heat diffusivity.
:cpp:`erf.transport_scalar` must be true for the scalar to be advected and
diffused at all.
