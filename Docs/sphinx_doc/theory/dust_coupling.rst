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
cell-centred velocities. :cpp:`erf.dust.zref` should equal
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
:ref:`sec:FireCoupling`), and the friction velocity is then recomputed from
the corrected wind by the log law

.. math::

   u_* = \frac{\kappa\, U}{\ln(z_\mathrm{ref}/z_0)}, \qquad z_0 = \texttt{erf.dust.z0\_dust}

which replaces the surface layer's value. As for the fire, these are
empirical stand-ins for flow that a resolved simulation already contains;
the flat-terrain cases set the four factors to 1.

Injection into the atmosphere
-----------------------------

The dust rides in the passive scalar slot after the first
(``RhoScalar_comp + 1``), one slot per bin when
:cpp:`erf.dust.transport_bins_separately` is true and a single total
otherwise. The per-bin emission flux is summed, averaged down to the
atmosphere grid, and added to the slow right-hand side of the lowest cell as

.. math::

   \left.\frac{\partial \rho_\mathrm{dust}}{\partial t}\right|_{k=0}
     = f_\mathrm{atm}\, \frac{F_\mathrm{dust}}{\Delta z_0}

with :math:`f_\mathrm{atm}` = :cpp:`erf.dust.atm_feedback`; setting it to 0
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
last entry repeats for extra bins), capped at 1 m/s. The tendency is the
first-order upwind divergence of the downward flux :math:`v_s \rho_\mathrm{dust}`
through the cell faces. With a single transported scalar the diameter of bin
0 is used.

Dry deposition
--------------

At the lowest cell the deposition velocity of Zhang et al. (2001) combines
settling with the aerodynamic and surface resistances,

.. math::

   v_d = v_s + \frac{1}{r_a + r_s + r_a r_s v_s}, \qquad
   r_a = \frac{1}{\kappa\, u_*}, \qquad r_s = \frac{1}{E_0\, u_*}

with :math:`E_0` = :cpp:`erf.dust.deposition_E0`. The flux
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
When :cpp:`erf.dust.use_dynamic_moisture` is set and a moisture scheme is
active, the flux gives a gravimetric water content
:math:`w = q_\mathrm{flux} / (L_v \rho_a)` and the Fecan (1999) factor

.. math::

   f_\mathrm{moist} = \sqrt{1 + 1.21 \max(w - 0.003, 0)}

multiplies the threshold; otherwise the static moisture raster is all that
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
