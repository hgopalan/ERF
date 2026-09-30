# ------------------  INPUTS TO MAIN PROGRAM  -------------------
#
# Over embedded-boundary terrain (a flat EB surface at 45 m): anelastic Smagorinsky LES where
# explicit diffusion, not advection, limits the time step (EB terrain takes no TKE closure). A shear
# of 0.1 1/s and a large Cs (2, for the test only) give K_theta near 480 m^2/s over 20 m cells: the
# step must follow erf.diffusive_cfl (about 0.1 s) where advection allows 0.3 s, at which a
# grid-scale checkerboard in theta grows. Theta starts at 300 K +- 0.5 K (random, fixed seed) and
# must stay within those bounds, from the first step on (the eddy diffusivities of the initial
# state are computed before the first time step, ERF::init_eddy_diffs_for_dt).
#
# Tests/CTestList.cmake runs it and checks the bounds (Tests/MultiLevelPropertyCheck.cpp, mode
# bounded); erf.diffusive_cfl = 0 (the limit off) fails that check.
#
erf.prob_name = "ABL"

max_step = 30

amrex.fpe_trap_invalid  = 1
amrex.fpe_trap_zero     = 1
amrex.fpe_trap_overflow = 1

geometry.prob_lo     = 0.0 0.0 0.0
geometry.prob_hi     = 320.0 320.0 320.0
amr.n_cell           = 16 16 16
amr.max_grid_size    = 256
amr.max_level        = 0
geometry.is_periodic = 1 1 0

zlo.type = "SlipWall"   # inside the EB solid

erf.terrain_type      = "EB"
erf.terrain_file_name = "terrain.txt"
erf.eb_boundary_type  = "SlipWall"
eb2.geometry          = terrain
eb2.small_volfrac     = 1.e-4
zhi.type = "SlipWall"

erf.anelastic      = 1
erf.anelastic_type = MidPoint
erf.vert_implicit  = false
erf.cfl            = 0.5

erf.v = 1
amr.v = 0

erf.check_int   = -1
erf.plot_file_1 = plt
erf.plot_int_1  = 30
erf.plot_vars_1 = density x_velocity y_velocity z_velocity theta KE

erf.use_gravity     = true    # needed by the sounding initialization; the buoyancy is small over 3 s
erf.molec_diff_type = "None"
erf.les_type        = "Smagorinsky"
erf.Cs              = 2.0
erf.theta_ref       = 300.0

erf.init_type           = "input_sounding"
erf.input_sounding_file = "input_sounding"
erf.sounding_type       = "Ideal"
erf.fix_random_seed = 1
prob.rho_0 = 1.16
prob.T_0   = 300.0
prob.V_0   = 0.0
prob.W_0   = 0.0
prob.T_0_Pert_Mag = 0.5
prob.U_0_Pert_Mag = 0.0
prob.V_0_Pert_Mag = 0.0
prob.W_0_Pert_Mag = 0.0
