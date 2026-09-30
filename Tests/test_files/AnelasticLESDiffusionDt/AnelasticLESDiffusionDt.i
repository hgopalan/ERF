# ------------------  INPUTS TO MAIN PROGRAM  -------------------
#
# Anelastic Deardorff LES where explicit diffusion, not advection, limits the time step: a large
# initial TKE (Ck 0.5, slow dissipation Ce 0.01) gives an eddy diffusivity near 45 m^2/s for theta
# over 20 m cells, while the 2 m/s wind allows 5 s. The anelastic integrator treats horizontal
# diffusion explicitly, so the step must follow erf.diffusive_cfl (0.4 s here); at the advective
# step a grid-scale checkerboard in theta grows. Theta starts at 300 K +- 0.5 K (random, fixed
# seed) and must stay within those bounds, from the first step on (the eddy diffusivities of the
# initial state are computed before the first time step, ERF::init_eddy_diffs_for_dt).
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

zlo.type = "SlipWall"
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

erf.use_gravity     = false   # no buoyancy: only the eddy diffusion moves theta
erf.molec_diff_type = "None"
erf.les_type        = "Deardorff"
erf.Ck              = 0.5
erf.Ce              = 0.01
erf.theta_ref       = 300.0

erf.init_type = Uniform
erf.fix_random_seed = 1
prob.rho_0 = 1.16
prob.T_0   = 300.0
prob.U_0   = 2.0
prob.V_0   = 0.0
prob.W_0   = 0.0
prob.KE_0  = 20.0
prob.T_0_Pert_Mag = 0.5
prob.U_0_Pert_Mag = 0.0
prob.V_0_Pert_Mag = 0.0
prob.W_0_Pert_Mag = 0.0
