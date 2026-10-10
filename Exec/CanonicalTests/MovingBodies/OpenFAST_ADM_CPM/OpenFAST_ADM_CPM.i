# The stub rotor as an actuator disk in a Smagorinsky LES whose turbulent inflow is made by the
# cell perturbation method: a steady 10 m/s inflow profile at x = 0 with perturbation boxes just
# inside it, outflow at x = 3000 m, the anelastic FFT solve on the non-periodic direction. Ten
# steps; the plotfile, the turbine's running statistics and the integrated-source identity are
# the regression.

max_step  = 10
stop_time = 5.0
erf.fixed_dt = 0.5

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 3000. 1200. 600.
amr.n_cell           = 60    24    12
amr.max_level        = 0
geometry.is_periodic = 0 1 0
xlo.type           = "Inflow"
xlo.dirichlet_file = "inflow_profile.txt"
xlo.density        = 1.0
xlo.theta          = 300.0
xhi.type = "Outflow"
zlo.type = "SlipWall"
zhi.type = "SlipWall"

erf.anelastic = 1
erf.use_fft   = true
erf.molec_diff_type = "None"
erf.les_type        = "Smagorinsky"
erf.Cs              = 0.17

erf.init_type = "uniform"
erf.prob_name = "ABL"
prob.rho_0 = 1.0
prob.T_0   = 300.0
prob.U_0   = 10.0

# cell perturbation method at the inflow: boxes of 4 x 4 x 3 cells, two layers from two cells in
erf.perturbation_type      = "CPM"
erf.perturbation_direction = 1 0 0 0 0 0
erf.perturbation_layers    = 2
erf.perturbation_offset    = 2
erf.perturbation_box_dims  = 4 4 3
erf.perturbation_Ug        = 10.0
erf.perturbation_klo       = 1
erf.perturbation_khi       = 10
erf.fix_random_seed        = 1

erf.sum_interval = -1
erf.check_int    = -1
erf.plot_file_1  = plt
erf.plot_int_1   = 10
# theta is left out: the perturbations draw from the C library's rand(), whose sequence differs between
# Linux and macOS even with erf.fix_random_seed, so theta cannot match one gold on both
erf.plot_vars_1  = density x_velocity y_velocity z_velocity

# The default sampling (disk_corrected) recovers the free stream; with epsilon = 2 cells of 50 m its
# filter width is about 2 rotor radii, past the 1.25 the factor was fitted for (the start-up log
# warns). That is fine for a regression of this feature; the calibrated set-up is OpenFAST_ADM_DiskCorrected.
erf.moving_bodies.bodies                     = T1
erf.moving_bodies.T1.type                    = openfast_turbine
erf.moving_bodies.T1.mode                    = adm
erf.moving_bodies.T1.fst_file                = stub_turbine.fst
erf.moving_bodies.T1.base_pos                = 750. 600. 0.
erf.moving_bodies.T1.num_force_points_blade  = 20
erf.moving_bodies.T1.num_points_t            = 16
erf.moving_bodies.T1.epsilon                 = 2.0
erf.moving_bodies.T1.output_root             = T1
erf.moving_bodies.diagnostics_dir            = moving_bodies
erf.moving_bodies.avg_start                  = 2.0
