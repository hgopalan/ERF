# The stub rotor as an actuator disk in a Smagorinsky LES with turbulent inflow: the x
# boundaries are inflow (read from the precursor's boundary planes) and outflow, the anelastic
# FFT solver handles the non-periodic direction. Ten steps; the plotfile, the turbine's
# running statistics and the integrated-source identity are the regression.

max_step  = 10
stop_time = 5.0
erf.fixed_dt = 0.5

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 3000. 1200. 600.
amr.n_cell           = 60    24    12
amr.max_level        = 0
geometry.is_periodic = 0 1 0
xlo.type = "Inflow"
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

erf.input_bndry_planes    = 1
erf.bndry_file            = "BndryFiles"
erf.bndry_input_var_names = density velocity temperature

erf.sum_interval = -1
erf.check_int    = -1
erf.plot_file_1  = plt
erf.plot_int_1   = 10
erf.plot_vars_1  = density x_velocity y_velocity z_velocity theta

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
