# The OpenFAST_ALM_Tower case with a kernel width for each kind of point: the blade points narrow
# (epsilon = 1 cell, 50 m), the tower force nodes wider (epsilon_tower = 2 cells) and the nacelle drag
# point in between (epsilon_nacelle = 1.5 cells, which also sets the nacelle's kernel correction). The
# integrated momentum source must still equal minus the total load (thrust + tower + nacelle, load_x
# in T1_erf.csv) in every row, the plotfile and the sampled flow log are the gold, and the twin with
# the deck's widths all set to 2 cells must reproduce OpenFAST_ALM_Tower's gold exactly.

max_step = 10
stop_time = 5.0
erf.fixed_dt = 0.5

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 3000. 1200. 600.
amr.n_cell           = 60    24    12
amr.max_level        = 0
geometry.is_periodic = 0 1 0
xlo.type     = "Inflow"
xlo.velocity = 10.0 0. 0.   # uniform freestream; conserved variables extrapolated from the interior
xhi.type     = "Outflow"
zlo.type = "SlipWall"
zhi.type = "SlipWall"

erf.anelastic = 1
erf.vert_implicit = true
erf.anelastic_type = MidPoint   # RK2 ignores the implicit vertical solve; MidPoint honours it
erf.use_fft   = true
erf.molec_diff_type = "None"
erf.les_type        = "None"

erf.init_type = "uniform"
erf.prob_name = "ABL"
prob.rho_0 = 1.0
prob.T_0   = 300.0
prob.U_0   = 10.0

erf.sum_interval = -1
erf.check_int    = -1
erf.plot_file_1  = plt
erf.plot_int_1   = 10
erf.plot_vars_1  = density x_velocity y_velocity z_velocity theta

# an IEA-15-MW-sized stub rotor with its tower and nacelle, sampling the flow, each kind of point with its own kernel
erf.moving_bodies.bodies                     = T1
erf.moving_bodies.T1.type                    = openfast_turbine
erf.moving_bodies.T1.mode                    = alm
erf.moving_bodies.T1.fst_file                = stub_turbine.fst
erf.moving_bodies.T1.base_pos                = 750. 600. 0.
erf.moving_bodies.T1.num_force_points_blade  = 20
erf.moving_bodies.T1.num_force_points_tower  = 10
erf.moving_bodies.T1.nacelle_cd              = 1.0
erf.moving_bodies.T1.nacelle_area            = 60.0
erf.moving_bodies.T1.epsilon                 = 1.0
erf.moving_bodies.T1.epsilon_tower           = 2.0
erf.moving_bodies.T1.epsilon_nacelle         = 1.5
erf.moving_bodies.T1.output_root             = T1
erf.moving_bodies.diagnostics_dir            = moving_bodies
