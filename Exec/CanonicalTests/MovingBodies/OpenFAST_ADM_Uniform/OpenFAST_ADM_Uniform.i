# Uniform 10 m/s anelastic flow through one stub OpenFAST turbine represented as an actuator
# disk (mode = adm): the rotor's loads, computed by the stub from the velocities sampled at its
# nodes, are spread onto the momentum sources as rings about the hub axis and slow the flow.
# The integrated momentum source must equal minus the turbine's thrust in every row; the
# plotfile carries the wake that starts to form.

max_step = 10
stop_time = 5.0
erf.fixed_dt = 0.5

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 3000. 1200. 600.
amr.n_cell           = 60    24    12
amr.max_level        = 0
geometry.is_periodic = 1 1 0
zlo.type = "SlipWall"
zhi.type = "SlipWall"

erf.anelastic = 1
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

# an IEA-15-MW-sized stub rotor, sampling the flow, its loads on the flow as a disk
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
