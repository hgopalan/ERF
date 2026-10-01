# Uniform 10 m/s anelastic flow through a prescribed uniform-Ct actuator disk (R = 120 m, hub
# 150 m, Ct = 0.75) whose force is spread onto the momentum sources with epsilon = 2 dx. The
# disk's log must show the integrated source equal to the thrust in every row (the exact
# normalisation of the spreading); the plotfile carries the wake that starts to form.

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

erf.moving_bodies.bodies          = D1
erf.moving_bodies.D1.type         = ct_disk
erf.moving_bodies.D1.base_pos     = 750. 600. 0.
erf.moving_bodies.D1.rotor_radius = 120.
erf.moving_bodies.D1.hub_height   = 150.
erf.moving_bodies.D1.ct           = 0.75
erf.moving_bodies.D1.epsilon      = 2.0
erf.moving_bodies.D1.output_root  = D1
