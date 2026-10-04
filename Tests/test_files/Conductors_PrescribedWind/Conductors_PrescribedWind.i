# A 300 m conductor span (795 kcmil ACSR, 1.5 m slack) hanging 30 m above flat ground in a
# uniform 10 m/s anelastic flow, blown by a prescribed 15 m/s crosswind handed to MoorDyn at
# every line node (the flow itself is not sampled in this version). The span's log must match
# its gold; the plotfile is the flow's own gold, since the span puts nothing into the flow.

max_step = 10
stop_time = 5.0
erf.fixed_dt = 0.5          # 10 m/s on 50 m cells: Courant 0.1

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 1500. 1000. 300.
amr.n_cell           = 30    20    12
amr.max_level        = 0
geometry.is_periodic = 0 1 0
xlo.type     = "Inflow"
xlo.velocity = 10.0 0. 0.
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

erf.conductors.lines               = S1
erf.conductors.S1.end_a            = 300. 500. 30.
erf.conductors.S1.end_b            = 600. 500. 30.
erf.conductors.S1.length           = 301.5
erf.conductors.S1.diameter         = 0.0281
erf.conductors.S1.mass_per_length  = 1.628
erf.conductors.S1.axial_stiffness  = 3.0e7
erf.conductors.S1.output_root      = S1
erf.conductors.diagnostics_dir     = conductors
erf.conductors.air_density         = 1.0
erf.conductors.prescribed_velocity = 0. 15. 0.
