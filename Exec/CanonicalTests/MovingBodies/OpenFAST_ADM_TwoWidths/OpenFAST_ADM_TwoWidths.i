# The OpenFAST_ADM_TwoTurbines farm with a kernel width per turbine: the upstream disk T1 spread over
# 2 cells (100 m) and the downstream disk T2 over 1 cell (50 m), which also sets T2's filtered-disk
# correction (filter width sqrt(6) x 50 m, about one rotor radius). The integrated source must equal
# minus the farm's total load (load_x in moving_bodies/total_load.csv) in every row and T2's sampled
# flow log is the gold; the twin with T2.epsilon = 2 must reproduce OpenFAST_ADM_TwoTurbines' gold exactly.

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

# two IEA-15-MW-sized stub rotors, sampling the flow, their loads on the flow as disks
# The default sampling (disk_corrected) recovers the free stream; T1's filter width (epsilon = 2 cells
# of 50 m) is about 2 rotor radii, past the 1.25 the factor was fitted for (the start-up log warns),
# T2's (1 cell) about 1 radius, inside it.
erf.moving_bodies.bodies                     = T1 T2
erf.moving_bodies.T1.type                    = openfast_turbine
erf.moving_bodies.T1.mode                    = adm
erf.moving_bodies.T1.fst_file                = stub_turbine.fst
erf.moving_bodies.T1.base_pos                = 750. 600. 0.
erf.moving_bodies.T1.num_force_points_blade  = 20
erf.moving_bodies.T1.num_points_t            = 16
erf.moving_bodies.T1.epsilon                 = 2.0
erf.moving_bodies.T1.output_root             = T1
erf.moving_bodies.T2.type                    = openfast_turbine
erf.moving_bodies.T2.mode                    = adm
erf.moving_bodies.T2.fst_file                = stub_turbine.fst
erf.moving_bodies.T2.base_pos                = 1230. 600. 0.
erf.moving_bodies.T2.num_force_points_blade  = 20
erf.moving_bodies.T2.num_points_t            = 16
erf.moving_bodies.T2.epsilon                 = 1.0
erf.moving_bodies.T2.output_root             = T2
erf.moving_bodies.diagnostics_dir            = moving_bodies
