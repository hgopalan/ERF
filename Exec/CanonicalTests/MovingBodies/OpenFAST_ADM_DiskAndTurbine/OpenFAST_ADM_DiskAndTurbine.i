# A stub OpenFAST turbine T1 (actuator disk, epsilon = 2 cells) and, two diameters downstream on its
# axis, a prescribed-Ct disk D2 spread with its own width (epsilon = 1 cell): the two kinds of body,
# each with its own kernel, on the same momentum sources. The integrated source must equal minus the
# total load of both (load_x in moving_bodies/total_load.csv) in every row, and D2's disk log is the gold.

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

# an IEA-15-MW-sized stub rotor as a disk and a uniform-Ct disk of the same size behind it, both
# sampling the flow; T1's default sampling (disk_corrected) has a filter width of about 2 rotor radii
# (epsilon = 2 cells of 50 m), past the 1.25 the factor was fitted for (the start-up log warns)
erf.moving_bodies.bodies                     = T1 D2
erf.moving_bodies.T1.type                    = openfast_turbine
erf.moving_bodies.T1.mode                    = adm
erf.moving_bodies.T1.fst_file                = stub_turbine.fst
erf.moving_bodies.T1.base_pos                = 750. 600. 0.
erf.moving_bodies.T1.num_force_points_blade  = 20
erf.moving_bodies.T1.num_points_t            = 16
erf.moving_bodies.T1.epsilon                 = 2.0
erf.moving_bodies.T1.output_root             = T1
erf.moving_bodies.D2.type                    = ct_disk
erf.moving_bodies.D2.base_pos                = 1230. 600. 0.
erf.moving_bodies.D2.rotor_radius            = 120.
erf.moving_bodies.D2.hub_height              = 150.
erf.moving_bodies.D2.ct                      = 0.75
erf.moving_bodies.D2.epsilon                 = 1.0
erf.moving_bodies.D2.air_density             = 1.0     # ERF's density here (prob.rho_0)
erf.moving_bodies.D2.output_root             = D2
erf.moving_bodies.diagnostics_dir            = moving_bodies
