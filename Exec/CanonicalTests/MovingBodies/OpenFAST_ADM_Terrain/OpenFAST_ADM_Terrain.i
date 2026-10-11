# OpenFAST_ADM_Uniform, with the default sampling (disk_corrected), on a terrain-fitted mesh: a
# 100 m Witch-of-Agnesi ridge across the flow with the stub turbine on its top. base_pos z = 0 is the
# terrain surface at the base, so the base is raised by the ridge height (moving_bodies/ground.csv
# records it), the nodes and the rings follow the fitted mesh, and the start-up audit measures the
# ground clearance from the terrain under the hub.

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

# a Witch-of-Agnesi ridge across the flow, 100 m high and 400 m half-width, centred at x = 1500 m
erf.terrain_type         = StaticFittedMesh
erf.terrain_smoothing    = 0
prob.custom_terrain_type = "WoA"
prob.dir                 = 0
prob.hmax                = 100.0
prob.L                   = 400.0

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
erf.plot_vars_1  = density x_velocity y_velocity z_velocity theta z_phys

# an IEA-15-MW-sized stub rotor, sampling the flow, its loads on the flow as a disk
# The default sampling (disk_corrected) recovers the free stream; with epsilon = 2 cells of 50 m its
# filter width is about 2 rotor radii, past the 1.25 the factor was fitted for (the start-up log
# warns). That is fine for a regression of this feature; the calibrated set-up is OpenFAST_ADM_DiskCorrected.
erf.moving_bodies.bodies                     = T1
erf.moving_bodies.T1.type                    = openfast_turbine
erf.moving_bodies.T1.mode                    = adm
erf.moving_bodies.T1.fst_file                = stub_turbine.fst
erf.moving_bodies.T1.base_pos                = 1500. 600. 0.   # on the ridge top; z = 0 is the terrain surface there (100 m)
erf.moving_bodies.T1.num_force_points_blade  = 20
erf.moving_bodies.T1.num_points_t            = 16
erf.moving_bodies.T1.epsilon                 = 2.0
erf.moving_bodies.T1.output_root             = T1
erf.moving_bodies.diagnostics_dir            = moving_bodies
