# The OpenFAST_ADM_Uniform case, with the default sampling (disk_corrected), on two levels: the 50 m
# level 0 and a level 1 of 25 m cells over a box around the rotor and its near wake (x 500..1500, y
# 300..900, z 0..450 m). The turbine lives on the finest level (the default anchor): its nodes are
# sampled there, its momentum sources are spread there with epsilon = 2 fine cells, and OpenFAST is
# stepped with the fine level's step (0.25 s, 25 OpenFAST steps); level 0 sees the rotor through the
# average-down of the state. The integrated momentum source must equal minus the thrust in every
# row; the two-level plotfile and the sampled flow log are the golds.

max_step = 10
stop_time = 5.0
erf.fixed_dt = 0.5

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 3000. 1200. 600.
amr.n_cell           = 60    24    12
amr.max_level        = 1
amr.ref_ratio        = 2
amr.n_error_buf      = 0
erf.refinement_indicators = rotor
erf.rotor.in_box_lo  = 500. 300. 0.
erf.rotor.in_box_hi  = 1500. 900. 450.
erf.rotor.max_level  = 1
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

# an IEA-15-MW-sized stub rotor on level 1 (the default anchor), its loads on the flow as a disk
# The default sampling (disk_corrected) recovers the free stream; with epsilon = 2 cells of 25 m its
# filter width is about 1 rotor radius, inside the range the factor was fitted for.
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
