# Linear shear u = 5 + 0.05 z, v = 1 + 0.01 z (input_sounding) in an anelastic box with one
# stub OpenFAST turbine whose node velocities are sampled from the flow. The sampler is exact
# for a linear field, so the CSV must show u = 12.5 and v = 2.5 at the hub (z = 150 m) and as the
# blade mean (the rotor is symmetric about the hub), and w = 0, at every step. The turbine adds
# no forcing yet, so the sheared flow stays stationary.

max_step = 5
stop_time = 1.0
erf.fixed_dt = 0.05

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 2400. 1200. 600.
amr.n_cell           = 24    12    6
amr.max_level        = 0
geometry.is_periodic = 1 1 0
zlo.type = "SlipWall"
zhi.type = "SlipWall"

erf.anelastic = 1
erf.use_gravity = true
erf.use_fft   = true
erf.molec_diff_type = "None"
erf.les_type        = "None"

erf.init_type           = "input_sounding"
erf.input_sounding_file = "input_sounding"

erf.sum_interval = -1
erf.check_int    = -1
erf.plot_file_1  = plt
erf.plot_int_1   = 5
erf.plot_vars_1  = density x_velocity y_velocity z_velocity theta

# an IEA-15-MW-sized stub rotor at hub height 150 m, velocities sampled from the flow
erf.moving_bodies.bodies                     = T1
erf.moving_bodies.T1.type                    = openfast_turbine
erf.moving_bodies.T1.fst_file                = stub_turbine.fst
erf.moving_bodies.T1.base_pos                = 600. 600. 0.
erf.moving_bodies.T1.num_force_points_blade  = 20
erf.moving_bodies.T1.output_root             = T1
