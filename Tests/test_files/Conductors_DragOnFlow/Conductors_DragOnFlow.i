# The coupled case of Conductors_FlowWind with the lines' drag put back into the flow
# (erf.conductors.drag_on_flow): a 300 m conductor span across a sheared anelastic crosswind
# v = 15 + 0.1 z m/s; each step the air's drag on every node, reversed, is spread with a Gaussian of
# two cells into the momentum sources of the next step. In every row of conductors/total_load.dat
# the integrated source must equal the force the lines put into the air; the span's log and the
# plotfile, which now carries the source (conductor_fx, _fy, _fz) and its wake, match their golds.

max_step = 10
stop_time = 5.0
erf.fixed_dt = 0.5          # at most 45 m/s on 50 m cells: Courant 0.45

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 1500. 1000. 300.
amr.n_cell           = 30    20    12
amr.max_level        = 0
geometry.is_periodic = 1 0 0
ylo.type     = "Inflow"
ylo.dirichlet_file = "inflow_profile"   # z u v w; no constant inflow density (it would rescale theta)
yhi.type     = "Outflow"
zlo.type = "SlipWall"
zhi.type = "SlipWall"

erf.anelastic = 1
erf.vert_implicit = true
erf.anelastic_type = MidPoint   # RK2 ignores the implicit vertical solve; MidPoint honours it
erf.use_fft   = true
erf.molec_diff_type = "None"
erf.les_type        = "None"

erf.init_type = "input_sounding"
erf.input_sounding_file = "input_sounding"   # the same profile as the inflow; surface pressure in hPa
erf.prob_name = "ABL"
erf.use_gravity = true                       # the sounding initialisation needs it; neutral at 300 K

erf.sum_interval = -1
erf.check_int    = -1
erf.plot_file_1  = plt
erf.plot_int_1   = 10
erf.plot_vars_1  = density x_velocity y_velocity z_velocity theta conductor_fx conductor_fy conductor_fz

erf.conductors.lines               = S1
erf.conductors.S1.end_a            = 600. 500. 30.
erf.conductors.S1.end_b            = 900. 500. 30.
erf.conductors.S1.length           = 301.5
erf.conductors.S1.diameter         = 0.0281
erf.conductors.S1.mass_per_length  = 1.628
erf.conductors.S1.axial_stiffness  = 3.0e7
erf.conductors.S1.output_root      = S1
erf.conductors.diagnostics_dir     = conductors
erf.conductors.air_density         = 1.0
erf.conductors.drag_on_flow        = true
erf.conductors.epsilon             = 2.0
erf.conductors.node_output_int     = 5
