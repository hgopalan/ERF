# A 300 m conductor span (795 kcmil ACSR, 1.5 m slack) hanging 30 m above flat ground across a
# sheared anelastic crosswind v = 15 + 0.1 z m/s entering through the y-low face (steady between
# slip walls without diffusion): the wind handed to MoorDyn is ERF's velocity sampled at the line's
# nodes where they are each step, so it changes as the span swings up and down through the shear.
# The span's log (mid-span position, sag, offset, swing angle, tensions and the sampled wind) must
# match its gold; the plotfile is the flow's gold, since nothing is put back into the flow.

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
erf.plot_vars_1  = density x_velocity y_velocity z_velocity theta

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
