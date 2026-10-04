# A circuit across a sheared anelastic crosswind v = 15 + 0.1 z m/s entering through the y-low face
# (steady between slip walls without diffusion): three phases of 795 kcmil (thousand circular
# mils) ACSR (aluminium conductor, steel reinforced), 6 m apart, each a line of three 300 m spans
# hanging from 2.5 m insulator strings at two suspension towers and dead-ended at both ends, and a
# steel shield wire clamped at the towers 7 m above the phases' attachment points. Each line samples
# the flow at its nodes where they are each step. The middle phase's middle span and strings, the
# shield wire's middle span and the closest-approach statistics of the pairs P1-P2 and P2-SW must
# match their golds; the flow is Conductors_FlowWind's, since nothing goes back into it.

max_step = 10
stop_time = 5.0
erf.fixed_dt = 0.5          # at most 45 m/s on 50 m cells: Courant 0.45

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 1500. 1000. 300.
amr.n_cell           = 30    20    12
amr.max_level        = 0
geometry.is_periodic = 1 0 0
ylo.type     = "Inflow"
ylo.dirichlet_file = "inflow_profile"   # z u v w; no constant inflow density, which would also need ylo.theta
yhi.type     = "Outflow"
zlo.type = "SlipWall"
zhi.type = "SlipWall"

erf.anelastic = 1
erf.vert_implicit = true
erf.anelastic_type = MidPoint   # the anelastic MidPoint scheme, as in the decks over hills
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

erf.conductors.lines               = P1 P2 P3 SW
erf.conductors.diagnostics_dir     = conductors
erf.conductors.air_density         = 1.0

# the phases: dead-ended at x = 300 and 1200, suspension towers at x = 600 and 900
erf.conductors.P1.end_a            = 300.  494. 30.
erf.conductors.P1.towers           = 600.  494. 30.   900.  494. 30.
erf.conductors.P1.end_b            = 1200. 494. 30.
erf.conductors.P2.end_a            = 300.  500. 30.
erf.conductors.P2.towers           = 600.  500. 30.   900.  500. 30.
erf.conductors.P2.end_b            = 1200. 500. 30.
erf.conductors.P3.end_a            = 300.  506. 30.
erf.conductors.P3.towers           = 600.  506. 30.   900.  506. 30.
erf.conductors.P3.end_b            = 1200. 506. 30.
erf.conductors.P1.length           = 301.5 301.5 301.5
erf.conductors.P1.diameter         = 0.0281
erf.conductors.P1.mass_per_length  = 1.628
erf.conductors.P1.axial_stiffness  = 3.0e7
erf.conductors.P1.insulator_length = 2.5
erf.conductors.P1.insulator_mass   = 60.
erf.conductors.P1.output_root      = P1
erf.conductors.P2.length           = 301.5 301.5 301.5
erf.conductors.P2.diameter         = 0.0281
erf.conductors.P2.mass_per_length  = 1.628
erf.conductors.P2.axial_stiffness  = 3.0e7
erf.conductors.P2.insulator_length = 2.5
erf.conductors.P2.insulator_mass   = 60.
erf.conductors.P2.output_root      = P2
erf.conductors.P3.length           = 301.5 301.5 301.5
erf.conductors.P3.diameter         = 0.0281
erf.conductors.P3.mass_per_length  = 1.628
erf.conductors.P3.axial_stiffness  = 3.0e7
erf.conductors.P3.insulator_length = 2.5
erf.conductors.P3.insulator_mass   = 60.
erf.conductors.P3.output_root      = P3

# the shield wire: 3/8 in. extra-high-strength steel, clamped to the tower peaks
erf.conductors.SW.end_a            = 300.  500. 37.
erf.conductors.SW.towers           = 600.  500. 37.   900.  500. 37.
erf.conductors.SW.end_b            = 1200. 500. 37.
erf.conductors.SW.length           = 301. 301. 301.
erf.conductors.SW.diameter         = 0.0095
erf.conductors.SW.mass_per_length  = 0.406
erf.conductors.SW.axial_stiffness  = 9.7e6
erf.conductors.SW.output_root      = SW
