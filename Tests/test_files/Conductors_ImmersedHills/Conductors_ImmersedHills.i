# A circuit over hills that are an immersed boundary on a flat mesh: the hills of Conductors_Terrain
# (ImmersedForcing, the wall law on the hill surface), a neutral log-law inflow through the x-low
# face and out through the x-high one, Deardorff LES, the implicit anelastic MidPoint scheme. Three
# phases and a shield wire hang from two shared lattice towers that bend; their attachments stand on
# the hills' surface, read from the terrain file the immersed boundary is built from, since the
# mesh is flat. The towers' loads and sway, the middle phase's middle span and the transformers'
# loads must match their golds; nothing goes back into the flow.

max_step = 10
stop_time = 4.5
erf.fixed_dt = 0.45         # at most 26 m/s aloft on 25 m cells: Courant 0.47

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 1500. 1000. 400.
amr.n_cell           = 60    40    16
amr.max_level        = 0
geometry.is_periodic = 0 0 0
xlo.type           = "Inflow"
xlo.dirichlet_file = "inflow_profile"   # z u v w theta; no constant inflow density (it would rescale theta)
xlo.KE             = 0.1
xhi.type           = "Outflow"
ylo.type           = "SlipWall"
yhi.type           = "SlipWall"
zhi.type           = "SlipWall"
zlo.type           = "surface_layer"    # the flat ground between the hills
erf.most.z0        = 0.1

erf.terrain_type             = ImmersedForcing
erf.terrain_file_name        = "terrain_hills.txt"
eb2.small_volfrac            = 0.005
erf.if_implicit_drag         = true     # point-implicit forcing in the hills: stable at the flow's step
erf.if_use_most              = true     # the wall law on the hills
erf.if_z0                    = 0.1

erf.anelastic      = 1
erf.anelastic_type = MidPoint
erf.vert_implicit  = true
erf.use_fft        = true

erf.init_type           = "input_sounding"
erf.input_sounding_file = "input_sounding"   # the inflow profile; surface pressure in hPa
erf.use_gravity         = true

erf.dycore_horiz_adv_type  = "Upwind_5th"
erf.dycore_vert_adv_type   = "Upwind_3rd"
erf.dryscal_horiz_adv_type = "Upwind_5th"
erf.dryscal_vert_adv_type  = "Upwind_3rd"
erf.molec_diff_type        = "None"
erf.les_type               = "Deardorff"
erf.rans_type              = "None"
erf.theta_ref              = 300.0

erf.sum_interval = -1
erf.check_int    = -1
erf.plot_file_1  = plt
erf.plot_int_1   = 10
erf.plot_vars_1  = density x_velocity y_velocity z_velocity theta KE

erf.conductors.diagnostics_dir = conductors
erf.conductors.air_density     = 1.2
FILE = network.inputs
