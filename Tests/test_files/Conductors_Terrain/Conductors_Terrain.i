# Two lines over hills, dead-ended on three transformers: one on a hilltop, two on flat ground,
# generated with Exec/CanonicalTests/PowerLines/make_case.py --seed 7 (a smaller domain, two hills,
# three transformers). A neutral boundary layer, a log law under a capping inversion, comes in
# through the x-low face and leaves through the x-high one over the terrain-following mesh, with
# the k-equation RANS closure, the surface layer and the implicit anelastic MidPoint scheme. Each
# line hangs from insulator strings on two suspension towers and samples the flow at its nodes;
# the transformers take the lines' pull. The middle span of L1, the transformers' log and the
# statistics of the hilltop transformer must match their golds; nothing goes back into the flow.

max_step = 10
stop_time = 9.0
erf.fixed_dt = 0.9          # at most 26 m/s aloft on 50 m cells: Courant 0.47

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 1500. 1000. 400.
amr.n_cell           = 30    20    16
amr.max_level        = 0
geometry.is_periodic = 0 0 0
xlo.type           = "Inflow"
xlo.dirichlet_file = "inflow_profile"   # z u v w theta; no constant inflow density (it would rescale theta)
xlo.KE             = 5.2523             # 3.3 u*^2, u* = 1.2616 m/s
xhi.type           = "Outflow"
ylo.type           = "SlipWall"
yhi.type           = "SlipWall"
zhi.type           = "SlipWall"
zlo.type           = "surface_layer"
erf.most.z0        = 0.1

erf.terrain_type      = StaticFittedMesh
erf.terrain_smoothing = 2
erf.terrain_file_name = "terrain_hills.txt"
erf.wall_dist_type    = terrain_height

erf.anelastic      = 1
erf.anelastic_type = MidPoint   # RK2 ignores the implicit vertical solve; MidPoint honours it
erf.vert_implicit  = true
erf.use_fft        = true

erf.init_type           = "input_sounding"
erf.input_sounding_file = "input_sounding"   # the inflow profile; surface pressure in hPa
erf.prob_name           = "ABL"
erf.use_gravity         = true

erf.dycore_horiz_adv_type  = "Upwind_5th"
erf.dycore_vert_adv_type   = "Upwind_3rd"
erf.dryscal_horiz_adv_type = "Upwind_5th"
erf.dryscal_vert_adv_type  = "Upwind_3rd"
erf.molec_diff_type        = "None"
erf.les_type               = "None"
erf.rans_type              = "kEqn"
erf.dirichlet_k            = true
erf.init_tke_from_ustar    = true
erf.rans_consistent_diffusivities = true
erf.rans_lscale_min        = 1.0
erf.max_geom_lscale        = 20.0   # 0.1 kappa zi for the 500 m inversion
erf.theta_ref              = 300.0

erf.sum_interval = -1
erf.check_int    = -1
erf.plot_file_1  = plt
erf.plot_int_1   = 10
erf.plot_vars_1  = density x_velocity y_velocity z_velocity theta KE

erf.conductors.diagnostics_dir = conductors
erf.conductors.air_density     = 1.2
FILE = network.inputs
