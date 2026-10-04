# The lines over hills of Conductors_Terrain (two sections dead-ended on three transformers, in a
# neutral boundary layer over terrain with the k-equation RANS closure), with lattice towers that
# bend: each sways in its first mode, at 2 Hz on a rigid foundation and lower on its footing,
# which can tilt (foundation_rotational_stiffness), under its members' drag and its line's pull,
# and MoorDyn moves the cross-arms the lines hang from as coupled points. The towers' loads and
# cross-arm displacements, the statistics of L1's first tower (tower_L1_t1) and the middle span of
# L1 must match their golds; nothing goes back into the flow.

max_step = 10
stop_time = 9.0
erf.fixed_dt = 0.9          # s: at most 26 m/s aloft on 50 m horizontal cells, Courant 0.47

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 1500. 1000. 400.
amr.n_cell           = 30    20    16
amr.max_level        = 0
geometry.is_periodic = 0 0 0
xlo.type           = "Inflow"
xlo.dirichlet_file = "inflow_profile"   # z u v w theta; no constant inflow density (it would rescale theta),
                                        # so the inflow k is zero-gradient
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
erf.max_geom_lscale        = 20.0   # m: 0.1 kappa zi with zi = 500 m, an inversion base above this domain
erf.theta_ref              = 300.0

erf.sum_interval = -1
erf.check_int    = -1
erf.plot_file_1  = plt
erf.plot_int_1   = 10
erf.plot_vars_1  = density x_velocity y_velocity z_velocity theta KE

erf.conductors.diagnostics_dir = conductors
erf.conductors.air_density     = 1.2
FILE = network.inputs

# the towers bend (the lattice type's other keys are in network.inputs)
erf.conductors.lattice.frequency                       = 2.0
erf.conductors.lattice.foundation_rotational_stiffness = 1.0e9
