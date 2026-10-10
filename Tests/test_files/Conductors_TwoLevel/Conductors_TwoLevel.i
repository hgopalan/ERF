# The span of Conductors_DragOnFlow on a refined level: a 300 m conductor span (795 kcmil ACSR, 1.5 m
# slack) 30 m above flat ground in a sheared compressible crosswind v = 15 + 0.1 z m/s (periodic in
# x and y, slip walls, no diffusion), under a static level-1 box (refinement 2, the level's steps
# subcycled) that holds the span and its drag's reach but stops 100 m below the domain's top. The
# lines step on the finest level (anchor_level -1), take its wind and put their drag into its
# flow (drag_on_flow); the samplers search each column only up to the fine box's top. In every row
# of conductors/total_load.dat the integrated source must equal the force the lines put into the
# air; the span's log and the two-level plotfile, which carries the source and its wake, match
# their golds. A restart parity carries the lines and their source across a checkpoint, and two
# abort tests refuse drag_on_flow on a coarser anchor and an anchor level that does not exist.

max_step = 10
stop_time = 5.0
erf.fixed_dt = 0.25         # level 0; level 1 takes two steps of 0.125 s

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 1500. 1000. 300.
amr.n_cell           = 30    20    12
amr.max_grid_size    = 64
amr.blocking_factor  = 2
geometry.is_periodic = 1 1 0
zlo.type = "SlipWall"
zhi.type = "SlipWall"

amr.max_level        = 1
amr.ref_ratio_vect   = 2 2 2
erf.dt_ref_ratio     = 2
erf.coupling_type    = "TwoWay"
erf.refinement_indicators = box1
erf.box1.max_level = 1
erf.box1.in_box_lo_indices_crse =  8  6 0    # x 400 to 1100 m, y 300 to 700 m, z 0 to 200 m
erf.box1.in_box_hi_indices_crse = 21 13 7

erf.molec_diff_type = "None"
erf.les_type        = "None"

erf.init_type = "input_sounding"
erf.input_sounding_file = "input_sounding"   # the crosswind; surface pressure in hPa
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
erf.conductors.epsilon             = 2.0      # the drag reaches 3 epsilon = 6 fine cells, 150 m
