# ------------------  INPUTS TO MAIN PROGRAM  -------------------
# Dry 2D warm bubble with a static level-1 box over the lower half of the domain.
# The level is made at the first regrid (erf.box1.start_time) and then kept: the
# Bubble initialization needs boxes of the full domain height, so the level cannot
# be made at start-up.
#
# Run by the SPLIT fine_z parity tests: level 1 split in z must give the same answer
# as level 1 left whole, because ERF joins fine boxes stacked in z into whole columns
# when the implicit substep or the implicit vertical diffusion is on.  Without the
# join, level 1 split in z was off by 0.18 m/s in w and 0.019 K in theta after
# 100 steps.

erf.prob_name = "Bubble"

max_step  = 20
stop_time = 3600.0

amrex.fpe_trap_invalid = 1

fabarray.mfiter_tile_size = 1024 1024 1024

# PROBLEM SIZE & GEOMETRY
geometry.prob_extent = 20000.0 400.0  10000.0
amr.n_cell           = 100     4      50
geometry.is_periodic = 0 1 0
xlo.type = "SlipWall"
xhi.type = "SlipWall"
zlo.type = "SlipWall"
zhi.type = "SlipWall"

amr.max_grid_size_x = 1048576
amr.max_grid_size_y = 1048576
amr.max_grid_size_z = 1048576

# TIME STEP CONTROL
erf.fixed_dt = 0.5
erf.fixed_mri_dt_ratio = 4

# DIAGNOSTICS & VERBOSITY
erf.sum_interval   = 1
erf.v              = 1
amr.v              = 1

# CHECKPOINT FILES
erf.check_file      = chk
erf.check_int       = -1

# PLOTFILES
erf.plot_file_1     = plt
erf.plot_int_1      = 20
erf.plot_vars_1     = density rhotheta x_velocity y_velocity z_velocity pressure theta

# SOLVER CHOICES
erf.use_gravity          = true

erf.dycore_horiz_adv_type    = "Upwind_5th"
erf.dycore_vert_adv_type     = "Upwind_5th"
erf.dryscal_horiz_adv_type   = "Upwind_5th"
erf.dryscal_vert_adv_type    = "Upwind_5th"

# PHYSICS OPTIONS
erf.les_type        = "None"
erf.pbl_type        = "None"
erf.buoyancy_type   = 1
erf.init_type       = Isentropic

# Nonzero so that the implicit vertical diffusion has something to do
erf.molec_diff_type   = "ConstantAlpha"
erf.dynamic_viscosity = 10.0 # [kg/(m-s)]
erf.alpha_T           = 10.0 # [m^2/s]
erf.alpha_C           = 10.0

# PROBLEM PARAMETERS
prob.T_pert =    20.0
prob.x_c    = 10000.0
prob.z_c    =  2000.0
prob.x_r    =  2000.0
prob.z_r    =  2000.0
prob.T_0    =   300.0
prob.do_moist_bubble = false
prob.T_pert_is_airtemp = false

# REFINEMENT: level 1 over x = 6-14 km, z = 0-5 km, made at the regrid of step 4
amr.max_level       = 1
amr.ref_ratio       = 2
erf.regrid_int      = 4
erf.coupling_type   = "OneWay"
erf.refinement_indicators = box1
erf.box1.max_level  = 1
erf.box1.start_time = 0.1
erf.box1.in_box_lo  =  6000.   0.     0.
erf.box1.in_box_hi  = 14000. 400.  5000.
