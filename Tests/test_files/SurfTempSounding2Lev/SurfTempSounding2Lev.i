# ------------------  INPUTS TO MAIN PROGRAM  -------------------
#
# Two levels (20 m over 10 m, the fine level over the whole width up to 100 m) initialised from an
# input sounding whose surface line is the surface temperature (303 K, as erf.most.surf_temp
# requires) over 300 K air from 1 m up. Every level must see the sounding at its own heights: the
# fine cell at 5 m starts at 300 K (it used to blend in the surface line, 301.5 K, and with the
# fixed anelastic density the average down then left the coarse cell 0.75 K cold).
#
# Tests/CTestList.cmake runs it for the initial state only and checks the lowest cells of both
# levels (Tests/MultiLevelPropertyCheck.cpp, mode sounding).
#
erf.prob_name = "ABL"

max_step = 0

amrex.fpe_trap_invalid  = 1
amrex.fpe_trap_zero     = 1
amrex.fpe_trap_overflow = 1

geometry.prob_lo     = 0.0 0.0 0.0
geometry.prob_hi     = 160.0 160.0 200.0
amr.n_cell           = 8 8 10
amr.max_grid_size    = 256
amr.blocking_factor  = 2
geometry.is_periodic = 1 1 0

zlo.type           = "surface_layer"
erf.most.z0        = 0.1
erf.most.surf_temp = 303.0
zhi.type           = "SlipWall"

erf.anelastic      = 1
erf.anelastic_type = MidPoint
erf.cfl            = 0.5

erf.init_type           = "input_sounding"
erf.input_sounding_file = "input_sounding"
erf.sounding_type       = "Ideal"

erf.v = 1
amr.v = 0

erf.check_int   = -1
erf.plot_file_1 = plt
erf.plot_int_1  = 1
erf.plot_vars_1 = density x_velocity y_velocity z_velocity theta

erf.use_gravity     = true
erf.molec_diff_type = "None"
erf.les_type        = "Smagorinsky"
erf.Cs              = 0.1

amr.max_level     = 1
amr.ref_ratio     = 2 2 2
erf.coupling_type = "TwoWay"
erf.refinement_indicators = box1
erf.box1.max_level = 1
erf.box1.in_box_lo_indices_crse = 0 0 0
erf.box1.in_box_hi_indices_crse = 7 7 4
