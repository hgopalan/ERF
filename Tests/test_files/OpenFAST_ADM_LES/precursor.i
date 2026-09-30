# Precursor for OpenFAST_ADM_LES: a periodic anelastic Smagorinsky LES box with a perturbed
# uniform 10 m/s flow that writes its boundary planes every step for the turbine run to read at
# its inflow. Same mesh as the turbine deck; the planes span the whole domain.

max_step  = 14
stop_time = 7.0
erf.fixed_dt = 0.5

geometry.prob_lo     = 0.    0.    0.
geometry.prob_hi     = 3000. 1200. 600.
amr.n_cell           = 60    24    12
amr.max_level        = 0
geometry.is_periodic = 1 1 0
zlo.type = "SlipWall"
zhi.type = "SlipWall"

erf.anelastic = 1
erf.use_fft   = true
erf.molec_diff_type = "None"
erf.les_type        = "Smagorinsky"
erf.Cs              = 0.17

erf.init_type = "uniform"
erf.prob_name = "ABL"
prob.rho_0 = 1.0
prob.T_0   = 300.0
prob.U_0   = 10.0
prob.U_0_Pert_Mag = 0.5
prob.V_0_Pert_Mag = 0.5
erf.fix_random_seed = 1

erf.sum_interval = -1
erf.check_int    = -1
erf.plot_int_1   = -1

erf.output_bndry_planes          = 1
erf.bndry_output_planes_interval = 1
erf.bndry_output_start_time      = 0.0
erf.bndry_output_planes_file     = "BndryFiles"
erf.bndry_output_var_names       = density velocity temperature
erf.bndry_output_box_lo          = 0.    0.
erf.bndry_output_box_hi          = 3000. 1200.
