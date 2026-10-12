# The IEA 15 MW reference turbine, run by the real OpenFAST library (not the stub), as an actuator
# disk on the summit of Askervein in a uniform 10 m/s inflow, on one level of 50 m cells. base_pos z
# = 0 is the terrain surface, so the base is raised by the hill (moving_bodies/ground.csv records
# it), the nodes and rings follow the fitted mesh, and the start-up audit measures the ground
# clearance from the terrain under the hub. Ten steps of a start-up from a uniform flow (the blades
# flap downwind once released: the thrust drops by 60 % and the torque by over 80 % in the first
# step, then recovers): a regression of the coupling, not a validation. The default sampling
# (disk_corrected) feeds OpenFAST the recovered free stream, so the model's AeroDyn runs its own BEM
# induction (IEA15-land-bem.fst, Wake_Mod 1). Plotfile, T1_flow.csv and T1_erf.csv against their
# golds; the integrated momentum source against minus the shaft component of the rotor thrust in
# every row.

max_step = 10
stop_time = 4.0
erf.fixed_dt = 0.4          # the blade tip (about 96 m/s) moves 38 m per step: 0.77 of a 50 m cell, 0.91 of the
                            # 42 m-tall cells over the summit; flow CFL about 0.1 (12 m/s at most)

# Askervein: the terrain file of Exec/CanonicalTests/Real_Terrain/Askervein (its summit, 125 m, at
# x = -306 m, y = 282 m), copied next to this deck by the test. Mass inflow 2.7 km upstream of the summit,
# pressure outflow 3.3 km (about 14 rotor diameters) downstream, no sponge.
geometry.prob_lo     = -3000. -1200.   0.
geometry.prob_hi     =  3000.  1800. 800.
amr.n_cell           =  120     60    16
amr.max_level        = 0
geometry.is_periodic = 0 0 0
xlo.type     = "Inflow"
xlo.velocity = 10.0 0. 0.   # uniform free stream; conserved variables extrapolated from the interior
xhi.type     = "Outflow"
ylo.type = "SlipWall"
yhi.type = "SlipWall"
zlo.type = "SlipWall"
zhi.type = "SlipWall"

erf.terrain_type      = StaticFittedMesh
erf.terrain_file_name = "askervein.txt"
erf.terrain_smoothing = 0

erf.anelastic = 1
erf.vert_implicit = true
erf.anelastic_type = MidPoint   # RK2 ignores the implicit vertical solve; MidPoint honours it
erf.use_fft   = true
erf.molec_diff_type = "None"
erf.les_type        = "None"
erf.dycore_horiz_adv_type  = "Upwind_3rd"
erf.dycore_vert_adv_type   = "Upwind_3rd"
erf.dryscal_horiz_adv_type = "Upwind_3rd"
erf.dryscal_vert_adv_type  = "Upwind_3rd"

erf.init_type = "uniform"
erf.prob_name = "ABL"
prob.rho_0 = 1.225          # the model's AirDens: the start-up audit compares them at the hub
prob.T_0   = 300.0
prob.U_0   = 10.0

erf.sum_interval = -1
erf.check_int    = -1
erf.plot_file_1  = plt
erf.plot_int_1   = 10
erf.plot_vars_1  = x_velocity y_velocity z_velocity   # density and theta stay uniform

erf.moving_bodies.bodies                     = T1
erf.moving_bodies.T1.type                    = openfast_turbine
erf.moving_bodies.T1.mode                    = adm
erf.moving_bodies.T1.fst_file                = IEA15_land_model/IEA15-land-bem.fst
erf.moving_bodies.T1.base_pos                = -306. 282. 0.   # the summit; z = 0 is the terrain surface there
erf.moving_bodies.T1.num_force_points_blade  = 20
erf.moving_bodies.T1.num_points_t            = 16
erf.moving_bodies.T1.epsilon                 = 2.0    # 2 cells of 50 m: filter width 2 rotor radii, past the 1.25 the
                                                       # disk factor was fitted for (the start-up log warns); fine here
erf.moving_bodies.T1.air_density             = 1.225
erf.moving_bodies.T1.output_root             = T1
erf.moving_bodies.diagnostics_dir            = moving_bodies
