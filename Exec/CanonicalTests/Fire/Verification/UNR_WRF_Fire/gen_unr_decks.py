#!/usr/bin/env python3
"""Write the ERF decks that reproduce the UNR WRF-Fire case studies 1-3
(https://unr-wrf-fire.readthedocs.io, Case Studies), each as a compressible run at
WRF's time step and an anelastic run at the same step, with the files they read:
the input soundings, the ignition lines and the ridge terrain of case 2. All output
goes next to this script (it is committed).

    python3 gen_unr_decks.py

Exact where ERF can be: grid spacing, fire-cell size, WRF's eta levels converted to
heights, sounding (with the 305 K header line), surface (WRF isfflx = 0 through
bulk_coeff and an imposed heat flux), diffusion, advection orders, fuel, moisture,
ignition, wind height and its log law below the first level (z0 = 0.1 m, WRF's land
roughness ZZLND), level-set scheme and viscosity, feedback scaled to WRF's heat,
case 3's start-up bubble. Not exact: the domain is enlarged to the nearest multiple
of the fire refinement (50 -> 52 cells, 125 -> 128), and ERF picks its own number of
acoustic substeps.
"""
import math, os

HERE = os.path.dirname(os.path.abspath(__file__))
G, CP, RD, P0 = 9.81, 1004.5, 287.0, 1.0e5

def theta(z):   # WRF sounding: 300 K to 1 km, +1 K per 100 m to 2.1 km
    return 300.0 if z <= 1000.0 else 300.0 + (z - 1000.0) / 100.0

def p_of_z(z, ps=1.0e5, n=4000):
    pi = (ps / P0) ** (RD / CP); dz = z / n; zz = 0.0
    for _ in range(n):
        pi -= G / (CP * theta(zz + 0.5 * dz)) * dz; zz += dz
    return P0 * pi ** (CP / RD)

def z_of_p(p, ztop):
    lo, hi = 0.0, ztop
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if p_of_z(mid, n=400) > p: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

def eta_levels(kde, hyp, zs):
    if hyp:   # module_initialize_fire.F, stretch_hyp
        return [-math.tanh(zs * ((k - 1) / (kde - 1) - 1.0)) / math.tanh(zs) for k in range(1, kde + 1)]
    return [(math.exp(-(k - 1) / (kde - 1) / zs) - math.exp(-1.0 / zs)) / (1.0 - math.exp(-1.0 / zs))
            for k in range(1, kde + 1)]   # exponential (stretch_grd default .true., z_grd_scale default 0.40)

def z_faces(kde, ztop, hyp, zs):
    ps, pt = 1.0e5, p_of_z(ztop)
    f = [0.0] + [z_of_p(pt + e * (ps - pt), ztop) for e in eta_levels(kde, hyp, zs)[1:-1]] + [ztop]
    return f

def sounding(path, u):
    rows = [(1.0, 300), (6.0, 300), (9.1, 300), (18.3, 300), (18.35, 300), (91.2, 300)] + \
           [(float(z), 300) for z in range(100, 1001, 100)] + [(float(z), theta(z)) for z in range(1100, 2101, 100)]
    with open(path, 'w') as o:
        o.write('1000.0  305.0  0.0\n')            # the WRF header line as published (surface 305 K)
        for z, t in rows: o.write(f'{z:8.2f}  {t:6.1f}  0.0  {u:4.1f}  0.0\n')

def ridge_file(path, L, nx, h=200.0, xs=2000.0, xe=3000.0):   # fire_mountain_type = 3, namelist height
    x = [L * i / (nx - 1) for i in range(nx)]
    zt = lambda xx: h * 0.5 * (1 + math.cos(math.pi + 2 * math.pi * max(0.0, min((xx - xs) / (xe - xs), 1.0))))
    with open(path, 'w') as o:
        o.write(f'{nx}\n{nx}\n' + ''.join(f'{v:.3f}\n' for v in x) + ''.join(f'{v:.3f}\n' for v in x))
        # ERF's ASCII terrain reader takes the heights with y varying fastest
        for i in range(nx):
            for j in range(nx): o.write(f'{zt(x[i]):.3f}\n')

HEAT = 1.0 / 1.08   # WRF sensible heat: burnt mass x (1 - M/(1+M)) x cmbcnst; the custom fuel already carries cmbcnst

TITLE = {1: 'flat ground, uniform 5 m/s wind, fire not coupled to the atmosphere',
         2: 'a 200 m north-south ridge, uniform 5 m/s wind, fire coupled to the atmosphere',
         3: 'LES of a convective boundary layer, fire coupled, ignited after 30 min of spin-up'}

def deck(case, solver, wind=None):
    if case in (1, 2):
        n, dx, ratio, dt, kde, hyp, zs, ztop, tstop = 52, 100.0, 4, 0.25, 101, False, 0.40, 2000.0, 3600.0
        line_x, half_w, t_ign, wind = 1050.0, 100.0, 10.0, 5.0
    else:
        n, dx, ratio, dt, kde, hyp, zs, ztop, tstop = 128, 40.0, 8, 0.5, 52, True, 1.09, 2000.0, 7200.0
        line_x, half_w, t_ign = 1020.0, 40.0, 1800.0
    L = n * dx
    variant = '' if case < 3 else ('_nowind' if wind == 0 else '_wind2p5')
    name = f'case{case}{variant}_{solver}'
    snd = 'input_sounding_case12' if case < 3 else f'input_sounding_case3{variant}'
    line = 'line_case12.csv' if case < 3 else 'line_case3.csv'
    sounding(os.path.join(HERE, snd), wind)
    open(os.path.join(HERE, line), 'w').write(f'# ignition line x = {line_x:.0f} m, y = 2000 to 3000 m\n{line_x} 2000.0\n{line_x} 3000.0\n')
    zf = z_faces(kde, ztop, hyp, zs)
    mg = 28 if case < 3 else 32           # 52 cells -> boxes of 28 + 24, each a multiple of the fire ratio 4
    lines = [f'# UNR WRF-Fire case study {case} in ERF ({solver}): {TITLE[case]}',
             f'# wind {wind:g} m/s; written by gen_unr_decks.py; run from this folder (README.md)',
             f'# first layer {zf[1]:.1f} m, top layer {zf[-1]-zf[-2]:.1f} m (WRF eta levels: {"hyperbolic 1.09" if hyp else "exponential 0.40"})',
             'erf.prob_name = "ABL"',
             'geometry.prob_lo = 0.0 0.0 0.0', f'geometry.prob_hi = {L:.1f} {L:.1f} {ztop:.1f}',
             f'amr.n_cell = {n} {n} {kde-1}', 'geometry.is_periodic = 1 1 0', 'amr.max_level = 0',
             f'amr.blocking_factor = {4 if case < 3 else 1}', f'amr.max_grid_size = {mg} {mg} {kde-1}',
             'erf.terrain_z_levels = ' + ' '.join(f'{z:.3f}' for z in zf)]
    if case == 2:
        ridge_file(os.path.join(HERE, 'terrain_ridge.txt'), L, 2 * n + 1)
        lines += ['erf.terrain_type = StaticFittedMesh', 'erf.terrain_file_name = "terrain_ridge.txt"',
                  'erf.fire.terrain_file_name = "terrain_ridge.txt"']
    cd, flux = (0.0, 0.0) if case < 3 else (0.005, 0.10)
    if case < 3:
        lines += ['erf.rayleigh_damp_W = true            # WRF damp_opt = 3 (default; not set in UNR cases 1/2): w damped everywhere',
                  'erf.rayleigh_zdamp = 5000.0           # WRF zdamp default 5000 m > ztop 2000 m',
                  'erf.rayleigh_dampcoef = 0.2           # WRF dampcoef default']
    lines += ['zlo.type = "surface_layer"', 'erf.surface_layer.flux_type = "bulk_coeff"   # WRF isfflx = 0',
              f'erf.most.Cd = {cd}                     # WRF tke_drag_coefficient', 'erf.most.Cq = 0.0',
              'erf.most.surf_temp = 305.0                # = the sounding header (ERF requires a match); the heat flux is imposed', 'erf.most.surf_moist = 0.0',
              f'erf.most.surf_temp_flux = {flux}         # WRF tke_heat_flux [K m/s]', 'erf.most.z0 = 0.1',
              'zhi.type = "SlipWall"                 # WRF rigid lid, w = 0 at ztop'
              + ('; no damping (WRF damp_opt = 0)' if case == 3 else '')] + (['erf.molec_diff_type = "None"'] if case == 3 else [])
    if case < 3:
        lines += ['erf.les_type = "None"                 # WRF diff_opt = 0, km_opt = 0',
                  'erf.molec_diff_type = "Constant"      # zero coefficients: no diffusion, but a surface_layer',
                  'erf.dynamic_viscosity = 0.0          # boundary needs a diffusion closure to run',
                  'erf.alpha_T = 0.0', 'erf.alpha_C = 0.0']
    else:
        lines += ['erf.les_type = "Deardorff"            # WRF diff_opt = 2, km_opt = 2', 'erf.Ck = 0.10', 'prob.KE_0 = 0.1',
                  'prob.bubble_dT = 0.5                  # WRF delt_perturbation: one theta bubble, no other start-up perturbation',
                  'prob.bubble_x_c = 2460.0   # WRF centres it on mass point i_c = (ide-ids)/2 = 62: (62 - 1/2) x 40 m',
                  'prob.bubble_y_c = 2460.0', 'prob.bubble_z_c = 40.0   # WRF hght_perturbation',
                  'prob.bubble_x_r = 2500.0', 'prob.bubble_y_r = 2500.0', 'prob.bubble_z_r = 40.0   # WRF x/y/zrad_perturbation']
    lines += ['erf.use_gravity = true', 'erf.use_coriolis = false', 'erf.init_type = "input_sounding"',
              f'erf.input_sounding_file = "{snd}"', 'erf.sounding_type = Ideal',
              f'erf.fixed_dt = {dt}                   # WRF time_step']
    if solver == 'anelastic':
        lines += ['erf.anelastic = 1', 'erf.use_fft = 1']
    else:
        lines += ['erf.substepping_cfl = 0.5            # ERF picks the acoustic substeps (WRF: time_step_sound)']
    lines += ['erf.dycore_horiz_adv_type  = "Upwind_5th"', 'erf.dycore_vert_adv_type   = "Upwind_3rd"',
              'erf.dryscal_horiz_adv_type = "Upwind_5th"', 'erf.dryscal_vert_adv_type  = "Upwind_3rd"',
              'max_step = 10000000', f'stop_time = {tstop}', 'erf.v = 0', 'amr.v = 0', 'erf.sum_interval = 400',
              'erf.check_int = -1', f'erf.plot_file_1 = plt_{name}_', f'erf.plot_per_1 = {600.0 if case == 3 else 120.0}',
              'erf.plot_vars_1 = density x_velocity y_velocity z_velocity theta' + (' rhoKE' if case == 3 else ''),
              'erf.fire.enable = true', f'erf.fire.grid_ratio = {ratio}               # WRF sr_x = sr_y',
              'erf.fire.propagation_method = "levelset"', 'erf.fire.directional_ros = true',
              'erf.fire.directional_wind_coupling = "advective"   # WRF fire_advection = 1 (default): phi_w(|U|) x cos to the normal',
              'erf.fire.fuel_model_id = 1000           # WRF fuel category 1 exactly (namelist.fire), with its heat content',
              'erf.fire.custom_fuel.codes = 1000', 'erf.fire.custom_fuel.1000.name = "wrf_fm1"',
              'erf.fire.custom_fuel.1000.w_1h_kg_m2 = 0.166', 'erf.fire.custom_fuel.1000.w_10h_kg_m2 = 0.0', 'erf.fire.custom_fuel.1000.w_100h_kg_m2 = 0.0',
              'erf.fire.custom_fuel.1000.w_live_herb_kg_m2 = 0.0', 'erf.fire.custom_fuel.1000.w_live_woody_kg_m2 = 0.0',
              'erf.fire.custom_fuel.1000.sav_1h_1_m = 11483.0   # savr 3500 /ft', 'erf.fire.custom_fuel.1000.depth_m = 0.305',
              'erf.fire.custom_fuel.1000.moisture_ext = 0.12', 'erf.fire.custom_fuel.1000.heat_content_J_kg = 17.433e6   # cmbcnst; Rothermel sees 7496 Btu/lb as WRF',
              'erf.fire.custom_fuel.1000.density_kg_m3 = 512.0', 'erf.fire.custom_fuel.1000.burnout_time_s = 7.0   # weight(1)',
              'erf.fire.burnout_model = "sfire"', 'erf.fire.burnout_time_to_efold = 0.85   # WRF fuel_time = weight/0.85',
              'erf.fire.heat_flux_alfg = 50.0          # WRF fire_ext_grnd default', 'erf.fire.levelset.reinit_every = 1   # WRF fire_lsm_reinit every step',
              'erf.fire.moisture_1hr = 0.08            # WRF fuelmc_g',
              'erf.fire.moisture_10hr = 0.08', 'erf.fire.moisture_100hr = 0.08', 'erf.fire.moisture_dynamic = false',
              f'erf.fire.ignition_x = {line_x}', 'erf.fire.ignition_y = 2500.0', 'erf.fire.ignition_r = 0.0',
              f'erf.fire.ignition.polygon_file = "{line}"', 'erf.fire.ignition.polygon_type = "polyline"',
              f'erf.fire.ignition.polyline_width = {half_w}   # WRF fire_ignition_radius1',
              f'erf.fire.ignition.polygon_time = {t_ign}      # WRF fire_ignition_start_time1',
              'erf.fire.wind_ref_ht = 6.5                # WRF fire_wind_height',
              'erf.fire.wind_below_first_cell = "log"     # WRF: log law from z0 below the first level',
              'erf.fire.wind_sample_z0 = 0.1              # WRF land roughness ZZLND',
              'erf.fire.use_waf = false                  # WRF master reads windrf but never applies it (module_fr_fire_phys.F)',
              'erf.fire.use_wind_limit = false', 'erf.fire.use_terrain_wind = false', 'erf.fire.ros_model = "rothermel"',
              'erf.fire.reaction_velocity_formula = "rothermel"   # WRF: A = 1/(4.774 s^0.1 - 7.27) (Rothermel 1972)',
              'erf.fire.wrf_bmst_compat = true                   # WRF: load net of its moisture share, (1 - M/(1+M)) fgi',
              f'erf.fire.levelset.gradient = "{"weno5z" if case < 3 else "weno5z_front"}"   # WRF fire_upwinding {7 if case < 3 else 9}',
              f'erf.fire.levelset.eps_visc = {0.4*dx/ratio:.3f}   # WRF fire_viscosity 0.4 is in fire cells (adds visc |R| dx lap phi); ERF eps is metres', f'erf.fire.levelset.eps_visc_front = {0.4*dx/ratio:.3f}   # WRF fire_viscosity_bg default 0.4 cells',
              'erf.fire.coupling_type = "lagged"', 'erf.fire.inject_latent = true',
              (f'erf.fire.fire_atm_feedback = {HEAT:.4f}   # coupled; WRF: burnt mass x (1 - M/(1+M)) x cmbcnst' if case > 1
               else 'erf.fire.fire_atm_feedback = 0.0     # uncoupled (WRF fire_atm_feedback = 0)'),
              'erf.fire.write_fire_stats_csv = true', f'erf.fire.fire_stats_csv_file = "fire_stats_{name}.csv"',
              f'erf.fire.plot_file = "plt_fire_{name}_"', 'erf.fire.plot_per = 60.0']
    open(os.path.join(HERE, f'inputs_{name}'), 'w').write('\n'.join(lines) + '\n')
    return name, zf

if __name__ == '__main__':
    for solver in ('compressible', 'anelastic'):
        for c in (1, 2):
            nm, zf = deck(c, solver); print(nm, f'first dz {zf[1]:.1f}  top dz {zf[-1]-zf[-2]:.1f}  n {len(zf)-1}')
        for w in (0.0, 2.5):
            nm, zf = deck(3, solver, w); print(nm, f'first dz {zf[1]:.1f}  top dz {zf[-1]-zf[-2]:.1f}  n {len(zf)-1}')
    print('feedback scale', round(HEAT, 4))
