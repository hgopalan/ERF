# Canonical RANS results by phase

Numbers from the check scripts, recorded as each phase of `PLAN.md` lands.
"Before" is the code as branched from `upstream/development` (c001e148d).

## Neutral_ABL_Flat

12 h physics run, 2 ranks, `check_neutral.py --physics plt08640`.

| check | target | tol | before (phase 1) | after phase 2 |
| --- | --- | --- | --- | --- |
| all fields finite; KE, Kmv, diss >= 0 | yes | exact | pass | pass |
| max abs(walldist - z_cc) [m] | 0 | 1e-8 | 0 | 0 |
| Lturb(k=0,1,2) vs capped kappa (z + z0) | equal | 0.1 % | pass (2.93, 7.30, 10.46 m) | pass |
| max Lturb [m] | <= 30 | 1e-9 | 26.9 | 25.4 |
| diss vs AL01 Eq. 19, interior cells | equal | 5 % | 2.2 % | 2.1 % |
| u* [m/s] | 0.35 | +/- 0.15 | 0.393 | 0.393 |
| KE(k=0) / u*^2 | 3.23 | 5 % | **2.60 (fail, 20 % low)** | 3.232 |
| wall cell k_start / k_end | 1 | 1 % | **1.244 (fail)** | 1.000 |
| abs(U)(k=0..3) vs log law | equal | 10 % | -0.3 %, +1.7 %, +4.2 %, +6.7 % | -0.3 %, +1.5 %, +4.1 %, +6.5 % |
| Kmv(k=1) / (rho kappa u* (z + z0)) | 1 | +/- 0.3 | 0.76 | 0.77 |

Smoke run (40 steps): every structural check passes on 1 and 2 ranks, and
the two rank counts agree to 2e-15 in every planar-averaged field.

Reading of the two phase-1 failures: the Dirichlet wall value of k was
written into the first cell once per step, then the one-sided diffusive
flux against a zero ghost cell drained a fifth of it before the step ended,
so the converged wall k sat 20 % below the AL01 value.

Phase 2 keeps the logical BC for RhoKE at foextrap (ghost cell equals the
first cell, zero diffusive flux through the wall face via the surface-layer
branch), re-imposes the first-cell value after every RK stage, and pins the
bottom row of the implicit vertical diffusion solve. Both failures clear;
the interior profile is unchanged to the digits shown. A 40-step run with a
checkpoint at step 20 and a restart from it matches the straight run to
1e-14 in every field. The anelastic integrator disables the implicit
vertical solve, so the pinned row was exercised with a compressible variant
of the deck (dt 0.5 s, 20 acoustic substeps, 2 h): see the note below.

Compressible check of the implicit path (2 h, dt 0.5 s, 20 acoustic
substeps, 2 ranks): `vert_implicit_fac 1 1 0` with `tke = 1` in the banner,
wall cell k_start/k_end = 1.000 and KE(0)/u*^2 = 3.232 with the implicit
KE solve on and off; the two runs differ by 1e-9 in KE, so the pinned row
is exercised and holds the wall value.

## Phase 3: robustness

Neutral_ABL_Flat after phase 3 (12 h, 2 ranks): every physics check passes
with the same numbers as phase 2 to the digits in the table above. The
closure refactor onto `ERF_RANSClosure.H` was bit-identical to phase 2
before the unstable-length bound went in; with the bound, KE differs by
2e-6 and Lturb by 4e-4 m at most, in cells where dtheta/dz noise makes N^2
slightly negative.

| run | dt [s] | integrator | KE diffusion | dissipation | outcome |
| --- | --- | --- | --- | --- | --- |
| deck | 5 | anelastic | explicit (forced by anelastic) | explicit | all checks pass |
| deck | 5 | anelastic | explicit | implicit | all checks pass, same numbers |
| 4x dt | 20 | anelastic | explicit | explicit | all checks pass |
| 4x dt | 20 | anelastic | explicit | implicit | **abort at 10.6 h, negative theta at 290 m** |
| large dt | 60 | compressible, 2400 substeps | implicit | explicit | all checks pass |
| large dt | 60 | compressible, 2400 substeps | implicit | implicit | all checks pass |

Reading: the anelastic integrator switches every vertical diffusion to
explicit, and dz^2 / (2 K/rho) is about 19 s mid-layer for this deck, so
dt = 20 s sits on the explicit diffusion limit. The explicit-dissipation
run survived it by a small margin (dissipation damps k and so K); the
implicit-dissipation run kept slightly more k and crossed it. Neither
result is about the dissipation itself: the compressible pair at dt = 60 s,
where vertical diffusion is implicit, passes both ways. In this deck the
dissipation time scale is never the binding one because the wall cell is
held by the Dirichlet condition and the second cell's scale is about a
minute. The option stays opt-in and verified equivalent; the practical
limit for RANS under the anelastic integrator is the explicit vertical
diffusion, which is a dycore matter outside this plan.

Unit tests (`erf_unit_tests --gtest_filter=RANSClosure*`, 7 tests): the
first version caught the Burchard & Petersen smoothing returning -2 for
Rt = -1e16 and +1.4e14 for Rt = -1e30 through cancellation; the
rearranged form `Rt_crit + a x / (x + a)` matches the original to
round-off up to |Rt| of 1e8 and holds Rt_min beyond.

Input validation: `erf.Rt_min = -4` now aborts with the pole message;
`erf.tke_floor = 1e-4` runs.

## Stable_ABL_Flat (phase 4)

9 h GABLS1 run, 2 ranks, `check_stable.py --physics plt16200`.

| check | target | measured |
| --- | --- | --- |
| u* [m/s] | 0.20 to 0.35 | 0.244 |
| theta(k=0) minus imposed surface theta [K] | 0 to 1.5 | 0.25 |
| min dtheta/dz below 200 m [K/m] | >= 0 | 0.011 |
| max wind over Ug (low-level jet) | >= 1.02 | 1.23 |
| height of the wind maximum [m] | 50 to 300 | 154 |
| BL depth from KE [m] | 80 to 300 | 134 |
| KE(k=0)/u*^2 | 3.23 within 5 % | 3.232 |
| Lturb over neutral length, 120 to 300 m | <= 1 | 0.12 |

GABLS1 LES ensemble for reference: u* 0.26 to 0.30, jet near 150 to 200 m,
depth 150 to 200 m.

## Convective_ABL_Flat (phase 4)

4 h run at dt = 2 s, 2 ranks, `check_convective.py --physics plt07200`.

| check | target | measured |
| --- | --- | --- |
| u* [m/s] | 0.30 to 0.80 | 0.485 |
| column heat gain over rho_sfc F t | 1 within 10 % | 0.9998 |
| inversion height [m] | 900 to 1250 | 1020 |
| theta spread in 0.2 to 0.7 zi [K] | <= 2 (local closure) | 1.12 |
| max dtheta/dz in 0.2 to 0.7 zi [K/m] | <= 0 | -0.0012 |
| mixed-layer warming over F t / zi | 1 within 30 % | 1.075 |
| KE(k=0)/u*^2 | >= 3.23 (buoyancy adds) | 3.82 |
| min KE in 0.1 to 0.8 zi [m2/s2] | >= 0.05 | 0.55 |
| max KE above 1.3 zi [m2/s2] | <= 0.05 | 2e-16 |
| max Lturb over bound | <= 1 | 0.79 |

At dt = 5 s the run aborts after 1.7 h with a negative theta at 290 m:
Kmv reaches 42 kg/m/s, i.e. K/rho at the explicit limit dz^2/(2 dt) = 40
m2/s of the anelastic integrator. The profile at 4 h keeps a
superadiabatic lapse of -1 to -3 K/km through the mixed layer (K_h 40 to
67 kg/m/s, Lturb 24 to 45 m under the PBL-height cap), the expected
behaviour of a local-K closure without countergradient transport.

The deck now runs at dt = 5 s with `erf.vert_implicit = true` (phase 9);
the 4 h numbers of that run are in the phase 9 section below.

## Neutral_Hill_2D (phase 5)

6 h run at dt = 1.5 s, 2 ranks, `check_hill.py --physics plt14400`.

| check | target | measured |
| --- | --- | --- |
| walldist vs exact ridge distance, max relative | <= 10 % | 5.2 % |
| walldist vs exact ridge distance, mean relative | <= 3 % | 1.0 % |
| walldist abs error within 100 m of the surface [m] | <= 3 | 2.36 |
| crest speed-up, k = 0, 1, 2 | 0.2 to 0.8 (2 h/L = 0.4) | 0.55, 0.45, 0.39 |
| speed-up positive in the lowest ten cells | > 0 | min 1.43 m/s |
| upstream u* from the wall k [m/s] | 0.25 to 0.55 | 0.346 |
| upstream wind vs log law, k = 0, 1, 2 | within 15 % | 1.5 %, 1.7 %, 3.7 % |
| max Lturb over bound | <= 1 | 0.76 |
| wall-cell k retention along the surface | within 1 % | 3e-16 |

Flat-fitted variant (prob.hmax = 1e-6, 40 steps): before the gradient fix
the Poisson distance was z (1 - dz/2H), 0.78 % short at every height;
after it, exact to 1e-6 m above the first cell and 1.5 cm long in the
first cell. The ridge mean error went from 1.5 % to 1.0 %; the speed-ups
did not change to three digits because the length is capped at 30 m.

## Neutral_Hill_3D (phase 6)

4 h run at dt = 1.5 s, 64 x 64 x 20 at 40 m, 2 ranks, `check_hill3d.py --physics plt09600`.

| check | target | measured |
| --- | --- | --- |
| walldist vs exact hill distance, max relative | <= 15 % | 7.4 % (terrain_height), 9.5 % (poisson) |
| walldist vs exact hill distance, mean relative | <= 3 % | 0.013 % (terrain_height), 0.31 % (poisson) |
| walldist abs error within 100 m of the surface [cells] | <= 0.2 | 0.03 (terrain_height), 0.12 (poisson) |
| crest speed-up, k = 0, 1, 2 | 0.16 to 0.64 (1.6 h/L = 0.32) | 0.41, 0.28, 0.22 |
| speed-up positive in the lowest eight cells | > 0 | min 0.71 m/s |
| upstream u* from the wall k [m/s] | 0.25 to 0.55 | 0.318 |
| upstream wind vs log law, k = 0, 1, 2 | within 15 % | -5.7 %, +4.0 %, +10.4 % |
| wall-cell k retention over the surface | within 1 % | 3e-16 |

Wall-distance methods on the 2D ridge (40-step run): terrain_height mean
0.02 %, max 2.3 %, 0.01 cells near the surface; poisson mean 1.0 %, max
5.2 %, 0.15 cells. On the flat fitted mesh terrain_height is exact to
2e-10, poisson 0.2 % in the first cell and 1e-6 above it.

Restart (2D and 3D terrain decks, checkpoint at 20, compare at 40):
every field identical to 1e-14.

Askervein (20 steps, 4 ranks, 26 s): walldist 7.9 to 707 m, KE up to
6.9 m2/s2, Kmv up to 9.4 kg/m/s, all finite.

Mesh finding (not a RANS matter, see PLAN phase 6): the 1.788e139
pre-projection divergence first seen at dz != dx was a BoxArray split in
z (unfilled momenta ghost faces in the initial projection and duplicated
planar surface-layer arrays), fixed in erf-model/ERF#3970; the
`terrain_height` wall distance had the same class of read and now
gathers the surface nodes onto every box. The Poisson wall-distance
multigrid does diverge at dx = 2 dz, with or without the split.


## Phase 9: implicit vertical diffusion of scalars under anelastic

All runs on 2 ranks with the final phase 9 code, which also changed the
explicit answers: the TKE buoyancy term is now the closure's cell-centred
`-K_h dtheta/dz` instead of the flux at the lower face (the explicit 12 h
neutral profile moved by up to 0.08 m/s and 0.03 K at the boundary-layer
top, the stable one by 0.08 m/s and 0.02 K, the convective one by 0.13
m/s and 0.18 K; every physics check still passes and the hill numbers
are unchanged in the digits quoted above).

Implicit scalars (`erf.vert_implicit = true`) against the explicit run,
max |difference| of the planar averages over the column:

| case | step | u [m/s] | theta [K] | KE [m2/s2] | Kmv [kg/m/s] |
| --- | --- | --- | --- | --- | --- |
| Neutral 12 h, dt 5 s implicit vs dt 5 s explicit | 5 | 2.9e-4 | 1.4e-4 | 6.3e-6 | 1.4e-3 |
| Neutral 12 h, dt 10 s implicit vs dt 5 s explicit | 10 | 1.3e-3 | 7.5e-4 | 9.9e-6 | 1.6e-3 |
| Stable 9 h, dt 2 s implicit vs explicit | 2 | 3.8e-5 | 3.9e-5 | 2.0e-6 | 1.5e-5 |
| Convective 4 h, dt 5 s implicit vs dt 2 s explicit | 5 / 2 | 3.2e-3 | 5.1e-3 | 2.7e-4 | 7.1e-3 |

Scales: u 12, 10 and 10 m/s; KE 0.5, 0.2 and 1.2 m2/s2; Kmv 7.5, 0.75
and 42 kg/m/s. Before the two fixes the same comparisons gave 0.34 m/s
(neutral), 0.47 m/s (stable) and 1.0 m/s with 1.3 K and a factor 3 in
KE (convective).

Key numbers, explicit / implicit:

| check | neutral dt 5 | neutral dt 10 (implicit) | stable dt 2 | convective |
| --- | --- | --- | --- | --- |
| u* [m/s] | 0.39312 / 0.39312 | 0.39313 | 0.24369 / 0.24370 | 0.4849 / 0.4848 |
| KE(k=0)/u*^2 | 3.2323 / 3.2323 | 3.2322 | | 3.816 / 3.817 |
| jet max U/Ug | | | 1.2286 / 1.2286 | |
| BL depth from KE [m] | | | 134 / 134 | |
| column heat gain over rho_sfc F t | | | | 0.9998 / 0.9995 |
| inversion height [m] | | | | 1020 / 1020 |
| theta spread 0.2 to 0.7 zi [K] | | | | 1.117 / 1.117 |

Stage decomposition of the convective one-step heat budget (ratio of the
column heat gain to rho_sfc F dt, dt = 5 s): before the fix
`erf.vert_implicit_fac` = `1 0 0` 0.975, `0 1 0` 1.449, `0 0.5 0` 1.212,
`1 1 0` 1.450; after the fix 0.975 for all four, 0.983 after ten steps
(explicit at dt = 2 s: 0.994).

Restart consistency test that exposed the buoyancy-term defect (neutral
deck restarted at 6 h, one step of 0.5 s, implicit minus explicit, max
over the column): KE 1.2e-11 with `erf.sigma_k = 1e9` (no KE diffusion)
against 1e-6 with it, growing linearly in time and the same at dt = 0.5,
1 and 2 s, i.e. an operator difference, not a time-stepping one; after
the fix 8e-9 in one step and 5e-7 after 10 s at any of the three steps.

Neutral deck at dt = 20 s with implicit scalars: NaN at 10.8 h (now
caught by `check_for_negative_theta`), from the explicit momentum
diffusion, whose limit is 10.4 s once K_m reaches 7.5 m2/s. Phase 10.

Gold tests that run Deardorff (`ABL_MOST`, `Deardorff_stationary`) and
the implicit-diffusion MYNN tests (`ABL_MOST_IMP_DIFF*`) pass; `ctest -L
rans` is 10 for 10 (new entry `RANS_Neutral_ABL_Flat_Implicit`, the
neutral deck at dt = 10 s with the solve on); 7 gtests pass.

## Phase 10: implicit vertical diffusion of momentum under anelastic

`erf.vert_implicit = true` under the anelastic integrator now covers u, v,
theta, k and moisture; w stays explicit. All runs 2 ranks.

Neutral deck, 12 h, max |difference| of the planar averages from the
anelastic explicit dt = 5 s run (scales: u 11.8, v 2.93 m/s, theta 309 K,
k 0.515 m2/s2):

| run | dt [s] | u [m/s] | v [m/s] | theta [K] | k [m2/s2] |
| --- | --- | --- | --- | --- | --- |
| anelastic implicit | 5 | 3.1e-4 | 4.8e-4 | 2.7e-4 | 8.9e-6 |
| anelastic implicit | 10 | 1.3e-3 | 1.2e-3 | 7.9e-4 | 8.4e-6 |
| anelastic implicit | 20 | 1.7e-3 | 1.8e-3 | 8.8e-4 | 1.2e-5 |
| anelastic implicit | 30 | 1.7e-3 | 1.8e-3 | 9.8e-4 | 1.4e-5 |
| anelastic implicit | 60 | 2.9e-3 | 3.0e-3 | 1.7e-3 | 2.6e-5 |
| compressible implicit | 5 | 1.1e-3 | 1.1e-3 | 6.4e-4 | 7.8e-6 |
| compressible implicit | 20 | 1.2e-3 | 1.3e-3 | 6.9e-4 | 1.5e-5 |
| compressible implicit | 60 | 1.8e-3 | 1.8e-3 | 1.0e-3 | 4.3e-5 |

Every one of these passes all 22 physics checks. u* runs 0.39312 at dt = 5 s
to 0.39322 at dt = 60 s and KE(0)/u*^2 stays 3.2322 to 3.2325. The figure
`plot_dt_overlay.py` produces from these runs shows the nine profiles on
top of each other with the differences below.

Other decks:

| deck | step reached | vs explicit reference | note |
| --- | --- | --- | --- |
| Convective, 4 h | dt 20 s (explicit was 2 s) | 1.8e-2 m/s, 2.4e-2 K | heat budget 0.998; ships at dt 5 s (3.5e-3 m/s) |
| Stable, 9 h | dt 8 s | 7.6e-4 m/s | u* 0.24379, jet 1.2286, depth 134 m |
| Neutral_Hill_2D, 6 h | dt 1.5 s (unchanged) | 1.2e-4 m/s on the 3D field | advection-limited, see below |

The 2D hill deck gains no time step: its CFL-1 step is 3.3 s against the
shipped 1.5 s, so advection binds long before vertical diffusion, and the
deck fails at dt = 3 s at the same step 82 with the solve on and off. What
the deck does show is that the terrain-fitted momentum solve reproduces
the explicit answer (1.2e-4 m/s in u on a scale of 10.4, 1.5e-5 m/s in w).

Restart at dt = 20 s with the momentum solve on: `fcompare` reports zero
absolute and relative difference on all fourteen fields.

Sensitivity to the box decomposition (one rank, one box against four x-y
boxes, 40 steps, dt 5 s, max |difference| of planar averages in u):

| configuration | difference in u | in theta |
| --- | --- | --- |
| explicit | 1.8e-15 | 5.7e-14 |
| implicit, k solve only | 1.8e-15 | 5.7e-14 |
| implicit, momentum solve only | 5.3e-15 | 5.7e-14 |
| implicit, theta solve only, `vert_implicit_fac = 1 0 0` | 1.8e-15 | 5.7e-14 |
| implicit, theta solve only, `1 1 0` | 8.0e-7 | 1.2e-5 |
| implicit as shipped, dt 5 s | 9.1e-7 | 1.1e-5 |
| implicit as shipped, dt 20 s | 1.3e-5 | 4.4e-5 |
| implicit as shipped, dt 60 s | 3.6e-4 | 1.8e-4 |

5.7e-14 is one ulp of rho theta, so the first four rows are as clean as
floating point allows. The cause was chased at length and not found; the
list of mechanisms ruled out by direct test, and what the evidence does
support, is in PLAN.md phase 10. In short: it reproduces on a single rank
so it is not MPI reduction order, it is not the MOST plane average, not
uninitialised memory, not the buoyancy term, not the Dirichlet wall value
of k, and not the k equation, since Deardorff behaves the same. It needs
both the implicit solve and the anelastic projection, and it scales with
the size of the implicit increment. At dt 60 s the spread is 3e-5 in
relative terms against an implicit-explicit difference of 2.5e-4, so it
moves no physics check, but this configuration will not reproduce bitwise
across decompositions.

### How large a step the solve allows

The explicit limit dz^2 / (2K) is gone for u, v, theta and k, so what is
left is the advective Courant number and the vertical diffusion of w,
which stays explicit. Pushing each deck until it fails:

| deck | shipped | highest passing | first failure | what binds |
| --- | --- | --- | --- | --- |
| Neutral_ABL_Flat | 5 s | 240 s (all checks) | 480 s at step 53 | nothing physical: the deck is horizontally uniform, so advection does no work |
| Convective_ABL_Flat | 5 s | 20 s (all checks) | 80 s still runs, only the dissipation-lag diagnostic fails (0.12 against 0.05) | as above |
| Neutral_Hill_2D | 1.5 s | 2 s | 3 s at step 82, identically with the solve on and off | advective Courant number (CFL-1 step 3.3 s) |

The flat numbers are not transferable: those decks are 8 x 8 in the
horizontal with a horizontally uniform state, so `u du/dx` is identically
zero and the printed CFL-1 step of 32 s means nothing for them. The hill
deck is the honest one: with the solve on, the step is set by the
advective Courant number that ERF prints each step ("Anelastic dt at
level 0 would be"), and a working choice is half to nine tenths of it.

### The gain is not specific to the k-eqn closure

The solve lives in the dycore, so every closure that sets a vertical
eddy diffusivity gets it. On the same neutral geometry, 1 h, anelastic,
running each closure explicit and implicit over a step sweep:

| closure | explicit | implicit |
| --- | --- | --- |
| Smagorinsky (`erf.Cs = 0.16`) | dt 5 s runs, 20 s fails at step 4 | 5, 20, 60 and 120 s all run |
| Deardorff | dt 5 s fails at step 89 | 5, 20, 60 and 120 s all run |
| kEqn (Axell & Liungman) | dt 10 s runs, 20 s fails | 5 to 240 s run |

Agreement: Smagorinsky implicit against explicit at dt = 5 s differs by
2.7e-2 m/s in wind (scale 10) after 1 h of spin-up, and implicit dt = 120 s
against implicit dt = 5 s by 5.3e-2 m/s. Deardorff implicit dt = 120 s
against dt = 5 s differs by 1.7e-2 m/s, 3.0e-3 K and 2.1e-3 m2/s2 in k.

### Anelastic against compressible, after the change

Both integrators carry the same slow time step, because the acoustic
waves in the compressible path are handled by substepping and never
limited the slow step; what limited the anelastic path was the vertical
diffusion being forced explicit. Neutral deck, 12 h, 2 ranks, wall time
for the whole run:

| integrator | dt [s] | acoustic substeps per step | wall time | outcome |
| --- | --- | --- | --- | --- |
| anelastic implicit | 5 | none | 25 s | all checks pass |
| anelastic implicit | 60 | none | 2 s | all checks pass |
| anelastic implicit | 240 | none | 1 s | all checks pass |
| anelastic implicit | 480 | none | - | fails at step 53 |
| compressible implicit | 60 | 2400 | 445 s | all checks pass |
| compressible implicit | 120 | 4800 | 433 s | all checks pass |
| compressible implicit | 240 | 9600 | 433 s | runs, 7 checks fail |

So the anelastic path now reaches a larger passing step than the
compressible one (240 s against 120 s) and costs about two orders of
magnitude less per unit simulated time, since it pays one FFT solve per
step instead of thousands of acoustic substeps. Before phases 9 and 10
the ordering was inverted: the compressible path had had the implicit
column solve for a long time, while `vert_implicit_fac` was zeroed for
anelastic, so the cheaper integrator was the one stuck near 10 s.

### Largest step per closure and integrator (`Timestep_Limits`)

A RANS-like neutral column (4 x 4 x 200, dx = 800 m, dz = 5 m), spun up for
1 h at dt = 5 s with the implicit solve and restarted over a ladder of steps
for 200 steps each; the step is the largest rung below the first failure.
`Timestep_Limits/sweep_dt.py`, 1 rank, first failing rung in brackets:

| closure | explicit anelastic | implicit anelastic | implicit compressible | dz^2 / (2 K/rho) |
| --- | --- | --- | --- | --- |
| kEqn | 2 s (4) | 256 s (512) | 64 s (128) | 2.13 s |
| Deardorff | 0.25 s (0.5) | 512 s (1024) | 64 s (128) | 0.339 s |
| MRF | 0.5 s (1) | 256 s (512) | 8 s (16) | 0.582 s |

The explicit anelastic step is 0.94, 0.74 and 0.86 of dz^2 / (2 K/rho) in
the restart state, so for all three closures it is the diffusion limit, and
the implicit solve raises it by two to three orders of magnitude under
anelastic. Under compressible, kEqn and Deardorff stop at 64 s and MRF at
8 s. The MRF run at 16 s keeps going with |u| in the thousands of m/s,
whether the substeps are ERF's own or pinned. Not investigated.
