# Canonical RANS: largest time step per closure and integrator

How far the implicit column solve moves the time step, for three closures
that set a vertical eddy diffusivity: the one-equation k model
(`erf.rans_type = kEqn`), Deardorff (`erf.les_type = Deardorff`) and the
MRF PBL scheme (`erf.pbl_type = MRF`), each under three integrators:

| integrator | inputs |
| --- | --- |
| explicit anelastic | `erf.anelastic = 1`, `erf.vert_implicit = false` |
| implicit anelastic | `erf.anelastic = 1`, `erf.vert_implicit = true`, `erf.anelastic_type = MidPoint` (RK2 anelastic turns the solve off) |
| implicit compressible | `erf.anelastic = 0`, `erf.vert_implicit = true`, acoustic substeps |

The grid is coarse in the horizontal and fine in the vertical, as in a RANS
run, so the vertical diffusion limit dz^2 / (2 K/rho) is what an explicit
run meets first.

| item | value |
| --- | --- |
| domain | 3200 x 3200 x 1000 m, doubly periodic |
| grid | 4 x 4 x 200, dx = 800 m, dz = 5 m, first cell centre 2.5 m |
| flow | neutral Ekman layer: G = 10 m/s, f = 1e-4 1/s, z0 = 0.1 m, the `Neutral_ABL_Flat` sounding (300 K to 700 m, 3 K per 100 m above) |
| kEqn | AL01 defaults, `erf.dirichlet_k = true`, `erf.init_tke_from_ustar = true` |
| Deardorff | defaults, `erf.init_tke_from_ustar = true` |
| MRF | defaults, `erf.pbl_mrf_coriolis_freq = 1e-4` |

## Method

`sweep_dt.py` does, for each closure:

1. **Spin-up.** 1 h (720 steps) at dt = 5 s with the implicit solve, once
   anelastic and once compressible, checkpointed at step 720. By then the
   eddy diffusivity has developed: the largest K/rho is 5.9 m2/s (kEqn,
   near 150 m), 37 m2/s (Deardorff, in the first cell) and 21 m2/s (MRF,
   near 320 m). A restart from the checkpoint reproduces the straight run
   bit for bit for all six spin-ups (compared at step 740).
2. **Ladder.** Restart from the checkpoint at dt = 0.125, 0.25, 0.5, ...,
   1024 s and take 200 steps, climbing until the first rung that fails.
   The reported step is the largest rung below that failure.
   - `erf.change_max` is lifted so that the new step applies at once;
     otherwise ERF grows the step by 10 % per step from the checkpoint's 5 s.
   - Compressible rungs pin the acoustic substeps at a 2 s fast step
     (`erf.fixed_mri_dt_ratio`). Sized from the state, a run that is going
     unstable asks for billions of substeps and hangs instead of aborting.
3. **Pass.** A rung passes if ERF exits cleanly and its last plotfile is
   finite, with |u| and |v| <= 2 G = 20 m/s, |w| <= 0.01 m/s (the column is
   horizontally uniform; the passing rungs reach at most 3e-6 m/s for kEqn
   and Deardorff and 1.1e-4 m/s for MRF), theta inside the sounding range
   +/- 0.5 K and Kmv >= 0.

## Checks

The exit code is non-zero if any of these fails, per closure:

| check | target |
| --- | --- |
| both spin-ups healthy | yes |
| every integrator passes dt = 0.125 s | yes (otherwise the setup is broken) |
| explicit anelastic fails somewhere on the ladder | yes |
| explicit anelastic step over dz^2 / (2 K/rho), K = max(Kmv, Khv) in the restart state | 0.5 to 2 |
| implicit anelastic step over explicit anelastic step | >= 8 |
| implicit compressible step over explicit anelastic step | >= 8 |

When an implicit integrator passes every rung, its step is the top rung,
which is a lower bound.

## Results

Largest step that runs 200 steps from the 1 h state, with the first failing
rung in brackets (1 rank, Release, 2026-09-15; implicit anelastic with
`erf.anelastic_type = MidPoint`):

| closure | explicit anelastic | implicit anelastic | implicit compressible | dz^2 / (2 K/rho) |
| --- | --- | --- | --- | --- |
| kEqn | 2 s (4) | 64 s (128) | 64 s (128) | 2.13 s |
| Deardorff | 0.25 s (0.5) | 64 s (128) | 64 s (128) | 0.339 s |
| MRF | 0.5 s (1) | 64 s (128) | 8 s (16) | 0.246 s |

- Explicit anelastic stops at the diffusion limit for all three closures.
  The passing step is 0.94, 0.74 and 2.03 of dz^2 / (2 K/rho), and the next
  rung aborts on a negative theta after 21, 80, 3 steps. The MRF ratio lies
  above the stated band of 0.5 to 2; the check reports a pass because its
  range comparison is half a band width too loose (erf-model/ERF#4027).
- The implicit solve raises the step by a factor of 32 (kEqn), 256
  (Deardorff) and 128 (MRF) under anelastic, and 32, 256 and 16 under
  compressible. Implicit anelastic stops at 64 s for every closure: the
  128 s rung makes rho theta negative after 10 to 13 steps.
- Before the midpoint stages (RK2 with a second-stage half step, removed
  when ERF-Fire took development's rule for anelastic implicit diffusion)
  implicit anelastic reached 256 s (kEqn, MRF) and 512 s (Deardorff). The
  anelastic spin-up now runs the midpoint stages too, which leaves MRF with a
  larger diffusivity at 1 h (K/rho 50.8 m2/s against 21.5 m2/s), hence its
  shorter diffusion limit.
- The column is horizontally uniform, so advection does no work and the
  implicit steps do not carry over to a real case, where the advective
  Courant number binds first (see the `Neutral_Hill_2D` case).
- MRF under implicit compressible stops at 8 s: a factor 8 below kEqn and
  Deardorff. The 16 s rung does not abort; after 200 steps |u| reaches
  4.6e3 m/s. The pinned substeps are not the cause (measured 2026-09-10).
  Not investigated further.

Wall time: 192 s for the three closures on one rank in Release.

## Running

```bash
python3 sweep_dt.py --exe /path/to/erf_exec
python3 sweep_dt.py --exe /path/to/erf_exec --closure MRF --steps 400
ctest -L dt_sweep
```

The first command sweeps all three closures. The runs go to
`dt_runs/<closure>/<integrator>/dt_<step>/`, each with the command line in
`cmd` and ERF's output in `log`. The script needs only the Python standard
library, plus `erf_plotfile.py` and `rans_checks.py` from `..`.

The CTest entries `RANS_Timestep_Limits_kEqn`, `_Deardorff` and `_MRF` carry
the labels `rans` and `dt_sweep` but not `regression`: the CI runs that
label in Debug, where each sweep would take far longer than the 40-step
cases.
