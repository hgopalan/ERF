# Canonical RANS: largest time step per closure and integrator

How far the implicit column solve moves the time step, for three closures
that set a vertical eddy diffusivity: the one-equation k model
(`erf.rans_type = kEqn`), Deardorff (`erf.les_type = Deardorff`) and the
MRF PBL scheme (`erf.pbl_type = MRF`), each under three integrators:

| integrator | inputs |
| --- | --- |
| explicit anelastic | `erf.anelastic = 1`, `erf.vert_implicit = false` |
| implicit anelastic | `erf.anelastic = 1`, `erf.vert_implicit = true` |
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
rung in brackets (1 rank, Release, 2026-09-10):

| closure | explicit anelastic | implicit anelastic | implicit compressible | dz^2 / (2 K/rho) |
| --- | --- | --- | --- | --- |
| kEqn | 2 s (4) | 256 s (512) | 64 s (128) | 2.13 s |
| Deardorff | 0.25 s (0.5) | 512 s (1024) | 64 s (128) | 0.339 s |
| MRF | 0.5 s (1) | 256 s (512) | 8 s (16) | 0.582 s |

- Explicit anelastic stops at the diffusion limit for all three closures.
  The passing step is 0.94, 0.74 and 0.86 of dz^2 / (2 K/rho), and the next
  rung aborts on a negative theta after 15, 75 and 11 steps.
- The implicit solve raises the step by a factor of 128 (kEqn), 2048
  (Deardorff) and 512 (MRF) under anelastic, and 32, 256 and 16 under
  compressible.
- The column is horizontally uniform, so advection does no work and the
  implicit steps do not carry over to a real case, where the advective
  Courant number binds first (see `Neutral_Hill_2D` in `../RESULTS.md`).
  What stops the implicit runs here was not identified.
- MRF under implicit compressible stops at 8 s: a factor 8 below kEqn and
  Deardorff, and 32 below its own anelastic step. The 16 s rung does not
  abort; after 200 steps |u| reaches 4.6e3 m/s. The pinned substeps are not
  the cause: with ERF's own substep count the same rung reaches 1.7e3 m/s,
  and with a 1 s fast step 6.3e3 m/s. Not investigated further.

Wall time: 57 to 87 s per closure on one rank in Release, 26 to 32 ERF runs
each.

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
