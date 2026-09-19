# fire-lsm-reinit-head worktree

Git worktree of ERF, branch `fire-lsm-reinit-head`, forked directly from
`hgopalan/ERF`'s (`origin`) current `ERF-Fire` tip (unlike the older
`ERF_fire_lsreinit` worktree, which forked ~1500 commits back and drifted
stale). Do not push to `origin` without explicit confirmation.

## What this worktree is for

Validating and tuning ERF's existing level-set reinitialization
(`Source/Fire/ERF_Reinitialize.H`, `fire_levelset::reinitialize_phi`) — see
`Exec/RegTests/FireLevelsetReinit/README.md` for the actual test and
findings. This worktree carries three source fixes, cherry-picked from
`ERF_fire_lsreinit`'s `445bda258` and then re-resolved by hand against the
current tip (that commit's own parent had two WRF-Fire-matching ParmParse
options, `reaction_velocity_formula`/`wrf_bmst_compat`, from a *separate*
in-flight PR (#439) that isn't merged into `ERF-Fire` yet — deliberately
dropped here so this branch has zero dependency on that PR or on WRF-Fire
matching at all):

1. **`ERF_Reinitialize.H`**: a WRF-Fire-style smoothed sign for the general
   reinit update, and a monotonicity clamp ("fire area can only increase")
   — fixes a severe long-run corner-erosion regression in the original
   scheme (412.5m eroded by t=1260s, worse than no reinit's 62.5m; now
   37.5m, better than no reinit).
2. **`ERF_PolygonIgnition.H`**: `init_phi_from_polyline`'s t=0 initial
   condition is now a true signed-distance function (was a discontinuous
   constant inside the ignition band).
3. **`ERF_FireLayer.cpp`**: a `fire_debug` print of the init-time Rothermel
   coefficients (R0/I_R/beta) — cosmetic, no behavior change.

Explicitly **not** carried over from `445bda258`: `reaction_velocity_formula`/
`wrf_bmst_compat` threading (belongs to PR #439, kept separate on purpose —
see [[erf_fire_advective_wrf_comparison]] / auto-memory for why these two
PRs must stay independent) and `heat_content_override_btu_lb` (a WRF-matching
knob that was dead code — declared and parsed but never actually wired into
any Rothermel call site — and unused by this worktree's own regtest, so
dropped rather than carried forward unused).

## Regtest: `Exec/RegTests/FireLevelsetReinit/`

Massively simplified from the original sweep (which lived in
`ERF_fire_lsreinit` and compared against WRF-Fire): a single fuel (FM1,
short grass), uncoupled (`fire_atm_feedback=0`, `prescribed_wind`), 120s
line fire, two decks (`inputs_noreinit`, `inputs_reinit` at ERF's compiled
defaults), one comparison — does reinit's head-position error against the
theoretical Rothermel `Rf` come in lower than no-reinit's at t=120s? No
WRF-Fire dependency anywhere; the theoretical `Rf` is ERF's own native
formula (Albini reaction-velocity exponent, no `wrf_bmst_compat`
deflation), computed independently in `check_regtest.py`.

**Gotcha already hit and fixed**: the theoretical head-position baseline
must be the *actual* t=0 front position (x=525, read from the t=0
plotfile), not the nominal ignition-line x-coordinate (x=500) —
`erf.fire.ignition.polyline_width=25` means the initial phi=0 front already
sits a half-line-width ahead of the nominal ignition line. Using the wrong
baseline flips the result (makes no-reinit look better than reinit at
120s, the opposite of the true behavior).

Run: `./run_regtest.sh /path/to/erf_exec` (needs `FI_PROVIDER=tcp`
exported — MPICH's default PSM3/OFI provider fails `MPI_Init` even for a
single rank in this environment) or `SKIP_RUN=1 ./run_regtest.sh x` once
the decks have already been run.

## Result (current source)

Reinit tracks the theoretical head position to ~0.03% at t=120s; no-reinit
lags by ~2%. Confirmed by direct perimeter plot as well as the numeric
check — see the worktree's own scratch plots from the original development
session (not checked in).
