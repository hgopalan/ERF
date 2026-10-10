# FireSplitHamiltonian

Rotation invariance of the advective level-set scheme, with and without
`erf.fire.directional_split_hamiltonian`. A finite ignition line burns in a
uniform wind, once along +x and once with the whole scenario rotated by 34 deg
about the domain centre, and the two burned regions are compared after
un-rotating.

```
MPIRUN="mpirun -np 1" ./run_firesplithamiltonian.sh /path/to/erf_exec   # four decks at once, then the checks
SKIP_RUN=1 ./run_firesplithamiltonian.sh x                              # checks only
SERIAL=1 ./run_firesplithamiltonian.sh /path/to/erf_exec                # one deck at a time
```

Needs `numpy`, `scipy` and `yt`. Each deck takes about 50 s on one core; the four
run concurrently by default, so the whole suite takes about a minute.

## The case

4800 x 4800 x 500 m, periodic sideways, a uniform 4.005 m/s wind
(`erf.fire.prescribed_wind`, so every deck sees the same magnitude and the
atmosphere cannot differ between them), `erf.fire.use_wind_limit = false`.
Flat ground, Anderson fuel model 1 at 6 % moisture (Rothermel head rate
Rf = 1.701 m/s), 25 m fire cells, one-way coupling. The 1 km ignition line sits
1200 m upwind of the domain centre, perpendicular to the wind, and the fire
runs 1200 s -- far from the periodic edges, so the square grid is the only thing
that breaks the rotational symmetry. `erf.fire.directional_wind_coupling =
"advective"` and the Jiang-Peng reinitialization every level-set substep
(`reinit_every = 1`, `reinit_scheme = "jiang_peng"`) are on in every deck:
that is the combination in which the baseline scheme grows a wing at an
oblique angle.

| deck | wind heading | `directional_split_hamiltonian` |
|---|---|---|
| `baseline_0` | 0 deg (+x) | false |
| `baseline_34` | 34 deg | false |
| `split_0` | 0 deg (+x) | true |
| `split_34` | 34 deg, ignition line and standoff rotated with it | true |

`line_0.csv` and `line_34.csv` are the same line rotated by 34 deg about
(2400, 2400); `make_ignition_lines.py` writes them.

## Why the baseline fails

With advective coupling the level-set equation is a two-term Hamiltonian,
`phi_t + R0 |grad phi| + max(V . grad phi, 0) = 0`, with `V` the wind-driven
velocity vector. The baseline builds one scalar `R(n)` from an estimated front
normal and multiplies it by a single Godunov-upwinded `|grad phi|`. That
upwinds the wind-driven term along the front normal, but its information
travels along `V`, including the component tangential to the front. The
tangential transport is not upwinded along `V`, and its error depends on the
angle between `V` and the grid axes: none at 0 deg, large at 34 deg.

`directional_split_hamiltonian` upwinds `R0 |grad phi|` (Godunov) and each
advective term (wind, slope) separately, by the sign of that term's own
velocity components. `V` depends only on the wind and slope, so it is built once
per advection call.

## What the checker measures

`check_firesplithamiltonian.py` reads every `fire_phi` plotfile.

- **Mismatch**: the area where the 0 deg burned region (`phi < 0`) and the
  un-rotated 34 deg burned region disagree, as a fraction of the 0 deg burned
  area, at every saved time from 300 s. The 34 deg field is sampled bilinearly
  at each 0 deg cell centre rotated about the domain centre.
- **Head rate**: the speed of the front along the wind ray from the ignition
  line's midpoint, fitted over t >= 600 s, against Rothermel's Rf from an
  independent port of the equations in `Source/Fire/ERF_Rothermel.cpp`.

| check | bound | measured |
|---|---|---|
| split mismatch, every time | <= 1 % | 0.00-0.10 % |
| baseline mismatch, t >= 600 s | >= 5 % | 9.1-13.4 % |
| baseline / split mismatch, t >= 600 s | >= 10x | >= 87x |
| split head rate, 0 deg and 34 deg | within 3 % of Rf | +0.02 %, +0.01 % |

The baseline mismatch is 4.6 % at 300 s and grows as the wing develops, which
is why its bound starts at 600 s. The baseline head rate is also within 0.03 %
of Rf at both angles (printed for reference, not asserted): the artifact is in
the flanks and corners, not the head, so the head-rate check confirms the split
scheme does not trade accuracy for invariance rather than discriminating
between the schemes.
