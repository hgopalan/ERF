# FireLineFire

The idealized surface line fire of Coen et al. (2013), the WRF-Fire paper,
without and with an ambient wind: short grass (Anderson fuel model 1) at
5.5 % moisture on flat ground, a 40 m wide line fire across the whole
(periodic) y extent, so the fire is a pair of straight fronts, a head fire
spreading with the wind and a backing fire against it. The WRF-Fire choices
are mirrored: the wind interpolated to 6.1 m and reduced by the fuel model's
wind reduction factor (WRF-Fire's `windrf(1) = 0.36`, here Andrews'
unsheltered factor 0.362 for the 1 ft grass bed), no midflame wind cap
(WRF-Fire caps R at 6 m/s only), a 0.03 m roughness length. Nine decks share
one base:

- `nowind`: no ambient wind. Both fronts must move at the no-wind rate R0.
- `wind2p5`: a 2.5 m/s sounding, Coen's Control.
- `wind5`: a 5 m/s sounding, Coen's WSHi.
- `wind5_cap`: `wind5` with ERF's default midflame wind limit
  (`erf.fire.use_wind_limit = true`): Rothermel's (1972, eq. 87) U <= 0.9 I_R,
  about 750 ft/min or 3.8 m/s for fuel model 1 at 5.5 %, which the 5 m/s
  sounding's midflame wind (about 1.6 m/s) never reaches, so `wind5_cap` is
  `wind5`. The head is checked at the bounded wind.
- `wind5_cap_fuel_class`: `wind5` with the fuel-class rule this code applied
  until 2026-10 (`erf.fire.wind_limit = fuel_class`, 300 ft/min or 1.52 m/s for
  fuel model 1). The head is checked at the capped wind, so the pair shows
  what that cap did.
- `nowind_2way`: no ambient wind with the heat coupled back, reported next to
  Coen's NoWind fire (0.02 m/s outward). Its own indraft blows into the burned
  area at both fronts (1.3 m/s at 6.1 m), which the directional model clips at
  zero, so both fronts stay at R0 = 0.0240 m/s.
- `wind2p5_2way`, `wind5_2way`: the Control and WSHi winds with the heat
  coupled back, in the 400 by 80 by 160 m box of the one-way decks. Reported,
  not checked: the box is too small for a coupled fire (see below).
- `wind2p5_open_2way`: the Control wind with the heat coupled back in a box
  the plume does not fill: 1600 m in x, a 480 m lid, a 200 m line in 400 m of
  y, 10 m cells, 600 s at a 0.5 s step. Reported, not checked.

```
MPIRUN="mpirun -np 4" ./run_linefire.sh /path/to/erf_exec
```

The one-way decks are one-way so that the fronts must move at Rothermel's
rates for the wind the fire samples: `check_linefire.py` takes the head and
backing rates from the arrival-time differences between consecutive probe
cells, evaluates Rothermel (1972) for fuel model 1 at 5.5 % (the same
equations as `Source/Fire/ERF_Rothermel.cpp`, ported independently) at the
effective wind the fire reports every step, and requires the backing fire at
R0 and the head at R0 (1 + phi_w) to 10 %. The surface layer slows the wind by
28 % over the 5 m/s run and phi_w grows about as the wind squared, so each
probe pair is held to the head rate averaged over its own arrival window, not
to the rate at the run-mean wind.

The reference numbers from the paper (Table 1 and section 4): a 1 km long,
40 m wide line in fuel model 1 at 5.5 % moisture in a 5 km periodic LES with
a convective boundary layer; NoWind crept outward at 0.02 m/s, Control (2.5
m/s) ran a 0.22 m/s head, halving the wind cut the rate by a fifth and
doubling it (WSHi, 5 m/s) raised it by four fifths. Those are coupled
results in turbulent winds; the one-way decks here reproduce Rothermel at the
sampled wind, and the gap between the two is the fire-atmosphere feedback
the two-way deck begins to show.

## Why the coupled head runs fast in the small box

With the heat coupled back at 2.5 m/s, the 400 by 80 by 160 m box gives a
0.365 m/s head against the paper's 0.22 m/s. The cause is the box, not the
fire. A rerun of `wind2p5_2way` with plotfiles every 20 s
shows how. The plume reaches the 160 m lid within 20 s (1.3 K warm at 157.5 m
above the line) and spreads along it; by 100 s a layer 4.5 K warm fills the
top of the whole 400 m box. With x periodic, the air the plume carries away
comes back from upwind: the 6.1 m wind 100 m upwind of the line falls from
2.5 m/s to 0.56 m/s at 80 s and turns to -1.0 m/s at 160 s, while the wind
the fire samples at the head reaches 8.5 to 8.7 m/s at 160 to 180 s. A
faster head releases more heat, which strengthens the loop: the head
advances 21 m in the first 100 s (0.21 m/s) and 53 m from 100 to 180 s
(0.66 m/s).

The table below shows when the head reached two probes, 18.75 and 38.75 m
ahead of the line's east edge, in 180 s runs that change the box or the heat,
and the rate between the two (from the `[FIRE PROBE]` lines of each run's log).

| Run (change from `wind2p5_2way`) | Arrival at 198.75 m (s) | Arrival at 218.75 m (s) | Head rate between (m/s) |
|---|---|---|---|
| none (400 x 80 x 160 m, 5 m cells) | 78.6 | 132.6 | 0.37 |
| x extent 1600 m | 77.8 | 167.7 | 0.22 |
| x extent 1600 m, 10 m cells | 77.0 | 165.2 | 0.23 |
| x extent 1600 m, heat scaled by 0.888 | 80.6 | 167.4 | 0.23 |
| x extent 1600 m, extinction depth 50 m | 79.0 | 171.5 | 0.22 |
| lid at 480 m | 88.9 | not by 180 s | below 0.22 |
| a 200 m line in 400 m of y | 89.0 | not by 180 s | below 0.22 |
| all three, 10 m cells (`wind2p5_open_2way`, 600 s) | 93.8 | 240.4 | 0.14 |
| all three, 40 m cells (the paper's), 20 m in z, 1920 x 640 x 480 m | 78.4 | not by 180 s | below 0.20 |

- **Opening either the x extent or the lid removes the speed-up.** In the
  small box the head then runs on to 238.75 m by 164.5 s, 0.63 m/s over that
  last pair; in the 1600 m box it holds 0.22 to 0.23 m/s on 5 and 10 m cells.
- **The heat released is not the cause.** Scaling the feedback by 0.888, so
  the heat matches WRF-Fire's (WRF-SFIRE takes the fuel's moisture share
  M/(1 + M) = 0.052 out of the load and burns it at 17.43 MJ/kg, against
  ERF's 18.6 MJ/kg: 0.948 x 17.43 / 18.6 = 0.888), or setting the extinction
  depth to 50 m (WRF-Fire's; ERF's default is 45 m) leaves the head at 0.22
  to 0.23 m/s. The scaling also lowers the latent heat, which WRF-Fire does
  not.
- **The checker's `U6.1` column is the largest 6.1 m wind in the domain**
  each step, not the wind at the head.
- **Every box starts the same way.** All runs reach 188.75 m at 38 to 40 s
  and 198.75 m at 77 to 94 s (0.20 to 0.24 m/s from the line's edge), before
  the boxes part.

`wind2p5_open_2way` runs the open box to 600 s. Its probe pairs give a 0.200 m/s
head (`check_linefire.py`; Rothermel at the domain's largest effective wind
gives 0.242), and the pairs after the start give 0.136 m/s (198.75 to
218.75 m) and 0.120 m/s (218.75 to 238.75 m). It runs in about 9 minutes on
4 ranks. That is below
the paper. The paper's LES has a convective boundary layer driven by
100 W/m2 of surface heating and a wind held at 2.5 m/s; this deck is
neutral, with no surface heating and no imposed turbulence, and nothing
holds its wind at 2.5 m/s. Its x boundary is still periodic: air leaving
downwind comes back upwind after about 1600 m / 2.5 m/s = 640 s, so 600 s is
about as long as this box allows. A fair comparison needs the paper's setup:
a box of a few km with a 1.2 km top, a 1 km line, the surface heating and a
driven wind, 30 min of spin-up and at least an hour of fire. That is left
for a later PR.
