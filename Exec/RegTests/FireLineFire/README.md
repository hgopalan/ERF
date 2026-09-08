# FireLineFire

The idealized surface line fire of Coen et al. (2013), the WRF-Fire paper,
without and with an ambient wind: short grass (Anderson fuel model 1) at
5.5 % moisture on flat ground, a 40 m wide line fire across the whole
(periodic) y extent, so the fire is a pair of straight fronts, a head fire
spreading with the wind and a backing fire against it. The WRF-Fire choices
are mirrored: the wind interpolated to 6.1 m and reduced by the fuel model's
wind reduction factor (WRF-Fire's `windrf(1) = 0.36`, here Andrews'
unsheltered factor 0.362 for the 1 ft grass bed), no midflame wind cap
(WRF-Fire caps R at 6 m/s only), a 0.03 m roughness length. Four decks share
one base:

- `nowind`: no ambient wind. Both fronts must move at the no-wind rate R0.
- `wind2p5`: a 2.5 m/s sounding, Coen's Control.
- `wind5`: a 5 m/s sounding, Coen's WSHi.
- `wind2p5_2way`: the Control wind with the heat coupled back. Reported,
  not checked: the 400 by 80 m box is far smaller than the paper's 5 km LES.

```
MPIRUN="mpirun -np 4" ./run_linefire.sh /path/to/erf_exec
```

The one-way decks are one-way so that the fronts must move at Rothermel's
rates for the wind the fire samples: `check_linefire.py` takes the head and
backing rates from the arrival-time differences between consecutive probe
cells, evaluates Rothermel (1972) for fuel model 1 at 5.5 % (the same
equations as `Source/Fire/ERF_Rothermel.cpp`, ported independently) at the
effective wind the fire reports, and requires the backing fire at R0 and the
head at R0 (1 + phi_w) to 10 %.

The reference numbers from the paper (Table 1 and section 4): a 1 km long,
40 m wide line in fuel model 1 at 5.5 % moisture in a 5 km periodic LES with
a convective boundary layer; NoWind crept outward at 0.02 m/s, Control (2.5
m/s) ran a 0.22 m/s head, halving the wind cut the rate by a fifth and
doubling it (WSHi, 5 m/s) raised it by four fifths. Those are coupled
results in turbulent winds; the one-way decks here reproduce Rothermel at the
sampled wind, and the gap between the two is the fire-atmosphere feedback
the two-way deck begins to show.
