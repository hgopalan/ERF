# FireFluxPartition

How the fire's heat release is split between the sensible and latent flux
handed to the atmosphere. A grass fire on flat ground with lagged coupling
is run with the two settings of `erf.fire.heat_flux_partition`:

- `legacy` (default): the sensible flux is the full heat release of the dry
  fuel, `Q = w h / tau`, and the latent flux is derived from it as in
  WRF-SFIRE, `Q_lat = (L_v / h) (b + (1 - b) 0.56) Q` with
  `b = M_f / (1 + M_f)`. This is the form ERF-Fire has always used.
- `cfbm`: the sensible flux is scaled by the dry-fuel fraction of the wet
  fuel mass, `1 / (1 + M_f)`, Eq. 4 of Jiménez y Muñoz et al. (2026,
  Geosci. Model Dev. 19, 3035), the Community Fire Behavior Model. Their
  Eq. 5 is the legacy latent flux, which is unchanged.

Only the flux handed to the atmosphere changes; the fire-grid heat flux in
the plotfiles and the flame diagnostics stay the unpartitioned release.

```
MPIRUN="mpirun -np 4" ./run_partition.sh /path/to/erf_exec
python3 plot_partition.py
```

Ten decks, 60 s each. Five run one-way (`erf.fire.fire_atm_feedback =
0`, fluxes computed and printed but not injected) so the fire evolves
identically and the checks are exact: `default` (no key), `legacy`, `cfbm`,
`cfbm_wet` (30 % moisture) and `cfbm_nolatent`. The script reads the
`[FIRE DEBUG]` flux lines and checks that the default deck reproduces the
legacy one line for line, that the cfbm sensible flux is `1/(1+M_f)` times
the legacy one at every step, that the latent flux is identical under both
partitions, that the factor is applied with the latent flux off, and that
the wet deck reports `1/1.30`. `legacy_2way` and `cfbm_2way` inject the
fluxes and are listed for comparison only: with the feedback on, the weaker
heating changes the wind a little and the fronts ignite cells a step apart,
so the flux maxima depart from the factor on a few dozen of the 480 steps
while the burned area stays the same. The plot shows the one-way flux
maxima against time and the sensible-flux ratio for both pairs.

Three `smoke_*` decks turn the smoke tracer on, one-way. The tracer is the
CFBM's smoke (2 % of the fuel burnt, Coen 2013, into the first atmospheric
layer). `smoke_legacy` and `smoke_cfbm` must give an identical smoke source
at every step: the emission divides the partition factor back out of the
lagged flux before forming the fuel burnt, so the smoke does not change with
`erf.fire.heat_flux_partition`. `smoke_cfbm_fuel` sets
`erf.fire.smoke_heat_from_fuel = true`, which takes the heat per kilogram
from the fuel model (18.608 MJ/kg for the Anderson models) instead of
`erf.fire.smoke_heat_of_comb` (18.7 MJ/kg); its source must be the ratio of
the two, 1.00494, times `smoke_cfbm` at every step.

| M_f | 1/(1+M_f) |
|-----|-----------|
| 0.06 | 0.943 |
| 0.08 | 0.926 |
| 0.12 | 0.893 |
| 0.30 | 0.769 |

The factor is mass bookkeeping rather than an energy balance: the heat spent
vaporising the fuel water is about `M_f L_v` per unit dry fuel, roughly 1.5 %
of the heat content at 10 % moisture, so the CFBM partition removes more
heat than evaporation costs. It is offered for parity with CFBM and WRF-SFIRE
results, not as a correction to the legacy form.
