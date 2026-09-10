# Smoke_Tracer

## Purpose
Exercises passive smoke-tracer emission driven by fire sensible heat release.

## Physics / Model Features Exercised
- Smoke emission proportional to fire heat flux
- Transport of the smoke scalar by the dycore
- Lagged fire-atmosphere coupling as the heat source

## Expected Results
Smoke should appear only where the fire is burning and be advected downwind.
Column-integrated smoke grows while fuel is being consumed and levels off once
the fuel behind the front is exhausted. Smoke mass should never be negative.

## Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `erf.fire.smoke_enable` | `true` | Enables the smoke tracer component. |
| `erf.fire.smoke_emission_factor` | `0.015` | Smoke mass emitted per unit fuel burned [kg/kg]. |
| `erf.fire.smoke_heat_of_comb` | `1.86e7` | Heat of combustion used to convert heat release to fuel mass [J/kg]. |
| `erf.fire.coupling_type` | `"lagged"` | Fire heat is injected into the atmosphere. |

## Checking the output
The atmosphere plotfiles `plt_1_NNNNN` carry `smoke`, the smoke mass
concentration [kg/m^3] as stored in the conserved state. After a run,

    python3 check_smoke_tracer.py

checks the fire plotfiles and statistics, then reads `smoke` from every
atmosphere plotfile. The field must be finite and present at the end. Its
negative mass must stay below 10% of its positive mass. Its domain total must
never decrease (the domain is periodic in x and y with walls top and bottom, so
smoke only enters, through the fire).

The default `Upwind_3rd` scalar advection is not monotone. It rings next to the
one-layer surface source, so single cells go negative: in a 60 s run the worst
cell reached 12% of the maximum, while the negative mass was 2-6% of the
positive mass.

## Notes
- Enabling smoke adds a conserved component, so plotfile component counts differ
  from the other coupling cases.
- `smoke` must be listed in `erf.plot_vars_1`. Before 2026-09-10 it could not be
  plotted at all (the name had no component mapping), and this deck set
  `erf.plot_vars`/`erf.plot_int`, which ERF does not read.

## References
- Urbanski 2014, Wildland fire emissions, carbon, and climate: Emission factors.
