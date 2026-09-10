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

The deck advects the scalars with `WENOZ5` (`erf.dryscal_horiz_adv_type` and
`erf.dryscal_vert_adv_type`; theta and momentum keep the dycore scheme). The
default `Upwind_3rd` is not monotone and rings next to the one-layer surface
source. In a 60 s run with gravity on, that ringing left 18% of the positive
smoke mass as negative mass, and the worst cell reached -26% of the maximum,
so the `positive` check failed. With `WENOZ5` the negative mass rounds to zero
(the worst cell was -4e-12 kg/m^3 against a maximum of 1.4e-4 kg/m^3). The
plume, the fire and the smoke total (38.3 kg) were the same under both schemes.

## Atmosphere and plume
The atmosphere is neutral (300 K) with a uniform 5 m/s westerly. It starts from
`input_sounding` in hydrostatic balance with `erf.use_gravity = true`, and the
geostrophic forcing holds the wind, so the fire's heat drives a buoyant plume
that lofts the smoke.

Until 2026-09-10 the deck used `erf.init_type = "uniform"`, which runs with
gravity off: `erf.use_gravity` defaults to false, and uniform init allows
gravity only when anelastic. The heat warmed the air but raised no plume. In
a 60 s run, the largest vertical velocity was 0.006 m/s with gravity off and
1.5 m/s with it on, in the bent-over plume 225 m downwind of the ignition. The
smoke's centre of mass rose from 5 m to 13 m. The wind in the lowest cell,
3.76 m/s everywhere with gravity off, sped up to 4.4 m/s just downwind of the
fire and slowed to 3.0 m/s under the plume. The head rate of spread in
`fire_stats_smoke.csv` rose from 0.188 to 0.216 m/s, and the peak heat flux
from 46 to 53 kW/m2. The burned area (0.42 ha) was unchanged at 60 s. Upwind_3rd
had kept the negative smoke mass at 2% of the positive mass with gravity off
(worst cell -3%), so the undershoot only mattered once the plume appeared.

## Notes
- Enabling smoke adds a conserved component, so plotfile component counts differ
  from the other coupling cases.
- `smoke` must be listed in `erf.plot_vars_1`. Before 2026-09-10 it could not be
  plotted at all (the name had no component mapping), and this deck set
  `erf.plot_vars`/`erf.plot_int`, which ERF does not read.

## References
- Urbanski 2014, Wildland fire emissions, carbon, and climate: Emission factors.
