# FireRosComparison

Compares rate-of-spread formulations against each other on one fixed scenario, so
that a change to any of them shows up as a change in the table rather than having
to be argued from the code.

## The scenario

A wind-driven grass fire on flat ground: 2 x 2 km, 20 x 20 x 30 cells, FM1 short
grass at 6% moisture, a neutral 8 m/s sounding, ignited as a 100 m disk and run
for 200 s on the level-set path with passive coupling. Every variant includes
`inputs_base` and overrides only the lines that distinguish it, so any difference
in the result is the formulation and nothing else. Each run takes a few seconds.

```
./run_comparison.sh /path/to/erf_exec
```

## What it compares

Two axes, crossed:

- **Balbi 2009 against Balbi 2020.** The 2009 form is the steady explicit one:
  no radiative base spread, and a wind response that saturates at twice its
  amplitude coefficient. The 2020 form adds the base term and removes the
  saturation.
- **Isotropic against direction-dependent spread.** With
  `erf.fire.directional_ros = false` the level set is handed a single scalar rate
  and applies it in every direction, so the fire grows as a disc at the head-fire
  rate. With it true the wind and slope are projected onto the front normal and
  the model is evaluated with the projected scalars, so the flanks and backing
  fire slow down.

A third group exercises the hybrid model (`erf.fire.ros_model = "hybrid"`), which
evaluates a primary and a secondary model on every cell and blends them with a
per-cell weight, and the per-fuel Rothermel table
(`erf.fire.rothermel_per_fuel`):

- **Identities.** `hybrid_none` gives the secondary model an empty region, so it
  must reproduce `rothermel_isotropic` exactly; `hybrid_all` gives it the whole
  domain and must reproduce `balbi2020_isotropic`; `rothermel_fuelmap` runs
  Rothermel on a uniform FM1 fuel map with per-fuel coefficients and must again
  match `rothermel_isotropic`. These are bit-for-bit checks, not tolerances.
- **Splits.** `hybrid_region` hands cells east of x = 800 m, the downwind edge
  of the ignition disc, to Balbi 2020; `hybrid_fuel` does the same by fuel
  code on a map that is FM1 west of that line and FM3 east of it. In both the
  head fire runs at the secondary rate while the flanks and backing fire keep
  the Rothermel rate, so the burned area falls between the two identities.

The `sec_cells` column is the number of fire cells whose weight exceeds one
half, read from the last `[FIRE DEBUG] Hybrid ROS` line; 48 of the 80 columns
lie east of the split, hence 3840.

Two further groups cover the direction-dependent path and the wind selector:

- **Directional identities and split.** `hybrid_none_directional` and
  `hybrid_all_directional` repeat the identities with
  `erf.fire.directional_ros = true`, where both members are rebuilt along the
  front normal at every Runge-Kutta stage, and must match
  `rothermel_directional` and `balbi2020_directional`.
  `hybrid_region_directional` is the split on that path.
- **Wind selector.** `hybrid_wind` ramps the weight from 0 at 1 m/s to 1 at
  3 m/s of midflame wind, rebuilt every fire step; the 8 m/s sounding gives
  about 2 m/s after the WAF, so every cell carries a weight near one half.
  `hybrid_wind_off` puts the band far above the midflame wind and is another
  Rothermel identity.

## Reference results

Burned cells after 200 s from a 50-cell ignition disk, and the head-fire rate of
spread:

| variant | burned cells | max ROS (m/s) | sec_cells |
|---|---|---|---|
| `rothermel_isotropic` | 132 | 0.250 | - |
| `rothermel_directional` | 78 | 0.250 | - |
| `balbi2009_isotropic` | 200 | 0.390 | - |
| `balbi2009_directional` | 84 | 0.390 | - |
| `balbi2020_isotropic` | 232 | 0.514 | - |
| `balbi2020_directional` | 138 | 0.514 | - |
| `rothermel_fuelmap` | 132 | 0.250 | - |
| `hybrid_none` | 132 | 0.250 | 0 |
| `hybrid_all` | 232 | 0.514 | 6400 |
| `hybrid_region` | 162 | 0.514 | 3840 |
| `hybrid_fuel` | 140 | 0.329 | 3840 |
| `hybrid_none_directional` | 78 | 0.250 | 0 |
| `hybrid_all_directional` | 138 | 0.514 | 6400 |
| `hybrid_region_directional` | 98 | 0.514 | 3840 |
| `hybrid_wind_off` | 132 | 0.250 | 0 |
| `hybrid_wind` | 216 | 0.389 | 6400 |

Three things to read out of it.

**Direction-dependence always burns less.** The head rate is unchanged in every
pair — the max ROS column is identical down each model — but the area is smaller
because the flanks no longer advance at the head rate. That is the whole effect
of the switch.

**Balbi 2020 spreads faster than 2009**, 0.514 against 0.390 m/s at this wind,
which is the unsaturated wind response of the newer form.

**The two Balbi forms respond very differently to the switch.** The 2009 form
loses 58% of its area (200 to 84) where the 2020 form loses 41% (232 to 138).
That is the base term doing exactly what it is there for: the 2009 form has no
no-wind spread at all, so once the wind is projected out on the flanks its
flanking rate is zero and the fire can only run downwind. The 2020 form retains
its radiative base rate on the flanks and keeps spreading sideways, slowly.

**The identities hold exactly.** `hybrid_none` and `rothermel_fuelmap` match
`rothermel_isotropic`, and `hybrid_all` matches `balbi2020_isotropic`, in both
columns. The blend is `(1 - w) R_p + w R_s`, so a weight of exactly 0 or 1
returns one model's value untouched.

**The directional identities hold exactly too**, at 78 and 138, which also
checks that moving the three direction-dependent drivers onto one shared
Runge-Kutta routine changed nothing. `hybrid_region_directional` burns 98
cells, between them, for the same reason as its isotropic counterpart.

**The wind selector blends rather than switches.** With every cell at a
weight near one half, `hybrid_wind` reports a head rate of 0.389 m/s, close to
the mean of 0.250 and 0.514, and a burned area between the two identities.

**The splits sit between the identities.** `hybrid_region` burns 162 cells:
more than Rothermel because the head runs at the Balbi rate from the first
step, fewer than Balbi because the flanks and the backing fire still spread at
0.25 m/s. Its max ROS is the Balbi value since the diagnostic is taken over
burning cells and the head is in the Balbi region. `hybrid_fuel` reports
0.329 m/s because Balbi 2020 evaluated on FM3 tall grass is slower than on FM1
at this wind, which is the per-fuel Balbi table doing its job.

## Backward compatibility

`erf.fire.directional_ros` defaults to false. Omitting it entirely and setting it
to false give the same answer, 132 cells for Rothermel, which is also what the
code produced before the switch existed. `erf.fire.ros_model` still defaults to
`rothermel` and `erf.fire.rothermel_per_fuel` to false, so runs that do not
name the hybrid or the per-fuel table are unchanged; without the flag the
Rothermel kernel keeps spreading with the domain `fuel_model_id` on every cell
of a fuel map, as before. The older Balbi-only flag
`erf.fire.balbi.directional` still works and is equivalent for that model: both
give 138 cells on `balbi2020_directional`.

## What this is not

The FARSITE path is not in the table. It carries its own directionality through
the Anderson length-to-width ellipse and is not comparable cell-for-cell with
the level-set path; see the note on anisotropy in
`Docs/sphinx/fire/propagation_methods.rst`.

The projection reproduces neither the observed saturation of the length-to-width
ratio nor a backing rate below the no-wind rate, both of which the empirical
ellipse carries as calibration. It is the physically consistent choice when the
wind is resolved, not a calibrated one.
