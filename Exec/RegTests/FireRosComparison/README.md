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

Further groups cover the direction-dependent path, the wind selector, the
other rate-of-spread models, the nearest-column wind mapping, two Balbi
options, hybrids built from two non-Balbi members, and a non-burnable
fuel-code strip:

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

Re-measured 2026-10-09 on one rank with the validated defaults: the Rothermel
rows (and BEHAVE on FM1, the Rothermel identities and `rothermel_code0_*`) carry
the Rothermel wind limit, 0.9 I_R, instead of the 300 ft/min fuel-class cap, so
the sounding's midflame wind counts in full and the head runs at 0.444 m/s
over 240 cells (0.250 m/s and 112 cells under the old cap, which
`erf.fire.wind_limit = fuel_class` restores); `cheney_gould` is the 1998 model
(0.927 m/s, 484 cells) and `grass_simple` the earlier fit it replaced (0.180
m/s, 88 cells, the `cheney_gould` rows of the 2026-09 table). The Balbi and
MacArthur rows are unchanged.

Burned cells after 200 s from a 50-cell ignition disk, and the head-fire rate of
spread, with the default hybrid WENO5-Z/first-order level set and the
near-front artificial viscosity of 0.1 (`erf.fire.levelset.gradient =
weno5z_front`, `eps_visc_front = 0.1`, 2026-09-05; the single viscosity
differs only in `balbi2020_directional` 120, `hybrid_all_directional` 120
and `hybrid_wind` 180). The first-order
scheme that preceded it burned more cells in every row (for example
`rothermel_isotropic` 132 against 112, `balbi2020_isotropic` 232 against
216) at the same head rate; the identities and the direction-dependence
ordering below are unchanged.

| variant | burned cells | max ROS (m/s) | sec_cells |
|---|---|---|---|
| `rothermel_isotropic` | 240 | 0.444 | - |
| `rothermel_directional` | 80 | 0.444 | - |
| `balbi2009_isotropic` | 172 | 0.390 | - |
| `balbi2009_directional` | 80 | 0.390 | - |
| `balbi2020_isotropic` | 216 | 0.516 | - |
| `balbi2020_directional` | 122 | 0.516 | - |
| `rothermel_fuelmap` | 240 | 0.444 | - |
| `hybrid_none` | 240 | 0.444 | 0 |
| `hybrid_all` | 216 | 0.516 | 6400 |
| `hybrid_region` | 238 | 0.516 | 3840 |
| `hybrid_fuel` | 218 | 0.444 | 3840 |
| `hybrid_none_directional` | 80 | 0.444 | 0 |
| `hybrid_all_directional` | 122 | 0.516 | 6400 |
| `hybrid_region_directional` | 86 | 0.516 | 3840 |
| `hybrid_wind_off` | 240 | 0.444 | 0 |
| `hybrid_wind` | 216 | 0.482 | 6400 |
| `macarthur_isotropic` | 680 | 1.015 | - |
| `macarthur_directional` | 156 | 1.015 | - |
| `cheney_gould_isotropic` | 484 | 0.927 | - |
| `cheney_gould_directional` | 116 | 0.927 | - |
| `grass_simple_isotropic` | 88 | 0.180 | - |
| `grass_simple_directional` | 84 | 0.180 | - |
| `behave_isotropic` | 240 | 0.444 | - |
| `behave_directional` | 80 | 0.444 | - |
| `rothermel_nearest` | 240 | 0.444 | - |
| `balbi2020_reference_wind` | 468 | 0.933 | - |
| `balbi2020_extinction_wet` | 52 | 0.000 | - |
| `hybrid_behave_cheney` | 320 | 0.927 | 3840 |
| `hybrid_behave_cheney_directional` | 104 | 0.927 | 3840 |
| `hybrid_blend_width` | 238 | 0.516 | 3840 |
| `rothermel_code0_legacy` | 240 | 0.444 | - |
| `rothermel_code0_masked` | 202 | 0.444 | - |

Three things to read out of it.

**Direction-dependence always burns less.** The head rate is unchanged in every
pair — the max ROS column is identical down each model — but the area is smaller
because the flanks no longer advance at the head rate. That is the whole effect
of the switch.

**Balbi 2020 spreads faster than 2009**, 0.516 against 0.390 m/s at this wind,
which is the unsaturated wind response of the newer form.

**The two Balbi forms respond very differently to the switch.** The 2009 form
loses 53% of its area (172 to 80) where the 2020 form loses 44% (216 to 122).
That is the base term doing exactly what it is there for: the 2009 form has no
no-wind spread at all, so once the wind is projected out on the flanks its
flanking rate is zero and the fire can only run downwind. The 2020 form retains
its radiative base rate on the flanks and keeps spreading sideways, slowly.

**The identities hold exactly.** `hybrid_none` and `rothermel_fuelmap` match
`rothermel_isotropic`, and `hybrid_all` matches `balbi2020_isotropic`, in both
columns. The blend is `(1 - w) R_p + w R_s`, so a weight of exactly 0 or 1
returns one model's value untouched.

**The directional identities hold exactly too**, at 80 and 122, which also
checks that moving the three direction-dependent drivers onto one shared
Runge-Kutta routine changed nothing. `hybrid_region_directional` burns 86
cells, between them, for the same reason as its isotropic counterpart.

**The wind selector blends rather than switches.** With every cell at a
weight near one half, `hybrid_wind` reports a head rate of 0.482 m/s, close to
the mean of 0.444 and 0.516, and a burned area at the Balbi identity's (216).

**The splits sit between the identities.** `hybrid_region` burns 238 cells,
between Rothermel's 240 and Balbi's 216 now that the two heads are within
15 % of each other. Its max ROS is the Balbi value since the diagnostic is
taken over burning cells and the head is in the Balbi region. `hybrid_fuel`
reports 0.444 m/s, Rothermel's head on FM1: Balbi 2020 evaluated on FM3 tall
grass (0.329 m/s) is the slower member at this wind, which is the per-fuel
Balbi table doing its job.

**The remaining models behave as their formulas say.** MacArthur has no
slope term and an exponential wind response, so it is by far the fastest here
and loses the most area to the directional switch (680 to 156). The
`grass_simple` fit is nearly isotropic by construction, its rate depending
weakly on direction, so the switch changes it least (88 to 84); the
Cheney-Gould 1998 model carries the full wind response and loses most of its
area to the switch (484 to 116). BEHAVE on FM1, a single dead class with no
live fuel, reduces exactly to Rothermel: 240 and 80, the same digits. `rothermel_nearest` matches `rothermel_isotropic` because the wind is
uniform, so the two horizontal mappings must agree; a difference here would
mean they disagree on a uniform field.

**Balbi options.** Consuming the reference-height wind instead of the
WAF-reduced midflame wind raises the head rate from 0.516 to 0.933 m/s and
the area to 468 cells. With the moisture-of-extinction cutoff on and the
1-hour moisture at 0.20, above the FM1 extinction value of 0.12, the rate is
exactly zero and the burned area stays at the ignition disc (52 cells against
the 50 stamped, the difference being the reinitialisation band).

**Hybrids of two generic members.** BEHAVE upwind of x = 800 m and
Cheney-Gould downwind gives 320 cells isotropic and 104 directional, between
the members' own results (240 and 80, 484 and 116) because the head runs at
the Cheney-Gould rate downwind while the flanks and the back run at BEHAVE's.
`hybrid_blend_width` ramps the weight over 200 m instead of stepping it and
lands at `hybrid_region`'s count (238).

**Non-burnable fuel codes.** `rothermel_code0_legacy` puts a 100 m strip of
fuel code 0 (the nodata value) one cell downwind of the ignition disc and
runs Rothermel without the per-fuel table, which is the historical setup: the
kernel spreads with the domain fuel model everywhere, the strip burns as
short grass, and the result is `rothermel_isotropic` to the cell (240).
`rothermel_code0_masked` lists code 0 in `erf.fire.fuel_map.nonburnable_codes`
and the head fire stops at the strip's upwind edge (202 cells, none of them
inside the strip). With `rothermel_per_fuel = true` code 0 has zero spread in
the per-fuel table and the strip stops the fire without the mask, but only
the mask also rejects ember landings and keeps the level set out of it.

## Backward compatibility

`erf.fire.directional_ros` defaults to false. Omitting it entirely and setting it
to false give the same answer, 240 cells for Rothermel (112 under the
fuel-class wind limit the code applied until 2026-10). `erf.fire.ros_model` still defaults to
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
the front propagation page of the Sphinx documentation (`Docs/sphinx_doc/theory/fire_propagation.rst`).

The projection reproduces neither the observed saturation of the length-to-width
ratio nor a backing rate below the no-wind rate, both of which the empirical
ellipse carries as calibration. It is the physically consistent choice when the
wind is resolved, not a calibrated one.
