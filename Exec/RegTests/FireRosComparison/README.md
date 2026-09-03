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

## Reference results

Burned cells after 200 s from a 50-cell ignition disk, and the head-fire rate of
spread:

| variant | burned cells | max ROS (m/s) |
|---|---|---|
| `rothermel_isotropic` | 132 | 0.250 |
| `rothermel_directional` | 78 | 0.250 |
| `balbi2009_isotropic` | 200 | 0.390 |
| `balbi2009_directional` | 84 | 0.390 |
| `balbi2020_isotropic` | 232 | 0.514 |
| `balbi2020_directional` | 138 | 0.514 |

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

## Backward compatibility

`erf.fire.directional_ros` defaults to false. Omitting it entirely and setting it
to false give the same answer, 132 cells for Rothermel, which is also what the
code produced before the switch existed. The older Balbi-only flag
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
