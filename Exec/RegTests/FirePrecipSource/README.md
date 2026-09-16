# FirePrecipSource

Where the rain that wets the dead fuel classes comes from,
`erf.fire.precip_source`:

- `uniform` (default): `erf.fire.precip_rate_mm_hr` on every fire cell, as
  before this option (0 by default, so no rain).
- `atmosphere`: the rain of each atmospheric column, the change per step of
  the microphysics surface precipitation accumulation (Kessler's
  `rain_accum` here; the rain, snow and graupel accumulations of SAM,
  Morrison, WSM6 and WDM6) in mm/hr, every fire cell taking its column's
  value. The accumulation the rate is measured from is checkpointed
  (`FirePrecipAccumPrev`). The rate of every fire cell is written as
  `fire_precip_mm_hr` and its maximum as the last column of the statistics
  CSV, `precip_max_mm_hr`.

## The scenario

The FireRestart grid (2 x 2 km, 20 x 20 x 30 cells of 100 x 100 x 20 m, one
rank) with Kessler microphysics, a neutral 8 m/s sounding at 88 % relative
humidity near the ground (20 g/kg, falling to 5 g/kg at 1 km) and a cold
column of air at the ground in the middle of the domain (`erf.prob_name =
Bubble`, -15 K air temperature over 300 m across and 250 m up). The column
saturates at once, and its cloud water rains out within the 20 s run; the
rest of the domain stays dry. The fire is passive, so the atmosphere is the
same in every variant. The dead classes start at 0.06 / 0.07 / 0.09 and wet
toward the humid air's equilibrium everywhere; the rain adds to that.

Five 40-step runs: `legacy` (no rain key: the historical deck), `uniform`
(2 mm/hr), `atmosphere`, `atmosphere_chk` (to a checkpoint at step 27, while
it rains) and `atmosphere_restart` (from it to step 40).

```
MPIRUN="mpirun -np 1" ./run_precip.sh /path/to/erf_exec
```

`check_precip.py` (pure Python, on the fire plotfiles at step 40, the
atmosphere plotfiles at steps 39 and 40 and the checkpoint) requires:

- `fire_precip_mm_hr` present with a rain source and absent on the legacy deck;
- `atmosphere`: some fire cells at or above the 0.1 mm/hr wetting threshold
  and others at zero, and the 1-hour moisture higher than `legacy` on every
  cell at or above the threshold and identical on every other cell;
- `uniform`: the 1-hour moisture higher than `legacy` on every cell by the
  same amount to 1 %, and the field equal to the deck's rate everywhere;
- the rate of every fire cell equal to the change of its column's
  `rain_accum` over the last step times 3600 / dt, to round-off;
- `FirePrecipAccumPrev` in the checkpoint, and the restarted run's level set,
  1-hour moisture and rain rate identical to the straight run's.

On a binary without the option the first check fails (no field) and the
script stops. A binary that writes the field but does not hand the rain to
the moisture update fails the `wet_atm` check. Before this branch a Kessler
run on a constant-dz mesh could not be restarted at all: the restart path set
the microphysics' minimum dz only on fitted meshes, so the sedimentation
substep count of the first restarted step was computed from an
uninitialised value and the step never finished; the `atmosphere_restart`
leg times out on such a binary.

## Reference results

Measured 2026-09-16 (Debug, MPI, one rank, macOS), 40 steps of 0.5 s:

| variant      | max rain [mm/hr] | fire cells >= 0.1 mm/hr | 1-h moisture minus legacy |
|--------------|------------------|-------------------------|---------------------------|
| `legacy`     | (no field)       | -                       | 0                         |
| `uniform`    | 2.000            | 6400 of 6400            | 1.108e-4 everywhere       |
| `atmosphere` | 0.291            | 96 of 6400              | >= 1.87e-6 under rain, 0 elsewhere |

The rain is light because the run is 20 s long: Kessler's autoconversion
turns the column's 11 g/kg of cloud water into rain at 1e-5 kg/kg/s, and
the rate at the surface is still growing at step 40. The 768 fire cells
between zero and the threshold (the column's fringe, where the rate is below
0.1 mm/hr) see no wetting, which the second check verifies as well.
