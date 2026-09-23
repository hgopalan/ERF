# FireCustomFuel

A fire on flat ground whose fuel model is described in the input deck
(`erf.fire.custom_fuel.*`) instead of coming from the Anderson 13 or the Scott
and Burgan 40 compiled tables. Fuel codes 1000-1015 take the fuel table slots
above the published sets, so every rate-of-spread model, the heat flux, the
fuel load and the burnout read them exactly as they read a published model.

Properties are given in SI and converted once to the units
`FuelModelParams` carries internally.

## Decks

| Deck | What it is |
| --- | --- |
| `inputs_base` | 320 x 160 x 160 m, 5 m cells, 1.25 m fire cells, one-way coupling, Anderson short grass, Rothermel |
| `inputs_anderson1` | the baseline: short grass from the compiled table |
| `inputs_custom_grass` | the same fuel written out in SI as code 1000; must reproduce the baseline |
| `inputs_custom_heavy` | a heavy bed of coarse woody material, which no published set can express |
| `inputs_custom_map` | a raster mixing Scott-Burgan GR2, a deck-defined block and the non-burnable codes, each cell starting with its own model's load |
| `inputs_bad_code` | a code outside 1000-1015 |
| `inputs_bad_missing` | a block missing a required property |
| `inputs_bad_depth` | a bed depth under the 0.01 m floor, where the Balbi models silently return zero spread |
| `inputs_bad_heat` | a heat content left in BTU/lb instead of J/kg |
| `inputs_bad_burnout` | `burnout_model = sfire` on a deck-defined model with no burn time |
| `inputs_bad_undeclared` | a raster holding a custom code the deck never defines |
| `inputs_bad_uniform_code` | `fuel_model_id` in the custom range with no block defining it |
| `inputs_custom_map_altid` | the mixed map with a different uniform `fuel_model_id`, which a per-fuel run must ignore |
| `inputs_undeclared_nonburnable` | the undeclared-code raster with that code declared non-burnable, which must run |

`fuel_map_mixed.asc` and `fuel_map_undeclared.asc` come from
`python3 make_fuel_map.py`.

## Running

```
MPIRUN="mpirun -np 2" FCOMPARE=/path/to/amrex_fcompare ./run_custom_fuel.sh /path/to/erf_exec
```

`SKIP_RUN=1` reruns only the checks. `FCOMPARE` adds the box-parity comparison.

## What the checks establish

`check_custom_fuel.py --selftest` exercises the checker's own pass/fail logic
first, with values just inside and just outside each band.

* **identity** — `custom_grass` reproduces `anderson1` over all 1920
  `[FIRE DEBUG]` numbers with a worst relative difference of 0. The fire
  plotfiles agree to 8.5e-14 absolute and 5.4e-16 relative, which is the
  round-off of the eight-digit SI values the deck carries, not bitwise
  equality: `0.16600262` is the written form of `0.034 lb/ft2 x 4.88243`.
* **summary** — the start-up line reports the deck's own SI back, at slot 54,
  which is above every published slot.
* **fuel** — the map run starts with 193430.6 kg on the grid, the sum of each
  cell's model load; 96% of it comes from the deck-defined block, so a run that
  ignored the block could not pass.
* **crossed** — the ignition circle straddles the block edge, so one front
  burns in both fuels. At 60 s it has burned 508 cells of the deck-defined code
  at 0.05287 m/s and 266 of GR2 at 0.1559 m/s: the deck's properties reach the
  per-cell kernel, not just the uniform path. The front is asymmetric because
  the sounding's wind carries the head fire east into the block and leaves a
  backing fire in GR2, so the burned counts are a wind effect, not a fuel one;
  the rates are the fuel one. Needs `yt`.
* **slower** — the coarse bed spreads at 0.0547 m/s against grass's 0.2160 m/s.
  The deck's properties drive the spread; they are not a published model in
  disguise.
* **box parity** — the map run on the default decomposition and on one box
  agree bitwise, which covers the per-cell slot lookup and the cross-rank
  reduction in the raster check.
* **uniform id is not read** — `custom_map_altid` differs from `custom_map`
  only in `erf.fire.fuel_model_id`, which a run with
  `erf.fire.rothermel_per_fuel` must ignore entirely. Every number agrees over
  1920 lines and every plotfile field is bitwise equal, except the two Byram
  diagnostics (see below). On the binary before the level-set path read the
  per-cell table this check fails: the front followed the uniform coefficients
  and the two runs burned 608 against 704 cells at 30 s.

  The deck uses Anderson 10 as the alternative because it shares Anderson 1's
  1 ft bed depth. The wind adjustment factor and the fuel wind height are still
  built from the uniform model's depth, so a depth-matched pair isolates the
  coefficients; with Anderson 13 the effective wind itself differs by 27%.
* **known limitation, exempted explicitly** — `fire_fireline_intensity` and
  `fire_flame_length` come from Byram's relation on the *uniform* initial fuel
  load, so under `load_from_map` they scale with `erf.fire.fuel_model_id`
  instead of the cell's own load and can be exactly zero. The identity check
  exempts `I_B_max` and `L_max` by name and reports an exemption that has
  stopped differing, so the exemption cannot outlive the defect. The rate of
  spread, the heat flux, the residence time and the fuel load are per cell.
* **aborts** — each of the six bad decks stops the run with a message naming
  the input. `bad_undeclared` was also run on one and two ranks: its reduction
  aborts on both rather than hanging on one.

* **model sweep** — ten 10 s runs pair grass against the coarse deck fuel for
  every rate-of-spread model. Each has to give a different rate, since each
  reads its fuel through the same `FuelModelParams`:

  | `ros_model` | grass [m/s] | deck fuel [m/s] |
  | --- | --- | --- |
  | `rothermel` | 0.2160244347 | 0.05470070803 |
  | `balbi` | 0.4534711647 | 0.00221418181 |
  | `behave` | 0.2160244347 | 0.06701721308 |
  | `macarthur` | 1.793275906 | 6.309411001 |
  | `cheney_gould` | 0.1962616261 | 0.2316395252 |

  MacArthur and Cheney-Gould rise rather than fall: they are grassland
  correlations driven by the load and the bed depth, so a deeper, heavier bed
  speeds them up where Rothermel's reaction term and Balbi's radiation, which
  read the surface-area-to-volume ratio, slow down.

## Measured numbers

| variant | active fire cells at 60 s | max ROS [m/s] | fuel at 0 s [kg] | fuel at 60 s [kg] |
| --- | --- | --- | --- | --- |
| `anderson1` | 774 | 0.2160244347 | 8496.8 | 8305.8 |
| `custom_grass` | 774 | 0.2160244347 | 8496.8 | 8305.8 |
| `custom_heavy` | 548 | 0.05470070803 | 1126379.1 | 1118114.9 |
| `custom_map` | 558 | 0.1558625579 | 193418.8 | 188870.3 |

`custom_map` is re-measured: with the level-set path reading the per-cell
coefficients it burns 558 cells at 60 s where it burned 774 before, since the
front no longer spreads at the uniform grass rate inside the coarse block.
At 30 s the count is 500 against the previous 608.

`custom_map`'s max ROS is GR2's rate, the fastest fuel on its grid; the
deck-defined block spreads at 0.05287 m/s in the same run.
