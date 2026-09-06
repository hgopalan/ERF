# FireScottBurgan

The Scott and Burgan (2005) 40 fuel models, the set LANDFIRE's FBFM40
rasters use (codes 101-109 GR, 121-124 GS, 141-149 SH, 161-165 TU, 181-189
TL, 201-204 SB; 91-99 non-burnable), behind `erf.fire.fuel_map.fuel_set =
scott_burgan40` (default `anderson13`, the historical Anderson 13 only).
With the set on, a fuel map or `erf.fire.fuel_model_id` may carry the 40
codes natively, the dynamic models move their cured herbaceous load to the
1-h class at the deck's live moisture, and 91-99 (and 0) are non-burnable
without listing them. `erf.fire.fuel_map.sb40_crosswalk = true` instead
translates the codes to the Anderson 13 at load, as the Community Fire
Behavior Model does. `erf.fire.fuel_map.load_from_map = true` takes each
cell's initial fuel load from its own model instead of the uniform one.

Six one-way decks, 60 s: uniform Anderson short grass (`anderson`, and
`anderson_key` with the set written out), uniform GR2 (`sb_gr2`), the map of
`make_fuel_maps.py` (GR2 west, SH2 east, TL3 band north, an urban strip and
water along the south) natively with per-cell loads (`sb_map`), the same
map through the crosswalk (`sb_map_crosswalk`), and the hand-crosswalked
map in Anderson codes (`anderson_map`).

```
MPIRUN="mpirun -np 4" ./run_sb40.sh /path/to/erf_exec
python3 plot_sb40.py
```

The script checks the line-for-line reproduction of the historical deck;
that the uniform GR2 parameters the code prints equal the table in
`check_sb40.py` (loads, surface-area-to-volume ratios, depth, extinction
moisture, heat content, curing applied); that the map deck's initial fuel is
the sum of the cells' model loads and its non-burnable cells never burn; and
that the crosswalk deck reproduces the hand-crosswalked map line for line.
