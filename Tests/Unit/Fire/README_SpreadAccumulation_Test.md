# FARSITE front update unit tests

`ERF_GTestFarsiteSpreadAccumulation.cpp` checks `advance_farsite_one_step()`
and `advance_fire_subcycle()` (`Source/Fire/ERF_FarsiteEllipse.H`). The default
update, `erf.fire.farsite.front_update = "front_cell"`:

1. front cells are the unburned, burnable cells with a burned 4-neighbour;
2. each accumulates the head rate times `dt` in `disp_accum` (component 0,
   component 1 flags a front cell), so the rate used is the mean since its
   first neighbour burned;
3. each gets an arrival time from its burned neighbours' arrival times with the
   Hopf-Lax update of the Richards spread shape, and burns, with that time as
   its arrival time, if it falls inside the substep;
4. `phi` is rebuilt as -1 on burned cells and +1 elsewhere.

The tests:

- `FrontCellsAreUnburnedNeighbours`: one burned cell, one short substep; the
  four neighbours are flagged with `R dt` accumulated, nothing else is, nothing
  burns, and unburned cells have `phi = +1`.
- `ArrivalTimeAcrossSubsteps`: the accumulator doubles over two substeps; a long
  substep burns the next column at exactly `dx / R`, not at the substep's start,
  and not the column after it.
- `PlanarFrontAdvancesRt`: a planar front runs one column per `dx / R` through
  the subcycle driver, `R t` rather than `2 R t`; the legacy update on the same
  case runs at least 1.5 times as far.
- `NoStallWhenBurnedCellsLoseTheirRate`: the same arrival times with zero rate
  in every burned cell, and with the rate only in burned cells.
- `FarsiteFrontCell.HeadFlankAndBackRates`: from one burning cell in a
  1.64 m/s wind the head, flank and back cells burn at `dx / (a R)`,
  `dx / (b R)` and `dx / (c R)`.
- `FarsiteFrontCell.DecompositionIndependent`: a windy, sloped, patchy case on
  one box and on 64 boxes gives identical arrival times.
- `SingleCellStampingRaceSafety` (legacy update) and `FireGridGeometryResolution`.

`main()` is the shared `Tests/Unit/ERF_GTestMain.cpp`.
