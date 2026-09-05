# FARSITE spread accumulation unit tests

`ERF_GTestFarsiteSpreadAccumulation.cpp` checks the bookkeeping of
`advance_farsite_one_step()` (`Source/Fire/ERF_FarsiteEllipse.H`), which
carries the FARSITE front between substeps as a per-cell displacement
accumulator rather than as spread vectors:

1. every substep adds `R * dt` along the front normal (scaled by the
   Richards ellipse under wind) to `disp_accum` at each front cell;
2. when the accumulated displacement of a cell reaches one cell width the
   cell writes an absolute target position into `farsite_work` and resets
   its accumulator;
3. the targets are gathered across ranks and stamped into `phi`, which is
   rebuilt from `arrival_time` at every substep, and the newly burned cells
   take the substep's time as their arrival time.

The tests: `SingleStepFrontDetection` (one short substep moves the four front
cells by exactly `R dt` and stamps nothing), `SpreadAccumulationAcrossSteps`
(the accumulator doubles over two substeps, then a substep that reaches the
threshold stamps, records arrival times and resets the accumulator),
`SingleCellStampingRaceSafety` and `FireGridGeometryResolution`.

`main()` is the shared `Tests/Unit/ERF_GTestMain.cpp`.
