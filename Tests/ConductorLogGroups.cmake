# The column groups of the conductor logs, for the normwise comparison of CompareDataLogs.cmake
# (ERF_DATALOG_GROUPS). Included by RunConductors.cmake and by CompareDataLogsGroupsSelfTest.cmake,
# which checks these groups against the conductor logs' own numbers. Include it after
# CompareDataLogs.cmake, which empties the list.
#
# The components of one vector, the loads one support carries and the statistics of one quantity
# compare normwise: each field to SIGDIGITS digits of the largest magnitude of its group in that
# row (CompareDataLogs.cmake). A suspension tower's line pull is the sum of the pulls of the spans
# on its two sides, tens of kN (kilonewtons) along the line that cancel to a few N; at time 0 the
# remainder is the residual of MoorDyn's stationary initial-condition solve, which differs
# between compilers by about 1e-7 of the pulls, more than SIGDIGITS digits of the remainder.
set(ERF_DATALOG_GROUPS
    # towers.dat drag_Fx..z and line_Fx..z, transformers.dat Fx..z and Fh (N)
    "^(.+)_F[xyzh]$" "\\1_F"
    # transformers.dat Mx, My and Mh (N m)
    "^(.+)_M[xyh]$" "\\1_M"
    # towers.dat foundation loads: the vertical load and the largest leg reactions (N)
    "^(.+_t[0-9]+)_(vertical|max_compression|max_uplift)$" "\\1_legs"
    # towers.dat cross-arm displacement of a tower that bends (m)
    "^(.+_t[0-9]+)_arm_d[xy]$" "\\1_arm"
    # span logs: the drag on the span (N), the wind at its middle node (m/s), and the middle
    # node's sag below and offset across the chord (m)
    "^drag_[xyz]$" "drag"
    "^mid_[uvw]$" "mid_wind"
    "^mid_(sag|offset)$" "mid_shift"
    # *_stats.csv: one quantity per row
    "^(mean|rms|min|max)$" "stats")
