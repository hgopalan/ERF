#!/bin/bash
# Run every variant of the obstacle deck and print the comparison.
#
#   ./run_obstacles.sh /path/to/erf_exec [extra erf args...]
#   SKIP_RUN=1 ./run_obstacles.sh x     # only rebuild the table from existing logs
#
# Columns: burned fire cells and head ROS at the end, the number of cells the
# hybrid hands to its secondary model, the number of burned cells inside the
# non-burnable mask (*_mask decks; must be 0), and the arrival time [s] at each probe
# in the order listed in inputs_base (u = upwind faces, g = gap midpoints,
# d = downwind faces; "-" means the probe never burned).

set -u

EXE=${1:?usage: [MPIRUN="mpirun -np 8"] run_obstacles.sh /path/to/erf_exec [extra args]}
shift || true

VARIANTS="rothermel_noib balbi_noib hybrid_noib rothermel_ib balbi_ib hybrid_ib
          rothermel_noib_mask balbi_noib_mask hybrid_noib_mask rothermel_ib_mask balbi_ib_mask hybrid_ib_mask"

printf "%-20s %5s %7s %8s %6s %7s  %s\n" variant exit cells max_ROS sec in_mask "u1 u2 u3 | g1 g2 g3 | d1 d2 d3"
printf "%-20s %5s %7s %8s %6s %7s  %s\n" -------------------- ----- ------- -------- ------ ------- ------------------------------------------

for v in $VARIANTS; do
    log="run_${v}.log"
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "$log" ]; then
        # Rebuild the table from existing logs (SKIP_RUN=1)
        rc=$(grep -q 'MPI_ABORT\|SIGABRT' "$log" && echo 6 || echo 0)
    else
        ${MPIRUN:-} "$EXE" "inputs_${v}" "$@" > "$log" 2>&1
        rc=$?
    fi
    cells=$(grep 'active fire cells' "$log" | tail -1 | awk '{print $NF}')
    ros=$(grep 'Rate-of-spread computed' "$log" | tail -1 | sed 's/.*Max: //; s/ m\/s.*//' | cut -c1-6)
    sec=$(grep 'Hybrid ROS:' "$log" | tail -1 | sed 's/.*secondary_cells=//')
    inmask=$(grep 'Non-burnable:' "$log" | tail -1 | sed 's/.*burned_inside=//')
    probes=""
    for n in 0 1 2 3 4 5 6 7 8; do
        t=$(grep "\[FIRE PROBE\] $n " "$log" | sed -n 's/.*arrival_time_s=//p' | head -1)
        if [ -n "$t" ]; then
            probes="$probes $(printf '%5.0f' "$t")"
        else
            probes="$probes     -"
        fi
        if [ $n -eq 2 ] || [ $n -eq 5 ]; then probes="$probes |"; fi
    done
    printf "%-20s %5s %7s %8s %6s %7s %s\n" "$v" "$rc" "${cells:-n/a}" "${ros:-n/a}" "${sec:--}" "${inmask:--}" "$probes"
done
