#!/bin/bash
# Run every variant of the near-wall deck and print the comparison.
#
#   ./run_nearwall.sh /path/to/erf_exec [extra erf args...]
#   MPIRUN="mpirun -np 4" ./run_nearwall.sh /path/to/erf_exec
#   SKIP_RUN=1 ./run_nearwall.sh x     # only rebuild the table from existing logs
#
# Columns: burned fire cells and head ROS at the end, the number of burned
# cells inside the non-burnable mask (must be 0 where a mask is on), the
# largest reference wind on the fire grid, and the arrival time [s] at each
# probe in the order listed in inputs_base (u = upwind faces, g = gap
# midpoints, d = downwind faces; "-" means the probe never burned). u3 is the
# wall-effect probe: the flank runs along the middle box's wall to reach it.

set -u

EXE=${1:?usage: [MPIRUN="mpirun -np 4"] run_nearwall.sh /path/to/erf_exec [extra args]}
shift || true

VARIANTS="noib noib_mask noib_mask_wall ib_mask ib_mask_wall ib_mask_wind ib_mask_wall_wind"

printf "%-20s %5s %7s %8s %7s %8s  %s\n" variant exit cells max_ROS in_mask max_wind "u1 u2 u3 | g1 g2 g3 | d1 d2 d3"
printf "%-20s %5s %7s %8s %7s %8s  %s\n" -------------------- ----- ------- -------- ------- -------- ------------------------------------------

for v in $VARIANTS; do
    log="run_${v}.log"
    if [ "${SKIP_RUN:-0}" = "1" ] && [ -f "$log" ]; then
        rc=$(grep -q 'MPI_ABORT\|SIGABRT' "$log" && echo 6 || echo 0)
    else
        ${MPIRUN:-} "$EXE" "inputs_${v}" "$@" > "$log" 2>&1
        rc=$?
    fi
    cells=$(grep 'active fire cells' "$log" | tail -1 | awk '{print $NF}')
    ros=$(grep 'Rate-of-spread computed' "$log" | tail -1 | sed 's/.*Max: //; s/ m\/s.*//' | cut -c1-6)
    inmask=$(grep 'Non-burnable:' "$log" | tail -1 | sed 's/.*burned_inside=//')
    wind=$(grep 'Max reference wind' "$log" | tail -1 | sed 's/.*wind: //; s/ m\/s.*//' | cut -c1-6)
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
    printf "%-20s %5s %7s %8s %7s %8s %s\n" "$v" "$rc" "${cells:-n/a}" "${ros:-n/a}" "${inmask:--}" "${wind:-n/a}" "$probes"
done
